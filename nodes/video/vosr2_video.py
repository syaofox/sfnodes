"""VOSR2 视频帧超分节点。

VOSR 2.0 是逐帧模型（无时序注意力），视频只是 IMAGE 批次。TE-Speed-VOSR2 的
视频优化在本节点等价实现：

- `frame_batch`：逐帧推理时一次拼批的帧数（默认按档位，manual 1 / speed 8）
- DINOv2 时序缓存：相邻帧同位置瓦片的像素签名差 <= cache_threshold 时复用上一帧
  特征（特征存 CPU），每 cache_refresh 帧强制刷新 —— 静帧/慢镜头显著省时
- 逐帧 seed = seed + 帧号；进度条按「帧 × 瓦片」报总进度
- 目标尺寸与图片节点同源（倍率/总像素/长边/短边四模式，见 SFVOSR2Settings.size_input_types）
"""

import torch

from ...sf_utils.logger import get_logger
from ..model.vosr2.inferencer import DinoTemporalCache, VOSR2Inferencer
from ..model.vosr2.settings import normalize_settings
from ..model.vosr2_settings import build_size_spec, size_input_types

logger = get_logger(__name__)

_CATEGORY = "sfnodes/video"


class SFVOSR2Video:
    """VOSR2 视频帧超分。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOSR2_MODEL", {"tooltip": "来自 SFVOSR2ModelLoader"}),
                "images": ("IMAGE", {"tooltip": "视频帧序列（IMAGE 批次，按帧序处理）"}),
                **size_input_types(),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**63 - 1,
                     "tooltip": "初始噪声种子；第 i 帧用 seed+i"},
                ),
                "color_alignment": (
                    ["wavelet", "adain", "none"],
                    {"default": "wavelet",
                     "tooltip": "相对双三次参考帧的色彩对齐：wavelet 低频替换（推荐）/ adain 统计匹配 / none"},
                ),
                "color_downsample": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4, "step": 1,
                     "tooltip": "wavelet 低频计算降采样倍数（1 = 全分辨率；2~4 更快）"},
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 0, "max": 4096, "step": 64,
                     "tooltip": "DiT 分块像素边长；0 = 关闭分块"},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 32, "min": 0, "max": 512, "step": 8,
                     "tooltip": "DiT 分块重叠像素（须小于 tile_size）"},
                ),
                "vae_tile_size": (
                    "INT",
                    {"default": 1024, "min": 0, "max": 8192, "step": 64,
                     "tooltip": "VAE 分块像素边长；0 = 整图解码（大图易 OOM）"},
                ),
                "vae_tile_overlap": (
                    "INT",
                    {"default": 32, "min": 0, "max": 512, "step": 8,
                     "tooltip": "VAE 分块重叠像素（须小于 vae_tile_size）"},
                ),
                "force_offload": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "执行完成后立即卸载模型释放显存（下次执行会重新加载）"},
                ),
            },
            "optional": {
                "settings": (
                    "VOSR2_SETTINGS",
                    {"tooltip": "来自 SFVOSR2Settings 的推理档位（frame_batch / 时序缓存等）；不接则用默认"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"
    CATEGORY = _CATEGORY
    DESCRIPTION = "VOSR 2.0 视频帧超分（逐帧 + DINOv2 时序缓存 + 分块批量，模型由 SFVOSR2ModelLoader 加载）"

    def upscale(self, model, images, size_mode, scale, total_pixels, longer_size, shorter_size,
                seed, color_alignment, color_downsample,
                tile_size, tile_overlap, vae_tile_size, vae_tile_overlap, force_offload,
                settings=None):
        settings = normalize_settings(settings)
        spec = build_size_spec(size_mode, scale, total_pixels, longer_size, shorter_size)
        model.set_memory_policy(settings.memory_policy)
        model.set_torch_compile(settings.torch_compile)

        cache = DinoTemporalCache(
            enabled=settings.temporal_cache,
            threshold=settings.cache_threshold,
            refresh=settings.cache_refresh,
        )
        frames = int(images.shape[0])
        logger.info(
            f"VOSR2 视频超分: {frames} 帧, {spec.describe()}, "
            f"tile={tile_size}/{tile_overlap}, vae_tile={vae_tile_size}/{vae_tile_overlap}, "
            f"{settings.describe()}"
        )
        inferencer = VOSR2Inferencer(model)
        result = inferencer.upscale(
            images, spec, seed, settings=settings,
            color_alignment=color_alignment, color_downsample=color_downsample,
            tile_size=tile_size, tile_overlap=tile_overlap,
            vae_tile_size=vae_tile_size, vae_tile_overlap=vae_tile_overlap,
            cache=cache, item_batch=settings.resolved_frame_batch(),
        )
        if settings.temporal_cache:
            total = cache.hits + cache.misses
            if total:
                logger.info(
                    f"VOSR2 时序缓存: 命中 {cache.hits}/{total} 瓦片 "
                    f"({cache.hits * 100.0 / total:.1f}%)"
                )
        if force_offload:
            model.offload()
        return (result.to(torch.float32),)
