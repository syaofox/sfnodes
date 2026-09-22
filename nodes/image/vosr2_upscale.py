"""VOSR2 图片超分节点。

VOSR 2.0 是 one-step 1.4B DiT（Qwen-Image 2D VAE 潜空间 + DINOv2-L 条件）的
生成式超分模型；本节点逐项处理 IMAGE 批次（第 i 项 seed+i），目标尺寸支持
倍率/总像素/长边/短边四种模式（见 SFVOSR2Settings.size_input_types），
另支持 DiT/VAE 分块与色彩对齐。推理档位由可选 settings 输入控制。
"""

import torch

from ...sf_utils.logger import get_logger
from ..model.vosr2.inferencer import VOSR2Inferencer
from ..model.vosr2.settings import normalize_settings
from ..model.vosr2_settings import build_size_spec, size_input_types

logger = get_logger(__name__)

_CATEGORY = "sfnodes/image"


class SFVOSR2Upscale:
    """VOSR2 图片超分。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOSR2_MODEL", {"tooltip": "来自 SFVOSR2ModelLoader"}),
                "image": ("IMAGE", {"tooltip": "输入图像，支持 batch 逐张处理"}),
                **size_input_types(),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**63 - 1,
                     "tooltip": "初始噪声种子；batch 第 i 项用 seed+i"},
                ),
                "color_alignment": (
                    ["wavelet", "adain", "none"],
                    {"default": "wavelet",
                     "tooltip": "相对双三次参考图的色彩对齐：wavelet 低频替换（推荐）/ adain 统计匹配 / none"},
                ),
                "color_downsample": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4, "step": 1,
                     "tooltip": "wavelet 低频计算降采样倍数（1 = 全分辨率；2~4 更快，色彩更平滑）"},
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 0, "max": 4096, "step": 64,
                     "tooltip": "DiT 分块像素边长；0 = 关闭分块。输出超过 512px 建议保持 512（原生训练分辨率）"},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 32, "min": 0, "max": 512, "step": 8,
                     "tooltip": "DiT 分块重叠像素（须小于 tile_size）"},
                ),
                "vae_tile_size": (
                    "INT",
                    {"default": 1024, "min": 0, "max": 8192, "step": 64,
                     "tooltip": "VAE 分块像素边长；0 = 整图解码（大图易 OOM）。输出超过 ~1024px 建议保持 1024"},
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
                    {"tooltip": "来自 SFVOSR2Settings 的推理档位；不接则用默认（manual + auto）"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = _CATEGORY
    DESCRIPTION = "VOSR 2.0 图片超分（one-step 1.4B DiT，支持分块与显存策略；模型由 SFVOSR2ModelLoader 加载）"

    def upscale(self, model, image, size_mode, scale, total_pixels, longer_size, shorter_size,
                seed, color_alignment, color_downsample,
                tile_size, tile_overlap, vae_tile_size, vae_tile_overlap, force_offload,
                settings=None):
        settings = normalize_settings(settings)
        spec = build_size_spec(size_mode, scale, total_pixels, longer_size, shorter_size)
        model.set_memory_policy(settings.memory_policy)
        model.set_torch_compile(settings.torch_compile)
        logger.info(
            f"VOSR2 图片超分: {image.shape[0]} 张, {spec.describe()}, "
            f"tile={tile_size}/{tile_overlap}, vae_tile={vae_tile_size}/{vae_tile_overlap}, "
            f"{settings.describe()}"
        )
        inferencer = VOSR2Inferencer(model)
        result = inferencer.upscale(
            image, spec, seed, settings=settings,
            color_alignment=color_alignment, color_downsample=color_downsample,
            tile_size=tile_size, tile_overlap=tile_overlap,
            vae_tile_size=vae_tile_size, vae_tile_overlap=vae_tile_overlap,
        )
        if force_offload:
            model.offload()
        return (result.to(torch.float32),)
