"""VOSR2 推理档位设置节点。

输出 `VOSR2_SETTINGS`，接给 SFVOSR2Upscale / SFVOSR2Video 的可选 settings 输入；
不接时各节点用默认档（manual + auto）。语义见 nodes/model/vosr2/settings.py：

- quality_profile：manual 保守批量 / speed 大批量换吞吐（显存峰值更高）
- memory_policy：auto / resident（DiT+DINO 常驻）/ staged（VAE 解码前释放驻留）
- tile_strategy：auto（>512px 自动分块）/ tiled（强制）/ full_frame（不分块）
- 各 batch 0 = 按档位自动；batch_override > 0 覆盖 image/frame 批量
"""

from .vosr2.settings import (
    MEMORY_POLICIES,
    QUALITY_PROFILES,
    TILE_STRATEGIES,
    VOSR2Settings,
    validate_settings,
)
from .vosr2.sizing import MAX_TARGET, MIN_TARGET, SIZE_MODES, TargetSizeSpec

_CATEGORY = "sfnodes/model"


def size_input_types():
    """目标尺寸四模式的公共 widget 定义（SFVOSR2Upscale / SFVOSR2Video 共用，禁止内联副本）。"""
    return {
        "size_mode": (
            list(SIZE_MODES),
            {
                "default": "scale",
                "tooltip": (
                    "目标尺寸模式：scale = 倍率（允许 <1 缩小）；total pixels = 目标总像素"
                    "（百万像素，1.00 = 1024×1024）；longer dimension = 长边；"
                    "shorter dimension = 短边。除 scale 外均保持源图宽高比"
                ),
            },
        ),
        "scale": (
            "FLOAT",
            {
                "default": 4.0, "min": 0.05, "max": 16.0, "step": 0.05,
                "tooltip": "缩放倍率（输出 = 输入 × scale，非整数亦可；<1 为缩小，质量未验证）；仅 size_mode=scale 生效",
            },
        ),
        "total_pixels": (
            "FLOAT",
            {
                "default": 1.0, "min": 0.01, "max": 64.0, "step": 0.01,
                "tooltip": "目标总像素（百万像素）：1.00 = 1024×1024 = 1,048,576 像素"
                           "（与原生 ImageScaleToTotalPixels 一致）；仅 size_mode=total pixels 生效",
            },
        ),
        "longer_size": (
            "INT",
            {
                "default": 1024, "min": MIN_TARGET, "max": MAX_TARGET, "step": 8,
                "tooltip": "长边目标像素数（保持宽高比）；仅 size_mode=longer dimension 生效",
            },
        ),
        "shorter_size": (
            "INT",
            {
                "default": 1024, "min": MIN_TARGET, "max": MAX_TARGET, "step": 8,
                "tooltip": "短边目标像素数（保持宽高比）；仅 size_mode=shorter dimension 生效",
            },
        ),
    }


def build_size_spec(size_mode, scale, total_pixels, longer_size, shorter_size):
    """由节点 widget 构造并校验 TargetSizeSpec（非法参数直接报错）。"""
    spec = TargetSizeSpec(
        mode=size_mode,
        scale=scale,
        total_pixels=total_pixels,
        longer_size=longer_size,
        shorter_size=shorter_size,
    )
    error = spec.validate()
    if error:
        raise ValueError(f"VOSR2 目标尺寸参数非法: {error}")
    return spec


class SFVOSR2Settings:
    """VOSR2 推理设置。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quality_profile": (
                    list(QUALITY_PROFILES),
                    {
                        "default": "manual",
                        "tooltip": "manual = 逐瓦片/逐帧保守批量；speed = 大瓦片批量（显存峰值更高，可能因换载变慢）",
                    },
                ),
                "memory_policy": (
                    list(MEMORY_POLICIES),
                    {
                        "default": "auto",
                        "tooltip": "auto = 按显存自动；resident = DiT/DINO 常驻；staged = VAE 解码前释放 DiT/DINO 驻留",
                    },
                ),
                "tile_strategy": (
                    list(TILE_STRATEGIES),
                    {
                        "default": "auto",
                        "tooltip": "auto = 输出超过 512px 自动分块；tiled = 强制分块；full_frame = 整图（pad 成方形）",
                    },
                ),
                "torch_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "用 torch.compile 包装 DiT 前向（需要 triton，首次调用编译较慢，失败自动回退 eager）",
                    },
                ),
                "vae_encode_amp": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "VAE 编解码使用 bf16 autocast（权重仍 fp32），降低显存占用",
                    },
                ),
                "auto_expand_vae_tile": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "空闲显存允许时自动扩大 VAE 瓦片（更少块、更快；OOM 会自动降级）",
                    },
                ),
                "dit_tile_batch": (
                    "INT",
                    {"default": 0, "min": 0, "max": 32,
                     "tooltip": "DiT 瓦片拼批数量；0 = 按档位（manual 1 / speed 4），OOM 自动减半"},
                ),
                "dino_batch": (
                    "INT",
                    {"default": 0, "min": 0, "max": 32,
                     "tooltip": "DINOv2 瓦片拼批数量；0 = 按档位（manual 1 / speed 4）"},
                ),
                "image_batch": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16,
                     "tooltip": "图片批次拼批数量（同尺寸才会拼批）；0 = 按档位"},
                ),
                "frame_batch": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64,
                     "tooltip": "视频帧拼批数量；0 = 按档位（manual 1 / speed 8）"},
                ),
                "batch_override": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64,
                     "tooltip": ">0 时直接覆盖 image_batch / frame_batch（统一强制批量）"},
                ),
                "temporal_cache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "视频帧间复用 DINOv2 特征（相邻帧相似时跳过重算）"},
                ),
                "cache_threshold": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "帧相似度阈值：瓦片像素签名最大差低于该值才复用缓存（越大越激进）"},
                ),
                "cache_refresh": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000,
                     "tooltip": "每 N 帧强制刷新 DINO 缓存；0 = 不强制刷新"},
                ),
            },
        }

    RETURN_TYPES = ("VOSR2_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "make"
    CATEGORY = _CATEGORY
    DESCRIPTION = "VOSR2 推理档位设置（档位 / 显存策略 / 分块策略 / 批量 / 视频时序缓存）"

    def make(self, quality_profile, memory_policy, tile_strategy, torch_compile,
             vae_encode_amp, auto_expand_vae_tile, dit_tile_batch, dino_batch,
             image_batch, frame_batch, batch_override, temporal_cache,
             cache_threshold, cache_refresh):
        settings = VOSR2Settings(
            quality_profile=quality_profile,
            memory_policy=memory_policy,
            tile_strategy=tile_strategy,
            torch_compile=torch_compile,
            vae_encode_amp=vae_encode_amp,
            auto_expand_vae_tile=auto_expand_vae_tile,
            dit_tile_batch=dit_tile_batch,
            dino_batch=dino_batch,
            image_batch=image_batch,
            frame_batch=frame_batch,
            batch_override=batch_override,
            temporal_cache=temporal_cache,
            cache_threshold=cache_threshold,
            cache_refresh=cache_refresh,
        )
        error = validate_settings(settings)
        if error:
            raise ValueError(f"VOSR2 设置非法: {error}")
        return (settings,)
