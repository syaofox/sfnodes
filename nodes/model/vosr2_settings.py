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

_CATEGORY = "sfnodes/model"


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
