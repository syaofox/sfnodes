"""VOSR2 推理档位设置 — 纯 Python（无 torch / comfy 依赖，可宿主机单测）。

对齐 TE-Speed-VOSR2 的 `TESpeedVOSR2Settings` 语义（README 的 manual / speed 两档
推理配置 + 显存策略 + 分块策略），但不依赖其编译实现：

- `quality_profile`：manual = 逐块/逐帧保守批量；speed = 放大批量换吞吐
  （显存峰值更高，README 明确提示 speed 可能因换载反而更慢）。
- `memory_policy`：auto = 按空闲显存自动；resident = DiT/DINO 常驻；
  staged = VAE 解码前释放 DiT/DINO 驻留（降低峰值）。
- `tile_strategy`：auto = 输出超过原生 512 像素时自动分块；tiled = 强制分块；
  full_frame = 不分块（pad 成方形整图推理）。
- 各 batch 字段 0 表示「按档位自动」，`batch_override` > 0 时覆盖 item 批量。
"""

from dataclasses import dataclass, fields, replace

QUALITY_PROFILES = ("manual", "speed")
MEMORY_POLICIES = ("auto", "resident", "staged")
TILE_STRATEGIES = ("auto", "tiled", "full_frame")

# 档位 → 各阶段批量（0 字段的解析结果）
PROFILE_BATCHES = {
    "manual": {"dit_tile_batch": 1, "dino_batch": 1, "image_batch": 1, "frame_batch": 1},
    "speed": {"dit_tile_batch": 4, "dino_batch": 4, "image_batch": 2, "frame_batch": 8},
}

# 输出超过该边长即认为超出 VOSR 2.0 原生训练分辨率（README 口径）
NATIVE_TILE_PIXELS = 512


@dataclass(frozen=True)
class VOSR2Settings:
    quality_profile: str = "manual"
    memory_policy: str = "auto"
    tile_strategy: str = "auto"
    torch_compile: bool = False
    vae_encode_amp: bool = True
    auto_expand_vae_tile: bool = True
    dit_tile_batch: int = 0
    dino_batch: int = 0
    image_batch: int = 0
    frame_batch: int = 0
    batch_override: int = 0
    temporal_cache: bool = True
    cache_threshold: float = 0.05
    cache_refresh: int = 0

    # ---- 档位解析 ----

    def _profile_value(self, name):
        profile = self.quality_profile if self.quality_profile in PROFILE_BATCHES else "manual"
        return PROFILE_BATCHES[profile][name]

    def resolved_dit_tile_batch(self):
        return self.dit_tile_batch or self._profile_value("dit_tile_batch")

    def resolved_dino_batch(self):
        return self.dino_batch or self._profile_value("dino_batch")

    def resolved_image_batch(self):
        if self.batch_override > 0:
            return self.batch_override
        return self.image_batch or self._profile_value("image_batch")

    def resolved_frame_batch(self):
        if self.batch_override > 0:
            return self.batch_override
        return self.frame_batch or self._profile_value("frame_batch")

    def use_tiling(self, output_h, output_w, tile_size):
        """按策略 + 输出尺寸决定是否分块。"""
        if self.tile_strategy == "tiled":
            return True
        if self.tile_strategy == "full_frame":
            return False
        if tile_size <= 0:
            return False
        return output_h > tile_size or output_w > tile_size

    def describe(self):
        return (
            f"profile={self.quality_profile} memory={self.memory_policy} "
            f"tiles={self.tile_strategy} compile={self.torch_compile} "
            f"batches=(dit {self.resolved_dit_tile_batch()}, dino {self.resolved_dino_batch()}, "
            f"image {self.resolved_image_batch()}, frame {self.resolved_frame_batch()})"
        )


def default_settings():
    return VOSR2Settings()


def normalize_settings(obj):
    """把可选输入（None / dict / VOSR2Settings）归一为 VOSR2Settings。

    - None → 全默认（节点未接 Settings 输入）。
    - dict → 只取已知字段（工作流反序列化 / 第三方构造）。
    - VOSR2Settings → 原样返回。
    """
    if obj is None:
        return default_settings()
    if isinstance(obj, VOSR2Settings):
        return obj
    if isinstance(obj, dict):
        known = {f.name for f in fields(VOSR2Settings)}
        return VOSR2Settings(**{k: v for k, v in obj.items() if k in known})
    raise TypeError(f"VOSR2 settings 输入类型不支持: {type(obj).__name__}")


def with_overrides(settings, **kwargs):
    """在给定设置上覆盖若干字段（None 值忽略）。"""
    settings = normalize_settings(settings)
    patch = {k: v for k, v in kwargs.items() if v is not None}
    return replace(settings, **patch) if patch else settings


def validate_settings(settings):
    """校验枚举/数值范围，返回错误信息（None 表示合法）。"""
    if settings.quality_profile not in QUALITY_PROFILES:
        return f"quality_profile 必须是 {QUALITY_PROFILES} 之一，得到 {settings.quality_profile!r}"
    if settings.memory_policy not in MEMORY_POLICIES:
        return f"memory_policy 必须是 {MEMORY_POLICIES} 之一，得到 {settings.memory_policy!r}"
    if settings.tile_strategy not in TILE_STRATEGIES:
        return f"tile_strategy 必须是 {TILE_STRATEGIES} 之一，得到 {settings.tile_strategy!r}"
    if not 0.0 <= settings.cache_threshold <= 1.0:
        return f"cache_threshold 必须在 [0, 1]，得到 {settings.cache_threshold}"
    for name in ("dit_tile_batch", "dino_batch", "image_batch", "frame_batch",
                 "batch_override", "cache_refresh"):
        value = getattr(settings, name)
        if not isinstance(value, int) or value < 0:
            return f"{name} 必须是非负整数，得到 {value!r}"
    return None
