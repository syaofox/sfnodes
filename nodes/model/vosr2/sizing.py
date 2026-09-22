"""VOSR2 目标尺寸解析 — 纯 Python（无 torch / comfy 依赖，可宿主机单测）。

四种模式（对齐 SFImageResizePlus 的原生语义，缩放函数复用 `sf_utils/resize_engine`）：

- `scale`：倍率（0.05~16，允许 <1 缩小）；取整 round
- `total pixels`：目标总像素，单位百万像素（**1.00 = 1024×1024 = 1,048,576 px**，
  与原生 ImageScaleToTotalPixels 的 binary-MP 约定一致，非 10⁶ SI-MP）
- `longer dimension`：长边目标像素数（保持宽高比）
- `shorter dimension`：短边目标像素数（保持宽高比）

解析结果统一钳制到 `[MIN_TARGET, MAX_TARGET]`：超上限时**等比**缩小（不破坏宽高比），
低于下限时逐维抬到下限（极端长宽比/极小倍率才会触发）。
"""

import math
from dataclasses import dataclass

from ....sf_utils.resize_engine import (
    longer_dimension_to_wh,
    multiplier_to_wh,
    shorter_dimension_to_wh,
    total_pixels_to_wh,
)

SIZE_MODES = ("scale", "total pixels", "longer dimension", "shorter dimension")

# 目标尺寸钳制范围（像素）：小于 16 无意义（管线还要 pad 到 16 倍数），
# 大于 8192 显存/耗时不可控（节点 tile_size 上限 4096）
MIN_TARGET = 16
MAX_TARGET = 8192

SCALE_RANGE = (0.05, 16.0)
TOTAL_PIXELS_RANGE = (0.01, 64.0)
SIDE_RANGE = (MIN_TARGET, MAX_TARGET)

MODE_DESCRIPTIONS = {
    "scale": "倍率",
    "total pixels": "总像素（MP）",
    "longer dimension": "长边像素",
    "shorter dimension": "短边像素",
}


def _clamp_target(w, h):
    """等比上限钳制 + 逐维下限抬升。返回 (w, h, clamped)。"""
    w, h = max(int(w), 1), max(int(h), 1)
    clamped = False
    if max(w, h) > MAX_TARGET:
        factor = MAX_TARGET / max(w, h)
        w = max(1, int(math.floor(w * factor)))
        h = max(1, int(math.floor(h * factor)))
        clamped = True
    if min(w, h) < MIN_TARGET:
        w, h = max(MIN_TARGET, w), max(MIN_TARGET, h)
        clamped = True
    return w, h, clamped


@dataclass(frozen=True)
class TargetSizeSpec:
    mode: str = "scale"
    scale: float = 4.0
    total_pixels: float = 1.0
    longer_size: int = 1024
    shorter_size: int = 1024

    def resolve(self, orig_w, orig_h):
        """解析目标尺寸，返回 (width, height, clamped)。非法输入抛 ValueError。"""
        if self.mode == "scale":
            wh = multiplier_to_wh(orig_w, orig_h, self.scale)
        elif self.mode == "total pixels":
            wh = total_pixels_to_wh(orig_w, orig_h, self.total_pixels)
        elif self.mode == "longer dimension":
            wh = longer_dimension_to_wh(orig_w, orig_h, self.longer_size)
        elif self.mode == "shorter dimension":
            wh = shorter_dimension_to_wh(orig_w, orig_h, self.shorter_size)
        else:
            raise ValueError(f"未知 size_mode: {self.mode!r}（可选 {SIZE_MODES}）")
        if wh is None:
            raise ValueError(
                f"无法解析目标尺寸：mode={self.mode!r}, 源图 {orig_w}x{orig_h}, "
                f"参数 scale={self.scale}, total_pixels={self.total_pixels}, "
                f"longer_size={self.longer_size}, shorter_size={self.shorter_size}"
            )
        return _clamp_target(*wh)

    def validate(self):
        """校验枚举与数值范围，返回错误信息（None = 合法）。"""
        if self.mode not in SIZE_MODES:
            return f"size_mode 必须是 {SIZE_MODES} 之一，得到 {self.mode!r}"
        try:
            scale = float(self.scale)
            total_pixels = float(self.total_pixels)
            longer = int(self.longer_size)
            shorter = int(self.shorter_size)
        except (TypeError, ValueError):
            return "scale/total_pixels/longer_size/shorter_size 必须是数值"
        if not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
            return f"scale 必须在 {SCALE_RANGE[0]}~{SCALE_RANGE[1]}，得到 {scale}"
        if not TOTAL_PIXELS_RANGE[0] <= total_pixels <= TOTAL_PIXELS_RANGE[1]:
            return f"total_pixels 必须在 {TOTAL_PIXELS_RANGE[0]}~{TOTAL_PIXELS_RANGE[1]} MP，得到 {total_pixels}"
        for name, value in (("longer_size", longer), ("shorter_size", shorter)):
            if not SIDE_RANGE[0] <= value <= SIDE_RANGE[1]:
                return f"{name} 必须在 {SIDE_RANGE[0]}~{SIDE_RANGE[1]}，得到 {value}"
        return None

    def describe(self):
        """人类可读的模式描述（日志用，不含源图尺寸）。"""
        if self.mode == "scale":
            return f"scale ×{float(self.scale):.2f}"
        if self.mode == "total pixels":
            return f"total pixels {float(self.total_pixels):.2f}MP"
        if self.mode == "longer dimension":
            return f"longer side {int(self.longer_size)}px"
        if self.mode == "shorter dimension":
            return f"shorter side {int(self.shorter_size)}px"
        return f"mode={self.mode!r}"


def coerce_target_size(target):
    """把节点/调用方的入参归一为 TargetSizeSpec（数字视作倍率，兼容旧调用）。"""
    if isinstance(target, TargetSizeSpec):
        return target
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        return TargetSizeSpec(mode="scale", scale=float(target))
    raise TypeError(f"target 必须是 TargetSizeSpec 或倍率数字，得到 {type(target).__name__}")
