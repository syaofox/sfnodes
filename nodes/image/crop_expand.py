"""SF Image Crop Expand — 复刻 ComfyUI-YCNodes_Toolkit ycImageCrop
(Load Image Crop Expand)。

在节点上直接加载图片（Load Image 按钮 / 拖放），拖拽一个可**出界**的裁剪框
（负坐标 / 超出边界 = 外扩区域），执行时输出 crop_w×crop_h 的画布：与源图的
交集贴回原像素，出界区域填充 fill_color；mask 白（1.0）= 扩展区、黑（0）=
原图区域——供外绘模型作为 inpaint 遮罩使用。

与原版的差异（已确认）：
  - 图片持久化：源图上传到 input/sfnodes_crop/（复用 SFImageCrop 的
    crop/upload_src 路由），隐藏 JSON 只存 src_path——工作流重载不丢图。
  - 不接受上游 IMAGE 输入（与原版一致，仅手动加载）。
  - 隐藏状态收敛为单个 SFCropExpandJson STRING（patterns §4 先例，
    替代原版 7 个隐藏 widget）；aspect_ratio 仅前端拖拽辅助，不进后端。

纯函数 _parse_state / _clamp_crop / _compose_expand 无 torch 依赖，可裸测。
"""

import os

import numpy as np
import torch
from PIL import Image

from ...sf_utils.common import _parse_fill_color, parse_json_dict as _parse_state  # 隐藏状态解析（单源，见 common）
from .crop import _safe_join

_CATEGORY = "sfnodes/image"

# 与原版 ycImageCrop 的输入域一致
_XY_LIMIT = 4096
_DIM_MAX = 8192


def _clamp_int(v, lo, hi, default=0):
    try:
        v = int(round(float(v)))
    except Exception:
        return default
    return max(lo, min(hi, v))


def _clamp_crop(meta):
    """Extract (x, y, w, h) from the state dict, clamped to the original
    node's declared domains (x/y ±4096, w/h 1..8192)."""
    x = _clamp_int(meta.get("crop_x", 0), -_XY_LIMIT, _XY_LIMIT, 0)
    y = _clamp_int(meta.get("crop_y", 0), -_XY_LIMIT, _XY_LIMIT, 0)
    w = _clamp_int(meta.get("crop_w", 512), 1, _DIM_MAX, 512)
    h = _clamp_int(meta.get("crop_h", 512), 1, _DIM_MAX, 512)
    return x, y, w, h


def _compose_expand(src, crop_x, crop_y, crop_w, crop_h, fill_rgb):
    """Composite the expansion canvas. Pure numpy.

    src: (H, W, 3) float32 0..1 RGB, or None (no source → pure fill canvas).
    Returns (image (crop_h, crop_w, 3) float32, mask (crop_h, crop_w) float32)
    with mask 1.0 = extended region (white), 0.0 = original-image region.
    """
    canvas = np.empty((crop_h, crop_w, 3), dtype=np.float32)
    canvas[...] = (np.array(fill_rgb, dtype=np.float32) / 255.0).reshape(1, 1, 3)

    mask = np.ones((crop_h, crop_w), dtype=np.float32)

    if src is not None and src.ndim == 3 and src.shape[2] >= 3:
        src = src[..., :3]
        sh, sw = int(src.shape[0]), int(src.shape[1])
        # Intersection of the source rect (0,0)-(sw,sh) and the crop rect.
        sx1 = max(0, crop_x)
        sy1 = max(0, crop_y)
        sx2 = min(sw, crop_x + crop_w)
        sy2 = min(sh, crop_y + crop_h)
        if sx2 > sx1 and sy2 > sy1:
            dx1 = sx1 - crop_x
            dy1 = sy1 - crop_y
            cw = sx2 - sx1
            ch = sy2 - sy1
            canvas[dy1:dy1 + ch, dx1:dx1 + cw, :] = src[sy1:sy2, sx1:sx2, :]
            mask[dy1:dy1 + ch, dx1:dx1 + cw] = 0.0

    return canvas, mask


class SFImageCropExpand:
    DESCRIPTION = (
        "在节点上直接加载图片（Load Image 按钮或拖放图片文件到节点），拖拽一个可"
        "超出图片边界的裁剪框——框内与原图重叠的部分保留原像素，出界区域填充纯色，"
        "mask 输出以白色标出扩展区域（黑=原图区域），供外绘模型作为重绘遮罩。\n\n"
        "面板提供常用比例（Free/1:1/16:9 等）与 Custom 自定义比例，非 Free 时拖拽"
        "保持比例；Color 按钮选择扩展区填充色（默认黑色，外绘建议中性灰）。\n\n"
        "图片持久化到 input/sfnodes_crop/，工作流保存/重载不丢图。输出 裁剪图、"
        "遮罩、宽、高，以及 filename——源图在 input 目录下的存储路径（可直连 "
        "LoadImage，未加载时为空串）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            # 隐藏状态输入：必须在 Python 侧声明，否则前端 validatePrompt 会把
            # schema 外输入从 prompt 剥离（patterns §4）。前端隐藏同名 STRING
            # widget 的值经标准 widget 通道收集。
            "hidden": {
                "SFCropExpandJson": ("STRING", {"default": "{}"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "filename")
    OUTPUT_TOOLTIPS = (
        "crop_w×crop_h 画布：与源图交集为原像素，出界区域为填充色",
        "白（1.0）=扩展区、黑（0.0）=原图区域——外绘重绘遮罩",
        "画布宽度（crop_w）",
        "画布高度（crop_h）",
        "源图在 input 目录下的存储路径（如 sfnodes_crop/crop_src_x.png），可直连 LoadImage；未加载源图时为空串",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    @classmethod
    def IS_CHANGED(cls, SFCropExpandJson="{}", **kwargs):
        """Re-run when the state (rect/fill/source file) changes. Keys on the
        source file's (mtime_ns, size) — patterns §3 禁 NaN."""
        meta = _parse_state(SFCropExpandJson)
        x, y, w, h = _clamp_crop(meta)
        key = f"{x}:{y}:{w}:{h}:{meta.get('fill_color', '')}"
        src_path = meta.get("src_path", "")
        full = _safe_join(src_path) if src_path else None
        if full and os.path.exists(full):
            st = os.stat(full)
            return f"{st.st_mtime_ns}:{st.st_size}:{key}"
        return key

    def _load_src(self, src_path):
        """Load the persisted source image as a float32 RGB array, or None."""
        full = _safe_join(src_path) if src_path else None
        if not full:
            return None
        try:
            pil = Image.open(full).convert("RGB")
            return np.array(pil).astype(np.float32) / 255.0
        except Exception as e:
            print(f"[SFImageCropExpand] source load failed: {e}")
            return None

    def execute(self, SFCropExpandJson="{}", **kwargs):
        meta = _parse_state(SFCropExpandJson)
        x, y, w, h = _clamp_crop(meta)
        try:
            fill_rgb = _parse_fill_color(meta.get("fill_color") or "#000000")
        except Exception:
            fill_rgb = (0, 0, 0)

        src = self._load_src(meta.get("src_path", ""))
        img_arr, mask_arr = _compose_expand(src, x, y, w, h, fill_rgb)

        image = torch.from_numpy(img_arr)[None,]   # [1, H, W, 3]
        mask = torch.from_numpy(mask_arr)[None,]   # [1, H, W]
        # filename 输出 = src_path 原样（input 相对路径，可直连 LoadImage）
        return (image, mask, w, h, meta.get("src_path", ""))
