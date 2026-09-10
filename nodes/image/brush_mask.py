"""SF Image Brush Mask — 复刻 ComfyUI-YCNodes_Toolkit ycimagebrushmask
(Load Image Brush Mask)。

在节点上直接加载图片（Load Image 按钮 / Browse 图片浏览器 / 拖放 /
Ctrl+V 粘贴），用画笔在图上直接绘制遮罩（brush=白/erase=擦除），执行时
输出源图 + 二值遮罩 + 宽高 + filename。

与原版的差异（已确认）：
  - 图片持久化：源图经 ``/api/sfnodes/crop/upload_src`` 落盘到
    ``input/sfnodes_crop/``（复用 SFImageCrop 的路由与 ``_safe_join``），
    隐藏 JSON 只存 ``src_path``——工作流保存/重载/刷新不丢图。原版把
    base64 塞进 widget（工作流巨大需手动清图）+ 会话内 Map 缓存（刷新即丢）。
  - 状态收敛为单个 ``SFBrushMaskJson`` STRING（patterns §4 先例，替代原版
    brush_data/brush_size/image_base64/image_width/image_height 5 个 widget）；
    前端经 graphToPrompt 注入（只注入影响结果的 lean 字段：src/strokes，
    预览用的 opacity/color 不进注入——改颜色不重跑）。
  - mask 恒二值（brush 置 1 / erase 置 0）：Opacity 与取色仅前端预览透明度，
    后端忽略（与原版语义一致，DESCRIPTION 注明）。
  - 输出多一个 ``filename``（src_path 原样 input 相对路径，可直连 LoadImage，
    未加载时为空串）。

纯函数 ``_parse_state`` 无 torch 依赖，可裸测；栅格化在
``sf_utils/brush_mask.py``（原版绘制语义的纯逻辑移植）。
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from ...sf_utils.brush_mask import parse_state_strokes
from .crop import _safe_join
from . import brush_mask_sam  # noqa: F401  # 副作用注册 /api/sfnodes/brush_mask/* 路由

_CATEGORY = "sfnodes/image"

_HIDDEN_INPUT = "SFBrushMaskJson"


def _parse_state(raw):
    """Parse the hidden SFBrushMaskJson STRING into a dict ({} on failure)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _lean_key(meta):
    """Stable cache key over result-affecting fields only (strokes + source).

    预览字段（opacity/color）故意排除——改预览不重跑（text §6 lean 注入先例）。
    SAM 结果即 fill 笔触（strokes 内），无额外键。
    """
    strokes = meta.get("strokes", [])
    try:
        strokes_key = json.dumps(strokes, sort_keys=True, separators=(",", ":"))
    except Exception:
        strokes_key = str(strokes)
    try:
        brush_size = int(float(meta.get("brush_size", 80)))
    except Exception:
        brush_size = 80
    return f"{meta.get('src_path', '')}|{meta.get('src_w', '')}|{meta.get('src_h', '')}|{brush_size}|{strokes_key}"


class SFImageBrushMask:
    DESCRIPTION = (
        "在节点上直接加载图片（Load Image 按钮 / Browse 图片浏览器 / 拖放图片"
        "文件到节点 / Ctrl+V 粘贴），用画笔在图上直接涂抹遮罩，无需打开遮罩"
        "编辑器。Brush 涂白（遮罩=1），Eraser 擦除；Clear 清空全部笔触，Undo "
        "撤销上一笔。\n\n"
        "Size 步进（S±，可悬停滚轮快调）调节笔刷直径；Opacity 与取色块仅改变预览叠加的透明度/颜色，"
        "不影响输出（输出遮罩恒为二值）。\n\n"
        "右键菜单可用文本 prompt 跑 SAM 分割（核心 SAM3_Detect，需 "
        "models/checkpoints/sam3.1_multiplex_fp16.safetensors），结果转为填充"
        "笔触并入列表统一管理（可擦除/撤销/清除）。\n\n"
        "图片持久化到 input/sfnodes_crop/，工作流保存/重载/刷新不丢图。输出 "
        "原图、遮罩、宽、高，以及 filename——源图在 input 目录下的存储路径"
        "（可直连 LoadImage，未加载时为空串）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            # 隐藏状态输入：必须在 Python 侧声明，否则前端 validatePrompt 会把
            # schema 外输入从 prompt 剥离（patterns §4）。前端隐藏同名 STRING
            # widget 的值经标准 widget 通道收集。
            "hidden": {
                _HIDDEN_INPUT: ("STRING", {"default": "{}"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "filename")
    OUTPUT_TOOLTIPS = (
        "加载的源图（未加载时为 512×512 空白图）",
        "笔触遮罩：白（1.0）=涂抹区、黑（0.0）=背景，恒二值",
        "源图宽度",
        "源图高度",
        "源图在 input 目录下的存储路径（如 sfnodes_crop/crop_src_x.png），可直连 LoadImage；未加载源图时为空串",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    @classmethod
    def IS_CHANGED(cls, SFBrushMaskJson="{}", **kwargs):
        """Re-run when strokes/source change. Keys on the source file's
        (mtime_ns, size) — patterns §3 禁 NaN；预览字段不进键。"""
        meta = _parse_state(SFBrushMaskJson)
        key = _lean_key(meta)
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
            print(f"[SFImageBrushMask] source load failed: {e}")
            return None

    def execute(self, SFBrushMaskJson="{}", **kwargs):
        from ...sf_utils.brush_mask import rasterize_strokes

        meta = _parse_state(SFBrushMaskJson)
        src_path = meta.get("src_path", "") or ""
        try:
            brush_default = int(float(meta.get("brush_size", 80)))
        except Exception:
            brush_default = 80

        src = self._load_src(src_path)
        if src is not None and src.ndim == 3 and src.shape[2] >= 3:
            src = src[..., :3]
            sh, sw = int(src.shape[0]), int(src.shape[1])
            image = torch.from_numpy(src)[None,]  # [1, H, W, 3]
        else:
            sw, sh = 512, 512
            image = torch.zeros((1, sh, sw, 3), dtype=torch.float32)

        # 状态 strokes 已是结构化数值；用纯逻辑做裁剪+栅格化（与旧串格式同语义）。
        state = {"src_w": sw, "src_h": sh, "brush_size": brush_default,
                 "strokes": meta.get("strokes", [])}
        strokes = parse_state_strokes(state, brush_default)
        mask_arr = rasterize_strokes(strokes, sw, sh)
        mask = torch.from_numpy(mask_arr)[None,]  # [1, H, W]
        return (image, mask, sw, sh, src_path)
