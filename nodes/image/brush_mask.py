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

纯函数解析/键值/栅格化全在 ``sf_utils/brush_mask.py``（``_parse_state`` 见
``common.parse_json_dict``；``_lean_key`` 已提升为 ``sf_utils.brush_mask.lean_key``
公共实现，与 SFImageCropExpandBrushMask 共用；``_load_src`` 收敛到
``crop.load_src_rgb``）。
"""

import os

import torch

from ...sf_utils.brush_mask import lean_key as _lean_key
from ...sf_utils.brush_mask import parse_state_strokes, rasterize_strokes
from ...sf_utils.common import parse_json_dict as _parse_state  # 隐藏状态解析（单源，见 common）
from .crop import _safe_join
from .crop import load_src_rgb
from . import brush_mask_sam  # noqa: F401  # 副作用注册 /api/sfnodes/brush_mask/* 路由
from . import brush_mask_tools  # noqa: F401  # 人物部位/YOLO/导入遮罩/统一卸载路由

_CATEGORY = "sfnodes/image"

_HIDDEN_INPUT = "SFBrushMaskJson"


class SFImageBrushMask:
    DESCRIPTION = (
        "在节点上直接加载图片（Load Image 按钮 / Browse 图片浏览器 / 拖放图片"
        "文件到节点 / Ctrl+V 粘贴），用画笔在图上直接涂抹遮罩，无需打开遮罩"
        "编辑器。Brush 涂白（遮罩=1），Eraser 擦除；Clear 清空全部笔触，Undo "
        "撤销上一笔。选中节点时快捷键：B 切 Brush、E 切 Eraser、[ ] 调节笔刷尺寸。\n\n"
        "Size 步进（S±，可悬停滚轮快调）调节笔刷直径；Opacity 与取色仅改变预览叠加的透明度/颜色，"
        "不影响输出（输出遮罩恒为二值）。\n\n"
        "右键菜单：SAM 文本/点选/框选分割（核心 SAM3_Detect，需 "
        "models/checkpoints/sam3.1_multiplex_fp16.safetensors；点选左键=正点、"
        "Shift+左键=负点、Enter 执行）、人物部位遮罩（MediaPipe：脸/发/身体/衣服/"
        "背景）、YOLO 检测/分割（models/ultralytics/{bbox,segm} 权重，需已装 "
        "ultralytics）、导入遮罩文件为笔触、反选遮罩（面板 Invert 按钮或右键菜单"
        "切换，ON 时按钮呈琥珀色提示输出已取反），结果均转为"
        "填充笔触并入列表统一管理（可擦除/撤销/清除）。多人用 person:3（:N 为"
        "每类最多检出数），多类用逗号分隔。工作流执行期间模型类菜单不可用"
        "（避免与运行时模型加载并发冲突），请等任务结束再试。\n\n"
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
        """Load the persisted source image as a float32 RGB array, or None
        (shared impl in crop.load_src_rgb, converged from three copies)."""
        return load_src_rgb(src_path, "[SFImageBrushMask]")

    def execute(self, SFBrushMaskJson="{}", **kwargs):
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
        if meta.get("invert"):
            mask_arr = 1.0 - mask_arr  # 反选：笔触层取反（空笔触 → 全白）
        mask = torch.from_numpy(mask_arr)[None,]  # [1, H, W]
        return (image, mask, sw, sh, src_path)
