"""SF Image Crop Expand Brush Mask — SFImageCropExpand 与 SFImageBrushMask 合体。

在节点上直接加载图片后，一个画布内完成外绘预处理的两件事：

  - 拖拽一个可**出界**的裁剪框（负坐标 / 超出边界 = 外扩区域），执行时输出
    crop_w×crop_h 画布：与源图交集贴回原像素，出界区域填充 fill_color；
  - 用画笔在源图上直接涂抹额外重绘区域（brush 涂白 / erase 擦除 / fill 多边形，
    与 SFImageBrushMask 完全同语义）。

mask 输出 = 扩展区 ∪ 笔触（白=重绘，恒二值）：扩展区结构性保留，Erase 只擦除
笔触层。笔触以**源图像素坐标**记录（前端钳制在源图内），随图移动，只有落在
裁剪框内的部分进入输出——与后端 `_compose_expand(overlay=)` 的贴回口径一致。

复用（零新增路由 / 零新依赖）：
  - `crop_expand._clamp_crop` / `_compose_expand(overlay=)`（扩展画布合成）
  - `sf_utils.brush_mask.parse_state_strokes` / `rasterize_strokes` / `lean_key`
  - `crop._safe_join` / `crop.load_src_rgb`（input/sfnodes_crop/ 源图读取）
  - `sf_utils.common._parse_fill_color` / `parse_json_dict as _parse_state`

状态收敛为单个隐藏输入 `SFCropExpandBrushMaskJson` STRING（前端 graphToPrompt
注入 lean 字段：src/crop/fill/strokes/brush_size；比例与画笔预览字段不进注入，
改画笔颜色不重跑）——patterns §4 先例。
"""

import os

import torch

from ...sf_utils.brush_mask import lean_key as _brush_lean_key
from ...sf_utils.brush_mask import parse_state_strokes, rasterize_strokes
from ...sf_utils.common import _parse_fill_color, parse_json_dict as _parse_state  # 隐藏状态解析（单源，见 common）
from .crop import _safe_join, load_src_rgb
from .crop_expand import _clamp_crop, _compose_expand

_CATEGORY = "sfnodes/image"

_HIDDEN_INPUT = "SFCropExpandBrushMaskJson"


def _state_key(meta):
    """IS_CHANGED 结果键（不含源文件签名）：裁剪域 + 填充色 + 笔触 lean
    （预览字段由 lean_key 排除）。"""
    x, y, w, h = _clamp_crop(meta)
    return f"{x}:{y}:{w}:{h}:{meta.get('fill_color', '')}|{_brush_lean_key(meta)}"


class SFImageCropExpandBrushMask:
    DESCRIPTION = (
        "在节点上直接加载图片（Load 按钮 / Browse 图片浏览器 / 拖放图片文件到"
        "节点 / Ctrl+V 粘贴），一个节点完成外绘预处理的两件事：拖拽一个可超出"
        "图片边界的裁剪框（出界区域填充纯色、框内与原图重叠处保留原像素），并用"
        "画笔在源图上直接涂抹需要额外重绘的区域。\n\n"
        "左列：裁剪比例预设（Free/1:1/16:9 等，非 Free 时拖拽保持比例）与 "
        "Custom 自定义比例（全局库，可保存/删除定义）、Reset（框回满幅）、Color"
        "（扩展区填充色）。右列：Crop/Brush/Erase 模式切换、Clear 清空与 Undo "
        "撤销笔触、笔刷 Size± 与预览 Opa± 步进（悬停滚轮快调；选中节点时 [ ] "
        "快捷键调尺寸）、BCol 笔刷取色。\n\n"
        "右键菜单可用文本 prompt 跑 SAM 分割（核心 SAM3_Detect，需 "
        "models/checkpoints/sam3.1_multiplex_fp16.safetensors），结果转为填充"
        "笔触并入列表统一管理（可擦除/撤销/清除）。工作流执行期间菜单不可用"
        "（避免与运行时模型加载并发冲突），请等任务结束再试。\n\n"
        "mask 输出 = 扩展区 ∪ 笔触（白=重绘，恒二值）——笔触以源图坐标记录并"
        "随图移动，只有落在裁剪框内的部分进入输出；Erase 只擦除笔触，扩展区"
        "始终保留（外绘掩码直接可用）。\n\n"
        "图片持久化到 input/sfnodes_crop/（复用 SFImageCrop 的上传路由），工作流"
        "保存/重载不丢图。输出 画布、遮罩、宽、高，以及 filename——源图在 input "
        "目录下的存储路径（可直连 LoadImage，未加载时为空串）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            # 隐藏状态输入：必须在 Python 侧声明，否则前端 validatePrompt 会把
            # schema 外输入从 prompt 剥离（patterns §4）。前端同名 STRING 值经
            # graphToPrompt 注入。
            "hidden": {
                _HIDDEN_INPUT: ("STRING", {"default": "{}"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "filename")
    OUTPUT_TOOLTIPS = (
        "crop_w×crop_h 画布：与源图交集为原像素，出界区域为填充色",
        "白（1.0）=重绘区：扩展区 ∪ 笔触——外绘/inpaint 遮罩",
        "画布宽度（crop_w）",
        "画布高度（crop_h）",
        "源图在 input 目录下的存储路径（如 sfnodes_crop/crop_src_x.png），可直连 LoadImage；未加载源图时为空串",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    @classmethod
    def IS_CHANGED(cls, SFCropExpandBrushMaskJson="{}", **kwargs):
        """Re-run when the crop/fill/strokes/source change. Keys on the source
        file's (mtime_ns, size) — patterns §3 禁 NaN；预览字段不进键。"""
        meta = _parse_state(SFCropExpandBrushMaskJson)
        key = _state_key(meta)
        src_path = meta.get("src_path", "")
        full = _safe_join(src_path) if src_path else None
        if full and os.path.exists(full):
            st = os.stat(full)
            return f"{st.st_mtime_ns}:{st.st_size}:{key}"
        return key

    def execute(self, SFCropExpandBrushMaskJson="{}", **kwargs):
        meta = _parse_state(SFCropExpandBrushMaskJson)
        x, y, w, h = _clamp_crop(meta)
        try:
            fill_rgb = _parse_fill_color(meta.get("fill_color") or "#000000")
        except Exception:
            fill_rgb = (0, 0, 0)

        src_path = meta.get("src_path", "") or ""
        src = load_src_rgb(src_path, "[SFImageCropExpandBrushMask]")

        # 笔触层：源图坐标系栅格化（brush/erase/fill 同 BrushMask 语义），交由
        # _compose_expand 在交集贴回处并入扩展区遮罩。缺源图时忽略（无坐标可贴）。
        overlay = None
        if src is not None and src.ndim == 3 and src.shape[2] >= 3:
            sh, sw = int(src.shape[0]), int(src.shape[1])
            try:
                brush_default = int(float(meta.get("brush_size", 80)))
            except Exception:
                brush_default = 80
            strokes = parse_state_strokes(
                {"src_w": sw, "src_h": sh, "brush_size": brush_default,
                 "strokes": meta.get("strokes", [])},
                brush_default,
            )
            if strokes:
                overlay = rasterize_strokes(strokes, sw, sh)

        img_arr, mask_arr = _compose_expand(src, x, y, w, h, fill_rgb, overlay)

        image = torch.from_numpy(img_arr)[None,]   # [1, H, W, 3]
        mask = torch.from_numpy(mask_arr)[None,]   # [1, H, W]
        # filename 输出 = src_path 原样（input 相对路径，可直连 LoadImage）
        return (image, mask, w, h, src_path)
