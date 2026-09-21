"""SF Image Crop Expand Brush Mask — SFImageCropExpand 与 SFImageBrushMask 合体。

在节点上直接加载图片后，一个画布内完成外绘预处理的两件事：

  - 拖拽一个可**出界**的裁剪框（负坐标 / 超出边界 = 外扩区域），执行时输出
    crop_w×crop_h 画布：与源图交集贴回原像素，出界区域填充 fill_color；
  - 用画笔在源图上直接涂抹额外重绘区域（brush 涂白 / erase 擦除 / fill 多边形
    添加 / fill_erase 多边形打洞，与 SFImageBrushMask 完全同语义）。

mask 输出 = 扩展区 ∪ 笔触（白=重绘，恒二值）：扩展区默认结构性保留（Ext 开关
可关掉并入，此时 mask = 笔触层），Erase 只擦除笔触层。笔触以**源图像素坐标**
记录（前端钳制在源图内），随图移动，只有落在裁剪框内的部分进入输出——与后端
`_compose_expand(overlay=)` 的贴回口径一致。

复用（零新增路由 / 零新依赖）：
  - `crop_expand._clamp_crop` / `_compose_expand(overlay=, include_ext=)`（扩展画布合成）
  - `sf_utils.brush_mask.parse_state_strokes` / `rasterize_strokes` / `lean_key`
  - `crop._safe_join` / `crop.load_src_rgb`（input/sfnodes_crop/ 源图读取）
  - `sf_utils.common._parse_fill_color` / `parse_json_dict as _parse_state`

状态收敛为单个隐藏输入 `SFCropExpandBrushMaskJson` STRING（前端 graphToPrompt
注入 lean 字段：src/crop/fill/strokes/brush_size/invert/include_ext；比例与画笔
预览字段不进注入，改画笔颜色不重跑）——patterns §4 先例。

可选接线宽高比（§118）：`aspect_w` / `aspect_h` 两个 INT 输入，两项都接且为
正整数时，裁剪框优先保持该比例（宽度为准、高度自动并垂直居中）——前端接入
即同步刷新裁剪框，后端 execute 用 `_apply_aspect_ratio` 同公式兜底（上游值
前端读不到时输出仍守比例）。见 experience/nodes-image.md §118。
"""

import math
import os

import numpy as np
import torch

from ...sf_utils.brush_mask import lean_key as _brush_lean_key
from ...sf_utils.brush_mask import parse_state_strokes, rasterize_strokes
from ...sf_utils.common import _parse_fill_color, parse_json_dict as _parse_state  # 隐藏状态解析（单源，见 common）
from .crop import _safe_join, load_src_rgb
from .crop_expand import _clamp_crop, _compose_expand, _DIM_MAX

_CATEGORY = "sfnodes/image"

_HIDDEN_INPUT = "SFCropExpandBrushMaskJson"

# 接线比例套用的高度下限（镜像前端 sf_crop_expand_lib.MIN_SIZE=10，
# 保证预览与输出同公式；`_clamp_crop` 的域下限 1 更大）
_RATIO_MIN_H = 10


def _js_round(v):
    """Math.round 镜像（.5 向上取整；Python round 是银行家舍入，不能直接用）。"""
    return int(math.floor(v + 0.5))


def _apply_aspect_ratio(x, y, w, h, aspect_w, aspect_h):
    """接线宽高比优先（§118）：`aspect_w`/`aspect_h` 两项都接且为正整数时，
    保持 x/w（宽度为准），高度按 `w / (aspect_w / aspect_h)` 计算并在原中心
    垂直居中——逐行镜像前端 `sf_crop_expand_lib.applyRatioToRect`。

    缺项 / 非数 / ≤0（只接一项、断开、上游缺值）原样返回，此时面板比例状态
    照常生效。高度另受 `_DIM_MAX` 夹紧防极端比例爆画布（前端无上限夹紧，
    仅 1:1000 类极端比例下预览与输出会差一个上限，安全性优先）。"""
    try:
        rw = int(aspect_w)
        rh = int(aspect_h)
    except (TypeError, ValueError):
        return x, y, w, h
    if rw <= 0 or rh <= 0:
        return x, y, w, h
    root = rw / rh
    new_h = max(_RATIO_MIN_H, min(_DIM_MAX, _js_round(w / root)))
    new_y = _js_round((y + h / 2) - new_h / 2)
    return x, new_y, w, new_h


def _state_key(meta):
    """IS_CHANGED 结果键（不含源文件签名）：裁剪域 + 填充色 + 笔触 lean
    （预览字段由 lean_key 排除）+ 扩展区开关（Ext，§113）。"""
    x, y, w, h = _clamp_crop(meta)
    ext = "1" if meta.get("include_ext", True) else "0"
    return f"{x}:{y}:{w}:{h}:{meta.get('fill_color', '')}|{_brush_lean_key(meta)}|ext={ext}"


class SFImageCropExpandBrushMask:
    DESCRIPTION = (
        "在节点上直接加载图片（Load 按钮 / Browse 图片浏览器 / 拖放图片文件到"
        "节点 / Ctrl+V 粘贴），一个节点完成外绘预处理的两件事：拖拽一个可超出"
        "图片边界的裁剪框（出界区域填充纯色、框内与原图重叠处保留原像素），并用"
        "画笔在源图上直接涂抹需要额外重绘的区域。\n\n"
        "左列：裁剪比例预设（Free/1:1/16:9 等，非 Free 时拖拽保持比例）与 "
        "Custom 自定义比例（全局库，可保存/删除定义）、Reset（框回满幅）、Color"
        "（扩展区填充色）。中列：Crop/Brush/Erase 模式切换、Clear 清空与 Undo "
        "撤销笔触、Invert 反选、Ext 扩展区遮罩开关（默认 ON=扩展区计入 mask；"
        "OFF 时 mask 只含笔触层、扩展区不重绘）、笔刷 Size± 与预览 Opa± 步进"
        "（悬停滚轮快调；选中节点时 C/B/E 快捷键切模式、[ ] 调尺寸）、Pen 笔刷"
        "取色。右列："
        "FlipH/FlipV 水平/垂直翻转与 RotL/RotR 逆/顺时针旋转 90°——源图整体"
        "变换（裁剪框与笔触随图联动、笔触仍粘在画面内容上），每次操作把变换后"
        "的源图另存为新文件并自动更新输出；旋转 90° 后比例预设复位为 Free。"
        "裁剪框"
        "（框线/九宫格/手柄/框外压暗）只在 Crop 模式显示，Brush/Erase 涂抹时"
        "自动隐藏以免干扰观察（扩展区白提示保留）。线条粗细可在设置页调："
        "sfnodes.Canvas.FrameWidth（框线/边界虚线/SAM 框选）与 "
        "sfnodes.Canvas.CursorWidth（画笔光环），默认均为 1.0。\n\n"
        "右键菜单：SAM 文本/点选/框选分割（核心 SAM3_Detect，需 "
        "models/checkpoints/sam3.1_multiplex_fp16.safetensors；点选左键=正点、"
        "Shift+左键=负点、Enter 执行）、人物部位遮罩（MediaPipe）、YOLO 检测/分割"
        "（models/ultralytics/{bbox,segm} 权重，需已装 ultralytics）、导入遮罩"
        "文件为笔触、反选遮罩（面板 Invert 按钮或右键菜单切换，ON 时按钮呈琥珀色）。"
        "反选只作用于笔触层：Ext ON（扩展区计入）时 mask = 扩展区 ∪ (1 - 笔触)，"
        "扩展区保留重绘；Ext OFF 时源图交集内 = 1 - 笔触、扩展区仍不重绘。"
        "识别/导入结果按当前模式并入列表统一管理（可擦除/"
        "撤销/清除）：Brush（及 Crop）模式转为填充笔触添加；Eraser 模式转为"
        "打洞笔触，从现有遮罩（笔触层）中减去识别区域（导入遮罩同样跟随）。"
        "工作流执行期间模型类菜单不可用（避免与运行时模型加载并发冲突），"
        "请等任务结束再试。\n\n"
        "mask 输出 = 扩展区 ∪ 笔触（白=重绘，恒二值；Ext 开关 OFF 时 = 笔触层）"
        "——笔触以源图坐标记录并随图移动，只有落在裁剪框内的部分进入输出；"
        "Erase 只擦除笔触，扩展区默认保留（外绘掩码直接可用）。\n\n"
        "可选接线宽高比：aspect_w / aspect_h 两个 INT 输入（如接分辨率节点的宽高）。"
        "两项都接且为正整数时，裁剪框优先保持该比例——以宽度为准、高度自动并在"
        "原中心垂直居中；接线期间面板比例预设只记住不生效（断开后恢复）。上游值"
        "前端读不到时（动态值）预览不约束，但执行时后端仍按接线值修正输出。\n\n"
        "图片持久化到 input/sfnodes_crop/（复用 SFImageCrop 的上传路由），工作流"
        "保存/重载不丢图。输出 画布、遮罩、宽、高，以及 filename——源图在 input "
        "目录下的存储路径（可直连 LoadImage，未加载时为空串）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            # 可选接线宽高比（§118）：forceInput 无 widget（widgets_values 位置
            # 不变，旧工作流零影响）；两项都接才生效，值经 execute 兜底修正。
            "optional": {
                "aspect_w": ("INT", {"forceInput": True, "tooltip": "可选宽高比分子（如分辨率节点的宽度）。与 aspect_h 同时接线时裁剪框优先保持 aspect_w:aspect_h——宽度为准、高度自动；接线期间面板比例预设只记住不生效。"}),
                "aspect_h": ("INT", {"forceInput": True, "tooltip": "可选宽高比分母（如分辨率节点的高度）。与 aspect_w 同时接线时裁剪框优先保持该比例；上游值前端不可读时执行阶段仍按接线值修正输出。"}),
            },
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
        """Re-run when the crop/fill/strokes/source/Ext switch change. Keys on the
        source file's (mtime_ns, size) — patterns §3 禁 NaN；预览字段不进键。"""
        meta = _parse_state(SFCropExpandBrushMaskJson)
        key = _state_key(meta)
        src_path = meta.get("src_path", "")
        full = _safe_join(src_path) if src_path else None
        if full and os.path.exists(full):
            st = os.stat(full)
            return f"{st.st_mtime_ns}:{st.st_size}:{key}"
        return key

    def execute(self, SFCropExpandBrushMaskJson="{}", aspect_w=None, aspect_h=None, **kwargs):
        meta = _parse_state(SFCropExpandBrushMaskJson)
        x, y, w, h = _clamp_crop(meta)
        # 接线宽高比优先（§118）：两项都接且为正整数时修正 x/y/w/h
        # （宽度为准、垂直居中，镜像前端 applyRatioToRect）
        x, y, w, h = _apply_aspect_ratio(x, y, w, h, aspect_w, aspect_h)
        # Ext 开关（默认 True=扩展区计入遮罩；前端面板切换，§113）
        include_ext = bool(meta.get("include_ext", True))
        try:
            fill_rgb = _parse_fill_color(meta.get("fill_color") or "#000000")
        except Exception:
            fill_rgb = (0, 0, 0)

        src_path = meta.get("src_path", "") or ""
        src = load_src_rgb(src_path, "[SFImageCropExpandBrushMask]")

        # 笔触层：源图坐标系栅格化（brush/erase/fill/fill_erase 同 BrushMask
        # 语义），交由 _compose_expand 在交集贴回处并入扩展区遮罩。缺源图时忽略。
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

        img_arr, mask_arr = _compose_expand(src, x, y, w, h, fill_rgb, overlay,
                                            include_ext=include_ext)
        if meta.get("invert"):
            # 反选只作用于笔触层（§113 与 Ext 开关联动）：
            # - Ext ON：扩展区 ∪ (1 - 笔触)（扩展区是结构性重绘区，不被反选取消）；
            # - Ext OFF：扩展区恒黑（反选不把它带回来），源图交集内 = 1 - 笔触。
            # inv 先乘 (1 - 扩展区) 把反选结果圈定在源图交集内，一处公式覆盖两态。
            _, ext_mask = _compose_expand(src, x, y, w, h, fill_rgb, None)
            inv = (1.0 - mask_arr) * (1.0 - ext_mask)
            mask_arr = np.maximum(ext_mask, inv) if include_ext else inv

        image = torch.from_numpy(img_arr)[None,]   # [1, H, W, 3]
        mask = torch.from_numpy(mask_arr)[None,]   # [1, H, W]
        # filename 输出 = src_path 原样（input 相对路径，可直连 LoadImage）
        return (image, mask, w, h, src_path)
