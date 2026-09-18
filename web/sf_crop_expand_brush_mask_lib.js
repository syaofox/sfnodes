// ==========================================================================
// sf_crop_expand_brush_mask_lib.js - SF Image Crop Expand Brush Mask 组合布局纯逻辑
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_crop_expand_brush_mask.js 使用，也供 tests/ 复制为 .mjs 直接测试。
//
// 组合节点 = CropExpand 比例列（列1）+ BrushMask 工具列（列2，前置 Crop 模式）
// 的双列布局。本模块只放"两列组合"特有的常量与薄包装：
//   - 几何/交互数学全部复用 sf_crop_expand_lib.js（显示坐标系、手柄、拖拽、
//     比例约束、ensureMinSize、drawCropBox/drawPlaceholder）；
//   - 画笔步进/命中/渲染复用 sf_brush_mask_lib.js（TOOL_COL/step*/hitStepper/
//     paintStrokeMask/colorTextStyle）。
// 禁止在本模块重写上述公式（双端/多节点语义分叉是 bug 温床）。
// ==========================================================================

import {
  LAYOUT as CROP_LAYOUT,
  MIN_NODE_WIDTH as CROP_MIN_WIDTH,
  computeDisplayMetrics as _baseMetrics,
  ensureMinSize as _baseEnsureMin,
} from "./sf_crop_expand_lib.js";
import { LAYOUT as BRUSH_LAYOUT, TOOL_COL as BRUSH_TOOL_COL } from "./sf_brush_mask_lib.js";

// 双列布局：列1（比例预设）与列2（画笔工具）各占 ratioColW/toolColW + 间距。
export const LAYOUT = {
  shiftLeft: CROP_LAYOUT.shiftLeft,
  shiftRight: CROP_LAYOUT.shiftRight,
  ratioColW: CROP_LAYOUT.ratioColW,
  ratioColGap: CROP_LAYOUT.ratioColGap,
  toolColW: BRUSH_LAYOUT.toolColW,
  toolColGap: BRUSH_LAYOUT.toolColGap,
  bottomH: CROP_LAYOUT.bottomH,
};

// 列2 起点 x（列1 面板右缘 + 间距）。
export const TOOL_COL_X = LAYOUT.shiftLeft + LAYOUT.ratioColW + LAYOUT.ratioColGap;

// 显示区相对 CropExpand 额外让出的左侧宽度（列2 宽 + 间距）。
export const EXTRA_LEFT = LAYOUT.toolColW + LAYOUT.toolColGap;

// 最小节点尺寸：
// - 宽度：CropExpand 320（显示区 190）再加列2 的 40 → 360；
// - 高度：由列2 的 12 项（Crop + BrushMask 11 项含 Poly，colTop=16 起、步进
//   22、底 276）+ 底行 26 + 边距驱动 → 320（列1 11 项底 254 不再是瓶颈；
//   Poly 加入前为 300，存量节点载入时自动抬升，见 §101）。
export const MIN_NODE_WIDTH = CROP_MIN_WIDTH + EXTRA_LEFT; // 360
export const MIN_NODE_HEIGHT = 320;                        // 320（列2 12 项）

// ensureMinSize(w, h) → [w, h]（组合节点下限；computeSize 包装与
// clampNodeSize 共用）。
export function ensureMinSize(w, h) {
  return _baseEnsureMin(w, h, MIN_NODE_WIDTH, MIN_NODE_HEIGHT);
}

// computeDisplayMetrics(state, nodeW, nodeH, frozen) → 显示坐标系
// （CropExpand 公式 + 列2 让宽；拖拽冻结快照原样透传）。
export function computeDisplayMetrics(state, nodeW, nodeH, frozen) {
  return _baseMetrics(state, nodeW, nodeH, frozen, EXTRA_LEFT);
}

// 列2 按钮顺序：Crop 模式置顶 + BrushMask 工具列原序（Brush/Erase/Clear/Undo/
// Size±/Opa±/BCol），三模式为单选按钮。
export const TOOL_COL = ["crop", ...BRUSH_TOOL_COL];
