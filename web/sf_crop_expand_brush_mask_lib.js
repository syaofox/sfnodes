// ==========================================================================
// sf_crop_expand_brush_mask_lib.js - SF Image Crop Expand Brush Mask 组合布局纯逻辑
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_crop_expand_brush_mask.js 使用，也供 tests/ 复制为 .mjs 直接测试。
//
// 组合节点 = CropExpand 比例列（列1）+ BrushMask 工具列（列2，前置 Crop 模式）
// + 翻转/旋转列（列3，§112）的三列布局。本模块只放"多列组合"特有的常量与薄包装：
//   - 几何/交互数学全部复用 sf_crop_expand_lib.js（显示坐标系、手柄、拖拽、
//     比例约束、ensureMinSize、drawCropBox/drawPlaceholder）；
//   - 画笔步进/命中/渲染复用 sf_brush_mask_lib.js（TOOL_COL/step*/hitStepper/
//     paintStrokeMask/colorTextStyle）。
// 禁止在本模块重写上述公式（双端/多节点语义分叉是 bug 温床）。
// ==========================================================================

import {
  LAYOUT as CROP_LAYOUT,
  MIN_NODE_WIDTH as CROP_MIN_WIDTH,
  RATIO_PRESETS_COL,
  computeDisplayMetrics as _baseMetrics,
  ensureMinSize as _baseEnsureMin,
} from "./sf_crop_expand_lib.js";
import {
  LAYOUT as BRUSH_LAYOUT,
  TOOL_COL as BRUSH_TOOL_COL,
  COL_TOP,
  COL_STEP,
} from "./sf_brush_mask_lib.js";

// 三列布局：列1（比例预设）/ 列2（画笔工具）/ 列3（翻转旋转）各占
// ratioColW/toolColW/orientColW + 间距（列3 见 §112）。
export const LAYOUT = {
  shiftLeft: CROP_LAYOUT.shiftLeft,
  shiftRight: CROP_LAYOUT.shiftRight,
  ratioColW: CROP_LAYOUT.ratioColW,
  ratioColGap: CROP_LAYOUT.ratioColGap,
  toolColW: BRUSH_LAYOUT.toolColW,
  toolColGap: BRUSH_LAYOUT.toolColGap,
  orientColW: BRUSH_LAYOUT.toolColW,
  orientColGap: BRUSH_LAYOUT.toolColGap,
  bottomH: CROP_LAYOUT.bottomH,
};

// 列2/列3 起点 x（前一列面板右缘 + 间距）。
export const TOOL_COL_X = LAYOUT.shiftLeft + LAYOUT.ratioColW + LAYOUT.ratioColGap;
export const ORIENT_COL_X = TOOL_COL_X + LAYOUT.toolColW + LAYOUT.toolColGap;

// 显示区相对 CropExpand 额外让出的左侧宽度（列2 + 列3 宽与间距）。
export const EXTRA_LEFT = LAYOUT.toolColW + LAYOUT.toolColGap + LAYOUT.orientColW + LAYOUT.orientColGap;

// ── 列头与分组（§109）──────────────────────────────────────────────────
// 三列视觉分层：列头文字占 HEADER_H（首行从 COL_TOP+HEADER_H 起），组间在既有
// COL_GAP 基础上再多让 GROUP_EXTRA（分隔线画在让出空隙的中央，主扩展绘制）。
// 按钮顺序/几何方法不变，只挪 y——点击命中共用 buttonRect，一处生效。
export const HEADER_H = 10;
export const GROUP_EXTRA = 3;
export const FIRST_ROW_Y = COL_TOP + HEADER_H; // 26

// 每列按语义分组的项数（之和必须等于对应列的按钮数）：
// - 列1：比例预设 | Custom/Reset/Fill；
// - 列2：模式 | Clear/Undo/Invert | S±/O± | Pen；
// - 列3：翻转/旋转 4 项一组（无组间分隔）。
export const COL1_GROUPS = [RATIO_PRESETS_COL.length, 3]; // [8, 3]
export const COL2_GROUPS = [4, 3, 4, 1];                  // = TOOL_COL 12 项
export const COL3_GROUPS = [4];                           // = ORIENT_COL 4 项

// columnYs(groups, topY) → 逐项 y 数组（组内步进 COL_STEP，组间额外 GROUP_EXTRA）。
// 主扩展按此排布按钮，分组分隔线由同一数组推算（避免绘制/命中两套公式）。
export function columnYs(groups, topY = FIRST_ROW_Y) {
  const ys = [];
  let y = topY;
  groups.forEach((n, gi) => {
    if (gi > 0) y += GROUP_EXTRA;
    for (let i = 0; i < n; i++) {
      ys.push(y);
      y += COL_STEP;
    }
  });
  return ys;
}

// 最小节点尺寸：
// - 宽度：CropExpand 320（显示区 190）再加列2 + 列3 的 80 → 400；
// - 高度：由列2 的 12 项（Crop + BrushMask 11 项含 Poly，首行 26 起、步进 22、
//   三处分隔 +3、底 295）+ 底行 26 + 边距驱动 → 340（列1 11 项底 267、列3 4 项
//   底 110 都不是瓶颈；§101 加 Poly 前为 300、§90 起为 320，存量节点载入时
//   自动抬升见 §109；§112 加列3 只加宽不加高）。
export const MIN_NODE_WIDTH = CROP_MIN_WIDTH + EXTRA_LEFT; // 400
export const MIN_NODE_HEIGHT = 340;                        // 340（列2 12 项 + 列头/分隔）

// ensureMinSize(w, h) → [w, h]（组合节点下限；computeSize 包装与
// clampNodeSize 共用）。
export function ensureMinSize(w, h) {
  return _baseEnsureMin(w, h, MIN_NODE_WIDTH, MIN_NODE_HEIGHT);
}

// computeDisplayMetrics(state, nodeW, nodeH, frozen) → 显示坐标系
// （CropExpand 公式 + 列2/列3 让宽；拖拽冻结快照原样透传）。
export function computeDisplayMetrics(state, nodeW, nodeH, frozen) {
  return _baseMetrics(state, nodeW, nodeH, frozen, EXTRA_LEFT);
}

// 列2 按钮顺序：Crop 模式置顶 + BrushMask 工具列原序（Brush/Erase/Clear/Undo/
// Size±/Opa±/BCol），三模式为单选按钮。
export const TOOL_COL = ["crop", ...BRUSH_TOOL_COL];

// ── 源图翻转/旋转（列3 ORIENT，§112）─────────────────────────────────────
// 操作语义 = 源图整体变换：裁剪框与笔触随图联动重映射（笔触仍粘在画面内容
// 上），输出画布随之变化。本函数只算状态（纯几何，无 DOM）：主扩展据结果
// 重写 properties，源图文件由 sf_crop_source.orientSource 重绘上传。
// 坐标口径与后端 rasterize_strokes 一致：裁剪框为连续坐标（允许出界），
// 笔触点为整型像素索引（0..W-1）→ 镜像用 W-1-x、旋转用 (H-1-y, x)。
// 旋转 90° 后宽高互换，比例预设不再成立 → aspect_ratio 复位 "free"（翻转不动）。
export const ORIENT_COL = ["flipH", "flipV", "rotL", "rotR"];

// orientState(st, op) → 变换后的状态补丁（含 strokes/aspect_ratio）或 null
// （维度非法/未知 op）。不改入参（笔触逐层新对象）。
export function orientState(st, op) {
  if (!ORIENT_COL.includes(op)) return null;
  const W = Number(st?.src_w) || 0;
  const H = Number(st?.src_h) || 0;
  if (W <= 0 || H <= 0) return null;
  const x = Number(st.crop_x) || 0;
  const y = Number(st.crop_y) || 0;
  const w = Number(st.crop_w) || 0;
  const h = Number(st.crop_h) || 0;
  const strokes = (Array.isArray(st.strokes) ? st.strokes : []).map((s) => ({
    ...s,
    points: (Array.isArray(s.points) ? s.points : []).map(([px, py]) => {
      const X = Number(px) || 0;
      const Y = Number(py) || 0;
      if (op === "flipH") return [W - 1 - X, Y];
      if (op === "flipV") return [X, H - 1 - Y];
      if (op === "rotL") return [Y, W - 1 - X];
      return [H - 1 - Y, X]; // rotR
    }),
  }));
  const aspect = st.aspect_ratio || "free";
  if (op === "flipH") {
    return { src_w: W, src_h: H, crop_x: W - x - w, crop_y: y, crop_w: w, crop_h: h, strokes, aspect_ratio: aspect };
  }
  if (op === "flipV") {
    return { src_w: W, src_h: H, crop_x: x, crop_y: H - y - h, crop_w: w, crop_h: h, strokes, aspect_ratio: aspect };
  }
  if (op === "rotL") {
    return { src_w: H, src_h: W, crop_x: y, crop_y: W - x - w, crop_w: h, crop_h: w, strokes, aspect_ratio: "free" };
  }
  return { src_w: H, src_h: W, crop_x: H - y - h, crop_y: x, crop_w: h, crop_h: w, strokes, aspect_ratio: "free" }; // rotR
}
