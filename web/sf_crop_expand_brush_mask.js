// ==========================================================================
// sf_crop_expand_brush_mask.js - SF Image Crop Expand Brush Mask 主扩展
// ==========================================================================
//
// SFImageCropExpand + SFImageBrushMask 合体：节点上加载图片后，一个画布内完成
//   ① 拖拽可出界的裁剪框（比例预设 / Reset / 填充色；出界区 = 扩展区）
//   ② 画笔涂抹额外重绘区（Crop/Brush/Erase 三模式；笔触以源图像素坐标记录，
//      随图移动，只有落在裁剪框内的部分进入输出）
// 输出扩展画布 + 并集遮罩（扩展区 ∪ 笔触，白=重绘，恒二值；面板 Ext 开关
// 可关掉扩展区并入，此时 mask = 笔触层，见 §113）。
//
// 三列布局（sf_crop_expand_brush_mask_lib）：列1 = 比例预设（CropExpand 同款），
// 列2 = 画笔工具（BrushMask TOOL_COL 前置 Crop 模式），列3 = 翻转/旋转
// （源图整体联动，§112）；显示区多让两列宽度。三列带列头（RATIO/TOOLS/
// ORIENT）与组间分隔线、底部歧义按钮重命名（Fill/Pen）、悬停底行改显按钮
// 中文说明，见 §109·§112。
//
// 状态真源 node.properties.sfCropExpandBrushMaskState（JSON 字符串，随工作流
// 保存），经 graphToPrompt 钩子注入隐藏输入 SFCropExpandBrushMaskJson（只注入
// lean 字段：src/crop/fill/brush_size/strokes——比例与画笔预览字段不进注入，
// 改画笔颜色不重跑；Python hidden 已声明，schema 内不被剥离）。
//
// 接线宽高比（可选输入 aspect_w/aspect_h，§118）：两项都接且可读时裁剪框
// 优先保持该比例——接入即时同步刷新（onConnectionsChange）、上游值变化下一拍
// 跟随、加载路径 onAfterGraphConfigured 补同步、节点内加载图片（onStored）后
// 即时重套；接线期间面板预设只记住不生效；槽名 ZW 隐藏（核心渲染会画在自定义
// 面板之上），信息栏显示 AR。
//
// 共享实现（禁止内联副本）：
//   - 几何/冻结快照防飘移/手柄/绘制：sf_crop_expand_lib.js
//   - 步进/命中/笔触渲染：sf_brush_mask_lib.js（paintStrokeMask/colorTextStyle）
//   - 组合布局：sf_crop_expand_brush_mask_lib.js
//   - 源图链路：sf_crop_source.js（Load/Browse/拖放/Ctrl+V → input/sfnodes_crop/；
//     翻/转经 orientSource 重绘上传新源图）
//   - 比例预设弹窗：sf_crop_expand_ratios.js
//   - AI/工具右键菜单：sf_brush_ai.js（与 SFImageBrushMask 单源：SAM 文本/
//     点选/框选、人物部位、YOLO、导入遮罩、反选、统一卸载；识别/导入结果按
//     当前模式并入——Eraser 模式 fill_erase 打洞减去（Crop/Brush 添加）；
//     工作流执行期间后端 409 熔断，见 experience/nodes-image.md §91·§93·§99）
//   - 多边形套索：sf_brush_poly.js（与 SFImageBrushMask 单源；Brush=fill 添加 /
//     Eraser=fill_erase 打洞，Crop 模式点击按钮自动切 Brush；落点限源图内，
//     见 experience/nodes-image.md §101）
//   - 画笔步长设置/[ ]/滚轮：sf_brush_tools.js
//   - 图索引：sf_pause_kit.js（buildClassNodeIndex/findNodeByPromptId 单源）
//   - 释放兜底/cursor 补丁：sf_common.js
// ==========================================================================

import { app } from "/scripts/app.js";
import {
  getSfAccent,
  sfThemeColors,
  installPasteHandler,
  primaryButtonReleased,
  installNodeReleaseGuard,
  removeNodeReleaseGuard,
  installResizeCornerCursor,
  pickColorInput,
  rgbStringToHex,
  hexToRgbString,
  sfFrameWidth,
  sfFrameThin,
  sfCursorWidth,
  registerSfLineWidthSettings,
  isGraphLoading,
  sfToast,
} from "./sf_common.js";
import { ZW } from "./sf_dropdown_lib.js";
import { pickFile, browseSource, restoreSourceImage, installSourceDrop, storeSource, orientSource } from "./sf_crop_source.js";
import { openCustomRatioDialog, ratioLabel } from "./sf_crop_expand_ratios.js";
import { installBrushMenu, handleSamPointer, drawSamOverlay, toggleInvert, cancelSamMode } from "./sf_brush_ai.js";
import { togglePoly, handlePolyPointer, handlePolyDblClick, drawPolyOverlay, cancelPoly, disposePoly } from "./sf_brush_poly.js";
import { buildClassNodeIndex, findNodeByPromptId } from "./sf_pause_kit.js";
import {
  RATIO_PRESETS_COL,
  ratioFromAspect,
  localToImage,
  imageToLocal,
  getHandleAtPoint,
  getCursorForHandle,
  updateCropByDrag,
  roundRect,
  applyRatioToRect,
  isExtended,
  hitResizeCornerSE,
  drawPlaceholder,
  drawCropBox,
} from "./sf_crop_expand_lib.js";
import {
  COL_W,
  COL_H,
  clampToImage,
  stepBrushSize,
  stepOpacity,
  paintStrokeMask,
  paintInvertMask,
  colorTextStyle,
  INVERT_ON_COLOR,
  POLY_ON_COLOR,
} from "./sf_brush_mask_lib.js";
import {
  LAYOUT,
  TOOL_COL,
  ORIENT_COL,
  TOOL_COL_X,
  ORIENT_COL_X,
  HEADER_H,
  COL1_GROUPS,
  COL2_GROUPS,
  COL3_GROUPS,
  columnYs,
  orientState,
  ensureMinSize,
  computeDisplayMetrics,
  RATIO_INPUTS,
  wiredAspect,
  effectiveRatio,
} from "./sf_crop_expand_brush_mask_lib.js";
import { registerBrushKeys, registerBrushStepSettings, brushSizeStep, brushOpacityStep } from "./sf_brush_tools.js";

const CLASS = "SFImageCropExpandBrushMask";
const HIDDEN_INPUT = "SFCropExpandBrushMaskJson"; // 必须与 crop_expand_brush_mask.py 的隐藏输入一致
const STATE_PROP = "sfCropExpandBrushMaskState";

const DEFAULT_STATE = {
  src_path: "",
  src_w: 512,
  src_h: 512,
  crop_x: 0,
  crop_y: 0,
  crop_w: 512,
  crop_h: 512,
  fill_color: "#000000",
  aspect_ratio: "free",
  custom_w: 1,
  custom_h: 1,
  brush_size: 80,
  strokes: [],
  // 以下仅预览/交互语义（不进 lean 注入，后端忽略）：
  brush_opacity: 0.5,
  brush_color: "255,255,255",
  brush_mode: "crop", // crop | brush | erase（三模式单选）
  // 多边形套索开关（交互语义；闭合前不影响输出，不进 lean 注入；Crop 模式
  // 点击入口按钮会自动切 Brush，见 §101）：
  brush_poly: false,
  // 反选（影响输出 → 进 lean 注入；合体节点语义 = 扩展区 ∪ (1 - 笔触)；
  // Ext OFF 时源图交集内 = 1 - 笔触、扩展区不带回，见 include_ext）：
  invert: false,
  // 扩展区遮罩开关（影响输出 → 进 lean 注入；默认 ON=扩展区计入 mask，
  // OFF=mask 只含笔触层、扩展区不重绘，§113）：
  include_ext: true,
  // 菜单参数记忆（不进 lean 注入；结果以 fill/fill_erase 笔触进 strokes——
  // 由 sf_brush_ai 按当前模式改写：Eraser=fill_erase 打洞，Crop/Brush=fill 添加）
  sam_prompt: "",
  sam_threshold: 0.5,
  sam_refine: 2,
  person_parts: [],
  person_confidence: 0.4,
  person_refine: false,
  yolo_kind: "bbox",
  yolo_model: "",
  yolo_conf: 0.25,
  yolo_box_shape: "rect",
  yolo_imgsz: 640,
  yolo_classes: [],
};

// ── 状态读写 ──────────────────────────────────────────────────────────────

function getState(node) {
  try {
    return { ...DEFAULT_STATE, ...JSON.parse(node.properties?.[STATE_PROP] || "{}") };
  } catch {
    return { ...DEFAULT_STATE };
  }
}

function setState(node, patch) {
  const next = { ...getState(node), ...patch };
  node.properties[STATE_PROP] = JSON.stringify(next);
  return next;
}

function stateChanged(node) {
  if (app.graph) app.graph.setDirtyCanvas(true, true);
}

// ── 接线宽高比（可选输入 aspect_w / aspect_h，§118）────────────────────────
// 两项都接且值可读 → 裁剪框即时按 w:h 刷新（onConnectionsChange 同步调用，
// 不靠下一次拖拽）；上游动态值读不到时输出由后端 execute 同公式兜底，信息栏
// 显示 "AR wire"。接线期间面板预设只记住不生效（断开后恢复）。

// 槽名会被核心渲染画在自定义面板之上（重叠 RATIO 列顶部）→ 零宽惯例隐藏
// （sf_dropdown 同款；diff 门控不无谓标脏），信息栏与 tooltip 承担说明。
function hideRatioSlotLabels(node) {
  for (const inp of node.inputs || []) {
    if (inp && RATIO_INPUTS.includes(inp.name) && inp.label !== ZW) inp.label = ZW;
  }
}

// 接线比例套到裁剪框（宽度为准、垂直居中，复用 applyRatioToRect）；逐字段
// diff 命中才写状态（工作流加载路径不为一致状态标脏）。返回是否真的改了。
function syncWiredRatio(node) {
  const wa = wiredAspect(node);
  if (!wa.ratio) return false;
  const st = getState(node);
  const rect = applyRatioToRect({ x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, wa.ratio);
  if (rect.x === st.crop_x && rect.y === st.crop_y && rect.w === st.crop_w && rect.h === st.crop_h) {
    return false;
  }
  setState(node, { crop_x: rect.x, crop_y: rect.y, crop_w: rect.w, crop_h: rect.h });
  stateChanged(node);
  return true;
}

function ratioOverrideToast() {
  sfToast({
    summary: "SF Crop Expand Brush Mask",
    detail: "比例由接线输入决定（aspect_w:aspect_h）；断开接线后本设置才生效",
    severity: "warn",
    fallbackTag: "SF Crop Expand Brush Mask",
  });
}

// lean 注入载荷：只含影响结果的字段（比例/画笔预览字段不进注入——改画笔颜色
// 不重跑；预览字段同口径排除在 IS_CHANGED 键外）
function leanState(st) {
  return {
    src_path: st.src_path || "",
    src_w: st.src_w || 512,
    src_h: st.src_h || 512,
    crop_x: st.crop_x || 0,
    crop_y: st.crop_y || 0,
    crop_w: st.crop_w || 512,
    crop_h: st.crop_h || 512,
    fill_color: st.fill_color || "#000000",
    brush_size: st.brush_size || 80,
    strokes: Array.isArray(st.strokes) ? st.strokes : [],
    invert: !!st.invert,
    include_ext: st.include_ext !== false,
  };
}

// ── 源图加载链路（sf_crop_source 共享实现）────────────────────────────────
// 换图清空笔触（BrushMask 同款语义）+ 裁剪框回满幅（CropExpand 同款语义）；
// 接线比例可读时在满幅基础上重套（与 resetCrop/applyOrientation 同语义，§118）。
const SOURCE_CFG = {
  uploadPrefix: "cebm_",
  logTag: "[SF Crop Expand Brush Mask]",
  toastTag: "SF Crop Expand Brush Mask",
  imgProp: "_sfCEBImg",
  getState,
  onStored: ({ srcPath, w, h }, node) => {
    cancelPoly(node);  // 换图后旧顶点坐标失效，丢弃未闭合会话
    setState(node, {
      src_path: srcPath,
      src_w: w,
      src_h: h,
      crop_x: 0,
      crop_y: 0,
      crop_w: w,
      crop_h: h,
      aspect_ratio: "free",
      strokes: [],
    });
    // 接线比例即时重套（diff 门控：一致不写状态）；seen 置空兜"此刻不可读、
    // 稍后就绪但 w:h 串未变"（绘制判定 null 即触发，幂等无副作用，§118.3.6 同款）
    node._sfCEBWiredRatioSeen = null;
    syncWiredRatio(node);
  },
};

// ── AI/工具右键菜单（共享 UI：sf_brush_ai.js；后端 brush_mask_sam/tools.py）──
// 结果按当前模式并入统一列表（Eraser=fill_erase 打洞减去，Crop/Brush=fill
// 添加，由 sf_brush_ai.mergeStrokes 单点改写）；extra 为菜单参数记忆字段
//（不进 lean 注入）；点/框模式经本 cfg 的坐标换算接入节点画布（含裁剪框出界偏移）。
const AI_CFG = {
  toastTag: "SF Crop Expand Brush Mask",
  logTag: "[SF Crop Expand Brush Mask]",
  getState,
  patchState: (node, patch) => {
    setState(node, patch);
    stateChanged(node);
  },
  addStrokes: (node, incoming, extra) => {
    const st = getState(node);
    setState(node, {
      strokes: incoming && incoming.length ? [...st.strokes, ...incoming] : st.strokes,
      ...(extra || {}),
    });
    stateChanged(node);
  },
  toImage: (node, lx, ly) => localToImage(lx, ly, metricsOf(node, null)),
  fromImage: (node, x, y) => imageToLocal(x, y, metricsOf(node, null)),
  inDisplay: (node, lx, ly) => {
    const m = metricsOf(node, null);
    return lx >= m.offsetX && lx <= m.offsetX + m.scaledDisplayWidth &&
      ly >= m.offsetY && ly <= m.offsetY + m.scaledDisplayHeight;
  },
  displayOrigin: (node) => {
    const m = metricsOf(node, null);
    return { x: m.offsetX, y: m.offsetY };
  },
  // 多边形套索落点（sf_brush_poly 用）：仅源图内有效（扩展区恒遮罩白，
  // 同画笔落笔限制），返回 null 表示拒绝该点
  toSource: (node, lx, ly) => {
    const st = getState(node);
    const p = localToImage(lx, ly, metricsOf(node, null));
    if (p.x < 0 || p.y < 0 || p.x > st.src_w - 1 || p.y > st.src_h - 1) return null;
    return clampToImage(p.x, p.y, st.src_w, st.src_h);
  },
  cancelPoly: (node) => cancelPoly(node),  // beginSamMode 互斥（Poly 会话丢弃）
};

// ── 控件（三列 + 底行：绘制与命中共用同一几何）───────────────────────────
// 列1（x=shiftLeft）：比例预设（Free 置顶）+ Custom/Reset/Fill；
// 列2（x=TOOL_COL_X）：Crop/Brush/Erase 三模式 + Clear/Undo/Invert/Ext +
// S±/O±/Pen（Ext = 扩展区计入遮罩开关，§113）；
// 列3（x=ORIENT_COL_X）：FlipH/FlipV/RotL/RotR（源图整体翻转/旋转，§112）；
// 底行（y 运行时解析）：Load/Browse + 信息文本。
// 每列按 COL1_GROUPS/COL2_GROUPS/COL3_GROUPS 分组排布（columnYs），主扩展按
// 同一数组画分组分隔线（§109）。

function toolText(id) {
  return {
    crop: "Crop",
    brush: "Brush",
    erase: "Erase",
    poly: "Poly",
    clear: "Clear",
    undo: "Undo",
    invert: "Invert",
    includeExt: "Ext",
    sizeMinus: "S−",
    sizePlus: "S+",
    opaMinus: "O−",
    opaPlus: "O+",
    brushColor: "Pen",
    flipH: "FlipH",
    flipV: "FlipV",
    rotL: "RotL",
    rotR: "RotR",
  }[id] || id;
}

// 悬停说明（底行信息文本临时替换为按钮用途；列头/分隔之外的"缩写按钮"解释，
// 中文文案与右侧菜单一致）。比例按钮（含 Custom）在 controlHint 单独拼。
const HINTS = {
  crop: "Crop 模式：拖拽裁剪框，框可拖出源图形成扩展区",
  brush: "Brush 模式：涂抹添加重绘遮罩",
  erase: "Erase 模式：擦除笔触（真擦除，按画序生效）",
  poly: "Poly 套索：多次点选，双击/Enter 闭合；Brush 添加 / Erase 打洞",
  clear: "清空全部笔触",
  undo: "撤销最后一笔（含套索与 AI 识别结果）",
  invert: "反选遮罩：反选笔触层（Ext ON 时扩展区仍保留重绘）",
  includeExt: "扩展区计入遮罩（ON：出界区重绘，外绘默认；OFF：mask 只含笔触层）",
  sizeMinus: "笔刷直径减小（[ / ] 或悬停滚轮）",
  sizePlus: "笔刷直径增大（[ / ] 或悬停滚轮）",
  opaMinus: "笔触预览透明度降低",
  opaPlus: "笔触预览透明度提高",
  brushColor: "画笔颜色（仅预览与 AI 染色，输出为二值遮罩）",
  reset: "裁剪框复位到整幅源图",
  fillColor: "扩展区填充色（裁剪框内、源图外）",
  custom: "自定义比例（全局预设库，可保存/删除）",
  load: "加载本地图片（也可拖放或 Ctrl+V 粘贴）",
  browse: "浏览输入目录图片",
  flipH: "水平翻转源图（左右镜像；裁剪框与笔触随图联动）",
  flipV: "垂直翻转源图（上下镜像；裁剪框与笔触随图联动）",
  rotL: "逆时针旋转源图 90°（比例预设复位 Free）",
  rotR: "顺时针旋转源图 90°（比例预设复位 Free）",
};

function controlHint(b, node) {
  if (b.isRatio) {
    const wa = node ? wiredAspect(node) : null;
    const wiredNote = wa?.wired
      ? `（接线比例优先：${wa.w != null ? `${wa.w}:${wa.h}` : "由上游值决定"}）` : "";
    if (b.ratioKey === "custom") return HINTS.custom + wiredNote;
    return `裁剪框比例：${b.text}${b.ratioKey === "free" ? "（不约束）" : ""}${wiredNote}`;
  }
  return HINTS[b.id] || "";
}

// 底行按钮的 y 标记为 "bottom"：节点高度运行时可变，绘制/命中时经 buttonRect
// 动态解析为 nodeH - shiftLeft - h。
const BOTTOM_Y = "bottom";

function bottomButtonY(node, h) {
  return node.size[1] - LAYOUT.shiftLeft - h;
}

// buttonRect(b, node) → [x, y, w, h]（解析 bottom 标记；绘制与命中共用）
function buttonRect(b, node) {
  return [b.x, b.y === BOTTOM_Y ? bottomButtonY(node, b.h) : b.y, b.w, b.h];
}

function buildControls() {
  const buttons = [];
  const col1Ys = columnYs(COL1_GROUPS);
  const col2Ys = columnYs(COL2_GROUPS);
  const colButton = (x, y, b) => ({ ...b, x, y, w: COL_W, h: COL_H });
  // 列1：Free 置顶 + 7 预设 + Custom/Reset/Fill 收尾（CropExpand 同款）
  RATIO_PRESETS_COL.forEach((key, i) => {
    buttons.push(colButton(LAYOUT.shiftLeft, col1Ys[i], {
      id: "ratio:" + key,
      text: ratioLabel(key),
      isRatio: true, ratioKey: key,
      action: (node) => setAspect(node, key),
    }));
  });
  buttons.push(
    colButton(LAYOUT.shiftLeft, col1Ys[RATIO_PRESETS_COL.length], {
      id: "ratio:custom",
      text: ratioLabel("custom"),
      isRatio: true, ratioKey: "custom",
      action: (node) => openRatioDialog(node),
    }),
    colButton(LAYOUT.shiftLeft, col1Ys[RATIO_PRESETS_COL.length + 1], {
      id: "reset",
      text: "Reset",
      action: (node) => resetCrop(node),
    }),
    colButton(LAYOUT.shiftLeft, col1Ys[RATIO_PRESETS_COL.length + 2], {
      id: "fillColor",
      text: "Fill", isColor: true,
      action: (node) => pickFillColor(node),
    }),
  );
  // 列2：Crop/Brush/Erase 三模式置顶 + BrushMask 工具列原序
  TOOL_COL.forEach((id, i) => {
    buttons.push(colButton(TOOL_COL_X, col2Ys[i], {
      id,
      text: toolText(id),
      isToggle: id === "crop" || id === "brush" || id === "erase",
      isPoly: id === "poly",
      isInvert: id === "invert",
      isColor: id === "brushColor",
    }));
  });
  // 列3：翻转/旋转（源图整体联动；isOrient 用 9px 小字号，§112）
  const col3Ys = columnYs(COL3_GROUPS);
  ORIENT_COL.forEach((id, i) => {
    buttons.push(colButton(ORIENT_COL_X, col3Ys[i], {
      id,
      text: toolText(id),
      isOrient: true,
    }));
  });
  // 底行：Load/Browse（与信息文本同排，y 运行时解析；按钮右缘 106）
  buttons.push(
    { id: "load", text: "Load", x: 10, y: BOTTOM_Y, w: 44, h: 21 },
    { id: "browse", text: "Browse", x: 58, y: BOTTOM_Y, w: 48, h: 21 },
  );
  return buttons;
}

function buttonAction(node, id) {
  const st = getState(node);
  if (id === "load") { pickFile(node, SOURCE_CFG); return; }
  if (id === "browse") { browseSource(node, SOURCE_CFG); return; }
  if (id === "reset") { resetCrop(node); return; }
  if (id === "fillColor") { pickFillColor(node); return; }
  if (id.startsWith("ratio:")) { setAspect(node, id.slice(6)); return; }
  if (ORIENT_COL.includes(id)) { applyOrientation(node, id); return; }
  if (id === "crop" || id === "brush" || id === "erase") setState(node, { brush_mode: id });
  else if (id === "clear") setState(node, { strokes: [] });
  else if (id === "undo") {
    if (st.strokes.length > 0) setState(node, { strokes: st.strokes.slice(0, -1) });
    else return;
  } else if (id === "sizeMinus") setState(node, { brush_size: stepBrushSizeFromSettings(st.brush_size, -1) });
  else if (id === "sizePlus") setState(node, { brush_size: stepBrushSizeFromSettings(st.brush_size, +1) });
  else if (id === "opaMinus") setState(node, { brush_opacity: stepOpacityFromSettings(st.brush_opacity, -1) });
  else if (id === "opaPlus") setState(node, { brush_opacity: stepOpacityFromSettings(st.brush_opacity, +1) });
  else if (id === "brushColor") { pickColor(node); return; }
  else if (id === "invert") { toggleInvert(AI_CFG, node); return; }  // 状态位按钮（与右键菜单同一实现）
  else if (id === "includeExt") setState(node, { include_ext: st.include_ext === false });  // ON=计入（默认）
  else if (id === "poly") {  // 多边形套索开关：Crop 下自动切 Brush（Poly 是画笔工具）
    if (!st.brush_poly) {
      cancelSamMode(node);  // 与 SAM 模式互斥
      if (st.brush_mode === "crop") setState(node, { brush_mode: "brush" });
    }
    togglePoly(AI_CFG, node);
    return;
  }
  else return;
  stateChanged(node);
}

// 步长设置三路同源（按钮 / [ ] / 滚轮都经 buttonAction；设置值读取在
// sf_brush_tools，勿内联步进公式——曾漏传设置值）
function stepBrushSizeFromSettings(cur, dir) {
  return stepBrushSize(cur, dir, brushSizeStep());
}

function stepOpacityFromSettings(cur, dir) {
  return stepOpacity(cur, dir, brushOpacityStep() / 100);
}

function resetCrop(node) {
  const st = getState(node);
  // 复位到整幅源图；接线比例可读时一并套用（预览=输出，否则等执行才被修正）
  const wa = wiredAspect(node);
  let rect = { x: 0, y: 0, w: st.src_w, h: st.src_h };
  if (wa.ratio) rect = applyRatioToRect(rect, wa.ratio);
  setState(node, {
    crop_x: rect.x,
    crop_y: rect.y,
    crop_w: rect.w,
    crop_h: rect.h,
    aspect_ratio: "free",
  });
  stateChanged(node);
}

function setAspect(node, key) {
  if (key === "custom") {
    openRatioDialog(node);
    return;
  }
  setState(node, { aspect_ratio: key });
  if (wiredAspect(node).wired) {
    // 接线优先：预设只记住（断开接线后生效），当前裁剪框不动
    ratioOverrideToast();
    stateChanged(node);
    return;
  }
  const st = getState(node);
  const ratio = ratioFromAspect(key);
  if (ratio) {
    const rect = applyRatioToRect({ x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, ratio);
    setState(node, { crop_x: rect.x, crop_y: rect.y, crop_w: rect.w, crop_h: rect.h });
  }
  stateChanged(node);
}

function pickFillColor(node) {
  pickColorInput(getState(node).fill_color || "#000000", (hex) => {
    setState(node, { fill_color: hex });
    stateChanged(node);
  });
}

function pickColor(node) {
  pickColorInput(rgbStringToHex(getState(node).brush_color), (hex) => {
    setState(node, { brush_color: hexToRgbString(hex) });
    stateChanged(node);
  }, "#ffffff");
}

// ── 源图翻转/旋转（列3 ORIENT，§112）─────────────────────────────────────
// 源图整体变换：canvas 重绘 → 上传新源图 → orientState 联动重映射裁剪框/
// 笔触/src_w/h（笔触粘内容）→ 替换预览 <img>。不删旧文件（与换图行为一致，
// 复制节点共享的源图不被就地改写）；连续点击用 busy 标记串行化（第二次点击
// 时第一次尚未落定，若并行会把旧图再变换一次）。patch 在上传成功后按当时
// 状态计算（上传等待期内的框/笔触编辑不被旧快照覆盖）；若等待期内源图被
// 替换（src_path 变了）则丢弃本次变换，避免把新图状态按旧图变换重映射。
async function applyOrientation(node, op) {
  if (node._sfCEBOrientBusy) return;
  const srcBefore = getState(node).src_path;
  node._sfCEBOrientBusy = true;
  try {
    const res = await orientSource(node, op, SOURCE_CFG);
    if (!res) return;
    const st = getState(node);
    if (st.src_path !== srcBefore) return;
    const patch = orientState(st, op);
    if (!patch) return;
    // 接线比例可读时套用到变换后的框：旋转会交换宽高（比例不再成立），
    // 不套用则预览与后端 execute 的修正结果不一致
    const wa = wiredAspect(node);
    if (wa.ratio) {
      const rect = applyRatioToRect(
        { x: patch.crop_x, y: patch.crop_y, w: patch.crop_w, h: patch.crop_h }, wa.ratio);
      Object.assign(patch, { crop_x: rect.x, crop_y: rect.y, crop_w: rect.w, crop_h: rect.h });
    }
    cancelPoly(node);      // 旧顶点坐标随图变换失效
    cancelSamMode(node);   // 与 SAM 点选/框选互斥
    node._sfCEBDrag = null;
    node._sfCEBDrawing = false;
    node._sfCEBCur = [];
    setState(node, { ...patch, src_path: res.srcPath });
    const img = new Image();
    img.onload = () => {
      node._sfCEBImg = img;
      if (app.graph) app.graph.setDirtyCanvas(true, true);
    };
    img.src = res.dataURL;
    stateChanged(node);
  } finally {
    node._sfCEBOrientBusy = false;
  }
}

// ── 自定义比例预设弹窗（sf_crop_expand_ratios 共享实现）────────────────────
function openRatioDialog(node) {
  const st = getState(node);
  openCustomRatioDialog({
    initialCustom: { w: st.custom_w ?? 1, h: st.custom_h ?? 1 },
    toastTag: "SF Crop Expand Brush Mask",
    applyCustom: (w, h) => {
      setState(node, { custom_w: w, custom_h: h, aspect_ratio: "custom" });
      if (wiredAspect(node).wired) {
        // 接线优先：自定义比例只记住（断开接线后生效），当前裁剪框不动
        ratioOverrideToast();
        stateChanged(node);
        return;
      }
      const s = getState(node);
      const rect = applyRatioToRect({ x: s.crop_x, y: s.crop_y, w: s.crop_w, h: s.crop_h }, w / h);
      setState(node, { crop_x: rect.x, crop_y: rect.y, crop_w: rect.w, crop_h: rect.h });
      stateChanged(node);
    },
  });
}

// ── 节点尺寸自适应 ────────────────────────────────────────────────────────

// 钳制节点尺寸不低于最小值（创建/恢复兜底；拖拽路径由 computeSize 包装钳住）
function clampNodeSize(node) {
  node.size = ensureMinSize(node.size?.[0], node.size?.[1]);
}

// ── 显示坐标系 ────────────────────────────────────────────────────────────

function metricsOf(node, frozen) {
  const st = getState(node);
  return computeDisplayMetrics(
    { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
    node.size[0], node.size[1], frozen || null);
}

// 拖放判定区 = 显示区
function sourceAreaOf(node) {
  const m = metricsOf(node, null);
  return { x: m.offsetX, y: m.offsetY, w: m.scaledDisplayWidth, h: m.scaledDisplayHeight };
}

const SOURCE_DROP_CFG = { ...SOURCE_CFG, getArea: sourceAreaOf };

// ── 绘制 ──────────────────────────────────────────────────────────────────

function drawButtons(ctx, node, st, th, accent) {
  for (const b of node._sfCEBCtrls) {
    const [bx, by, bw, bh] = buttonRect(b, node);
    if (b.isRatio && b.ratioKey === st.aspect_ratio) {
      ctx.fillStyle = accent;
    } else if (b.id === "fillColor") {
      ctx.fillStyle = st.fill_color || "#000000";
    } else if (b.isToggle && b.id === st.brush_mode) {
      ctx.fillStyle = accent;
    } else if (b.id === "includeExt" && st.include_ext !== false) {
      ctx.fillStyle = accent;  // Ext ON（默认）：强调色底，同模式按钮语言（§113）
    } else if (b.isPoly && st.brush_poly) {
      ctx.fillStyle = POLY_ON_COLOR;  // 套索 ON：状态色（绿，随开关变色，区别模式强调色）
    } else if (b.isInvert && st.invert) {
      ctx.fillStyle = INVERT_ON_COLOR;  // 反选 ON：状态色
    } else if (b.isColor) {
      const rgb = String(st.brush_color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
      ctx.fillStyle = `rgba(${rgb[0] || 0},${rgb[1] || 0},${rgb[2] || 0},0.9)`;
    } else {
      ctx.fillStyle = th.surface;
    }
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = th.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx, by, bw, bh);

    if (b.id === "fillColor") ctx.fillStyle = colorTextStyle(st.fill_color);
    else if (b.isColor) ctx.fillStyle = colorTextStyle(st.brush_color);
    else if ((b.isInvert && st.invert) || (b.isPoly && st.brush_poly)) ctx.fillStyle = "rgba(255,255,255,0.95)";
    else ctx.fillStyle = th.textStrong;

    ctx.font = b.y === BOTTOM_Y ? "11px Arial" : (b.isInvert || b.isOrient ? "9px Arial" : "10px Arial");
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    let text = b.text;
    if (b.ratioKey === "custom" && st.aspect_ratio === "custom") {
      text = `${st.custom_w || 1}:${st.custom_h || 1}`;
    }
    ctx.fillText(text, bx + bw / 2, by + bh / 2);
  }
}

function setupDrawing(node) {
  const { shiftLeft, shiftRight, ratioColW, bottomH } = LAYOUT;
  const BTN_H = 21; // 底行按钮高（与 buildControls 底行一致）

  node.onDrawForeground = (ctx) => {
    if (node.flags.collapsed) return false;

    const nodeW = node.size[0];
    const nodeH = node.size[1];
    const st = getState(node);
    const th = sfThemeColors();
    const accent = getSfAccent() || "rgba(100,150,255,0.9)";
    const dragging = !!node._sfCEBDrag;
    const m = metricsOf(node, dragging ? node._sfCEBDrag.frozen : null);

    // 接线宽高比（§118）：信息栏标记；上游 widget 值变化（无事件）靠绘制比对
    // 检测——发现比例变化时下一拍重同步裁剪框（不在绘制中改状态，避免重绘环）
    const wa = wiredAspect(node);
    const arText = wa.wired
      ? `AR ${wa.w != null ? `${wa.w}:${wa.h}` : "wire"}`
      : (wa.partial ? "AR 半接" : "");
    if (wa.wired && wa.ratio != null) {
      const sig = `${wa.w}:${wa.h}`;
      // 首次可读（如图片预览稍后载入才有尺寸）或比例变化 → 下一拍套用；
      // syncWiredRatio 内部 diff 门控，重复无副作用
      if (node._sfCEBWiredRatioSeen == null || node._sfCEBWiredRatioSeen !== sig) {
        setTimeout(() => syncWiredRatio(node), 0);
      }
      node._sfCEBWiredRatioSeen = sig;
    } else if (!wa.wired) {
      node._sfCEBWiredRatioSeen = null;
    }

    // 三列底条 + 底行背景（必须先于按钮：§45.8 半透明 chrome 按"背景→控件→
    // 内容→文本"分层，后画盖先画）
    const colTop = shiftLeft - 4;
    const colBottom = nodeH - shiftLeft - bottomH;
    const columns = [
      { x: shiftLeft - 4, w: ratioColW + 2, btnX: shiftLeft, groups: COL1_GROUPS, header: "RATIO", accent: true },
      { x: TOOL_COL_X - 4, w: LAYOUT.toolColW + 2, btnX: TOOL_COL_X, groups: COL2_GROUPS, header: "TOOLS", accent: false },
      { x: ORIENT_COL_X - 4, w: LAYOUT.orientColW + 2, btnX: ORIENT_COL_X, groups: COL3_GROUPS, header: "ORIENT", accent: false },
    ];
    for (const c of columns) {
      ctx.fillStyle = th.panel2;
      ctx.beginPath();
      ctx.roundRect(c.x, colTop, c.w, colBottom - colTop, 4);
      ctx.fill();
      ctx.strokeStyle = th.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(c.x, colTop, c.w, colBottom - colTop);
    }
    // 列头 chip（RATIO 用强调色、TOOLS 用中性面）——两列一眼可分（§109）
    const chipY = 8;
    const chipH = HEADER_H + 4;
    ctx.font = "bold 8px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const c of columns) {
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(c.x + 2, chipY, c.w - 4, chipH, 3);
      ctx.globalAlpha = c.accent ? 0.22 : 1;
      ctx.fillStyle = c.accent ? accent : th.surface;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = th.border;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = c.accent ? accent : th.textDim;
      ctx.fillText(c.header, c.x + c.w / 2, chipY + chipH / 2 + 0.5);
      ctx.restore();
    }
    // 组间分隔线（画在 columnYs 让出的 GROUP_EXTRA 空隙中央；同一数组推算，
    // 与按钮排布单源）
    ctx.strokeStyle = th.border;
    ctx.lineWidth = 1;
    for (const c of columns) {
      const ys = columnYs(c.groups);
      let idx = 0;
      for (let g = 0; g < c.groups.length - 1; g++) {
        idx += c.groups[g];
        // 上一组末行底部与下一组首行顶部的中央（组间空隙 = 常规间距 + GROUP_EXTRA）
        const y = (ys[idx - 1] + COL_H + ys[idx]) / 2 + 0.5;
        ctx.beginPath();
        ctx.moveTo(c.btnX + 2, y);
        ctx.lineTo(c.btnX + COL_W - 2, y);
        ctx.stroke();
      }
    }
    const bottomY = nodeH - shiftLeft - BTN_H;
    ctx.fillStyle = th.panel2;
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, BTN_H + 8, 4);
    ctx.fill();
    ctx.strokeStyle = th.border;
    ctx.strokeRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, BTN_H + 8);
    drawButtons(ctx, node, st, th, accent);

    // 显示区背景 + 网格（扩展区可见）
    ctx.fillStyle = th.surface;
    ctx.beginPath();
    ctx.roundRect(m.offsetX - 4, m.offsetY - 4, m.scaledDisplayWidth + 8, m.scaledDisplayHeight + 8, 4);
    ctx.fill();
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.strokeStyle = th.border;
    ctx.lineWidth = sfFrameThin();
    const gridSize = 32 * m.scale;
    for (let x = m.offsetX; x <= m.offsetX + m.scaledDisplayWidth; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, m.offsetY);
      ctx.lineTo(x, m.offsetY + m.scaledDisplayHeight);
      ctx.stroke();
    }
    for (let y = m.offsetY; y <= m.offsetY + m.scaledDisplayHeight; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(m.offsetX, y);
      ctx.lineTo(m.offsetX + m.scaledDisplayWidth, y);
      ctx.stroke();
    }
    ctx.restore();

    // 源图
    const srcX = m.offsetX + (0 - m.displayMinX) * m.scale;
    const srcY = m.offsetY + (0 - m.displayMinY) * m.scale;
    const srcW = st.src_w * m.scale;
    const srcH = st.src_h * m.scale;
    ctx.fillStyle = "rgba(20,20,20,0.9)";
    ctx.fillRect(srcX, srcY, srcW, srcH);
    const img = node._sfCEBImg;
    if (img && img.complete && img.naturalWidth > 0) {
      try {
        ctx.drawImage(img, srcX, srcY, srcW, srcH);
      } catch (e) {
        console.error("[SF Crop Expand Brush Mask] draw source failed:", e);
        drawPlaceholder(ctx, srcX, srcY, srcW, srcH, m.scale);
      }
    } else {
      drawPlaceholder(ctx, srcX, srcY, srcW, srcH, m.scale);
    }

    // 笔触遮罩预览（离屏源图像素画布 → 贴回源图区；真擦除与后端同画序）
    if ((st.strokes.length > 0 || (node._sfCEBCur && node._sfCEBCur.length > 0))
        && st.src_w > 0 && st.src_h > 0) {
      const maskCvs = ensureMaskCanvas(node, st.src_w, st.src_h);
      const rgb = String(st.brush_color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
      paintStrokeMask(maskCvs, st.strokes, {
        defaultSize: st.brush_size,
        paintStyle: `rgba(${rgb[0] || 0},${rgb[1] || 0},${rgb[2] || 0},1)`,
        current: node._sfCEBCur && node._sfCEBCur.length > 0
          ? { mode: st.brush_mode === "erase" ? "erase" : "brush", size: st.brush_size, points: node._sfCEBCur }
          : null,
      });
      const compositeCvs = st.invert ? invertComposite(node, maskCvs) : maskCvs;
      ctx.save();
      ctx.globalAlpha = st.brush_opacity;
      try {
        ctx.drawImage(compositeCvs, srcX, srcY, srcW, srcH);
      } catch (e) {
        console.error("[SF Crop Expand Brush Mask] draw mask composite failed:", e);
      }
      ctx.restore();
    }

    // 原图边界虚线
    ctx.strokeStyle = "rgba(100,150,255,0.6)";
    ctx.lineWidth = sfFrameWidth();
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(srcX, srcY, srcW, srcH);
    ctx.setLineDash([]);

    // 扩展区提示（裁剪框内、源图外的区域：扩展区计入 mask 时为白）——evenodd
    // 挖去源图；Ext OFF（扩展区不计入）时不画，提示跟随 mask 语义（§113）
    const rect = { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h };
    const x1 = m.offsetX + (rect.x - m.displayMinX) * m.scale;
    const y1 = m.offsetY + (rect.y - m.displayMinY) * m.scale;
    const x2 = x1 + rect.w * m.scale;
    const y2 = y1 + rect.h * m.scale;
    if (isExtended(rect, st.src_w, st.src_h) && st.include_ext !== false) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x1, y1, x2 - x1, y2 - y1);
      ctx.rect(srcX, srcY, srcW, srcH);
      ctx.clip("evenodd");
      ctx.fillStyle = "rgba(255,255,255,0.18)";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.restore();
    }

    // 裁剪框（框外压暗 + 框线/九宫格/手柄；sf_crop_expand_lib 共享绘制）
    // 仅 Crop 模式显示：Brush/Erase 时隐藏框组件（压暗/手柄/九宫格会干扰涂抹
    // 观察，手柄在非 Crop 模式本就不可交互）；扩展区白提示保留（§96，Ext OFF
    // 时不画见 §113）
    if (st.brush_mode === "crop") {
      drawCropBox(ctx, rect, st.src_w, st.src_h, m, sfFrameWidth());
    }

    // 笔刷光环：悬停显示区时显示实际笔刷直径（BrushMask 同款语义；离开节点
    // 后靠 canvas.node_over 门控隐藏）。SAM/Poly 模式不画光环（覆盖层语义）。
    const cursor = (node._sfAiSam || st.brush_poly) ? null : node._sfCEBCursor;
    const hovering = !app.canvas || app.canvas.node_over === node;
    if (cursor && hovering && (st.brush_mode === "brush" || st.brush_mode === "erase")) {
      const isErase = st.brush_mode === "erase";
      const ringR = Math.max(2, (st.brush_size / 2) * m.scale);
      ctx.save();
      ctx.lineWidth = sfCursorWidth();
      if (isErase) {
        ctx.strokeStyle = "#ffffff";
        ctx.setLineDash([4, 3]);
      } else {
        ctx.strokeStyle = accent;
        ctx.setLineDash([]);
      }
      ctx.beginPath();
      ctx.arc(cursor[0], cursor[1], ringR, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = isErase ? "#ffffff" : accent;
      ctx.beginPath();
      ctx.arc(cursor[0], cursor[1], 1.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // SAM 点选/框选覆盖层（点/橡皮筋/提示条）
    drawSamOverlay(node, ctx, (x, y) => imageToLocal(x, y, m), { x: m.offsetX, y: m.offsetY });

    // 多边形套索覆盖层（折线/橡皮筋/顶点/提示条；sf_brush_poly 共享实现）
    drawPolyOverlay(AI_CFG, node, ctx, (x, y) => imageToLocal(x, y, m), { x: m.offsetX, y: m.offsetY });

    // 信息文本（与底行按钮同排，右对齐到输出槽区前；空间不足时截断 "…"）。
    // 悬停控件时改显该按钮的中文说明（§109；拖拽/落笔期间不打扰）。
    const ext = isExtended(rect, st.src_w, st.src_h) ? " (Extended)" : "";
    const fullText = `Src: ${st.src_w}\u00d7${st.src_h} | Crop: ${Math.round(st.crop_w)}\u00d7${Math.round(st.crop_h)}${ext}` +
      ` | Brush: ${Math.round(st.brush_size)} | Strokes: ${st.strokes.length}` +
      `${st.brush_poly ? " | Poly" : ""}${st.invert ? " | Inv" : ""}${st.include_ext === false ? " | NoExt" : ""}` +
      `${arText ? ` | ${arText}` : ""}`;
    const hint = (hovering && !dragging && !node._sfCEBDrawing && node._sfCEBHover)
      ? controlHint(node._sfCEBHover, node) : "";
    const fullLabel = hint || fullText;
    ctx.fillStyle = hint ? th.textStrong : LiteGraph.NODE_TEXT_COLOR;
    ctx.font = "10px Arial";
    ctx.textAlign = "right";
    const maxTextW = nodeW - shiftRight - 6 - (106 + 6); // 底行按钮右缘 106 + 间隙 6
    let label = fullLabel;
    if (ctx.measureText(fullLabel).width > maxTextW) {
      while (label.length > 1 && ctx.measureText(label + "\u2026").width > maxTextW) {
        label = label.slice(0, -1);
      }
      label += "\u2026";
    }
    ctx.fillText(label, nodeW - shiftRight - 6, bottomY + BTN_H / 2 + 3.5);
  };
}

// 离屏遮罩画布（源图像素尺寸；尺寸变化重建）
function ensureMaskCanvas(node, w, h) {
  const mw = Math.max(1, Math.round(w) || 512);
  const mh = Math.max(1, Math.round(h) || 512);
  let cvs = node._sfCEBMaskCvs;
  if (!cvs || cvs.width !== mw || cvs.height !== mh) {
    cvs = document.createElement("canvas");
    cvs.width = mw;
    cvs.height = mh;
    node._sfCEBMaskCvs = cvs;
  }
  return cvs;
}

// 反相遮罩预览画布（离屏白底打洞；尺寸=源图，随尺寸重建）
function invertComposite(node, srcCvs) {
  let cvs = node._sfInvertCvs;
  if (!cvs || cvs.width !== srcCvs.width || cvs.height !== srcCvs.height) {
    cvs = document.createElement("canvas");
    cvs.width = Math.max(1, srcCvs.width);
    cvs.height = Math.max(1, srcCvs.height);
    node._sfInvertCvs = cvs;
  }
  return paintInvertMask(srcCvs, cvs);
}

// ── 交互 ──────────────────────────────────────────────────────────────────

function finalizeDrag(node, canvas) {
  const drag = node._sfCEBDrag;
  if (!drag) return false;
  node._sfCEBDrag = null;
  const rect = roundRect(drag.curRect);
  setState(node, { crop_x: rect.x, crop_y: rect.y, crop_w: rect.w, crop_h: rect.h });
  if (canvas) canvas.style.cursor = "default";
  stateChanged(node);
  return true;
}

function finalizeStroke(node, canvas) {
  if (!node._sfCEBDrawing) return false;
  node._sfCEBDrawing = false;
  const cur = node._sfCEBCur || [];
  node._sfCEBCur = [];
  if (cur.length > 0) {
    const st = getState(node);
    setState(node, {
      strokes: [...st.strokes, {
        mode: st.brush_mode === "erase" ? "erase" : "brush",
        size: st.brush_size,
        points: cur.map(([x, y]) => [Math.round(x), Math.round(y)]),
      }],
    });
  }
  if (canvas) canvas.style.cursor = "default";
  stateChanged(node);
  return true;
}

function setupInteractions(node) {
  node.onMouseDown = (e, localPos) => {
    const lp = localPos || [e.canvasX - node.pos[0], e.canvasY - node.pos[1]];
    const [lx, ly] = lp;

    // 控件命中（双列 + 底行共用 buttonRect 解析）
    for (const b of node._sfCEBCtrls) {
      const [bx, by, bw, bh] = buttonRect(b, node);
      if (lx >= bx && lx <= bx + bw && ly >= by && ly <= by + bh) {
        if (typeof b.action === "function") b.action(node);
        else buttonAction(node, b.id);
        return true;
      }
    }

    // SAM 点选/框选模式（显示区内消费；控件命中已先行）
    if (handleSamPointer(AI_CFG, node, "down", e, [lx, ly])) return true;

    // 多边形套索（显示区内消费；SAM 优先。落点限源图内/点首点闭合/右键取消）
    if (handlePolyPointer(AI_CFG, node, "down", e, [lx, ly])) return true;

    // 仅左键起拖/落笔（右键/中键交给原位/菜单；buttons 缺失的旧调用放行）
    if (e && e.button !== 0 && e.button !== undefined) return false;

    const st = getState(node);
    const m = metricsOf(node, null);

    if (st.brush_mode === "crop") {
      const p = localToImage(lx, ly, m);
      const handle = getHandleAtPoint(p.x, p.y,
        { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, m.scale);
      if (!handle) return false;
      node._sfCEBDrag = {
        handle,
        startImgX: p.x,
        startImgY: p.y,
        startRect: { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h },
        curRect: { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h },
        // 关键：冻结当前显示坐标系，整个拖拽过程共用（防自反馈飘移）
        frozen: m,
      };
      return true;
    }

    // Brush/Erase：仅源图区域内落笔（笔触以源图坐标记录并钳制在源图内；
    // 扩展区本就已是遮罩白，无需画笔）
    if (lx < m.offsetX || lx > m.offsetX + m.scaledDisplayWidth ||
        ly < m.offsetY || ly > m.offsetY + m.scaledDisplayHeight) {
      return false;
    }
    const p = localToImage(lx, ly, m);
    if (p.x < 0 || p.y < 0 || p.x > st.src_w - 1 || p.y > st.src_h - 1) return false;
    const c = clampToImage(p.x, p.y, st.src_w, st.src_h);
    node._sfCEBDrawing = true;
    node._sfCEBCur = [[c.x, c.y]];
    return true;
  };

  node.onMouseMove = (e, localPos, graphCanvas) => {
    const lp = localPos || [e.canvasX - node.pos[0], e.canvasY - node.pos[1]];
    const [lx, ly] = lp;

    // 释放丢失兜底：主键已松但拖拽/落笔状态残留 → 立即落定
    if ((node._sfCEBDrag || node._sfCEBDrawing) && primaryButtonReleased(e)) {
      finalizeDrag(node, graphCanvas?.canvas);
      finalizeStroke(node, graphCanvas?.canvas);
      return true;
    }

    // SAM 点选/框选模式优先（消费移动；框模式拖橡皮筋 + crosshair）
    if (handleSamPointer(AI_CFG, node, "move", e, lp)) return true;
    // 多边形套索：记录橡皮筋光标 + crosshair（覆盖层自取会话 cursor）
    if (handlePolyPointer(AI_CFG, node, "move", e, lp)) return true;

    const st = getState(node);
    const dragging = !!node._sfCEBDrag;
    const m = metricsOf(node, dragging ? node._sfCEBDrag.frozen : null);
    const p = localToImage(lx, ly, m);

    // 光环位置常驻记录（每次移动都记——落笔/拖框期间也要跟随鼠标；曾只在
    // 悬停分支更新，导致画笔拖动时圆环停在起笔前的位置，见 §93.5）
    node._sfCEBCursor =
      lx >= m.offsetX && lx <= m.offsetX + m.scaledDisplayWidth &&
      ly >= m.offsetY && ly <= m.offsetY + m.scaledDisplayHeight ? [lx, ly] : null;

    if (dragging) {
      const drag = node._sfCEBDrag;
      // 比例优先级：接线宽高比（可读）> 面板预设；接线不可读时不约束
      const ratio = effectiveRatio(st, wiredAspect(node));
      drag.curRect = updateCropByDrag(drag, drag.handle, p.x, p.y, ratio);
      setState(node, {
        crop_x: drag.curRect.x, crop_y: drag.curRect.y,
        crop_w: drag.curRect.w, crop_h: drag.curRect.h,
      });
      if (graphCanvas && graphCanvas.dirty_canvas !== true) {
        graphCanvas.setDirty(true, true);
      }
      return true;
    }

    if (node._sfCEBDrawing) {
      const c = clampToImage(p.x, p.y, st.src_w, st.src_h);
      const cur = node._sfCEBCur;
      const last = cur[cur.length - 1];
      if (Math.hypot(c.x - last[0], c.y - last[1]) > 1) {
        cur.push([c.x, c.y]);
        stateChanged(node);
      }
      return true;
    }

    // 控件悬停记录（底行信息文本改显按钮中文说明；只在变化时重绘，§109）
    let hoverBtn = null;
    for (const b of node._sfCEBCtrls) {
      const [bx, by, bw, bh] = buttonRect(b, node);
      if (lx >= bx && lx <= bx + bw && ly >= by && ly <= by + bh) { hoverBtn = b; break; }
    }
    if (hoverBtn !== node._sfCEBHover) {
      node._sfCEBHover = hoverBtn;
      if (app.graph) app.graph.setDirtyCanvas(true, true);
    }

    // 悬停 cursor：Crop 模式按手柄；其余模式写回 default（防模式切换后残留
    // 上一次的 resize cursor）
    if (graphCanvas?.canvas) {
      let handle = null;
      if (st.brush_mode === "crop") {
        handle = getHandleAtPoint(p.x, p.y,
          { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, m.scale);
      }
      graphCanvas.canvas.style.cursor = handle ? getCursorForHandle(handle) : "default";
    }
    return false;
  };

  node.onMouseUp = (_e, _lp, graphCanvas) => {
    if (handleSamPointer(AI_CFG, node, "up", _e, _lp || [0, 0])) return true;
    const a = finalizeDrag(node, graphCanvas?.canvas);
    const b = finalizeStroke(node, graphCanvas?.canvas);
    return a || b;
  };

  node.onDblClick = () => {
    if (handlePolyDblClick(AI_CFG, node)) return;
    finalizeDrag(node, null);
    finalizeStroke(node, null);
  };

  // 释放兜底（window capture，复用 sf_common）：画布 processMouseUp 会
  // stopPropagation 且仅对 node_over 回调 → 出界/纯点击释放收不到，bubble
  // 的 document 监听无效（见 sf_common.installNodeReleaseGuard 注释）
  installNodeReleaseGuard(node, () => {
    const a = finalizeDrag(node, null);
    const b = finalizeStroke(node, null);
    if (a || b) {
      if (app.graph) app.graph.setDirtyCanvas(true, true);
    }
  }, { hook: "_sfCEBReleaseGuard" });

  // 拖放图片文件到节点显示区加载（sf_crop_source 共享实现）
  installSourceDrop(node, SOURCE_DROP_CFG);
}

// ── graphToPrompt：注入隐藏输入（只注入 lean 字段，Export/分享共用同一份 output）──

if (!app._sfCEBPromptPatched) {
  app._sfCEBPromptPatched = true;
  const _origGraphToPrompt = app.graphToPrompt.bind(app);
  app.graphToPrompt = async function (...args) {
    const result = await _origGraphToPrompt(...args);
    try {
      const out = result?.output;
      if (out) {
        let index = null;
        for (const id in out) {
          const entry = out[id];
          if (!entry || entry.class_type !== CLASS) continue;
          if (!index) index = buildClassNodeIndex(CLASS);
          const node = findNodeByPromptId(index, id);
          const raw = node?.properties?.[STATE_PROP];
          let parsed = DEFAULT_STATE;
          try { parsed = { ...DEFAULT_STATE, ...JSON.parse(raw || "{}") }; } catch { /* 坏状态回退默认 */ }
          entry.inputs = entry.inputs || {};
          entry.inputs[HIDDEN_INPUT] = JSON.stringify(leanState(parsed));
        }
      }
    } catch (e) {
      console.warn("[SF Crop Expand Brush Mask] could not inject state:", (e && e.message) || e);
    }
    return result;
  };
}

// ── 右下角 resize cursor 视觉修正（sf_common 共享安装器）────────────────────
installResizeCornerCursor(CLASS, hitResizeCornerSE);

// ── 画笔工具共享安装（sf_brush_tools）：快捷键（[ ] 尺寸 / C 裁剪 / B 笔刷 /
// E 擦除，双通道 + 时间戳去重；buttonAction 对这三模式本就是确定性写入）与
// S±/O± 悬停滚轮快调；action 经 buttonAction 统一入口（步长设置三路同源）。
const brushKeyStep = registerBrushKeys({
  classNames: [CLASS],
  controlsProp: "_sfCEBCtrls",
  keyMap: { "]": "sizePlus", "[": "sizeMinus", "c": "crop", "b": "brush", "e": "erase" },
  applyAction: (n, action) => buttonAction(n, action),
});

// ── 注册 ──────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "sfnodes.CropExpandBrushMask",
  init() {
    registerBrushStepSettings();
    // 画布线条粗细设置（sfnodes.Canvas.*，三画布节点共用，幂等）
    registerSfLineWidthSettings();
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    // 拖拽 resize 的最小值取自 node.computeSize()（双端同款：onDrag 里 clamp 到
    // computeSize 再 setSize）——包装抬高到 MIN 一处钳住两条路径
    const origComputeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function (out) {
      const size = origComputeSize ? origComputeSize.call(this, out) : [0, 0];
      return ensureMinSize(size[0], size[1]);
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onNodeCreated) onNodeCreated.apply(this, []);
      this.properties = this.properties || {};
      if (!this.properties[STATE_PROP]) {
        this.properties[STATE_PROP] = JSON.stringify(DEFAULT_STATE);
      }
      clampNodeSize(this);
      hideRatioSlotLabels(this);  // 新增输入槽名隐藏（ZW），见 §118
      this._sfCEBCtrls = buildControls();
      setupDrawing(this);
      setupInteractions(this);
      // Ctrl+V 粘贴剪贴板图片：installPasteHandler（选中判定/防抢输入框/清扫
      // 自动 pasted 均在公共实现内），加载走既有链路
      installPasteHandler({
        comfyClass: CLASS,
        hook: "_sfCEBPaste",
        onPasteImage: (n, dataURL) => n._sfCEBPaste(dataURL),
      });
      this._sfCEBPaste = (dataURL) => storeSource(this, dataURL, SOURCE_CFG);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      // 加载期守卫：连接恢复不应触发比例同步（链接在 configure 之后才恢复，
      // 由 onAfterGraphConfigured 统一补同步）
      this._sfCEBConfiguring = true;
      try {
        if (onConfigure) onConfigure.apply(this, [info]);
        // 恢复源图预览（状态本体在 properties 里，已随 info 恢复）
        clampNodeSize(this);
        if (!this._sfCEBCtrls) {
          this._sfCEBCtrls = buildControls();
          setupDrawing(this);
          setupInteractions(this);
        }
        hideRatioSlotLabels(this);  // configure 可能重建 inputs（§118）
        restoreSourceImage(this, SOURCE_CFG);
      } finally {
        this._sfCEBConfiguring = false;
      }
    };

    // 接线事件：aspect_w/aspect_h 任一接入/断开 → 同步即时刷新裁剪框
    // （开启工作流时 configure 直赋 links 不触发本事件，由 onAfterGraphConfigured
    // 补一次；isGraphLoading 守卫加载期的杂散回调）
    const INPUT_TYPE = (typeof LiteGraph !== "undefined" && LiteGraph.INPUT != null) ? LiteGraph.INPUT : 1;
    const origConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, slotInfo) {
      const r = origConnectionsChange ? origConnectionsChange.apply(this, arguments) : undefined;
      const name = this.inputs?.[index]?.name ?? slotInfo?.name;
      if (type === INPUT_TYPE && name && RATIO_INPUTS.includes(name)
          && !this._sfCEBConfiguring && !isGraphLoading()) {
        syncWiredRatio(this);  // 主路径：同步套用（同一帧可见）
        // 补同步重试（全为幂等 + diff 门控）：0ms 兜新版前端 link 表滞后
        // （platform §12 坑 3）；200/1000ms 兜上游图片预览异步载入后才可读
        // （LoadImage → GetImageSize 链，§118）
        for (const d of [0, 200, 1000]) setTimeout(() => syncWiredRatio(this), d);
        if (app.graph) app.graph.setDirtyCanvas(true, true);
      }
      return r;
    };

    // 工作流加载/粘贴恢复连线：链路已就绪后按接线比例补同步（diff 门控，
    // 一致状态不写 properties → 打开工作流不标脏）
    const origAfterGraphConfigured = nodeType.prototype.onAfterGraphConfigured;
    nodeType.prototype.onAfterGraphConfigured = function () {
      const r = origAfterGraphConfigured ? origAfterGraphConfigured.apply(this, arguments) : undefined;
      hideRatioSlotLabels(this);
      syncWiredRatio(this);
      // 图片预览/上游 widgets 在配置完成后才逐步就绪 → 幂等补同步
      for (const d of [200, 1000]) setTimeout(() => syncWiredRatio(this), d);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (onRemoved) onRemoved.apply(this, []);
      removeNodeReleaseGuard(this, { hook: "_sfCEBReleaseGuard" });
      disposePoly(this);  // 清套索会话 + 解绑 Poly 键盘监听
    };

    // 官方快捷键通道：画布 processKey 把 keydown 分发给选中节点的 onKeyDown。
    // core 的 addNodeKeyHandler 会再包一层（`=== false` 表示已处理），链式兼容。
    const origKeyDown = nodeType.prototype.onKeyDown;
    nodeType.prototype.onKeyDown = function (e) {
      if (brushKeyStep(e)) return false;
      if (origKeyDown) return origKeyDown.apply(this, arguments);
    };

    // 右键菜单（共享安装器：SAM/人物/YOLO/导入/反选/卸载）
    installBrushMenu(AI_CFG, nodeType);
  },
});
