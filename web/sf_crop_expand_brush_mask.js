// ==========================================================================
// sf_crop_expand_brush_mask.js - SF Image Crop Expand Brush Mask 主扩展
// ==========================================================================
//
// SFImageCropExpand + SFImageBrushMask 合体：节点上加载图片后，一个画布内完成
//   ① 拖拽可出界的裁剪框（比例预设 / Reset / 填充色；出界区 = 扩展区）
//   ② 画笔涂抹额外重绘区（Crop/Brush/Erase 三模式；笔触以源图像素坐标记录，
//      随图移动，只有落在裁剪框内的部分进入输出）
// 输出扩展画布 + 并集遮罩（扩展区 ∪ 笔触，白=重绘，恒二值）。
//
// 双列布局（sf_crop_expand_brush_mask_lib）：列1 = 比例预设（CropExpand 同款），
// 列2 = 画笔工具（BrushMask TOOL_COL 前置 Crop 模式）；显示区多让一列宽度。
//
// 状态真源 node.properties.sfCropExpandBrushMaskState（JSON 字符串，随工作流
// 保存），经 graphToPrompt 钩子注入隐藏输入 SFCropExpandBrushMaskJson（只注入
// lean 字段：src/crop/fill/brush_size/strokes——比例与画笔预览字段不进注入，
// 改画笔颜色不重跑；Python hidden 已声明，schema 内不被剥离）。
//
// 共享实现（禁止内联副本）：
//   - 几何/冻结快照防飘移/手柄/绘制：sf_crop_expand_lib.js
//   - 步进/命中/笔触渲染：sf_brush_mask_lib.js（paintStrokeMask/colorTextStyle）
//   - 组合布局：sf_crop_expand_brush_mask_lib.js
//   - 源图链路：sf_crop_source.js（Load/Browse/拖放/Ctrl+V → input/sfnodes_crop/）
//   - 比例预设弹窗：sf_crop_expand_ratios.js
//   - AI/工具右键菜单：sf_brush_ai.js（与 SFImageBrushMask 单源：SAM 文本/
//     点选/框选、人物部位、YOLO、导入遮罩、反选、统一卸载；工作流执行期间
//     后端 409 熔断，见 experience/nodes-image.md §91·§93）
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
} from "./sf_common.js";
import { pickFile, browseSource, restoreSourceImage, installSourceDrop, storeSource } from "./sf_crop_source.js";
import { openCustomRatioDialog, ratioLabel } from "./sf_crop_expand_ratios.js";
import { installBrushMenu, handleSamPointer, drawSamOverlay, toggleInvert } from "./sf_brush_ai.js";
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
  COL_TOP,
  COL_W,
  COL_H,
  COL_STEP,
  clampToImage,
  stepBrushSize,
  stepOpacity,
  paintStrokeMask,
  paintInvertMask,
  colorTextStyle,
  INVERT_ON_COLOR,
} from "./sf_brush_mask_lib.js";
import {
  LAYOUT,
  TOOL_COL,
  TOOL_COL_X,
  ensureMinSize,
  computeDisplayMetrics,
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
  // 反选（影响输出 → 进 lean 注入；合体节点语义 = 扩展区 ∪ (1 - 笔触)）：
  invert: false,
  // 菜单参数记忆（不进 lean 注入；结果均以 fill 笔触进 strokes）
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
  };
}

// ── 源图加载链路（sf_crop_source 共享实现）────────────────────────────────
// 换图清空笔触（BrushMask 同款语义）+ 裁剪框回满幅（CropExpand 同款语义）。
const SOURCE_CFG = {
  uploadPrefix: "cebm_",
  logTag: "[SF Crop Expand Brush Mask]",
  toastTag: "SF Crop Expand Brush Mask",
  imgProp: "_sfCEBImg",
  getState,
  onStored: ({ srcPath, w, h }, node) => {
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
  },
};

// ── AI/工具右键菜单（共享 UI：sf_brush_ai.js；后端 brush_mask_sam/tools.py）──
// fill 笔触并入统一列表；extra 为菜单参数记忆字段（不进 lean 注入）；
// 点/框模式经本 cfg 的坐标换算接入节点画布（含裁剪框出界偏移）。
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
};

// ── 控件（双列 + 底行：绘制与命中共用同一几何）───────────────────────────
// 列1（x=shiftLeft）：比例预设（Free 置顶）+ Custom/Reset/Color；
// 列2（x=TOOL_COL_X）：Crop/Brush/Erase 三模式 + Clear/Undo/S±/O±/BCol；
// 底行（y 运行时解析）：Load/Browse + 信息文本。

function toolText(id) {
  return {
    crop: "Crop",
    brush: "Brush",
    erase: "Erase",
    clear: "Clear",
    undo: "Undo",
    invert: "Invert",
    sizeMinus: "S−",
    sizePlus: "S+",
    opaMinus: "O−",
    opaPlus: "O+",
    brushColor: "BCol",
  }[id] || id;
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
  const colButton = (i, x, b) => ({ ...b, x, y: COL_TOP + i * COL_STEP, w: COL_W, h: COL_H });
  // 列1：Free 置顶 + 7 预设 + Custom/Reset/Color 收尾（CropExpand 同款）
  RATIO_PRESETS_COL.forEach((key, i) => {
    buttons.push(colButton(i, LAYOUT.shiftLeft, {
      id: "ratio:" + key,
      text: ratioLabel(key),
      isRatio: true, ratioKey: key,
      action: (node) => setAspect(node, key),
    }));
  });
  buttons.push(
    colButton(RATIO_PRESETS_COL.length, LAYOUT.shiftLeft, {
      id: "ratio:custom",
      text: ratioLabel("custom"),
      isRatio: true, ratioKey: "custom",
      action: (node) => openRatioDialog(node),
    }),
    colButton(RATIO_PRESETS_COL.length + 1, LAYOUT.shiftLeft, {
      id: "reset",
      text: "Reset",
      action: (node) => resetCrop(node),
    }),
    colButton(RATIO_PRESETS_COL.length + 2, LAYOUT.shiftLeft, {
      id: "fillColor",
      text: "Color", isColor: true,
      action: (node) => pickFillColor(node),
    }),
  );
  // 列2：Crop/Brush/Erase 三模式置顶 + BrushMask 工具列原序
  TOOL_COL.forEach((id, i) => {
    buttons.push(colButton(i, TOOL_COL_X, {
      id,
      text: toolText(id),
      isToggle: id === "crop" || id === "brush" || id === "erase",
      isInvert: id === "invert",
      isColor: id === "brushColor",
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
  setState(node, {
    crop_x: 0,
    crop_y: 0,
    crop_w: st.src_w,
    crop_h: st.src_h,
    aspect_ratio: "free",
  });
  stateChanged(node);
}

function setAspect(node, key) {
  if (key === "custom") {
    openRatioDialog(node);
    return;
  }
  const st = setState(node, { aspect_ratio: key });
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

// ── 自定义比例预设弹窗（sf_crop_expand_ratios 共享实现）────────────────────
function openRatioDialog(node) {
  const st = getState(node);
  openCustomRatioDialog({
    initialCustom: { w: st.custom_w ?? 1, h: st.custom_h ?? 1 },
    toastTag: "SF Crop Expand Brush Mask",
    applyCustom: (w, h) => {
      const s = setState(node, { custom_w: w, custom_h: h, aspect_ratio: "custom" });
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
    else if (b.isInvert && st.invert) ctx.fillStyle = "rgba(255,255,255,0.95)";
    else ctx.fillStyle = th.textStrong;

    ctx.font = b.y === BOTTOM_Y ? "11px Arial" : (b.isInvert ? "9px Arial" : "10px Arial");
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

    // 双列底条 + 底行背景（必须先于按钮：§45.8 半透明 chrome 按"背景→控件→
    // 内容→文本"分层，后画盖先画）
    const colTop = shiftLeft - 4;
    const colBottom = nodeH - shiftLeft - bottomH;
    for (const [x, w] of [
      [shiftLeft - 4, ratioColW + 2],
      [TOOL_COL_X - 4, LAYOUT.toolColW + 2],
    ]) {
      ctx.fillStyle = th.panel2;
      ctx.beginPath();
      ctx.roundRect(x, colTop, w, colBottom - colTop, 4);
      ctx.fill();
      ctx.strokeStyle = th.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, colTop, w, colBottom - colTop);
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

    // 扩展区提示（裁剪框内、源图外的区域：mask 恒为白）——evenodd 挖去源图
    const rect = { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h };
    const x1 = m.offsetX + (rect.x - m.displayMinX) * m.scale;
    const y1 = m.offsetY + (rect.y - m.displayMinY) * m.scale;
    const x2 = x1 + rect.w * m.scale;
    const y2 = y1 + rect.h * m.scale;
    if (isExtended(rect, st.src_w, st.src_h)) {
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
    // 观察，手柄在非 Crop 模式本就不可交互）；扩展区白提示保留（§96）
    if (st.brush_mode === "crop") {
      drawCropBox(ctx, rect, st.src_w, st.src_h, m, sfFrameWidth());
    }

    // 笔刷光环：悬停显示区时显示实际笔刷直径（BrushMask 同款语义；离开节点
    // 后靠 canvas.node_over 门控隐藏）
    const cursor = node._sfAiSam ? null : node._sfCEBCursor;
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

    // 信息文本（与底行按钮同排，右对齐到输出槽区前；空间不足时截断 "…"）
    ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
    ctx.font = "10px Arial";
    ctx.textAlign = "right";
    const ext = isExtended(rect, st.src_w, st.src_h) ? " (Extended)" : "";
    const fullText = `Src: ${st.src_w}\u00d7${st.src_h} | Crop: ${Math.round(st.crop_w)}\u00d7${Math.round(st.crop_h)}${ext}` +
      ` | Brush: ${Math.round(st.brush_size)} | Strokes: ${st.strokes.length}${st.invert ? " | Inv" : ""}`;
    const maxTextW = nodeW - shiftRight - 6 - (106 + 6); // 底行按钮右缘 106 + 间隙 6
    let label = fullText;
    if (ctx.measureText(fullText).width > maxTextW) {
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
      const ratio = ratioFromAspect(st.aspect_ratio, st.custom_w, st.custom_h);
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
      if (onConfigure) onConfigure.apply(this, [info]);
      // 恢复源图预览（状态本体在 properties 里，已随 info 恢复）
      clampNodeSize(this);
      if (!this._sfCEBCtrls) {
        this._sfCEBCtrls = buildControls();
        setupDrawing(this);
        setupInteractions(this);
      }
      restoreSourceImage(this, SOURCE_CFG);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (onRemoved) onRemoved.apply(this, []);
      removeNodeReleaseGuard(this, { hook: "_sfCEBReleaseGuard" });
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
