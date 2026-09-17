// ==========================================================================
// sf_brush_mask.js - SF Image Brush Mask 主扩展
// ==========================================================================
//
// 复刻 ComfyUI-YCNodes_Toolkit ycimagebrushmask（Load Image Brush Mask）：
// 节点上直接加载图片（Load Image 按钮 / Browse 图片浏览器 / 拖放文件到
// 节点 / Ctrl+V 粘贴），画笔在图上直接涂抹遮罩（brush 涂白 / eraser 擦除）。
//
// 控件布局向 SFImageCropExpand 看齐：左侧工具竖列（Brush/Eraser 模式 →
// Clear/Undo → Size±/Opa± 步进 → 双取色块，节点顶直通画布区底）+ 底行
//（Load Image/Browse 与信息文本同排，实时数值进信息文本）。原版的横向
// Size/Opacity 拖拽滑块在 30px 竖列里放不下，改为步进器（Size± 步长 2 /
// Opa± 步长 5%，纯函数在 lib）。
//
// 状态真源 node.properties.sfBrushMaskState（JSON 字符串，随工作流保存），
// 经 graphToPrompt 钩子注入隐藏输入 SFBrushMaskJson（只注入影响结果的
// lean 字段：src_path/src_w/src_h/brush_size/strokes——预览用的
// brush_opacity/brush_color 不进注入，改预览不重跑，
// 同 sf_outpaint.js / sf_crop_expand.js 先例；Python hidden 已声明，
// schema 内不被剥离）。
// 源图持久化：dataURL 经 CropAPI.uploadSrc 落盘 input/sfnodes_crop/
// （复用 SFImageCrop 的路由，零新增后端路由），状态只存 src_path——
// 工作流重载经 /view 恢复预览（原版 base64 进 workflow + 会话 Map 缓存，
// 文件巨大且刷新丢图，已确认差异）。
//
// 共享实现（与 SFImageCropExpand/SFImageCropExpandBrushMask 单源）：
//   - 源图加载链路 sf_crop_source.js；画笔步进/命中/绘制与取色文字色在
//     sf_brush_mask_lib.js（无 app 依赖可 .mjs 直测）；步长设置与 [ ]/滚轮
//     快调在 sf_brush_tools.js；AI/工具右键菜单（SAM 文本/点选/框选、人物
//     部位、YOLO、导入遮罩、反选、统一卸载，忙时熔断预检）在 sf_brush_ai.js
//     （与 SFImageCropExpandBrushMask 同一实现，见 §91·§93）；
//     右下角 cursor 补写在 sf_common。
//   - 最小尺寸钳制（computeSize 包装）同 sf_crop_expand.js（§44 同款）。
// ==========================================================================

import { app } from "/scripts/app.js";
import { getSfAccent, installPasteHandler, primaryButtonReleased, installNodeReleaseGuard, removeNodeReleaseGuard, installResizeCornerCursor, pickColorInput, rgbStringToHex, hexToRgbString } from "./sf_common.js";
import { installBrushMenu, handleSamPointer, drawSamOverlay, toggleInvert } from "./sf_brush_ai.js";
import { pickFile, browseSource, restoreSourceImage, installSourceDrop, storeSource } from "./sf_crop_source.js";
import { registerBrushKeys, registerBrushStepSettings, brushSizeStep, brushOpacityStep } from "./sf_brush_tools.js";
import { buildClassNodeIndex, findNodeByPromptId } from "./sf_pause_kit.js";
import {
  LAYOUT,
  TOOL_COL,
  COL_TOP,
  COL_W,
  COL_H,
  COL_STEP,
  ensureMinSize,
  hitResizeCornerSE,
  computeDisplayMetrics,
  localToImage,
  imageToLocal,
  clampToImage,
  stepBrushSize,
  stepOpacity,
  paintStrokeMask,
  paintInvertMask,
  colorTextStyle,
  INVERT_ON_COLOR,
} from "./sf_brush_mask_lib.js";

const CLASS = "SFImageBrushMask";
const HIDDEN_INPUT = "SFBrushMaskJson"; // 必须与 brush_mask.py 的隐藏输入一致
const STATE_PROP = "sfBrushMaskState";

// 步长设置读取/注册在 sf_brush_tools.js（与 SFImageCropExpandBrushMask 共享
// 同一组用户设置键）。

const DEFAULT_STATE = {
  src_path: "",
  src_w: 512,
  src_h: 512,
  brush_size: 80,
  strokes: [],
  // 以下仅预览语义（不进 lean 注入，后端忽略）：
  brush_opacity: 0.5,
  brush_color: "255,255,255",
  brush_mode: "brush",
  // 反选（影响输出 → 进 lean 注入）：
  invert: false,
  // eraser_color 惰性遗留（ECol 按钮已随真擦除预览移除，无读取方，旧工作流无感）
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

// lean 注入载荷：只含影响结果的字段（改 opacity/颜色/模式/记忆项不重跑；
// SAM 结果即 fill 笔触，随 strokes 进键）
function leanState(st) {
  return {
    src_path: st.src_path || "",
    src_w: st.src_w || 512,
    src_h: st.src_h || 512,
    brush_size: st.brush_size || 80,
    strokes: Array.isArray(st.strokes) ? st.strokes : [],
    invert: !!st.invert,
  };
}

function stateChanged(node) {
  if (app.graph) app.graph.setDirtyCanvas(true, true);
}

// ── 源图加载链路（sf_crop_source 共享实现）────────────────────────────────
// 换图清空笔触（原版同款语义）；加载后状态回写与预览交给共享链路。
const SOURCE_CFG = {
  uploadPrefix: "brushmask_",
  logTag: "[SF Brush Mask]",
  toastTag: "SF Brush Mask",
  imgProp: "_sfBrushImg",
  getState,
  onStored: ({ srcPath, w, h }, node) => {
    setState(node, { src_path: srcPath, src_w: w, src_h: h, strokes: [] });
  },
};

// 拖放判定区 = 显示区（metrics 动态计算）
function sourceAreaOf(node) {
  const st = getState(node);
  const m = computeDisplayMetrics({ srcW: st.src_w, srcH: st.src_h }, node.size[0], node.size[1]);
  return { x: m.offsetX, y: m.offsetY, w: m.scaledW, h: m.scaledH };
}

const SOURCE_DROP_CFG = { ...SOURCE_CFG, getArea: sourceAreaOf };

// ── 控件（左竖列 + 底行：绘制与命中共用同一几何）──────────────────────────
// 左竖列（x=shiftLeft, w=30, h=18，步进 22，列顶 16）：Brush/Erase 模式 →
// Clear/Undo → Size±/Opa± 步进 → BCol 取色（背景即当前色）。
// 底行（y 运行时解析为 nodeH-shiftLeft-21）：Load/Browse + 信息文本。

function toolText(id) {
  return {
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

// 底行按钮的 y 标记为 "bottom"：节点高度运行时可变，
// 绘制/命中时经 buttonRect 动态解析为 nodeH - shiftLeft - h。
const BOTTOM_Y = "bottom";

function bottomButtonY(node, h) {
  return node.size[1] - LAYOUT.shiftLeft - h;
}

// buttonRect(b, node) → [x, y, w, h]（解析 bottom 标记；绘制与命中共用）
function buttonRect(b, node) {
  return [b.x, b.y === BOTTOM_Y ? bottomButtonY(node, b.h) : b.y, b.w, b.h];
}

function buildControls() {
  const buttons = TOOL_COL.map((id, i) => ({
    id,
    text: toolText(id),
    x: LAYOUT.shiftLeft,
    y: COL_TOP + i * COL_STEP,
    w: COL_W,
    h: COL_H,
    isToggle: id === "brush" || id === "erase",
    isInvert: id === "invert",
    isColor: id === "brushColor" ? "brush" : null,
  }));
  // 底行：Load/Browse（与信息文本同排，y 运行时解析；短文案保 MIN 320）
  buttons.push(
    { id: "load", text: "Load", x: 10, y: BOTTOM_Y, w: 44, h: 21 },
    { id: "browse", text: "Browse", x: 58, y: BOTTOM_Y, w: 48, h: 21 },
  );
  return buttons;
}

function buttonAction(node, id) {
  const st = getState(node);
  if (id === "load") pickFile(node, SOURCE_CFG);
  else if (id === "browse") browseSource(node, SOURCE_CFG);
  else if (id === "brush") setState(node, { brush_mode: "brush" });
  else if (id === "modeErase") setState(node, { brush_mode: "erase" }); // 键盘 E：确定性切换（鼠标 Erase 按钮保持 toggle）
  else if (id === "erase") setState(node, { brush_mode: st.brush_mode === "erase" ? "brush" : "erase" });
  else if (id === "clear") setState(node, { strokes: [] });
  else if (id === "undo") {
    if (st.strokes.length > 0) setState(node, { strokes: st.strokes.slice(0, -1) });
    else return;
  } else if (id === "sizeMinus") setState(node, { brush_size: stepBrushSize(st.brush_size, -1, brushSizeStep()) });
  else if (id === "sizePlus") setState(node, { brush_size: stepBrushSize(st.brush_size, +1, brushSizeStep()) });
  else if (id === "opaMinus") setState(node, { brush_opacity: stepOpacity(st.brush_opacity, -1, brushOpacityStep() / 100) });
  else if (id === "opaPlus") setState(node, { brush_opacity: stepOpacity(st.brush_opacity, +1, brushOpacityStep() / 100) });
  else if (id === "brushColor") { pickColor(node); return; }
  else if (id === "invert") { toggleInvert(AI_CFG, node); return; }  // 状态位按钮（与右键菜单同一实现）
  else return;
  stateChanged(node);
}

function pickColor(node) {
  pickColorInput(rgbStringToHex(getState(node).brush_color), (hex) => {
    setState(node, { brush_color: hexToRgbString(hex) });
    stateChanged(node);
  }, "#ffffff");
}

// 取色按钮文字色 colorTextStyle 提升到 sf_brush_mask_lib.js（两节点共用）。

// ── AI/工具右键菜单（共享 UI：sf_brush_ai.js；后端 brush_mask_sam/tools.py）──
// fill 笔触并入统一列表；extra 为菜单参数记忆字段（不进 lean 注入）；
// 点/框模式经本 cfg 的坐标换算接入节点画布。
const AI_CFG = {
  toastTag: "SF Brush Mask",
  logTag: "[SF Brush Mask]",
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
  toImage: (node, lx, ly) => localToImage(lx, ly, metricsOf(node)),
  fromImage: (node, x, y) => imageToLocal(x, y, metricsOf(node)),
  inDisplay: (node, lx, ly) => {
    const m = metricsOf(node);
    return lx >= m.offsetX && lx <= m.offsetX + m.scaledW &&
      ly >= m.offsetY && ly <= m.offsetY + m.scaledH;
  },
  displayOrigin: (node) => {
    const m = metricsOf(node);
    return { x: m.offsetX, y: m.offsetY };
  },
};

// ── 节点尺寸自适应 ────────────────────────────────────────────────────────

// 钳制节点尺寸不低于最小值（创建/恢复兜底；拖拽路径由 computeSize 包装钳住）
function clampNodeSize(node) {
  node.size = ensureMinSize(node.size?.[0], node.size?.[1]);
}

// ── 绘制 ──────────────────────────────────────────────────────────────────

function drawPlaceholder(ctx, x, y, w, h, scale) {
  ctx.fillStyle = "rgba(100,100,100,0.3)";
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "rgba(150,150,150,0.2)";
  ctx.lineWidth = 1;
  const grid = 32 * scale;
  for (let gx = x; gx <= x + w; gx += grid) {
    ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx, y + h); ctx.stroke();
  }
  for (let gy = y; gy <= y + h; gy += grid) {
    ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x + w, gy); ctx.stroke();
  }
}

// 笔触绘制/离屏遮罩合成提升到 sf_brush_mask_lib.js（drawStrokePath /
// paintStrokeMask，两节点共用同一画序与真擦除语义）。

function setupDrawing(node) {
  const { shiftLeft, shiftRight, bottomH } = LAYOUT;

  node.onDrawForeground = (ctx) => {
    if (node.flags.collapsed) return false;
    const nodeW = node.size[0];
    const nodeH = node.size[1];
    const st = getState(node);
    const m = computeDisplayMetrics({ srcW: st.src_w, srcH: st.src_h }, nodeW, nodeH);
    const accent = getSfAccent() || "rgba(100,150,255,0.9)";

    // 左侧竖列底条（节点顶到画布区底缘，与图片区同高）
    const colTop = shiftLeft - 4;
    const colBottom = nodeH - shiftLeft - bottomH;
    ctx.fillStyle = "rgba(40,40,40,0.9)";
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, colTop, LAYOUT.toolColW + 2, colBottom - colTop, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(100,100,100,0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(shiftLeft - 4, colTop, LAYOUT.toolColW + 2, colBottom - colTop);

    // 底信息行背景（必须画在底行按钮之前：后画盖先画，半透明底栏
    // 若盖住按钮会导致按钮发虚 + 边框错层残影，见 §45.8）
    const bottomY = nodeH - shiftLeft - 21;
    ctx.fillStyle = "rgba(40,40,40,0.9)";
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, 21 + 8, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(100,100,100,0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, 21 + 8);

    // 竖列 + 底行按钮（底行按钮落在底栏背景之上）
    for (const b of node._sfBrushCtrls) {
      const [bx, by, bw, bh] = buttonRect(b, node);
      if (b.isColor) {
        const rgb = String(st.brush_color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
        ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.9)`;
      } else if (b.isInvert && st.invert) {
        ctx.fillStyle = INVERT_ON_COLOR;  // 反选 ON：状态色
      } else if (b.isToggle && (
        (b.id === "brush" && st.brush_mode !== "erase") ||
        (b.id === "erase" && st.brush_mode === "erase"))) {
        ctx.fillStyle = accent;
      } else {
        ctx.fillStyle = "rgba(60,60,60,0.7)";
      }
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = "rgba(150,150,150,0.6)";
      ctx.strokeRect(bx, by, bw, bh);
      ctx.fillStyle = b.isColor ? colorTextStyle(st.brush_color)
        : (b.isInvert && st.invert ? "rgba(255,255,255,0.95)" : "rgba(220,220,220,0.9)");
      ctx.font = b.y === BOTTOM_Y ? "11px Arial" : (b.isInvert ? "9px Arial" : "10px Arial");
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(b.text, bx + bw / 2, by + bh / 2);
    }

    // 图片区背景 + 网格
    ctx.fillStyle = "rgba(60,60,60,0.8)";
    ctx.beginPath();
    ctx.roundRect(m.offsetX - 4, m.offsetY - 4, m.scaledW + 8, m.scaledH + 8, 4);
    ctx.fill();
    const img = node._sfBrushImg;
    if (img && img.complete && img.naturalWidth > 0) {
      try {
        ctx.drawImage(img, m.offsetX, m.offsetY, m.scaledW, m.scaledH);
      } catch (e) {
        console.error("[SF Brush Mask] draw source failed:", e);
        drawPlaceholder(ctx, m.offsetX, m.offsetY, m.scaledW, m.scaledH, m.scale);
      }
    } else {
      drawPlaceholder(ctx, m.offsetX, m.offsetY, m.scaledW, m.scaledH, m.scale);
    }

    // 遮罩合成（真擦除预览；sf_brush_mask_lib.paintStrokeMask 共享实现——
    // 离屏按源图像素绘制，贴回时一次缩放到显示区，与后端画序逐像素一致）
    const mw = Math.max(1, st.src_w || 512);
    const mh = Math.max(1, st.src_h || 512);
    let maskCvs = node._sfMaskCvs;
    if (!maskCvs || maskCvs.width !== mw || maskCvs.height !== mh) {
      maskCvs = document.createElement("canvas");
      maskCvs.width = mw;
      maskCvs.height = mh;
      node._sfMaskCvs = maskCvs;
    }
    const paintRGB = String(st.brush_color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
    paintStrokeMask(maskCvs, st.strokes, {
      defaultSize: st.brush_size,
      paintStyle: `rgba(${paintRGB[0]},${paintRGB[1]},${paintRGB[2]},1)`,
      current: node._sfBrushCur && node._sfBrushCur.length > 0
        ? { mode: st.brush_mode === "erase" ? "erase" : "brush", size: st.brush_size, points: node._sfBrushCur }
        : null,
    });
    const compositeCvs = st.invert ? invertComposite(node, maskCvs) : maskCvs;
    ctx.save();
    ctx.globalAlpha = st.brush_opacity;
    try {
      ctx.drawImage(compositeCvs, m.offsetX, m.offsetY, m.scaledW, m.scaledH);
    } catch (e) {
      console.error("[SF Brush Mask] draw mask composite failed:", e);
    }
    ctx.restore();

    // 笔刷光环：悬停图片区时显示实际笔刷直径（inpaint _drawCursor 同款语义）。
    // 半径随显示 scale 自适应，Size 步进/滚轮实时生效；离开节点后靠
    // canvas.node_over 门控隐藏（无额外监听，hover 切换自带重绘）。
    // app.canvas 为空时（冒烟测试）视为悬停。
    const cursor = node._sfAiSam ? null : node._sfBrushCursor;
    const hovering = !app.canvas || app.canvas.node_over === node;
    if (cursor && hovering) {
      const isErase = st.brush_mode === "erase";
      const ringR = Math.max(2, (st.brush_size / 2) * m.scale);
      ctx.save();
      ctx.lineWidth = 1.5;
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

    // SAM 点选/框选覆盖层（点/橡皮筋/提示条；模式激活时不画光环）
    drawSamOverlay(node, ctx, (x, y) => imageToLocal(x, y, m), { x: m.offsetX, y: m.offsetY });

    // 底信息行文本（右对齐截断，落在按钮之上，两者无重叠）
    ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
    ctx.font = "10px Arial";
    ctx.textAlign = "right";
    const fullText = `Brush ${Math.round(st.brush_size)} · Op ${Math.round(st.brush_opacity * 100)}% · Strokes ${st.strokes.length}` +
      `${st.invert ? " · Inv" : ""} · ${st.src_w}\u00d7${st.src_h}`;
    const maxTextW = nodeW - shiftRight - 6 - (106 + 6); // 底行按钮右缘 106 + 间隙 6
    let label = fullText;
    if (ctx.measureText(fullText).width > maxTextW) {
      while (label.length > 1 && ctx.measureText(label + "\u2026").width > maxTextW) {
        label = label.slice(0, -1);
      }
      label += "\u2026";
    }
    ctx.fillText(label, nodeW - shiftRight - 6, bottomY + 21 / 2 + 3.5);
  };
}

// ── 交互 ──────────────────────────────────────────────────────────────────

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

function metricsOf(node) {
  const st = getState(node);
  return computeDisplayMetrics({ srcW: st.src_w, srcH: st.src_h }, node.size[0], node.size[1]);
}

function finalizeStroke(node, canvas) {
  if (!node._sfBrushDrawing) return false;
  node._sfBrushDrawing = false;
  const cur = node._sfBrushCur || [];
  node._sfBrushCur = [];
  if (cur.length > 0) {
    const st = getState(node);
    setState(node, {
      strokes: [...st.strokes, {
        mode: st.brush_mode || "brush",
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

    // 控件命中（竖列 + 底行共用 buttonRect 解析）
    for (const b of node._sfBrushCtrls) {
      const [bx, by, bw, bh] = buttonRect(b, node);
      if (lx >= bx && lx <= bx + bw && ly >= by && ly <= by + bh) {
        buttonAction(node, b.id);
        return true;
      }
    }

    // SAM 点选/框选模式（显示区内消费；控件命中已先行）
    if (handleSamPointer(AI_CFG, node, "down", e, [lx, ly])) return true;

    // 画布区：左键落笔
    const st = getState(node);
    const m = metricsOf(node);
    if (lx < m.offsetX || lx > m.offsetX + m.scaledW ||
        ly < m.offsetY || ly > m.offsetY + m.scaledH) {
      return false;
    }
    if (e.button !== 0 && e.button !== undefined) return false;
    const p = localToImage(lx, ly, m);
    const c = clampToImage(p.x, p.y, st.src_w, st.src_h);
    node._sfBrushDrawing = true;
    node._sfBrushCur = [[c.x, c.y]];
    return true;
  };

  node.onMouseMove = (e, localPos) => {
    const lp = localPos || [e.canvasX - node.pos[0], e.canvasY - node.pos[1]];
    const [lx, ly] = lp;
    // SAM 点选/框选模式优先（消费移动；框模式拖橡皮筋 + crosshair）
    if (handleSamPointer(AI_CFG, node, "move", e, lp)) return true;
    // 光环位置常驻记录（画与不画都记；图片区外置空，绘制侧再经 node_over 门控）
    const mm = metricsOf(node);
    node._sfBrushCursor =
      lx >= mm.offsetX && lx <= mm.offsetX + mm.scaledW &&
      ly >= mm.offsetY && ly <= mm.offsetY + mm.scaledH ? [lx, ly] : null;
    // 释放丢失兜底：主键已松但落笔状态残留 → 立即落定，绝不再续笔
    if (node._sfBrushDrawing && primaryButtonReleased(e)) {
      finalizeStroke(node, null);
      return true;
    }
    if (!node._sfBrushDrawing) return false;
    const st = getState(node);
    const m = metricsOf(node);
    const p = localToImage(lx, ly, m);
    const c = clampToImage(p.x, p.y, st.src_w, st.src_h);
    const cur = node._sfBrushCur;
    const last = cur[cur.length - 1];
    const dist = Math.hypot(c.x - last[0], c.y - last[1]);
    if (dist > 1) {
      cur.push([c.x, c.y]);
      stateChanged(node);
    }
    return true;
  };

  node.onMouseUp = (_e, _lp, graphCanvas) => {
    if (handleSamPointer(AI_CFG, node, "up", _e, _lp || [0, 0])) return true;
    return finalizeStroke(node, graphCanvas?.canvas);
  };

  node.onDblClick = () => finalizeStroke(node, null);

  // 释放兜底（window capture，复用 sf_common）：画布 processMouseUp 会
  // stopPropagation 且仅对 node_over 回调 → 出界/纯点击释放收不到，bubble
  // 的 document 监听无效（见 sf_common.installNodeReleaseGuard 注释）
  installNodeReleaseGuard(node, () => {
    if (finalizeStroke(node, null)) {
      if (app.graph) app.graph.setDirtyCanvas(true, true);
    }
  }, { hook: "_sfBrushReleaseGuard" });

  // 拖放图片文件到节点显示区加载（sf_crop_source 共享实现）
  installSourceDrop(node, SOURCE_DROP_CFG);
}

// ── graphToPrompt：注入隐藏输入（只注入 lean 字段，Export/分享共用同一份 output）──
// 图索引复用 sf_pause_kit.buildClassNodeIndex/findNodeByPromptId（四闸门单源，
// 复合 id + 子图环守卫）。

if (!app._sfBrushMaskPromptPatched) {
  app._sfBrushMaskPromptPatched = true;
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
          const raw = node?.properties?.[STATE_PROP] || JSON.stringify(DEFAULT_STATE);
          let parsed = DEFAULT_STATE;
          try { parsed = { ...DEFAULT_STATE, ...JSON.parse(raw) }; } catch { /* 坏状态回退默认 */ }
          entry.inputs = entry.inputs || {};
          entry.inputs[HIDDEN_INPUT] = JSON.stringify(leanState(parsed));
        }
      }
    } catch (e) {
      console.warn("[SF Brush Mask] could not inject state:", (e && e.message) || e);
    }
    return result;
  };
}

// ── 右下角 resize cursor 视觉修正（sf_common 共享安装器）────────────────────
// 命中区维持原生 15×15（不做扩大）、区内直接写 style.cursor 的修法见
// sf_common.installResizeCornerCursor 注释（§44）。
installResizeCornerCursor(CLASS, hitResizeCornerSE);

// ── 画笔工具共享安装（sf_brush_tools）：快捷键（[ ] 尺寸 / B 笔刷 / E 擦除，
// 双通道 + 时间戳去重）与 S±/O± 悬停滚轮快调；action 经 buttonAction 统一
// 入口（步长设置三路同源）。
const brushKeyStep = registerBrushKeys({
  classNames: [CLASS],
  controlsProp: "_sfBrushCtrls",
  keyMap: { "]": "sizePlus", "[": "sizeMinus", "b": "brush", "e": "modeErase" },
  applyAction: (n, action) => buttonAction(n, action),
});

// ── 注册 ──────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "sfnodes.BrushMask",
  init() {
    registerBrushStepSettings();
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    // 拖拽 resize 的最小值取自 node.computeSize()（双端同款）——包装抬高到
    // MIN 一处钳住两条路径（§44）
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
      this._sfBrushCtrls = buildControls();
      setupDrawing(this);
      setupInteractions(this);
      // Ctrl+V 粘贴剪贴板图片：installPasteHandler（选中判定/防抢输入框/
      // 清扫自动 pasted/ LoadImage 均在公共实现内），加载走既有链路
      installPasteHandler({
        comfyClass: CLASS,
        hook: "_sfBrushPaste",
        onPasteImage: (n, dataURL) => n._sfBrushPaste(dataURL),
      });
      this._sfBrushPaste = (dataURL) => storeSource(this, dataURL, SOURCE_CFG);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      if (onConfigure) onConfigure.apply(this, [info]);
      // 恢复源图预览（状态本体在 properties 里，已随 info 恢复）
      clampNodeSize(this);
      if (!this._sfBrushCtrls) {
        this._sfBrushCtrls = buildControls();
        setupDrawing(this);
        setupInteractions(this);
      }
      restoreSourceImage(this, SOURCE_CFG);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (onRemoved) onRemoved.apply(this, []);
      removeNodeReleaseGuard(this, { hook: "_sfBrushReleaseGuard" });
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
