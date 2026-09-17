// ==========================================================================
// sf_crop_expand.js - SF Image Crop Expand 主扩展
// ==========================================================================
//
// 复刻 ComfyUI-YCNodes_Toolkit ycImageCrop（Load Image Crop Expand）：节点上
// 直接加载图片（Load 按钮 / 拖放文件到节点），拖拽一个可出界的裁剪框，
// 左侧竖列提供比例按钮（Free 置顶/预设/Custom）与填充色/重置按钮；底行为
// Load/Browse 与信息文本同排。
//
// 状态真源 node.properties.sfCropExpandState（JSON 字符串，随工作流保存），
// 经 graphToPrompt 钩子注入隐藏输入 SFCropExpandJson（只注入不剪枝，同
// sf_outpaint.js 先例；Python hidden 已声明，schema 内不被剥离）。
//
// 共享实现（与 SFImageBrushMask / SFImageCropExpandBrushMask 单源）：
//   - 源图持久化链路 sf_crop_source.js（Load/Browse/拖放/粘贴 → input/
//     sfnodes_crop/ → /view 恢复）
//   - 比例预设弹窗 sf_crop_expand_ratios.js（全局库读写）
//   - 右下角 cursor 补丁 sf_common.installResizeCornerCursor
//   - 绘制 drawPlaceholder/drawCropBox 与交互数学（含拖拽冻结快照防飘移）
//     在纯库 sf_crop_expand_lib.js
// ==========================================================================

import { app } from "/scripts/app.js";
import { getSfAccent, sfThemeColors, installPasteHandler, primaryButtonReleased, installNodeReleaseGuard, removeNodeReleaseGuard, installResizeCornerCursor, pickColorInput } from "./sf_common.js";
import { pickFile, browseSource, restoreSourceImage, installSourceDrop, storeSource } from "./sf_crop_source.js";
import { openCustomRatioDialog, ratioLabel } from "./sf_crop_expand_ratios.js";
import { buildClassNodeIndex, findNodeByPromptId } from "./sf_pause_kit.js";
import {
  RATIO_PRESETS_COL,
  LAYOUT,
  ratioFromAspect,
  computeDisplayMetrics,
  ensureMinSize,
  hitResizeCornerSE,
  localToImage,
  getHandleAtPoint,
  getCursorForHandle,
  updateCropByDrag,
  roundRect,
  applyRatioToRect,
  isExtended,
  drawPlaceholder,
  drawCropBox,
} from "./sf_crop_expand_lib.js";

const CLASS = "SFImageCropExpand";
const HIDDEN_INPUT = "SFCropExpandJson"; // 必须与 crop_expand.py 的隐藏输入一致
const STATE_PROP = "sfCropExpandState";

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

// ── 源图加载链路（sf_crop_source 共享实现）────────────────────────────────
// 加载后控制框自动填满整图；不改节点大小（显示区 scale 动态计算）。
const SOURCE_CFG = {
  uploadPrefix: "cropexpand_",
  logTag: "[SF Crop Expand]",
  toastTag: "SF Crop Expand",
  imgProp: "_sfExpandImg",
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
    });
  },
};

// 拖放判定区 = 显示区（metrics 动态计算）
function sourceAreaOf(node) {
  const st = getState(node);
  const m = computeDisplayMetrics(
    { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
    node.size[0], node.size[1], null);
  return { x: m.offsetX, y: m.offsetY, w: m.scaledDisplayWidth, h: m.scaledDisplayHeight };
}

const SOURCE_DROP_CFG = { ...SOURCE_CFG, getArea: sourceAreaOf };

// ── 面板按钮 ──────────────────────────────────────────────────────────────

// 底行按钮（Load/Browse）的 y 标记为 "bottom"：节点高度运行时可变，
// 绘制/命中时经 buttonRect 动态解析为 nodeH - shiftLeft - h。
const BOTTOM_Y = "bottom";

function bottomButtonY(node, h) {
  return node.size[1] - LAYOUT.shiftLeft - h;
}

// buttonRect(b, node) → [x, y, w, h]（解析 bottom 标记；绘制与命中共用）
function buttonRect(b, node) {
  return [b.x, b.y === BOTTOM_Y ? bottomButtonY(node, b.h) : b.y, b.w, b.h];
}

function buildButtons(node) {
  const buttons = [];
  // 左侧竖列：Free 置顶 + 7 预设 + Custom/Reset/Color 收尾（从节点顶直通画布区底）
  const colH = 18;
  const colGap = 4;
  const colTop = LAYOUT.shiftLeft + 6;
  const colButton = (i, b) => ({ ...b, x: LAYOUT.shiftLeft, y: colTop + i * (colH + colGap), w: 30, h: colH });
  RATIO_PRESETS_COL.forEach((key, i) => {
    buttons.push(colButton(i, {
      text: ratioLabel(key),
      isRatio: true, ratioKey: key,
      action: () => setAspect(node, key),
    }));
  });
  buttons.push(
    colButton(RATIO_PRESETS_COL.length, {
      text: ratioLabel("custom"),
      isRatio: true, ratioKey: "custom",
      action: () => openRatioDialog(node),
    }),
    colButton(RATIO_PRESETS_COL.length + 1, {
      text: "Reset",
      action: () => resetCrop(node),
    }),
    colButton(RATIO_PRESETS_COL.length + 2, {
      text: "Color", isColor: true,
      action: () => pickFillColor(node),
    }),
  );
  // 底行：Load/Browse（与信息文本同排，y 运行时解析；按钮右缘 106，
  // 其右为信息文本窗——MIN 宽度下文本不足时截断，见 setupDrawing）
  buttons.push(
    { text: "Load", x: 10, y: BOTTOM_Y, w: 44, h: 21, action: () => pickFile(node, SOURCE_CFG) },
    { text: "Browse", x: 58, y: BOTTOM_Y, w: 48, h: 21, action: () => browseSource(node, SOURCE_CFG) },
  );
  return buttons;
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

// ── 自定义比例预设弹窗（sf_crop_expand_ratios 共享实现）────────────────────
// 弹窗读写全局库 user/sfnodes/crop_expand_presets.json（跨工作流）；套用动作
// 回写本节点状态与裁剪框。
function openRatioDialog(node) {
  const st = getState(node);
  openCustomRatioDialog({
    initialCustom: { w: st.custom_w ?? 1, h: st.custom_h ?? 1 },
    toastTag: "SF Crop Expand",
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

// ── 绘制 ──────────────────────────────────────────────────────────────────


function setupDrawing(node) {
  const { shiftLeft, shiftRight } = LAYOUT;
  const BTN_H = 21; // 底行按钮高（与 buildButtons 底行一致）

  node.onDrawForeground = (ctx) => {
    if (node.flags.collapsed) return false;

    const nodeW = node.size[0];
    const nodeH = node.size[1];
    const dragging = !!node._sfExpandDrag;
    const st = getState(node);
    const th = sfThemeColors();

    const m = computeDisplayMetrics(
      { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
      nodeW, nodeH,
      dragging ? node._sfExpandDrag.frozen : null,
    );

    // 比例竖列底条（节点顶到画布区底缘，与图片区同高）
    const colTop = shiftLeft - 4;
    const colBottom = nodeH - shiftLeft - LAYOUT.bottomH;
    ctx.fillStyle = th.panel2;
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, colTop, LAYOUT.ratioColW + 2, colBottom - colTop, 4);
    ctx.fill();
    ctx.strokeStyle = th.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(shiftLeft - 4, colTop, LAYOUT.ratioColW + 2, colBottom - colTop);

    // 扩展区背景 + 网格
    ctx.fillStyle = th.surface;
    ctx.beginPath();
    ctx.roundRect(m.offsetX - 4, m.offsetY - 4, m.scaledDisplayWidth + 8, m.scaledDisplayHeight + 8, 4);
    ctx.fill();
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.strokeStyle = th.border;
    ctx.lineWidth = 1;
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
    const sourceX = m.offsetX + (0 - m.displayMinX) * m.scale;
    const sourceY = m.offsetY + (0 - m.displayMinY) * m.scale;
    const srcW = st.src_w * m.scale;
    const srcH = st.src_h * m.scale;
    ctx.fillStyle = "rgba(20,20,20,0.9)";
    ctx.fillRect(sourceX, sourceY, srcW, srcH);
    const img = node._sfExpandImg;
    if (img && img.complete && img.naturalWidth > 0) {
      try {
        ctx.drawImage(img, sourceX, sourceY, srcW, srcH);
      } catch (e) {
        console.error("[SF Crop Expand] draw source failed:", e);
        drawPlaceholder(ctx, sourceX, sourceY, srcW, srcH, m.scale);
      }
    } else {
      drawPlaceholder(ctx, sourceX, sourceY, srcW, srcH, m.scale);
    }

    // 原图边界虚线
    ctx.strokeStyle = "rgba(100,150,255,0.6)";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(sourceX, sourceY, srcW, srcH);
    ctx.setLineDash([]);

    drawCropBox(ctx,
      { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h },
      st.src_w, st.src_h, m);

    // 底行背景条（Load/Browse 按钮与信息文本同排）
    const bottomY = nodeH - shiftLeft - BTN_H;
    ctx.fillStyle = th.panel2;
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, BTN_H + 8, 4);
    ctx.fill();
    ctx.strokeStyle = th.border;
    ctx.strokeRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, BTN_H + 8);

    drawButtons(ctx, node);

    // 信息文本（与底行按钮同排，右对齐到输出槽区前；空间不足时截断 "…"）
    ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
    ctx.font = "10px Arial";
    ctx.textAlign = "right";
    const ext = isExtended(
      { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, st.src_w, st.src_h)
      ? " (Extended)" : "";
    const fullText = `Source: ${st.src_w}\u00d7${st.src_h} | Crop: ${Math.round(st.crop_w)}\u00d7${Math.round(st.crop_h)}${ext}`;
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

// ── 交互 ──────────────────────────────────────────────────────────────────

function finalizeDrag(node, canvas) {
  const drag = node._sfExpandDrag;
  if (!drag) return false;
  node._sfExpandDrag = null;
  const rect = roundRect(drag.curRect);
  setState(node, { crop_x: rect.x, crop_y: rect.y, crop_w: rect.w, crop_h: rect.h });
  if (canvas) canvas.style.cursor = "default";
  stateChanged(node);
  return true;
}

function setupInteractions(node) {
  node.onMouseDown = (e, localPos) => {
    // 仅左键起拖（右键/中键交给原位/菜单；buttons 缺失的旧调用放行）
    if (e && e.button !== 0 && e.button !== undefined) return false;
    for (const b of node._sfExpandButtons) {
      const [bx, by, bw, bh] = buttonRect(b, node);
      if (localPos[0] >= bx && localPos[0] <= bx + bw &&
          localPos[1] >= by && localPos[1] <= by + bh) {
        b.action();
        return true;
      }
    }
    const st = getState(node);
    const m = computeDisplayMetrics(
      { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
      node.size[0], node.size[1], null);
    const p = localToImage(localPos[0], localPos[1], m);
    const handle = getHandleAtPoint(p.x, p.y,
      { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, m.scale);
    if (handle) {
      node._sfExpandDrag = {
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
    return false;
  };

  node.onMouseMove = (e, localPos, graphCanvas) => {
    // 释放丢失兜底：主键已松但拖拽状态残留 → 立即落定，绝不再改框
    if (node._sfExpandDrag && primaryButtonReleased(e)) {
      finalizeDrag(node, graphCanvas?.canvas);
      return true;
    }
    const st = getState(node);
    const dragging = !!node._sfExpandDrag;
    const m = computeDisplayMetrics(
      { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
      node.size[0], node.size[1],
      dragging ? node._sfExpandDrag.frozen : null);
    const p = localToImage(localPos[0], localPos[1], m);

    if (!dragging) {
      const handle = getHandleAtPoint(p.x, p.y,
        { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, m.scale);
      if (graphCanvas?.canvas) {
        graphCanvas.canvas.style.cursor = handle ? getCursorForHandle(handle) : "default";
      }
      return false;
    }

    const drag = node._sfExpandDrag;
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
  };

  node.onMouseUp = (e, localPos, graphCanvas) =>
    finalizeDrag(node, graphCanvas?.canvas);

  node.onDblClick = () => finalizeDrag(node, null);

  // 释放兜底（window capture，复用 sf_common）：画布 processMouseUp 会
  // stopPropagation 且仅对 node_over 回调 → 出界/纯点击释放收不到，bubble
  // 的 document 监听无效（见 sf_common.installNodeReleaseGuard 注释）
  installNodeReleaseGuard(node, () => {
    if (finalizeDrag(node, null)) {
      if (app.graph) app.graph.setDirtyCanvas(true, true);
    }
  }, { hook: "_sfExpandReleaseGuard" });

  // 拖放图片文件到节点显示区加载（sf_crop_source 共享实现；加载链路同
  // Load/Browse/粘贴）
  installSourceDrop(node, SOURCE_DROP_CFG);
}

// ── 工作流恢复 ────────────────────────────────────────────────────────────

// ── graphToPrompt：注入隐藏输入（只注入 lean 字段，Export/分享共用同一份 output）──
// 图索引复用 sf_pause_kit.buildClassNodeIndex/findNodeByPromptId（四闸门单源，
// 复合 id + 子图环守卫）。

if (!app._sfCropExpandPromptPatched) {
  app._sfCropExpandPromptPatched = true;
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
          const state = node?.properties?.[STATE_PROP] || JSON.stringify(DEFAULT_STATE);
          entry.inputs = entry.inputs || {};
          entry.inputs[HIDDEN_INPUT] = state;
        }
      }
    } catch (e) {
      console.warn("[SF Crop Expand] could not inject state:", (e && e.message) || e);
    }
    return result;
  };
}

// ── 右下角 resize cursor 视觉修正（sf_common 共享安装器）────────────────────
// 命中区维持原生 15×15（不做扩大）、区内直接写 style.cursor 的修法见
// sf_common.installResizeCornerCursor 注释（§44）。
installResizeCornerCursor(CLASS, hitResizeCornerSE);


// ── 注册 ──────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "sfnodes.CropExpand",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    // 拖拽 resize 的最小值取自 node.computeSize()（双端同款：onDrag 里
    // clamp 到 computeSize 再 setSize）——包装抬高到 MIN 一处钳住两条路径
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
      this._sfExpandButtons = buildButtons(this);
      setupDrawing(this);
      setupInteractions(this);
      // Ctrl+V 粘贴剪贴板图片：installPasteHandler（选中判定/防抢输入框/
      // 清扫自动 pasted/ LoadImage 均在公共实现内），加载走既有链路
      installPasteHandler({
        comfyClass: CLASS,
        hook: "_sfExpandPaste",
        onPasteImage: (n, dataURL) => n._sfExpandPaste(dataURL),
      });
      this._sfExpandPaste = (dataURL) => storeSource(this, dataURL, SOURCE_CFG);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      if (onConfigure) onConfigure.apply(this, [info]);
      // 恢复源图预览（状态本体在 properties 里，已随 info 恢复）
      clampNodeSize(this);
      if (!this._sfExpandButtons) {
        this._sfExpandButtons = buildButtons(this);
        setupDrawing(this);
        setupInteractions(this);
      }
      restoreSourceImage(this, SOURCE_CFG);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (onRemoved) onRemoved.apply(this, []);
      removeNodeReleaseGuard(this, { hook: "_sfExpandReleaseGuard" });
    };
  },
});
