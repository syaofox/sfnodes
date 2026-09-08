// ==========================================================================
// sf_crop_expand.js - SF Image Crop Expand 主扩展
// ==========================================================================
//
// 复刻 ComfyUI-YCNodes_Toolkit ycImageCrop（Load Image Crop Expand）：节点上
// 直接加载图片（Load Image 按钮 / 拖放文件到节点），拖拽一个可出界的裁剪框，
// 面板提供比例按钮（Free/预设/Custom）与填充色按钮。
//
// 状态真源 node.properties.sfCropExpandState（JSON 字符串，随工作流保存），
// 经 graphToPrompt 钩子注入隐藏输入 SFCropExpandJson（只注入不剪枝，同
// sf_outpaint.js 先例；Python hidden 已声明，schema 内不被剥离）。
// 源图持久化：dataURL 经 /api/sfnodes/crop/upload_src 落盘
// input/sfnodes_crop/，状态只存 src_path——工作流重载经 /view 恢复预览。
// 交互数学（含拖拽冻结快照防飘移）在纯库 sf_crop_expand_lib.js。
// ==========================================================================

import { app } from "/scripts/app.js";
import { CropAPI } from "./sf_crop_core.js";
import { sfToast, buildSourceURL, getSfAccent } from "./sf_common.js";
import { attachPopupDismiss } from "./sf_popup.js";
import {
  ASPECT_RATIOS,
  RATIO_PRESETS_ROW2,
  LAYOUT,
  MIN_NODE_WIDTH,
  MIN_NODE_HEIGHT,
  ratioFromAspect,
  computeDisplayMetrics,
  localToImage,
  getHandleAtPoint,
  getCursorForHandle,
  updateCropByDrag,
  roundRect,
  applyRatioToRect,
  isExtended,
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

// src_path → /view 记录（upload_src 返回 "sfnodes_crop/<file>"，前缀即子目录）
function srcViewPart(srcPath) {
  if (!srcPath) return null;
  const norm = String(srcPath).replace(/\\/g, "/");
  const slash = norm.indexOf("/");
  return {
    filename: slash >= 0 ? norm.slice(slash + 1) : norm,
    subfolder: slash >= 0 ? norm.slice(0, slash) : "",
    type: "input",
  };
}

// ── 图片加载（按钮 / 拖放共用）────────────────────────────────────────────

function imageDims(dataURL) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = reject;
    img.src = dataURL;
  });
}

async function loadAndStoreImage(node, dataURL) {
  try {
    const dims = await imageDims(dataURL);
    const res = await CropAPI.uploadSrc("cropexpand_" + Date.now(), dataURL);
    const srcPath = res?.path || "";
    if (!srcPath) {
      sfToast({ summary: "SF Crop Expand", detail: "源图上传失败，已取消加载", severity: "error", fallbackTag: "SF Crop Expand" });
      return;
    }
    setState(node, {
      src_path: srcPath,
      src_w: dims.w,
      src_h: dims.h,
      // 加载后控制框自动填满整图
      crop_x: 0,
      crop_y: 0,
      crop_w: dims.w,
      crop_h: dims.h,
      aspect_ratio: "free",
    });
    const img = new Image();
    img.onload = () => {
      node._sfExpandImg = img;
      updateNodeSize(node);
      stateChanged(node);
    };
    img.src = dataURL;
  } catch (err) {
    console.error("[SF Crop Expand] load image failed:", err);
    sfToast({ summary: "SF Crop Expand", detail: "加载图片失败", severity: "error", fallbackTag: "SF Crop Expand" });
  }
}

function pickFile(node) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => loadAndStoreImage(node, event.target.result);
    reader.readAsDataURL(file);
  };
  input.click();
}

// ── 面板按钮 ──────────────────────────────────────────────────────────────

function ratioLabel(key) {
  return ASPECT_RATIOS.find((r) => r.key === key)?.label || key;
}

function buildButtons(node) {
  const y1 = 10;
  const h1 = 21;
  const h2 = 18;
  const buttons = [
    { text: "Load Image", x: 10, y: y1, w: 80, h: h1, action: () => pickFile(node) },
    { text: "Reset", x: 95, y: y1, w: 50, h: h1, action: () => resetCrop(node) },
    { text: "Color", x: 150, y: y1, w: 50, h: h1, isColor: true, action: () => pickFillColor(node) },
    { text: "Free", x: 205, y: y1 + 2, w: 30, h: h2, isRatio: true, ratioKey: "free", action: () => setAspect(node, "free") },
    { text: "Custom", x: 240, y: y1 + 2, w: 50, h: h2, isRatio: true, ratioKey: "custom", action: () => openCustomRatioDialog(node) },
  ];
  let x = 10;
  const y2 = y1 + h1 + 5;
  for (const key of RATIO_PRESETS_ROW2) {
    buttons.push({
      text: ratioLabel(key),
      x, y: y2, w: 30, h: h2,
      isRatio: true, ratioKey: key,
      action: () => setAspect(node, key),
    });
    x += 35;
  }
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
    openCustomRatioDialog(node);
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
  const input = document.createElement("input");
  input.type = "color";
  input.value = getState(node).fill_color || "#000000";
  input.onchange = (e) => {
    setState(node, { fill_color: e.target.value });
    stateChanged(node);
  };
  input.click();
}

// ── Custom 比例弹窗（sf_popup 三关闭 + Enter 确认）────────────────────────

function openCustomRatioDialog(node) {
  if (document.getElementById("sf-crop-expand-ratio-overlay")) return;
  const st0 = getState(node);

  const overlay = document.createElement("div");
  overlay.id = "sf-crop-expand-ratio-overlay";
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:9999;";

  const dialog = document.createElement("div");
  dialog.style.cssText = "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);" +
    "background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:12px 14px;" +
    "box-shadow:0 4px 20px rgba(0,0,0,0.5);width:auto;";
  dialog.innerHTML = `
    <div style="color:#ddd;font-size:13px;margin-bottom:10px;font-weight:bold;">Custom Aspect Ratio</div>
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px;">
      <div>
        <label style="color:#aaa;font-size:10px;display:block;margin-bottom:3px;">Width</label>
        <input type="number" id="sf-ce-ratio-w" value="${st0.custom_w ?? 1}" min="0.1" step="0.1"
          style="width:100px;padding:5px;background:#1a1a1a;border:1px solid #555;border-radius:3px;color:#ddd;font-size:13px;box-sizing:border-box;">
      </div>
      <div style="color:#888;font-size:16px;margin-top:14px;">:</div>
      <div>
        <label style="color:#aaa;font-size:10px;display:block;margin-bottom:3px;">Height</label>
        <input type="number" id="sf-ce-ratio-h" value="${st0.custom_h ?? 1}" min="0.1" step="0.1"
          style="width:100px;padding:5px;background:#1a1a1a;border:1px solid #555;border-radius:3px;color:#ddd;font-size:13px;box-sizing:border-box;">
      </div>
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;">
      <button id="sf-ce-ratio-cancel" style="padding:5px 12px;background:#444;border:none;border-radius:3px;color:#ddd;cursor:pointer;font-size:12px;">Cancel</button>
      <button id="sf-ce-ratio-ok" style="padding:5px 12px;background:#4a90e2;border:none;border-radius:3px;color:white;cursor:pointer;font-size:12px;">OK</button>
    </div>`;
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    overlay.remove();
  };
  // 三关闭（外部点击 / Esc / 滚轮）复用公共库
  attachPopupDismiss(overlay, { onClose: close });

  const wInput = dialog.querySelector("#sf-ce-ratio-w");
  const hInput = dialog.querySelector("#sf-ce-ratio-h");
  setTimeout(() => wInput.focus(), 100);

  const apply = () => {
    const w = parseFloat(wInput.value);
    const h = parseFloat(hInput.value);
    if (isNaN(w) || isNaN(h) || w <= 0 || h <= 0) {
      wInput.style.borderColor = "#e74c3c";
      hInput.style.borderColor = "#e74c3c";
      return;
    }
    const st = setState(node, { custom_w: w, custom_h: h, aspect_ratio: "custom" });
    const rect = applyRatioToRect(
      { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, w / h);
    setState(node, { crop_x: rect.x, crop_y: rect.y, crop_w: rect.w, crop_h: rect.h });
    stateChanged(node);
    close();
  };

  dialog.querySelector("#sf-ce-ratio-ok").onclick = apply;
  dialog.querySelector("#sf-ce-ratio-cancel").onclick = close;
  // Enter 确认 / Esc 关闭；放行 ctrl/meta/alt 组合键（Ctrl+S 不能漏成浏览器行为）
  for (const el of [wInput, hInput]) {
    el.onkeydown = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") apply();
      else if (e.key === "Escape") close();
    };
    el.oninput = () => {
      wInput.style.borderColor = "#555";
      hInput.style.borderColor = "#555";
    };
  }
}

// ── 节点尺寸自适应 ────────────────────────────────────────────────────────

// 钳制节点尺寸不低于最小值（LiteGraph 按 schema 算的初始尺寸偏小，按钮外溢）
function clampNodeSize(node) {
  const w = Math.max(MIN_NODE_WIDTH, node.size?.[0] || 0);
  const h = Math.max(MIN_NODE_HEIGHT, node.size?.[1] || 0);
  node.size = [w, h];
}

function updateNodeSize(node) {
  const st = getState(node);
  const { shiftLeft, shiftRight, panelHeight } = LAYOUT;
  const maxDisplaySize = 500;
  const scale = Math.min(
    maxDisplaySize / Math.max(1, st.src_w),
    maxDisplaySize / Math.max(1, st.src_h),
    1.0,
  );
  const w = Math.max(MIN_NODE_WIDTH, Math.min(st.src_w * scale + shiftRight + shiftLeft, 800));
  const h = Math.max(MIN_NODE_HEIGHT, Math.min(st.src_h * scale + shiftLeft * 2 + panelHeight, 800));
  node.size = [w, h];
}

// ── 绘制 ──────────────────────────────────────────────────────────────────

function drawPlaceholder(ctx, x, y, width, height, scale) {
  ctx.fillStyle = "rgba(100,100,100,0.3)";
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = "rgba(150,150,150,0.2)";
  ctx.lineWidth = 1;
  const gridSize = 32 * scale;
  for (let gx = x; gx <= x + width; gx += gridSize) {
    ctx.beginPath();
    ctx.moveTo(gx, y);
    ctx.lineTo(gx, y + height);
    ctx.stroke();
  }
  for (let gy = y; gy <= y + height; gy += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, gy);
    ctx.lineTo(x + width, gy);
    ctx.stroke();
  }
}

function drawButtons(ctx, node) {
  const st = getState(node);
  const accent = getSfAccent() || "rgba(100,150,255,0.8)";
  for (const b of node._sfExpandButtons) {
    if (b.isRatio && b.ratioKey === st.aspect_ratio) {
      ctx.fillStyle = accent;
    } else if (b.isColor) {
      ctx.fillStyle = st.fill_color || "#000000";
    } else {
      ctx.fillStyle = "rgba(60,60,60,0.7)";
    }
    ctx.fillRect(b.x, b.y, b.w, b.h);
    ctx.strokeStyle = "rgba(150,150,150,0.6)";
    ctx.lineWidth = 1;
    ctx.strokeRect(b.x, b.y, b.w, b.h);

    if (b.isColor) {
      // 颜色按钮文字按背景亮度取黑/白
      const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(String(st.fill_color || "#000000"));
      if (m) {
        const r = parseInt(m[1], 16), g = parseInt(m[2], 16), bl = parseInt(m[3], 16);
        const brightness = (r * 299 + g * 587 + bl * 114) / 1000;
        ctx.fillStyle = brightness > 128 ? "rgba(0,0,0,0.9)" : "rgba(255,255,255,0.9)";
      } else {
        ctx.fillStyle = "rgba(255,255,255,0.9)";
      }
    } else {
      ctx.fillStyle = "rgba(220,220,220,0.9)";
    }

    ctx.font = b.isRatio ? "10px Arial" : "11px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    let text = b.text;
    if (b.ratioKey === "custom" && st.aspect_ratio === "custom") {
      text = `${st.custom_w || 1}:${st.custom_h || 1}`;
    }
    ctx.fillText(text, b.x + b.w / 2, b.y + b.h / 2);
  }
}

function drawCropBox(ctx, node, m) {
  const st = getState(node);
  const x1 = m.offsetX + (st.crop_x - m.displayMinX) * m.scale;
  const y1 = m.offsetY + (st.crop_y - m.displayMinY) * m.scale;
  const x2 = x1 + st.crop_w * m.scale;
  const y2 = y1 + st.crop_h * m.scale;

  const imgX1 = m.offsetX + (0 - m.displayMinX) * m.scale;
  const imgY1 = m.offsetY + (0 - m.displayMinY) * m.scale;
  const imgX2 = imgX1 + st.src_w * m.scale;
  const imgY2 = imgY1 + st.src_h * m.scale;

  // 裁切框外（原图内）的半透明遮罩
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  if (y1 > imgY1) ctx.fillRect(imgX1, imgY1, imgX2 - imgX1, y1 - imgY1);
  if (y2 < imgY2) ctx.fillRect(imgX1, y2, imgX2 - imgX1, imgY2 - y2);
  if (x1 > imgX1) ctx.fillRect(imgX1, Math.max(y1, imgY1), x1 - imgX1, Math.min(y2, imgY2) - Math.max(y1, imgY1));
  if (x2 < imgX2) ctx.fillRect(x2, Math.max(y1, imgY1), imgX2 - x2, Math.min(y2, imgY2) - Math.max(y1, imgY1));

  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // 九宫格辅助线
  ctx.strokeStyle = "rgba(255,255,255,0.4)";
  ctx.lineWidth = 1;
  const w3 = (x2 - x1) / 3;
  const h3 = (y2 - y1) / 3;
  for (let i = 1; i < 3; i++) {
    ctx.beginPath();
    ctx.moveTo(x1 + w3 * i, y1);
    ctx.lineTo(x1 + w3 * i, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1 + h3 * i);
    ctx.lineTo(x2, y1 + h3 * i);
    ctx.stroke();
  }

  // 控制点
  const hs = 10;
  const handles = [
    { x: x1, y: y1 }, { x: x2, y: y1 }, { x: x1, y: y2 }, { x: x2, y: y2 },
    { x: (x1 + x2) / 2, y: y1 }, { x: (x1 + x2) / 2, y: y2 },
    { x: x1, y: (y1 + y2) / 2 }, { x: x2, y: (y1 + y2) / 2 },
  ];
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.strokeStyle = "rgba(0,0,0,0.8)";
  ctx.lineWidth = 1;
  for (const p of handles) {
    ctx.fillRect(p.x - hs / 2, p.y - hs / 2, hs, hs);
    ctx.strokeRect(p.x - hs / 2, p.y - hs / 2, hs, hs);
  }
}

function setupDrawing(node) {
  const { shiftLeft, shiftRight, panelHeight } = LAYOUT;

  node.onDrawForeground = (ctx) => {
    if (node.flags.collapsed) return false;

    const nodeW = node.size[0];
    const nodeH = node.size[1];
    const dragging = !!node._sfExpandDrag;
    const st = getState(node);

    // 控制面板背景
    ctx.fillStyle = "rgba(40,40,40,0.9)";
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, shiftLeft - 4, nodeW - shiftRight - shiftLeft + 8, panelHeight, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(100,100,100,0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(shiftLeft - 4, shiftLeft - 4, nodeW - shiftRight - shiftLeft + 8, panelHeight);

    const m = computeDisplayMetrics(
      { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
      nodeW, nodeH,
      dragging ? node._sfExpandDrag.frozen : null,
    );

    // 扩展区背景 + 网格
    ctx.fillStyle = "rgba(60,60,60,0.8)";
    ctx.beginPath();
    ctx.roundRect(m.offsetX - 4, m.offsetY - 4, m.scaledDisplayWidth + 8, m.scaledDisplayHeight + 8, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(80,80,80,0.3)";
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

    drawCropBox(ctx, node, m);
    drawButtons(ctx, node);

    // 信息文本
    ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
    ctx.font = "10px Arial";
    ctx.textAlign = "center";
    const ext = isExtended(
      { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, st.src_w, st.src_h)
      ? " (Extended)" : "";
    ctx.fillText(
      `Source: ${st.src_w}\u00d7${st.src_h} | Crop: ${Math.round(st.crop_w)}\u00d7${Math.round(st.crop_h)}${ext}`,
      nodeW / 2,
      m.offsetY + m.scaledDisplayHeight + 15,
    );
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
    for (const b of node._sfExpandButtons) {
      if (localPos[0] >= b.x && localPos[0] <= b.x + b.w &&
          localPos[1] >= b.y && localPos[1] <= b.y + b.h) {
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

  // 全局释放兜底：鼠标移出节点区域松开也能落定
  if (!node._sfExpandGlobalUp) {
    node._sfExpandGlobalUp = () => {
      if (finalizeDrag(node, null)) {
        if (app.graph) app.graph.setDirtyCanvas(true, true);
      }
    };
    document.addEventListener("mouseup", node._sfExpandGlobalUp);
  }

  // 拖放图片文件到节点显示区加载
  node.onDragOver = (e) => {
    const st = getState(node);
    const m = computeDisplayMetrics(
      { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
      node.size[0], node.size[1], null);
    const localX = e.canvasX - node.pos[0];
    const localY = e.canvasY - node.pos[1];
    const inArea = localX >= m.offsetX && localX <= m.offsetX + m.scaledDisplayWidth &&
      localY >= m.offsetY && localY <= m.offsetY + m.scaledDisplayHeight;
    if (inArea && e.dataTransfer?.types && Array.from(e.dataTransfer.types).includes("Files")) {
      e.preventDefault();
      e.stopPropagation();
      return true;
    }
    return false;
  };

  node.onDragDrop = (e) => {
    const st = getState(node);
    const m = computeDisplayMetrics(
      { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
      node.size[0], node.size[1], null);
    const localX = e.canvasX - node.pos[0];
    const localY = e.canvasY - node.pos[1];
    if (localX < m.offsetX || localX > m.offsetX + m.scaledDisplayWidth ||
        localY < m.offsetY || localY > m.offsetY + m.scaledDisplayHeight) {
      return false;
    }
    const file = e.dataTransfer?.files?.[0];
    if (!file) return false;
    if (!file.type.startsWith("image/")) {
      console.warn("[SF Crop Expand] only image files are supported");
      return false;
    }
    const reader = new FileReader();
    reader.onload = (event) => loadAndStoreImage(node, event.target.result);
    reader.onerror = (err) => console.error("[SF Crop Expand] read file failed:", err);
    reader.readAsDataURL(file);
    e.preventDefault();
    e.stopPropagation();
    return true;
  };
}

// ── 工作流恢复 ────────────────────────────────────────────────────────────

function restoreImage(node) {
  const st = getState(node);
  const part = srcViewPart(st.src_path);
  if (!part) return;
  const url = buildSourceURL(part, true);
  if (!url) return;
  const img = new Image();
  img.onload = () => {
    node._sfExpandImg = img;
    if (app.graph) app.graph.setDirtyCanvas(true, true);
  };
  img.src = url;
}

// ── graphToPrompt：注入隐藏输入（只注入不剪枝，Export/分享共用同一份 output）──

function buildNodeIndex() {
  const index = new Map();
  const visit = (graph) => {
    if (!graph) return;
    for (const n of graph._nodes || graph.nodes || []) {
      if (!n) continue;
      if (n.comfyClass === CLASS || n.type === CLASS) index.set(String(n.id), n);
      const inner = n.subgraph || n.graph || n._graph;
      if (inner && inner !== graph) visit(inner);
    }
  };
  visit(app.graph);
  return index;
}

function findNodeById(index, id) {
  const s = String(id);
  if (index.has(s)) return index.get(s);
  const tail = s.includes(":") ? s.slice(s.lastIndexOf(":") + 1) : null;
  return tail && index.has(tail) ? index.get(tail) : null;
}

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
          if (!index) index = buildNodeIndex();
          const node = findNodeById(index, id);
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

// ── 注册 ──────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "sfnodes.CropExpand",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

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
      restoreImage(this);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (onRemoved) onRemoved.apply(this, []);
      if (this._sfExpandGlobalUp) {
        document.removeEventListener("mouseup", this._sfExpandGlobalUp);
        this._sfExpandGlobalUp = null;
      }
    };
  },
});
