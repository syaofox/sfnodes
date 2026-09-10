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
// brush_opacity/brush_color/eraser_color 不进注入，改预览不重跑，
// 同 sf_outpaint.js / sf_crop_expand.js 先例；Python hidden 已声明，
// schema 内不被剥离）。
// 源图持久化：dataURL 经 CropAPI.uploadSrc 落盘 input/sfnodes_crop/
// （复用 SFImageCrop 的路由，零新增后端路由），状态只存 src_path——
// 工作流重载经 /view 恢复预览（原版 base64 进 workflow + 会话 Map 缓存，
// 文件巨大且刷新丢图，已确认差异）。
// 交互数学在纯库 sf_brush_mask_lib.js（无 app 依赖可 .mjs 直测）。
// 最小尺寸钳制 + 右下角 cursor 补写同 sf_crop_expand.js（§44 同款）。
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { CropAPI } from "./sf_crop_core.js";
import { sfToast, buildSourceURL, getSfAccent, installPasteHandler, sfApiUrl } from "./sf_common.js";
import { showImageBrowser } from "./image_browser.js";
import { parseAnnotatedImageValue } from "./sf_common.js";
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
  hitStepper,
  wheelDir,
  wheelAction,
} from "./sf_brush_mask_lib.js";

const CLASS = "SFImageBrushMask";
const HIDDEN_INPUT = "SFBrushMaskJson"; // 必须与 brush_mask.py 的隐藏输入一致
const STATE_PROP = "sfBrushMaskState";

const DEFAULT_STATE = {
  src_path: "",
  src_w: 512,
  src_h: 512,
  brush_size: 80,
  strokes: [],
  // 以下仅预览语义（不进 lean 注入，后端忽略）：
  brush_opacity: 0.5,
  brush_color: "255,255,255",
  eraser_color: "255,50,50",
  brush_mode: "brush",
  // SAM 对话框记忆（不进 lean 注入；SAM 结果即 fill 笔触进 strokes）
  sam_prompt: "",
  sam_threshold: 0.5,
  sam_refine: 2,
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
  };
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

// ── 图片加载（按钮 / 拖放 / 粘贴 / Browse 共用）────────────────────────────

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
    const res = await CropAPI.uploadSrc("brushmask_" + Date.now(), dataURL);
    const srcPath = res?.path || "";
    if (!srcPath) {
      sfToast({ summary: "SF Brush Mask", detail: "源图上传失败，已取消加载", severity: "error", fallbackTag: "SF Brush Mask" });
      return;
    }
    setState(node, {
      src_path: srcPath,
      src_w: dims.w,
      src_h: dims.h,
      strokes: [], // 换图清空笔触（原版同款语义）
    });
    const img = new Image();
    img.onload = () => {
      node._sfBrushImg = img;
      stateChanged(node);
    };
    img.src = dataURL;
  } catch (err) {
    console.error("[SF Brush Mask] load image failed:", err);
    sfToast({ summary: "SF Brush Mask", detail: "加载图片失败", severity: "error", fallbackTag: "SF Brush Mask" });
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

// Browse 按钮：复用 SF Load Image Browser 弹窗（选择器模式），
// 选中后经 /view 取原始字节 → dataURL → 既有落盘+状态链路
function browseImage(node) {
  showImageBrowser(node, {
    onPick: async (annotated) => {
      const part = parseAnnotatedImageValue(annotated);
      const url = buildSourceURL(part);
      if (!url) return;
      try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();
        const dataURL = await new Promise((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => resolve(r.result);
          r.onerror = reject;
          r.readAsDataURL(blob);
        });
        await loadAndStoreImage(node, dataURL);
      } catch (err) {
        console.error("[SF Brush Mask] browse load failed:", err);
        sfToast({ summary: "SF Brush Mask", detail: "从图片浏览器加载失败", severity: "error", fallbackTag: "SF Brush Mask" });
      }
    },
  });
}

// ── 控件（左竖列 + 底行：绘制与命中共用同一几何）──────────────────────────
// 左竖列（x=shiftLeft, w=30, h=18，步进 22，列顶 16）：Brush/Erase 模式 →
// Clear/Undo → Size±/Opa± 步进 → BCol/ECol 取色（背景即当前色）。
// 底行（y 运行时解析为 nodeH-shiftLeft-21）：Load Image/Browse + 信息文本。

function toolText(id) {
  return {
    brush: "Brush",
    erase: "Erase",
    clear: "Clear",
    undo: "Undo",
    sizeMinus: "S−",
    sizePlus: "S+",
    opaMinus: "O−",
    opaPlus: "O+",
    brushColor: "BCol",
    eraserColor: "ECol",
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
    isColor: id === "brushColor" ? "brush" : id === "eraserColor" ? "erase" : null,
  }));
  // 底行：Load Image/Browse（与信息文本同排，y 运行时解析）
  buttons.push(
    { id: "load", text: "Load Image", x: 10, y: BOTTOM_Y, w: 72, h: 21 },
    { id: "browse", text: "Browse", x: 87, y: BOTTOM_Y, w: 48, h: 21 },
  );
  return buttons;
}

function buttonAction(node, id) {
  const st = getState(node);
  if (id === "load") pickFile(node);
  else if (id === "browse") browseImage(node);
  else if (id === "brush") setState(node, { brush_mode: "brush" });
  else if (id === "erase") setState(node, { brush_mode: st.brush_mode === "erase" ? "brush" : "erase" });
  else if (id === "clear") setState(node, { strokes: [] });
  else if (id === "undo") {
    if (st.strokes.length > 0) setState(node, { strokes: st.strokes.slice(0, -1) });
    else return;
  } else if (id === "sizeMinus") setState(node, { brush_size: stepBrushSize(st.brush_size, -1) });
  else if (id === "sizePlus") setState(node, { brush_size: stepBrushSize(st.brush_size, +1) });
  else if (id === "opaMinus") setState(node, { brush_opacity: stepOpacity(st.brush_opacity, -1) });
  else if (id === "opaPlus") setState(node, { brush_opacity: stepOpacity(st.brush_opacity, +1) });
  else if (id === "brushColor") { pickColor(node, "brush"); return; }
  else if (id === "eraserColor") { pickColor(node, "erase"); return; }
  else return;
  stateChanged(node);
}

function pickColor(node, which) {
  const st = getState(node);
  const cur = which === "erase" ? st.eraser_color : st.brush_color;
  const rgb = String(cur || "255,255,255").split(",").map((c) => parseInt(String(c).trim(), 10));
  const hex = "#" + rgb.map((c) => Math.max(0, Math.min(255, c || 0)).toString(16).padStart(2, "0")).join("");
  const input = document.createElement("input");
  input.type = "color";
  input.value = /^#[0-9a-f]{6}$/i.test(hex) ? hex : "#ffffff";
  input.onchange = (e) => {
    const h = e.target.value;
    const c = `${parseInt(h.substr(1, 2), 16)},${parseInt(h.substr(3, 2), 16)},${parseInt(h.substr(5, 2), 16)}`;
    if (which === "erase") setState(node, { eraser_color: c });
    else setState(node, { brush_color: c });
    stateChanged(node);
  };
  input.click();
}

// 取色按钮文字按背景亮度取黑/白（CropExpand Color 按钮同款）
function colorTextStyle(color) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(String(color || ""));
  if (!m) {
    const rgb = String(color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
    if (rgb.length === 3 && rgb.every((v) => Number.isFinite(v))) {
      const brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000;
      return brightness > 128 ? "rgba(0,0,0,0.9)" : "rgba(255,255,255,0.9)";
    }
    return "rgba(255,255,255,0.9)";
  }
  const brightness = (parseInt(m[1], 16) * 299 + parseInt(m[2], 16) * 587 + parseInt(m[3], 16) * 114) / 1000;
  return brightness > 128 ? "rgba(0,0,0,0.9)" : "rgba(255,255,255,0.9)";
}

// ── SAM（右键菜单 → 核心 SAM3_Detect → fill 笔触并入列表统一管理）────────
// 后端见 nodes/image/brush_mask_sam.py（三路由 sam/sam_unload/sam_status）。

async function samPost(path, body) {
  const res = await api.fetchApi(sfApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = null;
  try { data = await res.json(); } catch { /* 非 JSON 回退 */ }
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data || {};
}

async function runSamMask(node, prompt, threshold, refine) {
  const st = getState(node);
  if (!st.src_path) {
    sfToast({ summary: "SF Brush Mask", detail: "先加载源图再跑 SAM", severity: "warn", fallbackTag: "SF Brush Mask" });
    return;
  }
  sfToast({ summary: "SF Brush Mask", detail: `SAM 推理中：${prompt || "object"}…（首次需加载 1.7GB 模型）`, severity: "info", life: 5000, fallbackTag: "SF Brush Mask" });
  try {
    const data = await samPost("/api/sfnodes/brush_mask/sam", {
      src_path: st.src_path, prompt, threshold, refine_iterations: refine,
    });
    const incoming = Array.isArray(data && data.strokes) ? data.strokes : [];
    if (!incoming.length) {
      setState(node, { sam_prompt: prompt, sam_threshold: threshold, sam_refine: refine });
      stateChanged(node);
      sfToast({ summary: "SF Brush Mask", detail: "SAM 未检出目标（空结果，笔触不变）", severity: "warn", fallbackTag: "SF Brush Mask" });
      return;
    }
    setState(node, {
      strokes: [...st.strokes, ...incoming],
      sam_prompt: prompt,
      sam_threshold: threshold,
      sam_refine: refine,
    });
    stateChanged(node);
    const cov = data.coverage != null ? `覆盖 ${Math.round(data.coverage * 100)}%，` : "";
    sfToast({ summary: "SF Brush Mask", detail: `SAM 并入 ${incoming.length} 个填充笔触（${cov}可擦除/撤销）`, severity: "success", fallbackTag: "SF Brush Mask" });
  } catch (err) {
    console.error("[SF Brush Mask] sam failed:", err);
    sfToast({ summary: "SF Brush Mask", detail: `SAM 失败：${(err && err.message) || err}`, severity: "error", life: 6000, fallbackTag: "SF Brush Mask" });
  }
}

async function unloadSamModel() {
  try {
    const data = await samPost("/api/sfnodes/brush_mask/sam_unload", {});
    sfToast({
      summary: "SF Brush Mask",
      detail: data && data.unloaded ? "SAM 模型已卸载，显存已释放" : "SAM 模型未在驻留，无需卸载",
      severity: "info", fallbackTag: "SF Brush Mask",
    });
  } catch (err) {
    console.error("[SF Brush Mask] sam unload failed:", err);
    sfToast({ summary: "SF Brush Mask", detail: `卸载失败：${(err && err.message) || err}`, severity: "error", fallbackTag: "SF Brush Mask" });
  }
}

// SAM prompt 对话框（prompt 文本 + threshold；Enter 确认 / Esc 关闭；
// 放行 ctrl/meta/alt 组合键。样式同 CropExpand Custom 比例弹窗。）
function openSamDialog(node) {
  const st = getState(node);
  if (!st.src_path) {
    sfToast({ summary: "SF Brush Mask", detail: "先加载源图再跑 SAM", severity: "warn", fallbackTag: "SF Brush Mask" });
    return;
  }
  if (document.getElementById("sf-brush-mask-sam-overlay")) return;

  const overlay = document.createElement("div");
  overlay.id = "sf-brush-mask-sam-overlay";
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:9999;";

  const dialog = document.createElement("div");
  dialog.style.cssText = "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);" +
    "background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:12px 14px;" +
    "box-shadow:0 4px 20px rgba(0,0,0,0.5);width:300px;box-sizing:border-box;";
  const lastPrompt = st.sam_prompt || "";
  const lastThr = st.sam_threshold ?? 0.5;
  const lastRefine = st.sam_refine ?? 2;
  dialog.innerHTML = `
    <div style="color:#ddd;font-size:13px;margin-bottom:10px;font-weight:bold;">SAM 蒙版：文本选择</div>
    <div style="margin-bottom:10px;">
      <label style="color:#aaa;font-size:10px;display:block;margin-bottom:3px;">Prompt（英文，如 person / car，为空按 object）</label>
      <input type="text" id="sf-bm-sam-prompt" value="${String(lastPrompt).replace(/"/g, "&quot;")}" placeholder="person"
        style="width:100%;padding:5px;background:#1a1a1a;border:1px solid #555;border-radius:3px;color:#ddd;font-size:13px;box-sizing:border-box;">
      <div style="color:#777;font-size:10px;margin-top:3px;">多人用 person:3（:N = 每类最多 N 个），多类用逗号分隔</div>
    </div>
    <div style="margin-bottom:10px;">
      <label style="color:#aaa;font-size:10px;display:block;margin-bottom:3px;">Threshold（0-1，越低越多）</label>
      <input type="number" id="sf-bm-sam-thr" value="${lastThr}" min="0" max="1" step="0.05"
        style="width:100%;padding:5px;background:#1a1a1a;border:1px solid #555;border-radius:3px;color:#ddd;font-size:13px;box-sizing:border-box;">
    </div>
    <div style="margin-bottom:10px;">
      <label style="color:#aaa;font-size:10px;display:block;margin-bottom:3px;">Refine（0-5，SAM 解码精修轮数，0=用粗蒙版）</label>
      <input type="number" id="sf-bm-sam-refine" value="${lastRefine}" min="0" max="5" step="1"
        style="width:100%;padding:5px;background:#1a1a1a;border:1px solid #555;border-radius:3px;color:#ddd;font-size:13px;box-sizing:border-box;">
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;">
      <button id="sf-bm-sam-cancel" style="padding:5px 12px;background:#444;border:none;border-radius:3px;color:#ddd;cursor:pointer;font-size:12px;">Cancel</button>
      <button id="sf-bm-sam-ok" style="padding:5px 12px;background:#4a90e2;border:none;border-radius:3px;color:white;cursor:pointer;font-size:12px;">Run SAM</button>
    </div>`;
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    overlay.remove();
  };
  overlay.addEventListener("pointerdown", (e) => { if (e.target === overlay) close(); });
  window.addEventListener("keydown", function esc(e) {
    if (e.key === "Escape") { close(); window.removeEventListener("keydown", esc); }
  });

  const promptInput = dialog.querySelector("#sf-bm-sam-prompt");
  const thrInput = dialog.querySelector("#sf-bm-sam-thr");
  const refineInput = dialog.querySelector("#sf-bm-sam-refine");
  setTimeout(() => promptInput.focus(), 100);

  const apply = () => {
    let thr = parseFloat(thrInput.value);
    if (!Number.isFinite(thr)) thr = 0.5;
    thr = Math.max(0, Math.min(1, thr));
    let refine = parseInt(refineInput.value, 10);
    if (!Number.isFinite(refine)) refine = 2;
    refine = Math.max(0, Math.min(5, refine));
    const p = promptInput.value || "";
    close();
    runSamMask(node, p, thr, refine);
  };
  dialog.querySelector("#sf-bm-sam-ok").onclick = apply;
  dialog.querySelector("#sf-bm-sam-cancel").onclick = close;
  for (const el of [promptInput, thrInput, refineInput]) {
    el.onkeydown = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") {
        // 吞掉回车：否则 keydown 上浮到 window，会触发全局键位
        // （如用户自绑的 Enter 队列）导致与对话框无关的报错
        e.preventDefault();
        e.stopPropagation();
        apply();
      } else if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        close();
      }
    };
  }
}

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

function drawStrokePath(ctx, pts, m, lineW, style, fill) {
  if (!pts || pts.length === 0) return;
  // fill 笔触（SAM 并入）：整体填充闭合多边形
  if (fill) {
    if (pts.length === 1) {
      const p = imageToLocal(pts[0][0], pts[0][1], m);
      ctx.fillStyle = style;
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(1, lineW / 2), 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    ctx.fillStyle = style;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const p = imageToLocal(pts[i][0], pts[i][1], m);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fill();
    return;
  }
  ctx.lineWidth = Math.max(1, lineW);
  ctx.strokeStyle = style;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (pts.length === 1) {
    const p = imageToLocal(pts[0][0], pts[0][1], m);
    // 单点：画一个直径=线宽的圆盘（后端同款：单点印章）
    ctx.fillStyle = style;
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(1, lineW / 2), 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const p = imageToLocal(pts[i][0], pts[i][1], m);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
}

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
        const src = b.isColor === "erase" ? st.eraser_color : st.brush_color;
        const rgb = String(src || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
        ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.9)`;
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
      ctx.fillStyle = b.isColor
        ? colorTextStyle(b.isColor === "erase" ? st.eraser_color : st.brush_color)
        : "rgba(220,220,220,0.9)";
      ctx.font = b.y === BOTTOM_Y ? "11px Arial" : "10px Arial";
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

    // 已落笔触：先刷后擦（与原版同序；擦以红色预览叠加；fill 整体填充）
    const brushRGB = String(st.brush_color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
    const brushStyle = `rgba(${brushRGB[0]},${brushRGB[1]},${brushRGB[2]},${st.brush_opacity})`;
    for (const s of st.strokes) {
      if ((s.mode || "brush") === "erase") continue;
      drawStrokePath(ctx, s.points, m, (s.size || st.brush_size) * m.scale, brushStyle, (s.mode || "brush") === "fill");
    }
    const eraserRGB = String(st.eraser_color || "255,50,50").split(",").map((v) => parseInt(String(v).trim(), 10));
    const eraserStyle = `rgba(${eraserRGB[0]},${eraserRGB[1]},${eraserRGB[2]},${st.brush_opacity})`;
    for (const s of st.strokes) {
      if ((s.mode || "brush") !== "erase") continue;
      drawStrokePath(ctx, s.points, m, (s.size || st.brush_size) * m.scale, eraserStyle);
    }
    // 进行中笔触
    if (node._sfBrushCur && node._sfBrushCur.length > 0) {
      drawStrokePath(ctx, node._sfBrushCur, m, st.brush_size * m.scale,
        st.brush_mode === "erase" ? eraserStyle : brushStyle);
    }

    // 底信息行文本（右对齐截断，落在按钮之上，两者无重叠）
    ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
    ctx.font = "10px Arial";
    ctx.textAlign = "right";
    const fullText = `Brush ${Math.round(st.brush_size)} · Op ${Math.round(st.brush_opacity * 100)}% · Strokes ${st.strokes.length} · ${st.src_w}\u00d7${st.src_h}`;
    const maxTextW = nodeW - shiftRight - 6 - (135 + 6); // 底行按钮右缘 135 + 间隙 6
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
    if (!node._sfBrushDrawing) return false;
    const lp = localPos || [e.canvasX - node.pos[0], e.canvasY - node.pos[1]];
    const [lx, ly] = lp;
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

  node.onMouseUp = (_e, _lp, graphCanvas) => finalizeStroke(node, graphCanvas?.canvas);

  node.onDblClick = () => finalizeStroke(node, null);

  // 全局释放兜底：鼠标移出节点区域松开也能落定
  if (!node._sfBrushGlobalUp) {
    node._sfBrushGlobalUp = () => {
      if (finalizeStroke(node, null)) {
        if (app.graph) app.graph.setDirtyCanvas(true, true);
      }
    };
    document.addEventListener("mouseup", node._sfBrushGlobalUp);
  }

  // 拖放图片文件到节点显示区加载
  node.onDragOver = (e) => {
    const m = metricsOf(node);
    const lx = e.canvasX - node.pos[0];
    const ly = e.canvasY - node.pos[1];
    const inArea = lx >= m.offsetX && lx <= m.offsetX + m.scaledW &&
      ly >= m.offsetY && ly <= m.offsetY + m.scaledH;
    if (inArea && e.dataTransfer?.types && Array.from(e.dataTransfer.types).includes("Files")) {
      e.preventDefault();
      e.stopPropagation();
      return true;
    }
    return false;
  };

  node.onDragDrop = (e) => {
    const m = metricsOf(node);
    const lx = e.canvasX - node.pos[0];
    const ly = e.canvasY - node.pos[1];
    if (lx < m.offsetX || lx > m.offsetX + m.scaledW ||
        ly < m.offsetY || ly > m.offsetY + m.scaledH) {
      return false;
    }
    const file = e.dataTransfer?.files?.[0];
    if (!file) return false;
    if (!file.type.startsWith("image/")) {
      console.warn("[SF Brush Mask] only image files are supported");
      return false;
    }
    const reader = new FileReader();
    reader.onload = (event) => loadAndStoreImage(node, event.target.result);
    reader.onerror = (err) => console.error("[SF Brush Mask] read file failed:", err);
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
    node._sfBrushImg = img;
    if (app.graph) app.graph.setDirtyCanvas(true, true);
  };
  img.src = url;
}

// ── graphToPrompt：注入隐藏输入（只注入 lean 字段，Export/分享共用同一份 output）──

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
          if (!index) index = buildNodeIndex();
          const node = findNodeById(index, id);
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

// ── 右下角 resize cursor 视觉修正（§44 同款：命中区维持原生 15×15， ──────
// 后注册 mousemove 在区内直接写 style.cursor，绕过 resizeDirection 被清空链路）

if (!app._sfBrushMaskCursorPatch) {
  app._sfBrushMaskCursorPatch = true;
  let _sfCursorOwned = false;
  window.addEventListener("mousemove", () => {
    const canvas = app.canvas;
    const pointer = canvas?.pointer;
    if (!pointer || pointer.eDown) return;
    const mx = canvas.graph_mouse?.[0], my = canvas.graph_mouse?.[1];
    if (mx == null || my == null) return;
    let inSE = false;
    for (const n of app.graph?._nodes || []) {
      if (n.comfyClass !== CLASS) continue;
      if (hitResizeCornerSE(mx - n.pos[0], my - n.pos[1], n.size[0], n.size[1])) {
        inSE = true;
        break;
      }
    }
    if (inSE) {
      if (pointer.resizeDirection !== "SE") pointer.resizeDirection = "SE";
      canvas.canvas.style.cursor = "nwse-resize";
      _sfCursorOwned = true;
    } else if (_sfCursorOwned) {
      _sfCursorOwned = false;
      if (canvas.canvas.style.cursor === "nwse-resize") canvas.canvas.style.cursor = "";
    }
  });
}

// ── 步进器滚轮快调 ────────────────────────────────────────────────────────
//
// 悬停左竖列 S±/O± 步进器时滚轮直接调值（上滚增大/下滚减小，每 tick 一步）：
// 引擎无节点级 onMouseWheel 钩子，故用 window capture 先手拦截（同
// installPasteHandler 先例）；仅命中四个步进器且无 Ctrl/Meta（捏合缩放手势）
// 时拦截，其余一律放行（画布缩放不受影响）。passive:false 否则 preventDefault
// 无效；折叠节点跳过（控件不可见）。

if (!app._sfBrushMaskWheelPatch) {
  app._sfBrushMaskWheelPatch = true;
  window.addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.metaKey) return;
    const dir = wheelDir(e.deltaY);
    if (!dir) return;
    const canvas = app.canvas;
    if (!canvas) return;
    const mx = canvas.graph_mouse?.[0], my = canvas.graph_mouse?.[1];
    if (mx == null || my == null) return;
    for (const n of app.graph?._nodes || []) {
      if (n.comfyClass !== CLASS && n.type !== CLASS) continue;
      if (n.flags?.collapsed || !n._sfBrushCtrls) continue;
      const lx = mx - n.pos[0], ly = my - n.pos[1];
      if (lx < 0 || ly < 0 || lx > n.size[0] || ly > n.size[1]) continue;
      const hovered = hitStepper(n._sfBrushCtrls, lx, ly);
      if (!hovered) continue;
      const action = wheelAction(hovered, dir);
      if (!action) continue;
      e.preventDefault();
      e.stopPropagation();
      buttonAction(n, action);
      return;
    }
  }, { passive: false, capture: true });
}

// ── 注册 ──────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "sfnodes.BrushMask",
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
      this._sfBrushPaste = (dataURL) => loadAndStoreImage(this, dataURL);
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
      restoreImage(this);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (onRemoved) onRemoved.apply(this, []);
      if (this._sfBrushGlobalUp) {
        document.removeEventListener("mouseup", this._sfBrushGlobalUp);
        this._sfBrushGlobalUp = null;
      }
    };

    // 右键菜单（any_pack.js 同款 getExtraMenuOptions 包装）
    const origMenu = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
      if (origMenu) origMenu.apply(this, arguments);
      if (!Array.isArray(options)) return;
      options.push({
        content: "SAM 蒙版：文本选择…",
        callback: () => openSamDialog(this),
      });
      options.push({
        content: "卸载 SAM 模型",
        callback: () => unloadSamModel(),
      });
    };
  },
});
