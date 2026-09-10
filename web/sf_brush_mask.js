// ==========================================================================
// sf_brush_mask.js - SF Image Brush Mask 主扩展
// ==========================================================================
//
// 复刻 ComfyUI-YCNodes_Toolkit ycimagebrushmask（Load Image Brush Mask）：
// 节点上直接加载图片（Load Image 按钮 / Browse 图片浏览器 / 拖放文件到
// 节点 / Ctrl+V 粘贴），画笔在图上直接涂抹遮罩（brush 涂白 / eraser 擦除）。
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
import { CropAPI } from "./sf_crop_core.js";
import { sfToast, buildSourceURL, getSfAccent, installPasteHandler } from "./sf_common.js";
import { showImageBrowser } from "./image_browser.js";
import { parseAnnotatedImageValue } from "./sf_common.js";
import {
  LAYOUT,
  ensureMinSize,
  hitResizeCornerSE,
  computeDisplayMetrics,
  localToImage,
  imageToLocal,
  clampToImage,
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

// lean 注入载荷：只含影响结果的字段（改 opacity/颜色/模式不重跑）
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

// ── 面板控件（按钮/滑块/色块：绘制与命中共用同一几何）─────────────────────
// 行1（y=8,h=21）：Load(56) Browse(52) Clear(44) Undo(44) Eraser(52) + 右侧色块列(40)
// 行2（y=34,h=12+label）：Size(150) Opacity(150)

function buildControls() {
  return {
    buttons: [
      { id: "load", text: "Load", x: 10, y: 8, w: 56, h: 21 },
      { id: "browse", text: "Browse", x: 70, y: 8, w: 52, h: 21 },
      { id: "clear", text: "Clear", x: 126, y: 8, w: 44, h: 21 },
      { id: "undo", text: "Undo", x: 174, y: 8, w: 44, h: 21 },
      { id: "eraser", text: "Eraser", x: 222, y: 8, w: 52, h: 21, isToggle: true },
    ],
    sliders: [
      { id: "size", label: "Size", x: 10, y: 34, w: 150, h: 12, min: 1, max: 200 },
      { id: "opacity", label: "Opacity", x: 170, y: 34, w: 150, h: 12, min: 0.1, max: 1.0 },
    ],
    colorW: 40,
  };
}

function buttonAction(node, id) {
  if (id === "load") pickFile(node);
  else if (id === "browse") browseImage(node);
  else if (id === "clear") {
    setState(node, { strokes: [] });
    stateChanged(node);
  } else if (id === "undo") {
    const st = getState(node);
    if (st.strokes.length > 0) {
      setState(node, { strokes: st.strokes.slice(0, -1) });
      stateChanged(node);
    }
  } else if (id === "eraser") {
    const st = getState(node);
    setState(node, { brush_mode: st.brush_mode === "erase" ? "brush" : "erase" });
    stateChanged(node);
  }
}

function sliderValue(node, st, id) {
  return id === "size" ? st.brush_size : st.brush_opacity;
}

function applySliderValue(node, id, v) {
  if (id === "size") setState(node, { brush_size: Math.max(1, Math.round(v)) });
  else setState(node, { brush_opacity: Math.max(0.1, Math.min(1, Math.round(v * 100) / 100)) });
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

function drawStrokePath(ctx, pts, m, lineW, style) {
  if (!pts || pts.length === 0) return;
  ctx.lineWidth = Math.max(1, lineW);
  ctx.strokeStyle = style;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  if (pts.length === 1) {
    const p = imageToLocal(pts[0][0], pts[0][1], m);
    // 单点：画一个直径=线宽的圆盘（后端同款：单点印章）
    ctx.fillStyle = style;
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(1, lineW / 2), 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  for (let i = 0; i < pts.length; i++) {
    const p = imageToLocal(pts[i][0], pts[i][1], m);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
}

function setupDrawing(node) {
  const { shiftLeft, shiftRight, panelH, bottomH } = LAYOUT;

  node.onDrawForeground = (ctx) => {
    if (node.flags.collapsed) return false;
    const nodeW = node.size[0];
    const nodeH = node.size[1];
    const st = getState(node);
    const ctrls = node._sfBrushCtrls;
    const m = computeDisplayMetrics({ srcW: st.src_w, srcH: st.src_h }, nodeW, nodeH);
    const accent = getSfAccent() || "rgba(100,150,255,0.9)";

    // 顶面板背景
    const panelW = nodeW - shiftRight - shiftLeft + 8;
    ctx.fillStyle = "rgba(40,40,40,0.9)";
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, shiftLeft - 4, panelW, panelH, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(100,100,100,0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(shiftLeft - 4, shiftLeft - 4, panelW, panelH);

    // 按钮行
    for (const b of ctrls.buttons) {
      const active = b.isToggle && st.brush_mode === "erase";
      ctx.fillStyle = active ? accent : "rgba(60,60,60,0.7)";
      ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.strokeStyle = "rgba(150,150,150,0.6)";
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.fillStyle = "rgba(220,220,220,0.9)";
      ctx.font = "11px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(b.text, b.x + b.w / 2, b.y + b.h / 2);
    }

    // 色块列（右对齐到面板右缘）
    const groupX = shiftLeft - 4 + panelW - ctrls.colorW - 4;
    const colorDefs = [
      { which: "brush", label: "Brush", y: 8, h: 10, color: st.brush_color },
      { which: "erase", label: "Eraser", y: 19, h: 10, color: st.eraser_color },
    ];
    node._sfBrushColorRects = colorDefs.map((c) => ({ ...c, x: groupX, w: ctrls.colorW }));
    for (const c of node._sfBrushColorRects) {
      const rgb = String(c.color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.9)`;
      ctx.fillRect(c.x, c.y, c.w, c.h);
      ctx.strokeStyle = "rgba(150,150,150,0.6)";
      ctx.strokeRect(c.x, c.y, c.w, c.h);
      ctx.fillStyle = "rgba(40,40,40,0.9)";
      ctx.font = "9px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(c.label, c.x + c.w / 2, c.y + 1);
    }

    // 滑块行
    for (const s of ctrls.sliders) {
      const val = sliderValue(node, st, s.id);
      const ratio = (val - s.min) / (s.max - s.min);
      const thumb = 14;
      const thumbX = s.x + ratio * (s.w - thumb);
      ctx.fillStyle = "rgba(50,50,50,0.8)";
      ctx.fillRect(s.x, s.y, s.w, s.h);
      ctx.strokeStyle = "rgba(100,100,100,0.6)";
      ctx.strokeRect(s.x, s.y, s.w, s.h);
      ctx.fillStyle = "rgba(100,150,255,0.4)";
      ctx.fillRect(s.x, s.y, ratio * s.w, s.h);
      ctx.fillStyle = "rgba(120,170,255,0.9)";
      ctx.beginPath();
      ctx.roundRect(thumbX, s.y - 1, thumb, thumb + 2, 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(150,200,255,1.0)";
      ctx.strokeRect(thumbX, s.y - 1, thumb, thumb + 2);
      ctx.fillStyle = "rgba(200,200,200,0.9)";
      ctx.font = "11px Arial";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      const label = s.id === "size" ? `${s.label}: ${Math.round(val)}` : `${s.label}: ${Math.round(val * 100)}%`;
      ctx.fillText(label, s.x, s.y + s.h + 2);
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

    // 已落笔触：先刷后擦（与原版同序；擦以红色预览叠加）
    const brushRGB = String(st.brush_color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
    const brushStyle = `rgba(${brushRGB[0]},${brushRGB[1]},${brushRGB[2]},${st.brush_opacity})`;
    for (const s of st.strokes) {
      if ((s.mode || "brush") === "erase") continue;
      drawStrokePath(ctx, s.points, m, (s.size || st.brush_size) * m.scale, brushStyle);
    }
    for (const s of st.strokes) {
      if ((s.mode || "brush") !== "erase") continue;
      const c = String(st.eraser_color || "255,50,50").split(",").map((v) => parseInt(String(v).trim(), 10));
      drawStrokePath(ctx, s.points, m, (s.size || st.brush_size) * m.scale,
        `rgba(${c[0]},${c[1]},${c[2]},${st.brush_opacity})`);
    }
    // 进行中笔触
    if (node._sfBrushCur && node._sfBrushCur.length > 0) {
      const isErase = st.brush_mode === "erase";
      const src = isErase ? st.eraser_color : st.brush_color;
      const c = String(src || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
      drawStrokePath(ctx, node._sfBrushCur, m, st.brush_size * m.scale,
        `rgba(${c[0]},${c[1]},${c[2]},${st.brush_opacity})`);
    }

    // 底信息行（与 CropExpand 同款：背景条 + 右对齐截断文本）
    const bottomY = nodeH - shiftLeft - 21;
    ctx.fillStyle = "rgba(40,40,40,0.9)";
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, 21 + 8, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(100,100,100,0.5)";
    ctx.strokeRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, 21 + 8);
    ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
    ctx.font = "10px Arial";
    ctx.textAlign = "right";
    const fullText = `Source: ${st.src_w}\u00d7${st.src_h} | Strokes: ${st.strokes.length}`;
    const maxTextW = nodeW - shiftRight - 6 - 6;
    let label = fullText;
    if (ctx.measureText(fullText).width > maxTextW) {
      while (label.length > 1 && ctx.measureText(label + "\u2026").width > maxTextW) {
        label = label.slice(0, -1);
      }
      label += "\u2026";
    }
    ctx.fillText(label, nodeW - shiftRight - 6, bottomY + 21 / 2 + 3.5);
    void bottomH;
  };
}

// ── 交互 ──────────────────────────────────────────────────────────────────

function metricsOf(node) {
  const st = getState(node);
  return computeDisplayMetrics({ srcW: st.src_w, srcH: st.src_h }, node.size[0], node.size[1]);
}

function sliderAt(node, lx, ly) {
  for (const s of node._sfBrushCtrls.sliders) {
    if (lx >= s.x && lx <= s.x + s.w && ly >= s.y - 5 && ly <= s.y + s.h + 15) return s;
  }
  return null;
}

function updateSliderFromX(node, s, lx) {
  const thumb = 14;
  let ratio = (lx - s.x - thumb / 2) / (s.w - thumb);
  ratio = Math.max(0, Math.min(1, ratio));
  applySliderValue(node, s.id, s.min + ratio * (s.max - s.min));
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

    // 面板：滑块优先（拖拽），其次色块，再次按钮
    const s = sliderAt(node, lx, ly);
    if (s) {
      node._sfBrushSlider = s;
      updateSliderFromX(node, s, lx);
      return true;
    }
    for (const c of node._sfBrushColorRects || []) {
      if (lx >= c.x && lx <= c.x + c.w && ly >= c.y && ly <= c.y + c.h) {
        pickColor(node, c.which);
        return true;
      }
    }
    for (const b of node._sfBrushCtrls.buttons) {
      if (lx >= b.x && lx <= b.x + b.w && ly >= b.y && ly <= b.y + b.h) {
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
    const lp = localPos || [e.canvasX - node.pos[0], e.canvasY - node.pos[1]];
    const [lx, ly] = lp;
    if (node._sfBrushSlider) {
      updateSliderFromX(node, node._sfBrushSlider, lx);
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

  const up = (_e, _lp, graphCanvas) => {
    if (node._sfBrushSlider) {
      node._sfBrushSlider = null;
      return true;
    }
    return finalizeStroke(node, graphCanvas?.canvas);
  };
  node.onMouseUp = up;
  node.onDblClick = () => finalizeStroke(node, null);

  // 全局释放兜底：鼠标移出节点区域松开也能落定
  if (!node._sfBrushGlobalUp) {
    node._sfBrushGlobalUp = () => {
      node._sfBrushSlider = null;
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
  },
});
