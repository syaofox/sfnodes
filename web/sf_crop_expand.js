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
// 源图持久化：dataURL 经 /api/sfnodes/crop/upload_src 落盘
// input/sfnodes_crop/，状态只存 src_path——工作流重载经 /view 恢复预览。
// 交互数学（含拖拽冻结快照防飘移）在纯库 sf_crop_expand_lib.js。
// ==========================================================================

import { app } from "/scripts/app.js";
import { CropAPI } from "./sf_crop_core.js";
import { sfToast, sfApiUrl, buildSourceURL, getSfAccent, parseAnnotatedImageValue, installPasteHandler } from "./sf_common.js";
import { attachPopupDismiss } from "./sf_popup.js";
import { showImageBrowser } from "./image_browser.js";
import {
  ASPECT_RATIOS,
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
  normalizeRatioPresets,
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
      // 不改节点大小：显示区 scale 动态计算，自动适配现有画布区域
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
        console.error("[SF Crop Expand] browse load failed:", err);
        sfToast({ summary: "SF Crop Expand", detail: "从图片浏览器加载失败", severity: "error", fallbackTag: "SF Crop Expand" });
      }
    },
  });
}

// ── 面板按钮 ──────────────────────────────────────────────────────────────

function ratioLabel(key) {
  return ASPECT_RATIOS.find((r) => r.key === key)?.label || key;
}

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
      action: () => openCustomRatioDialog(node),
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
    { text: "Load", x: 10, y: BOTTOM_Y, w: 44, h: 21, action: () => pickFile(node) },
    { text: "Browse", x: 58, y: BOTTOM_Y, w: 48, h: 21, action: () => browseImage(node) },
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

// ── 自定义比例预设：全局库 user/sfnodes/crop_expand_presets.json（跨工作流）──
//
// 真源为全局库（经 /api/sfnodes/crop_expand_presets 读写）；点击已存定义即把
// 该比例套用到本节点（写 properties 状态，随工作流保存）。后端不可用时列表
// 降级为空、手输照常可用。

const RATIO_PRESETS_API = "/api/sfnodes/crop_expand_presets";

async function fetchRatioPresets() {
  try {
    const r = await fetch(sfApiUrl(RATIO_PRESETS_API), { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    return normalizeRatioPresets(j);
  } catch (e) {
    console.warn("[SF Crop Expand] 自定义比例预设库加载失败，手输仍可用:", e);
    return null; // null = 后端不可用
  }
}

async function apiSaveRatioPreset(name, w, h) {
  try {
    const r = await fetch(sfApiUrl(RATIO_PRESETS_API), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, w, h }),
    });
    return r.ok;
  } catch (e) {
    console.warn("[SF Crop Expand]", e);
    return false;
  }
}

async function apiDeleteRatioPreset(name) {
  try {
    const r = await fetch(sfApiUrl(`${RATIO_PRESETS_API}?name=${encodeURIComponent(name)}`), { method: "DELETE" });
    return r.ok;
  } catch (e) {
    console.warn("[SF Crop Expand]", e);
    return false;
  }
}

// ── Custom 比例管理弹窗（左已存定义列表 + 右编辑器；sf_popup 三关闭）──────

const _BTN = "padding:5px 12px;border:none;border-radius:3px;cursor:pointer;font-size:12px;";
const _INPUT = "padding:5px;background:#1a1a1a;border:1px solid #555;border-radius:3px;color:#ddd;font-size:13px;box-sizing:border-box;";

function markRatioFields(wInput, hInput, bad) {
  const color = bad ? "#e74c3c" : "#555";
  wInput.style.borderColor = color;
  hInput.style.borderColor = color;
}

function openCustomRatioDialog(node) {
  if (document.getElementById("sf-crop-expand-ratio-overlay")) return;
  const st0 = getState(node);

  const overlay = document.createElement("div");
  overlay.id = "sf-crop-expand-ratio-overlay";
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:9999;";

  const dialog = document.createElement("div");
  dialog.style.cssText = "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);" +
    "background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:12px 14px;" +
    "box-shadow:0 4px 20px rgba(0,0,0,0.5);width:460px;color:#ddd;font-size:13px;box-sizing:border-box;";
  dialog.innerHTML = `
    <div style="font-weight:bold;margin-bottom:10px;">Custom Aspect Ratio</div>
    <div style="display:flex;gap:12px;align-items:stretch;">
      <div style="width:180px;display:flex;flex-direction:column;min-width:0;">
        <label style="color:#aaa;font-size:10px;margin-bottom:3px;">Saved Definitions</label>
        <div id="sf-ce-ratio-list" style="flex:1;min-height:150px;max-height:240px;overflow-y:auto;background:#1a1a1a;border:1px solid #555;border-radius:3px;padding:3px;"></div>
      </div>
      <div style="flex:1;display:flex;flex-direction:column;gap:8px;min-width:0;">
        <div>
          <label style="color:#aaa;font-size:10px;display:block;margin-bottom:3px;">Name</label>
          <input type="text" id="sf-ce-ratio-name" placeholder="definition name"
            style="width:100%;${_INPUT}">
        </div>
        <div style="display:flex;gap:10px;align-items:flex-end;">
          <div>
            <label style="color:#aaa;font-size:10px;display:block;margin-bottom:3px;">Width</label>
            <input type="number" id="sf-ce-ratio-w" value="${st0.custom_w ?? 1}" min="0.1" step="0.1"
              style="width:100px;${_INPUT}">
          </div>
          <div style="color:#888;font-size:16px;padding-bottom:6px;">:</div>
          <div>
            <label style="color:#aaa;font-size:10px;display:block;margin-bottom:3px;">Height</label>
            <input type="number" id="sf-ce-ratio-h" value="${st0.custom_h ?? 1}" min="0.1" step="0.1"
              style="width:100px;${_INPUT}">
          </div>
        </div>
        <div style="display:flex;gap:8px;">
          <button id="sf-ce-ratio-save" style="${_BTN}background:#3a5f8a;color:#fff;">Save</button>
          <button id="sf-ce-ratio-del" style="${_BTN}background:#5a2f2f;color:#f0b0b0;">Delete</button>
        </div>
      </div>
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:12px;">
      <button id="sf-ce-ratio-cancel" style="${_BTN}background:#444;color:#ddd;">Cancel</button>
      <button id="sf-ce-ratio-ok" style="${_BTN}background:#4a90e2;color:white;">Apply</button>
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

  const listEl = dialog.querySelector("#sf-ce-ratio-list");
  const nameInput = dialog.querySelector("#sf-ce-ratio-name");
  const wInput = dialog.querySelector("#sf-ce-ratio-w");
  const hInput = dialog.querySelector("#sf-ce-ratio-h");
  const saveBtn = dialog.querySelector("#sf-ce-ratio-save");
  const delBtn = dialog.querySelector("#sf-ce-ratio-del");

  let presets = [];
  let selectedIndex = -1;
  let routeOk = true;

  const parseFields = () => {
    const w = parseFloat(wInput.value);
    const h = parseFloat(hInput.value);
    const ok = !isNaN(w) && !isNaN(h) && w > 0 && h > 0;
    markRatioFields(wInput, hInput, !ok);
    return ok ? { w, h } : null;
  };

  // 套用当前字段比例（Apply / 双击列表项共用）
  const applyRatio = (w, h) => {
    const st = setState(node, { custom_w: w, custom_h: h, aspect_ratio: "custom" });
    const rect = applyRatioToRect(
      { x: st.crop_x, y: st.crop_y, w: st.crop_w, h: st.crop_h }, w / h);
    setState(node, { crop_x: rect.x, crop_y: rect.y, crop_w: rect.w, crop_h: rect.h });
    stateChanged(node);
    close();
  };

  const apply = () => {
    const v = parseFields();
    if (!v) return;
    applyRatio(v.w, v.h);
  };

  const renderList = () => {
    listEl.replaceChildren();
    if (presets.length === 0) {
      const empty = document.createElement("div");
      empty.style.cssText = "color:#888;font-size:11px;padding:8px 4px;text-align:center;";
      empty.textContent = routeOk ? "No saved definitions" : "Preset store unavailable";
      listEl.appendChild(empty);
      return;
    }
    presets.forEach((p, idx) => {
      const fmt = (n) => (Number.isInteger(n) ? String(n) : String(Math.round(n * 100) / 100));
      const item = document.createElement("div");
      item.style.cssText = "padding:4px 6px;border-radius:3px;cursor:pointer;margin-bottom:2px;" +
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px;" +
        (idx === selectedIndex ? "background:#3a5f8a;" : "");
      item.textContent = `${p.name}  (${fmt(p.w)}:${fmt(p.h)})`;
      item.title = `${p.name} ${fmt(p.w)}:${fmt(p.h)}`;
      item.onmouseenter = () => { if (idx !== selectedIndex) item.style.background = "#3a3a3a"; };
      item.onmouseleave = () => { item.style.background = idx === selectedIndex ? "#3a5f8a" : ""; };
      // 单击选中并回填字段；双击直接套用
      item.onclick = () => {
        selectedIndex = idx;
        nameInput.value = p.name;
        wInput.value = p.w;
        hInput.value = p.h;
        markRatioFields(wInput, hInput, false);
        renderList();
      };
      item.ondblclick = () => applyRatio(p.w, p.h);
      listEl.appendChild(item);
    });
  };

  const reload = async (selectName) => {
    const list = await fetchRatioPresets();
    if (list === null) {
      routeOk = false;
      renderList();
      return;
    }
    routeOk = true;
    presets = list;
    selectedIndex = selectName ? presets.findIndex((p) => p.name === selectName) : -1;
    renderList();
  };

  const save = async () => {
    const name = nameInput.value.trim();
    if (!name) {
      nameInput.style.borderColor = "#e74c3c";
      sfToast({ summary: "SF Crop Expand", detail: "请填写定义名称", severity: "warn", fallbackTag: "SF Crop Expand" });
      return;
    }
    nameInput.style.borderColor = "#555";
    const v = parseFields();
    if (!v) {
      sfToast({ summary: "SF Crop Expand", detail: "宽度/高度必须为正数", severity: "warn", fallbackTag: "SF Crop Expand" });
      return;
    }
    if (!(await apiSaveRatioPreset(name, v.w, v.h))) {
      sfToast({ summary: "SF Crop Expand", detail: "保存失败（后端路由不可用？重启 ComfyUI 后重试）", severity: "error", fallbackTag: "SF Crop Expand" });
      return;
    }
    sfToast({ summary: "SF Crop Expand", detail: `已保存定义「${name}」`, fallbackTag: "SF Crop Expand" });
    await reload(name);
  };

  const remove = async () => {
    if (selectedIndex < 0 || selectedIndex >= presets.length) {
      sfToast({ summary: "SF Crop Expand", detail: "请先在左侧选择要删除的定义", severity: "warn", fallbackTag: "SF Crop Expand" });
      return;
    }
    const name = presets[selectedIndex].name;
    if (!confirm(`删除定义「${name}」？`)) return;
    if (!(await apiDeleteRatioPreset(name))) {
      sfToast({ summary: "SF Crop Expand", detail: "删除失败（后端路由不可用？）", severity: "error", fallbackTag: "SF Crop Expand" });
      return;
    }
    sfToast({ summary: "SF Crop Expand", detail: `已删除定义「${name}」`, fallbackTag: "SF Crop Expand" });
    nameInput.value = "";
    await reload();
  };

  dialog.querySelector("#sf-ce-ratio-ok").onclick = apply;
  dialog.querySelector("#sf-ce-ratio-cancel").onclick = close;
  saveBtn.onclick = save;
  delBtn.onclick = remove;

  // Enter 应用 / Esc 关闭；放行 ctrl/meta/alt 组合键（Ctrl+S 不能漏成浏览器行为）
  for (const el of [wInput, hInput, nameInput]) {
    el.onkeydown = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") apply();
      else if (e.key === "Escape") close();
    };
    el.oninput = () => {
      wInput.style.borderColor = "#555";
      hInput.style.borderColor = "#555";
      nameInput.style.borderColor = "#555";
    };
  }
  setTimeout(() => (presets.length ? wInput : nameInput).focus(), 100);

  reload();
}

// ── 节点尺寸自适应 ────────────────────────────────────────────────────────

// 钳制节点尺寸不低于最小值（创建/恢复兜底；拖拽路径由 computeSize 包装钳住）
function clampNodeSize(node) {
  node.size = ensureMinSize(node.size?.[0], node.size?.[1]);
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
    const [bx, by, bw, bh] = buttonRect(b, node);
    if (b.isRatio && b.ratioKey === st.aspect_ratio) {
      ctx.fillStyle = accent;
    } else if (b.isColor) {
      ctx.fillStyle = st.fill_color || "#000000";
    } else {
      ctx.fillStyle = "rgba(60,60,60,0.7)";
    }
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = "rgba(150,150,150,0.6)";
    ctx.lineWidth = 1;
    ctx.strokeRect(bx, by, bw, bh);

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
    ctx.fillText(text, bx + bw / 2, by + bh / 2);
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
  const { shiftLeft, shiftRight } = LAYOUT;
  const BTN_H = 21; // 底行按钮高（与 buildButtons 底行一致）

  node.onDrawForeground = (ctx) => {
    if (node.flags.collapsed) return false;

    const nodeW = node.size[0];
    const nodeH = node.size[1];
    const dragging = !!node._sfExpandDrag;
    const st = getState(node);

    const m = computeDisplayMetrics(
      { cropX: st.crop_x, cropY: st.crop_y, cropW: st.crop_w, cropH: st.crop_h, srcW: st.src_w, srcH: st.src_h },
      nodeW, nodeH,
      dragging ? node._sfExpandDrag.frozen : null,
    );

    // 比例竖列底条（节点顶到画布区底缘，与图片区同高）
    const colTop = shiftLeft - 4;
    const colBottom = nodeH - shiftLeft - LAYOUT.bottomH;
    ctx.fillStyle = "rgba(40,40,40,0.9)";
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, colTop, LAYOUT.ratioColW + 2, colBottom - colTop, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(100,100,100,0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(shiftLeft - 4, colTop, LAYOUT.ratioColW + 2, colBottom - colTop);

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

    // 底行背景条（Load/Browse 按钮与信息文本同排）
    const bottomY = nodeH - shiftLeft - BTN_H;
    ctx.fillStyle = "rgba(40,40,40,0.9)";
    ctx.beginPath();
    ctx.roundRect(shiftLeft - 4, bottomY - 4, nodeW - shiftRight - (shiftLeft - 4) - 2, BTN_H + 8, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(100,100,100,0.5)";
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

// ── 右下角 resize cursor 视觉修正 ─────────────────────────────────────────
//
// 命中区维持原生 15×15（不做扩大）；仅修视觉。原生链路 dir→帧尾
// updateCursorStyle→css 不可靠：pointer.resizeDirection 会被两处反复清空
// （诊断栈实锤）——① LGraphCanvas.updateMouseOverNodes（hover 判定切换时无
// 条件清）；② 第三方扩展（如 Comfyui_LG_Tools/queue_shortcut.js）重放
// processMouseMove，每次物理移动跑 ≥2 遍，第二遍清掉刚设的 dir。
// 此 listener 在 LiteGraph 与第三方扩展处理完之后（后注册）运行：
// - 原生 15×15 区内：直接写 style.cursor（不经 dir 间接层，绕过清空/时序），
//   并补写 dir=SE 作双保险；
// - 区外仅在 cursor 是我们写入时恢复 ""（交还原生/页面管理），不覆盖其它
//   cursor 状态（槽 grab 等）。

if (!app._sfCropExpandCursorPatch) {
  app._sfCropExpandCursorPatch = true;
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
      this._sfExpandPaste = (dataURL) => loadAndStoreImage(this, dataURL);
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
