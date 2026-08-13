// SFRegionalLoRA — in-node multi-region editor.
// Draw N boxes on the canvas, assign one LoRA per box; each LoRA only takes
// effect inside its own box (activation-delta injection, see Python node).
//
// State model: the hidden SFRegionsJson STRING widget (Python "hidden" input)
// is the single source of truth — region rows and the canvas read/write it
// via lib.readRegions/writeRegions, so values survive workflow save/load/
// copy through standard widget collection (no graphToPrompt injection
// needed). Transient row widgets never serialize.
//
// Interaction (box move/8-way resize/draw-to-create) follows the
// ZIT-Ideogram pattern verified by RegioCraft: mousedown stopPropagation +
// document-level move/up listeners.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { isGraphLoading } from "./sf_common.js";
import * as lib from "./sf_regional_lora_lib.js";

// 版本标记：诊断"浏览器是否加载了新 JS"（硬刷新后 window.__sfRegionalLoRAVersion
// 应为 3；undefined/旧值 = 缓存）。每次改 web/ 时递增。
const EXT_VERSION = 3;
window.__sfRegionalLoRAVersion = EXT_VERSION;

const NODE_TYPE = "SFRegionalLoRA";
const HANDLE = 12;
const MINSIZE = 0.04;
const CLEARBTN = 18;

function hueColor(i, n, alpha = 1) {
  const hue = (i / Math.max(1, n)) * 300; // stop before wrapping back to red
  return `hsla(${hue}, 70%, 60%, ${alpha})`;
}

function markTransient(w) {
  w.__rc_row = true;
  w.serialize = false;
  if (!w.options) w.options = {};
  w.options.serialize = false;
  return w;
}

// ---------------------------------------------------------------------------
// LoRA list (single shared array instance — see lib.ensureLoraList)
// ---------------------------------------------------------------------------
async function loraListLoader() {
  const resp = await api.fetchApi("/object_info/LoraLoader");
  const info = await resp.json();
  const names = info?.LoraLoader?.input?.required?.lora_name?.[0];
  return Array.isArray(names) ? names.filter((n) => n !== "None") : [];
}

// ---------------------------------------------------------------------------
// background image (Load latest output / drag-drop) for face alignment
// ---------------------------------------------------------------------------
async function fetchLatestOutputImageURL() {
  try {
    const res = await fetch("/history?max_items=30");
    if (!res.ok) return null;
    const hist = await res.json();
    const entries = Object.values(hist);
    for (let i = entries.length - 1; i >= 0; i--) {
      const outputs = (entries[i] && entries[i].outputs) || {};
      for (const nodeId of Object.keys(outputs)) {
        const imgs = outputs[nodeId].images;
        if (imgs && imgs.length) {
          const img = imgs[imgs.length - 1];
          const params = new URLSearchParams({
            filename: img.filename, subfolder: img.subfolder || "", type: img.type || "output",
          });
          return "/view?" + params.toString();
        }
      }
    }
  } catch (e) {
    console.error("[SFRegionalLoRA] failed to fetch latest output:", e);
  }
  return null;
}

function imageOutputURLFromExecutedEvent(detail) {
  const imgs = detail && detail.output && detail.output.images;
  if (!imgs || !imgs.length) return null;
  const img = imgs[imgs.length - 1];
  const params = new URLSearchParams({
    filename: img.filename, subfolder: img.subfolder || "", type: img.type || "output",
  });
  return "/view?" + params.toString();
}

// ---------------------------------------------------------------------------
// box-drawing canvas
// ---------------------------------------------------------------------------
function buildCanvasWidget(node) {
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.display = "block";
  canvas.style.marginTop = "4px";
  canvas.style.borderRadius = "6px";
  canvas.style.touchAction = "none";
  canvas.style.cursor = "crosshair";

  let bgImage = null; // visual reference only, never serialized

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth || 260;
    const chh = canvas.clientHeight || 220;
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(chh * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, chh);
    ctx.fillStyle = "#15151a";
    ctx.fillRect(0, 0, cw, chh);

    if (bgImage) {
      // "contain" fit: show the WHOLE image, letterboxed — never crop, since
      // faces can sit anywhere in the frame.
      const ir = bgImage.width / bgImage.height, cr = cw / chh;
      let dw, dh, dx, dy;
      if (ir > cr) { dw = cw; dh = dw / ir; dx = 0; dy = (chh - dh) / 2; }
      else { dh = chh; dw = dh * ir; dx = (cw - dw) / 2; dy = 0; }
      ctx.drawImage(bgImage, dx, dy, dw, dh);
      ctx.fillStyle = "rgba(0,0,0,0.2)";
      ctx.fillRect(dx, dy, dw, dh);
    }

    ctx.strokeStyle = "#3a3a42";
    ctx.strokeRect(0.5, 0.5, cw - 1, chh - 1);

    const regions = lib.readRegions(node);
    regions.forEach((reg, i) => {
      const col = hueColor(i, regions.length);
      const x = (reg.x ?? 0) * cw, y = (reg.y ?? 0) * chh;
      const w = (reg.w ?? 0.3) * cw, h = (reg.h ?? 0.3) * chh;
      ctx.globalAlpha = reg.enable !== false ? 1 : 0.35;
      ctx.fillStyle = hueColor(i, regions.length, bgImage ? 0.08 : 0.15);
      ctx.fillRect(x, y, w, h);
      ctx.lineWidth = 2;
      ctx.strokeStyle = col;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = col;
      ctx.fillRect(x + w - HANDLE, y + h - HANDLE, HANDLE, HANDLE);
      ctx.font = "11px sans-serif";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#111";
      ctx.fillText(`${i + 1} ${lib.shortName(reg.lora)}`, x + 5, y + 4);
      ctx.globalAlpha = 1;
    });

    if (!regions.length) {
      ctx.fillStyle = "#888";
      ctx.font = "11px sans-serif";
      ctx.fillText("click '+ Add Region' below or drag on the canvas to draw boxes", 6, chh - 6);
    }
    if (bgImage) {
      ctx.fillStyle = "#000a";
      ctx.fillRect(cw - CLEARBTN - 6, 6, CLEARBTN, CLEARBTN);
      ctx.strokeStyle = "#aaa"; ctx.lineWidth = 1.5;
      const cx0 = cw - CLEARBTN - 6 + 5, cy0 = 6 + 5, cx1 = cw - 6 - 5, cy1 = 6 + CLEARBTN - 5;
      ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx1, cy1);
      ctx.moveTo(cx1, cy0); ctx.lineTo(cx0, cy1); ctx.stroke();
    } else {
      ctx.fillStyle = "#666";
      ctx.font = "10px sans-serif";
      ctx.fillText("drop an image here, or load the latest output, to line up faces", 6, chh - 6);
    }
  }

  function onImageLoaded(img) {
    bgImage = img;
    draw();
    node.setDirtyCanvas(true, true);
  }

  const container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.appendChild(canvas);

  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:4px;";

  const loadBtn = document.createElement("button");
  loadBtn.textContent = "↻ Load latest output";
  loadBtn.style.cssText = "font-size:11px;padding:3px 8px;cursor:pointer;background:#2a2a32;"
    + "color:#ddd;border:1px solid #444;border-radius:4px;";
  loadBtn.onclick = async (e) => {
    e.stopPropagation();
    loadBtn.textContent = "…";
    const url = await fetchLatestOutputImageURL();
    loadBtn.textContent = "↻ Load latest output";
    if (!url) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => onImageLoaded(img);
    img.src = url;
  };

  const autoLabel = document.createElement("label");
  autoLabel.style.cssText = "font-size:11px;color:#999;display:flex;align-items:center;gap:4px;cursor:pointer;";
  const autoCheckbox = document.createElement("input");
  autoCheckbox.type = "checkbox";
  autoLabel.appendChild(autoCheckbox);
  autoLabel.appendChild(document.createTextNode("auto after each run"));

  toolbar.appendChild(loadBtn);
  toolbar.appendChild(autoLabel);
  container.appendChild(toolbar);

  let executedHandler = null;
  autoCheckbox.addEventListener("change", () => {
    if (autoCheckbox.checked) {
      executedHandler = (e) => {
        const url = imageOutputURLFromExecutedEvent(e.detail);
        if (!url) return;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => onImageLoaded(img);
        img.src = url;
      };
      api.addEventListener("executed", executedHandler);
    } else if (executedHandler) {
      api.removeEventListener("executed", executedHandler);
      executedHandler = null;
    }
  });
  const oldRemoved = node.onRemoved;
  node.onRemoved = function () {
    if (executedHandler) api.removeEventListener("executed", executedHandler);
    oldRemoved && oldRemoved.apply(this, arguments);
  };

  const widget = node.addDOMWidget("region_canvas", "rc_canvas", container, {
    getValue() { return ""; },
    setValue() {},
    getMinHeight() {
      const w = node.size ? node.size[0] - 20 : 220;
      return Math.round(Math.max(140, Math.min(w * 1.1, 380))) + 26;
    },
    hideOnZoom: false,
  });
  widget.serialize = false;
  widget.__rc_canvas = true;

  // -- interaction: move / resize boxes, click-to-clear-bg, drag-drop image --
  let drag = null;
  const toNorm = (e) => {
    const r = canvas.getBoundingClientRect();
    return [lib.clamp01((e.clientX - r.left) / r.width), lib.clamp01((e.clientY - r.top) / r.height)];
  };
  const onDown = (e) => {
    if (drag) onUp(e); // stale drag from a lost up-event

    const r = canvas.getBoundingClientRect();
    const [nx, ny] = toNorm(e);

    if (bgImage) {
      const px = nx * r.width, py = ny * r.height;
      const bx0 = r.width - CLEARBTN - 6, by0 = 6;
      if (px >= bx0 && px <= bx0 + CLEARBTN && py >= by0 && py <= by0 + CLEARBTN) {
        bgImage = null; draw();
        e.preventDefault(); e.stopPropagation();
        return;
      }
    }

    const regions = lib.readRegions(node);
    const hit = lib.hitTestRegions(regions, nx, ny, r.width, r.height, HANDLE);
    if (hit && hit.mode === "move") {
      const reg = regions[hit.i];
      drag = { i: hit.i, mode: "move", ox: nx - (reg.x ?? 0), oy: ny - (reg.y ?? 0) };
    } else if (hit) {
      const reg = regions[hit.i];
      drag = {
        i: hit.i, mode: hit.mode, startNorm: { nx, ny },
        start: { x: reg.x ?? 0, y: reg.y ?? 0, w: reg.w ?? 0.3, h: reg.h ?? 0.3 },
      };
    } else {
      // Empty canvas: start drawing a brand-new region here; dropped in onUp
      // if it never grows past an accidental-click size.
      const nb = lib.defaultRegion(regions.length, regions.length + 1);
      nb.x = nx; nb.y = ny; nb.w = 0; nb.h = 0;
      regions.push(nb);
      lib.writeRegions(node, regions);
      drag = {
        i: regions.length - 1, mode: "resize-br", isNew: true,
        startNorm: { nx, ny }, start: { x: nx, y: ny, w: 0, h: 0 },
      };
    }
    if (drag) {
      // stopPropagation (not just preventDefault): without it the mousedown
      // bubbles into LiteGraph's canvas which competes for the gesture
      // (the "magnet" bug). Plain mouse events + document listeners.
      e.preventDefault();
      e.stopPropagation();
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    }
  };
  const onMove = (e) => {
    if (!drag) return;
    const [nx, ny] = toNorm(e);
    const regions = lib.readRegions(node);
    const reg = regions[drag.i];
    if (!reg) return;
    if (drag.mode === "move") {
      const rw = reg.w ?? 0.3, rh = reg.h ?? 0.3;
      reg.x = lib.clamp01(nx - drag.ox);
      reg.y = lib.clamp01(ny - drag.oy);
      if (reg.x + rw > 1) reg.x = 1 - rw;
      if (reg.y + rh > 1) reg.y = 1 - rh;
    } else {
      const dx = nx - drag.startNorm.nx, dy = ny - drag.startNorm.ny;
      const nb = lib.applyResize(drag.mode, drag.start, dx, dy);
      reg.x = nb.x; reg.y = nb.y; reg.w = nb.w; reg.h = nb.h;
    }
    lib.writeRegions(node, regions);
    draw();
  };
  const onUp = (e) => {
    if (drag) {
      const regions = lib.readRegions(node);
      const reg = regions[drag.i];
      if (drag.isNew) {
        if (!reg || reg.w < MINSIZE || reg.h < MINSIZE) {
          if (reg) regions.splice(drag.i, 1);
          lib.writeRegions(node, regions);
        } else {
          // Real new region: build its control row only now (rebuilding on
          // every mousemove would churn the widgets dozens of times a second).
          rebuildRows(node);
        }
      }
    }
    drag = null;
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    draw();
  };
  canvas.addEventListener("mousedown", onDown);
  canvas.addEventListener("mousemove", (e) => {
    if (drag) return;
    const r = canvas.getBoundingClientRect();
    const [nx, ny] = toNorm(e);
    const hit = lib.hitTestRegions(lib.readRegions(node), nx, ny, r.width, r.height, HANDLE);
    canvas.style.cursor = hit ? (hit.mode === "move" ? "move" : "nwse-resize") : "crosshair";
  });

  canvas.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  canvas.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => onImageLoaded(img);
        img.src = reader.result;
      };
      reader.readAsDataURL(file);
      return;
    }
    const url = e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain");
    if (url) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => onImageLoaded(img);
      img.src = url;
    }
  });

  try { new ResizeObserver(() => draw()).observe(canvas); } catch (e) {}
  setTimeout(draw, 50);
  node.__rc_draw = draw;

  const oldResize = node.onResize;
  node.onResize = function () { oldResize && oldResize.apply(this, arguments); draw(); };

  return widget;
}

// ---------------------------------------------------------------------------
// per-region control rows (enable / lora / strength / remove)
// ---------------------------------------------------------------------------
function rebuildRows(node) {
  if (node.widgets) {
    node.widgets = node.widgets.filter((w) => !w.__rc_row);
  }
  const regions = lib.readRegions(node);
  regions.forEach((region, idx) => {
    const enableW = node.addWidget(
      "toggle", `region ${idx + 1} enabled`, region.enable !== false,
      (v) => { const r = lib.readRegions(node); if (r[idx]) { r[idx].enable = v; lib.writeRegions(node, r); } node.setDirtyCanvas(true, true); },
      { on: "on", off: "off" }
    );
    markTransient(enableW);
    lib.bindRegionValue(node, enableW, idx, "enable", region.enable !== false);

    const loraW = node.addWidget(
      "combo", `region ${idx + 1} lora`, region.lora || "None",
      (v) => { const r = lib.readRegions(node); if (r[idx]) { r[idx].lora = v; lib.writeRegions(node, r); } node.setDirtyCanvas(true, true); },
      { values: lib.getLoraList() }
    );
    markTransient(loraW);
    lib.bindRegionValue(node, loraW, idx, "lora", region.lora || "None");

    const strW = node.addWidget(
      "number", `region ${idx + 1} strength`,
      typeof region.strength === "number" ? region.strength : 1.0,
      (v) => { const r = lib.readRegions(node); if (r[idx]) { r[idx].strength = v; lib.writeRegions(node, r); } },
      { min: -10.0, max: 10.0, step: 0.1, precision: 2 }
    );
    markTransient(strW);
    lib.bindRegionValue(node, strW, idx, "strength",
      typeof region.strength === "number" ? region.strength : 1.0);

    const rmW = node.addWidget("button", `✕ remove region ${idx + 1}`, null, () => {
      const r = lib.readRegions(node);
      r.splice(idx, 1);
      lib.writeRegions(node, r);
      rebuildRows(node);
      node.setDirtyCanvas(true, true);
    });
    markTransient(rmW);
  });

  const sz = node.computeSize();
  node.size[1] = Math.max(node.size[1], sz[1]);
  node.setDirtyCanvas(true, true);
}

// 工作流加载值恢复兜底：轮询等 isGraphLoading() 结束（loadGraphData 完成 +
// 300ms 尾窗，此时 SFRegionsJson 的保存值必已恢复），再幂等重建行控件。
// 加载结束即自动停止；非加载路径（手动加节点/复制粘贴）首次即停。值守卫
// （safeRebuildRows）保证多时机触发零重复。onRemoved 清理。
function startRestorePoll(node) {
  node.__rc_stopPoll?.();
  const timer = setInterval(() => {
    if (isGraphLoading()) return; // 加载中/尾窗：值尚未恢复，继续等
    if (lib.safeRebuildRows(node, rebuildRows)) {
      node.__rc_draw && node.__rc_draw();
    }
    clearInterval(timer);
    node.__rc_stopPoll = null;
  }, 250);
  node.__rc_stopPoll = () => clearInterval(timer);
}

// ---------------------------------------------------------------------------
// per-region control rows (enable / lora / strength / remove)
// ---------------------------------------------------------------------------
// extension registration
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "SFRegionalLoRA.editor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    await lib.ensureLoraList(loraListLoader);

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const node = this;

      // Hidden state widget (data carrier, saved/loaded/copied with the
      // workflow). Python declares SFRegionsJson in "hidden" so the value
      // travels via standard widget collection — no prompt injection needed.
      // Seed ONLY when truly absent: transient empty states during multi-tab
      // canvas restore must not wipe real region data (RegioCraft lesson).
      const jw = node.widgets?.find((w) => w.name === lib.REGIONS_WIDGET);
      if (!jw) {
        const sfw = node.addWidget("STRING", lib.REGIONS_WIDGET, lib.defaultRegionsJson(2), () => {});
        sfw.hidden = true;
        sfw.computeSize = () => [0, -4];
        if (!sfw.options) sfw.options = {};
        sfw.options.canvasOnly = true;
        // NOTE: never markTransient here — the widget must keep default
        // serialize (workflow persistence) and must NOT carry __rc_row
        // (rebuildRows would drop it).
      } else if (jw.value === undefined || jw.value === null || jw.value === "") {
        jw.value = lib.defaultRegionsJson(2);
      }

      buildCanvasWidget(node);
      lib.safeRebuildRows(node, rebuildRows); // 首次必重建 + 记录值标记
      startRestorePoll(node); // 值恢复兜底（isGraphLoading 结束即重建）

      // Not transient: must survive rebuildRows (which filters __rc_row).
      node.addWidget("button", "+ Add Region", null, () => {
        const regions = lib.readRegions(node);
        regions.push(lib.defaultRegion(regions.length, regions.length + 1));
        lib.writeRegions(node, regions);
        rebuildRows(node);
        node.__rc_draw && node.__rc_draw();
      });

      // 清理恢复轮询（链式包装，buildCanvasWidget 已有一次 onRemoved 包装）
      const oldRemoved = node.onRemoved;
      node.onRemoved = function () {
        node.__rc_stopPoll?.();
        oldRemoved && oldRemoved.apply(this, arguments);
      };

      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (o) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      // 值恢复时序保险：SFRegionsJson 的保存值可能在 onConfigure 之后才恢复
      // （Vue 前端），多时机幂等重建（值守卫，未变即跳过）。同 crop 的
      // microtask + 250ms 双保险模式，再加一层更晚的兜底。
      const syncRows = () => {
        if (lib.safeRebuildRows(this, rebuildRows)) {
          this.__rc_draw && this.__rc_draw();
        }
      };
      queueMicrotask(syncRows);
      setTimeout(syncRows, 0);
      setTimeout(syncRows, 300);
      setTimeout(syncRows, 1500);
      return r;
    };

    // 整图加载完成（值必已恢复）后的最终兜底同步。ComfyUI 标准钩子，
    // any_pack/regex_extract 同款包装模式。
    const onAfterGraphConfigured = nodeType.prototype.onAfterGraphConfigured;
    nodeType.prototype.onAfterGraphConfigured = function (o) {
      const r = onAfterGraphConfigured ? onAfterGraphConfigured.apply(this, arguments) : undefined;
      setTimeout(() => {
        if (lib.safeRebuildRows(this, rebuildRows)) {
          this.__rc_draw && this.__rc_draw();
        }
      }, 0);
      return r;
    };
  },
});
