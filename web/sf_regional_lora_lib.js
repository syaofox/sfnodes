// SFRegionalLoRA — pure helpers (no app/DOM), Node-testable.
// The hidden SFRegionsJson STRING widget is the single source of truth for
// region data (lora/strength/enable/box), saved/loaded/copied with the
// workflow via standard widget collection (Python declares it in "hidden").

export const REGIONS_WIDGET = "SFRegionsJson";

export function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

export function defaultRegion(i, n) {
  const cols = Math.max(1, n);
  return {
    lora: "None",
    strength: 1.0,
    enable: true,
    x: i / cols, y: 0.0, w: 1.0 / cols, h: 1.0,
  };
}

export function defaultRegions(n = 2) {
  const arr = [];
  for (let i = 0; i < n; i++) arr.push(defaultRegion(i, n));
  return arr;
}

export function defaultRegionsJson(n = 2) {
  return JSON.stringify(defaultRegions(n), null, 2);
}

export function readRegions(node) {
  const w = node.widgets?.find((x) => x.name === REGIONS_WIDGET);
  if (!w) return [];
  // 值字符串不变则复用上次解析结果（每帧 getter 调用零成本）
  if (w.__rc_parsed && w.__rc_parsed[0] === w.value) return w.__rc_parsed[1];
  try {
    const parsed = JSON.parse(w.value || "[]");
    const arr = Array.isArray(parsed) ? parsed : [];
    w.__rc_parsed = [w.value, arr];
    return arr;
  } catch (e) {
    return [];
  }
}

export function writeRegions(node, regions) {
  const w = node.widgets?.find((x) => x.name === REGIONS_WIDGET);
  if (!w) return;
  w.value = JSON.stringify(regions, null, 2);
  if (w.inputEl) w.inputEl.value = w.value;
}

// Live-bind a row widget's `value` to the regions JSON: the displayed value
// ALWAYS mirrors readRegions(node)[idx][key]. Workflow-load value restore is
// asynchronous and runs AFTER row widgets are created; a stale assignment to
// widget.value could otherwise stick forever (observed: combo showed "None"
// while the JSON and the run were correct, and even a rebuild that read the
// restored JSON was followed by another external write-back of "None").
// The setter only stages the intermediate value for LiteGraph's internal
// flow — the real JSON write happens in the widget's callback.
export function bindRegionValue(node, widget, idx, key, initial) {
  let _staged = initial;
  Object.defineProperty(widget, "value", {
    get() {
      const r = readRegions(node);
      const cur = r[idx] ? r[idx][key] : undefined;
      if (cur === undefined || cur === null) {
        if (key === "enable") return !!_staged;
        if (key === "strength") return typeof _staged === "number" ? _staged : Number(_staged || 0);
        return String(_staged);
      }
      if (key === "enable") return !!cur;
      if (key === "strength") return typeof cur === "number" ? cur : Number(cur);
      return String(cur);
    },
    set(v) { _staged = v; },
    configurable: true,
  });
  return widget;
}

// Idempotent row-rebuild guard for workflow-load value-restore timing.
// On graph load the SFRegionsJson widget's saved value is restored AFTER
// onNodeCreated and possibly after onConfigure's microtask/short timers —
// row widgets created before the restore read the default JSON and show
// "None" forever (observed bug: LoRA dropdown displays None while the data
// and the run are correct). Call this from several post-load timings
// (queueMicrotask / setTimeout 0 / 300ms / onAfterGraphConfigured); it only
// rebuilds when the widget value actually changed since the last rebuild,
// so repeated calls are free and no flicker occurs.
export function safeRebuildRows(node, rebuild) {
  const w = node.widgets?.find((x) => x.name === REGIONS_WIDGET);
  if (!w) return false; // 异常状态（缺真源）：不重建、不动标记
  const json = w.value;
  if (node.__rc_lastJson === json) return false;
  node.__rc_lastJson = json;
  if (typeof rebuild === "function") rebuild(node);
  return true;
}

// Collapse a possibly-inverted drag into a normal positive-size box, clamped
// into the canvas. Same idea as KJ's normalizeBox in ZIT-Ideogram.
export function normalizeRect(b) {
  let { x, y, w, h } = b;
  if (w < 0) { x += w; w = -w; }
  if (h < 0) { y += h; h = -h; }
  x = clamp01(x); y = clamp01(y);
  w = Math.min(w, 1 - x); h = Math.min(h, 1 - y);
  return { x, y, w: Math.max(0, w), h: Math.max(0, h) };
}

// Apply a resize-handle drag (any of the 8 directions) to a starting rect.
export function applyResize(mode, start, dx, dy) {
  let { x, y, w, h } = start;
  switch (mode) {
    case "resize-br": w += dx; h += dy; break;
    case "resize-tl": x += dx; y += dy; w -= dx; h -= dy; break;
    case "resize-tr": y += dy; w += dx; h -= dy; break;
    case "resize-bl": x += dx; w -= dx; h += dy; break;
    case "resize-t": y += dy; h -= dy; break;
    case "resize-b": h += dy; break;
    case "resize-l": x += dx; w -= dx; break;
    case "resize-r": w += dx; break;
  }
  return normalizeRect({ x, y, w, h });
}

// Which of a region's 8 handles (if any) sits under a normalized point, or
// "move" if inside the region body. Handles are fixed CSS pixels, so the
// pixel radius is divided by the canvas's on-screen width/height each call.
export function hitTestRegions(regions, nx, ny, wPx, hPx, handlePx = 12) {
  const rxr = handlePx / wPx, ryr = handlePx / hPx;
  for (let i = regions.length - 1; i >= 0; i--) {
    const reg = regions[i];
    const x1 = reg.x ?? 0, y1 = reg.y ?? 0;
    const x2 = x1 + (reg.w ?? 0.3), y2 = y1 + (reg.h ?? 0.3);
    const near = (cx, cy) => Math.abs(nx - cx) < rxr && Math.abs(ny - cy) < ryr;
    if (near(x1, y1)) return { i, mode: "resize-tl" };
    if (near(x2, y1)) return { i, mode: "resize-tr" };
    if (near(x1, y2)) return { i, mode: "resize-bl" };
    if (near(x2, y2)) return { i, mode: "resize-br" };
    if (nx >= x1 && nx <= x2 && Math.abs(ny - y1) < ryr) return { i, mode: "resize-t" };
    if (nx >= x1 && nx <= x2 && Math.abs(ny - y2) < ryr) return { i, mode: "resize-b" };
    if (ny >= y1 && ny <= y2 && Math.abs(nx - x1) < rxr) return { i, mode: "resize-l" };
    if (ny >= y1 && ny <= y2 && Math.abs(nx - x2) < rxr) return { i, mode: "resize-r" };
    if (nx >= x1 && nx <= x2 && ny >= y1 && ny <= y2) return { i, mode: "move" };
  }
  return null;
}

export function shortName(p) {
  if (!p || typeof p !== "string" || p === "None") return "";
  const s = p.split(/[\\/]/).pop().replace(/\.(safetensors|safetensor|ckpt|pt|pth|bin|sft)$/i, "");
  return s.length > 14 ? s.slice(0, 13) + "…" : s;
}

// LoRA list management: ONE array instance, mutated in place, so every combo
// widget's `values` reference always observes fresh entries (no stale-array
// race when the fetch completes after node creation).
const LORA_LIST = ["None"];
let _loraLoaded = false;

export function getLoraList() {
  return LORA_LIST;
}

export async function ensureLoraList(loader) {
  if (_loraLoaded) return LORA_LIST;
  _loraLoaded = true;
  try {
    const names = await loader();
    if (Array.isArray(names) && names.length) {
      const seen = new Set(LORA_LIST);
      for (const n of names) {
        if (!seen.has(n)) {
          seen.add(n);
          LORA_LIST.push(n);
        }
      }
    }
  } catch (e) {
    console.warn("[SFRegionalLoRA] could not fetch lora list:", e);
  }
  return LORA_LIST;
}
