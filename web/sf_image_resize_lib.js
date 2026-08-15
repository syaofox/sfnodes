// SF Image Resize — pure-function lib (no app/DOM dependency, Node-testable).
// JS mirror of Python `sf_utils.resize_engine._apply_wired_size` and the
// wired-size precedence in nodes/image/resize_image.py — keep in lockstep.
// DOM-free helpers (gcd/ratioLabel/aspectRectDims/roundRectPath) are shared
// by the legacy slot-dead-space painter and the Nodes 2.0 cards canvas.

import { previewResize } from "./sf_load_image_resize.js";

// ── state read/write (pattern mirrors sf_load_image.js, prop name parameterised) ──

export function readState(node, stateProp, DEFAULT_STATE) {
  const v = node?.properties?.[stateProp];
  if (typeof v === "string" && v) {
    try { return { ...DEFAULT_STATE, ...JSON.parse(v) }; } catch { /* fall through */ }
  }
  return { ...DEFAULT_STATE };
}

export function writeState(node, stateProp, state) {
  if (!node.properties) node.properties = {};
  node.properties[stateProp] = JSON.stringify(state);
}

// ── wired inputs ─────────────────────────────────────────────────────────────

export function isWired(node, name) {
  const inp = node?.inputs?.find((i) => i.name === name);
  return !!(inp && inp.link != null);
}

// Best-effort read of a wired INT input's value at edit time. Plain INT widget
// sources are only trusted when there's exactly ONE numeric widget
// (unambiguous); multi-widget nodes (seed/steps/cfg…) and combo/string sources
// return null — callers then fall back to a "set by wires" readout / post-run
// dims rather than a wrong number.
export function readWiredInt(node, name) {
  const inp = node?.inputs?.find((i) => i.name === name);
  if (!inp || inp.link == null) return null;
  let l = node?.graph?.links?.[inp.link];
  if (!l && typeof node?.graph?.links?.get === "function") l = node.graph.links.get(inp.link);
  if (!l) return null;
  const up = node.graph.getNodeById(l.origin_id);
  if (!up) return null;
  const nums = (up.widgets || []).filter((x) => typeof x.value === "number");
  // 镜像 Python _apply_wired_size 的 int() 截断（2.7 -> 2）：Math.round 会
  // 把 2.7 报成 3，而真实执行是 2，预览对输出说谎。
  return nums.length === 1 && Number.isFinite(nums[0].value) ? Math.trunc(nums[0].value) : null;
}

// Central wired-input state: which axes are wired + their best-effort values.
export function wireInfo(node) {
  const wiredW = isWired(node, "width");
  const wiredH = isWired(node, "height");
  const wiredLongest = isWired(node, "longest_side");
  return {
    wiredW, wiredH, wiredLongest,
    count: (wiredW ? 1 : 0) + (wiredH ? 1 : 0),
    valW: wiredW ? readWiredInt(node, "width") : null,
    valH: wiredH ? readWiredInt(node, "height") : null,
    valLongest: wiredLongest ? readWiredInt(node, "longest_side") : null,
  };
}

// JS mirror of Python `_apply_wired_size` (keep in lockstep). Returns the
// effective state to feed previewResize so the OUTPUT card matches Python.
// Single wire = aspect scale to that dim; both = exact box (Fit/Crop);
// longest_side wins over width/height. Unreadable wires return the state
// unchanged — callers fall back to "set by wires" / cached dims.
export function effectiveWiredState(state, info, ow, oh) {
  if (info.wiredLongest) {
    if (info.valLongest == null) return state;
    if (info.valLongest <= 0) return { ...state, mode: "off" }; // 0/neg = no target
    return { ...state, mode: "longest_side", longest_side: info.valLongest };
  }
  if (!info.wiredW && !info.wiredH) return state;
  if (info.wiredW !== info.wiredH) { // exactly one wired
    const v = info.wiredW ? info.valW : info.valH;
    const od = info.wiredW ? ow : oh;
    if (v == null || !od) return state;
    if (v <= 0) return { ...state, mode: "off" };
    return { ...state, mode: "scale_factor", scale_factor: v / od };
  }
  if (info.valW == null || info.valH == null) return state;
  if (state.mode === "fit_inside") return { ...state, fit_w: info.valW, fit_h: info.valH };
  return { ...state, mode: "cover", cover_w: info.valW, cover_h: info.valH };
}

// What the painter should show for the INPUT→OUTPUT cards. Returns a dual-card
// info object, or null when nothing can be shown (caller decides the message):
//   - live = { w, h } from the upstream image preview (naturalWidth/NaturalHeight)
//   - cached = { in_w, in_h, out_w, out_h } from the last run (executed payload)
//   - wi = wireInfo(node) — wired inputs drive the size when readable
export function getReadoutInfo(state, cached, live, wi) {
  if ((wi.count > 0 || wi.wiredLongest) && live) {
    const needL = wi.wiredLongest && wi.valLongest == null;
    const needW = !wi.wiredLongest && wi.wiredW && wi.valW == null;
    const needH = !wi.wiredLongest && wi.wiredH && wi.valH == null;
    if (!needW && !needH && !needL) {
      const eff = effectiveWiredState(state, wi, live.w, live.h);
      const { w, h } = previewResize(live.w, live.h, eff);
      return { mode: "dual", inW: live.w, inH: live.h, outW: w, outH: h };
    }
    // Wired from a source we can't read at edit time: show the real dims if a
    // run happened, else say it's wire-driven (no wrong number).
    if (cached) return { mode: "dual", inW: live.w, inH: live.h, outW: cached.out_w, outH: cached.out_h };
    return { mode: "msg", text: "尺寸由接线输入决定" };
  }
  if (live) {
    const { w, h } = previewResize(live.w, live.h, state);
    return { mode: "dual", inW: live.w, inH: live.h, outW: w, outH: h };
  }
  if (cached) return { mode: "dual", inW: cached.in_w, inH: cached.in_h, outW: cached.out_w, outH: cached.out_h };
  return null;
}

// ── drawing helpers (shared by legacy onDrawForeground + Nodes 2.0 canvas) ──

export function gcd(a, b) {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { const t = b; b = a % b; a = t; }
  return a || 1;
}

export function ratioLabel(w, h) {
  const g = gcd(w, h);
  const rw = w / g, rh = h / g;
  const known = ["1:1","16:9","9:16","2:1","1:2","3:2","2:3","4:3","3:4","4:5","5:4","21:9"];
  const s = `${rw}:${rh}`;
  if (known.includes(s)) return s;
  const r = w / h;
  return r >= 1 ? `~${r.toFixed(2)}:1` : `~1:${(1 / r).toFixed(2)}`;
}

// Rectangle scaled to a w:h aspect inside a max box (the little ratio shape).
export function aspectRectDims(w, h, maxW, maxH) {
  const a = w / h;
  let rw, rh;
  if (a >= maxW / maxH) { rw = maxW; rh = maxW / a; }
  else { rh = maxH; rw = maxH * a; }
  return { rw: Math.max(2, Math.round(rw)), rh: Math.max(2, Math.round(rh)) };
}

// Rounded rectangle (all corners).
export function roundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
