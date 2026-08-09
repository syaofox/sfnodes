// SF Image Resize — DOM UI layer. Builds the node body: mode chips, the
// per-mode panel (reusing sf_load_image_resize.js buildModePanel), the
// snap/resample/upscale row (reusing sf_load_image_ui.js
// renderGlobalControls), wired-input panels, and the INPUT→OUTPUT readout
// painter (shared by the legacy onDrawForeground and the Nodes 2.0 canvas).
// Pure maths lives in sf_image_resize_lib.js; the extension + graphToPrompt
// injection live in sf_image_resize.js.
//
// Ported from comfyui-pixaroma js/image_resize/index.js + ui.mjs; class names
// use the sf-ir-* family for the chrome and the shared sf-li-* family for the
// per-mode panels (their CSS is injected by sf_load_image_resize.js).

import { buildModePanel, previewResize } from "./sf_load_image_resize.js";
import {
  applyInlineLabel, applyWHLayout, applyCoverControls,
  renderGlobalControls, injectCSS as injectLoadImageChromeCSS,
} from "./sf_load_image_ui.js";
import { sfAccent } from "./sf_common.js";
import {
  readState, writeState, wireInfo, effectiveWiredState, getReadoutInfo,
  ratioLabel, aspectRectDims, roundRectPath,
} from "./sf_image_resize_lib.js";

export const STATE_PROP = "sfImageResizeState";
export const HIDDEN_INPUT = "SFImageResizeState";

export const DEFAULT_STATE = {
  version: 1,
  mode: "off",
  max_mp: 1.0,
  longest_side: 1024,
  scale_factor: 1.0,
  fit_w: 1024, fit_h: 1024,
  cover_w: 1024, cover_h: 1024,
  ratio_preset: "1:1",
  ratio_w: 1, ratio_h: 1,
  ratio_action: "crop",
  pad_color: "#808080",
  pad_top: 0, pad_bottom: 0, pad_left: 0, pad_right: 0,
  crop_anchor: "center", crop_scale: true,
  snap: 0,
  resample: "auto",
  allow_upscale: true,
};

// Modes that consume an explicit W x H target. Wired width/height feed these.
const WH_MODES = new Set(["fit_inside", "cover"]);

let _cssInjected = false;
export function injectCSS() {
  if (_cssInjected) return;
  _cssInjected = true;
  const css = `
    .sf-ir-root{width:100%;box-sizing:border-box;padding:2px 8px 8px;background:#2a2a2a;
      border-radius:4px;color:#ddd;font-family:ui-sans-serif,system-ui,sans-serif;
      font-size:11px;display:flex;flex-direction:column;gap:8px;}
    .sf-ir-chips{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;}
    .sf-ir-chip{background:#1d1d1d;border:1px solid #444;
      border-radius:4px;padding:6px 3px;font-size:9.5px;color:#ccc;
      text-align:center;cursor:pointer;user-select:none;transition:background .08s,border-color .08s;}
    .sf-ir-chip:hover{border-color:${"var(--sf-acc, #f66744)"};color:#ddd;}
    .sf-ir-chip.active{background:${"var(--sf-acc, #f66744)"};color:#fff;border-color:${"var(--sf-acc, #f66744)"};}
    /* Disabled while width/height are wired (mode doesn't apply). */
    .sf-ir-chip.disabled{opacity:.32;pointer-events:none;}
    /* Single-wire summary panel: read-only W / H rows. */
    .sf-ir-wirepanel{display:flex;flex-direction:column;gap:6px;}
    .sf-ir-wirerow{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#1d1d1d;border:1px solid #444;border-radius:4px;}
    .sf-ir-wirelbl{color:${"var(--sf-acc, #f66744)"};font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;width:14px;flex:none;}
    /* Wide variant for full-word labels (e.g. "LONGEST SIDE"). */
    .sf-ir-wirelbl.is-wide{width:auto;white-space:nowrap;}
    .sf-ir-wireval{color:#e0e0e0;font-size:13px;font-weight:600;flex:1;}
    .sf-ir-wiretag{color:#888;font-size:9px;text-transform:uppercase;letter-spacing:.5px;}
    /* Shared panels render in a 1fr grid with 1px borders — border-box so the
       border can't push the last column to clip. */
    .sf-ir-root .sf-li-quickpick,
    .sf-ir-root .sf-li-ratio-chip{box-sizing:border-box;}
    /* Locked (wire-driven) numeric fields dim to read as disabled. */
    .sf-ir-root .sf-li-numinput{opacity:.55;transition:opacity .1s;}
    .sf-ir-cards-canvas{display:block;width:100%;}
  `;
  const s = document.createElement("style");
  s.id = "sf-image-resize-css";
  s.textContent = css;
  document.head.appendChild(s);
  // The per-mode panel chrome (sf-li-*) + the global controls row styles.
  injectLoadImageChromeCSS();
}

// Sum children intrinsic heights (NOT scrollHeight — LiteGraph stretches the
// root, so scrollHeight feeds back). Root is a flex column with 8px gap +
// 2px/8px padding.
export function measureContentHeight(root) {
  if (!root) return 120;
  let h = 0;
  for (const c of root.children) h += c.offsetHeight;
  h += Math.max(0, root.children.length - 1) * 8; // row gaps
  h += 10; // root padding (2 top + 8 bottom)
  return Math.max(120, h);
}

const MODE_CHIPS = [
  { id: "off",          label: "Off" },
  { id: "max_mp",       label: "Max MP" },
  { id: "longest_side", label: "Longest side" },
  { id: "scale_factor", label: "Scale ×" },
  { id: "fit_inside",   label: "Fit inside" },
  { id: "cover",        label: "Crop to fill" },
  { id: "match_ratio",  label: "Match ratio" },
  { id: "pad",          label: "Pad" },
];

// Mode chips. When width/height are wired the modes are restricted (mirrors
// the Python _apply_wired_size precedence):
//   longest_side wired -> force Longest side display, lock every chip.
//   1 of width/height wired -> fixed aspect scale, NO mode applies.
//   both wired -> exact box, only Fit inside / Crop to fill apply.
export function buildChips(state, wi) {
  const wrap = document.createElement("div");
  wrap.className = "sf-ir-chips";
  for (const c of MODE_CHIPS) {
    const el = document.createElement("div");
    el.className = "sf-ir-chip" + (state.mode === c.id ? " active" : "");
    el.dataset.mode = c.id;
    el.textContent = c.label;
    wrap.appendChild(el);
  }
  if (wi.wiredLongest) {
    for (const c of wrap.querySelectorAll(".sf-ir-chip")) {
      c.classList.add("disabled");
      c.classList.toggle("active", c.dataset.mode === "longest_side");
    }
  } else if (wi.count === 1) {
    for (const c of wrap.querySelectorAll(".sf-ir-chip")) {
      c.classList.add("disabled");
      c.classList.remove("active");
    }
  } else if (wi.count === 2) {
    for (const c of wrap.querySelectorAll(".sf-ir-chip")) {
      const m = c.dataset.mode;
      c.classList.toggle("disabled", m !== "fit_inside" && m !== "cover");
      c.classList.toggle("active", m === (WH_MODES.has(state.mode) ? state.mode : "cover"));
    }
  }
  return wrap;
}

// The INPUT half of the Input -> Output card: the upstream image dims at edit
// time. Falls back to null (message shown) when unreadable.
export function getInputDims(node) {
  const inp = node.inputs?.find((i) => i.name === "image");
  if (!inp || inp.link == null) return null;
  let l = node.graph?.links?.[inp.link];
  if (!l && typeof node.graph?.links?.get === "function") l = node.graph.links.get(inp.link);
  if (!l) return null;
  const up = node.graph.getNodeById(l.origin_id);
  const img = up?.imgs?.[0];
  if (img?.naturalWidth) return { w: img.naturalWidth, h: img.naturalHeight };
  return null;
}

// Single-wire summary panel: shows the wired dimension + the auto-computed
// other dimension (keeps aspect). Read-only — no mode applies here.
export function buildSingleWirePanel(node, info, live) {
  const panel = document.createElement("div");
  panel.className = "sf-li-panel sf-ir-wirepanel";
  const state = readState(node, STATE_PROP, DEFAULT_STATE);
  const wv = info.wiredW ? info.valW : info.valH;
  let aw = null, ah = null;
  if (live && wv != null) {
    // Compute via the same path as the OUTPUT card (snap-aware) so the first
    // render matches the value the painter will show — no one-frame flash.
    const r = previewResizeOf(node, live, info, state);
    aw = r.w; ah = r.h;
  } else if (info.wiredW) { aw = wv; } else { ah = wv; }
  const mkRow = (label, val, tag) => {
    const r = document.createElement("div");
    r.className = "sf-ir-wirerow";
    const l = document.createElement("span"); l.className = "sf-ir-wirelbl"; l.textContent = label;
    const v = document.createElement("span"); v.className = "sf-ir-wireval"; v.textContent = val == null ? "—" : String(val);
    const t = document.createElement("span"); t.className = "sf-ir-wiretag"; t.textContent = tag;
    r.append(l, v, t);
    return { row: r, valEl: v };
  };
  const wRow = mkRow("W", aw, info.wiredW ? "来自接线" : "自动 · 保持比例");
  const hRow = mkRow("H", ah, info.wiredW ? "自动 · 保持比例" : "来自接线");
  panel.append(wRow.row, hRow.row);
  // Cache the value cells so the draw-loop poll can refresh them live when the
  // upstream wired value changes (DOM has no event for that).
  node._sfIrWireCells = { wEl: wRow.valEl, hEl: hRow.valEl };
  return panel;
}

// longest_side wired summary: one row "LONGEST SIDE | value | from wire". The
// OUTPUT card paints the resulting W×H; this just confirms the wired target.
export function buildLongestWirePanel(node, info) {
  const panel = document.createElement("div");
  panel.className = "sf-li-panel sf-ir-wirepanel";
  const row = document.createElement("div");
  row.className = "sf-ir-wirerow";
  const l = document.createElement("span"); l.className = "sf-ir-wirelbl is-wide"; l.textContent = "LONGEST SIDE";
  const v = document.createElement("span"); v.className = "sf-ir-wireval";
  v.textContent = info.valLongest == null ? "—" : String(info.valLongest);
  const t = document.createElement("span"); t.className = "sf-ir-wiretag"; t.textContent = "来自接线";
  row.append(l, v, t);
  panel.append(row);
  node._sfIrLongestCell = v;
  return panel;
}

// previewResize helper bound to the node's live input dims + wired state
// (shared by the single-wire panel and the readout).
function previewResizeOf(node, live, info, state) {
  const eff = effectiveWiredState(state, info, live.w, live.h);
  return previewResize(live.w, live.h, eff);
}

// Lock the W/H field(s) of the active Fit/Crop panel to the wire(s) (both-wired
// case). buildModePanel renders Width then Height as the first two text inputs.
function lockField(inp, val) {
  inp.readOnly = true;
  inp.title = "由接线输入驱动";
  if (val != null) inp.value = String(val);
  const wrap = inp.closest(".sf-li-numinput");
  if (wrap) { wrap.style.opacity = "0.55"; wrap.title = "由接线输入驱动"; }
}

export function applyWiredLocks(node, root, info) {
  const numEls = [...root.querySelectorAll(".sf-li-numinput input")];
  const wInp = info.wiredW ? numEls[0] : null;
  const hInp = info.wiredH ? numEls[1] : null;
  if (wInp) lockField(wInp, info.valW);
  if (hInp) lockField(hInp, info.valH);
  // Cache so the draw-loop poll can refresh the shown value live.
  node._sfIrLockedInputs = wInp || hInp ? { wInp, hInp } : null;
}

// Re-read the wired inputs and mirror the live values into the DOM readout
// cells (the single-wire / longest-side summaries + the locked W/H fields).
// Polled from the legacy onDrawForeground loop and the Nodes 2.0 setInterval.
export function refreshReadout(node) {
  const wi = wireInfo(node);
  const state = readState(node, STATE_PROP, DEFAULT_STATE);
  const cached = node.properties?.sfIrDims;
  const live = getInputDims(node);
  let info = readoutOf(node, state, cached, live, wi);
  if (info?.mode === "dual" && node._sfIrWireCells) {
    const c = node._sfIrWireCells, w = String(info.outW), h = String(info.outH);
    if (c.wEl.textContent !== w) c.wEl.textContent = w;
    if (c.hEl.textContent !== h) c.hEl.textContent = h;
  }
  if (node._sfIrLongestCell && wi.valLongest != null) {
    const s = String(wi.valLongest);
    if (node._sfIrLongestCell.textContent !== s) node._sfIrLongestCell.textContent = s;
  }
  if (node._sfIrLockedInputs) {
    const li = node._sfIrLockedInputs;
    if (li.wInp && wi.valW != null && li.wInp.value !== String(wi.valW)) li.wInp.value = String(wi.valW);
    if (li.hInp && wi.valH != null && li.hInp.value !== String(wi.valH)) li.hInp.value = String(wi.valH);
  }
  return { wi, info };
}

// What the painter shows, resolved against the node (null -> caller message).
function readoutOf(node, state, cached, live, wi) {
  if (!isImageWired(node)) return { mode: "msg", text: "连接图片后显示尺寸" };
  const info = getReadoutInfo(state, cached, live, wi);
  if (info) return info;
  if (cached) return { mode: "dual", inW: cached.in_w, inH: cached.in_h, outW: cached.out_w, outH: cached.out_h };
  return { mode: "msg", text: "运行一次后显示尺寸" };
}

function isImageWired(node) {
  const inp = node.inputs?.find((i) => i.name === "image");
  return !!(inp && inp.link != null);
}

// Paint the INPUT -> OUTPUT readout (two joined mini cards, or a single message
// box) centered horizontally in width W, vertically at midY. The two-card
// design aligns INPUT over column 2 / OUTPUT over column 3 of a 4-column grid
// spanning W (same grid the mode chips use), joined by a center bridge.
// SHARED by the legacy slot-dead-space paint (W = node.size[0], midY = 54) and
// the Nodes 2.0 cards canvas (W = canvas width, midY = canvas center).
export function paintReadout(ctx, info, W, midY) {
  const acc = sfAccent();
  const cx = W / 2;
  const fam = "ui-sans-serif, system-ui, sans-serif";
  const capFont = `8px ${fam}`;
  const dimsFont = `bold 10px ${fam}`;
  const ratioFont = `8px ${fam}`;
  ctx.save();
  ctx.textBaseline = "middle";

  if (info.mode === "msg") {
    ctx.font = `13px ${fam}`;
    const tw = ctx.measureText(info.text).width;
    const bw = tw + 26, bh = 28;
    roundRectPath(ctx, cx - bw / 2, midY - bh / 2, bw, bh, 8);
    ctx.fillStyle = "#1d1d1d"; ctx.fill();
    ctx.textAlign = "center"; ctx.fillStyle = acc;
    ctx.fillText(info.text, cx, midY);
    ctx.restore();
    return;
  }

  const GRID_PAD = 16, COL_GAP = 5, NECK_INSET = 3;
  const gridW = Math.max(40, W - GRID_PAD * 2);
  const colW = (gridW - COL_GAP * 3) / 4;
  const cardW = colW - NECK_INSET, cardH = 90;
  const L1 = GRID_PAD + colW + COL_GAP;          // INPUT left  (column 2 left)
  const R1 = L1 + cardW;                          // INPUT right (inset)
  const R2 = GRID_PAD + 3 * colW + 2 * COL_GAP;   // OUTPUT right (column 3 right)
  const L2 = R2 - cardW;                          // OUTPUT left (inset)
  const arrowCx = (R1 + L2) / 2;                  // bridge center
  const cardY = midY - cardH / 2, T = cardY, Bm = cardY + cardH;
  const R = 6, bridgeH = 22, bT = midY - bridgeH / 2, bB = midY + bridgeH / 2;
  const rectMaxW = 46, rectMaxH = 30;

  // Single connected outline: rounded OUTER corners on both cards, joined by a
  // center bridge (square inner junctions), with the gap above and below the
  // bridge left open.
  ctx.beginPath();
  ctx.moveTo(L1 + R, T);
  ctx.lineTo(R1 - R, T);
  ctx.arcTo(R1, T, R1, T + R, R);          // INPUT top-right
  ctx.lineTo(R1, bT);                       // down to bridge top
  ctx.lineTo(L2, bT);                       // bridge top across
  ctx.lineTo(L2, T + R);                    // up OUTPUT inner edge
  ctx.arcTo(L2, T, L2 + R, T, R);          // OUTPUT top-left
  ctx.lineTo(R2 - R, T);
  ctx.arcTo(R2, T, R2, T + R, R);          // OUTPUT top-right
  ctx.lineTo(R2, Bm - R);
  ctx.arcTo(R2, Bm, R2 - R, Bm, R);        // OUTPUT bottom-right
  ctx.lineTo(L2 + R, Bm);
  ctx.arcTo(L2, Bm, L2, Bm - R, R);        // OUTPUT bottom-left
  ctx.lineTo(L2, bB);                       // up to bridge bottom
  ctx.lineTo(R1, bB);                       // bridge bottom across
  ctx.lineTo(R1, Bm - R);                   // down INPUT inner edge
  ctx.arcTo(R1, Bm, R1 - R, Bm, R);        // INPUT bottom-right
  ctx.lineTo(L1 + R, Bm);
  ctx.arcTo(L1, Bm, L1, Bm - R, R);        // INPUT bottom-left
  ctx.lineTo(L1, T + R);
  ctx.arcTo(L1, T, L1 + R, T, R);          // INPUT top-left
  ctx.closePath();
  ctx.fillStyle = "#1d1d1d"; ctx.fill();
  ctx.strokeStyle = "#444"; ctx.lineWidth = 1; ctx.stroke();

  const drawContent = (x, label, w, h, accent) => {
    const ccx = x + cardW / 2;
    ctx.textAlign = "center";
    const maxTxt = cardW - 8;
    ctx.font = capFont; ctx.fillStyle = "#9a9a9a";
    ctx.fillText(label, ccx, cardY + 15, maxTxt);
    ctx.font = dimsFont; ctx.fillStyle = acc;
    ctx.fillText(`${w}×${h}`, ccx, cardY + 27, maxTxt);
    const { rw, rh } = aspectRectDims(w, h, rectMaxW, rectMaxH);
    const rx = Math.round(ccx - rw / 2) + 0.5, ry = Math.round(cardY + 53 - rh / 2) + 0.5;
    if (accent) { ctx.fillStyle = "rgba(246,103,68,0.20)"; ctx.fillRect(rx, ry, rw, rh); }
    ctx.strokeStyle = accent ? acc : "rgba(200,200,200,0.7)"; ctx.lineWidth = 1;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.font = ratioFont; ctx.fillStyle = "#9a9a9a";
    ctx.fillText(ratioLabel(w, h), ccx, cardY + 77, maxTxt);
  };

  const changed = info.inW !== info.outW || info.inH !== info.outH;
  drawContent(L1, "INPUT", info.inW, info.inH, false);
  drawContent(L2, "OUTPUT", info.outW, info.outH, changed);

  // Compact ">" chevron centered on the bridge.
  ctx.strokeStyle = "#9a9a9a"; ctx.lineWidth = 1;
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(arrowCx - 2.5, midY - 4);
  ctx.lineTo(arrowCx + 2.5, midY);
  ctx.lineTo(arrowCx - 2.5, midY + 4);
  ctx.stroke();

  ctx.restore();
}

// Render the whole node body. `onChange` is the non-destructive repaint hook
// (readout refresh); `onRefit` is called after a genuine user action that
// changes panel height (mode switch), never on the load path.
export function renderUI(node, { onChange, onRefit } = {}) {
  const root = node._sfIrRoot;
  if (!root) return;
  const state = readState(node, STATE_PROP, DEFAULT_STATE);
  // Null the live-refresh caches BEFORE wiping the DOM (their old elements are
  // about to be detached); the panels below re-set whichever applies.
  node._sfIrWireCells = null;
  node._sfIrLockedInputs = null;
  node._sfIrLongestCell = null;
  root.innerHTML = "";

  // Nodes 2.0: re-assert the readout cards canvas as the FIRST child (root was
  // just wiped). Force a redraw next frame, once it has laid out.
  if (node._sfIrCardsCanvas) {
    root.appendChild(node._sfIrCardsCanvas);
    requestAnimationFrame(() => node._sfIrRenderCards?.(true));
  }

  const info = wireInfo(node);
  const live = getInputDims(node);
  const rs = (n) => readState(n, STATE_PROP, DEFAULT_STATE);
  const ws = (n, s) => writeState(n, STATE_PROP, s);
  const repaint = () => { node.setDirtyCanvas?.(true, true); onChange?.(); };

  const chips = buildChips(state, info);
  root.appendChild(chips);

  // Display mode under wiring: longest_side wire wins; 2 wired = exact box
  // (honour Fit if active, else show Crop to fill); 1 wired = no mode.
  let dispMode = state.mode;
  if (info.wiredLongest) dispMode = "longest_side";
  else if (info.count === 2) dispMode = WH_MODES.has(state.mode) ? state.mode : "cover";

  let panel = null;
  if (info.wiredLongest) {
    panel = buildLongestWirePanel(node, info);
  } else if (info.count === 1) {
    panel = buildSingleWirePanel(node, info, live);
  } else {
    panel = buildModePanel(dispMode, node, state, ws, repaint, STATE_PROP,
      { previewMaxW: 134, previewMaxH: 96, cropOnly: true, inputDims: live, oneLine: true });
    if (panel) {
      applyInlineLabel(panel, dispMode);
      if (dispMode === "fit_inside" || dispMode === "cover") applyWHLayout(panel);
      if (dispMode === "cover") applyCoverControls(node, panel, rs, ws, repaint);
      // No redundant title row — the highlighted chip names the mode.
      panel.querySelector(".sf-li-panel-label")?.remove();
    }
  }
  if (panel) root.appendChild(panel);

  const globals = renderGlobalControls(node, state, ws, repaint, STATE_PROP);
  root.appendChild(globals);

  applyWiredLocks(node, root, info);
  repaint();

  // ── wiring ──
  chips.addEventListener("click", (e) => {
    const c = e.target.closest(".sf-ir-chip");
    if (!c || c.classList.contains("disabled")) return;
    ws(node, { ...rs(node), mode: c.dataset.mode });
    renderUI(node, { onChange, onRefit });
    onRefit?.();
  });
}
