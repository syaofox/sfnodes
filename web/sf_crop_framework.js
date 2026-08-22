// SF Image Crop — 精简版编辑器框架（移植自 comfyui-pixaroma js/framework/，
// 仅包含 Crop 编辑器用到的部分；图标 data URI 内联，无 pixaroma 品牌/资产依赖）。

import { injectCSSOnce } from "./sf_common.js";

// ── 图标（内联 data URI，项目惯例见 sf_workflows_ui.js） ──────────────────
const ICONS = {
  save: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M50.871,1.325c1.347.603,2.188,1.758,3.202,2.776l5.076,5.096c1.076,1.08,1.9,2.084,2.19,3.718v44.323c-.563,3.105-2.648,5.31-5.757,5.907H8.405c-3.019-.684-5.706-3.082-5.71-6.362V7.372C3.088,3.654,5.967.876,9.684.886h8.216v17.49c.003,1.979,1.835,3.595,3.788,3.595h20.629c1.979,0,3.779-1.547,3.782-3.606V1.495c0-.133.071-.497.184-.529.732-.204,3.428-.161,4.587.358ZM52.257,54.993v-19.669c.004-1.973-1.219-3.574-3.072-4.095h-31.677c-.886,0-1.546-.184-2.484.002-1.776.352-3.216,1.992-3.213,3.988v19.621c.002,1.777,1.491,3.811,3.321,3.813h33.335c1.905.002,3.787-1.596,3.791-3.66ZM40.759,18.21h-4.118c-.958-.003-1.565-.768-1.766-1.678V6.446c.001-.845.758-1.406,1.467-1.55h4.506c.718.181,1.298.862,1.299,1.679v9.846c0,.85-.546,1.554-1.388,1.79Z"/></svg>'),
  upload: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M58.115,1.482H5.885C3.239,1.482,1.094,3.627,1.094,6.273v51.453c0,2.646,2.145,4.791,4.791,4.791h52.23c2.646,0,4.791-2.145,4.791-4.791V6.273c0-2.646-2.145-4.791-4.791-4.791ZM49.641,28.696h-11.702v24.054c0,1.147-.93,2.077-2.077,2.077h-7.726c-1.147,0-2.077-.93-2.077-2.077v-24.054h-11.702c-2.409,0-3.487-3.024-1.62-4.547l17.641-14.398c.472-.384,1.046-.577,1.62-.577s1.149.193,1.62.577l17.641,14.398c1.867,1.523.789,4.547-1.62,4.547Z"/></svg>'),
  swap: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M13.622,36.23L.925,20.673c-.339-.416-.509-.922-.509-1.429s.17-1.013.509-1.429L13.622,2.259c1.344-1.646,4.01-.696,4.01,1.429v10.319h41.824c1.027,0,1.86.833,1.86,1.86v6.756c0,1.027-.833,1.86-1.86,1.86H17.632v10.319c0,2.125-2.666,3.075-4.01,1.429ZM63.491,43.327l-12.697-15.557c-1.344-1.646-4.01-.696-4.01,1.429v10.319H4.96c-1.027,0-1.86.833-1.86,1.86v6.756c0,1.027.833,1.86,1.86,1.86h41.824v10.319c0,2.125,2.667,3.075,4.01,1.429l12.697-15.557c.339-.416.509-.922.509-1.429s-.17-1.013-.509-1.429Z"/></svg>'),
  help: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M58.115,1.482H5.885C3.239,1.482,1.094,3.627,1.094,6.273v51.453c0,2.646,2.145,4.791,4.791,4.791h52.23c2.646,0,4.791-2.145,4.791-4.79V6.273c0-2.646-2.145-4.791-4.791-4.791ZM31.568,55.964c-2.992,0-5.417-2.425-5.417-5.417s2.425-5.417,5.417-5.417,5.417,2.425,5.417,5.417-2.425,5.417-5.417,5.417ZM45.58,25.271c-3.529,7.741-9.903,6.913-10.121,15.722h-8.529c.08-11.174,5.01-11.133,8.593-16.076,6.349-8.782-8.514-13.088-8.625-3.557h-9.752c.312-21.491,36.915-14.63,28.435,3.911Z"/></svg>'),
  download: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M5.885,62.518h52.23c2.646,0,4.791-2.145,4.791-4.791V6.273c0-2.646-2.145-4.791-4.791-4.791H5.885C3.239,1.482,1.094,3.627,1.094,6.273v51.453c0,2.646,2.145,4.791,4.791,4.791ZM14.359,35.304h11.702V11.25c0-1.147.93-2.077,2.077-2.077h7.726c1.147,0,2.077.93,2.077,2.077v24.054h11.702c2.409,0,3.487,3.024,1.62,4.547l-17.641,14.398c-.472.384-1.046.577-1.62.577s-1.149-.193-1.62-.577l-17.641-14.398c-1.867-1.523-.789-4.547,1.62-4.547Z"/></svg>'),
  undo: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M52.383,56.044c-.059-3.655-.657-7.104-1.973-10.348-4.123-10.162-13.964-16.644-24.909-16.314l-.026,10.478c-.003,1.004-.877,1.806-1.533,2.109-.993.459-2.056.281-2.866-.375l-3.002-2.428L1.517,25.896c-1.188-.988-1.321-2.412-.437-3.742L21.05,6.14c.82-.658,1.812-.845,2.82-.43.693.285,1.591,1.14,1.595,2.184l.037,10.243c4.927.012,9.747.734,14.311,2.543,6.938,2.75,12.817,7.393,17.032,13.549,4.426,6.464,6.744,14.158,6.645,21.991-.016,1.287-1.155,2.283-2.336,2.284l-6.297.007c-1.351.001-2.452-1.043-2.475-2.466Z"/></svg>'),
  redo: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M11.617,56.044c.059-3.655.657-7.104,1.973-10.348,4.123-10.162,13.964-16.644,24.909-16.314l.026,10.478c.003,1.004.877,1.806,1.533,2.109.993.459,2.056.281,2.866-.375l3.002-2.428,16.557-13.269c1.188-.988,1.321-2.412.437-3.742L42.95,6.14c-.82-.658-1.812-.845-2.82-.43-.693.285-1.591,1.14-1.595,2.184l-.037,10.243c-4.927.012-9.747.734-14.311,2.543-6.938,2.75-12.817,7.393-17.032,13.549C2.729,40.693.411,48.387.51,56.22c.016,1.287,1.155,2.283,2.336,2.284l6.297.007c1.351.001,2.452-1.043,2.475-2.466Z"/></svg>'),
  minus: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect x="1.948" y="26.223" width="60.104" height="11.553" rx="2.309" ry="2.309"/></svg>'),
  plus: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M59.743,26.224h-21.966V4.257c0-1.275-1.034-2.309-2.309-2.309h-6.935c-1.275,0-2.309,1.034-2.309,2.309v21.966H4.257c-1.275,0-2.309,1.034-2.309,2.309v6.935c0,1.275,1.034,2.309,2.309,2.309h21.966v21.966c0,1.275,1.034,2.309,2.309,2.309h6.935c1.275,0,2.309-1.034,2.309-2.309v-21.966h21.966c1.275,0,2.309-1.034,2.309-2.309v-6.935c0-1.275-1.034-2.309-2.309-2.309Z"/></svg>'),
};

// ── 品牌主色 ────────────────────────────────────────────────────────────
export const BRAND = "#f66744";

/**
 * Creates an <img> element pointing to a UI icon SVG.
 * @param {string} name - Filename inside the UI icons folder (e.g. "save.svg")
 * @param {number} [size=14] - Width and height in px
 * @returns {HTMLImageElement}
 */
export function _uiIcon(name, size = 14) {
  const img = document.createElement("img");
  img.src = ICONS[name] || "";
  img.style.cssText = `width:${size}px;height:${size}px;pointer-events:none;`;
  img.draggable = false;
  return img;
}

/** ID used for the injected <style> element — prevents duplicate injection. */
const STYLE_ID = "sf-crop-framework-css";

// ═════════════════════════════════════════════════════════════════
//  CSS Injection
// ═════════════════════════════════════════════════════════════════

export function injectFrameworkStyles() {
  injectCSSOnce(STYLE_ID, `
/* ═══════════════════════════════════════════════════════
   Pixaroma Editor Framework — Shared Stylesheet
   ═══════════════════════════════════════════════════════ */

/* ── CSS Custom Properties ──────────────────────────── */
.sf-px-overlay {
  --sf-px-accent: var(--sf-acc, #f66744);
  --sf-px-accent-hover: #e05535;
  --sf-px-bg-darkest: #131415;
  --sf-px-bg-dark: #171718;
  --sf-px-bg-sidebar: #181a1b;
  --sf-px-bg-panel: #242628;
  --sf-px-bg-input: #111;
  --sf-px-bg-btn: #353535;
  --sf-px-border: #3a3d40;
  --sf-px-border-subtle: #2a2c2e;
  --sf-px-border-titlebar: #2e3033;
  --sf-px-text: #e0e0e0;
  --sf-px-text-dim: #888;
  --sf-px-text-dimmer: #666;
  --sf-px-text-label: #999;
  --sf-px-select-bg: #2a1800;
  --sf-px-select-border: var(--sf-acc, #f66744);
  --sf-px-multi-bg: #0a1a2a;
  --sf-px-multi-border: #0ea5e9;
  --sf-px-danger: #d46060;
  --sf-px-danger-bg: #2a1a1a;
  --sf-px-font: 'Segoe UI', system-ui, sans-serif;
  --sf-px-font-mono: monospace;
}

/* ── Overlay (fullscreen editor) ────────────────────── */
.sf-px-overlay {
  position: fixed; inset: 0; z-index: 11000;
  display: flex; flex-direction: column;
  background: var(--sf-px-bg-dark);
  font-family: var(--sf-px-font);
  color: var(--sf-px-text);
  overflow: hidden; user-select: none;
}

/* ── Titlebar ───────────────────────────────────────── */
.sf-px-titlebar {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 12px; background: var(--sf-px-bg-darkest);
  border-bottom: 1px solid var(--sf-px-border-titlebar);
  flex-shrink: 0; height: 38px;
}
.sf-px-title {
  color: #fff; font-size: 13px; font-weight: bold;
  display: flex; align-items: center; gap: 6px;
  flex-shrink: 0;
}
.sf-px-title-brand { color: var(--sf-px-accent); }
.sf-px-title-logo { width: 20px; height: 20px; }
.sf-px-titlebar-center {
  flex: 1; display: flex; align-items: center; justify-content: center; gap: 6px;
  min-width: 0;
}
.sf-px-titlebar-actions {
  display: flex; align-items: center; gap: 6px;
  flex-shrink: 0;
}
.sf-px-titlebar-zoom {
  display: flex; align-items: center; gap: 3px;
  background: rgba(255,255,255,0.05); border: 1px solid var(--sf-px-border);
  border-radius: 5px; padding: 2px 4px;
}
.sf-px-titlebar-zoom .sf-px-zoom-label {
  font-size: 10px; color: var(--sf-px-text-dim);
  min-width: 36px; text-align: center;
}
.sf-px-titlebar-sep {
  width: 1px; height: 18px; background: var(--sf-px-border); flex-shrink: 0;
  margin: 0 4px;
}

/* ── Top options bar (below titlebar, e.g. Paint brush opts) ── */
.sf-px-top-options {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  padding: 4px 10px; background: var(--sf-px-bg-darkest);
  border-bottom: 1px solid var(--sf-px-border-subtle);
  flex-shrink: 0; min-height: 34px;
}

/* ── Body (sidebars + workspace) ────────────────────── */
.sf-px-body {
  display: flex; flex: 1; overflow: hidden; min-height: 0;
}

/* ── Sidebars ───────────────────────────────────────── */
.sf-px-sidebar {
  flex-shrink: 0; background: var(--sf-px-bg-sidebar);
  display: flex; flex-direction: column;
  overflow-y: auto; overflow-x: hidden;
  scrollbar-gutter: stable;
  position: relative; z-index: 5;
}
.sf-px-sidebar-left { border-right: 1px solid var(--sf-px-border-subtle); }
.sf-px-sidebar-right { border-left: 1px solid var(--sf-px-border-subtle); overflow-y: hidden; }

/* Sidebar scrollbar */
.sf-px-sidebar::-webkit-scrollbar { width: 5px; }
.sf-px-sidebar::-webkit-scrollbar-track { background: var(--sf-px-bg-input); }
.sf-px-sidebar::-webkit-scrollbar-thumb { background: var(--sf-px-border); border-radius: 3px; }
.sf-px-sidebar::-webkit-scrollbar-thumb:hover { background: var(--sf-px-accent); }

/* ── Workspace (center canvas area) ─────────────────── */
.sf-px-workspace {
  flex: 1; position: relative; overflow: hidden;
  background: #111315;
  display: flex; align-items: center; justify-content: center;
}

/* Sidebar footer (save/close/help — always at bottom) */
.sf-px-sidebar-footer {
  padding: 10px 12px; margin-top: auto;
  border-top: 1px solid var(--sf-px-border-titlebar);
  display: flex; flex-direction: column; gap: 6px;
  flex-shrink: 0;
}

/* ── Tool info (floating tooltip in workspace, bottom-left) ── */
.sf-px-tool-info {
  position: absolute; bottom: 10px; left: 10px;
  background: rgba(0,0,0,0.75); color: #ccc;
  padding: 5px 12px; border-radius: 5px;
  font-size: 10px; font-family: var(--sf-px-font-mono);
  pointer-events: none; z-index: 5;
  max-width: 80%;
  transition: color 0.15s ease;
}
/* Editors that never write status text (AudioReact) leave this element
   empty — its padding + dark background still rendered a small box that
   overlapped AudioReact's bottom-left transport buttons. :empty hides it
   until an editor actually writes text via setStatusText(). */
.sf-px-tool-info:empty { display: none; }
.sf-px-tool-info.warn { color: var(--sf-acc, #f66744); }
.sf-px-tool-info.error { color: #f08080; }

/* ── Panel / Section ────────────────────────────────── */
.sf-px-panel {
  padding: 8px 10px;
  border-bottom: 1px solid var(--sf-px-border-subtle);
}
.sf-px-panel-title {
  font-size: 9px; color: var(--sf-px-accent); font-weight: bold;
  text-transform: uppercase; letter-spacing: .06em;
  margin-bottom: 6px; cursor: default;
  display: flex; align-items: center; gap: 4px;
}
.sf-px-panel-title-arrow {
  font-size: 8px; transition: transform .15s; display: inline-block;
}
.sf-px-panel.collapsed .sf-px-panel-title-arrow { transform: rotate(-90deg); }
.sf-px-panel.collapsed .sf-px-panel-content { display: none; }
.sf-px-panel-title.clickable { cursor: pointer; }
.sf-px-panel-title.clickable:hover { color: #fff; }

/* ── Buttons ────────────────────────────────────────── */
.sf-px-btn, .sf-px-btn-full, .sf-px-btn-sm {
  font-family: inherit; cursor: pointer;
  border-radius: 5px; border: 1px solid var(--sf-px-border);
  transition: all .15s ease; white-space: nowrap;
  display: inline-flex; align-items: center; justify-content: center; gap: 5px;
}
.sf-px-btn:disabled, .sf-px-btn-full:disabled, .sf-px-btn-sm:disabled {
  opacity: 0.35; cursor: default; pointer-events: none;
}
.sf-px-btn img, .sf-px-btn-full img, .sf-px-btn-sm img {
  width: 14px; height: 14px; filter: brightness(0) invert(0.7);
  pointer-events: none;
}
.sf-px-btn:hover img, .sf-px-btn-full:hover img, .sf-px-btn-sm:hover img {
  filter: brightness(0) invert(1);
}

.sf-px-btn {
  background: var(--sf-px-bg-btn); color: #ccc;
  padding: 6px 14px; font-size: 12px;
}
.sf-px-btn:hover { background: #2e3033; color: var(--sf-px-accent); border-color: var(--sf-px-accent); }

.sf-px-btn.sf-px-btn-accent, .sf-px-btn-accent {
  background: var(--sf-px-accent); border-color: var(--sf-px-accent);
  color: #fff; font-weight: bold;
}
.sf-px-btn.sf-px-btn-accent:hover, .sf-px-btn-accent:hover {
  background: var(--sf-px-accent-hover); border-color: var(--sf-px-accent-hover);
}
.sf-px-btn-accent img { filter: brightness(0) invert(1); }

.sf-px-btn.sf-px-btn-danger, .sf-px-btn-full.sf-px-btn-danger, .sf-px-btn-sm.sf-px-btn-danger {
  background: #1e2022 !important; color: #ccc !important;
  border-color: #d93523 !important;
}
.sf-px-btn.sf-px-btn-danger:hover, .sf-px-btn-full.sf-px-btn-danger:hover, .sf-px-btn-sm.sf-px-btn-danger:hover {
  background: #d93523 !important; color: #fff !important;
  border-color: #d93523 !important;
}
.sf-px-btn-danger img, .sf-px-btn-danger svg {
  filter: none !important;
}
.sf-px-btn-danger:hover img, .sf-px-btn-danger:hover svg {
  filter: brightness(0) invert(1) !important;
}

.sf-px-btn-full {
  width: 100%; padding: 7px 10px; font-size: 11px;
  background: #1e2022; color: #ccc;
}
.sf-px-btn-full:hover { background: #2e3033; color: var(--sf-px-accent); border-color: var(--sf-px-accent); }

.sf-px-btn-sm {
  min-width: 28px; height: 28px; padding: 0 4px; flex-shrink: 0;
  background: var(--sf-px-bg-panel); color: #ccc; font-size: 13px;
}
.sf-px-btn-sm:hover { background: #2e3033; color: var(--sf-px-accent); border-color: var(--sf-px-accent); }

.sf-px-btn-icon {
  background: none; border: none; color: #ccc; padding: 4px;
  cursor: pointer; font-size: 16px; border-radius: 4px; transition: all .15s;
  display: inline-flex; align-items: center; justify-content: center;
}
.sf-px-btn-icon:hover { color: var(--sf-px-accent); background: rgba(255,255,255,0.05); }
.sf-px-btn-icon:disabled { opacity: 0.3; cursor: default; pointer-events: none; }

.sf-px-btn-row { display: flex; gap: 6px; }
.sf-px-btn-row > .sf-px-btn, .sf-px-btn-row > .sf-px-btn-full { flex: 1; }

.sf-px-btn.active { background: var(--sf-px-accent); border-color: var(--sf-px-accent); color: #fff; }
.sf-px-btn.active img { filter: brightness(0) invert(1); }

/* ── Pill grid ─────────────────────────────────────── */
.sf-px-pill-grid { display: grid; gap: 4px; }
.sf-px-pill {
  font-size: 10px; background: #1e2022; border: 1px solid var(--sf-px-border);
  color: #aaa; border-radius: 3px; padding: 4px 0; cursor: pointer;
  transition: all .1s; text-align: center; font-family: inherit;
}
.sf-px-pill:hover { background: #444; color: #fff; }
.sf-px-pill.active { background: var(--sf-px-accent); border-color: var(--sf-px-accent); color: #fff; }

/* ── Slider row ─────────────────────────────────────── */
.sf-px-slider-row {
  display: flex; align-items: center; gap: 5px; margin-bottom: 5px;
}
.sf-px-slider-label {
  font-size: 10px; color: var(--sf-px-text-dim); flex-shrink: 0;
}
.sf-px-slider-row input[type=number] {
  width: 48px; background: var(--sf-px-bg-input); color: var(--sf-px-text);
  border: 1px solid var(--sf-px-border); border-radius: 4px;
  padding: 3px 4px; font-size: 10px; font-family: var(--sf-px-font-mono);
  flex-shrink: 0; text-align: center;
}
.sf-px-slider-row input[type=number]:focus {
  border-color: var(--sf-px-accent); outline: none;
}

/* ── Unified slider styling ──────────────────────────── */
.sf-px-overlay input[type=range] {
  -webkit-appearance: none; appearance: none;
  flex: 1; min-width: 0; height: 6px; cursor: pointer;
  background: linear-gradient(to right,
    var(--sf-px-accent) 0%, var(--sf-px-accent) var(--sf-px-fill, 50%),
    var(--sf-px-border) var(--sf-px-fill, 50%), var(--sf-px-border) 100%);
  border-radius: 3px; border: none; outline: none;
}
.sf-px-overlay input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 12px; height: 12px; border-radius: 50%;
  background: var(--sf-px-accent); border: none;
  box-shadow: 0 0 3px rgba(0,0,0,0.5);
  cursor: pointer; margin-top: -3px;
}
.sf-px-overlay input[type=range]::-moz-range-thumb {
  width: 12px; height: 12px; border-radius: 50%;
  background: var(--sf-px-accent); border: none;
  box-shadow: 0 0 3px rgba(0,0,0,0.5);
  cursor: pointer;
}
.sf-px-overlay input[type=range]::-webkit-slider-runnable-track {
  height: 6px; border-radius: 3px; background: transparent;
}
.sf-px-overlay input[type=range]::-moz-range-track {
  height: 6px; border-radius: 3px; background: transparent;
}
.sf-px-overlay input[type=range]::-moz-range-progress {
  height: 6px; border-radius: 3px; background: var(--sf-px-accent);
}

/* ── Number input ───────────────────────────────────── */
.sf-px-input-num {
  width: 55px; background: var(--sf-px-bg-input); color: var(--sf-px-text);
  border: 1px solid var(--sf-px-border); border-radius: 3px;
  padding: 3px 4px; font-size: 10px; font-family: var(--sf-px-font-mono);
  text-align: center;
}

/* ── Select dropdown ────────────────────────────────── */
.sf-px-select {
  background: var(--sf-px-bg-input); color: var(--sf-px-text);
  border: 1px solid var(--sf-px-border); border-radius: 4px;
  padding: 4px 6px; font-size: 11px; font-family: inherit;
  cursor: pointer; width: 100%;
}

/* ── Color input ────────────────────────────────────── */
.sf-px-color-input {
  width: 50px; height: 22px; cursor: pointer;
  border: 1px solid var(--sf-px-border); border-radius: 4px;
  background: var(--sf-px-bg-input); padding: 0;
}

/* ── Row (label + content) ──────────────────────────── */
.sf-px-row {
  display: flex; align-items: center; gap: 6px; margin-bottom: 5px;
}
.sf-px-row-label {
  font-size: 10px; color: var(--sf-px-text-dim); flex-shrink: 0;
}

/* ── Button row ─────────────────────────────────────── */
.sf-px-btn-row { display: flex; gap: 6px; }

/* ── Tool button ────────────────────────────────────── */
.sf-px-tool-btn {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 1px;
  height: 38px; background: #1c1e1f; border: 1px solid var(--sf-px-border);
  color: #ccc; border-radius: 4px; cursor: pointer;
  font-family: inherit; font-size: 10px; transition: all .12s;
  padding: 2px;
}
.sf-px-tool-btn:hover { background: #2e3033; color: #fff; border-color: #555; }
.sf-px-tool-btn.active { background: var(--sf-px-accent); border-color: var(--sf-px-accent); color: #fff; }
.sf-px-tool-btn-icon { font-size: 14px; line-height: 1; }
.sf-px-tool-btn-label { font-size: 8px; line-height: 1; }

/* ── Tool grid ──────────────────────────────────────── */
.sf-px-tool-grid { display: grid; gap: 4px; }

/* ── Layer Panel (unified Photoshop-style) ──────────── */
.sf-px-layer-panel {
  display: flex; flex-direction: column; min-height: 0; flex: 1;
}
.sf-px-layer-blend-row {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 8px; border-bottom: 1px solid var(--sf-px-border-subtle);
}
.sf-px-layer-blend-select {
  flex: 1; background: var(--sf-px-bg-input); color: var(--sf-px-text);
  border: 1px solid var(--sf-px-border); border-radius: 4px;
  padding: 4px 6px; font-size: 11px; font-family: inherit;
}
.sf-px-layer-opacity-row {
  display: flex; align-items: center; gap: 5px;
  padding: 6px 8px; border-bottom: 1px solid var(--sf-px-border-subtle);
}
.sf-px-layer-opacity-label {
  font-size: 9px; color: var(--sf-px-text-dim); flex-shrink: 0;
}
.sf-px-layer-opacity-row input[type=number] {
  width: 42px; background: var(--sf-px-bg-input); color: var(--sf-px-text);
  border: 1px solid var(--sf-px-border); border-radius: 4px;
  padding: 3px 4px; font-size: 10px; font-family: var(--sf-px-font-mono);
  text-align: center; flex-shrink: 0;
}
.sf-px-layer-opacity-row input[type=number]:focus {
  border-color: var(--sf-px-accent); outline: none;
}

/* Layer list */
.sf-px-layers-list {
  overflow-y: auto; min-height: 40px; flex: 1;
  padding: 2px 0;
}
.sf-px-layers-resize {
  height: 1px; background: var(--sf-px-border-subtle); flex-shrink: 0;
}
.sf-px-layers-list::-webkit-scrollbar { width: 4px; }
.sf-px-layers-list::-webkit-scrollbar-track { background: transparent; }
.sf-px-layers-list::-webkit-scrollbar-thumb { background: var(--sf-px-border); border-radius: 2px; }

/* Layer item */
.sf-px-layer-item {
  display: flex; align-items: center; gap: 4px;
  padding: 3px 6px; border-radius: 4px;
  border: 1px solid transparent; cursor: pointer;
  font-size: 11px; transition: background .1s;
  min-height: 30px;
}
.sf-px-layer-item:hover { background: rgba(255,255,255,0.04); }
.sf-px-layer-item.active {
  background: var(--sf-px-select-bg); border-color: var(--sf-px-select-border);
}
.sf-px-layer-item.multi-selected {
  background: var(--sf-px-multi-bg); border-color: var(--sf-px-multi-border);
}
.sf-px-layer-item.drag-over-top { border-top: 2px solid var(--sf-px-accent); }
.sf-px-layer-item.drag-over-bottom { border-bottom: 2px solid var(--sf-px-accent); }
.sf-px-layer-item.dragging { opacity: 0.35; }

/* Layer icon buttons (eye, lock) */
.sf-px-layer-icon {
  width: 16px; height: 16px; flex-shrink: 0; cursor: pointer;
  opacity: 0.5; transition: opacity .15s;
  display: flex; align-items: center; justify-content: center;
}
.sf-px-layer-icon:hover { opacity: 1; }
.sf-px-layer-icon img {
  width: 12px; height: 12px; display: block;
  filter: brightness(0) invert(0.7);
}
.sf-px-layer-icon:hover img { filter: brightness(0) invert(1); }
.sf-px-layer-item.active .sf-px-layer-icon img { filter: brightness(0) invert(0.9); }

/* Layer thumbnail */
.sf-px-layer-thumb {
  width: 28px; height: 28px; flex-shrink: 0; border-radius: 3px;
  background: repeating-conic-gradient(#333 0% 25%, #222 0% 50%) 50% / 8px 8px;
  overflow: hidden; border: 1px solid rgba(255,255,255,0.06);
}

/* Layer name */
.sf-px-layer-name {
  flex: 1; font-size: 11px; color: #ccc;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  padding: 2px 4px; min-width: 0;
}
.sf-px-layer-name-input {
  flex: 1; background: var(--sf-px-bg-input); color: var(--sf-px-text);
  border: 1px solid var(--sf-px-accent); border-radius: 3px;
  font-size: 11px; padding: 2px 6px; outline: none;
  font-family: inherit; min-width: 0;
}

/* Action bar (add, dup, delete, up, down, merge) */
.sf-px-layers-actions {
  display: flex; gap: 2px; padding: 5px 4px;
  border-top: 1px solid var(--sf-px-border-subtle);
  justify-content: center;
}
.sf-px-layer-action-btn {
  width: 28px; height: 28px; padding: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--sf-px-bg-panel); border: 1px solid var(--sf-px-border);
  border-radius: 4px; cursor: pointer; transition: all .12s;
}
.sf-px-layer-action-btn:hover {
  background: #2e3033; border-color: var(--sf-px-accent);
}
.sf-px-layer-action-btn:disabled { opacity: 0.3; cursor: default; pointer-events: none; }
.sf-px-layer-action-btn img {
  width: 14px; height: 14px; display: block;
  filter: brightness(0) invert(0.7);
}
.sf-px-layer-action-btn:hover img { filter: brightness(0) invert(1); }
.sf-px-layer-action-btn.danger { border-color: #d93523; }
.sf-px-layer-action-btn.danger:hover {
  background: #d93523; border-color: #d93523;
}
.sf-px-layer-action-btn.danger:hover img {
  filter: brightness(0) invert(1) !important;
}

/* ── Canvas Toolbar (Add Image, BG Color, Clear) ───── */
.sf-px-canvas-toolbar {
  display: flex; flex-direction: column; gap: 5px;
  padding: 8px 10px; border-bottom: 1px solid var(--sf-px-border-subtle);
}
.sf-px-canvas-toolbar-row {
  display: flex; align-items: center; gap: 6px;
}
.sf-px-canvas-toolbar .sf-px-btn-full {
  font-size: 11px; padding: 6px 8px;
}
/* Drag & drop overlay on workspace */
.sf-px-drop-overlay {
  display: none; position: absolute; inset: 0; z-index: 50;
  background: rgba(246, 103, 68, 0.08);
  border: 3px dashed var(--sf-px-accent);
  border-radius: 8px;
  pointer-events: none;
  align-items: center; justify-content: center;
}
.sf-px-drop-overlay.active { display: flex; }
.sf-px-drop-label {
  background: rgba(0,0,0,0.7); color: var(--sf-px-accent);
  padding: 12px 24px; border-radius: 8px;
  font-size: 14px; font-weight: bold;
}

/* ── Help overlay (unified modal, 2-column layout) ───── */
.sf-px-help-overlay {
  display: none; position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  background: #171718; border: 1px solid var(--sf-px-accent);
  border-radius: 10px; padding: 0;
  width: 960px; max-width: 95%; max-height: 86vh;
  z-index: 100; overflow: hidden;
  box-shadow: 0 12px 40px rgba(0,0,0,0.6);
  font-family: var(--sf-px-font);
}
.sf-px-help-header {
  display: flex; align-items: center; padding: 14px 20px;
  border-bottom: 1px solid #2a2a2a;
}
.sf-px-help-header h3 { flex: 1; color: var(--sf-px-accent); font-size: 14px; margin: 0; font-weight: 600; }
.sf-px-help-content {
  padding: 18px 24px; overflow-y: auto;
  max-height: calc(86vh - 110px);
  font-size: 11px; line-height: 1.7; color: #ccc;
  column-count: 2; column-gap: 36px;
}
.sf-px-help-section {
  break-inside: avoid; margin-bottom: 14px;
}
.sf-px-help-section:last-child { margin-bottom: 0; }
.sf-px-help-section h4 {
  color: var(--sf-px-accent);
  margin: 0 0 6px 0; font-size: 11px; font-weight: 700;
  letter-spacing: 0.6px; text-transform: uppercase;
}
.sf-px-help-grid {
  display: grid; grid-template-columns: max-content 1fr;
  gap: 3px 14px;
}
.sf-px-help-grid b { color: #eee; white-space: nowrap; font-weight: 600; }
.sf-px-help-grid span { color: #bbb; }
.sf-px-help-content kbd {
  background: #2a2c2e; border: 1px solid #444; border-radius: 3px;
  padding: 1px 5px; font-size: 10px; color: var(--sf-px-text);
  font-family: var(--sf-px-font-mono, monospace);
}
.sf-px-help-content b { color: #eee; }
.sf-px-help-footer {
  padding: 10px 20px; border-top: 1px solid #2a2a2a;
  font-size: 10px; color: #666; text-align: center; line-height: 1.6;
  flex-shrink: 0;
}
.sf-px-help-footer a { color: var(--sf-px-accent); text-decoration: none; }
.sf-px-help-footer a:hover { text-decoration: underline; }

/* ── Zoom controls ──────────────────────────────────── */
.sf-px-zoom-bar {
  position: absolute; bottom: 8px; left: 50%;
  transform: translateX(-50%);
  display: flex; align-items: center; gap: 4px;
  background: rgba(20,20,22,0.85); border: 1px solid var(--sf-px-border);
  border-radius: 6px; padding: 3px 6px;
}
.sf-px-zoom-label {
  font-size: 10px; color: var(--sf-px-text-dim);
  min-width: 36px; text-align: center;
}

/* ── Checkbox ───────────────────────────────────────── */
.sf-px-check-row {
  display: flex; align-items: center; gap: 6px; cursor: pointer;
  font-size: 11px; color: #ccc;
}
.sf-px-check-row input[type=checkbox] { accent-color: var(--sf-px-accent); }

/* ── Divider ────────────────────────────────────────── */
.sf-px-divider {
  height: 1px; background: var(--sf-px-border-subtle);
  margin: 6px 0; flex-shrink: 0;
}

/* ── Info text ──────────────────────────────────────── */
.sf-px-info { font-size: 10px; color: var(--sf-px-text-dim); line-height: 1.6; }
.sf-px-info b { color: #ccc; font-weight: 600; }

/* ── Canvas Frame (orange border + dimension label + gray masks) ── */
.sf-px-canvas-frame {
  position: absolute; pointer-events: none; z-index: 2;
  box-sizing: border-box;
  border: 2px solid rgba(249, 115, 22, 0.45);
}
.sf-px-canvas-frame-label {
  position: absolute; bottom: -18px; right: 0;
  font-size: 12px; color: rgba(249, 115, 22, 0.6);
  font-family: var(--sf-px-font-mono); white-space: nowrap;
  transform-origin: bottom right;
}
.sf-px-canvas-mask {
  position: absolute; pointer-events: none; z-index: 1;
  background: rgba(0, 0, 0, 0.4);
}

/* ── Canvas Settings component ──────────────────────── */
.sf-px-canvas-settings { display: flex; flex-direction: column; gap: 6px; }
.sf-px-ratio-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; }
.sf-px-ratio-btn {
  font-size: 10px; background: #1e2022; border: 1px solid var(--sf-px-border);
  color: #aaa; border-radius: 4px; padding: 5px 0; cursor: pointer;
  transition: all .12s; text-align: center; font-family: inherit;
  font-weight: 500;
}
.sf-px-ratio-btn:hover { background: #444; color: #fff; border-color: #555; }
.sf-px-ratio-btn.active {
  background: var(--sf-px-accent); border-color: var(--sf-px-accent); color: #fff;
}
.sf-px-size-row {
  display: flex; align-items: center; gap: 4px;
}
.sf-px-size-input {
  flex: 1; background: var(--sf-px-bg-input); color: var(--sf-px-text);
  border: 1px solid var(--sf-px-border); border-radius: 4px;
  padding: 5px 6px; font-size: 11px; font-family: var(--sf-px-font-mono);
  text-align: center; min-width: 0;
}
.sf-px-size-label {
  font-size: 9px; color: var(--sf-px-text-dim); width: 14px; flex-shrink: 0;
  text-align: center;
}
.sf-px-size-x { font-size: 10px; color: var(--sf-px-text-dimmer); flex-shrink: 0; }
.sf-px-swap-btn {
  width: 100%; padding: 5px; font-size: 11px; text-align: center;
  background: #1e2022; border: 1px solid var(--sf-px-border); color: #aaa;
  border-radius: 4px; cursor: pointer; transition: all .12s; font-family: inherit;
}
.sf-px-swap-btn:hover { background: var(--sf-px-accent); border-color: var(--sf-px-accent); color: #fff; }
  `);

  // ── Slider Fill System ──────────────────────────────────────
  if (!window._sfPxSliderFillInit) {
    window._sfPxSliderFillInit = true;
    window._sfPxUpdateFill = function (input) {
      const mn = parseFloat(input.min) || 0,
        mx = parseFloat(input.max) || 100;
      const v = parseFloat(input.value) || 0;
      input.style.setProperty(
        "--sf-px-fill",
        Math.max(0, Math.min(100, ((v - mn) / (mx - mn)) * 100)) + "%",
      );
    };
    document.addEventListener("input", (e) => {
      if (e.target.type === "range" && e.target.closest(".sf-px-overlay"))
        window._sfPxUpdateFill(e.target);
    });
    const desc = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    );
    const origSet = desc.set;
    desc.set = function (v) {
      origSet.call(this, v);
      if (this.type === "range" && this.closest(".sf-px-overlay"))
        window._sfPxUpdateFill(this);
    };
    Object.defineProperty(HTMLInputElement.prototype, "value", desc);
  }
}


// ── 组件（components.mjs 裁剪） ─────────────────────────────────

export function createButton(text, opts = {}) {
  const btn = document.createElement("button");
  const variantClass =
    {
      standard: "sf-px-btn",
      accent: "sf-px-btn sf-px-btn-accent",
      danger: "sf-px-btn sf-px-btn-danger",
      sm: "sf-px-btn-sm",
      icon: "sf-px-btn-icon",
      full: "sf-px-btn-full",
    }[opts.variant || "standard"] || "sf-px-btn";

  btn.className = variantClass;
  if (opts.iconSrc) {
    const img = document.createElement("img");
    img.src = opts.iconSrc;
    img.draggable = false;
    btn.appendChild(img);
  }
  if (text) btn.appendChild(document.createTextNode(text));
  if (opts.title) btn.title = opts.title;
  if (opts.onClick) btn.addEventListener("click", opts.onClick);
  return btn;
}

export function createPanel(title, opts = {}) {
  const el = document.createElement("div");
  el.className = "sf-px-panel" + (opts.collapsed ? " collapsed" : "");

  const titleEl = document.createElement("div");
  titleEl.className =
    "sf-px-panel-title" + (opts.collapsible ? " clickable" : "");

  if (opts.collapsible) {
    const arrow = document.createElement("span");
    arrow.className = "sf-px-panel-title-arrow";
    arrow.textContent = "▼";
    titleEl.appendChild(arrow);
  }

  const titleText = document.createTextNode(title);
  titleEl.appendChild(titleText);
  el.appendChild(titleEl);

  const content = document.createElement("div");
  content.className = "sf-px-panel-content";
  el.appendChild(content);

  if (opts.collapsible) {
    titleEl.addEventListener("click", () => {
      el.classList.toggle("collapsed");
    });
  }

  return {
    el,
    content,
    setCollapsed(b) {
      el.classList.toggle("collapsed", b);
    },
  };
}

export function createSliderRow(label, min, max, value, onChange, opts = {}) {
  const row = document.createElement("div");
  row.className = "sf-px-slider-row";

  const lbl = document.createElement("label");
  lbl.className = "sf-px-slider-label";
  lbl.textContent = label;
  if (opts.labelWidth) lbl.style.width = opts.labelWidth;
  row.appendChild(lbl);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = min;
  slider.max = max;
  slider.value = value;
  if (opts.step) slider.step = opts.step;

  const numInput = document.createElement("input");
  numInput.type = "number";
  numInput.min = min;
  numInput.max = max;
  numInput.value = value;
  if (opts.step) numInput.step = opts.step;

  function _syncFill() {
    const mn = parseFloat(slider.min) || 0,
      mx = parseFloat(slider.max) || 100;
    const v = parseFloat(slider.value) || 0;
    slider.style.setProperty("--sf-px-fill", ((v - mn) / (mx - mn)) * 100 + "%");
  }
  _syncFill();

  slider.addEventListener("input", () => {
    numInput.value = slider.value;
    _syncFill();
    if (onChange) onChange(parseFloat(slider.value));
  });
  numInput.addEventListener("input", () => {
    slider.value = numInput.value;
    _syncFill();
    if (onChange) onChange(parseFloat(numInput.value));
  });

  row.appendChild(slider);
  row.appendChild(numInput);

  return {
    el: row,
    slider,
    numInput,
    setValue(n) {
      slider.value = n;
      numInput.value = n;
      _syncFill();
    },
    setRange(newMin, newMax) {
      slider.min = newMin;
      slider.max = newMax;
      numInput.min = newMin;
      numInput.max = newMax;
      _syncFill();
    },
  };
}

export function createPillGrid(options, columns, onChange, opts = {}) {
  const grid = document.createElement("div");
  grid.className = "sf-px-pill-grid";
  grid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;

  const pills = [];
  let activeValue = opts.activeValue;

  options.forEach((opt) => {
    const pill = document.createElement("button");
    pill.className = "sf-px-pill" + (opt.value === activeValue ? " active" : "");
    pill.textContent = opt.label;
    pill.addEventListener("click", () => {
      activeValue = opt.value;
      pills.forEach((p, i) =>
        p.classList.toggle("active", options[i].value === activeValue),
      );
      if (onChange) onChange(activeValue);
    });
    grid.appendChild(pill);
    pills.push(pill);
  });

  return {
    el: grid,
    pills,
    setActive(value) {
      activeValue = value;
      pills.forEach((p, i) =>
        p.classList.toggle("active", options[i].value === activeValue),
      );
    },
  };
}

export function createInfo(html = "") {
  const el = document.createElement("div");
  el.className = "sf-px-info";
  el.innerHTML = html;
  return {
    el,
    setHTML(s) {
      el.innerHTML = s;
    },
  };
}

function _dangerIcon(pathD) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 64 64");
  svg.style.cssText =
    "width:14px;height:14px;flex-shrink:0;transition:fill .15s;";
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", pathD);
  path.setAttribute("fill", "#999");
  svg.appendChild(path);
  requestAnimationFrame(() => {
    const btn = svg.closest("button");
    if (btn) {
      btn.addEventListener("mouseenter", () =>
        path.setAttribute("fill", "#ffffff"),
      );
      btn.addEventListener("mouseleave", () =>
        path.setAttribute("fill", "#999"),
      );
    }
  });
  return svg;
}

// Export _dangerIcon for canvas.js


// ── Canvas 组件（canvas.mjs 裁剪） ───────────────────────────────────

const CANVAS_RATIOS = [
  { label: "Free", w: 0, h: 0 },
  { label: "1:1", w: 1, h: 1 },
  { label: "4:3", w: 4, h: 3 },
  { label: "3:2", w: 3, h: 2 },
  { label: "16:9", w: 16, h: 9 },
  { label: "4:5", w: 4, h: 5 },
  { label: "3:4", w: 3, h: 4 },
  { label: "2:3", w: 2, h: 3 },
  { label: "9:16", w: 9, h: 16 },
  { label: "5:4", w: 5, h: 4 },
];

export function createCanvasSettings(config) {
  const {
    width: initW = 1024,
    height: initH = 1024,
    ratioIndex: initRatio = 0,
    minSize = 64,
    maxSize = 8192,
    startCollapsed = true,
    onChange,
  } = config;

  let curW = initW,
    curH = initH,
    curRatio = initRatio;

  const panel = createPanel("Canvas Settings", {
    collapsible: true,
    collapsed: startCollapsed,
  });
  const wrapper = document.createElement("div");
  wrapper.className = "sf-px-canvas-settings";

  // ── Ratio buttons ──
  const ratioGrid = document.createElement("div");
  ratioGrid.className = "sf-px-ratio-grid";
  const ratioBtns = [];

  CANVAS_RATIOS.forEach((r, i) => {
    const btn = document.createElement("button");
    btn.className = "sf-px-ratio-btn" + (i === curRatio ? " active" : "");
    btn.textContent = r.label;
    btn.addEventListener("click", () => _setRatio(i));
    ratioGrid.appendChild(btn);
    ratioBtns.push(btn);
  });
  wrapper.appendChild(ratioGrid);

  // ── Width x Height row ──
  const sizeRow = document.createElement("div");
  sizeRow.className = "sf-px-size-row";

  const wLabel = document.createElement("span");
  wLabel.className = "sf-px-size-label";
  wLabel.textContent = "W";

  const wInput = document.createElement("input");
  wInput.type = "number";
  wInput.className = "sf-px-size-input";
  wInput.value = curW;
  wInput.min = minSize;
  wInput.max = maxSize;

  const xSign = document.createElement("span");
  xSign.className = "sf-px-size-x";
  xSign.textContent = "\u00d7";

  const hLabel = document.createElement("span");
  hLabel.className = "sf-px-size-label";
  hLabel.textContent = "H";

  const hInput = document.createElement("input");
  hInput.type = "number";
  hInput.className = "sf-px-size-input";
  hInput.value = curH;
  hInput.min = minSize;
  hInput.max = maxSize;

  const swapBtn = createButton("", {
    variant: "icon",
    iconSrc: ICONS["swap"],
    onClick: () => _swap(),
    title: "Swap width and height",
  });
  sizeRow.append(wLabel, wInput, xSign, hLabel, hInput, swapBtn);
  wrapper.appendChild(sizeRow);

  panel.content.appendChild(wrapper);

  // ── Internal logic ──

  function _clamp(v) {
    return Math.max(minSize, Math.min(maxSize, Math.round(v) || minSize));
  }

  function _getActiveRatio() {
    const r = CANVAS_RATIOS[curRatio];
    if (!r || r.w === 0) return 0;
    return r.w / r.h;
  }

  function _updateBtns() {
    ratioBtns.forEach((b, i) => b.classList.toggle("active", i === curRatio));
  }

  function _fire() {
    wInput.value = curW;
    hInput.value = curH;
    _updateBtns();
    if (onChange) onChange({ width: curW, height: curH, ratioIndex: curRatio });
  }

  function _setRatio(idx) {
    curRatio = idx;
    const ratio = _getActiveRatio();
    if (ratio > 0) {
      curH = _clamp(curW / ratio);
      if (Math.abs(curH / curW - 1 / ratio) > 0.01) {
        curW = _clamp(curH * ratio);
      }
    }
    _fire();
  }

  function _swap() {
    const tmp = curW;
    curW = curH;
    curH = tmp;
    const r = CANVAS_RATIOS[curRatio];
    if (r && r.w > 0) {
      const invIdx = CANVAS_RATIOS.findIndex((p) => p.w === r.h && p.h === r.w);
      if (invIdx >= 0) curRatio = invIdx;
    }
    _fire();
  }

  wInput.addEventListener("change", () => {
    curW = _clamp(parseInt(wInput.value));
    const ratio = _getActiveRatio();
    if (ratio > 0) {
      curH = _clamp(curW / ratio);
    }
    _fire();
  });

  hInput.addEventListener("change", () => {
    curH = _clamp(parseInt(hInput.value));
    const ratio = _getActiveRatio();
    if (ratio > 0) {
      curW = _clamp(curH * ratio);
    }
    _fire();
  });

  return {
    el: panel.el,
    getWidth() {
      return curW;
    },
    getHeight() {
      return curH;
    },
    getRatioIndex() {
      return curRatio;
    },
    setSize(w, h) {
      curW = _clamp(w);
      curH = _clamp(h);
      wInput.value = curW;
      hInput.value = curH;
    },
    setRatio(index) {
      _setRatio(index);
    },
    swap() {
      _swap();
    },
  };
}

export function createCanvasToolbar(config) {
  const {
    onAddImage,
    onBgColorChange,
    onClear,
    onReset,
    bgColor = "#ffffff",
    showBgColor = true,
    showClear = true,
    showReset = true,
    addImageLabel = "Add Image",
    clearLabel = "Clear Canvas",
    resetLabel = "Reset to Default",
  } = config;

  const wrapper = document.createElement("div");
  wrapper.className = "sf-px-canvas-toolbar";

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.style.display = "none";
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file && onAddImage) onAddImage(file);
    fileInput.value = "";
  });
  wrapper.appendChild(fileInput);

  const addBtn = createButton(addImageLabel, {
    variant: "full",
    iconSrc: ICONS["upload"],
    onClick: () => fileInput.click(),
    title: "Browse for an image file",
  });

  let colorInput = null;
  let _bgColor = bgColor;
  let _transparentBg = false;
  if (showBgColor) {
    const addRow = document.createElement("div");
    addRow.className = "sf-px-canvas-toolbar-row";
    addBtn.style.flex = "1";
    const label = document.createElement("span");
    label.style.cssText = "font-size:10px;color:#888;flex-shrink:0;";
    label.textContent = "BG:";
    colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = bgColor;
    colorInput.className = "sf-px-color-input";
    colorInput.style.cssText = "width:36px;height:28px;flex-shrink:0;";
    colorInput.addEventListener("input", () => {
      _bgColor = colorInput.value;
      if (onBgColorChange) onBgColorChange(colorInput.value);
    });
    addRow.append(addBtn, label, colorInput);
    wrapper.appendChild(addRow);

    const transpRow = document.createElement("label");
    transpRow.className = "sf-px-check-row";
    transpRow.title = "Save to Disk with transparent background (no background color)";
    transpRow.style.cssText = "margin:4px 0 0 2px;font-size:11px;opacity:0.85;";
    const transpCb = document.createElement("input");
    transpCb.type = "checkbox";
    transpCb.addEventListener("change", () => { _transparentBg = transpCb.checked; });
    transpRow.appendChild(transpCb);
    transpRow.append("Transparent BG (Save to Disk)");
    wrapper.appendChild(transpRow);
  } else {
    wrapper.appendChild(addBtn);
  }

  if ((showClear && onClear) || (showReset && onReset)) {
    const dangerRow = document.createElement("div");
    dangerRow.className = "sf-px-canvas-toolbar-row";
    dangerRow.style.cssText = "gap:4px;";

    if (showClear && onClear) {
      const clearBtn = createButton(clearLabel, {
        variant: "full",
        onClick: onClear,
        title: "Clear all content",
      });
      clearBtn.classList.add("sf-px-btn-danger");
      clearBtn.style.flex = "1";
      clearBtn.insertBefore(
        _dangerIcon(
          "M11.4,21.4h41.2l-5.1,38.2c-.3,1.9-1.9,3.3-3.9,3.3h-23.2c-1.9,0-3.6-1.4-3.9-3.3l-5.1-38.2ZM50.1,6.9h-13v-2.9c0-1.2-1-2.1-2.1-2.1h-6c-1.2,0-2.1,1-2.1,2.1v2.9h-13c-3.9.2-7,3.5-7,7.4v3h50.3v-3c0-3.9-3.1-7.2-7-7.4Z",
        ),
        clearBtn.firstChild,
      );
      dangerRow.appendChild(clearBtn);
    }

    if (showReset && onReset) {
      const resetBtn = createButton(resetLabel, {
        variant: "full",
        onClick: onReset,
        title: "Reset all settings to default",
      });
      resetBtn.classList.add("sf-px-btn-danger");
      resetBtn.style.flex = "1";
      resetBtn.insertBefore(
        _dangerIcon(
          "M5.1,36.2h8c-.1,8,5.1,15,12.2,17.7,7.8,2.9,16.4.6,21.5-5.8,3.3-4.1,4.6-9.2,4-14.4-1-8.6-7.8-15.3-16.4-16.4v6.5c0,.6-.6,1.3-1.1,1.4-.5.2-1.5.2-1.9-.2l-12-10.2c-.6-.5-.8-1.1-.8-1.9,0-.7.4-1.3,1-1.8l11.6-9.9c.6-.5,1.4-.6,2.1-.3.5.2,1,.9,1,1.6v6.4c4.6.5,9,1.9,12.8,4.5,6.5,4.5,10.6,11.2,11.6,19,.3,2.7.4,5,0,7.6-.9,6.2-3.9,12-8.4,16.2-12.2,11.1-30.9,8.9-40.4-4.6-3.1-4.4-4.8-9.7-4.8-15.5ZM38.7,41.7v-9.2c0-1.1-.7-1.9-1.7-2.2h-10.1c-1,.2-1.7,1.1-1.7,2.1v9.3c0,1.2.9,2.1,2.1,2.1h9.1c1.2,0,2.3-1,2.3-2.2Z",
        ),
        resetBtn.firstChild,
      );
      dangerRow.appendChild(resetBtn);
    }

    wrapper.appendChild(dangerRow);
  }

  function setupDropZone(workspace) {
    if (!workspace || !onAddImage) return;

    const overlay = document.createElement("div");
    overlay.className = "sf-px-drop-overlay";
    overlay.innerHTML = '<span class="sf-px-drop-label">Drop image here</span>';
    workspace.appendChild(overlay);

    let dragCounter = 0;
    workspace.addEventListener("dragenter", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter++;
      if (e.dataTransfer?.types?.includes("Files"))
        overlay.classList.add("active");
    });
    workspace.addEventListener("dragleave", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        overlay.classList.remove("active");
      }
    });
    workspace.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.stopPropagation();
      e.dataTransfer.dropEffect = "copy";
    });
    workspace.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter = 0;
      overlay.classList.remove("active");
      const file = e.dataTransfer?.files?.[0];
      if (file && file.type.startsWith("image/")) onAddImage(file);
    });

    const overlayEl = workspace.closest(".sf-px-overlay");
    if (overlayEl) {
      ["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
        overlayEl.addEventListener(evt, (e) => {
          e.preventDefault();
          e.stopPropagation();
        });
      });
    }

    const _pasteHandler = (e) => {
      if (!overlayEl?.isConnected) {
        window.removeEventListener("paste", _pasteHandler, true);
        return;
      }
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          e.preventDefault();
          e.stopPropagation();
          const file = item.getAsFile();
          if (file) onAddImage(file);
          break;
        }
      }
    };
    window.addEventListener("paste", _pasteHandler, true);
  }

  return {
    el: wrapper,
    fileInput,
    get transparentBg() { return _transparentBg; },
    setBgColor(hex) {
      _bgColor = hex;
      if (colorInput) colorInput.value = hex;
    },
    getBgColor() {
      return _bgColor;
    },
    setupDropZone,
  };
}


// ── 编辑器布局（layout.mjs，去 pixaroma 品牌） ───────────────────────

export function createEditorLayout(config) {
  injectFrameworkStyles();

  const {
    editorName = "Editor",
    editorId,
    leftWidth = 260,
    rightWidth = 260,
    showUndoRedo = true,
    showZoomBar = true,
    showStatusBar = true,
    showTopOptionsBar = false,
    onSave,
    onClose,
    onUndo,
    onRedo,
    onZoomIn,
    onZoomOut,
    onZoomFit,
    helpContent = "",
    helpTitle = "",   // optional; defaults to "<editorName> — Shortcuts" (back-compat)
  } = config;

  // ── Overlay ──
  const overlay = document.createElement("div");
  overlay.className = "sf-px-overlay";
  if (editorId) overlay.id = editorId;

  // ── Titlebar ──
  const titlebar = document.createElement("div");
  titlebar.className = "sf-px-titlebar";

  const title = document.createElement("span");
  title.className = "sf-px-title";
  title.textContent = ` ${editorName} `;
  titlebar.appendChild(title);

  // Center slot (editors can add tools here, e.g. align bar)
  const titlebarCenter = document.createElement("div");
  titlebarCenter.className = "sf-px-titlebar-center";
  titlebar.appendChild(titlebarCenter);

  // Right actions: zoom + undo/redo
  const actions = document.createElement("div");
  actions.className = "sf-px-titlebar-actions";

  // Zoom controls in titlebar
  let zoomBarEl = null,
    zoomLabelEl = null;
  if (showZoomBar) {
    const zoomWrap = document.createElement("div");
    zoomWrap.className = "sf-px-titlebar-zoom";
    const zoomOut = createButton("", {
      variant: "sm",
      title: "Zoom out",
      iconSrc: ICONS["minus"],
      onClick: () => {
        if (onZoomOut) onZoomOut();
      },
    });
    const zoomFit = createButton("Fit", {
      variant: "accent",
      title: "Fit to view",
      onClick: () => {
        if (onZoomFit) onZoomFit();
      },
    });
    zoomLabelEl = document.createElement("span");
    zoomLabelEl.className = "sf-px-zoom-label";
    zoomLabelEl.textContent = "100%";
    const zoomIn = createButton("", {
      variant: "sm",
      title: "Zoom in",
      iconSrc: ICONS["plus"],
      onClick: () => {
        if (onZoomIn) onZoomIn();
      },
    });
    zoomWrap.append(zoomOut, zoomFit, zoomLabelEl, zoomIn);
    actions.appendChild(zoomWrap);
    zoomBarEl = zoomWrap;

    // Separator between zoom and undo/redo
    const sep = document.createElement("div");
    sep.className = "sf-px-titlebar-sep";
    sep.style.cssText = "margin-left: 15px; margin-right: 15px;";
    actions.appendChild(sep);
  }

  // Undo / Redo buttons
  let undoBtn = null,
    redoBtn = null;
  if (showUndoRedo) {
    undoBtn = createButton("Undo", {
      variant: "accent",
      iconSrc: ICONS["undo"],
      title: "Undo (Ctrl+Z)",
      onClick: onUndo,
    });
    redoBtn = createButton("Redo", {
      variant: "accent",
      iconSrc: ICONS["redo"],
      title: "Redo (Ctrl+Shift+Z)",
      onClick: onRedo,
    });
    undoBtn.style.cssText = "padding:5px 14px;font-size:12px;";
    redoBtn.style.cssText = "padding:5px 14px;font-size:12px;";
    actions.append(undoBtn, redoBtn);
  }

  // Header close button (close without saving)
  const headerCloseBtn = createButton(`✕ Close ${editorName}`, {
    variant: "danger",
    title: `Close ${editorName} (does not close ComfyUI)`,
    onClick: () => {
      if (onClose) onClose();
    },
  });
  headerCloseBtn.style.cssText = "padding:5px 12px;font-size:12px;font-weight:bold;margin-left:8px;";
  actions.appendChild(headerCloseBtn);

  titlebar.appendChild(actions);
  overlay.appendChild(titlebar);

  // ── Top options bar ──
  let topOptionsBar = null;
  if (showTopOptionsBar) {
    topOptionsBar = document.createElement("div");
    topOptionsBar.className = "sf-px-top-options";
    overlay.appendChild(topOptionsBar);
  }

  // ── Body ──
  const body = document.createElement("div");
  body.className = "sf-px-body";

  // Left sidebar
  const leftSidebar = document.createElement("div");
  leftSidebar.className = "sf-px-sidebar sf-px-sidebar-left";
  leftSidebar.style.width = leftWidth + "px";
  body.appendChild(leftSidebar);

  // Workspace
  const workspace = document.createElement("div");
  workspace.className = "sf-px-workspace";

  // Help overlay in workspace
  const helpPanel = document.createElement("div");
  helpPanel.className = "sf-px-help-overlay";
  if (helpContent) {
    helpPanel.innerHTML = `
      <div class="sf-px-help-header">
        <h3>${helpTitle || `${editorName} — Shortcuts`}</h3>
        <button class="sf-px-btn-sm" style="flex-shrink:0;">✕</button>
      </div>
      <div class="sf-px-help-content">${helpContent}</div>
      <div class="sf-px-help-footer">SF Nodes</div>
    `;
    helpPanel
      .querySelector(".sf-px-help-header button")
      .addEventListener("click", () => {
        helpPanel.style.display = "none";
      });
  }
  workspace.appendChild(helpPanel);

  // Zoom bar reference (lives in titlebar, not workspace)
  let zoomBar = zoomBarEl,
    zoomLabel = null;

  body.appendChild(workspace);

  // Right sidebar
  const rightSidebar = document.createElement("div");
  rightSidebar.className = "sf-px-sidebar sf-px-sidebar-right";
  rightSidebar.style.width = (rightWidth || 220) + "px";
  body.appendChild(rightSidebar);

  overlay.appendChild(body);

  // ── Tool info (floating tooltip in workspace, bottom-left) ──
  const statusText = document.createElement("div");
  statusText.className = "sf-px-tool-info";
  workspace.appendChild(statusText);
  const statusBar = null;

  // ── Footer (Save / Close / Help) — always bottom of right sidebar ──
  const sidebarFooter = document.createElement("div");
  sidebarFooter.className = "sf-px-sidebar-footer";

  const helpBtn = createButton("Help", {
    variant: "standard",
    iconSrc: ICONS["help"],
    onClick: () => toggleHelp(),
  });
  helpBtn.style.width = "100%";

  const footerBtnRow = document.createElement("div");
  footerBtnRow.className = "sf-px-btn-row";

  const saveBtn = createButton("Save", {
    variant: "accent",
    iconSrc: ICONS["save"],
    onClick: onSave,
  });
  saveBtn.style.flex = "1";
  const closeBtn = createButton("Save to Disk", {
    variant: "standard",
    iconSrc: ICONS["download"],
    title: "Save image to disk",
    onClick: () => {
      if (layout.onSaveToDisk) layout.onSaveToDisk();
    },
  });
  closeBtn.style.flex = "1";

  footerBtnRow.append(saveBtn, closeBtn);
  sidebarFooter.append(helpBtn, footerBtnRow);
  rightSidebar.appendChild(sidebarFooter);

  // ── Methods ──
  function toggleHelp() {
    helpPanel.style.display =
      helpPanel.style.display === "block" ? "none" : "block";
  }

  const layout = {
    overlay,
    titlebar,
    titlebarCenter,
    topOptionsBar,
    body,
    leftSidebar,
    workspace,
    rightSidebar,
    sidebarFooter,
    statusBar,
    statusText,
    helpPanel,
    undoBtn,
    redoBtn,
    saveBtn,
    closeBtn,
    zoomBar,
    zoomLabel,

    mount() {
      document.body.appendChild(overlay);
      installFocusTrap(overlay);
      // Block ALL keyboard events from reaching ComfyUI while editor is open
      layout._kbBlock = (e) => {
        e.stopPropagation();
      };
      window.addEventListener("keydown", layout._kbBlock, { capture: true });
      window.addEventListener("keyup", layout._kbBlock, { capture: true });
      window.addEventListener("keypress", layout._kbBlock, { capture: true });
      requestAnimationFrame(() => {
        overlay.querySelectorAll("input[type=range]").forEach((s) => {
          if (window._sfPxUpdateFill) window._sfPxUpdateFill(s);
        });
      });
    },
    unmount() {
      if (layout._kbBlock) {
        window.removeEventListener("keydown", layout._kbBlock, {
          capture: true,
        });
        window.removeEventListener("keyup", layout._kbBlock, { capture: true });
        window.removeEventListener("keypress", layout._kbBlock, {
          capture: true,
        });
      }
      if (layout.onCleanup) layout.onCleanup();
      overlay.remove();
    },
    onCleanup: null,
    onSaveToDisk: null,
    setStatus(text, type) {
      if (statusText) {
        statusText.textContent = text;
        statusText.classList.remove("warn", "error");
        if (type === "warn") statusText.classList.add("warn");
        else if (type === "error") statusText.classList.add("error");
      }
    },
    setUndoState({ canUndo, canRedo }) {
      if (undoBtn) undoBtn.disabled = !canUndo;
      if (redoBtn) redoBtn.disabled = !canRedo;
    },
    toggleHelp,
    setZoomLabel(text) {
      if (zoomLabelEl) zoomLabelEl.textContent = text;
    },
    setSaving() {
      if (saveBtn) {
        saveBtn.disabled = true;
        saveBtn.textContent = "";
        saveBtn.appendChild(_uiIcon("save.svg"));
        saveBtn.appendChild(document.createTextNode("Saving..."));
      }
      layout.setStatus("Saving...");
    },
    setSaved(autoClose = true) {
      if (saveBtn) {
        saveBtn.disabled = false;
        saveBtn.textContent = "";
        saveBtn.appendChild(_uiIcon("save.svg"));
        saveBtn.appendChild(document.createTextNode("Saved!"));
      }
      layout.setStatus("Saved!");
      if (autoClose) setTimeout(() => layout.unmount(), 500);
    },
    setSaveError(msg) {
      if (saveBtn) {
        saveBtn.disabled = false;
        saveBtn.textContent = "";
        saveBtn.appendChild(_uiIcon("save.svg"));
        saveBtn.appendChild(document.createTextNode("Save"));
      }
      layout.setStatus(msg || "Save failed", "error");
    },
  };

  return layout;
}


// ── 工具函数（utils.mjs 内联） ──────────────────────────────────────

export const allow_debug = false;

export function installFocusTrap(overlay) {
  const trap = document.createElement("textarea");
  trap.dataset.sfCropTrap = "1";
  trap.setAttribute("aria-hidden", "true");
  trap.style.cssText =
    "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;z-index:-1;";
  overlay.appendChild(trap);
  trap.focus();
  const refocus = (e) => {
    const t = e.target;
    const tag = t?.tagName;
    if (t?.isContentEditable || t?.closest?.('[contenteditable="true"]')) return;
    if (tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") {
      requestAnimationFrame(() => trap.focus());
    }
  };
  overlay.addEventListener("mouseup", refocus);
  return trap;
}

// 下载 dataURL 为文件：实现收敛于 sf_common.js（showSaveFilePicker 优先 +
// <a download> 回退）。此处 re-export 保持调用方（sf_crop.js / sf_inpaint.js）
// 零改动。
export { downloadDataURL } from "./sf_common.js";

export function createDummyWidget(titleText, subtitleText, instructionText) {
  const container = document.createElement("div");
  container.style.cssText = `
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 4px;
      padding: 20px;
      background-color: #121212;
      border-radius: 8px;
      width: 100%;
      height: 100%;
      color: #ffffff;
      font-family: sans-serif;
      text-align: center;
      box-sizing: border-box;
    `;
  const title = document.createElement("div");
  title.innerText = titleText;
  title.style.cssText = `
      font-size: 22px;
      font-weight: 700;
      margin: 0;
      line-height: 1.2;
    `;
  container.appendChild(title);
  const subtitle = document.createElement("div");
  subtitle.innerText = subtitleText;
  subtitle.style.cssText = `
      font-size: 18px;
      font-weight: 700;
      color: var(--sf-acc, #f66744);
      margin: 0;
      line-height: 1.2;
    `;
  container.appendChild(subtitle);
  const instruction = document.createElement("div");
  instruction.innerText = instructionText;
  instruction.style.cssText = `
      font-size: 10px;
      color: #555555;
      margin-top: 12px;
      white-space: pre-line;
      line-height: 1.4;
    `;
  container.appendChild(instruction);
  return container;
}
