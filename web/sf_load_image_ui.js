// SF Load Image Resize — DOM UI (ported from comfyui-pixaroma js/load_image/ui.mjs
// + js/load_image/panel_polish.mjs). No pixaroma-plugin dependencies: brand
// colour is fixed, icons are inline data URIs, settings use ComfyUI's own
// app.ui.settings.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { updateNativePreview, setSelectedImage, splitFilenameSubfolder } from "./sf_load_image_api.js";

// 图标内联 data URI（本项目无资产服务路由，惯例见 sf_workflows_ui.js）
const ICON_UPLOAD = "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M58.115,1.482H5.885C3.239,1.482,1.094,3.627,1.094,6.273v51.453c0,2.646,2.145,4.791,4.791,4.791h52.23c2.646,0,4.791-2.145,4.791-4.791V6.273c0-2.646-2.145-4.791-4.791-4.791ZM49.641,28.696h-11.702v24.054c0,1.147-.93,2.077-2.077,2.077h-7.726c-1.147,0-2.077-.93-2.077-2.077v-24.054h-11.702c-2.409,0-3.487-3.024-1.62-4.547l17.641-14.398c.472-.384,1.046-.577,1.62-.577s1.149.193,1.62.577l17.641,14.398c1.867,1.523.789,4.547-1.62,4.547Z"/></svg>');
const ICON_SWAP = "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M13.622,36.23L.925,20.673c-.339-.416-.509-.922-.509-1.429s.17-1.013.509-1.429L13.622,2.259c1.344-1.646,4.01-.696,4.01,1.429v10.319h41.824c1.027,0,1.86.833,1.86,1.86v6.756c0,1.027-.833,1.86-1.86,1.86H17.632v10.319c0,2.125-2.666,3.075-4.01,1.429ZM63.491,43.327l-12.697-15.557c-1.344-1.646-4.01-.696-4.01,1.429v10.319H4.96c-1.027,0-1.86.833-1.86,1.86v6.756c0,1.027.833,1.86,1.86,1.86h41.824v10.319c0,2.125,2.667,3.075,4.01,1.429l12.697-15.557c.339-.416.509-.922.509-1.429s-.17-1.013-.509-1.429Z"/></svg>');
const ICON_MAGNET = "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M47.374,44.366v16.382h-17.717c-8.086-.123-15.546-3.579-20.886-9.379C-2.891,38.703-.815,18.774,13.166,8.738c4.612-3.311,10.005-5.282,15.879-5.494h18.329v16.358h-17.768c-3.033.264-5.706,1.331-7.901,3.368-3.232,2.999-4.652,7.339-3.698,11.679,1.183,5.378,5.755,9.212,11.26,9.717h18.107ZM60.934,60.745c1.159-.073,1.924-.914,1.861-2.07v-12.166c.087-1.093-.71-2.148-1.86-2.15h-9.945v16.386h9.944ZM61.252,19.605c.997-.198,1.615-1.05,1.545-2.064V5.265c.026-1.005-.7-2.027-1.785-2.03h-10.021v16.37h10.262Z"/></svg>');

// 绝对安全的 URL：api.apiURL 处理托管部署基址，失败降级原样返回
function pixApiUrl(route) {
  try {
    if (typeof api?.apiURL === "function") return api.apiURL(route);
  } catch {
    /* 降级 */
  }
  return route;
}

let _cssInjected = false;

export function injectCSS() {
  if (_cssInjected) return;
  _cssInjected = true;
  const css = `
    .sf-li-root {
      width: 100%;
      box-sizing: border-box;
      position: relative;
      background: #2a2a2a;
      border-radius: 4px;
      color: #ddd;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 11px;
    }
    /* The flex column lives on an inner layer that fills the root (absolute
       inset:0). ComfyUI forces the widget ROOT to display:block on rebuild /
       collapse, which would kill a flex column ON the root (the 7px gaps
       collapse and the flex:1 image canvas drops to its min height, then
       visibly grows back when flex is restored - the flicker). The inner is
       never touched by ComfyUI, so the layout is ALWAYS flex: no transition,
       no flicker. Padding lives here so the absolute inner respects it. */
    .sf-li-inner {
      position: absolute;
      inset: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 7px;
      /* Small top padding pulls the Upload button up tight under the output
         dots (the body can't sit higher than the slot area). */
      padding: 2px 8px 8px;
      box-sizing: border-box;
    }
    .sf-li-upload-btn {
      width: 100%;
      background: #f66744;
      border: none;
      border-radius: 4px;
      padding: 7px 8px;
      font-size: 11px;
      color: #fff;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      font-family: inherit;
      transition: background 0.08s;
    }
    .sf-li-upload-btn:hover { background: #ff7e5a; }
    .sf-li-upload-btn .ico {
      width: 14px; height: 14px;
      background-color: currentColor;
      -webkit-mask: url("${ICON_UPLOAD}") center/14px 14px no-repeat;
              mask: url("${ICON_UPLOAD}") center/14px 14px no-repeat;
    }
    .sf-li-hint {
      font-size: 9px;
      color: #777;
      text-align: center;
      letter-spacing: 0.3px;
      margin-top: -3px;
    }
    .sf-li-hint kbd {
      color: #aaa;
      font-family: inherit;
      background: transparent;
      padding: 0;
    }
    /* File row: [◀] [ dropdown ] [▶] - arrow buttons let the user flip
       through images visually, matching native ComfyUI LoadImage. */
    .sf-li-filerow {
      display: flex;
      gap: 6px;
      align-items: stretch;
    }
    .sf-li-filerow .sf-li-dropdown { flex: 1; min-width: 0; }
    /* File nav arrows match the resample picker exactly (orange, 30px, solid). */
    .sf-li-nav {
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      color: #f66744;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      width: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
      transition: background 0.08s, border-color 0.08s, color 0.08s;
      flex-shrink: 0;
    }
    .sf-li-nav:hover:not(.disabled) { border-color: #f66744; }
    .sf-li-nav.disabled { opacity: 0.3; cursor: default; }
    .sf-li-dropdown {
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      padding: 6px 10px;
      font-size: 11px;
      color: #ccc;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
    }
    .sf-li-dropdown:hover { border-color: #f66744; }
    .sf-li-dropdown .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sf-li-dropdown .arrow { color: #f66744; font-size: 13px; margin-left: 6px; line-height: 1; }
    .sf-li-dropdown .counter {
      color: #777;
      font-size: 9px;
      margin-left: 6px;
      flex-shrink: 0;
    }
    .sf-li-chips {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 5px;
    }
    .sf-li-chip {
      box-sizing: border-box;
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      padding: 6px 3px;
      text-align: center;
      font-size: 9.5px;
      color: #ccc;
      cursor: pointer;
      user-select: none;
      transition: background 0.08s, border-color 0.08s;
    }
    .sf-li-chip:hover { border-color: #f66744; color: #ddd; }
    .sf-li-chip.active {
      background: #f66744;
      color: #fff;
      border-color: #f66744;
    }
    .sf-li-panel {
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      padding: 8px 10px;
    }
    .sf-li-panel-row { display: flex; align-items: center; gap: 8px; }
    .sf-li-panel-label {
      font-size: 9px;
      color: #f66744;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }
    .sf-li-panel input[type="range"] {
      flex: 1;
      accent-color: #f66744;
    }
    .sf-li-panel input[type="text"], .sf-li-panel input[type="number"] {
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 3px;
      padding: 4px 6px;
      color: #f66744;
      font-size: 12px;
      font-weight: 600;
      text-align: center;
      font-family: inherit;
      box-sizing: border-box;
    }
    .sf-li-panel input[type="text"]:focus, .sf-li-panel input[type="number"]:focus {
      outline: none;
      border-color: #f66744;
    }
    .sf-li-panel-readout {
      font-size: 9px;
      color: #888;
      font-family: inherit;
      text-align: center;
      margin-top: 6px;
    }
    .sf-li-quickpicks {
      display: grid;
      gap: 3px;
      margin-bottom: 8px;
    }
    .sf-li-quickpick {
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 3px;
      color: #aaa;
      padding: 4px 0;
      text-align: center;
      font-size: 10px;
      cursor: pointer;
      font-family: inherit;
    }
    .sf-li-quickpick:hover { border-color: #666; color: #ddd; }
    .sf-li-quickpick.active {
      background: #f66744;
      color: #fff;
      border-color: #f66744;
    }
    .sf-li-value {
      font-family: inherit;
      font-size: 12px;
      color: #f66744;
      font-weight: 600;
      min-width: 50px;
      text-align: right;
    }
    .sf-li-ratio-chips {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 3px;
      margin-bottom: 8px;
    }
    .sf-li-ratio-chip {
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 3px;
      padding: 4px 0;
      text-align: center;
      font-size: 9px;
      color: #aaa;
      cursor: pointer;
      font-family: inherit;
    }
    .sf-li-ratio-chip:hover { border-color: #666; color: #ddd; }
    .sf-li-ratio-chip.active {
      background: #f66744;
      color: #fff;
      border-color: #f66744;
    }
    .sf-li-cropped {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 3px;
      overflow: hidden;
      margin-bottom: 6px;
    }
    .sf-li-cropped > div {
      text-align: center;
      font-size: 10px;
      padding: 5px 0;
      color: #aaa;
      cursor: pointer;
      user-select: none;
    }
    .sf-li-cropped > div.active { background: #f66744; color: #fff; }
    .sf-li-pad-row {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 10px;
      color: #888;
    }
    .sf-li-pad-swatch {
      width: 22px; height: 22px;
      border-radius: 3px;
      border: 1px solid #444;
      cursor: pointer;
    }
    .sf-li-custom-ratio-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      margin-bottom: 6px;
    }
    /* Custom ratio inputs sit inside a .sf-li-numinput wrapper — the
       wrapper supplies border/background; we just fix the width. */
    .sf-li-custom-ratio-input-wrap { width: 64px; }
    .sf-li-custom-ratio-swap {
      width: 24px;
      height: 22px;
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 3px;
      color: #aaa;
      cursor: pointer;
      position: relative;
      padding: 0;
      display: inline-block;
    }
    .sf-li-custom-ratio-swap::before {
      content: "";
      position: absolute;
      inset: 0;
      background-color: currentColor;
      -webkit-mask: url("${ICON_SWAP}") center/14px 14px no-repeat;
              mask: url("${ICON_SWAP}") center/14px 14px no-repeat;
      pointer-events: none;
    }
    .sf-li-custom-ratio-swap:hover { color: #f66744; border-color: #f66744; }
    /* Center single-input panel rows (Max MP / Longest side / Scale by ×). */
    .sf-li-panel-row.sf-li-centered { justify-content: center; }
    .sf-li-input-wide {
      width: 70% !important;
      max-width: 200px;
    }
    /* makeNumericInput wrapper — flex row with input + stacked +/- spinners. */
    .sf-li-numinput {
      display: inline-flex;
      align-items: stretch;
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 4px;
      overflow: hidden;
      box-sizing: border-box;
    }
    .sf-li-numinput:focus-within { border-color: #f66744; }
    .sf-li-numinput input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      padding: 2px 6px;
      color: #f66744;
      font-size: 11px;
      font-weight: 600;
      text-align: center;
      font-family: inherit;
      width: 100%;
      min-width: 0;
    }
    .sf-li-spin {
      display: flex;
      flex-direction: column;
      width: 12px;
      border-left: 1px solid #444;
    }
    .sf-li-spin > button {
      flex: 1;
      background: #232323;
      border: none;
      padding: 0;
      cursor: pointer;
      color: #aaa;
      font-size: 8px;
      line-height: 1;
      position: relative;
    }
    .sf-li-spin > button:hover { background: #333; color: #f66744; }
    .sf-li-spin-up { border-bottom: 1px solid #444; }
    /* CSS chevron arrows (no extra SVG needed). */
    .sf-li-spin-up::before,
    .sf-li-spin-down::before {
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      width: 6px;
      height: 6px;
      transform: translate(-50%, -50%) rotate(-45deg);
      border-top: 1px solid currentColor;
      border-right: 1px solid currentColor;
    }
    .sf-li-spin-up::before {
      transform: translate(-50%, -25%) rotate(-45deg);
    }
    .sf-li-spin-down::before {
      transform: translate(-50%, -75%) rotate(135deg);
    }
    /* Width × Height panels (Fit inside, Crop to fill) with swap between. */
    .sf-li-wh-row {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 6px;
      align-items: end;
    }
    .sf-li-wh-field {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .sf-li-wh-label {
      font-size: 9px;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      text-align: center;
    }
    /* W/H input is inside a .sf-li-numinput wrap — wrap provides
       border/background. The default 11px input font is fine. */
    .sf-li-wh-input-wrap { width: 100%; }
    /* Generic swap button used between W and H inputs. Height matches
       the trimmed .sf-li-numinput control height (~22 px). */
    .sf-li-swap {
      width: 26px;
      height: 22px;
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 4px;
      color: #aaa;
      cursor: pointer;
      padding: 0;
      position: relative;
      align-self: end;
    }
    .sf-li-swap::before {
      content: "";
      position: absolute;
      inset: 0;
      background-color: currentColor;
      -webkit-mask: url("${ICON_SWAP}") center/12px 12px no-repeat;
              mask: url("${ICON_SWAP}") center/12px 12px no-repeat;
      pointer-events: none;
    }
    .sf-li-swap:hover { color: #f66744; border-color: #f66744; }
    /* Aspect-ratio preview block under W / H fields. */
    .sf-li-wh-preview {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      margin-top: 8px;
    }
    .sf-li-wh-rect {
      background: color-mix(in srgb, #f66744 18%, transparent);
      border: 1px solid #f66744;
      border-radius: 2px;
      transition: width 0.12s ease, height 0.12s ease;
    }
    .sf-li-wh-rect-label {
      font-size: 9px;
      color: #999;
      font-family: inherit;
    }
    /* Tiny aspect-ratio shape rendered INSIDE each Match-ratio chip,
       same idea Resolution Pixaroma uses to make every ratio recognisable
       at a glance without reading the label. */
    .sf-li-ratio-chip {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
    }
    .sf-li-shape {
      display: inline-block;
      background: rgba(180,180,180,0.25);
      border: 1px solid #888;
      border-radius: 1px;
      box-sizing: border-box;
      flex-shrink: 0;
    }
    .sf-li-ratio-chip.active .sf-li-shape {
      background: rgba(255,255,255,0.4);
      border-color: rgba(255,255,255,0.85);
    }
    /* Custom chip has no shape (no fixed aspect) — keep text-only. */
    .sf-li-ratio-chip.sf-li-ratio-custom-chip { display: block; }
    .sf-li-global {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .sf-li-rs-popup {
      position: fixed;
      z-index: 99999;
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      font-size: 11px;
      color: #ccc;
      min-width: 200px;
      overflow: hidden;
    }
    .sf-li-rs-item {
      padding: 6px 10px;
      cursor: pointer;
      border-bottom: 1px solid #2a2a2a;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .sf-li-rs-item:last-child { border-bottom: none; }
    .sf-li-rs-item:hover { background: #2a2a2a; }
    .sf-li-rs-item.active .sf-li-rs-item-label { color: #f66744; font-weight: 600; }
    .sf-li-rs-item-label { font-size: 11px; }
    .sf-li-rs-item-hint { font-size: 9px; color: #777; }
    /* ── Image Resize design language, scoped to .sf-li-root ── */
    /* Centered snap footer: magnet + "Snap" + chips. */
    .sf-li-foot { display:flex; align-items:center; justify-content:center; gap:6px; flex-wrap:wrap; }
    .sf-li-snap2 { display:inline-flex; align-items:center; gap:5px; }
    .sf-li-snap-icon { display:inline-block; width:12px; height:12px; background-color:#888; flex:none;
      -webkit-mask:url("${ICON_MAGNET}") center/12px 12px no-repeat;
              mask:url("${ICON_MAGNET}") center/12px 12px no-repeat; }
    .sf-li-snap-lbl { font-size:9px; color:#7d7d7d; text-transform:uppercase; letter-spacing:.5px; }
    .sf-li-schip { background:#1d1d1d; border:1px solid #444; border-radius:3px; color:#aaa;
      font-size:8.5px; padding:3px 5px; min-width:16px; text-align:center; cursor:pointer; user-select:none; }
    .sf-li-schip:hover { border-color:#f66744; color:#ddd; }
    .sf-li-schip.active { background:#f66744; color:#fff; border-color:#f66744; }
    /* Resample picker: [◀] [ Resample: Auto ▾ ] [▶] */
    .sf-li-rs2-row { display:flex; align-items:stretch; gap:6px; }
    .sf-li-rs2-nav { flex:0 0 30px; background:#1d1d1d; border:1px solid #444; border-radius:4px;
      color:#f66744; font-size:11px; cursor:pointer; display:flex; align-items:center; justify-content:center; padding:0; }
    .sf-li-rs2-nav:hover { border-color:#f66744; }
    .sf-li-rs2-dd { flex:1; display:flex; align-items:center; justify-content:space-between;
      background:#1d1d1d; border:1px solid #444; border-radius:4px; padding:6px 10px; cursor:pointer; user-select:none; }
    .sf-li-rs2-dd:hover { border-color:#f66744; }
    .sf-li-rs2-value { color:#ddd; font-size:11px; }
    .sf-li-rs2-arrow { color:#f66744; font-size:13px; margin-left:6px; line-height:1; }
    /* Upscaling toggle button. */
    .sf-li-upbtn { align-self:center; background:#1d1d1d; border:1px solid #444; border-radius:5px;
      color:#aaa; font-size:11px; padding:7px 18px; cursor:pointer; user-select:none; transition:background .08s,border-color .08s; }
    .sf-li-upbtn:hover { border-color:#f66744; color:#ddd; }
    .sf-li-upbtn.is-on, .sf-li-upbtn.is-on:hover { background:#f66744; border-color:#f66744; color:#fff; }
    /* Per-mode panel overrides (mirror image_resize .pix-ir-root .sf-li-* block). */
    .sf-li-root .sf-li-panel { background:rgba(255,255,255,.04); border:none; border-radius:6px; padding:9px 10px; }
    .sf-li-root .sf-li-panel-readout { display:none; }
    .sf-li-root .sf-li-ratio-chips { margin-bottom:0; }
    .sf-li-root .sf-li-custom-ratio-row { margin:8px 0 0; }
    .sf-li-root .sf-li-input-wide { width:100% !important; max-width:none; }
    .sf-li-root .sf-li-numinput { background:#1d1d1d !important; align-items:center; min-height:28px; }
    .sf-li-root .sf-li-numinput .sf-li-spin { align-self:stretch; }
    .sf-li-root .sf-li-numinput input { line-height:1.2; background:transparent !important; border:none !important; border-radius:0 !important; }
    .sf-li-root .sf-li-inline-label { display:flex; align-items:center; color:#f66744; font-size:9px; font-weight:600;
      text-transform:uppercase; letter-spacing:.5px; padding:0 4px 0 9px; white-space:nowrap; flex:none; }
    .sf-li-root .sf-li-num-labeled input { text-align:right !important; padding-right:8px !important; }
    .sf-li-root .sf-li-swap { background:#1d1d1d !important; }
    .sf-li-root .sf-li-wh-header { text-align:center !important; color:#d6d6d6 !important; }
    .sf-li-root .sf-li-wh-rect { background:color-mix(in srgb, #f66744 35%, transparent); border-width:2px; }
    .sf-li-root .sf-li-wh-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; align-items:center; }
    .sf-li-root .sf-li-wh-col { display:flex; flex-direction:column; gap:6px; min-width:0; }
    .sf-li-root .sf-li-wh-col .sf-li-swap { width:100%; height:24px; align-self:auto; }
    .sf-li-root .sf-li-wh-grid .sf-li-wh-preview { margin-top:0; justify-content:center; }
    /* Filled triangle spinner glyphs (replace shared outline chevrons). NOTE:
       use the literal triangle characters - a backslash-escape inside a JS
       template literal throws (CLAUDE.md UI Pattern #12). */
    .sf-li-root .sf-li-spin { width:16px; border-left:none; }
    .sf-li-root .sf-li-spin > button { background:transparent; }
    .sf-li-root .sf-li-spin-up::before, .sf-li-root .sf-li-spin-down::before {
      border:none; width:auto; height:auto; font-size:8px; line-height:1; transform:translate(-50%,-50%); }
    .sf-li-root .sf-li-spin-up::before { content:"▲"; }
    .sf-li-root .sf-li-spin-down::before { content:"▼"; }
    /* Crop-to-fill extras: Fill/Crop toggle + 3x3 anchor grid. */
    .sf-li-root .sf-li-swaprow { display:flex; gap:6px; align-items:stretch; }
    .sf-li-root .sf-li-wh-col .sf-li-swaprow .sf-li-swap { flex:0 0 46px; width:auto; height:auto; align-self:stretch; }
    .sf-li-root .sf-li-fillcrop { flex:1; display:grid; grid-template-columns:1fr 1fr; background:#1d1d1d; border:1px solid #444; border-radius:4px; overflow:hidden; }
    .sf-li-root .sf-li-fillcrop > div { display:flex; align-items:center; justify-content:center; font-size:9.5px; padding:5px 0; color:#aaa; cursor:pointer; user-select:none; }
    .sf-li-root .sf-li-fillcrop > div:hover { color:#ddd; background:rgba(255,255,255,.08); }
    .sf-li-root .sf-li-fillcrop > div.active { background:#f66744; color:#fff; }
    .sf-li-root .sf-li-anchor { display:grid; grid-template-columns:repeat(3,1fr); grid-template-rows:repeat(3,1fr); gap:3px;
      width:100%; max-width:96px; aspect-ratio:1; margin:0 auto; background:#1d1d1d; border:1px solid #444; border-radius:5px; padding:5px; box-sizing:border-box; }
    .sf-li-root .sf-li-anchor-cell { background:rgba(255,255,255,.07); border-radius:2px; cursor:pointer; transition:background .08s; }
    .sf-li-root .sf-li-anchor-cell:hover { background:rgba(255,255,255,.18); }
    .sf-li-root .sf-li-anchor-cell.active { background:#f66744; }
    /* Bring shared quick-pick + ratio chips in line (orange hover). */
    .sf-li-root .sf-li-quickpick { box-sizing:border-box; }
    .sf-li-root .sf-li-quickpick:hover { border-color:#f66744; color:#ddd; }
    .sf-li-root .sf-li-ratio-chip { box-sizing:border-box; }
    .sf-li-root .sf-li-ratio-chip:hover { border-color:#f66744; color:#ddd; }
    /* ── B2 thumbnail dropdown popup ── */
    .sf-li-popup {
      background:#1d1d1d; border:1px solid #444; border-radius:6px;
      box-shadow:0 4px 16px rgba(0,0,0,.4); font-size:11px;
      font-family:ui-sans-serif,system-ui,sans-serif; color:#ccc; overflow:hidden;
    }
    .sf-li-pop-search { display:flex; align-items:center; gap:7px; padding:7px 9px;
      background:#161616; border-bottom:1px solid #333; }
    .sf-li-pop-mag { width:12px; height:12px; flex:none; border:1.6px solid #777; border-radius:50%; position:relative; }
    .sf-li-pop-mag::after { content:""; position:absolute; width:5px; height:1.6px; background:#777;
      transform:rotate(45deg); right:-3px; bottom:0; }
    .sf-li-pop-search input { flex:1; min-width:0; background:transparent; border:none; outline:none;
      color:#ddd; font-size:11px; font-family:inherit; }
    .sf-li-pop-search input::placeholder { color:#777; }
    .sf-li-pop-sizetoggle { flex:none; display:flex; border:1px solid #444; border-radius:4px; overflow:hidden; }
    .sf-li-pop-sizetoggle span { padding:2px 7px; font-size:10px; color:#aaa; cursor:pointer; user-select:none; line-height:1.4; }
    .sf-li-pop-sizetoggle span.on { background:#f66744; color:#fff; }
    .sf-li-pop-sizetoggle span:not(.on):hover { color:#ddd; }
    .sf-li-bsplit { display:flex; max-height:320px; }
    .sf-li-bfolders { width:104px; flex:none; border-right:1px solid #333; background:#141414; overflow:auto; }
    .sf-li-bfolder { padding:8px 9px; font-size:10.5px; color:#aaa; cursor:pointer; display:flex;
      justify-content:space-between; gap:4px; align-items:center; }
    .sf-li-bfolder:hover { background:#2a2a2a; }
    .sf-li-bfolder.on { background:color-mix(in srgb, #f66744 16%, transparent); color:#f66744; border-left:2px solid #f66744; padding-left:7px; }
    .sf-li-bfolder.all { border-bottom:1px solid #333; color:#cfcfcf; }
    .sf-li-bfolder.all.on { color:#f66744; }
    .sf-li-bfolder-n { color:#888; font-size:9px; flex:none; }
    .sf-li-bfolder > span:first-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
    .sf-li-bpane { flex:1; min-width:0; overflow:auto; max-height:320px; }
    .sf-li-pop-sec { padding:5px 10px 4px; font-size:9px; color:#777; text-transform:uppercase; letter-spacing:.5px;
      background:#141414; border-bottom:1px solid #333; display:flex; align-items:center; gap:6px; position:sticky; top:0; z-index:1; }
    .sf-li-pop-sec-c { margin-left:auto; color:#888; }
    .sf-li-imgrow { display:flex; align-items:center; gap:9px; padding:4px 10px; cursor:pointer; }
    .sf-li-imgrow:hover { background:#2a2a2a; }
    .sf-li-imgrow.cur { background:color-mix(in srgb, #f66744 12%, transparent); }
    .sf-li-imgrow.cur .sf-li-imgrow-lbl { color:#f66744; font-weight:600; }
    .sf-li-imgrow-lbl { flex:1; min-width:0; font-size:11px; color:#ccc; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-li-pop-thumb { position:relative; flex:none; border-radius:4px; overflow:hidden;
      background:linear-gradient(135deg,#3a3f4a,#222); }
    .sf-li-pop-thumb img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; }
    .sf-li-pop-glyph { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
      font-size:10px; color:rgba(255,255,255,.3); }
    .sf-li-bsplit.thumb-sm .sf-li-pop-thumb { width:32px; height:32px; }
    .sf-li-bsplit.thumb-lg .sf-li-pop-thumb { width:48px; height:48px; }
    .sf-li-pop-empty { padding:10px; color:#666; text-align:center; }
    .sf-li-pop-foot { padding:6px 10px; font-size:9px; color:#777; background:#141414;
      border-top:1px solid #333; text-align:center; }
  `;
  const el = document.createElement("style");
  el.id = "sf-load-image-css";
  el.textContent = css;
  document.head.appendChild(el);
}

export function buildRoot() {
  const root = document.createElement("div");
  root.className = "sf-li-root";

  // Upload button (orange, prominent, primary action).
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "sf-li-upload-btn";
  const ico = document.createElement("span");
  ico.className = "ico";
  const lbl = document.createElement("span");
  lbl.textContent = "上传图片";
  btn.append(ico, lbl);
  root.appendChild(btn);

  // Hint line for alternate upload methods (shown when no image yet).
  const hint = document.createElement("div");
  hint.className = "sf-li-hint";
  hint.dataset.role = "hint";
  hint.innerHTML = `或拖放文件到此处 · <kbd>Ctrl+V</kbd> 粘贴`;
  root.appendChild(hint);

  // File row: [◀ prev] [ filename dropdown ] [▶ next]. Arrow buttons cycle
  // through input/ images so users can flip through them visually, matching
  // native ComfyUI LoadImage. Both the prev/next arrows and PageUp/PageDown
  // (wired in index.js) route through setSelectedImage so the bottom preview
  // updates immediately.
  const fileRow = document.createElement("div");
  fileRow.className = "sf-li-filerow";

  const prev = document.createElement("button");
  prev.type = "button";
  prev.className = "sf-li-nav";
  prev.dataset.role = "prev";
  prev.title = "上一张图片 (PageUp)";
  prev.textContent = "◀";

  const dd = document.createElement("div");
  dd.className = "sf-li-dropdown";
  dd.dataset.role = "dropdown";
  dd.innerHTML = `<span class="name">— 未选择图片 —</span><span class="counter" data-role="counter"></span><span class="arrow">▼</span>`;

  const next = document.createElement("button");
  next.type = "button";
  next.className = "sf-li-nav";
  next.dataset.role = "next";
  next.title = "下一张图片 (PageDown)";
  next.textContent = "▶";

  fileRow.append(prev, dd, next);
  root.appendChild(fileRow);

  // The input/output size readout is no longer a DOM bar here — it is painted
  // by onDrawForeground (INPUT → OUTPUT cards in the output-slot dead space).

  return root;
}

// Hides every auto-created widget so we can render our own UI in the DOM
// widget. `image_upload: True` creates TWO widgets in INPUT_TYPES on the
// Vue frontend: the `image` combo + a separate `upload` button widget — both
// need to be hidden, plus any other auto-created widget that isn't ours.
//
// Uses the same multi-technique pattern as shared/utils.mjs `hideJsonWidget`:
// setting `canvasOnly` alone is not enough for canvas drawing on the current
// Vue frontend — must also set `hidden=true`, zero `computeSize`, and hide
// any DOM element. Returns the `image` combo widget so callers can read /
// write its `.value` (that drives the actual file selection).
export function hideNativeImageCombo(node) {
  let imageWidget = null;
  for (const w of (node.widgets || [])) {
    if (!w) continue;
    if (w.name === "image") imageWidget = w;
    w.hidden = true;
    w.computeSize = () => [0, -4];
    if (!w.options) w.options = {};
    w.options.canvasOnly = true;
    if (w.element) w.element.style.display = "none";
  }
  // Vue may DOM-render an upload widget AFTER nodeCreated — re-hide on the
  // next animation frame as a belt-and-braces. Mirrors hideJsonWidget.
  requestAnimationFrame(() => {
    for (const w of (node.widgets || [])) {
      if (!w || w.name === "sf_load_image_ui") continue;
      const _el = w.element || w.inputEl; // prefer .element; .inputEl only on old builds (no deprecation warning)
      if (_el) _el.style.display = "none";
    }
  });
  return imageWidget;
}

// Group combo values by subfolder so the popup renders:
//   ─ root ─
//      file1.png
//      file2.png
//   ─ Studio1 ─
//      bunny.png
// Returns an array of { folder, files } in display order (root first, then
// folders alphabetised). Each `files` entry is { full, name } where `full`
// is the value to write back and `name` is the bare filename to display.
function groupValuesByFolder(values) {
  const map = new Map();
  for (const v of values) {
    const { subfolder, filename } = splitFilenameSubfolder(v);
    if (!map.has(subfolder)) map.set(subfolder, []);
    map.get(subfolder).push({ full: v, name: filename });
  }
  // Sort the file lists alphabetically. Folders: root first, then ABC.
  for (const list of map.values()) list.sort((a, b) => a.name.localeCompare(b.name));
  const folders = [...map.keys()].sort((a, b) => {
    if (a === "" && b !== "") return -1;
    if (a !== "" && b === "") return 1;
    return a.localeCompare(b);
  });
  return folders.map((folder) => ({ folder, files: map.get(folder) }));
}

// Build a same-origin /view URL for a combo value like "3d/cat.png".
// Relative path → works on whatever host/port ComfyUI runs on. No cache-buster
// so the browser caches thumbnails across re-opens.
function thumbURL(full) {
  const { subfolder, filename } = splitFilenameSubfolder(full);
  return pixApiUrl(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);
}

// Read/write the persisted thumbnail size ("Small" | "Large"). Falls back to
// "Large" if the setting isn't registered yet or settings aren't ready.
function getThumbSize() {
  try {
    const v = app.ui.settings.getSettingValue("sfnodes.LoadImage.ThumbSize");
    return v === "Small" ? "Small" : "Large";
  } catch (_e) { return "Large"; }
}
function setThumbSize(v) {
  try {
    const s = app.ui.settings;
    // Prefer the async setter when present (newer ComfyUI persists reliably
    // through it; the sync form can no-op on disk on some builds).
    if (typeof s.setSettingValueAsync === "function") s.setSettingValueAsync("sfnodes.LoadImage.ThumbSize", v);
    else s.setSettingValue("sfnodes.LoadImage.ThumbSize", v);
  } catch (_e) { /* ignore */ }
}

// 注册缩略图大小设置项（原版通过 pixaroma 自己的设置面板注册；本项目用
// ComfyUI 原生 app.ui.settings.addSetting，惯例见 web/multi_lora_tree.js）。
// 幂等：扩展 setup 阶段调用一次即可。
let _thumbSettingRegistered = false;
export function registerThumbSizeSetting() {
  if (_thumbSettingRegistered) return;
  _thumbSettingRegistered = true;
  try {
    app.ui.settings.addSetting({
      id: "sfnodes.LoadImage.ThumbSize",
      name: "SF Load Image Resize: thumbnail size in the image dropdown",
      defaultValue: "Large",
      type: "combo",
      options: () => [
        { value: "Small", text: "Small", selected: getThumbSize() === "Small" },
        { value: "Large", text: "Large", selected: getThumbSize() === "Large" },
      ],
    });
  } catch (_e) { /* 设置系统不可用则退化为会话内记忆 */ }
}

// Open a thumbnail picker popup anchored below the file row.
//  - Search box filters across ALL folders while it has text.
//  - Left sidebar (only when subfolders exist) lists All + each folder.
//  - Right pane shows thumbnail rows; opens to "All" grouped by folder.
//  - Small/Large thumbnail size toggle, persisted via the ThumbSize setting.
// Reuses groupValuesByFolder, splitFilenameSubfolder, setSelectedImage.
export function openImageDropdown(node, anchorEl, onPick) {
  const imageWidget = node._sfLiImageWidget;
  if (!imageWidget) return;
  const values = imageWidget.options?.values || [];

  // Close any existing popup. Call its stored cleanup (not a bare remove) so
  // the prior popup's document-level capture listeners are detached too.
  const _existingPopup = document.querySelector(".sf-li-popup");
  if (_existingPopup) {
    if (typeof _existingPopup._pixClose === "function") _existingPopup._pixClose();
    else _existingPopup.remove();
  }

  const popup = document.createElement("div");
  popup.className = "sf-li-popup";
  const rect = anchorEl.getBoundingClientRect();
  const width = Math.max(rect.width, 360); // widen so sidebar + thumbs fit
  Object.assign(popup.style, {
    position: "fixed",
    zIndex: 99999,
    left: `${rect.left}px`,
    top: `${rect.bottom + 2}px`,
    width: `${width}px`,
  });

  // ── close handling (defined early so row click handlers can call it) ──
  function closePopup() {
    popup.remove();
    document.removeEventListener("mousedown", onDocDown, true);
    document.removeEventListener("pointerdown", onDocDown, true);
    document.removeEventListener("wheel", onWheel, true);
    document.removeEventListener("keydown", onKey, true);
  }
  const onDocDown = (e) => { if (!popup.contains(e.target)) closePopup(); };
  const onWheel = (e) => { if (!popup.contains(e.target)) closePopup(); };
  const onKey = (e) => { if (e.key === "Escape") closePopup(); };
  popup._pixClose = closePopup; // so a later open can detach our listeners

  if (values.length === 0) {
    const empty = document.createElement("div");
    empty.className = "sf-li-pop-empty";
    empty.textContent = "（尚未上传图片）";
    popup.appendChild(empty);
    document.body.appendChild(popup);
    setTimeout(() => {
      document.addEventListener("mousedown", onDocDown, true);
      document.addEventListener("pointerdown", onDocDown, true);
      document.addEventListener("wheel", onWheel, true);
      document.addEventListener("keydown", onKey, true);
    }, 0);
    return;
  }

  const groups = groupValuesByFolder(values); // [{folder, files:[{full,name}]}], root first
  const curVal = imageWidget.value;
  const hasSubfolders = groups.some((g) => g.folder !== "");

  // ── state ──
  let activeFolder = "__all"; // "__all" | "" (root) | "<FolderName>"
  let query = "";
  let thumbSize = getThumbSize();
  let scrollTarget = null;

  // ── search row (filter + size toggle) ──
  const searchRow = document.createElement("div");
  searchRow.className = "sf-li-pop-search";
  const mag = document.createElement("span");
  mag.className = "sf-li-pop-mag";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "筛选图片…";
  const sizeToggle = document.createElement("div");
  sizeToggle.className = "sf-li-pop-sizetoggle";
  const segS = document.createElement("span"); segS.textContent = "S"; segS.title = "小缩略图";
  const segL = document.createElement("span"); segL.textContent = "L"; segL.title = "大缩略图";
  sizeToggle.append(segS, segL);
  searchRow.append(mag, input, sizeToggle);
  popup.appendChild(searchRow);

  // ── body: sidebar (optional) + scrollable pane ──
  const body = document.createElement("div");
  body.className = "sf-li-bsplit";
  const sidebar = document.createElement("div");
  sidebar.className = "sf-li-bfolders";
  const pane = document.createElement("div");
  pane.className = "sf-li-bpane";
  if (hasSubfolders) body.append(sidebar, pane);
  else body.append(pane);
  popup.appendChild(body);

  // ── footer ──
  const footer = document.createElement("div");
  footer.className = "sf-li-pop-foot";
  popup.appendChild(footer);

  // ── element builders ──
  const makeRow = (entry) => {
    const row = document.createElement("div");
    row.className = "sf-li-imgrow" + (entry.full === curVal ? " cur" : "");
    const th = document.createElement("span");
    th.className = "sf-li-pop-thumb";
    const glyph = document.createElement("span");
    glyph.className = "sf-li-pop-glyph";
    glyph.textContent = "▣";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.onerror = () => { img.style.display = "none"; };
    img.src = thumbURL(entry.full);
    th.append(glyph, img);
    const lbl = document.createElement("span");
    lbl.className = "sf-li-imgrow-lbl";
    lbl.textContent = entry.name;
    lbl.title = entry.full;
    row.append(th, lbl);
    if (entry.full === curVal) scrollTarget = row;
    row.addEventListener("click", (e) => {
      e.stopPropagation();
      setSelectedImage(node, entry.full);
      closePopup();
      if (onPick) onPick(entry.full);
    });
    return row;
  };
  const makeSec = (label, count) => {
    const s = document.createElement("div");
    s.className = "sf-li-pop-sec";
    s.textContent = label;
    const c = document.createElement("span");
    c.className = "sf-li-pop-sec-c";
    c.textContent = String(count);
    s.appendChild(c);
    return s;
  };
  const folderLabel = (key) => (key === "" ? "根目录" : key);
  // Scroll the current image's row into view (deferred so the pane is laid out).
  // Called on every non-search render so a folder switch / search-clear re-centers it.
  const scrollCurrentIntoView = () => {
    if (!scrollTarget) return;
    const t = scrollTarget;
    queueMicrotask(() => { try { t.scrollIntoView({ block: "nearest" }); } catch (_e) { /* ignore */ } });
  };

  // ── renderers ──
  const renderSidebar = () => {
    if (!hasSubfolders) return;
    sidebar.replaceChildren();
    const entries = [["__all", "全部", values.length]];
    for (const g of groups) entries.push([g.folder, folderLabel(g.folder), g.files.length]);
    for (const [key, label, count] of entries) {
      const f = document.createElement("div");
      f.className = "sf-li-bfolder"
        + (key === "__all" ? " all" : "")
        + (key === activeFolder && !query.trim() ? " on" : "");
      const t = document.createElement("span");
      t.textContent = label;
      const n = document.createElement("span");
      n.className = "sf-li-bfolder-n";
      n.textContent = String(count);
      f.append(t, n);
      f.addEventListener("click", (e) => {
        e.stopPropagation();
        activeFolder = key;
        input.value = "";
        query = "";
        renderSidebar();
        renderPane();
      });
      sidebar.appendChild(f);
    }
  };

  const renderPane = () => {
    pane.replaceChildren();
    scrollTarget = null;
    const q = query.trim().toLowerCase();

    if (q) {
      // Search across ALL folders; show only matches, grouped by folder.
      let matches = 0;
      for (const g of groups) {
        const hit = g.files.filter((f) => f.name.toLowerCase().includes(q));
        if (hit.length === 0) continue;
        matches += hit.length;
        pane.appendChild(makeSec(folderLabel(g.folder), hit.length));
        for (const entry of hit) pane.appendChild(makeRow(entry));
      }
      if (matches === 0) {
        const none = document.createElement("div");
        none.className = "sf-li-pop-empty";
        none.textContent = "（无匹配结果）";
        pane.appendChild(none);
      }
      footer.textContent = `${matches} 个匹配`;
      return;
    }

    if (!hasSubfolders) {
      // Flat input/ — plain thumbnail list, no sidebar.
      for (const g of groups) for (const entry of g.files) pane.appendChild(makeRow(entry));
      footer.textContent = `${values.length} 张图片`;
      scrollCurrentIntoView();
      return;
    }

    if (activeFolder === "__all") {
      // All images, grouped by folder with sticky section headers.
      for (const g of groups) {
        pane.appendChild(makeSec(folderLabel(g.folder), g.files.length));
        for (const entry of g.files) pane.appendChild(makeRow(entry));
      }
      footer.textContent = `${values.length} 张图片 · 全部`;
    } else {
      const g = groups.find((x) => x.folder === activeFolder);
      const files = g ? g.files : [];
      for (const entry of files) pane.appendChild(makeRow(entry));
      footer.textContent = `${files.length} 张图片 · ${folderLabel(activeFolder)}`;
    }
    scrollCurrentIntoView();
  };

  const applyThumbSize = () => {
    body.classList.toggle("thumb-sm", thumbSize === "Small");
    body.classList.toggle("thumb-lg", thumbSize !== "Small");
    segS.classList.toggle("on", thumbSize === "Small");
    segL.classList.toggle("on", thumbSize !== "Small");
  };

  // ── events ──
  input.addEventListener("input", () => { query = input.value; renderSidebar(); renderPane(); });
  input.addEventListener("click", (e) => e.stopPropagation());
  // Keep LiteGraph's canvas shortcuts (Delete/Backspace = delete the selected
  // node, arrows = nudge, etc.) from firing while typing in the filter box —
  // the node is selected whenever this popup is open. Mirrors makeNumericInput
  // (Load Image Pattern #6). Enter picks the first listed match.
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      pane.querySelector(".sf-li-imgrow")?.click();
    }
    e.stopImmediatePropagation();
  });
  segS.addEventListener("click", (e) => {
    e.stopPropagation();
    thumbSize = "Small"; setThumbSize(thumbSize); applyThumbSize();
  });
  segL.addEventListener("click", (e) => {
    e.stopPropagation();
    thumbSize = "Large"; setThumbSize(thumbSize); applyThumbSize();
  });

  // ── initial render ──
  applyThumbSize();
  renderSidebar();
  renderPane();

  document.body.appendChild(popup);

  // Keep the popup on-screen: clamp horizontally and flip above the row when it
  // would overflow the bottom of the viewport (now measurable post-append).
  const pr = popup.getBoundingClientRect();
  let left = rect.left;
  if (left + pr.width > window.innerWidth - 4) left = Math.max(4, window.innerWidth - pr.width - 4);
  popup.style.left = `${left}px`;
  if (pr.bottom > window.innerHeight - 4) {
    const above = rect.top - pr.height - 2;
    popup.style.top = `${above >= 4 ? above : Math.max(4, window.innerHeight - pr.height - 4)}px`;
  }

  // renderPane() already scrolls the current row into view. Focus the filter box.
  queueMicrotask(() => { try { input.focus(); } catch (_e) { /* ignore */ } });

  // Attach close listeners after the opening click settles (capture phase so we
  // preempt LiteGraph; each gated on !popup.contains so inside scroll/clicks
  // don't close — Load Image Pattern #14).
  setTimeout(() => {
    document.addEventListener("mousedown", onDocDown, true);
    document.addEventListener("pointerdown", onDocDown, true);
    document.addEventListener("wheel", onWheel, true);
    document.addEventListener("keydown", onKey, true);
  }, 0);
}

const MODE_CHIPS = [
  { id: "off",          label: "Off",          title: "不缩放。（若设置了 Snap 仍会生效。）" },
  { id: "max_mp",       label: "Max MP",       title: "缩放使总像素数不超过百万像素上限，保持宽高比。" },
  { id: "longest_side", label: "Longest side", title: "缩放使最长边等于该像素值，保持宽高比。" },
  { id: "scale_factor", label: "Scale by ×",   title: "两维同时乘以该倍率，保持宽高比。" },
  { id: "fit_inside",   label: "Fit inside",   title: "不裁剪地缩放到 W×H 范围内，保持宽高比。" },
  { id: "cover",        label: "Crop to fill", title: "精确缩放到 W×H。Fill 先缩放再裁剪溢出部分；Crop 直接切取 1:1 像素区域。锚点决定保留哪部分。" },
  { id: "match_ratio",  label: "Match ratio",  title: "将图片裁剪到目标宽高比（不缩放）。" },
  { id: "pad",          label: "Pad",          title: "在指定边添加像素边框，新增区域成为白色修复遮罩区域。" },
];

export function renderChips(state) {
  const wrap = document.createElement("div");
  wrap.className = "sf-li-chips";
  for (const c of MODE_CHIPS) {
    const el = document.createElement("div");
    el.className = "sf-li-chip" + (state.mode === c.id ? " active" : "");
    el.dataset.modeId = c.id;
    el.textContent = c.label;
    el.title = c.title || "";
    wrap.appendChild(el);
  }
  return wrap;
}

const SNAP_OPTIONS = [0, 8, 16, 32, 64];
const RESAMPLE_OPTIONS = [
  { id: "auto",     label: "Auto",     hint: "缩小用 Lanczos，放大用 Bilinear" },
  { id: "nearest",  label: "Nearest",  hint: "像素级精确，无平滑" },
  { id: "bilinear", label: "Bilinear", hint: "快速，平滑" },
  { id: "bicubic",  label: "Bicubic",  hint: "较慢，更锐利" },
  { id: "lanczos",  label: "Lanczos",  hint: "最慢，最锐利" },
];

// Custom resample dropdown popup. Same look as the file dropdown popup —
// fixed-position list anchored to the row, click an item to commit.
function openResamplePopup(anchorEl, currentValue, onPick) {
  document.querySelector(".sf-li-rs-popup")?.remove();

  const popup = document.createElement("div");
  popup.className = "sf-li-rs-popup";
  const rect = anchorEl.getBoundingClientRect();
  popup.style.left = `${rect.left}px`;
  popup.style.top  = `${rect.bottom + 2}px`;
  popup.style.width = `${rect.width}px`;

  for (const opt of RESAMPLE_OPTIONS) {
    const item = document.createElement("div");
    item.className = "sf-li-rs-item" + (opt.id === currentValue ? " active" : "");
    const lbl = document.createElement("span");
    lbl.className = "sf-li-rs-item-label";
    lbl.textContent = opt.label;
    const hint = document.createElement("span");
    hint.className = "sf-li-rs-item-hint";
    hint.textContent = opt.hint;
    item.append(lbl, hint);
    item.addEventListener("click", (e) => {
      e.stopPropagation();
      onPick(opt.id);
      close();
    });
    popup.appendChild(item);
  }

  document.body.appendChild(popup);

  function close() {
    popup.remove();
    document.removeEventListener("mousedown", onDocDown, true);
    document.removeEventListener("pointerdown", onDocDown, true);
    document.removeEventListener("wheel", onWheel, true);
    document.removeEventListener("keydown", onKey, true);
  }
  const onDocDown = (e) => {
    if (!popup.contains(e.target)) close();
  };
  const onWheel = (e) => {
    if (!popup.contains(e.target)) close();
  };
  const onKey = (e) => {
    if (e.key === "Escape") close();
  };
  setTimeout(() => {
    document.addEventListener("mousedown", onDocDown, true);
    document.addEventListener("pointerdown", onDocDown, true);
    document.addEventListener("wheel", onWheel, true);
    document.addEventListener("keydown", onKey, true);
  }, 0);
}

export function renderGlobalControls(node, state, writeState, onChange) {
  const wrap = document.createElement("div");
  wrap.className = "sf-li-global";

  // Centered snap footer: magnet + "Snap" + chips.
  const foot = document.createElement("div");
  foot.className = "sf-li-foot";
  const snap = document.createElement("div");
  snap.className = "sf-li-snap2";
  const icon = document.createElement("span");
  icon.className = "sf-li-snap-icon";
  snap.appendChild(icon);
  const lbl = document.createElement("span");
  lbl.className = "sf-li-snap-lbl";
  lbl.textContent = "吸附";
  snap.appendChild(lbl);
  for (const v of SNAP_OPTIONS) {
    const b = document.createElement("div");
    b.className = "sf-li-schip" + (v === (state.snap || 0) ? " active" : "");
    b.dataset.v = String(v);
    b.textContent = v === 0 ? "Off" : String(v);
    b.title = v === 0 ? "不吸附。"
      : `将输出尺寸向下取整为 ${v} px 的倍数（保持潜空间对齐）。`;
    snap.appendChild(b);
  }
  foot.appendChild(snap);
  wrap.appendChild(foot);

  // Resample picker: [◀] [ Resample: Auto ▾ ] [▶]
  const rsRow = document.createElement("div");
  rsRow.className = "sf-li-rs2-row";
  const prev = document.createElement("button");
  prev.type = "button"; prev.className = "sf-li-rs2-nav"; prev.title = "上一个重采样滤波器"; prev.textContent = "◀";
  const dd = document.createElement("div");
  dd.className = "sf-li-rs2-dd";
  dd.title = "缩放时使用的重采样滤波器。点击选择，或用箭头切换。";
  const rsValue = document.createElement("span");
  rsValue.className = "sf-li-rs2-value";
  rsValue.textContent = "重采样：" + resampleLabel(state.resample || "auto");
  const rsArrow = document.createElement("span");
  rsArrow.className = "sf-li-rs2-arrow"; rsArrow.textContent = "▼";
  dd.append(rsValue, rsArrow);
  const next = document.createElement("button");
  next.type = "button"; next.className = "sf-li-rs2-nav"; next.title = "下一个重采样滤波器"; next.textContent = "▶";
  rsRow.append(prev, dd, next);
  wrap.appendChild(rsRow);

  // Upscaling toggle button.
  const upBtn = document.createElement("button");
  upBtn.type = "button";
  upBtn.title = "允许图片放大到超过原始尺寸。关闭 = 永不放大。";
  const upOn = state.allow_upscale !== false;
  upBtn.className = "sf-li-upbtn" + (upOn ? " is-on" : "");
  upBtn.textContent = upOn ? "允许放大：开" : "允许放大：关";
  wrap.appendChild(upBtn);

  // ── events ──
  const RESAMPLE_IDS = RESAMPLE_OPTIONS.map((o) => o.id);
  const setResample = (id) => {
    const s = readStateLocal(node);
    writeState(node, { ...s, resample: id });
    rsValue.textContent = "重采样：" + resampleLabel(id);
    onChange?.();
  };
  const cycleResample = (delta) => {
    const cur = (readStateLocal(node).resample) || "auto";
    let i = RESAMPLE_IDS.indexOf(cur); if (i < 0) i = 0;
    i = (i + delta + RESAMPLE_IDS.length) % RESAMPLE_IDS.length;
    setResample(RESAMPLE_IDS[i]);
  };
  foot.addEventListener("click", (e) => {
    const b = e.target.closest(".sf-li-schip");
    if (!b) return;
    e.stopPropagation();
    const v = parseInt(b.dataset.v, 10);
    for (const x of foot.querySelectorAll(".sf-li-schip")) x.classList.toggle("active", x === b);
    writeState(node, { ...readStateLocal(node), snap: v });
    onChange?.();
  });
  dd.addEventListener("click", (e) => {
    e.stopPropagation();
    openResamplePopup(dd, (readStateLocal(node).resample) || "auto", setResample);
  });
  prev.addEventListener("click", (e) => { e.stopPropagation(); cycleResample(-1); });
  next.addEventListener("click", (e) => { e.stopPropagation(); cycleResample(1); });
  upBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const on = !(readStateLocal(node).allow_upscale !== false);
    writeState(node, { ...readStateLocal(node), allow_upscale: on });
    upBtn.classList.toggle("is-on", on);
    upBtn.textContent = on ? "允许放大：开" : "允许放大：关";
    onChange?.();
  });

  return wrap;
}

// Local state read for event handlers (avoids depending on index.js import).
function readStateLocal(node) {
  try { return JSON.parse(node.properties?.sfLoadImageResizeState || "{}"); }
  catch { return {}; }
}

// Map a resample id to its display label.
function resampleLabel(id) {
  const o = RESAMPLE_OPTIONS.find((x) => x.id === id) || RESAMPLE_OPTIONS[0];
  return o.label;
}
// Per-mode panel post-processors for Load Image Pixaroma. Adapted from
// js/image_resize/index.js; class names use the .sf-li-* family that the
// scoped CSS in ui.mjs targets. Keeps Image Resize untouched.

const INLINE_LABELS = {
  max_mp: "Max megapixels",
  longest_side: "Longest side",
  scale_factor: "Scale by ×",
};

// Single-input modes: drop the section header, move the name INTO the input.
export function applyInlineLabel(panel, mode) {
  const label = INLINE_LABELS[mode];
  if (!label) return;
  panel.querySelector(".sf-li-panel-label")?.remove();
  const num = panel.querySelector(".sf-li-numinput");
  if (!num || num.querySelector(".sf-li-inline-label")) return;
  const lab = document.createElement("span");
  lab.className = "sf-li-inline-label";
  lab.textContent = label;
  num.insertBefore(lab, num.firstChild);
  num.classList.add("sf-li-num-labeled");
}

// Fit/Crop (W × H): W/H labels inside inputs, drop redundant size text, reflow
// into two columns (W/H/swap stacked left, aspect rect right).
export function applyWHLayout(panel) {
  const fields = [...panel.querySelectorAll(".sf-li-wh-field")];
  const tags = ["W", "H"];
  fields.forEach((f, i) => {
    f.querySelector(".sf-li-wh-label")?.remove();
    const num = f.querySelector(".sf-li-numinput");
    if (num && !num.querySelector(".sf-li-inline-label")) {
      const lab = document.createElement("span");
      lab.className = "sf-li-inline-label";
      lab.textContent = tags[i] || "";
      num.insertBefore(lab, num.firstChild);
      num.classList.add("sf-li-num-labeled");
    }
  });
  panel.querySelector(".sf-li-wh-rect-label")?.remove();

  const row = panel.querySelector(".sf-li-wh-row");
  const swap = panel.querySelector(".sf-li-swap");
  const preview = panel.querySelector(".sf-li-wh-preview");
  if (row && fields.length === 2 && preview && !panel.querySelector(".sf-li-wh-grid")) {
    const grid = document.createElement("div");
    grid.className = "sf-li-wh-grid";
    const col = document.createElement("div");
    col.className = "sf-li-wh-col";
    col.append(fields[0], fields[1]);
    if (swap) col.append(swap);
    grid.append(col, preview);
    row.replaceWith(grid);
  }
}

// Crop-to-fill extras: Fill/Crop scale toggle + 3×3 anchor picker.
export function applyCoverControls(node, panel, readState, writeState, onChange) {
  const state = readState(node);

  const swap = panel.querySelector(".sf-li-swap");
  if (swap && !panel.querySelector(".sf-li-fillcrop")) {
    const row = document.createElement("div");
    row.className = "sf-li-swaprow";
    const toggle = document.createElement("div");
    toggle.className = "sf-li-fillcrop";
    const fillOpt = document.createElement("div");
    fillOpt.textContent = "Fill"; fillOpt.dataset.cropScale = "1";
    fillOpt.title = "精确缩放到填满，裁剪溢出部分";
    const cropOpt = document.createElement("div");
    cropOpt.textContent = "Crop"; cropOpt.dataset.cropScale = "0";
    cropOpt.title = "直接切取 1:1 像素区域，不缩放";
    const scaleOn = state.crop_scale !== false;
    fillOpt.classList.toggle("active", scaleOn);
    cropOpt.classList.toggle("active", !scaleOn);
    toggle.append(fillOpt, cropOpt);
    swap.replaceWith(row);
    row.append(swap, toggle);
    toggle.addEventListener("click", (e) => {
      const opt = e.target.closest("[data-crop-scale]");
      if (!opt) return;
      const on = opt.dataset.cropScale === "1";
      writeState(node, { ...readState(node), crop_scale: on });
      fillOpt.classList.toggle("active", on);
      cropOpt.classList.toggle("active", !on);
      onChange?.();
    });
  }

  const preview = panel.querySelector(".sf-li-wh-preview");
  if (preview && !panel.querySelector(".sf-li-anchor")) {
    const ANCHORS = [
      "top-left", "top", "top-right",
      "left", "center", "right",
      "bottom-left", "bottom", "bottom-right",
    ];
    const cur = state.crop_anchor || "center";
    const grid = document.createElement("div");
    grid.className = "sf-li-anchor";
    grid.title = "从哪个位置裁剪";
    for (const a of ANCHORS) {
      const cell = document.createElement("div");
      cell.className = "sf-li-anchor-cell" + (a === cur ? " active" : "");
      cell.dataset.anchor = a;
      cell.title = { "top-left": "左上", "top": "上", "top-right": "右上", "left": "左", "center": "居中", "right": "右", "bottom-left": "左下", "bottom": "下", "bottom-right": "右下" }[a] || a;
      grid.appendChild(cell);
    }
    preview.replaceWith(grid);
    grid.addEventListener("click", (e) => {
      const cell = e.target.closest(".sf-li-anchor-cell");
      if (!cell) return;
      writeState(node, { ...readState(node), crop_anchor: cell.dataset.anchor });
      for (const c of grid.children) c.classList.toggle("active", c === cell);
      onChange?.();
    });
  }
}
