// ==========================================================================
// sf_outpaint.js - SF Image Outpaint 前端（移植自 comfyui-pixaroma
// js/outpaint/index.js，精简版：去掉 Pixaroma 品牌功能——accent 色、
// 设置面板、比例/MP 列表管理）。
// ==========================================================================
//
// 状态存 node.properties.outpaintState（随工作流保存），提交时由
// graphToPrompt 钩子注入隐藏输入 SFOutpaintState（只注入不剪枝）。
// 预览两层：上游 imgs[0]（Load Image 等）或 Python 每次运行存到 temp/ 的
// 输入帧（executed 事件携带 sf_outpaint_base）。
//
// 纯数学镜像在 sf_outpaint_core.js（可 .mjs 直测），本文件只做 UI。

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { injectResizePanelCSS, makeNumericInput } from "./sf_load_image_resize.js";
import {
  DEFAULT_STATE, DEFAULT_RATIOS, LIMITS, MAX_PAD, SNAPS, STATE_PROP,
  readState, writeState, parseRatio, anchorAxis, remapAnchor,
  padsForState, finalSize, sidePad,
} from "./sf_outpaint_core.js";

const CLASS = "SFImageOutpaint";
const HIDDEN_INPUT = "SFOutpaintState"; // 必须与 outpaint.py 的隐藏输入一致

// 品牌主色（原版 var(--pix-op-acc) 会随设置面板变化，本项目无 accent 系统，
// 固定值，惯例见 sf_load_image_resize.js）
const BRAND = "#f66744";

const DEFAULT_W = 305;
const MIN_W = 305;
const DEFAULT_H = 421;
const PAD = 9;      // .sf-op-inner 内边距，上下
const ROW_GAP = 6;  // 行间距
const PREVIEW_MIN = 120;
const FLOOR_MIN = 60;
const FLOOR_CAP = 460;

// ── 内联小工具（移植自 pixaroma js/shared/，去除插件专属依赖，同 sf_crop.js）─

// 工作流加载守卫：wrap app.loadGraphData 一次，加载 + 300ms 尾窗内
// isGraphLoading() 为 true。
let _sfOpGraphLoading = false;
if (app && app.loadGraphData && !app._sfOpGraphLoadWrapped) {
  app._sfOpGraphLoadWrapped = true;
  const _origLoadGraphData = app.loadGraphData.bind(app);
  app.loadGraphData = function (...args) {
    _sfOpGraphLoading = true;
    let r;
    try {
      r = _origLoadGraphData(...args);
    } finally {
      Promise.resolve(r).finally(() => setTimeout(() => { _sfOpGraphLoading = false; }, 300));
    }
    return r;
  };
}
function isGraphLoading() {
  return _sfOpGraphLoading;
}

// Nodes 2.0 (Vue) 渲染器检测 + canvasOnly 自适应
function isVueNodes() {
  return !!window.LiteGraph?.vueNodesMode;
}
function applyAdaptiveCanvasOnly(widget) {
  if (!widget || !widget.options) return widget;
  try {
    Object.defineProperty(widget.options, "canvasOnly", {
      configurable: true,
      enumerable: true,
      get() {
        return !window.LiteGraph?.vueNodesMode;
      },
    });
  } catch (e) {
    widget.options.canvasOnly = !window.LiteGraph?.vueNodesMode;
  }
  return widget;
}

// 滚轮缩放透传（仅 Classic 渲染器）
function installCanvasZoomPassthrough(root) {
  if (!root || typeof root.addEventListener !== "function") return () => {};
  const onWheel = (e) => {
    if (isVueNodes()) return;
    const canvasEl = app?.canvas?.canvas;
    if (!canvasEl) return;
    e.preventDefault();
    e.stopPropagation();
    const { clientX, clientY, deltaX, deltaY, deltaMode, ctrlKey, metaKey, shiftKey } = e;
    canvasEl.dispatchEvent(new WheelEvent("wheel", {
      clientX, clientY, deltaX, deltaY, deltaMode,
      ctrlKey, metaKey, shiftKey, bubbles: true, cancelable: true,
    }));
  };
  root.addEventListener("wheel", onWheel, { passive: false });
  return () => root.removeEventListener("wheel", onWheel);
}

// 画布 backing store 缩放：dpr * 图缩放（Vue 节点体被 CSS transform 缩放，
// 只按布局像素画会在大图缩放时发糊），长边封顶防深缩放分配巨型画布。
const CANVAS_BACKING_CAP = 6000;
function canvasBackingScale(cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const zoom = Math.max(1, app.canvas?.ds?.scale || 1);
  let s = dpr * zoom;
  const longCss = Math.max(cssW || 0, cssH || 0);
  if (longCss > 0 && longCss * s > CANVAS_BACKING_CAP) s = CANVAS_BACKING_CAP / longCss;
  return s;
}

// 图缩放变化时逐帧重绘（ResizeObserver 对图缩放不触发：布局尺寸没变，只有
// CSS transform 变了）。每帧只 diff 缩放，无 DOM 读。
function installZoomRepaint(node, getSize, render, rafKey) {
  void getSize;
  let lastZoom = -1;
  const tick = () => {
    const zoom = Math.max(1, app.canvas?.ds?.scale || 1);
    if (Math.abs(zoom - lastZoom) > 0.005) { lastZoom = zoom; render(); }
    node[rafKey] = requestAnimationFrame(tick);
  };
  node[rafKey] = requestAnimationFrame(tick);
  return () => {
    try { cancelAnimationFrame(node[rafKey]); } catch (_e) { /* ignore */ }
    node[rafKey] = null;
  };
}

// 轻量取色：点击 swatch 弹出原生取色器（原版为 Photoshop 风格 modal，
// 其 1000+ 行依赖不属于本项目基础设施；原生 input[type=color] 等效）。
// input 必须 append 到 document 才会触发原生取色面板，用完即移除。
function openSimpleColorPicker({ initialColor, onPick }) {
  const inp = document.createElement("input");
  inp.type = "color";
  inp.value = /^#[0-9a-f]{6}$/i.test(initialColor || "") ? initialColor : "#000000";
  inp.style.position = "fixed";
  inp.style.left = "-9999px";
  inp.style.top = "0";
  inp.addEventListener("change", () => { onPick(inp.value); inp.remove(); });
  inp.addEventListener("input", () => { onPick(inp.value); });
  document.body.appendChild(inp);
  try { inp.click(); } catch (_e) { /* ignore */ }
  setTimeout(() => inp.remove(), 60000);
}

// 重置按钮图标（内联 data URI，本项目无资产服务路由）
const ICON_RESET = "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M5.076,36.174h7.951c-.055,7.957,5.073,14.981,12.165,17.655,7.796,2.94,16.386.61,21.503-5.823,3.292-4.139,4.582-9.187,3.971-14.41-1.006-8.613-7.761-15.308-16.394-16.422v6.522c-.005.556-.627,1.25-1.062,1.41-.492.182-1.518.167-1.943-.194l-11.966-10.184c-.593-.505-.816-1.117-.82-1.857-.003-.68.401-1.278,1.043-1.825L31.164,1.139c.559-.476,1.425-.561,2.095-.312.476.177,1.01.956,1.01,1.62v6.426c4.626.52,9.001,1.929,12.753,4.544,6.46,4.503,10.607,11.215,11.617,18.995.345,2.661.41,4.965.007,7.63-.945,6.242-3.878,11.989-8.439,16.154-12.151,11.097-30.942,8.938-40.35-4.563-3.056-4.386-4.798-9.678-4.781-15.459ZM38.682,41.707v-9.247c.003-1.069-.713-1.902-1.696-2.175h-10.107c-1.019.222-1.738,1.087-1.738,2.147v9.334c0,1.196.921,2.138,2.127,2.138h9.073c1.213,0,2.339-.973,2.342-2.197Z"/></svg>');

// ── preview ────────────────────────────────────────────────────────────────
const PREVIEW_INSET = 6;   // 构图四周的留白
const BAND_TEXT_MIN = 24;  // 低于此厚度的色带放不下文字，数字跳到图上

// 色带上的 pad 数字用色。填充色是用户设置，固定颜色不可行：近黑/近白里
// 选对比度更高的那个（WCAG 相对亮度）。简单亮度阈值会把默认的中灰
// #808080（亮度 128）判给白色——那是最差选择（对黑 3.95:1 vs 对白 5.32:1
// 的黑色反而更好）。量对比度选黑色，且对任意用户颜色都成立。
function relLuminance(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || "").trim());
  if (!m) return 0.5; // 解析失败：当中灰，谁都不占
  const n = parseInt(m[1], 16);
  const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  return 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
}

function bandInk(fill) {
  const L = relLuminance(fill);
  const vsBlack = (L + 0.05) / 0.05;
  const vsWhite = 1.05 / (L + 0.05);
  return vsBlack >= vsWhite ? "#0f0f0f" : "#ffffff";
}

// ── source image ───────────────────────────────────────────────────────────
// 有没有接线？与图片本身分开：没线是"接个图进来"，线连着但图没到是"跑一次"
// ——把两者说错会把用户引到错的地方。
function hasWire(node) {
  const slot = (node.inputs || []).find((i) => i && i.name === "image");
  return !!slot && slot.link != null;
}

// 第一层：上游节点已有的图（Load Image、Preview Image）。与 sourceImage 分开，
// 因为 executed 处理器问的正是这个问题——"浏览器是不是已经有了？"——绝不能被
// 我们自己缓存的 base 帧满足，否则第二次运行会跳过存档、预览永远停在第一张图。
function upstreamImage(node) {
  if (!hasWire(node)) return null;
  try {
    const slot = (node.inputs || []).find((i) => i && i.name === "image");
    const graph = node.graph || app.graph;
    // 新版前端的 graph.links 可能是 Map（Vue Compat #3）。
    let link = graph?.links?.[slot.link];
    if (!link && typeof graph?.links?.get === "function") link = graph.links.get(slot.link);
    const up = link && graph.getNodeById?.(link.origin_id);
    // MUTED（mode 2）/ BYPASSED（mode 4）的上游不再产出它仍在显示的图：
    // bypass 把自己的输入直接透传，它的预览是一张永远到不了这里的图。
    // 本节点要从那张图报尺寸（源尺寸、画布尺寸、哪个轴增长），信它就是在
    // 自信地说一个错的尺寸、画一张错的预览。返回 null 落到第二层。
    if (up && (up.mode === 2 || up.mode === 4)) return null;
    const img = up?.imgs?.[0];
    if (img && img.naturalWidth > 0 && img.naturalHeight > 0) return img;
  } catch (_e) { /* 未解析的线不是错误，只是未知的图 */ }
  return null;
}

// 要画的图，两层（Text Overlay 模式）：
//   1. 上游填了 imgs[0]（Load Image、Preview Image）——瞬时，无需运行
//   2. 没有（VAE Decode 之类链路中段）——Python 上次运行存到 temp/ 的帧，
//      由下面的 executed 处理器缓存
// 两者都以存在线为前提：上游已被移除后残留的 base 帧会是节点不再收到的
// 东西的图。
function sourceImage(node) {
  if (!hasWire(node)) return null;
  const up = upstreamImage(node);
  if (up) return up;
  const base = node._sfOpBaseImg;
  return base && base.naturalWidth > 0 ? base : null;
}

// 那张图的尺寸，未知时返回 null。
function sourceSize(node) {
  const img = sourceImage(node);
  return img ? { w: img.naturalWidth, h: img.naturalHeight } : null;
}

// 面板当前画着什么。刻意便宜：只读属性，无布局，watcher 可以无限跑下去
// 而不花一次 reflow。
function sourceSig(node) {
  const img = sourceImage(node);
  return img ? (img.src || "?") + "|" + img.naturalWidth + "x" + img.naturalHeight : "none";
}

// 上游的图异步到达且随时会变（用户在 Load Image 里换文件、一次运行换帧），
// 没有任何事件通知（Vue Compat #1），轮询是文档化答案。永久轮询而非短暂
// 爆发："加载后出现"与"换文件后跟上"是同一个问题。只在图真的变了才重绘，
// 节点离开图时自清。
function watchSource(node) {
  clearInterval(node._sfOpPoll);
  node._sfOpSrcSig = sourceSig(node);
  node._sfOpPoll = setInterval(() => {
    if (!node.graph) {
      clearInterval(node._sfOpPoll);
      node._sfOpPoll = null;
      return;
    }
    const sig = sourceSig(node);
    if (sig === node._sfOpSrcSig) return;
    // renderFace 重建所有行，会移除用户正在输入 pad 的字段、丢掉打了一半的
    // 字（移除聚焦元素不触发 blur，静默发生）。先只重画图、签名保持不变，
    // 下一跳焦点移走后重试整面重建。
    if (focusedPadInput(node)) { requestPreviewRedraw(node); return; }
    node._sfOpSrcSig = sig;
    // 行也依赖源（Add space 三件套跟随源宽高比），所以重画整面而非仅预览。
    renderFace(node);
  }, 400);
}

// ── CSS ────────────────────────────────────────────────────────────────────
// 模板字符串内不能有任何反引号（会提前结束字面量、静默禁用整个扩展），
// 也不能有 CSS unicode 转义（在模板字面量里是非法的八进制转义）。
function injectCSS() {
  if (document.getElementById("sf-outpaint-css")) return;
  const css = `
    .sf-op-root { position:relative; width:100%; height:100%; box-sizing:border-box;
      background:#1d1d1d; border-radius:4px; color:#ddd;
      font-family: ui-sans-serif, system-ui, sans-serif; font-size:11px; }
    /* flex 列在这里，绝不在 root 上：ComfyUI 每次重建/折叠都会把 root 强制成
       内联 display:block，会杀掉它。 */
    .sf-op-inner { position:absolute; inset:0; box-sizing:border-box;
      display:flex; flex-direction:column; gap:${ROW_GAP}px; padding:${PAD}px;
      user-select:none; }
    .sf-op-row { display:flex; align-items:stretch; gap:5px; flex:0 0 auto;
      flex-wrap:wrap; }

    /* 芯片：闲置 / 悬停 / 激活。悬停移动边框、提亮文字——填充会读成"激活"。 */
    .sf-op-chip { flex:1 1 auto; min-width:0; box-sizing:border-box;
      display:flex; align-items:center; justify-content:center;
      padding:6px 4px; border-radius:5px;
      background:#1d1d1d; border:1px solid #444; color:#aaa;
      cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
      transition:background .08s, border-color .08s, color .08s; }
    .sf-op-chip:hover { border-color:${BRAND}; color:#ddd; }
    .sf-op-chip.on { background:${BRAND}; border-color:${BRAND}; color:#fff; }
    /* 没有可点的：无指针、无悬停承诺。 */
    .sf-op-chip.dim { opacity:.4; cursor:default; }
    .sf-op-chip.dim:hover { border-color:#444; color:#aaa; }
    .sf-op-chip.dim.on:hover { border-color:${BRAND}; color:#fff; }

    /* 折叠箭头：固定宽，模式芯片拿到每个剩余像素。 */
    .sf-op-sq { flex:0 0 auto; width:30px; padding:6px 0; font-size:14px; line-height:1; }
    .sf-op-alabel { flex:0 0 auto; display:flex; align-items:center;
      color:#8a8a8a; padding-right:1px; white-space:nowrap; }

    /* By side 的 L/T/R/B 输入。复用共享 makeNumericInput（数学输入 + 键隔离），
       但丢掉其 13px 步进列：四个字段 + 重置在节点最小宽度下只有去掉箭头才
       放得下一行。min-width 让五个一起排一行而非换行。 */
    .sf-op-pad { flex:1 1 0; min-width:44px; min-height:28px; box-sizing:border-box;
      display:flex; align-items:center; gap:3px;
      background:#1d1d1d; border:1px solid #444; border-radius:5px;
      padding:0 4px 0 6px; }
    .sf-op-pad:focus-within { border-color:${BRAND}; }
    .sf-op-pad-l { flex:0 0 auto; font-size:9px; font-weight:700; color:#8a8a8a;
      letter-spacing:.5px; pointer-events:none; }
    /* 剥掉共享 wrapper 自己的盒子，只让 .sf-op-pad 画一个。 */
    .sf-op-pad .sf-li-numinput { flex:1 1 auto; min-width:0;
      border:none !important; background:transparent !important;
      border-radius:0 !important; }
    .sf-op-pad .sf-li-spin { display:none !important; }
    /* .sf-op-inner 上是 user-select:none 且会继承，会阻止用户拖选正要替换的数字。 */
    .sf-op-pad .sf-li-numinput input { padding:0 !important;
      text-align:right !important; color:${BRAND};
      user-select:text; }

    /* 重置：一键把四边归零。与箭头同款方块，放在它清空的数字旁边。 */
    .sf-op-reset { flex:0 0 auto; width:26px; min-height:28px; box-sizing:border-box;
      display:flex; align-items:center; justify-content:center;
      background:#1d1d1d; border:1px solid #444; border-radius:5px;
      color:#aaa; cursor:pointer; padding:0; font:inherit;
      transition:border-color .08s, color .08s; }
    .sf-op-reset:hover:not(:disabled) { border-color:${BRAND}; color:${BRAND}; }
    .sf-op-reset:focus-visible { border-color:${BRAND}; color:${BRAND}; outline:none; }
    /* 没有可重置的：真正惰性，而非只是变淡。 */
    .sf-op-reset:disabled { opacity:.4; cursor:default; }
    .sf-op-reset-ic { width:12px; height:12px; background-color:currentColor;
      -webkit-mask: url("${ICON_RESET}") center/12px 12px no-repeat;
              mask: url("${ICON_RESET}") center/12px 12px no-repeat; }

    /* 填充色 swatch。limit 行上可点（开取色器）。 */
    .sf-op-swatch { flex:0 0 auto; width:26px; border-radius:5px;
      border:1px solid #444; cursor:default; }
    .sf-op-swatch-btn { cursor:pointer; }
    .sf-op-swatch-btn:hover { border-color:${BRAND}; }

    /* 唯一 grower：flex:1 1 0 把行没用掉的每个像素都给它。min-height 必须 0，
       下限在 measureFloor 里。真 CSS min 看着诱人但会反噬：flex 项不能缩到
       它之下，每当节点体比下限紧（Nodes 2.0 的 chrome 比 legacy computeSize
       估计高），预览拒绝收缩、溢出到分类 chip 上。设 0 只会变小，是优雅
       降级而非坏掉的节点。（min-height 默认 auto = 内容高，必须显式设。） */
    .sf-op-prev { position:relative; flex:1 1 0; min-height:0;
      border-radius:4px; background:#151515; overflow:hidden; }
    /* 按 inset 填充而非 flex：canvas 不关心宿主给父级什么 display。 */
    .sf-op-prev canvas { position:absolute; inset:0; width:100%; height:100%;
      display:block; }
  `;
  const s = document.createElement("style");
  s.id = "sf-outpaint-css";
  s.textContent = css;
  document.head.appendChild(s);
}

// ── row builders ───────────────────────────────────────────────────────────
function chip(text, on, title) {
  const el = document.createElement("div");
  el.className = "sf-op-chip" + (on ? " on" : "");
  el.textContent = text;
  if (title) el.title = title;
  return el;
}

function row(host) {
  const el = document.createElement("div");
  el.className = "sf-op-row";
  host.appendChild(el);
  return el;
}

function apply(node, patch) {
  writeState(node, patch);
  renderFace(node);
  node.setDirtyCanvas?.(true, true);
}

// 0 显示 "Off"，否则 "N MP"——每个值都带 MP 后缀，自定义 1.3 读作 "1.3 MP"
// 而非光秃秃的 "1.3"。
function limitLabel(v) {
  return v === 0 ? "Off" : v + " MP";
}

function renderModeRow(node, host) {
  const st = readState(node);
  const folded = !!st.collapsed;

  // ▾ 展开，▸ 折叠。永远存活——折叠控件自己绝不能被藏起来。
  const chevron = chip(folded ? "▸" : "▾", false,
    folded ? "Expand the settings" : "Collapse to the picture");
  chevron.classList.add("sf-op-sq");
  chevron.onclick = () => toggleFold(node);
  host.appendChild(chevron);

  if (folded) {
    // 摘要而非空白：折叠的节点仍要说清自己设了什么。这里只读。
    const shape = st.mode === "ratio" ? st.ratio : "By side";
    const sc = chip(shape, true, "Current: " + (st.mode === "ratio" ? "grow to " + shape : "add by side"));
    sc.style.cursor = "default";
    host.appendChild(sc);
    const lc = chip(limitLabel(st.limit), st.limit !== 0, st.limit === 0
      ? "No megapixel limit" : "Capped at " + st.limit + " MP");
    lc.style.cursor = "default";
    host.appendChild(lc);
    const sw = document.createElement("div");
    sw.className = "sf-op-swatch";
    sw.style.background = st.color;
    sw.title = "Fill colour: " + st.color;
    host.appendChild(sw);
  } else {
    for (const [value, text, tip] of [
      ["ratio", "To ratio", "Grow the image to a target shape"],
      ["sides", "By side", "Add an exact number of pixels per edge"],
    ]) {
      const c = chip(text, st.mode === value, tip);
      c.onclick = () => apply(node, { mode: value });
      host.appendChild(c);
    }
  }
}

function renderRatioRow(node, host) {
  const st = readState(node);

  // By side 模式完全无视 ratio——两个引擎都按 mode 分支、那边从不读它——所以
  // 这些芯片点了毫无作用：点 3:2 节点不动。隐藏它们，正如 Add space 行已
  // 隐藏自己，而不是在面板上留死控件。所选 ratio 保留在状态里，切回 To ratio
  // 时恢复。
  host.style.display = st.mode === "ratio" ? "" : "none";
  if (st.mode !== "ratio") return;

  for (const r of DEFAULT_RATIOS) {
    const c = chip(r, st.ratio === r, "Grow the image to " + r);
    c.onclick = () => apply(node, { ratio: r });
    host.appendChild(c);
  }
}

function renderAnchorRow(node, host) {
  const st = readState(node);
  const src = sourceSize(node);

  // By side 模式：每边数字已经说清一切，这里再放 anchor 就是第二个、会打架
  // 的说法。
  host.style.display = st.mode === "ratio" ? "" : "none";
  if (st.mode !== "ratio") return;

  // null 覆盖两种不同的情况，绝不能混淆：
  //   src === null  -> 源尺寸未知（还没接线）
  //   axis === null -> 源已知，且这个比例什么都不增长
  const axis = src ? anchorAxis(st.ratio, src.w, src.h) : null;
  const grows = !!axis;
  const shown = axis || "h"; // 源未知：显示横向三件套

  // "Both" 而非 "Centre"：中间选项把新空间平分到两边，而 "add space in the
  // centre" 会读成在图片中间加。
  const labels = shown === "v"
    ? [["top", "Top"], ["middle", "Both"], ["bottom", "Bottom"]]
    : [["left", "Left"], ["centre", "Both"], ["right", "Right"]];

  // 持久化 remap：3:2 -> 9:16 翻转时保持"贴远边"而非静默重置到中间。只在活动
  // 轴真正已知时：未接线的节点显示横向三件套作占位，对着那个猜测 remap 存储
  // 的纵向 anchor 会毁掉它。加载路径绝不写（Vue Compat #18）。
  const live = grows ? remapAnchor(st.anchor, axis) : st.anchor;
  if (live !== st.anchor && !isGraphLoading()) writeState(node, { anchor: live });

  // 行高亮什么，总是用当前显示三件套的词汇，让存储的跨轴 anchor 仍点亮一颗
  // 芯片。仅显示，绝不写。
  const sel = remapAnchor(live, shown);

  const lbl = document.createElement("span");
  lbl.className = "sf-op-alabel";
  lbl.textContent = "Add space"; // 不是 "Anchor"——见 padsForRatio 的注释
  host.appendChild(lbl);

  for (const [value, text] of labels) {
    const c = chip(text, sel === value);
    if (!grows) {
      c.classList.add("dim");
      c.title = src
        ? "This ratio matches the image, so there is nothing to add"
        : "Wire an image in to choose which side the new space goes on";
    } else {
      c.title = value === "centre" || value === "middle"
        ? "Split the new space evenly across both sides"
        : "Put the new space on the " + text.toLowerCase();
      c.onclick = () => apply(node, { anchor: value });
    }
    host.appendChild(c);
  }
}

// ── By side：键入四个 pad 值 ───────────────────────────────────────────────
// 数字按眼睛读一帧的顺序：左 上 右 下。
const PAD_SIDES = [["left", "L", "Left"], ["top", "T", "Top"],
                   ["right", "R", "Right"], ["bottom", "B", "Bottom"]];

// 只在 By side 模式显示。ratio 模式下 padsForState 无视全部四个值，那里的
// 字段就是可见地什么都不做的控件——与 ratio 和 Add space 行在另一方向隐藏
// 自己的理由相同。ratio 模式重置已是一颗比例芯片的事，所以重置跟它们一起藏。
function renderPadRow(node, host) {
  const st = readState(node);
  host.style.display = st.mode === "sides" ? "" : "none";
  if (st.mode !== "sides") return;

  const inputs = {};
  for (const [key, letter, name] of PAD_SIDES) {
    const cell = document.createElement("div");
    cell.className = "sf-op-pad";
    const tip = name + " edge: pixels of fill to add. Maths works, e.g. 512*2.";
    cell.title = tip;
    const lab = document.createElement("span");
    lab.className = "sf-op-pad-l";
    lab.textContent = letter;
    cell.appendChild(lab);

    // opts 对象被持有而非丢弃：makeNumericInput 会改这个对象本身，在输入文本
    // 解析失败时读 opts.value 作回退（也是空字段箭头步进的基础）。syncPadInputs
    // 必须同时刷新 opts.value 与可见文本——只写 el.value 会把回退值困在行构建
    // 时的数字上。
    //
    // sidePad 与 Python 解析状态时的夹紧相同，输入值永远预览不了运行会丢弃的 pad。
    const opts = {
      value: sidePad(st[key]),
      min: 0, max: MAX_PAD, step: 1,
      format: (v) => String(Math.round(v)),
      onCommit: (v) => commitPad(node, key, v),
    };
    const built = makeNumericInput(opts);
    built.input.title = tip;
    cell.appendChild(built.wrap);
    inputs[key] = { el: built.input, opts };
    // 点字段不能落到下面的 LiteGraph 画布上，否则光标点进框的瞬间节点就被
    // 拖走了。
    cell.addEventListener("pointerdown", (e) => e.stopPropagation());
    host.appendChild(cell);
  }
  node._sfOpPadInputs = inputs;

  // 真 <button> 而非样式 div：免费得到可访问名、焦点环与 Enter/Space，
  // disabled 让"没什么可重置"真正惰性而非仅变淡。
  const rst = document.createElement("button");
  rst.type = "button";
  rst.className = "sf-op-reset";
  rst.setAttribute("aria-label", "Reset all four edges to 0");
  const ic = document.createElement("span");
  ic.className = "sf-op-reset-ic";
  rst.appendChild(ic);
  rst.onclick = () => resetPads(node);
  host.appendChild(rst);
  refreshResetState(node); // 按实时状态设 disabled + title
}

// 当前获得焦点的 pad 输入，或 null。用来避免两种悄悄吃掉打了一半值的方式。
function focusedPadInput(node) {
  const inputs = node._sfOpPadInputs;
  if (!inputs) return null;
  for (const [key] of PAD_SIDES) {
    const f = inputs[key];
    if (f && f.el === document.activeElement) return f.el;
  }
  return null;
}

// 输入值刻意不走 apply()：那会调用 renderFace 重建每一行——毁掉用户正在输入
// 的框、连光标一起带走。走拖拽的路：写状态、重画图与尺寸徽章，行保持原样。
function commitPad(node, key, value) {
  const v = sidePad(value);
  if (sidePad(readState(node)[key]) === v) return; // 实际没动
  writeState(node, { [key]: v });
  requestPreviewRedraw(node);         // 重画图（含尺寸徽章）
  node.setDirtyCanvas?.(true, true);
  refreshResetState(node);            // 0 -> 非 0 重新启用重置按钮
}

// 一键回到没有填充。一次性动作，完整 renderFace 没问题——也正是它把四个框
// 重新填 0、按钮重新变淡。
function resetPads(node) {
  const st = readState(node);
  if (!st.left && !st.top && !st.right && !st.bottom) return;
  apply(node, { left: 0, top: 0, right: 0, bottom: 0 });
}

// 不重建行地启用/变淡重置（重建会把光标从输入中踢出去）。每次 commit 与
// 每次拖拽帧都调用都够便宜。
function refreshResetState(node) {
  const ui = node._sfOpUI;
  const btn = ui && ui.inner.querySelector(".sf-op-reset");
  if (!btn) return;
  const st = readState(node);
  const empty = !st.left && !st.top && !st.right && !st.bottom;
  btn.disabled = empty;
  btn.title = empty ? "Nothing to reset - all four edges are already 0"
                    : "Reset all four edges to 0";
}

// 拖拽写状态时不重建行（120Hz 绝不能），框会停在旧数字上。把实时值直接推进
// 输入框。
function syncPadInputs(node) {
  const inputs = node._sfOpPadInputs;
  if (!inputs) return;
  const st = readState(node);
  for (const [key] of PAD_SIDES) {
    const f = inputs[key];
    // 绝不跟用户正在输入的字段打架。
    if (!f || !f.el.isConnected || f.el === document.activeElement) continue;
    const v = sidePad(st[key]);
    // 两个都写，总是：可见文本与字段自己的解析失败回退。见 opts 构建处的
    // 注释——只更新文本会留下一个能让拖拽前功尽弃的旧回退值。
    f.opts.value = v;
    const text = String(v);
    if (f.el.value !== text) f.el.value = text;
  }
  refreshResetState(node);
}

function renderLimitRow(node, host) {
  const st = readState(node);
  // Number(st.limit) 让存成字符串的值也能匹配。
  const active = Number(st.limit);
  for (const v of LIMITS) {
    const c = chip(limitLabel(v), v === active, v === 0
      ? "Keep the padded size"
      : "Scale the padded image to " + v + " megapixels");
    c.onclick = () => apply(node, { limit: v });
    host.appendChild(c);
  }
  // 填充色与改变它的唯一控件。这是外绘模型重画的颜色，用的是完整调色板
  // （中性色与鲜艳色都要）——LoRA 可能想要纯绿、纯白或纯黑。
  const sw = document.createElement("div");
  sw.className = "sf-op-swatch sf-op-swatch-btn";
  sw.style.background = st.color;
  sw.title = "Fill colour (click to change): " + st.color;
  sw.onclick = () => {
    openSimpleColorPicker({
      initialColor: readState(node).color,
      // 重置回到中性灰默认：那是不带色调的安全填充。
      onPick: (c) => {
        const v = /^#[0-9a-f]{6}$/i.test(c || "") ? c : DEFAULT_STATE.color;
        writeState(node, { color: v });
        renderFace(node); // 重新上色 swatch 与预览的 band + ink
        node.setDirtyCanvas?.(true, true);
      },
    });
  };
  host.appendChild(sw);
}

// ── preview drawing ────────────────────────────────────────────────────────
// 构图在预览框内的位置。纯函数（框、源、pad），绘制与命中测试不会漂移。
function previewGeom(cssW, cssH, src, pads) {
  const padW = src.w + pads.left + pads.right;
  const padH = src.h + pads.top + pads.bottom;
  // 适配 PADDED 矩形而非图片：预览是构图，填充必须与真实输出同比例地
  // 待在框内。
  const scale = Math.min((cssW - PREVIEW_INSET * 2) / padW, (cssH - PREVIEW_INSET * 2) / padH);
  const dw = padW * scale;
  const dh = padH * scale;
  return { scale, dw, dh, ox: (cssW - dw) / 2, oy: (cssH - dh) / 2 };
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") { ctx.roundRect(x, y, w, h, r); return; }
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// 纯数字串在 textBaseline "middle" 下视觉偏高：em 盒给下行部分留了数字永不
// 使用的空间，且间隙随字号增长。按真实字形盒居中。本预览每个读数都是数字，
// 所以这里是默认而非例外。
function fillTextVCenter(ctx, text, cx, cyMid) {
  const m = ctx.measureText(text);
  if (m.actualBoundingBoxAscent != null && m.actualBoundingBoxDescent != null) {
    ctx.textBaseline = "alphabetic";
    ctx.fillText(text, cx, cyMid + (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2);
  } else {
    ctx.textBaseline = "middle"; // 很老的浏览器：略高胜过不画
    ctx.fillText(text, cx, cyMid);
  }
}

const PILL_H = 15;
const PILL_GAP = 4; // 跳出的数字距图片边缘的距离

function pillW(ctx, text) { return ctx.measureText(text).width + 8; }

// 文字背后的深色胶囊。数字一离开填充色就必须有它：落在照片上时，任何固定
// 颜色墨水都会在某种图上消失。
function pill(ctx, text, cx, cyMid) {
  const w = pillW(ctx, text);
  ctx.fillStyle = "rgba(0,0,0,.72)";
  roundRect(ctx, cx - w / 2, cyMid - PILL_H / 2, w, PILL_H, 3);
  ctx.fill();
  // 中性而非着色：它待在照片上、有自己的深色胶囊，与填充色无关。
  ctx.fillStyle = "#f0f0f0";
  fillTextVCenter(ctx, text, cx, cyMid);
}

// 一个 pad 数字。够厚的带把字画在填充上（近黑色）；太薄带不下，数字跳到
// 图片内侧的胶囊上——这让 32px 的 pad 可读而非剪成一条糊。
function bandNumber(ctx, px, thick, onCx, onCy, offCx, offCy, ink) {
  if (px <= 0) return;
  const text = String(px);
  if (thick >= BAND_TEXT_MIN) {
    ctx.fillStyle = ink;
    fillTextVCenter(ctx, text, onCx, onCy);
  } else {
    pill(ctx, text, offCx, offCy);
  }
}

function drawBandNumbers(ctx, pads, scale, ox, oy, dw, dh, ink) {
  ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  const midX = ox + dw / 2;
  const midY = oy + dh / 2;
  const t = pads.top * scale, b = pads.bottom * scale;
  const l = pads.left * scale, r = pads.right * scale;

  bandNumber(ctx, pads.top, t, midX, oy + t / 2,
    midX, oy + t + PILL_GAP + PILL_H / 2, ink);
  bandNumber(ctx, pads.bottom, b, midX, oy + dh - b / 2,
    midX, oy + dh - b - PILL_GAP - PILL_H / 2, ink);
  bandNumber(ctx, pads.left, l, ox + l / 2, midY,
    ox + l + PILL_GAP + pillW(ctx, String(pads.left)) / 2, midY, ink);
  bandNumber(ctx, pads.right, r, ox + dw - r / 2, midY,
    ox + dw - r - PILL_GAP - pillW(ctx, String(pads.right)) / 2, midY, ink);
}

// 真相，相对于图：百万像素封顶后真实输出比上面构图暗示的要小，所以最终
// 数字必须直接说出来，而非从画面上推断。
function drawSizeBadge(ctx, cssW, cssH, fin) {
  const text = fin.w + " × " + fin.h;
  ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  const w = ctx.measureText(text).width + 12;
  const h = 17;
  const cx = cssW - PREVIEW_INSET - w / 2;
  const cy = cssH - PREVIEW_INSET - h / 2;
  ctx.fillStyle = "rgba(0,0,0,.72)";
  roundRect(ctx, cx - w / 2, cy - h / 2, w, h, 3);
  ctx.fill();
  ctx.fillStyle = "#ddd";
  fillTextVCenter(ctx, text, cx, cy);
}

function drawEmptyPreview(ctx, w, h, wired) {
  ctx.save();
  ctx.strokeStyle = "#3a3a3a";
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  roundRect(ctx, 4.5, 4.5, Math.max(0, w - 9), Math.max(0, h - 9), 4);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "#6a6a6a";
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle"; // 有下行，数字修正不适用
  // 有线无图与无线是不同的问题，让用户修错的地方浪费他们的时间。
  ctx.fillText(wired ? "Run once to see the preview" : "Connect an image", w / 2, h / 2);
}

function renderPreview(node) {
  const ui = node._sfOpUI;
  if (!ui || !ui.prev || !ui.canvas) return;
  const cssW = ui.prev.clientWidth;
  const cssH = ui.prev.clientHeight;
  if (cssW <= 0 || cssH <= 0) return; // 还没布局——observer 会回调

  // backing store 按 DPR x 图缩放：节点体被 CSS transform 缩放，只按布局像素
  // 画会在放大时发糊。
  const s = canvasBackingScale(cssW, cssH);
  const bw = Math.max(1, Math.round(cssW * s));
  const bh = Math.max(1, Math.round(cssH * s));
  if (ui.canvas.width !== bw || ui.canvas.height !== bh) {
    ui.canvas.width = bw;
    ui.canvas.height = bh;
  }
  const ctx = ui.canvas.getContext("2d");
  ctx.setTransform(s, 0, 0, s, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const img = sourceImage(node);
  if (!img) { drawEmptyPreview(ctx, cssW, cssH, hasWire(node)); return; }

  const st = readState(node);
  const src = { w: img.naturalWidth, h: img.naturalHeight };
  const pads = padsForState(st, src.w, src.h);
  const { scale, dw, dh, ox, oy } = previewGeom(cssW, cssH, src, pads);

  // 填充色在底下、图片在上面：四条带就是图片没盖住的填充色，绝不会与数学
  // 脱节。
  ctx.fillStyle = st.color;
  ctx.fillRect(ox, oy, dw, dh);
  ctx.drawImage(img, ox + pads.left * scale, oy + pads.top * scale,
    src.w * scale, src.h * scale);

  ctx.strokeStyle = "rgba(255,255,255,.14)";
  ctx.lineWidth = 1;
  ctx.strokeRect(ox + 0.5, oy + 0.5, Math.max(0, dw - 1), Math.max(0, dh - 1));

  drawBandNumbers(ctx, pads, scale, ox, oy, dw, dh, bandInk(st.color));
  drawSizeBadge(ctx, cssW, cssH, finalSize(src.w, src.h, pads, st.limit, st.snap));
}

// ── drag the green edges ───────────────────────────────────────────────────
const HANDLE_HIT = 7;      // 距边多少算抓住它
const CURSORS = { left: "ew-resize", right: "ew-resize", top: "ns-resize", bottom: "ns-resize" };

// 指针下的边，或 null。四条都可抓，包括还没有填充的：把边向外拖就是加空间
// 的方式。
function hitEdge(lx, ly, g) {
  const { ox, oy, dw, dh } = g;
  const nearY = ly >= oy - HANDLE_HIT && ly <= oy + dh + HANDLE_HIT;
  const nearX = lx >= ox - HANDLE_HIT && lx <= ox + dw + HANDLE_HIT;
  // 角上左右优先，任意但一致。
  if (nearY && Math.abs(lx - ox) <= HANDLE_HIT) return "left";
  if (nearY && Math.abs(lx - (ox + dw)) <= HANDLE_HIT) return "right";
  if (nearX && Math.abs(ly - oy) <= HANDLE_HIT) return "top";
  if (nearX && Math.abs(ly - (oy + dh)) <= HANDLE_HIT) return "bottom";
  return null;
}

// getBoundingClientRect 报的是屏幕像素而预览按布局像素画，节点体又被图缩放
// 的 CSS transform 缩放——不修正的话，抓取点随缩放越大偏得越远。
function localPos(el, e) {
  const r = el.getBoundingClientRect();
  const sx = r.width ? el.clientWidth / r.width : 1;
  const sy = r.height ? el.clientHeight / r.height : 1;
  return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
}

// 指针需要的当前构图的一切，或 null（没有可抓的东西）。
function previewState(node) {
  const ui = node._sfOpUI;
  const img = ui && sourceImage(node);
  if (!img) return null;
  const cssW = ui.prev.clientWidth, cssH = ui.prev.clientHeight;
  if (cssW <= 0 || cssH <= 0) return null;
  const st = readState(node);
  const src = { w: img.naturalWidth, h: img.naturalHeight };
  const pads = padsForState(st, src.w, src.h);
  return { st, src, pads, g: previewGeom(cssW, cssH, src, pads) };
}

function installDrag(node) {
  const prev = node._sfOpUI.prev;

  prev.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    // 在读几何之前先提交并释放仍持焦点的 pad 框。否则 syncPadInputs 故意跳过
    // 聚焦字段 + 下面 preventDefault 阻止浏览器把焦点移走——那个框在整个手势
    // 期间显示拖拽前的数字，最终 blur 时提交旧文本，静默撤销拖拽。先 blur 也
    // 意味着 d.pads 从用户真正输入的数字起步。
    focusedPadInput(node)?.blur();
    const ps = previewState(node);
    if (!ps) return;
    const [lx, ly] = localPos(prev, e);
    const side = hitEdge(lx, ly, ps.g);
    if (!side) return;
    // 不阻止的话下面的画布吃掉按下事件，整个节点从光标下被拖走。
    e.stopPropagation();
    e.preventDefault();
    try { prev.setPointerCapture(e.pointerId); } catch (_e) { /* 没有也行 */ }
    node._sfOpDrag = {
      side, x: lx, y: ly, pads: { ...ps.pads },
      // 抓取瞬间的 scale 在整个手势期间持有。构图会随 pad 增长重新缩放以
      // 保持放入，实时读 scale 会让同样的光标位移在不同帧值不同像素——
      // 拖拽会感觉在滑。
      scale: ps.g.scale,
      needsFlip: ps.st.mode === "ratio",
    };
  });

  prev.addEventListener("pointermove", (e) => {
    const d = node._sfOpDrag;
    if (!d) {
      // 空闲：只广告哪些边有动作。
      const ps = previewState(node);
      const side = ps ? hitEdge(...localPos(prev, e), ps.g) : null;
      prev.style.cursor = side ? CURSORS[side] : "";
      return;
    }
    const [lx, ly] = localPos(prev, e);
    // 光标位移 -> 源像素。这条边向外 = 更多填充；向外拉长画布、缩小框内
    // 图片，读作放大缩小——正是问模型要的东西。
    const dx = (lx - d.x) / d.scale, dy = (ly - d.y) / d.scale;
    const grow = d.side === "right" ? dx : d.side === "left" ? -dx
      : d.side === "bottom" ? dy : -dy;
    const patch = {};
    if (d.needsFlip) {
      // ratio 模式拖拽的第一下移动切到 By side，带上 ratio 算出的数字。
      // 不带的话，用户一碰边，其余三边立刻归零。
      d.needsFlip = false;
      Object.assign(patch, d.pads, { mode: "sides" });
    }
    patch[d.side] = Math.max(0, Math.min(Math.round(d.pads[d.side] + grow), MAX_PAD));
    writeState(node, patch);
    // 模式芯片和 Add space 行只在翻转时变化；其余每次移动都只有图在变，
    // 每秒 120 次重建行毫无必要。pad 框仍要跟上，直接写。
    if (patch.mode) renderFace(node);
    else { syncPadInputs(node); requestPreviewRedraw(node); }
  });

  const end = (e) => {
    if (!node._sfOpDrag) return;
    node._sfOpDrag = null;
    try { prev.releasePointerCapture(e.pointerId); } catch (_e) { /* already released */ }
    node.setDirtyCanvas?.(true, true);
  };
  prev.addEventListener("pointerup", end);
  prev.addEventListener("pointercancel", end);
  prev.addEventListener("pointerleave", () => {
    if (!node._sfOpDrag) prev.style.cursor = "";
  });
}

function renderFace(node) {
  const ui = node._sfOpUI;
  if (!ui) return;
  const inner = ui.inner;
  // 只重建行。预览元素是持久的：它持有 canvas（还有 ResizeObserver 与拖拽
  // 监听），每次点芯片都重建会每次漏一个 observer、白白扔掉好的 backing store。
  for (const el of [...inner.children]) {
    if (el !== ui.prev) el.remove();
  }
  // pad 字段在刚被移除的行上，缓存的引用已死，直到 renderPadRow 交还新鲜的。
  // 在这里清（而非 renderPadRow 里）也覆盖折叠路径——折叠从不调用它——
  // 否则 syncPadInputs 会写进已脱离的输入。
  node._sfOpPadInputs = null;
  renderModeRow(node, row(inner));
  // 折叠丢三个控制行，保留模式摘要 + 预览。上面的模式行已经渲染了摘要形；
  // 跳过其余就是折叠的全部。renderFace 每条路径都读 collapsed（含 onConfigure），
  // 所以加载时折叠立即生效。
  if (!readState(node).collapsed) {
    renderRatioRow(node, row(inner));
    renderAnchorRow(node, row(inner));
    // 仅 By side。总高度无增减：该模式隐藏上面两行，三行对四行。
    renderPadRow(node, row(inner));
    renderLimitRow(node, row(inner));
  }
  inner.appendChild(ui.prev);
  renderPreview(node);
}

// ── fold ───────────────────────────────────────────────────────────────────
// 纯用户动作。折叠记住打开时的高度，展开时恢复同样的大预览而非弹到裸地板。
// 两个写都是用户驱动的，绝不弄脏加载（Vue Compat #18）。renderFace 做视觉
// 半边，这里做高度半边。
function toggleFold(node) {
  const st = readState(node);
  const collapsed = !st.collapsed;
  writeState(node, collapsed
    ? { collapsed: true, openH: node.size[1] } // 先藏起来再缩小
    : { collapsed: false });
  renderFace(node);
  fitFoldHeight(node);
  node.setDirtyCanvas?.(true, true);
}

// 按折叠状态调整尺寸。绝不在加载路径上（保存的高度已匹配保存的折叠状态；
// 那里写 size 会把重开的工作流标"已修改"），也绝不在 LiteGraph 标题折叠时
// （所有子元素都读不到，测量只返回标题栏高度）。整个数组 setSize：
// 只写 node.size[1] 会绕过 Vue 布局 store。
function fitFoldHeight(node) {
  if (isGraphLoading() || node.flags?.collapsed) return;
  const st = readState(node);
  const w = node.size[0];

  // 展开：直接回到折叠前的高度，预览恢复旧的大尺寸而非裸地板。
  if (!st.collapsed) {
    node.setSize?.([w, st.openH || DEFAULT_H]);
    node.setDirtyCanvas?.(true, true);
    return;
  }

  // 折叠：缩到内容地板。一帧后隐藏行真正消失，再按地板与当前根高度之差
  // 调整 node.size（measureFloor 按预览最小值而非那帧 flex 长出的高度数）。
  requestAnimationFrame(() => {
    if (!node.graph) return;
    const ui = node._sfOpUI;
    if (!ui || !ui.root.isConnected || ui.root.clientWidth === 0) return;
    const deficit = measureFloor(node) - ui.root.clientHeight;
    node.setSize?.([w, Math.round(node.size[1] + deficit)]);
    node.setDirtyCanvas?.(true, true);
  });
}

// ── height ─────────────────────────────────────────────────────────────────
// 累加已布局的行。拒绝测量未挂载或零宽的 root：行会对着零宽换行、总和爆炸、
// 节点永久膨胀。4px 取整防字体抖动让每次打开工作流都长高一点（Vue Compat #18）。
function measureFloor(node) {
  const ui = node._sfOpUI;
  if (!ui || !ui.root.isConnected || ui.root.clientWidth === 0) {
    return ui?._floorCache ?? FLOOR_MIN;
  }
  let h = 0;
  let shown = 0;
  for (const child of ui.inner.children) {
    if (child.style.display === "none") continue; // By side 模式下的 anchor 行
    // 预览按它的最小值计，绝不按长出的高度。它是 flex grower，offsetHeight
    // 是节点恰好有的余量——把它反馈成地板会棘轮：节点能长不能缩，因为每次
    // 测量都把上次的尺寸报成新最小值。
    h += (child === ui.prev) ? PREVIEW_MIN : child.offsetHeight;
    shown++;
  }
  if (!shown) return ui._floorCache ?? FLOOR_MIN;
  h += (shown - 1) * ROW_GAP + PAD * 2;
  ui._floorCache = Math.min(Math.max(Math.round(h / 4) * 4, FLOOR_MIN), FLOOR_CAP);
  return ui._floorCache;
}

// ComfyUI 的 loadGraphData 对每个节点跑一次 fit 过：size = max(保存的,
// computeSize())。保存得比自身 computeSize 短的节点下次打开会长高，干净的
// 工作流被标"已修改"（Vue Compat #18）。本节点出生就短，因为两条尺寸路径
// 不一致：live _arrangeWidgets 落在 slots+widget，而 computeSize 加了略大的
// chrome 估计。出生时镜像一次加载过，保存的高度已是加载会产出的高度。
// 仅新节点——configure() 拥有已加载节点的尺寸。
function snapFresh(node, tries = 0) {
  requestAnimationFrame(() => {
    if (!node.graph || node._sfOpConfigured || isGraphLoading()) return;
    const ui = node._sfOpUI;
    // computeSize 只在 widget 有宽度后才可信：measureFloor 在那之前拒绝猜测。
    // 给布局几帧，然后照拍（拖到屏外的节点永远没有）。
    if ((!ui || !ui.root.isConnected || ui.root.clientWidth === 0) && tries < 20) {
      snapFresh(node, tries + 1);
      return;
    }
    let want = node.computeSize?.()?.[1] || 0;
    // computeSize 估计的是 legacy chrome。Nodes 2.0 给节点体包了更多（自己的
    // 槽带、分类 chip 页脚），同样的高度在那下面 widget 区会短、预览被挤成
    // 一条。量真实差额而非硬编码 chrome 常量——前端更新会悄悄烂掉。
    if (ui && ui.root.isConnected && ui.root.clientWidth > 0) {
      const deficit = measureFloor(node) - ui.root.clientHeight;
      if (deficit > 1) want = Math.max(want, node.size[1] + deficit);
    }
    if (want > 0 && node.size[1] < want - 1) {
      node.setSize?.([node.size[0], want]);
      node.setDirtyCanvas?.(true, true);
      // 一次修正过：setSize 重跑布局，Nodes 2.0 第一次测量对着旧节点体高，
      // 差额只在新的落定后显现。tries 有界。
      if (tries < 20) snapFresh(node, tries + 1);
    }
  });
}

// ── setup ──────────────────────────────────────────────────────────────────
function setupNode(node) {
  const root = document.createElement("div");
  root.className = "sf-op-root";
  const inner = document.createElement("div");
  inner.className = "sf-op-inner";
  root.appendChild(inner);

  const prev = document.createElement("div");
  prev.className = "sf-op-prev";
  const canvas = document.createElement("canvas");
  prev.appendChild(canvas);
  inner.appendChild(prev);

  node._sfOpUI = { root, inner, prev, canvas, _floorCache: FLOOR_MIN };

  const repaintCanvases = () => { renderPreview(node); };

  // node.onResize 对 DOM widget 不可靠（Vue Compat #13），直接观察元素：
  // 节点缩放、渲染器 reflow、切标签页都抓到，无论起因。
  node._sfOpRO = new ResizeObserver(repaintCanvases);
  node._sfOpRO.observe(prev);

  // 图缩放不改任何布局盒，上面的 observer 永远看不到——但会改 backing scale，
  // 不处理的话像素停在首画分辨率、放大发糊。
  node._sfOpZoomOff = installZoomRepaint(
    node, () => [prev.clientWidth, prev.clientHeight], repaintCanvases, "_sfOpZoomRaf");

  // 监听器挂在持久的预览元素上，一次装好终身使用——renderFace 刻意绕开它。
  installDrag(node);

  // 无自定义 computeSize 也无 getMaxHeight：二者都会让 widget 在 legacy 变
  // 固定高度，节点只能长不能缩。minWidth 1 否则保存的节点宽度不能往返。
  const w = node.addDOMWidget("outpaint_ui", "sf_image_outpaint", root, {
    serialize: false,
    getMinHeight: () => measureFloor(node),
  });
  w.computeLayoutSize = () => ({ minHeight: measureFloor(node), minWidth: 1 });
  applyAdaptiveCanvasOnly(w);
  // 预览上的滚轮仍要缩放画布（Classic；Nodes 2.0 空操作）。与绿色边拖拽
  // 无关，后者是指针驱动的。
  installCanvasZoomPassthrough(root);

  // 仅新节点，且同步：configure() 在 onNodeCreated 之后运行，把已加载节点的
  // 保存尺寸盖回来。微任务会跑到 configure() 之后、每次打开工作流都覆盖用户
  // 自己的尺寸。用下标赋值而非替换数组——reactive proxy 可能持有它。
  if (node.size[0] < MIN_W) node.size[0] = DEFAULT_W;
  if (node.size[1] < DEFAULT_H) node.size[1] = DEFAULT_H;

  // 首画推迟到 configure() 之后，让恢复的工作流渲染其保存状态而非默认
  // （Vue Compat #8）。
  queueMicrotask(() => {
    renderFace(node);
    watchSource(node);
    snapFresh(node);
  });
}

// ── 扩展注册 ───────────────────────────────────────────────────────────────
app.registerExtension({
  name: "sfnodes.SFImageOutpaint",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;
    if (nodeType.prototype._sfOpPatched) return; // 热重载守卫
    nodeType.prototype._sfOpPatched = true;

    injectCSS();
    injectResizePanelCSS(); // makeNumericInput 的 .sf-li-* 基础样式

    const _origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      // 来自保存工作流的节点尺寸已定：snapFresh 不得碰它。
      this._sfOpConfigured = true;
      const r = _origConfigure?.apply(this, arguments);
      // 只画：renderFace 不碰序列化状态，里面的 anchor remap 以 isGraphLoading
      // 为门。
      if (this._sfOpUI) { renderFace(this); watchSource(this); }
      return r;
    };

    const _origConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link, ioSlot) {
      const r = _origConn?.apply(this, arguments);
      // 接线的图决定 Add space 行显示哪个三件套，任何连线变化都重画。加载
      // 重放期间安全（Vue Compat #19）：只画，remap 写以 isGraphLoading 为门。
      if (this._sfOpUI) { renderFace(this); watchSource(this); }
      return r;
    };

    const _origRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      // 释放每个 watcher。各自持有本节点（observer 还钉住构建好的 DOM），
      // 中途删除节点会整包泄漏。
      clearInterval(this._sfOpPoll);
      this._sfOpPoll = null;
      try { this._sfOpRO?.disconnect(); } catch (_e) { /* already gone */ }
      this._sfOpRO = null;
      this._sfOpZoomOff?.();
      this._sfOpZoomOff = null;
      if (this._sfOpRaf) cancelAnimationFrame(this._sfOpRaf);
      this._sfOpRaf = null;
      this._sfOpDrag = null;
      this._sfOpPadInputs = null;
      return _origRemoved?.apply(this, arguments);
    };
  },

  // 右键菜单：折叠与重置的第二种入口。重置只在 By side 模式可用（ratio 模式
  // 的 pad 由比例推导，一颗比例芯片就是回去的路）。
  getNodeMenuItems(node) {
    if (node?.comfyClass !== CLASS) return [];
    const st = readState(node);
    const folded = !!st.collapsed;
    const padded = !!(st.left || st.top || st.right || st.bottom);
    return [
      null,
      { content: folded ? "▸ Expand" : "▾ Collapse", callback: () => toggleFold(node) },
      {
        content: "↺ Reset padding",
        disabled: st.mode !== "sides" || !padded,
        callback: () => resetPads(node),
      },
    ];
  },

  nodeCreated(node) {
    if (node.comfyClass !== CLASS) return;
    setupNode(node);
  },
});

// ── executed：拾起存档的输入帧 ─────────────────────────────────────────────
// 预览第二层。节点吃张量，生成图（KSampler -> VAE Decode）的上游从不填 imgs[0]、
// 第一层找不到东西。Python 把本次运行的输入帧写到 temp/ 并在 ui payload 里
// 报名字；这里把名字变成预览要画的图。
if (!app._sfOpExecPatched) {
  app._sfOpExecPatched = true;   // 热重载守卫：一个监听器，而非每次加载一个
  api.addEventListener("executed", ({ detail }) => {
    try {
      const entry = detail?.output?.sf_outpaint_base?.[0];
      if (!entry || !entry.filename) return;
      // Vue 把节点 id 以字符串交给 handler，legacy 是数字。
      const graph = app.graph;
      const node = graph?.getNodeById?.(detail.node) ??
                   graph?.getNodeById?.(parseInt(detail.node, 10));
      if (!node || node.comfyClass !== CLASS) return;
      // Python 不知道浏览器是否已有图，每次运行都存档。仅在第一层为空时解码：
      // 上游是 Load Image 时帧已上屏，每次运行解码全尺寸 PNG 再扔掉是纯浪费。
      // upstreamImage 而非 sourceImage：后者会把自己的缓存 base 帧也算进去，
      // 从第二次运行起回答"已经有了"、把预览冻结在第一张生成图上。
      if (upstreamImage(node)) return;
      const img = new Image();
      img.onload = () => {
        if (!node.graph) return; // 加载期间被删
        node._sfOpBaseImg = img;
        // 让 watcher 跟上，否则 400ms 后又白画同一张。
        node._sfOpSrcSig = sourceSig(node);
        renderFace(node);
      };
      img.src = `/view?filename=${encodeURIComponent(entry.filename)}` +
        `&type=${encodeURIComponent(entry.type || "temp")}` +
        `&subfolder=${encodeURIComponent(entry.subfolder || "")}`;
    } catch (e) {
      // 预览永远不值得破坏 executed 处理器——其他节点的监听器都靠这个事件。
      console.warn("[SF Image Outpaint] base preview failed:", (e && e.message) || e);
    }
  });
}

// ── graphToPrompt：注入每个节点的状态 ──────────────────────────────────────
// 只注入，绝不剪枝：Export（API）也序列化这同一份 output，剪枝会剪掉导出的
// 工作流。
function buildIndex() {
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

function findNode(index, id) {
  const s = String(id);
  if (index.has(s)) return index.get(s);
  const tail = s.includes(":") ? s.slice(s.lastIndexOf(":") + 1) : null;
  return tail && index.has(tail) ? index.get(tail) : null;
}

if (!app._sfOpPromptPatched) {
  app._sfOpPromptPatched = true;
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
          if (!index) index = buildIndex();
          const node = findNode(index, id);
          const state = node?.properties?.[STATE_PROP] || JSON.stringify(DEFAULT_STATE);
          entry.inputs = entry.inputs || {};
          entry.inputs[HIDDEN_INPUT] = state;
        }
      }
    } catch (e) {
      console.warn("[SF Image Outpaint] could not inject state:", (e && e.message) || e);
    }
    return result;
  };
}
