// ==========================================================================
// sf_dropdown_ui.js - SFValueDropdown 节点面（一行 DOM）与输出点对齐
// ==========================================================================
//
// DOM widget 在两个渲染器都渲染，所以一行实现同时服务两者。把输出点放上这行
// 是两个完全独立的机制：
//
//   CLASSIC   LiteGraph 尊重硬编码的 `output.pos`（getConnectionPos 原样返回
//             node.pos + slot.pos）并跳过自动堆叠已定位的输出，所以我们把点
//             停在该行的 Y 上。
//
//   NODES 2.0 没有官方方式移动输出——NodeSlots.vue 把所有输出渲染在右上角列，
//             也没有输入那种 widget-socket 模型。所以我们 NUDGE DOM。纯装饰
//             且 try/catch 包裹：未来前端若破解了它，点只是回到角落，节点照常
//             工作。
//
// 两种机制都源自 Control Panel（Pixaroma），它是全包里唯一这么做的节点。
// 它记载的每个陷阱在这里都适用。
//
// 内联 shared 辅助（参照 sf_pause_text.js 先例，不引入 pixaroma 的 shared 库）：
//   isVueNodes / applyAdaptiveCanvasOnly / popupZoom+placeZoomedPopup /
//   installCanvasZoomPassthrough（固定默认：可滚动区域滚动，否则转发 canvas）
// 图标用 data URI（sfnodes 无资产服务路由）。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import {
    ROW_H, MIN_W, BODY_PAD, readState, writeState, shownIndex,
    MODE_LETTERS, MODE_LABELS, MODES,
} from "./sf_dropdown_lib.js";
import { SOCKET_LABELS, previewText, readable } from "./sf_dropdown_lib.js";

// 固定强调色（复刻范围外：无 accent 颜色设置，参照 sf_pause_text_ui.js）。
const ACCENT = "#f66744";

// Classic 在行上方插入的内容：node.widgets_start_y（index.js 里设为 2）加
// BaseDOMWidgetImpl.DEFAULT_MARGIN（10）。在真正要紧处（alignOutputLegacy）
// 从活 widget 读 margin；这个常量只用于下方预留对应的空间。
const TOP_INSET = 12;

const ROW_CLASS = "sf-dd-row";
const WIDGET_NAME = "dropdown_ui";
// 加命名空间，未来前端不能开始占用这个类型名并渲染它自己的 widget
// 而不是我们的元素（Show Text bug）。
const WIDGET_TYPE = "sf_value_dropdown";

const GEAR_SVG = "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>' +
    "</svg>"
);

let _cssDone = false;

// ── 内联 shared 辅助（与 sf_pause_text.js 相同的实现）────────────────────
// Nodes 2.0（Vue）渲染器判定，由设置 Comfy.VueNodes.Enabled 驱动；实时读取，
// 运行时切换渲染器也尊重
export function isVueNodes() {
    return !!window.LiteGraph?.vueNodesMode;
}
// adaptive canvasOnly：legacy 下 true（不进 Parameters tab），Nodes 2.0 下 false
// （否则 Vue 根本不渲染该 widget）。实时 getter，渲染时求值
export function applyAdaptiveCanvasOnly(widget) {
    if (!widget || !widget.options) return widget;
    try {
        Object.defineProperty(widget.options, "canvasOnly", {
            configurable: true,
            enumerable: true,
            get() {
                return !isVueNodes();
            },
        });
    } catch { /* ignore */ }
    return widget;
}

// 12px 匹配原生节点 widget——Pixaroma 行在 100% 缩放下就是这么大。
const POPUP_BASE_FONT_PX = 12;
// 下限 1：打开来读的弹出列表不能随图缩小。上限 2.5：深放大不产生海报文字。
const POPUP_ZOOM_MIN = 1;
const POPUP_ZOOM_MAX = 2.5;

// 被钳制的画布缩放（缺失/非有限/零防御：画布未就绪时的早期调用）。
function popupZoom() {
    const s = Number(app.canvas?.ds?.scale);
    if (!isFinite(s) || s <= 0) return 1;
    return Math.min(POPUP_ZOOM_MAX, Math.max(POPUP_ZOOM_MIN, s));
}

// 从缩放设置弹出列表根字体。列表内部尺寸（文字、行距、间距）必须用 em 编写，
// 让这一个数字一起缩放它们——行上放回 px 就是悄悄让那一行退出缩放。
function applyPopupZoom(pop, opts = {}) {
    const zoom = popupZoom();
    const base = opts.baseFontPx || POPUP_BASE_FONT_PX;
    pop.style.fontSize = Math.round(base * zoom * 10) / 10 + "px";
    if (opts.baseMaxHeightPx) {
        if (zoom > 1) {
            const vh = opts.maxHeightVh == null ? 0.6 : opts.maxHeightVh;
            pop.style.maxHeight =
                Math.round(Math.min(opts.baseMaxHeightPx * zoom, window.innerHeight * vh)) + "px";
        } else {
            // 清掉而非保留：zoom 1 时 stylesheet 自己的 max-height 才是正确答案。
            pop.style.maxHeight = "";
        }
    }
    return zoom;
}

// 缩放字体 + 自适应宽度 + 落位，按唯一正确的顺序。在列表行都进去且已挂入
// document 之后调用（要测量），anchor 传列表所属元素。陷阱：
//   1. 缩放字体与自适应宽度必须一起工作。缩放的文字 + 仍锁在锚点宽度的列表
//      会把自适应刚放开的行重新切断。
//   2. 字体要先设置再测量：向上翻转分支读 offsetHeight，它依赖已应用的字体。
//   3. 长内容列表会探出窗口右缘，left 必须在宽度已知后钳制。
//   4. CSS min-width 压过 max-width：锚点 rect 是屏幕 px，宽节点高缩放时能
//      超过上限，所以先算上限、下限在它下面钳。
export function placeZoomedPopup(pop, anchorEl, opts = {}) {
    const zoom = applyPopupZoom(pop, opts);

    const r = anchorEl.getBoundingClientRect();
    const margin = opts.margin == null ? 8 : opts.margin;
    const gap = opts.gap == null ? 4 : opts.gap;

    const maxW = opts.baseMaxWidthPx
        ? Math.min(Math.round(window.innerWidth * 0.9), Math.round(opts.baseMaxWidthPx * zoom))
        : Math.round(window.innerWidth * 0.9);
    pop.style.maxWidth = maxW + "px";
    if (opts.anchorWidthIsMin !== false) {
        const wantMin = Math.max(opts.minWidthPx || 200, Math.round(r.width));
        pop.style.minWidth = Math.min(wantMin, maxW) + "px";
    }

    // 陷阱 3：只有增长后的宽度可测量了才钳 left。
    const pw = pop.offsetWidth;
    let left = Math.round(r.left);
    if (left + pw > window.innerWidth - margin) left = Math.max(margin, window.innerWidth - margin - pw);
    pop.style.left = left + "px";

    // 下方空间不足时向上翻转。
    const h = pop.offsetHeight;
    const below = window.innerHeight - r.bottom;
    pop.style.top = (below < h + margin && r.top > h + margin)
        ? Math.round(r.top - h - gap) + "px"
        : Math.round(r.bottom + gap) + "px";

    return zoom;
}

// 滚轮穿透（Classic only）：DOM widget 覆盖在 canvas 上，滚轮被 widget 消费，
// 画布缩放就停了（ComfyUI 的 wheel-to-zoom 监听在 <canvas> 元素上）。除非光标
// 在仍有滚动余地的可滚动区域内（长 textarea / 列表照常滚动），否则把 wheel
// 转发给 canvas。Nodes 2.0 自带转发，no-op。
function installCanvasZoomPassthrough(root) {
    if (!root || typeof root.addEventListener !== "function") return () => {};
    const onWheel = (e) => {
        if (isVueNodes()) return;                  // Nodes 2.0 自己转发给 canvas
        if (scrollRegionWantsWheel(e.target, root, e.deltaX, e.deltaY)) return;
        const canvasEl = app?.canvas?.canvas;      // 惰性读取；canvas 可被重建
        if (!canvasEl) return;
        e.preventDefault();                        // 需要非 passive 监听器（下面）
        e.stopPropagation();
        // 向 LiteGraph canvas 重发合成 wheel 使其缩放——与 ComfyUI 自己的
        // forwardEventToCanvas 对预览节点做的完全一样。
        const { clientX, clientY, deltaX, deltaY, deltaMode, ctrlKey, metaKey, shiftKey } = e;
        canvasEl.dispatchEvent(new WheelEvent("wheel", {
            clientX, clientY, deltaX, deltaY, deltaMode,
            ctrlKey, metaKey, shiftKey, bubbles: true, cancelable: true,
        }));
    };
    root.addEventListener("wheel", onWheel, { passive: false });
    return () => root.removeEventListener("wheel", onWheel);
}

// target 与 root 之间（含）某元素可滚动且在该方向仍有滚动余量——滚轮应滚动它
// 而不是缩放画布。
function scrollRegionWantsWheel(target, root, deltaX, deltaY) {
    const vertical = Math.abs(deltaY) >= Math.abs(deltaX);
    let el = target;
    while (el && el !== root.parentElement) {
        if (el.nodeType === 1) {
            const cs = getComputedStyle(el);
            if (vertical) {
                const oy = cs.overflowY;
                if ((oy === "auto" || oy === "scroll") && el.scrollHeight > el.clientHeight + 1) {
                    const atTop = el.scrollTop <= 0;
                    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
                    if ((deltaY < 0 && !atTop) || (deltaY > 0 && !atBottom)) return true;
                }
            } else {
                const ox = cs.overflowX;
                if ((ox === "auto" || ox === "scroll") && el.scrollWidth > el.clientWidth + 1) {
                    const atLeft = el.scrollLeft <= 0;
                    const atRight = el.scrollLeft + el.clientWidth >= el.scrollWidth - 1;
                    if ((deltaX < 0 && !atLeft) || (deltaX > 0 && !atRight)) return true;
                }
            }
        }
        el = el.parentElement;
    }
    return false;
}

// ── CSS ───────────────────────────────────────────────────────────────────
export function injectCSS() {
    if (_cssDone) return;
    _cssDone = true;
    const css = `
  .${ROW_CLASS}{
    /* 确定高度，永不用 100%。Nodes 2.0 的 widget 行是 min-content grid track，
       高度不确定的行会塌陷（实测 2px 行）。Legacy 用显式元素高度掩盖了 bug。 */
    height:${ROW_H}px; min-height:${ROW_H}px; box-sizing:border-box;
    display:flex; align-items:center; gap:5px;
    font:12px 'Segoe UI',sans-serif; user-select:none;
    /* 输出点在节点右缘上，已经在行外侧（Classic 按 widget.margin 内缩 DOM
       widget）。这里再留 16px 会在类型词与点之间留出可见空洞。 */
    padding-right:2px;
  }
  .sf-dd-arrow{
    flex:none; width:13px; text-align:center; cursor:pointer;
    color:${ACCENT}; font-size:10px; line-height:1; background:none; border:none; padding:0;
  }
  .sf-dd-arrow:hover{ filter:brightness(1.35); }
  .sf-dd-arrow.dim{ opacity:.28; cursor:default; }
  .sf-dd-arrow.dim:hover{ filter:none; }

  .sf-dd-field{
    flex:1 1 auto; min-width:0; height:${ROW_H - 4}px; box-sizing:border-box;
    display:flex; align-items:center; justify-content:space-between; gap:5px;
    background:#1d1d1d; border:1px solid #444; border-radius:4px;
    padding:0 6px; cursor:pointer;
  }
  .sf-dd-field:hover{ border-color:${ACCENT}; }
  .sf-dd-field.open{ border-color:${ACCENT}; }
  .sf-dd-name{
    flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; color:#ddd; font-size:12px;
  }
  .sf-dd-name.empty{ color:#777; font-style:italic; }
  .sf-dd-caret{ flex:none; color:${ACCENT}; font-size:8px; }

  /* 内联 data URI 齿轮，与节点选中工具栏的 ⚙ 一致，而非各平台渲染不同的 emoji。 */
  .sf-dd-gear{
    flex:none; width:14px; height:14px; padding:0; margin:0;
    background:none; border:none; cursor:pointer; line-height:0;
  }
  .sf-dd-gear::before{
    content:""; display:block; width:100%; height:100%;
    background:#aaa; -webkit-mask:url("${GEAR_SVG}") center/contain no-repeat;
    mask:url("${GEAR_SVG}") center/contain no-repeat;
  }
  .sf-dd-gear:hover::before{ background:${ACCENT}; }

  /* 常显且可点：循环 F -> I -> R。Fixed 是安静的默认，不喊橙色徽章；两种会
     偷偷换值的模式被填充，因为下次 Run 发出不同东西的节点必须明说。 */
  .sf-dd-mode{
    flex:none; width:16px; height:16px; padding:0; box-sizing:border-box;
    display:flex; align-items:center; justify-content:center;
    border-radius:3px; border:1px solid #4a4a4a; background:none; color:#999;
    font:11px 'Segoe UI',sans-serif; cursor:pointer; line-height:1;
  }
  .sf-dd-mode:hover{ border-color:${ACCENT}; color:#ddd; }
  .sf-dd-mode.on{ background:${ACCENT}; border-color:${ACCENT}; color:#fff; }
  .sf-dd-mode.on:hover{ filter:brightness(1.12); color:#fff; }

  .sf-dd-type{
    flex:none; color:${ACCENT}; font-size:11px; letter-spacing:.02em;
    max-width:50px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }

  /* ── 选项弹出列表（挂在 document.body，节点外）──────────────────────── */
  /* 内部尺寸与内边距刻意用 em：openPopup 从画布缩放设置根字号，这一个数字
     一起缩放行、间距与内边距。不要把 px 放回行上。 */
  .sf-dd-pop{
    position:fixed; z-index:1200; box-sizing:border-box;
    background:#1d1d1d; border:1px solid #555; border-radius:6px; padding:.35em;
    max-height:320px; overflow-y:auto; overflow-x:hidden;
    font:12px 'Segoe UI',sans-serif; box-shadow:0 6px 20px rgba(0,0,0,.45);
  }
  .sf-dd-opt{
    display:flex; align-items:baseline; gap:.85em;
    padding:.5em .75em; border-radius:4px; cursor:pointer;
  }
  .sf-dd-opt:hover{ background:#2a2a2a; }
  .sf-dd-opt.sel{ background:${ACCENT}; }
  .sf-dd-oname{
    flex:none; max-width:100%; overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; color:#ddd; font-size:1em;
  }
  .sf-dd-opt.sel .sf-dd-oname{ color:#fff; }
  /* 值读不成类型的行上的警告后缀。 */
  .sf-dd-obad{ flex:none; color:#e0703a; font-size:.9em; }
  .sf-dd-opt.sel .sf-dd-obad{ color:#fff; }
  .sf-dd-pop-empty{ padding:.7em .85em; color:#777; font-size:.92em; font-style:italic; }

  /* ── Nodes 2.0 only ─────────────────────────────────────────────────────
     每个 widget 行为输入点预留 12px 列。本节点没有输入，塌掉它，否则行被
     无故缩进 12px。 */
  .lg-node:has(.${ROW_CLASS}) .lg-node-widget > div:first-child{
    width:0 !important; min-width:0 !important; overflow:hidden !important;
  }
  /* 移动后的输出槽不得画标签（我们的行已显示类型）也不得吞掉行上的指针事件
     ——只有它的点可以。 */
  .lg-node:has(.${ROW_CLASS}) .lg-slot--output{ padding-left:0 !important; pointer-events:none; }
  .lg-node:has(.${ROW_CLASS}) .lg-slot--output > div:first-child{ display:none !important; }
  .lg-node:has(.${ROW_CLASS}) .lg-slot--output [data-testid="slot-connection-dot"]{ pointer-events:auto; }
  `;
    const tag = document.createElement("style");
    tag.id = "sf-dd-css";
    tag.textContent = css;
    document.head.appendChild(tag);
}

/**
 * 节点体高度。一行，但两个渲染器内缩方式不同。
 *
 * CLASSIC 把行放在 widgets_start_y + widget.margin（这里是 2 + 10）处，所以
 * 体下面要再留同样空间，否则行明显偏高（此前测得上方 12px、下方 2px）。
 *
 * NODES 2.0 没有这种 margin；它自己的 chrome 在调用处加。
 */
export function bodyHeight() {
    return isVueNodes() ? ROW_H + BODY_PAD * 2 : TOP_INSET * 2 + ROW_H;
}

// ── 一行 ──────────────────────────────────────────────────────────────────

export function ensureRow(node) {
    if (node._sfDropdownRow?.isConnected) return node._sfDropdownRow;
    // 即便尚未连入 DOM 也回退到持有的元素：首次渲染在元素入 DOM 前发生，
    // 在这里退出会让体永久空白（Sizes bug）。
    return node._sfDropdownRow || null;
}

export function buildRow(node, onOpenSettings) {
    injectCSS();

    const row = document.createElement("div");
    row.className = ROW_CLASS;

    const prev = document.createElement("button");
    prev.className = "sf-dd-arrow sf-dd-prev";
    prev.textContent = "◀";
    prev.title = "Previous entry";

    const field = document.createElement("div");
    field.className = "sf-dd-field";
    field.title = "Click to choose from your list";
    const name = document.createElement("span");
    name.className = "sf-dd-name";
    const caret = document.createElement("span");
    caret.className = "sf-dd-caret";
    caret.textContent = "▼";
    field.append(name, caret);

    const next = document.createElement("button");
    next.className = "sf-dd-arrow sf-dd-next";
    next.textContent = "▶";
    next.title = "Next entry";

    const gear = document.createElement("button");
    gear.className = "sf-dd-gear";
    gear.title = "Edit the list and what it sends out";

    const mode = document.createElement("button");
    mode.className = "sf-dd-mode";

    const type = document.createElement("span");
    type.className = "sf-dd-type";

    // 顺序：步进箭头、列表、运行模式徽章、设置、类型词（挨着它的点）。
    row.append(prev, field, next, mode, gear, type);

    node._sfDropdownRow = row;
    node._sfDropdownParts = { prev, field, name, next, gear, mode, type };

    // 一个委托监听器。每个分支都停止传播，使点击不会到达画布引发节点拖拽。
    row.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        const t = e.target;
        if (t.closest(".sf-dd-prev")) { e.stopPropagation(); step(node, -1); return; }
        if (t.closest(".sf-dd-next")) { e.stopPropagation(); step(node, +1); return; }
        // 齿轮自己关掉弹出列表。只有 FIELD 豁免于外部点击处理器，其他都不。
        if (t.closest(".sf-dd-gear")) { e.stopPropagation(); closePopup(); onOpenSettings?.(node); return; }
        if (t.closest(".sf-dd-mode")) { e.stopPropagation(); cycleMode(node); return; }
        if (t.closest(".sf-dd-field")) {
            e.stopPropagation();
            // 空列表没什么可显示，直接把用户送去能修的地方，而不是打开一个
            // 什么也不说的弹出列表。
            if (!readState(node).options.length) onOpenSettings?.(node);
            else togglePopup(node);
        }
    });

    const w = node.addDOMWidget(WIDGET_NAME, WIDGET_TYPE, row, {
        serialize: false,                       // 不进 API prompt
        getValue: () => "",
        setValue: () => {},
    });
    if (w) {
        // 与 options.serialize 不同的另一个标志：这个让 widget 不进已保存的
        // 工作流，文件不会多一个 widgets_values 槽。
        w.serialize = false;
        // legacy 中固定高度。
        w.computeSize = () => [node.size?.[0] || MIN_W, ROW_H];
        // 自身属性遮蔽 DOMWidget 原型方法。定义了它，行在 Nodes 2.0 中成为
        // 'auto' grid track，吃掉节点富余高度而不是贴合内容。
        w.computeLayoutSize = undefined;
        applyAdaptiveCanvasOnly(w);
        node._sfDropdownWidget = w;
    }

    // 没有这个，Classic 中行上的滚轮停止缩放画布。
    installCanvasZoomPassthrough(row);

    return row;
}

/** 从状态重绘行。只动 DOM——绝不触碰序列化节点状态。 */
export function renderRow(node) {
    const parts = node._sfDropdownParts;
    if (!parts) return;
    const st = readState(node);
    // 已排队或上次运行的牌，而非盲信存的那个：Random / In-order 下节点否则会
    // 一直显示一条它不会发送的条目。
    const opt = st.options[shownIndex(node)];

    if (!st.options.length) {
        parts.name.textContent = "No options yet, press the gear";
        parts.name.classList.add("empty");
        parts.field.title = "Open the settings and add your first entry";
    } else {
        parts.name.textContent = opt?.name?.trim() || "(unnamed)";
        parts.name.classList.remove("empty");
        parts.field.title = opt ? `Sends: ${previewText(opt.value, st.type)}` : "";
    }

    parts.mode.textContent = MODE_LETTERS[st.mode] || "F";
    parts.mode.title = `${MODE_LABELS[st.mode] || ""}\nClick to change`;
    parts.mode.classList.toggle("on", st.mode !== "fixed");

    parts.type.textContent = SOCKET_LABELS[st.type] || st.type;
    parts.type.title = `This node sends ${SOCKET_LABELS[st.type]}. Change it in the settings.`;

    const many = st.options.length > 1;
    parts.prev.classList.toggle("dim", !many);
    parts.next.classList.toggle("dim", !many);
}

/** 步进选择。回绕，所以短列表也能快速循环。 */
export function step(node, delta) {
    const st = readState(node);
    if (st.options.length < 2) return;
    const n = st.options.length;
    // 从节点正显示的位置步进。Random / In-order 下那是已排队/上次运行的牌，
    // 不是存的那个，所以箭头从用户目光所在处移动。
    writeState(node, { index: ((shownIndex(node) + delta) % n + n) % n });
    // 手工选择压过任何飞行中的序列，否则下次 Run 会无视刚做的选择。
    node._sfDropdownPending = null;
    node._sfDropdownCursor = null;
    renderRow(node);
    closePopup();
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

/**
 * 从节点面循环运行模式 F -> I -> R -> F，不打开设置也能改。
 *
 * 丢掉持有的牌与序列位置，与面板按钮完全一致：切换模式应从节点正显示的条目
 * 重新开始，而不是继续用户刚抛弃的序列。
 */
export function cycleMode(node) {
    const cur = readState(node).mode;
    const next = MODES[(Math.max(0, MODES.indexOf(cur)) + 1) % MODES.length];
    writeState(node, { mode: next });
    node._sfDropdownPending = null;
    node._sfDropdownCursor = null;
    renderRow(node);
    node.setDirtyCanvas?.(true, true);
    // 无需重绘设置面板：徽章在它外面，面板自己的外部点击守卫已关掉它——
    // 与点击节点上任何其他地方一样。
}

// ── 选项弹出列表 ─────────────────────────────────────────────────────────

let _pop = null;
let _popNode = null;

export function closePopup() {
    if (_pop) {
        _pop.remove();
        document.removeEventListener("pointerdown", _outsidePointer, true);
        document.removeEventListener("wheel", _outsideWheel, true);
        document.removeEventListener("keydown", _onKey, true);
    }
    _popNode?._sfDropdownParts?.field?.classList?.remove("open");
    _pop = null;
    _popNode = null;
}

export function closePopupFor(node) {
    if (_popNode === node) closePopup();
}

// 点击任何不在列表本身或拥有它的 field 上的地方。
//
// 只有 FIELD 豁免，不是整行。这个处理器在 CAPTURE 阶段运行，先于行自己的
// pointerdown：豁免 field 正是让对它第二次点击可以关掉列表，而不是这个处理器
// 关掉它、行处理器瞬间重开（看起来像点击没反应）。豁免整行则过头了——类型
// 标签、控件间隙、为输出点预留的内边距都在外面，行处理器对它们没有分支，
// 点那里会把列表卡在开着。
function _outsidePointer(e) {
    if (!_pop) return;
    if (_pop.contains(e.target)) return;
    if (_popNode?._sfDropdownParts?.field?.contains(e.target)) return;
    closePopup();
}

// 任何不在列表本身上的滚轮。
//
// 刻意没有 field/行豁免。列表是 position:fixed 且坐标只写一次（来自 field 的
// rect），任何移动画布的东西都会把它搁浅在一个不存在的节点旁。行上的滚轮
// 缩放画布（穿透），所以行也必须关掉它——只有滚动列表本身豁免，否则列表
// 根本滚不动。
function _outsideWheel(e) {
    if (!_pop) return;
    if (_pop.contains(e.target)) return;
    closePopup();
}

function _onKey(e) {
    if (e.key === "Escape") { e.stopPropagation(); closePopup(); }
}

function togglePopup(node) {
    if (_popNode === node) { closePopup(); return; }
    openPopup(node);
}

export function openPopup(node) {
    closePopup();
    const parts = node._sfDropdownParts;
    if (!parts) return;
    const st = readState(node);

    const pop = document.createElement("div");
    pop.className = "sf-dd-pop";

    if (!st.options.length) {
        const empty = document.createElement("div");
        empty.className = "sf-dd-pop-empty";
        empty.textContent = "Nothing in the list yet.";
        pop.appendChild(empty);
    } else {
        const shown = shownIndex(node);
        st.options.forEach((o, i) => {
            const item = document.createElement("div");
            // shownIndex 而非 st.index：In-order / Random 下节点面显示已排队或
            // 上次运行的条目，列表高亮另一个会让节点看起来无视模式。
            const ok = readable(o.value, st.type);
            item.className = "sf-dd-opt" + (i === shown ? " sel" : "") + (ok ? "" : " bad");
            const nm = document.createElement("span");
            nm.className = "sf-dd-oname";
            nm.textContent = o.name?.trim() || "(unnamed)";
            item.append(nm);
            // 只放名字——值预览尝试过又同天移除（用户：它把列表搞复杂了）。
            // 值悬停 title 可见，设置里也始终可见。坏行仍留标记：静默发回退值
            // 的条目不能看起来与正常条目相同。
            if (!ok) item.append(Object.assign(document.createElement("span"), {
                className: "sf-dd-obad", textContent: "⚠",
            }));
            item.title = ok
                ? (o.value || "")
                : `Does not read as ${SOCKET_LABELS[st.type]}, so this one sends ${previewText(o.value, st.type)}.`;
            item.addEventListener("pointerdown", (e) => {
                e.stopPropagation();
                writeState(node, { index: i });
                // 与箭头同规则：手工选择压过序列。
                node._sfDropdownPending = null;
                node._sfDropdownCursor = null;
                renderRow(node);
                closePopup();
                node.setDirtyCanvas?.(true, true);
            });
            pop.appendChild(item);
        });
    }

    document.body.appendChild(pop);
    // 缩放字体 + 自适应 + 落位，按唯一能工作的顺序。field 宽度是最小值而非
    // 宽度：缩放字体下锁在 field 上的列表会重切自适应刚放开的行。列表增长到
    // 最长行，受视口约束，增长后的列表被钳制不能探出窗口右缘。
    placeZoomedPopup(pop, parts.field, {
        baseMaxHeightPx: 320,
        baseMaxWidthPx: 640,
        minWidthPx: 200,
    });

    parts.field.classList.add("open");
    _pop = pop;
    _popNode = node;

    // 延迟注册，否则打开它的那次点击立即关掉它。
    setTimeout(() => {
        document.addEventListener("pointerdown", _outsidePointer, true);
        document.addEventListener("wheel", _outsideWheel, true);
        document.addEventListener("keydown", _onKey, true);
    }, 0);
}

// ── 输出点对齐 ───────────────────────────────────────────────────────────

/**
 * CLASSIC：把点停在行的 Y 上。
 *
 * 小心 MARGIN。Legacy 按 widget.margin（默认 10）内缩 DOM widget 的 ELEMENT：
 * 元素画在 node.pos + margin + widget.y 而 widget.y 不带 margin。因此放在
 * widget.y + ROW_H/2 的点会落在行真正中心之上整整 10px——26px 的行上几乎是
 * 它的顶边。Control Panel 恰好带过这个 bug，用户一眼抓到。Nodes 2.0 无此
 * margin，所以同样的数学在那里看着是对的。
 */
export function alignOutputLegacy(node) {
    const w = node._sfDropdownWidget;
    const out = node.outputs?.[0];
    if (!w || !out) return;
    const y = w.y;
    if (!Number.isFinite(y)) return;
    const margin = Number.isFinite(w.margin) ? w.margin : 10;
    const nx = node.size[0];
    const ny = y + margin + ROW_H * 0.5;
    const pos = out.pos;
    // diff 门控：output.pos 会被序列化，重写相同值在某些构建上仍算一次变更。
    if (!pos || pos[0] !== nx || Math.abs(pos[1] - ny) > 0.5) out.pos = [nx, ny];
}

function isAligned(rowEl, dot) {
    const rr = rowEl.getBoundingClientRect();
    const dd = dot.getBoundingClientRect();
    return Math.abs((rr.top + rr.height / 2) - (dd.top + dd.height / 2)) < 1;
}

/**
 * NODES 2.0：nudge。把槽位块拉出文档流，停止它把行往下推，然后把点平移上
 * 行。
 *
 * 顺序要紧：给槽位定尺寸会改变块高度，所以块必须在定尺寸后测量。先测量会
 * 少拉一行槽位，点神秘地偏高——Control Panel 追了好几个回合的 bug。
 */
export function alignOutput(node) {
    if (!isVueNodes()) return;
    try {
        const el = document.querySelector(`.lg-node[data-node-id="${node.id}"]`);
        if (!el) return;
        const rowEl = el.querySelector(`.${ROW_CLASS}`);
        const outs = el.querySelectorAll(".lg-slot--output");
        if (!rowEl || !outs.length) return;
        if (isAligned(rowEl, outs[0])) return;

        const col = outs[0].parentElement;
        const block = col?.parentElement;
        if (!col || !block) return;

        // 先复位：下面每个测量都必须对着自然布局，不能对着上次的 nudge，
        // 否则修正会叠加。
        block.style.marginBottom = "0px";
        col.style.transform = "none";
        col.style.gap = "0px";
        block.style.pointerEvents = "none";
        col.style.pointerEvents = "auto";

        // 我们写的是 LAYOUT px；getBoundingClientRect 返回 SCREEN px，因为节点
        // 被图缩放 CSS 缩放。从 layout 高度已知的元素上量比例，而不是信
        // ds.scale——任何缩放下都正确。
        const rowH = rowEl.offsetHeight || ROW_H;
        const toLayout = rowH / (rowEl.getBoundingClientRect().height || rowH);

        // 第一步：把点的槽位定为一行。改变块的高度。
        for (const o of outs) {
            o.style.height = rowH + "px";
            o.style.minHeight = rowH + "px";
            o.style.marginBottom = "0px";
        }
        // 第二步：把现在尺寸正确的块拉出文档流。
        block.style.marginBottom = (-block.offsetHeight) + "px";
        // 第三步：把点放上行。
        const delta =
            (rowEl.getBoundingClientRect().top - outs[0].getBoundingClientRect().top) * toLayout;
        col.style.transform = `translateY(${delta}px)`;
    } catch {
        /* nudge 失败——点留在角落，节点照常工作 */
    }
}

/**
 * 节点构建的那一帧行尚未布局，立即测量得到的是过期偏移，而没有任何东西会
 * 再纠正它。
 */
export function scheduleAlign(node) {
    alignOutput(node);
    requestAnimationFrame(() => {
        alignOutput(node);
        setTimeout(() => alignOutput(node), 120);
    });
}

/**
 * MutationObserver 不够：Vue 重渲染会替换节点元素，静默孤立任何绑在旧元素上
 * 的 observer。因此用自愈轮询。alignOutput 在无变化时早退，稳态成本是每
 * 350ms 一次 rect 读取。
 */
export function watchAlign(node) {
    if (node._sfDropdownPoll) return;
    // 刻意不门控 isVueNodes()。渲染器可在节点已存在时切换，切换不重跑
    // onNodeCreated/onConfigure——在 Classic 构建、后切到 Nodes 2.0 的节点
    // 会永远没有对齐器，点永远留在角落。alignOutput 在 Classic 早退，所以
    // 两种渲染器下跑轮询的稳态成本是一次布尔检查。
    node._sfDropdownPoll = setInterval(() => {
        if (!node.graph) { unwatchAlign(node); return; }
        alignOutput(node);
    }, 350);
    scheduleAlign(node);
}

export function unwatchAlign(node) {
    if (node._sfDropdownPoll) clearInterval(node._sfDropdownPoll);
    node._sfDropdownPoll = null;
}
