// SF Image Resize — main extension (ported from comfyui-pixaroma
// js/image_resize/index.js, PixaromaImageResize). Builds on the shared SF
// resize infra: buildModePanel from sf_load_image_resize.js (the Load Image
// port) + the pure wired-state maths in sf_image_resize_lib.js + the DOM body
// in sf_image_resize_ui.js.
//
// Nodes 2.0 (Vue) renders the INPUT→OUTPUT readout into a DOM cards canvas
// (setupVueCards); legacy paints it in the slot dead-space via
// onDrawForeground. The hidden state rides node.properties and is injected
// into the prompt by the graphToPrompt hook below (subgraph-safe).

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { renderUI, injectCSS, measureContentHeight, refreshReadout, paintReadout } from "./sf_image_resize_ui.js";
import { STATE_PROP, HIDDEN_INPUT, DEFAULT_STATE } from "./sf_image_resize_ui.js";

injectCSS();

const MIN_W = 360; // minimum node width (the two IN/OUT cards need the room)
const CARDS_CANVAS_H = 100;

// ── 内联小工具（移植自 pixaroma js/shared/ + sf_load_image.js，去除插件专属依赖）──

function hideJsonWidget(widgets, widgetName) {
  const w = (widgets || []).find((x) => x.name === widgetName);
  if (w) {
    w.hidden = true;
    w.computeSize = () => [0, -4];
    if (!w.options) w.options = {};
    w.options.canvasOnly = true;
    const hideEl = () => { const el = w.element || w.inputEl; if (el) el.style.display = "none"; };
    hideEl();
    requestAnimationFrame(hideEl);
  }
  return w;
}

// 工作流加载守卫：wrap app.loadGraphData 一次，加载 + 300ms 尾窗内
// isGraphLoading() 为 true。加载路径上的序列化状态变更必须被此守卫门控
// （连接恢复发生在 onConfigure 之后，若不加守卫会把已保存的接线误当用户
// 新连接而自动断开）。
let _sfIrGraphLoading = false;
if (app && app.loadGraphData && !app._sfIrGraphLoadWrapped) {
  app._sfIrGraphLoadWrapped = true;
  const _origLoadGraphData = app.loadGraphData.bind(app);
  app.loadGraphData = function (...args) {
    _sfIrGraphLoading = true;
    let r;
    try {
      r = _origLoadGraphData(...args);
    } finally {
      Promise.resolve(r).finally(() => setTimeout(() => { _sfIrGraphLoading = false; }, 300));
    }
    return r;
  };
}
function isGraphLoading() {
  return _sfIrGraphLoading;
}

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
const CANVAS_BACKING_CAP = 6000;
function canvasBackingScale(cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const zoom = Math.max(1, app.canvas?.ds?.scale || 1);
  let s = dpr * zoom;
  const longCss = Math.max(cssW || 0, cssH || 0);
  if (longCss > 0 && longCss * s > CANVAS_BACKING_CAP) s = CANVAS_BACKING_CAP / longCss;
  return s;
}

// 滚轮缩放透传（仅 Classic 渲染器）：滚轮事件被 DOM widget 吞掉时转发到
// canvas；光标在仍有滚动余量的可滚动区域上时允许其正常滚动。Nodes 2.0
// 由前端自身转发，这里 no-op。
function installCanvasZoomPassthrough(root) {
  if (!root || typeof root.addEventListener !== "function") return () => {};
  const onWheel = (e) => {
    if (isVueNodes()) return;
    let el = e.target;
    const vertical = Math.abs(e.deltaY) >= Math.abs(e.deltaX);
    let scrollable = false;
    while (el && el !== root.parentElement) {
      if (el.nodeType === 1) {
        const cs = getComputedStyle(el);
        if (vertical) {
          const oy = cs.overflowY;
          if ((oy === "auto" || oy === "scroll") && el.scrollHeight > el.clientHeight + 1) {
            const atTop = el.scrollTop <= 0;
            const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
            if ((e.deltaY < 0 && !atTop) || (e.deltaY > 0 && !atBottom)) { scrollable = true; break; }
          }
        } else {
          const ox = cs.overflowX;
          if ((ox === "auto" || ox === "scroll") && el.scrollWidth > el.clientWidth + 1) {
            const atLeft = el.scrollLeft <= 0;
            const atRight = el.scrollLeft + el.clientWidth >= el.scrollWidth - 1;
            if ((e.deltaX < 0 && !atLeft) || (e.deltaX > 0 && !atRight)) { scrollable = true; break; }
          }
        }
      }
      el = el.parentElement;
    }
    if (scrollable) return;
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

// ── 节点行为 ─────────────────────────────────────────────────────────────────

// Refit node height to content. ONLY call on genuine user actions — never on
// the load path (resizing during configure dirties the saved workflow). rAF so
// the freshly-rendered panel has laid out before measuring.
function refit(node) {
  if (!node._sfIrRoot) return;
  requestAnimationFrame(() => {
    if (!node._sfIrRoot) return;
    const sz = node.computeSize();
    if (Math.abs(node.size[1] - sz[1]) > 1) {
      node.size[1] = sz[1];
      node.setDirtyCanvas(true, true);
    }
  });
}

// Small toast. Silent no-op if the toast API isn't present, so it never throws.
function toast(msg) {
  const t = app?.extensionManager?.toast;
  if (t?.add) t.add({ severity: "info", summary: "SF Image Resize", detail: msg, life: 2500 });
}

// Disconnect a named input if it currently has a wire. Returns true if it did.
function disconnectInputByName(node, name) {
  const i = node.inputs?.findIndex((inp) => inp?.name === name);
  if (i != null && i >= 0 && node.inputs[i]?.link != null) {
    node.disconnectInput(i);
    return true;
  }
  return false;
}

// Nodes 2.0 only: the readout cards can't paint in the slot dead-space (slots
// are Vue-rendered, no canvas band), so render them into a <canvas> child of the
// controls panel (renderUI prepends it). A ResizeObserver keeps it DPR-sized,
// and a low-rate poll replaces the legacy onDrawForeground loop for live
// upstream-value changes. Change-gated so the poll is cheap.
function setupVueCards(node) {
  const cv = document.createElement("canvas");
  cv.className = "sf-ir-cards-canvas";
  cv.style.cssText = `width:100%; height:${CARDS_CANVAS_H}px; display:block;`;
  node._sfIrCardsCanvas = cv;

  const render = (force) => {
    const canvas = node._sfIrCardsCanvas;
    if (!canvas || !canvas.isConnected) return;
    const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
    if (!cssW || !cssH) return;
    const { info } = refreshReadout(node);
    const sig = info.mode === "msg"
      ? `m:${info.text}`
      : `d:${info.inW}x${info.inH}>${info.outW}x${info.outH}`;
    // Backing store at dpr x graph-zoom so the cards stay crisp when zoomed in.
    // The ResizeObserver doesn't fire on zoom, so fold the scale into the
    // size-sig — the existing 250ms poll then re-renders on a zoom change.
    const s = canvasBackingScale(cssW, cssH);
    const sizeSig = `${cssW}x${cssH}@${s.toFixed(2)}`;
    if (!force && sig === node._sfIrLastSig && sizeSig === node._sfIrLastSize) return;
    node._sfIrLastSig = sig;
    node._sfIrLastSize = sizeSig;
    const W = Math.round(cssW * s), H = Math.round(cssH * s);
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(s, 0, 0, s, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    paintReadout(ctx, info, cssW, cssH / 2);
  };
  node._sfIrRenderCards = render;

  const ro = new ResizeObserver(() => render());
  ro.observe(cv);
  node._sfIrCardsRO = ro;
  node._sfIrPoll = setInterval(() => render(), 250);
}

app.registerExtension({
  name: "sfnodes.ImageResize",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SFImageResize") return;

    const _origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = _origCreated?.apply(this, arguments);
      hideJsonWidget(this.widgets, HIDDEN_INPUT);
      const root = document.createElement("div");
      root.className = "sf-ir-root";
      installCanvasZoomPassthrough(root);
      const w = this.addDOMWidget("sf_image_resize_ui", "custom", root, {
        serialize: false,
        getMinHeight: () => measureContentHeight(root),
      });
      applyAdaptiveCanvasOnly(w);
      this._sfIrRoot = root;
      // Nodes 2.0 only: build the INPUT->OUTPUT cards canvas (renderUI prepends
      // it into the panel). Legacy paints the cards in the slot dead-space via
      // onDrawForeground instead.
      if (isVueNodes()) setupVueCards(this);
      // Fresh-node default size (saved workflows restore their own via configure).
      if (!this.size || this.size[0] < MIN_W) this.size = [360, 340];
      // Deferred initial render so configure() can land the saved state first.
      // By microtask time configure() has run for a loaded node, so the state
      // prop tells fresh drops from restores — refit ONLY on a fresh drop.
      queueMicrotask(() => {
        const wasConfigured = this.properties?.[STATE_PROP] !== undefined;
        renderUI(this, { onRefit: () => refit(this) });
        if (!wasConfigured) refit(this);
      });
      return r;
    };

    const _origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      this._sfIrConfiguring = true;
      try {
        const res = _origConfigure?.apply(this, arguments);
        if (this._sfIrRoot) renderUI(this); // render only — no refit (load path)
        return res;
      } finally {
        this._sfIrConfiguring = false;
      }
    };

    const INPUT_TYPE = (typeof LiteGraph !== "undefined" && LiteGraph.INPUT != null) ? LiteGraph.INPUT : 1;
    const _origConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, idx, connected, link, ioSlot) {
      const r = _origConn?.apply(this, arguments);
      // Auto-swap sizing sources: longest_side and width/height are competing
      // ways to set the size, so connecting one drops the other(s). width and
      // height may coexist (exact box) — only longest_side is exclusive vs them.
      // Only on a genuine user connect; never during configure/load. Three
      // guards: _sfIrConfiguring (this node's onConfigure window),
      // isGraphLoading() (the graph-level link-restore window that fires AFTER
      // onConfigure), and _sfIrAutoSwapping (re-entrancy from disconnectInput).
      if (type === INPUT_TYPE && connected && !this._sfIrConfiguring && !this._sfIrAutoSwapping && !isGraphLoading()) {
        const name = this.inputs?.[idx]?.name || ioSlot?.name;
        this._sfIrAutoSwapping = true;
        try {
          if (name === "longest_side") {
            const dW = disconnectInputByName(this, "width");
            const dH = disconnectInputByName(this, "height");
            if (dW || dH) toast("longest_side 现在驱动尺寸，width/height 已断开。");
          } else if (name === "width" || name === "height") {
            if (disconnectInputByName(this, "longest_side"))
              toast("width/height 现在驱动尺寸，longest_side 已断开。");
          }
        } finally {
          this._sfIrAutoSwapping = false;
        }
      }
      if (!this._sfIrConfiguring && !this._sfIrAutoSwapping && this._sfIrRoot) {
        renderUI(this, { onRefit: () => refit(this) }); // re-render for the new wire count
        refit(this);
        // Upstream loader may populate its image a tick after the wire lands;
        // re-read the size shortly after so the readout updates without a run.
        setTimeout(() => this.setDirtyCanvas(true, true), 200);
      }
      return r;
    };

    const _origRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this._sfIrRoot = null;
      this._sfIrWireCells = null;
      this._sfIrLockedInputs = null;
      this._sfIrLongestCell = null;
      // Nodes 2.0 cards-canvas teardown: stop the poll + observer so a deleted
      // node doesn't keep re-reading wires / leak the ResizeObserver.
      if (this._sfIrPoll) { clearInterval(this._sfIrPoll); this._sfIrPoll = null; }
      if (this._sfIrCardsRO) { this._sfIrCardsRO.disconnect(); this._sfIrCardsRO = null; }
      this._sfIrCardsCanvas = null;
      this._sfIrRenderCards = null;
      return _origRemoved?.apply(this, arguments);
    };

    // Belt-and-braces minimum width: onResize is unreliable in the Vue
    // frontend, so clamp here too.
    const _origResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      if (this.size[0] < MIN_W) this.size[0] = MIN_W;
      return _origResize?.apply(this, arguments);
    };

    // Paint the size readout in the empty space between the input and output
    // slot columns (legacy only; Nodes 2.0 uses the DOM cards canvas).
    const _origDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = _origDraw?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      if (isVueNodes()) return r;
      if (this.size[0] < MIN_W) { this.size[0] = MIN_W; this.setDirtyCanvas(true, true); }
      const { info } = refreshReadout(this);
      // midY=54: the vertical center of the 5 slot rows (TOP_PAD 4 + 5*20/2).
      paintReadout(ctx, info, this.size[0], 54);
      return r;
    };
  },
});

// ── executed payload: learn real in/out dims ──
api.addEventListener("executed", ({ detail }) => {
  const frames = detail?.output?.sf_image_resize;
  if (!frames || !frames.length) return;
  let node = app.graph.getNodeById(detail.node);
  if (!node && typeof detail.node === "string") node = app.graph.getNodeById(parseInt(detail.node, 10));
  if (!node || node.comfyClass !== "SFImageResize") return;
  const f = frames[0];
  if (!node.properties) node.properties = {};
  node.properties.sfIrDims = { in_w: f.in_w, in_h: f.in_h, out_w: f.out_w, out_h: f.out_h };
  node.setDirtyCanvas(true, true);
  node._sfIrRenderCards?.(true); // Nodes 2.0: refresh the cards canvas now (no-op in legacy)
});

// ── graphToPrompt: inject state into the hidden input (subgraph-safe) ──
const _origG2P = app.graphToPrompt.bind(app);
app.graphToPrompt = async function (...args) {
  const result = await _origG2P(...args);
  // FAIL OPEN — a throw here rejects ComfyUI's own graphToPrompt and breaks
  // Run for the whole workflow. Never wrap the `await _origG2P` above.
  try {
    const out = result?.output;
    if (out) {
      let index = null;
      const buildIndex = () => {
        const m = new Map();
        const visit = (g) => {
          if (!g) return;
          for (const n of (g._nodes || g.nodes || [])) {
            if (!n) continue;
            if (n.comfyClass === "SFImageResize" || n.type === "SFImageResize")
              m.set(String(n.id), n);
            const inner = n.subgraph || n.graph || n._graph;
            if (inner && inner !== g) visit(inner);
          }
        };
        visit(app.graph);
        return m;
      };
      for (const id in out) {
        if (out[id]?.class_type !== "SFImageResize") continue;
        if (!index) index = buildIndex();
        const sId = String(id);
        let node = index.get(sId);
        if (!node && sId.includes(":")) node = index.get(sId.slice(sId.lastIndexOf(":") + 1));
        const state = node?.properties?.[STATE_PROP] || JSON.stringify(DEFAULT_STATE);
        out[id].inputs = out[id].inputs || {};
        out[id].inputs[HIDDEN_INPUT] = state;
      }
    }
  } catch (e) {
    console.error("[SFImageResize] prompt injection failed; prompt sent unchanged", e);
  }
  return result;
};
