// ============================================================
// sf_common.js — 复刻节点共享的公共小工具
// 原 pixaroma js/shared/ 的内联副本去重（sf_crop / sf_inpaint /
// sf_load_image 曾各持一份，现已收敛到此处，避免复制后语义分叉）。
// 纯工具模块（无扩展行为），由使用者 import（同 sf_dynamic_slots.js 惯例）。
// 注意：本文件顶层自动安装 loadGraphData 守卫（幂等），任何模块 import 一次
// 即可全局生效。
// ============================================================
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// ── URL ──────────────────────────────────────────────────────────────
// 绝对安全的 URL：api.apiURL 处理托管部署基址，失败降级原样返回
export function sfApiUrl(route) {
  try {
    if (typeof api?.apiURL === "function") return api.apiURL(route);
  } catch {
    /* 降级 */
  }
  return route;
}

// ── 渲染器检测（Vue 2.0 vs Classic）──────────────────────────────────
export function isVueNodes() {
  return !!window.LiteGraph?.vueNodesMode;
}
export function applyAdaptiveCanvasOnly(widget) {
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

// ── 工作流加载守卫（全局单例，幂等）──────────────────────────────────
// 包装 app.loadGraphData 一次，加载 + 300ms 尾窗内 isGraphLoading() 为 true。
// 加载路径上的序列化状态变更必须被此守卫门控（连接恢复发生在 onConfigure
// 之后，无此守卫打开工作流会误断已保存的线）。
let _sfGraphLoading = false;
let _sfGraphLoadWrapped = false;
export function installGraphLoadingGuard() {
  if (_sfGraphLoadWrapped || !app || typeof app.loadGraphData !== "function") return;
  _sfGraphLoadWrapped = true;
  const _origLoadGraphData = app.loadGraphData.bind(app);
  app.loadGraphData = function (...args) {
    _sfGraphLoading = true;
    let r;
    try {
      r = _origLoadGraphData(...args);
    } finally {
      Promise.resolve(r).finally(() => setTimeout(() => { _sfGraphLoading = false; }, 300));
    }
    return r;
  };
}
export function isGraphLoading() {
  return _sfGraphLoading;
}
installGraphLoadingGuard();

// ── 滚轮缩放透传（仅 Classic 渲染器；增强版：滚动容器穿透检测）───────
// 根内部有可滚动区域（面板列表、编辑器等）时让容器滚动而非缩放画布；
// 无滚动容器时行为等价于直接转发（简单版），是后者的功能超集。
export function installCanvasZoomPassthrough(root) {
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

// ── 图片值解析（LoadImage widget 值 → {filename, subfolder, type}）────
// 支持 "subfolder/name.png [output]" 注解与反斜杠路径。
export function parseAnnotatedImageValue(value) {
  let v = String(value || "");
  let type = "input";
  const m = v.match(/\s*\[(input|output|temp)\]\s*$/i);
  if (m) { type = m[1].toLowerCase(); v = v.slice(0, m.index); }
  v = v.replace(/\\/g, "/").trim();
  const i = v.lastIndexOf("/");
  return {
    filename: i >= 0 ? v.slice(i + 1) : v,
    subfolder: i >= 0 ? v.slice(0, i) : "",
    type,
  };
}

// ── /view URL 构建 ────────────────────────────────────────────────────
// Build a /view URL from a {filename, subfolder, type} record. Adds a fresh
// cache-buster timestamp at runtime; persisted records (in node.properties)
// store only the structural parts so workflow JSON stays clean.
export function buildSourceURL(part, withCacheBust) {
  if (!part || !part.filename) return null;
  // The cache-buster is part of the ROUTE handed to sfApiUrl, never appended to
  // its RESULT: a hosted ComfyUI adds its auth token to the finished url, so
  // concatenating afterwards writes our param on the far side of that token
  // (see js/shared/api_url.mjs). Locally the two produce the identical string.
  return sfApiUrl(`/view?filename=${encodeURIComponent(part.filename)}` +
              `&subfolder=${encodeURIComponent(part.subfolder || "")}` +
              `&type=${encodeURIComponent(part.type || "temp")}` +
              (withCacheBust ? `&t=${Date.now()}` : ""));
}

// ── 上游图片 URL 解析 ─────────────────────────────────────────────────
// 优先活着的接线源（刚换过的 Load Image / 实时预览），cachedUrl 是执行期
// 缓存（生成型上游上次运行保存的 temp PNG / 粘贴拖放的磁盘源）。不按这个
// 顺序，换 Load Image 文件后显示的是上次运行的旧图直到重跑。
export function getUpstreamImageURL(node, cachedUrl) {
  const input = (node.inputs || []).find((i) => i.name === "image");
  const graph = node.graph;
  if (input && input.link != null && graph) {
    // Vue Compat #3: graph.links can be a Map in newer ComfyUI versions.
    let link = graph.links?.[input.link];
    if (!link && typeof graph.links?.get === "function") link = graph.links.get(input.link);
    const src = link && graph.getNodeById(link.origin_id);
    if (src) {
      if (src.comfyClass === "LoadImage" || src.type === "LoadImage") {
        const w = (src.widgets || []).find((x) => x.name === "image");
        if (w && w.value) return buildSourceURL(parseAnnotatedImageValue(w.value), true);
      }
      if (src.imgs && src.imgs.length > 0) {
        const img = src.imgs[link.origin_slot] || src.imgs[0];
        if (typeof img === "string") return img;
        if (img && img.src) return img.src;
      }
    }
  }
  return cachedUrl || null;
}

// ── 全局粘贴 handler（剪贴板图片 → 选中的节点）────────────────────────
// Capture phase + stopImmediatePropagation 不能完全压住 ComfyUI 自己的
// paste handler（它注册更早），因此额外快照图节点 id，之后删掉 ComfyUI 从
// 同一 paste 事件自动创建的 LoadImage（"pasted/" 文件名）。
// 参数化：comfyClass 匹配活动节点；onPasteImage(node, dataURL) 执行上传；
// allowPaste(node) 返回 false 时跳过（如 inpaint 编辑器打开时）。
// 单例键是 comfyClass:hook 而非全局一次：SFImageCrop 与 SFInpaintCrop 等
// 多个类各自要一个监听器（去重为单一布尔曾让后注册的类粘贴静默失效——
// 同节点类多实例的重复安装仍幂等）。
const _pasteHandlerKeys = new Set();
export function installPasteHandler({ comfyClass, hook, onPasteImage, allowPaste }) {
  const key = `${comfyClass}:${hook || ""}`;
  if (_pasteHandlerKeys.has(key)) return;
  _pasteHandlerKeys.add(key);
  window.addEventListener("paste", async (e) => {
    // Don't steal paste from form fields (panel inputs, editor inputs, etc.)
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;

    const node = findActiveNode(comfyClass, hook);
    if (!node) return;
    if (allowPaste && !allowPaste(node)) return;

    const items = e.clipboardData?.items || [];
    const imageItem = Array.from(items).find((it) => it.type?.startsWith("image/"));
    if (!imageItem) return;

    e.preventDefault();
    e.stopImmediatePropagation();

    // If upstream wire is connected, disconnect it — pasting an image is an
    // unambiguous "use this image now" override. Without this, Python would
    // keep using the upstream tensor and the paste would have no effect on
    // workflow output.
    const imgInputIdx = (node.inputs || []).findIndex((i) => i.name === "image");
    if (imgInputIdx >= 0 && node.inputs[imgInputIdx].link != null) {
      try { node.disconnectInput(imgInputIdx); } catch {}
    }

    // Snapshot existing graph node IDs so we can remove any LoadImage that
    // ComfyUI auto-creates from this same paste event.
    const idsBefore = new Set((app.graph?._nodes || []).map((n) => n.id));

    const blob = imageItem.getAsFile();
    if (!blob) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      onPasteImage(node, ev.target.result);
    };
    reader.readAsDataURL(blob);

    // Schedule a sweep for the auto-created LoadImage node (if any).
    // 50ms is enough for ComfyUI's createNode + widget setup to settle.
    setTimeout(() => {
      const after = app.graph?._nodes || [];
      for (const n of after) {
        if (idsBefore.has(n.id)) continue;
        if (n.comfyClass !== "LoadImage" && n.type !== "LoadImage") continue;
        const w = (n.widgets || []).find((x) => x.name === "image");
        const v = w?.value;
        if (typeof v === "string" && v.startsWith("pasted/")) {
          try { app.graph.remove(n); } catch {}
        }
      }
    }, 50);
  }, true); // capture phase
}

// Find the "active" node of a class from any of the selection sources
// ComfyUI might use across versions/frontends:
//   1. app.canvas.selected_nodes  (object/array/map of selected nodes)
//   2. app.canvas.current_node    (LiteGraph's last-clicked node)
//   3. node_over                  (hovered node)
//   4. Iterate all nodes and pick one with `.is_selected` (Vue may set this)
// `hook` is the node's paste method name; matching requires it to be present
// (defensive: nodeCreated may not have finished for a just-created node).
// Returns the first matching node, or null.
function findActiveNode(comfyClass, hook) {
  const c = app.canvas;
  if (!c) return null;
  const isTarget = (n) =>
    n && n.comfyClass === comfyClass && (!hook || typeof n[hook] === "function");

  // 1. selected_nodes — try Object.values, Array, and Map .values()
  const sel = c.selected_nodes;
  if (sel) {
    let iter = null;
    if (Array.isArray(sel)) iter = sel;
    else if (typeof sel.values === "function") iter = Array.from(sel.values());
    else if (typeof sel === "object") iter = Object.values(sel);
    if (iter) {
      const hit = iter.find(isTarget);
      if (hit) return hit;
    }
  }

  // 2. current_node (LiteGraph internal)
  if (isTarget(c.current_node)) return c.current_node;

  // 3. node_over (hovered)
  if (isTarget(c.node_over)) return c.node_over;

  // 4. Fallback — scan all nodes for an is_selected flag
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    if (isTarget(n) && (n.is_selected || n.flags?.is_selected)) return n;
  }
  return null;
}
