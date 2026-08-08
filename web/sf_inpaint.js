// ============================================================
// SF Inpaint Crop — 节点入口（Open mask editor 按钮、预览、源缓存、持久化）
// 移植自 comfyui-pixaroma js/inpaint_crop/index.js，遵循项目 sf_crop 系列
// 的隐藏 STRING widget 双通道模式。
// ============================================================
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { InpaintCropEditor, INPAINT_PREVIEW_COLORS } from "./sf_inpaint_core.js";
import "./sf_inpaint_paint.js";   // mixin: 画笔 / 遮罩 / 快捷键
import "./sf_inpaint_render.js";  // mixin: canvas 渲染 + 保存
import {
  createNodePreview, showNodePreview, restoreNodePreview, clearNodePreview, activateNodePreview,
} from "./sf_crop_preview.js";
import { downloadDataURL } from "./sf_crop_framework.js";

// ── 内联小工具（移植自 pixaroma js/shared/，去除插件专属依赖）─────────────

// 工作流加载守卫：wrap app.loadGraphData 一次，加载 + 300ms 尾窗内
// isGraphLoading() 为 true。
let _sfInpaintGraphLoading = false;
if (app && app.loadGraphData && !app._sfInpaintGraphLoadWrapped) {
  app._sfInpaintGraphLoadWrapped = true;
  const _origLoadGraphData = app.loadGraphData.bind(app);
  app.loadGraphData = function (...args) {
    _sfInpaintGraphLoading = true;
    let r;
    try {
      r = _origLoadGraphData(...args);
    } finally {
      Promise.resolve(r).finally(() => setTimeout(() => { _sfInpaintGraphLoading = false; }, 300));
    }
    return r;
  };
}
function isGraphLoading() {
  return _sfInpaintGraphLoading;
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

// 绝对安全的 URL：api.apiURL 处理托管部署基址，失败降级原样返回
function pixApiUrl(route) {
  try {
    if (typeof api?.apiURL === "function") return api.apiURL(route);
  } catch {
    /* 降级 */
  }
  return route;
}

const SIZE_MODE_MAP = {
  "keep shape (long side)": "keep",
  "force size (square)": "force",
  "free (multiple only)": "free",
};

// 节点 widget 标签 <-> 内部 blend_mode 键（编辑器胶囊用内部键）
const BLEND_MODE_MAP = { "mask": "mask", "whole crop": "whole_crop" };

function readParams(node) {
  const g = (n) => node.widgets?.find((w) => w.name === n)?.value;
  return {
    size_mode: SIZE_MODE_MAP[g("size_mode")] || "keep",
    target: parseInt(g("target")) || 1024,
    multiple: parseInt(g("multiple")) || 8,
    context_px: g("context_px") != null ? parseInt(g("context_px")) : 24,
    mask_grow: g("mask_grow") != null ? parseInt(g("mask_grow")) : 4,
    mask_blur: g("mask_blur") != null ? parseInt(g("mask_blur")) : 4,
    blend: g("softness") != null ? parseInt(g("softness")) : 16,
    // blend_mode 也是节点 widget（编辑器胶囊镜像它）。
    blend_mode: BLEND_MODE_MAP[g("blend_mode")] || "mask",
  };
}

// 友好标签 <- 内部 size-mode 键，把编辑器选择写回 widget
const SIZE_MODE_LABEL = Object.fromEntries(
  Object.entries(SIZE_MODE_MAP).map(([k, v]) => [v, k]));
const BLEND_MODE_LABEL = Object.fromEntries(
  Object.entries(BLEND_MODE_MAP).map(([k, v]) => [v, k]));

function setNodeWidget(node, name, value) {
  const w = node.widgets?.find((x) => x.name === name);
  if (w && w.value !== value) { w.value = value; w.callback?.(value); }
}

// 把编辑器镜像的几何旋钮写回原生节点 widget
function writeBackWidgets(node, extra) {
  if (!extra) return;
  if (extra.context_px != null) setNodeWidget(node, "context_px", extra.context_px);
  if (extra.mask_grow != null) setNodeWidget(node, "mask_grow", extra.mask_grow);
  if (extra.mask_blur != null) setNodeWidget(node, "mask_blur", extra.mask_blur);
  if (extra.softness != null) setNodeWidget(node, "softness", extra.softness);
  if (extra.target != null) setNodeWidget(node, "target", extra.target);
  if (extra.multiple != null) setNodeWidget(node, "multiple", extra.multiple);
  if (extra.size_mode != null)
    setNodeWidget(node, "size_mode", SIZE_MODE_LABEL[extra.size_mode] || "keep shape (long side)");
  if (extra.blend_mode != null)
    setNodeWidget(node, "blend_mode", BLEND_MODE_LABEL[extra.blend_mode] || "mask");
}

// 复制出来的节点要有自己的磁盘空间。state_json 里的 project_id 给磁盘上的
// src/mask 文件做键；复制（alt-drag、右键 Duplicate、Ctrl+C/V）会原样拷贝，
// 从副本的编辑器保存会覆盖父节点的文件。若同类型另一个存活的节点已持有该
// id，重新铸造 id + 清空路径，副本从空白隔离状态开始。干净的工作流加载
// 不会有两个节点共享 id，因此加载时是无操作（不写）-> 加载不脏。
function dedupeInpaintProjectId(node) {
  try {
    const w = node.widgets?.find((x) => x.name === "InpaintCropWidget");
    if (!w) return;
    let meta;
    try { meta = JSON.parse(w.value?.state_json || "{}"); } catch { return; }
    const myId = meta?.project_id;
    if (!myId) return;
    const g = node.graph || app.graph;
    const nodes = g?._nodes || g?.nodes || [];
    const collides = nodes.some((n) => {
      if (n === node || n?.comfyClass !== node.comfyClass) return false;
      const ow = n.widgets?.find((x) => x.name === "InpaintCropWidget");
      if (!ow) return false;
      let om; try { om = JSON.parse(ow.value?.state_json || "{}"); } catch { return false; }
      return om?.project_id === myId;
    });
    if (!collides) return;
    meta.project_id = "inpaint_" + Date.now() + "_" + Math.random().toString(36).slice(2, 9);
    meta.src_path = "";
    meta.mask_path = "";
    w.value = { state_json: JSON.stringify(meta) };
    if (typeof node._sfInpaintJsonSync === "function") node._sfInpaintJsonSync(JSON.stringify(meta));
    // 同时清掉拷贝过来的缓存源引用，否则节点缩略图一直显示父节点的图
    // （restoreNodePreview 无法抹掉已画好的图）。然后重画：接入上游就画
    // 上游，否则空白占位。
    node._pixInpaintSourceURL = null;
    if (node.properties) delete node.properties.pixInpaintSource;
    if (getUpstreamImageURL(node)) node._pixInpaintRefresh?.();
    else node._pixInpaintClearPreview?.();
  } catch (e) { console.warn("[InpaintCrop] dedupe project id failed:", e); }
}

function buildSourceURL(part, bust) {
  if (!part || !part.filename) return null;
  // 缓存破坏参数属于交给 pixApiUrl 的 ROUTE 部分，绝不追加到其 RESULT：
  // 托管 ComfyUI 会给完成的 url 追加认证 token，之后拼接会把参数写在 token
  // 另一侧。本地两条路径产出相同字符串。
  return pixApiUrl(`/view?filename=${encodeURIComponent(part.filename)}` +
    `&subfolder=${encodeURIComponent(part.subfolder || "")}` +
    `&type=${encodeURIComponent(part.type || "temp")}` +
    (bust ? `&t=${Date.now()}` : ""));
}

// LoadImage combo 值可能是 "name.png"、"sub/name.png"，或带标注如
// "clipspace/clipboard.png [input]"（粘贴到 LoadImage 的产物——子目录 +
// 后缀）。/view 要 filename + subfolder 分开、无标注，所以拆开，否则编辑器
// 404 "Failed to load the source image"（只在粘贴时出现，普通选择没有
// 子目录/后缀）。
function parseAnnotatedImageValue(value) {
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

function getUpstreamImageURL(node) {
  // 优先存活的接线源，刚换过的 Load Image（或任何实时预览）才是编辑器打开的
  // 图。下面的执行期缓存 URL 只是生成型上游的回退（其像素只存在于 Python
  // 节点上次运行保存的 temp PNG）。不按这个顺序，换 Load Image 文件后显示
  // 的是上次运行的旧图直到重跑。
  const input = (node.inputs || []).find((i) => i.name === "image");
  const graph = node.graph;
  if (input && input.link != null && graph) {
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
  // 回退：上次 Python 执行保存的源 PNG（生成型上游，或实时预览出现前），
  // 以及粘贴 / 拖放 / 恢复的情形。
  if (node._pixInpaintSourceURL) return node._pixInpaintSourceURL;
  return null;
}

// ── 剪贴板粘贴 → 选中的 SF Inpaint Crop 节点 ──
let _pasteInstalled = false;
function installPasteHandler() {
  if (_pasteInstalled) return;
  _pasteInstalled = true;
  window.addEventListener("paste", async (e) => {
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    const node = findActiveNode();
    if (!node) return;
    // 编辑器开着 -> 让编辑器自己的 paste 处理器把图加载进 canvas
    if (node._pixInpaintEditor?.el?.overlay?.isConnected) return;
    const items = e.clipboardData?.items || [];
    const it = Array.from(items).find((x) => x.type?.startsWith("image/"));
    if (!it) return;
    e.preventDefault(); e.stopImmediatePropagation();
    const idx = (node.inputs || []).findIndex((i) => i.name === "image");
    if (idx >= 0 && node.inputs[idx].link != null) { try { node.disconnectInput(idx); } catch {} }
    const idsBefore = new Set((app.graph?._nodes || []).map((n) => n.id));
    const blob = it.getAsFile();
    if (!blob) return;
    const reader = new FileReader();
    reader.onload = (ev) => node._pixInpaintPaste(ev.target.result);
    reader.readAsDataURL(blob);
    setTimeout(() => {
      for (const n of app.graph?._nodes || []) {
        if (idsBefore.has(n.id)) continue;
        if (n.comfyClass !== "LoadImage" && n.type !== "LoadImage") continue;
        const w = (n.widgets || []).find((x) => x.name === "image");
        if (typeof w?.value === "string" && w.value.startsWith("pasted/")) { try { app.graph.remove(n); } catch {} }
      }
    }, 50);
  }, true);
}

function findActiveNode() {
  const c = app.canvas;
  if (!c) return null;
  const ok = (n) => n && n.comfyClass === "SFInpaintCrop" && typeof n._pixInpaintPaste === "function";
  const sel = c.selected_nodes;
  if (sel) {
    let iter = Array.isArray(sel) ? sel : (typeof sel.values === "function" ? Array.from(sel.values()) : Object.values(sel));
    const hit = iter?.find(ok);
    if (hit) return hit;
  }
  if (ok(c.current_node)) return c.current_node;
  if (ok(c.node_over)) return c.node_over;
  for (const n of app.graph?._nodes || []) if (ok(n) && (n.is_selected || n.flags?.is_selected)) return n;
  return null;
}

app.registerExtension({
  name: "sfnodes.InpaintCrop",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SFInpaintCrop") return;
    const origExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExec?.apply(this, arguments);
      this.imgs = null;
    };
    const origCfg = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
      const ret = origCfg?.apply(this, arguments);
      this.imgs = null;
      if (!this._pixInpaintSourceURL && this.properties?.pixInpaintSource) {
        this._pixInpaintSourceURL = buildSourceURL(this.properties.pixInpaintSource, true);
      }
      if (this._pixInpaintRefresh) {
        queueMicrotask(() => this._pixInpaintRefresh());
        setTimeout(() => this._pixInpaintRefresh?.(), 250);
      }
      // 恢复隐藏状态 widget（SFInpaintJson）→ 闭包。工作流加载时 nodeCreated
      // 早于 widget 值恢复，因此延迟到 configure 之后读取（microtask +
      // setTimeout 双保险，同 sf_pause / sf_crop 模式）。
      const _restoreJson = () => {
        if (typeof this._sfInpaintJsonSync !== "function") return;
        const jw = this.widgets?.find((w) => w.name === "SFInpaintJson");
        const saved = jw?.value;
        if (typeof saved === "string" && saved && saved !== "{}") {
          this._sfInpaintJsonSync(saved);
        }
      };
      queueMicrotask(_restoreJson);
      setTimeout(_restoreJson, 250);
      // After the node settles, give a DUPLICATE its own project_id (see
      // dedupeInpaintProjectId). Deferred a microtask so a clipboard paste has
      // added the node to its graph (so this.graph + the sibling are both live).
      queueMicrotask(() => { if (!isGraphLoading()) dedupeInpaintProjectId(this); });
      return ret;
    };
  },

  async nodeCreated(node) {
    if (node.comfyClass !== "SFInpaintCrop") return;
    node.imgs = null;
    // 只有新建落子时才设默认尺寸；加载路径绝不设（configure 恢复保存的尺寸，
    // 加载时写会弄脏工作流）。按惯例改数组元素，不替换数组。
    if (!isGraphLoading() && node.size) { node.size[0] = 330; node.size[1] = 500; }

    if (!(node.inputs || []).some((i) => i.name === "image")) node.addInput("image", "IMAGE");
    if (!(node.inputs || []).some((i) => i.name === "mask")) node.addInput("mask", "MASK");

    const parts = createNodePreview(
      "Inpaint Crop", "SF",
      "接入 IMAGE 并运行一次，\n或点击 'Open mask editor' 加载并涂抹",
    );
    // 让 dedupe（模块作用域）在复制后清掉此节点的缩略图
    node._pixInpaintClearPreview = () => clearNodePreview(parts, node);

    let stateJson = "{}";
    let widget;

    const refreshSourcePreview = () => {
      const url = getUpstreamImageURL(node);
      if (url) showNodePreview(parts, url, null, node);
    };

    // ── 隐藏状态 widget（数据载体，随 workflow 保存/加载/复制）。Vue 前端
    // 对 DOM widget（InpaintCropWidget）的 serializeValue 值序列化不可靠，
    // 因此用隐藏 STRING widget 持久化 + graphToPrompt / queuePrompt 注入覆盖
    // （项目惯例，同 sf_pause_* / sf_crop 系列）。Python hidden 已声明
    // SFInpaintJson，schema 内不被 validatePrompt 剥离。
    const sfJsonWidget = node.addWidget("STRING", "SFInpaintJson", "{}", () => {});
    sfJsonWidget.hidden = true;
    sfJsonWidget.computeSize = () => [0, -4];
    if (!sfJsonWidget.options) sfJsonWidget.options = {};
    sfJsonWidget.options.canvasOnly = true;
    // 统一保存路径：更新闭包 + 隐藏 widget（随 workflow 持久化）。绝不能
    // 在这里写 DOM widget 的 .value（Vue 的 setValue 回调链会无限递归）。
    node._sfInpaintJsonSync = (jsonStr) => {
      stateJson = jsonStr;
      sfJsonWidget.value = jsonStr;
    };
    node._sfInpaintJsonGet = () => stateJson;

    // ── Open mask editor 按钮 ──
    node.addWidget("button", "Open mask editor", null, () => {
      if (node._pixInpaintEditor?.el?.overlay?.isConnected) return;
      refreshSourcePreview();   // 同步节点缩略图到当前上游图
      const editor = new InpaintCropEditor();
      node._pixInpaintEditor = editor;
      // 笔刷大小 / 不透明度跨打开在此节点持久化
      const captureBrush = () => {
        node._pixInpaintBrush = { brushSize: editor.brushSize, maskOpacity: editor.maskOpacity };
      };

      // 预览色调（仅显示）——从设置读初值，变更时持久化
      const colName = app.ui.settings?.getSettingValue?.("sfnodes.Inpaint.PreviewColor") || "Red";
      const colHex = INPAINT_PREVIEW_COLORS[colName] || INPAINT_PREVIEW_COLORS.Red;
      editor.previewColor = colHex;
      editor._cropBoxColor = (colHex === INPAINT_PREVIEW_COLORS.Orange) ? "#ffffff" : null;
      editor.onPreviewColor = (name) => {
        try { app.ui.settings?.setSettingValueAsync?.("sfnodes.Inpaint.PreviewColor", name); } catch {}
      };

      editor.onSave = (jsonStr, extra, preview) => {
        node._sfInpaintJsonSync(jsonStr);
        writeBackWidgets(node, extra);
        if (preview) showNodePreview(parts, preview, null, node);
        if (app.graph) { app.graph.setDirtyCanvas(true, true); app.graph.change?.(); }
        captureBrush();
      };
      editor.onSaveToDisk = (d) => downloadDataURL(d, "sf_inpaint_crop");
      editor.onLoadImage = () => {
        const idx = (node.inputs || []).findIndex((i) => i.name === "image");
        if (idx >= 0 && node.inputs[idx].link != null) { try { node.disconnectInput(idx); } catch {} }
      };
      editor.onClose = () => { captureBrush(); node._pixInpaintEditor = null; node.setDirtyCanvas(true, true); };

      editor.open(stateJson, getUpstreamImageURL(node),
        readParams(node), node._pixInpaintBrush);
    });

    // ── 迷你预览 DOM widget（同时携带隐藏状态）──
    installCanvasZoomPassthrough(parts.container);
    widget = node.addDOMWidget("InpaintCropWidget", "custom", parts.container, {
      getValue: () => ({ state_json: stateJson }),
      setValue: (v) => {
        if (!v || typeof v !== "object") return;
        node._sfInpaintJsonSync(v.state_json || "{}");
        const imgInput = (node.inputs || []).find((i) => i.name === "image");
        if (imgInput && imgInput.link != null) queueMicrotask(refreshSourcePreview);
        else restoreNodePreview(parts, stateJson, node);   // 有 src_path 时从磁盘重建
      },
      getMinHeight: () => 200,
      margin: 5,
    });
    applyAdaptiveCanvasOnly(widget);
    activateNodePreview(parts, node);

    // ── 直接往节点上粘贴 / 拖放源图 ──
    installPasteHandler();
    node._pixInpaintPaste = async (dataURL) => {
      try {
        const r = await api.fetchApi(pixApiUrl("/api/sfnodes/inpaint/upload_src"), {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project_id: "inpaint_paste_" + Date.now() + "_" + Math.random().toString(36).slice(2, 9), image: dataURL }),
        });
        const d = await r.json();
        const srcPath = d.path || "";
        let meta = {};
        try { meta = JSON.parse(stateJson) || {}; } catch {}
        meta.src_path = srcPath;
        meta.mask_path = "";  // 新源清掉旧画的遮罩
        node._sfInpaintJsonSync(JSON.stringify(meta));
        if (srcPath) {
          const part = { filename: srcPath.split(/[\\/]/).pop(), subfolder: "sfnodes_inpaint", type: "input" };
          node._pixInpaintSourceURL = buildSourceURL(part, true);
          if (!node.properties) node.properties = {};
          node.properties.pixInpaintSource = part;
        }
        showNodePreview(parts, dataURL, null, node);
        if (app.graph) app.graph.setDirtyCanvas(true, true);
      } catch (err) { console.warn("[InpaintCrop] paste failed:", err); }
    };
    const dropTarget = parts?.container;
    if (dropTarget) {
      dropTarget.addEventListener("dragover", (e) => { if (e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); e.stopPropagation(); } });
      dropTarget.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation();
        const file = e.dataTransfer?.files?.[0];
        if (!file || !file.type?.startsWith("image/")) return;
        const idx = (node.inputs || []).findIndex((i) => i.name === "image");
        if (idx >= 0 && node.inputs[idx].link != null) { try { node.disconnectInput(idx); } catch {} }
        const reader = new FileReader();
        reader.onload = (ev) => node._pixInpaintPaste(ev.target.result);
        reader.readAsDataURL(file);
      });
    }

    // ── 执行期源 URL 缓存 + 刷新钩子 ──
    node._pixInpaintRefresh = () => {
      if (getUpstreamImageURL(node)) refreshSourcePreview();
      // 传真实状态（不是 "{}"）让 restoreNodePreview 能按 src_path 重建
      // ——如编辑器 Load Image 加载的源，此时连线已断。
      else restoreNodePreview(parts, stateJson, node);
    };
    const onExec = (event) => {
      const detail = event?.detail;
      if (!detail?.output) return;
      const matched = app.graph.getNodeById(detail.node) || app.graph.getNodeById(parseInt(detail.node, 10));
      if (matched !== node) return;
      const frames = detail.output.sf_inpaint_source;
      if (!frames?.length) return;
      const f = frames[0];
      const part = { filename: f.filename, subfolder: f.subfolder || "", type: f.type || "temp" };
      node._pixInpaintSourceURL = buildSourceURL(part, true);
      if (!node.properties) node.properties = {};
      node.properties.pixInpaintSource = part;
      refreshSourcePreview();
    };
    api.addEventListener("executed", onExec);

    // wrap（不要覆盖）原型/其他扩展已有的处理器；转发所有参数，然后跑
    // 我们的 image 输入源预览逻辑。
    const origConnChange = node.onConnectionsChange;
    node.onConnectionsChange = function (type, slotIndex, connected) {
      const r = origConnChange?.apply(this, arguments);
      if (type === LiteGraph.INPUT && node.inputs?.[slotIndex]?.name === "image" && !isGraphLoading()) {
        node._pixInpaintSourceURL = null;
        if (node.properties) delete node.properties.pixInpaintSource;
        if (connected) refreshSourcePreview();
        else restoreNodePreview(parts, "{}", node);
      }
      return r;
    };

    const origRemoved = node.onRemoved;
    node.onRemoved = () => {
      try { if (node._pixInpaintEditor?.el?.overlay?.isConnected) node._pixInpaintEditor._close(); } catch (e) {}
      try { parts?.resizeObserver?.disconnect(); } catch (e) {}
      origRemoved?.call(node);
      try { api.removeEventListener("executed", onExec); } catch {}
    };
  },
});

// ── app.graphToPrompt hook（隐藏状态数据载体，双保险） ─────────────────────
// SFInpaintJson 隐藏 STRING widget 的值走标准 widget 收集（Python hidden
// 声明保证前端 validatePrompt 不剥离）。这里在提交时覆盖为最新闭包值——
// 覆盖加载/保存时序差，fail-open 不影响 Run。
const _origInpaintGraphToPrompt = app.graphToPrompt.bind(app);
app.graphToPrompt = async function (...args) {
  const result = await _origInpaintGraphToPrompt(...args);
  const out = result?.output;
  if (out) {
    let index = null;
    for (const id in out) {
      const entry = out[id];
      if (!entry || entry.class_type !== "SFInpaintCrop") continue;
      if (!index) {
        index = new Map();
        const visit = (graph) => {
          if (!graph) return;
          const nodes = graph._nodes || graph.nodes || [];
          for (const n of nodes) {
            if (!n) continue;
            if (n.comfyClass === "SFInpaintCrop" || n.type === "SFInpaintCrop") index.set(String(n.id), n);
            const inner = n.subgraph || n.graph || n._graph;
            if (inner && inner !== graph) visit(inner);
          }
        };
        visit(app.graph);
      }
      const node = index.get(String(id)) || null;
      entry.inputs = entry.inputs || {};
      // SFInpaintJson 已在 Python hidden 声明（schema 内，前端不剥离）；这里
      // 覆盖原生收集值作为双保险。InpaintCropWidget 不在 schema 会被剥离，
      // 不再注入（后端 IS_CHANGED 的 CropWidget 回退仅兼容旧 prompt）。
      entry.inputs.SFInpaintJson = node?._sfInpaintJsonGet?.() || "{}";
    }
  }
  return result;
};

// ── api.queuePrompt 提交时注入（最终漏斗，双保险） ─────────────────────────
// api.queuePrompt 是所有浏览器 run 的唯一漏斗（项目 sf_pause / sf_crop 先例），
// 提交前在此兜底覆盖 SFInpaintJson（Python hidden 声明，schema 内不被剥离）。
if (!api._sfInpaintQueueWrapped) {
  api._sfInpaintQueueWrapped = true;
  const _origInpaintQueuePrompt = api.queuePrompt.bind(api);
  api.queuePrompt = async function (...args) {
    try {
      const out = args[1]?.output;
      if (out) {
        let index = null;
        for (const id in out) {
          const entry = out[id];
          if (!entry || entry.class_type !== "SFInpaintCrop") continue;
          if (!index) {
            index = new Map();
            const visit = (graph) => {
              if (!graph) return;
              const nodes = graph._nodes || graph.nodes || [];
              for (const n of nodes) {
                if (!n) continue;
                if (n.comfyClass === "SFInpaintCrop" || n.type === "SFInpaintCrop") index.set(String(n.id), n);
                const inner = n.subgraph || n.graph || n._graph;
                if (inner && inner !== graph) visit(inner);
              }
            };
            visit(app.graph);
          }
          const node = index.get(String(id)) || null;
          entry.inputs = entry.inputs || {};
          entry.inputs.SFInpaintJson = node?._sfInpaintJsonGet?.() || "{}";
        }
      }
    } catch (err) {
      // 注入失败绝不能挡住用户的 run
      console.error("[SFInpaintCrop] submit-time state injection failed", err);
    }
    return _origInpaintQueuePrompt(...args);
  };
}
