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
import {
  sfApiUrl,
  isGraphLoading,
  isVueNodes,
  applyAdaptiveCanvasOnly,
  installCanvasZoomPassthrough,
  installPasteHandler,
  getUpstreamImageURL,
  buildSourceURL,
  parseAnnotatedImageValue,
} from "./sf_common.js";

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
    node._sfInpaintSourceURL = null;
    if (node.properties) {
      delete node.properties.sfInpaintSource;
      delete node.properties.pixInpaintSource;
    }
    if (getUpstreamImageURL(node, node._sfInpaintSourceURL)) node._sfInpaintRefresh?.();
    else node._sfInpaintClearPreview?.();
  } catch (e) { console.warn("[InpaintCrop] dedupe project id failed:", e); }
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
      // 恢复缓存源 URL（Vue Compat #11）。sfInpaintSource 是当前键；旧键
      // pixInpaintSource 仍读兜底（rename 前保存的工作流）。
      if (!this._sfInpaintSourceURL) {
        if (this.properties?.sfInpaintSource) {
          this._sfInpaintSourceURL = buildSourceURL(this.properties.sfInpaintSource, true);
        } else if (this.properties?.pixInpaintSource) {
          this._sfInpaintSourceURL = buildSourceURL(this.properties.pixInpaintSource, true);
        }
      }
      if (this._sfInpaintRefresh) {
        queueMicrotask(() => this._sfInpaintRefresh());
        setTimeout(() => this._sfInpaintRefresh?.(), 250);
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
    node._sfInpaintClearPreview = () => clearNodePreview(parts, node);

    let stateJson = "{}";
    let widget;

    const refreshSourcePreview = () => {
      const url = getUpstreamImageURL(node, node._sfInpaintSourceURL);
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
      // 编辑器 Load Image / 粘贴更新了 src_path 时同步预览缓存：否则节点预览
      // 停留在旧源图（保存退出不运行也错，运行后靠 executed 事件兜底修复）。
      // src_path 不变（如拖动遮罩）则跳过，零成本。
      let prevSrc = "", newSrc = "";
      try { prevSrc = JSON.parse(stateJson || "{}").src_path || ""; } catch {}
      try { newSrc = JSON.parse(jsonStr || "{}").src_path || ""; } catch {}
      if (newSrc && newSrc !== prevSrc) {
        const part = {
          filename: newSrc.split(/[\\/]/).pop(),
          subfolder: "sfnodes_inpaint",
          type: "input",
        };
        node._sfInpaintSourceURL = buildSourceURL(part, true);
        if (!node.properties) node.properties = {};
        node.properties.sfInpaintSource = part;
        delete node.properties.pixInpaintSource;
        // 节点无轮询（crop 有 pollInterval 兜底），主动刷新预览；加载路径
        // 由 onConfigure 的恢复逻辑负责，这里门控跳过。
        if (!isGraphLoading()) refreshSourcePreview();
      }
      stateJson = jsonStr;
      sfJsonWidget.value = jsonStr;
    };
    node._sfInpaintJsonGet = () => stateJson;

    // ── Open mask editor 按钮 ──
    node.addWidget("button", "Open mask editor", null, () => {
      if (node._sfInpaintEditor?.el?.overlay?.isConnected) return;
      refreshSourcePreview();   // 同步节点缩略图到当前上游图
      const editor = new InpaintCropEditor();
      node._sfInpaintEditor = editor;
      // 笔刷大小 / 不透明度跨打开在此节点持久化
      const captureBrush = () => {
        node._sfInpaintBrush = { brushSize: editor.brushSize, maskOpacity: editor.maskOpacity };
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
      editor.onClose = () => { captureBrush(); node._sfInpaintEditor = null; node.setDirtyCanvas(true, true); };

      editor.open(stateJson, getUpstreamImageURL(node, node._sfInpaintSourceURL),
        readParams(node), node._sfInpaintBrush);
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
    installPasteHandler({
      comfyClass: "SFInpaintCrop",
      hook: "_sfInpaintPaste",
      // 编辑器开着 -> 让编辑器自己的 paste 处理器把图加载进 canvas
      allowPaste: (n) => !(n._sfInpaintEditor?.el?.overlay?.isConnected),
      onPasteImage: (n, dataURL) => n._sfInpaintPaste(dataURL),
    });
    node._sfInpaintPaste = async (dataURL) => {
      try {
        const r = await api.fetchApi(sfApiUrl("/api/sfnodes/inpaint/upload_src"), {
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
          node._sfInpaintSourceURL = buildSourceURL(part, true);
          if (!node.properties) node.properties = {};
          node.properties.sfInpaintSource = part;
          delete node.properties.pixInpaintSource;
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
        reader.onload = (ev) => node._sfInpaintPaste(ev.target.result);
        reader.readAsDataURL(file);
      });
    }

    // ── 执行期源 URL 缓存 + 刷新钩子 ──
    node._sfInpaintRefresh = () => {
      if (getUpstreamImageURL(node, node._sfInpaintSourceURL)) refreshSourcePreview();
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
      node._sfInpaintSourceURL = buildSourceURL(part, true);
      if (!node.properties) node.properties = {};
      node.properties.sfInpaintSource = part;
      delete node.properties.pixInpaintSource;
      refreshSourcePreview();
    };
    api.addEventListener("executed", onExec);

    // wrap（不要覆盖）原型/其他扩展已有的处理器；转发所有参数，然后跑
    // 我们的 image 输入源预览逻辑。
    const origConnChange = node.onConnectionsChange;
    node.onConnectionsChange = function (type, slotIndex, connected) {
      const r = origConnChange?.apply(this, arguments);
      if (type === LiteGraph.INPUT && node.inputs?.[slotIndex]?.name === "image" && !isGraphLoading()) {
        node._sfInpaintSourceURL = null;
        if (node.properties) {
          delete node.properties.sfInpaintSource;
          delete node.properties.pixInpaintSource;
        }
        if (connected) refreshSourcePreview();
        else restoreNodePreview(parts, "{}", node);
      }
      return r;
    };

    const origRemoved = node.onRemoved;
    node.onRemoved = () => {
      try { if (node._sfInpaintEditor?.el?.overlay?.isConnected) node._sfInpaintEditor._close(); } catch (e) {}
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
