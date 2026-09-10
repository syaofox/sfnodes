// SFIDClothingSelector 前端（复刻孤海 IDPhotoClothingSelector 交互，复用 styles JSON 生态）
// - 隐藏 SFIDClothingState/SFIDClothingPrompt STRING widget 为值真源（Python
//   "hidden" 声明，标准 widget 收集进 prompt，随 workflow 保存）；DOM widget
//   纯交互不承担值传输（规避 Vue DOMWidget value setter 链，见 experience/nodes-image.md §11）
// - 单选画廊：搜索过滤 / Reset 清空 / 选中置顶 / 悬停缩略图预览（结构对齐
//   sf_styles_selector.js，CSS 类前缀 sf-idc- 隔离）
// - 提示词可手改：编辑框输入写入草稿 widget（空=回落模板原词，切换模板即清空，
//   对齐 SFTextPreset text_override 语义）；后端草稿优先输出
// - 纯逻辑（解析/单选/过滤/语言化）在 sf_id_clothing_lib.js

import { app } from "/scripts/app.js";
import { applyAdaptiveCanvasOnly, hideJsonWidget, injectCSSOnce, installWheelZoomPassthrough, isGraphLoading, isVueNodes, sfApiUrl } from "./sf_common.js";
import * as lib from "./sf_id_clothing_lib.js";

const NODE_TYPE = "SFIDClothingSelector";
const LIST_H = 300; // 列表固定高度（滚动容器）
const TOOLS_H = 34; // 工具条 + 与列表的 gap
const PROMPT_H = 64; // 提示词编辑框固定高度
const GAP = 6;
const ROOT_PAD = 8; // root padding 上下各 4
// 声称高度必须 ≥ 内容实际高度——低于内容时节点边框按声称高度绘制、
// 底部内容溢出被裁（对齐 sf_styles_selector.js §18.6 注释）
const WIDGET_H = LIST_H + TOOLS_H + PROMPT_H + GAP * 2 + ROOT_PAD + 4;
const MIN_W = 260;
const VIEW_PROP = "sfIDClothingView"; // Grid/List 显示模式（随 workflow 保存，不注入 prompt）
const EMPTY_IMG =
  "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";

function injectCSS() {
  injectCSSOnce("sf-id-clothing-css", `
.sf-idc-root{display:flex;flex-direction:column;gap:6px;height:100%;box-sizing:border-box;padding:4px 6px;position:relative;overflow:visible;}
.sf-idc-tools{display:flex;gap:6px;flex:0 0 auto;}
.sf-idc-search{flex:1;min-width:0;resize:none;font:12px sans-serif;color:#ddd;background:#1d1d1d;border:1px solid #333;border-radius:5px;padding:4px 6px;height:28px;box-sizing:border-box;outline:none;}
.sf-idc-reset{flex:0 0 auto;font:11px sans-serif;color:var(--sf-acc, #f66744);background:color-mix(in srgb, var(--sf-acc, #f66744) 12%, transparent);border:1px solid color-mix(in srgb, var(--sf-acc, #f66744) 45%, transparent);border-radius:5px;padding:0 10px;cursor:pointer;height:28px;}
.sf-idc-reset:hover{background:color-mix(in srgb, var(--sf-acc, #f66744) 22%, transparent);}
.sf-idc-viewseg{flex:0 0 auto;display:flex;border:1px solid #444;border-radius:5px;overflow:hidden;height:28px;}
.sf-idc-viewseg button{font:11px sans-serif;color:#c8c8c8;background:#2a2a2a;border:none;padding:0 8px;cursor:pointer;}
.sf-idc-viewseg button:hover{background:#3a3a3a;}
.sf-idc-viewseg button.sf-idc-viewon{background:color-mix(in srgb, var(--sf-acc, #f66744) 25%, transparent);color:#fff;}
.sf-idc-prompt{flex:0 0 auto;width:100%;height:${PROMPT_H}px;min-height:${PROMPT_H}px;max-height:${PROMPT_H}px;resize:none;font:12px/1.5 sans-serif;color:#ddd;background:#1d1d1d;border:1px solid #333;border-radius:5px;padding:6px 8px;box-sizing:border-box;outline:none;overflow-y:auto;}
.sf-idc-list{flex:0 0 auto;min-height:150px;height:calc(100% - 12px);overflow-y:auto;overflow-x:hidden;display:flex;flex-direction:column;gap:2px;padding:2px;box-sizing:border-box;}
.sf-idc-list.sf-idc-grid{display:grid !important;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));grid-auto-rows:max-content;gap:8px;align-content:start;padding:4px 0;}
.sf-idc-card{display:flex;flex-direction:column;gap:3px;padding:4px;border-radius:6px;cursor:pointer;background:#222;border:1px solid #3a3a3a;overflow:hidden;flex:0 0 auto;}
.sf-idc-card img{width:100%;height:auto;aspect-ratio:1/1;object-fit:contain;border-radius:4px;background:#1a1a1a;flex:0 0 auto;display:block;}
.sf-idc-card span{font:11px sans-serif;color:#ccc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center;user-select:none;}
.sf-idc-card:hover{border-color:#555;}
.sf-idc-cardsel{border-color:var(--sf-acc, #f66744);background:color-mix(in srgb, var(--sf-acc, #f66744) 12%, transparent);}
.sf-idc-cardsel span{color:#fff;}
.sf-idc-tag{display:flex;align-items:center;gap:6px;padding:3px 6px;border-radius:4px;cursor:pointer;font:12px sans-serif;color:#ccc;user-select:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:0 0 auto;}
.sf-idc-tag input{flex:0 0 auto;accent-color:var(--sf-acc, #f66744);pointer-events:none;}
.sf-idc-tag span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.sf-idc-tag:hover{background:#2c2c2c;}
.sf-idc-sel{background:color-mix(in srgb, var(--sf-acc, #f66744) 16%, transparent);color:#fff;}
.sf-idc-hide{display:none;}
.sf-idc-pop{position:absolute;display:none;pointer-events:none;width:210px;border-radius:8px;border:1px solid #4a4a4a;background:#202020;box-shadow:0 8px 22px rgba(0,0,0,.65);z-index:10;padding:6px;box-sizing:border-box;}
.sf-idc-popimg{width:100%;height:118px;object-fit:contain;background:#131313;border-radius:6px;display:block;}
.sf-idc-popname{display:block;font:12px sans-serif;color:#fff;margin:5px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sf-idc-poppos{font:10px/1.4 sans-serif;margin:2px 0;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;word-wrap:break-word;}
.sf-idc-poppos b{color:#7bd88f;}
.sf-idc-poppos span{color:#9ecfa8;}`);
}

// 模板库列表缓存（promise 级，避免加载期重复请求；失败缓存空列表会话内不重试）
const libraryCache = {};
function fetchLibrary(name) {
  if (!libraryCache[name]) {
    libraryCache[name] = (async () => {
      try {
        const resp = await fetch(sfApiUrl(`${lib.STYLES_API}?name=${encodeURIComponent(name)}`));
        if (resp.ok) {
          const data = await resp.json();
          return Array.isArray(data) ? data : [];
        }
      } catch (e) {
        console.warn("sfnodes.IDClothingSelector: 拉取模板库失败", name, e);
      }
      return [];
    })();
  }
  return libraryCache[name];
}

function isZh() {
  return (navigator.language || "en").toLowerCase().startsWith("zh");
}

function stateWidget(node) {
  return (node.widgets || []).find((w) => w.name === lib.STATE_WIDGET) || null;
}

function promptWidget(node) {
  return (node.widgets || []).find((w) => w.name === lib.PROMPT_WIDGET) || null;
}

function readSelection(node) {
  const w = stateWidget(node);
  return lib.firstSelected(w ? w.value : "");
}

function writeSelection(node, name) {
  const w = stateWidget(node);
  if (w) w.value = lib.serializeSingle(name);
}

function readDraft(node) {
  const w = promptWidget(node);
  return w && w.value != null ? String(w.value) : "";
}

function writeDraft(node, text) {
  const w = promptWidget(node);
  if (w) w.value = String(text != null ? text : "");
}

function currentLibraryName(node) {
  const w = (node.widgets || []).find((x) => x.name === "library");
  return w && w.value ? String(w.value) : "";
}

// Grid/List 显示模式：存 node.properties 随 workflow 保存（对齐 styles 实现）
function viewMode(node) {
  const v = node.properties && node.properties[VIEW_PROP];
  return v === "list" ? "list" : "grid";
}

function setViewMode(node, mode) {
  if (!node.properties) node.properties = {};
  node.properties[VIEW_PROP] = mode;
}

// 单选切换：点已选项=取消（输出空），点新项=选中并清空草稿（切换即弃，
// 对齐 SFTextPreset text_override 语义）；加载期门控点击防覆盖刚恢复的选择
function toggleSelect(node, ctx, name) {
  if (isGraphLoading()) return;
  const cur = readSelection(node);
  if (cur === name) {
    writeSelection(node, "");
  } else {
    writeSelection(node, name);
    writeDraft(node, "");
  }
  renderPrompt(ctx);
  renderList(ctx);
}

// hover 信息浮窗（缩略图 + 名称 + 提示词）
const POP_W = 214;
const POP_H = 240;

function placePop(pop, root, e) {
  const r = root.getBoundingClientRect();
  const scale = window.LiteGraph?.ds?.scale || 1;
  let x = (e.clientX - r.left + 14) / scale;
  let y = (e.clientY - r.top - 8) / scale;
  if (x + POP_W > r.width / scale - 4) x = r.width / scale - POP_W - 4;
  if (y + POP_H > r.height / scale - 4) y = r.height / scale - POP_H - 4;
  x = Math.max(4, x);
  y = Math.max(4, y);
  pop.style.left = `${x}px`;
  pop.style.top = `${y}px`;
}

function showPop(ctx, item, e) {
  const { popEl, root } = ctx;
  const raw = item.raw || {};
  const thumb = lib.thumbnailOf(raw);
  const img = popEl.querySelector(".sf-idc-popimg");
  if (img.dataset.src !== thumb) {
    img.dataset.src = thumb;
    img.src = thumb ? (lib.isRemoteThumb(thumb) ? thumb : sfApiUrl(thumb)) : EMPTY_IMG;
  }
  popEl.querySelector(".sf-idc-popname").textContent = item.label;
  const pos = popEl.querySelector(".sf-idc-poppos");
  const prompt = raw.prompt ? String(raw.prompt) : "";
  pos.style.display = prompt ? "" : "none";
  if (prompt) pos.querySelector("span").textContent = prompt;
  placePop(popEl, root, e);
  popEl.style.display = "block";
}

function hidePop(ctx) {
  ctx.popEl.style.display = "none";
}

// 提示词编辑框显示值：草稿非空优先，否则选中模板原词（后端同语义镜像）
function renderPrompt(ctx) {
  const { node, promptEl, styles } = ctx;
  if (!promptEl) return;
  const sel = readSelection(node);
  const text = lib.displayPrompt(styles || [], sel, readDraft(node));
  if (promptEl.value !== text) promptEl.value = text;
}

function makeTag(item, ctx) {
  const { node } = ctx;
  const label = document.createElement("label");
  label.className = "sf-idc-tag" + (item.selected ? " sf-idc-sel" : "") + (item.hidden ? " sf-idc-hide" : "");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = item.selected;
  const span = document.createElement("span");
  span.textContent = item.label;
  label.append(cb, span);

  label.onclick = (e) => {
    // 阻止 label 默认激活 checkbox：默认行为会合成 input.click() 并冒泡回
    // label，造成 onclick 二次触发（选中又取消、表现"点不动"）
    e.preventDefault();
    toggleSelect(node, ctx, item.name);
  };

  label.onmouseenter = (e) => showPop(ctx, item, e);
  label.onmousemove = (e) => {
    if (ctx.popEl.style.display !== "none") placePop(ctx.popEl, ctx.root, e);
  };
  label.onmouseleave = () => hidePop(ctx);
  return label;
}

// Grid 视图卡片：缩略图 + 名字（loading="lazy" 避免大库一次性拉取）
function makeCard(item, ctx) {
  const { node } = ctx;
  const card = document.createElement("div");
  card.className = "sf-idc-card" + (item.selected ? " sf-idc-cardsel" : "") + (item.hidden ? " sf-idc-hide" : "");
  const img = document.createElement("img");
  img.loading = "lazy";
  const thumb = lib.thumbnailOf(item.raw || {});
  img.src = thumb ? (lib.isRemoteThumb(thumb) ? thumb : sfApiUrl(thumb)) : EMPTY_IMG;
  img.onerror = () => {
    img.src = EMPTY_IMG;
  };
  const span = document.createElement("span");
  span.textContent = item.label;
  span.title = item.label;
  card.append(img, span);
  card.onclick = () => {
    toggleSelect(node, ctx, item.name);
  };
  card.onmouseenter = (e) => showPop(ctx, item, e);
  card.onmousemove = (e) => {
    if (ctx.popEl.style.display !== "none") placePop(ctx.popEl, ctx.root, e);
  };
  card.onmouseleave = () => hidePop(ctx);
  return card;
}

function renderList(ctx) {
  const { node, listEl, searchEl, styles } = ctx;
  if (!listEl) return;
  listEl.className = "sf-idc-list" + (viewMode(node) === "grid" ? " sf-idc-grid" : "");
  const sel = readSelection(node);
  const items = lib.filterAndSort(styles || [], searchEl.value, sel, isZh());
  listEl.innerHTML = "";
  const make = viewMode(node) === "grid" ? makeCard : makeTag;
  for (const item of items) {
    listEl.append(make(item, ctx));
  }
}

function ensureLoaded(ctx) {
  const name = currentLibraryName(ctx.node);
  if (ctx.name === name && ctx.styles) {
    renderList(ctx);
    renderPrompt(ctx);
    return;
  }
  if (ctx.pending) {
    ctx.pending.then(() => {
      if (currentLibraryName(ctx.node) !== ctx.name) ensureLoaded(ctx); // 加载期间库被切换：重新加载
      else {
        renderList(ctx);
        renderPrompt(ctx);
      }
    });
    return;
  }
  ctx.pending = fetchLibrary(name).then((data) => {
    ctx.pending = null;
    if (currentLibraryName(ctx.node) !== name) return; // 竞态：期间已切换库
    ctx.name = name;
    ctx.styles = data;
    renderList(ctx);
    renderPrompt(ctx);
  });
}

function setupNode(node) {
  injectCSS();

  // ── 隐藏真源 widget（Python hidden 声明，自动存在；缺则补建）──
  let sw = stateWidget(node);
  if (!sw) {
    sw = node.addWidget("STRING", lib.STATE_WIDGET, "[]", () => {});
    sw.hidden = true;
    sw.computeSize = () => [0, -4];
    if (!sw.options) sw.options = {};
    sw.options.canvasOnly = true;
  } else {
    hideJsonWidget(node.widgets, lib.STATE_WIDGET);
  }
  let pw = promptWidget(node);
  if (!pw) {
    pw = node.addWidget("STRING", lib.PROMPT_WIDGET, "", () => {});
    pw.hidden = true;
    pw.computeSize = () => [0, -4];
    if (!pw.options) pw.options = {};
    pw.options.canvasOnly = true;
  } else {
    hideJsonWidget(node.widgets, lib.PROMPT_WIDGET);
  }

  const root = document.createElement("div");
  root.className = "sf-idc-root";

  const tools = document.createElement("div");
  tools.className = "sf-idc-tools";
  const resetBtn = document.createElement("button");
  resetBtn.className = "sf-idc-reset";
  resetBtn.textContent = isZh() ? "重置" : "Reset";
  resetBtn.title = isZh() ? "清空已选模板" : "Clear selected template";
  const searchEl = document.createElement("textarea");
  searchEl.className = "sf-idc-search";
  searchEl.rows = 1;
  searchEl.placeholder = isZh() ? "🔎 搜索模板 ..." : "🔎 Search templates ...";
  installWheelZoomPassthrough(searchEl);

  // Grid/List 显示模式切换
  const zh = isZh();
  const viewSeg = document.createElement("div");
  viewSeg.className = "sf-idc-viewseg";
  const syncViewBtns = () => {
    const cur = viewMode(node);
    for (const btn of viewSeg.children) {
      btn.classList.toggle("sf-idc-viewon", btn.dataset.mode === cur);
    }
  };
  const viewBtn = (mode, icon, label) => {
    const b = document.createElement("button");
    b.dataset.mode = mode;
    b.textContent = icon;
    b.title = label;
    b.onclick = () => {
      setViewMode(node, mode);
      syncViewBtns();
      renderList(ctx);
    };
    viewSeg.append(b);
    return b;
  };
  viewBtn("grid", "▦", zh ? "网格视图" : "Grid view");
  viewBtn("list", "☰", zh ? "列表视图" : "List view");
  syncViewBtns();

  tools.append(resetBtn, searchEl, viewSeg);

  // 提示词编辑框：手改写入草稿 widget（空=回落模板原词）
  const promptEl = document.createElement("textarea");
  promptEl.className = "sf-idc-prompt";
  promptEl.placeholder = isZh()
    ? "模板提示词（选中后自动填入，可手改；清空=回落模板原词）..."
    : "Template prompt (auto-filled on select; edit to override; clear to fall back) ...";
  promptEl.addEventListener("keydown", (e) => e.stopPropagation());
  installWheelZoomPassthrough(promptEl);

  const listEl = document.createElement("div");
  listEl.className = "sf-idc-list";

  // hover 信息浮窗（图 + 名称 + 提示词）
  const popEl = document.createElement("div");
  popEl.className = "sf-idc-pop";
  const popImg = document.createElement("img");
  popImg.className = "sf-idc-popimg";
  popImg.src = EMPTY_IMG;
  popImg.onerror = () => {
    popImg.src = EMPTY_IMG;
  };
  const popName = document.createElement("span");
  popName.className = "sf-idc-popname";
  const popPos = document.createElement("div");
  popPos.className = "sf-idc-poppos";
  const popPosB = document.createElement("b");
  popPosB.textContent = "Prompt: ";
  const popPosT = document.createElement("span");
  popPos.append(popPosB, popPosT);
  popEl.append(popImg, popName, popPos);

  root.append(tools, promptEl, listEl, popEl);

  const ctx = {
    node,
    root,
    listEl,
    searchEl,
    promptEl,
    popEl,
    styles: null,
    name: null,
    pending: null,
  };

  resetBtn.onclick = () => {
    if (isGraphLoading()) return;
    writeSelection(node, "");
    writeDraft(node, "");
    renderPrompt(ctx);
    renderList(ctx);
  };
  searchEl.oninput = () => renderList(ctx);
  promptEl.oninput = () => {
    writeDraft(node, promptEl.value);
  };

  // 列表高度显式管理：不用 CSS 百分比（父容器高度不确定时 calc 失效）
  const fitListHeight = () => {
    if (!root.clientHeight) return;
    const toolsH = tools.offsetHeight || TOOLS_H - 6;
    const h = Math.max(150, root.clientHeight - toolsH - PROMPT_H - GAP * 2 - ROOT_PAD);
    listEl.style.height = h + "px";
  };
  const listRo = new ResizeObserver(fitListHeight);
  listRo.observe(root);

  // 内容高度测量：工具条实测 + 提示词框 + 列表固定高 + padding/gap
  const measureContentHeight = () => {
    const toolsH = tools.offsetHeight || TOOLS_H - 6;
    return ROOT_PAD + toolsH + GAP + PROMPT_H + GAP + LIST_H;
  };

  const widget = node.addDOMWidget(lib.DOM_WIDGET, lib.DOM_WIDGET, root, {
    serialize: false,
    getValue: () => null,
    setValue: () => {},
    getMinHeight: measureContentHeight,
    getMaxHeight: () => Math.max(measureContentHeight(), (node.size ? node.size[1] : WIDGET_H + 60) - 75),
    margin: 4,
  });
  applyAdaptiveCanvasOnly(widget);

  if (typeof node.setSize === "function") node.setSize([420, WIDGET_H + 60]);
  else { node.size[0] = 420; node.size[1] = WIDGET_H + 60; }

  // library 库切换 → 清空选择+草稿，重新拉取并渲染
  const libW = (node.widgets || []).find((x) => x.name === "library");
  if (libW) {
    const origCb = libW.callback;
    libW.callback = function () {
      const r = origCb?.apply(this, arguments);
      writeSelection(node, "");
      writeDraft(node, "");
      ctx.styles = null; // 强制重拉
      ctx.name = null;
      renderPrompt(ctx);
      ensureLoaded(ctx);
      return r;
    };
  }

  node._sfIDClothingCtx = ctx;
  ensureLoaded(ctx);
  renderPrompt(ctx);
}

app.registerExtension({
  name: "sfnodes.IDClothingSelector",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origCreated?.apply(this, arguments);
      setupNode(this);
      return r;
    };

    // 工作流加载：widget 值恢复发生在 configure，恢复后重渲染选中态
    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = origConfigure?.apply(this, arguments);
      const ctx = this._sfIDClothingCtx;
      if (ctx) ensureLoaded(ctx);
      return r;
    };

    // 自愈最小尺寸。只抬升过小的尺寸，已保存（>= min）的尺寸永不变更 -> 不脏加载
    const origResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      // LEGACY ONLY：Nodes 2.0 的渲染尺寸在 Vue 布局 store 里而非 node.size
      if (!isVueNodes()) {
        if (size[0] < MIN_W) size[0] = MIN_W;
        if (size[1] < WIDGET_H) size[1] = WIDGET_H;
      }
      return origResize?.apply(this, arguments);
    };
  },
});
