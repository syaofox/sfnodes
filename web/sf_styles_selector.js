// SFStylesSelector 前端（复刻 ComfyUI-Easy-Use easy stylesSelector）
// - 隐藏 SFStylesState STRING widget 为值真源（Python "hidden" 声明，标准 widget
//   收集进 prompt，随 workflow 保存）；DOM widget 纯交互不承担值传输（规避
//   Vue DOMWidget value setter 链，见 experience/nodes-image.md §11）
// - 标签多选列表：搜索过滤 / Reset 清空 / 选中置顶 / 悬停缩略图预览（预览图
//   挂在 widget 内部，修复原版全局 id 的多节点冲突）
// - Grid/List 显示模式切换（对齐 v2 的 stylesSelectorDisplay 设置语义；本实现
//   存 node.properties.sfStylesView 随 workflow 保存而非全局设置）
// - 纯逻辑（解析/排序/过滤/语言化）在 sf_styles_selector_lib.js

import { app } from "/scripts/app.js";
import { applyAdaptiveCanvasOnly, injectCSSOnce, installWheelZoomPassthrough, isGraphLoading, isVueNodes, sfApiUrl } from "./sf_common.js";
import * as lib from "./sf_styles_selector_lib.js";

const NODE_TYPE = "SFStylesSelector";
const LIST_H = 300; // 列表固定高度（滚动容器），widget 声称高度 = 头部 + 列表
const TOOLS_H = 34; // 工具条 + 与列表的 gap
const ROOT_PAD = 8; // root padding 上下各 4
// 声称高度必须 ≥ 内容实际高度（ROOT_PAD + TOOLS_H + LIST_H）——低于内容时
// 节点边框按声称高度绘制、底部内容溢出被裁（网格卡片缩略图显示不全，§18.6）
const WIDGET_H = LIST_H + TOOLS_H + ROOT_PAD + 4;
const MIN_W = 260;
const VIEW_PROP = "sfStylesView"; // Grid/List 显示模式（随 workflow 保存，不注入 prompt）
const GRID_COLS_SETTING = "sfnodes.StylesSelector.GridColumns"; // 网格固定列数（全局设置）
const EMPTY_IMG =
  "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";

// 网格固定列数：设置 combo 值 Auto/4/5/6/8/10/12 → 0 = Auto（CSS 自适应）
function gridCols() {
  try {
    const v = app.ui.settings.getSettingValue(GRID_COLS_SETTING);
    const n = parseInt(v, 10);
    return Number.isFinite(n) && n >= 2 ? n : 0;
  } catch {
    return 0;
  }
}

// 设置变化/异步加载完成后：重渲染全图 SFStylesSelector 节点（网格列数变化）
function refreshAll() {
  for (const n of app.graph?._nodes || []) {
    if (n.type === NODE_TYPE && n._sfStylesCtx?.reRender) n._sfStylesCtx.reRender();
  }
}

function injectCSS() {
  injectCSSOnce("sf-styles-selector-css", `
.sf-ss-root{display:flex;flex-direction:column;gap:6px;height:100%;box-sizing:border-box;padding:4px 6px;position:relative;overflow:visible;}
.sf-ss-tools{display:flex;gap:6px;flex:0 0 auto;}
.sf-ss-search{flex:1;min-width:0;resize:none;font:12px sans-serif;color:#ddd;background:#1d1d1d;border:1px solid #333;border-radius:5px;padding:4px 6px;height:28px;box-sizing:border-box;outline:none;}
.sf-ss-reset{flex:0 0 auto;font:11px sans-serif;color:var(--sf-acc, #f66744);background:color-mix(in srgb, var(--sf-acc, #f66744) 12%, transparent);border:1px solid color-mix(in srgb, var(--sf-acc, #f66744) 45%, transparent);border-radius:5px;padding:0 10px;cursor:pointer;height:28px;}
.sf-ss-reset:hover{background:color-mix(in srgb, var(--sf-acc, #f66744) 22%, transparent);}
.sf-ss-viewseg{flex:0 0 auto;display:flex;border:1px solid #444;border-radius:5px;overflow:hidden;height:28px;}
.sf-ss-viewseg button{font:11px sans-serif;color:#c8c8c8;background:#2a2a2a;border:none;padding:0 8px;cursor:pointer;}
.sf-ss-viewseg button:hover{background:#3a3a3a;}
.sf-ss-viewseg button.sf-ss-viewon{background:color-mix(in srgb, var(--sf-acc, #f66744) 25%, transparent);color:#fff;}
/* 列表区域：min-height 兜底 + 自身滚动（裁剪只发生在列表容器，root 不裁）*/
.sf-ss-list{flex:0 0 auto;min-height:150px;height:calc(100% - 12px);overflow-y:auto;overflow-x:hidden;display:flex;flex-direction:column;gap:2px;padding:2px;box-sizing:border-box;}
/* 网格覆盖用复合选择器 + !important 加固（防止被 flex 规则/注入顺序压掉，
   卡片被 flex-shrink 压缩成细条）。grid-auto-rows:max-content 是关键：
   grid 容器高度确定时 auto 行会收缩到 min-content（flex 卡片的 min-content
   高度 ≈ 0 → 行高 10px、缩略图溢出被裁成细条），max-content 强制按内容撑开 */
.sf-ss-list.sf-ss-grid{display:grid !important;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));grid-auto-rows:max-content;gap:8px;align-content:start;padding:4px 0;}
.sf-ss-card{display:flex;flex-direction:column;gap:3px;padding:4px;border-radius:6px;cursor:pointer;background:#222;border:1px solid #3a3a3a;overflow:hidden;flex:0 0 auto;}
/* 完整展示缩略图（不裁切）：contain 整图可见；aspect-ratio 1/1（实测
   672 张缩略图全部为正方形）让图区高度随列宽联动且零留白——列少卡片宽
   则图高、列多卡片窄则图矮。行高由 grid-auto-rows:max-content 按内容
   撑开（见 .sf-ss-list.sf-ss-grid 注释） */
.sf-ss-card img{width:100%;height:auto;aspect-ratio:1/1;object-fit:contain;border-radius:4px;background:#1a1a1a;flex:0 0 auto;display:block;}
.sf-ss-card span{font:11px sans-serif;color:#ccc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center;user-select:none;}
.sf-ss-card:hover{border-color:#555;}
.sf-ss-cardsel{border-color:var(--sf-acc, #f66744);background:color-mix(in srgb, var(--sf-acc, #f66744) 12%, transparent);}
.sf-ss-cardsel span{color:#fff;}
.sf-ss-tag{display:flex;align-items:center;gap:6px;padding:3px 6px;border-radius:4px;cursor:pointer;font:12px sans-serif;color:#ccc;user-select:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:0 0 auto;}
.sf-ss-tag input{flex:0 0 auto;accent-color:var(--sf-acc, #f66744);pointer-events:none;}
.sf-ss-tag span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.sf-ss-tag:hover{background:#2c2c2c;}
.sf-ss-sel{background:color-mix(in srgb, var(--sf-acc, #f66744) 16%, transparent);color:#fff;}
.sf-ss-hide{display:none;}
/* hover 信息浮窗（对齐 Easy-Use v2 previewer：缩略图 + 名称 + 正/负提示词）*/
.sf-ss-pop{position:absolute;display:none;pointer-events:none;width:210px;border-radius:8px;border:1px solid #4a4a4a;background:#202020;box-shadow:0 8px 22px rgba(0,0,0,.65);z-index:10;padding:6px;box-sizing:border-box;}
.sf-ss-popimg{width:100%;height:118px;object-fit:contain;background:#131313;border-radius:6px;display:block;}
.sf-ss-popname{display:block;font:12px sans-serif;color:#fff;margin:5px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sf-ss-poppos,.sf-ss-popneg{font:10px/1.4 sans-serif;margin:2px 0;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;word-wrap:break-word;}
.sf-ss-poppos b{color:#7bd88f;}
.sf-ss-poppos span{color:#9ecfa8;}
.sf-ss-popneg b{color:#e8928a;}
.sf-ss-popneg span{color:#cfa39e;}`);
}

// 样式库列表缓存（promise 级，避免加载期重复请求；失败缓存空列表会话内不重试）
const stylesListCache = {};
function fetchStylesList(name) {
  if (!stylesListCache[name]) {
    stylesListCache[name] = (async () => {
      try {
        const resp = await fetch(sfApiUrl(`${lib.STYLES_API}?name=${encodeURIComponent(name)}`));
        if (resp.ok) {
          const data = await resp.json();
          return Array.isArray(data) ? data : [];
        }
      } catch (e) {
        console.warn("sfnodes.StylesSelector: 拉取样式库失败", name, e);
      }
      return [];
    })();
  }
  return stylesListCache[name];
}

function isZh() {
  return (navigator.language || "en").toLowerCase().startsWith("zh");
}

function stateWidget(node) {
  return (node.widgets || []).find((w) => w.name === lib.STATE_WIDGET) || null;
}

function readState(node) {
  const w = stateWidget(node);
  return lib.parseState(w ? w.value : "");
}

function writeState(node, names) {
  const w = stateWidget(node);
  if (w) w.value = lib.serializeState(names);
}

function currentLibraryName(node) {
  const w = (node.widgets || []).find((x) => x.name === "styles");
  return w && w.value ? String(w.value) : "fooocus_styles";
}

// Grid/List 显示模式：存 node.properties 随 workflow 保存（对齐 v2 的
// stylesSelectorDisplay 设置语义，但不做全局设置——显示偏好按工作流区分）
function viewMode(node) {
  const v = node.properties && node.properties[VIEW_PROP];
  return v === "list" ? "list" : "grid";
}

function setViewMode(node, mode) {
  if (!node.properties) node.properties = {};
  node.properties[VIEW_PROP] = mode;
}

function toggleSelect(node, name) {
  if (isGraphLoading()) return; // 加载/尾窗：值尚未恢复，忽略点击
  const names = readState(node);
  const i = names.indexOf(name);
  if (i >= 0) names.splice(i, 1);
  else names.push(name);
  writeState(node, names);
}

// hover 信息浮窗（对齐 Easy-Use v2 previewer：缩略图 + 名称 + 正/负提示词）
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
  const img = popEl.querySelector(".sf-ss-popimg");
  if (img.dataset.src !== thumb) {
    img.dataset.src = thumb;
    img.src = thumb ? (lib.isRemoteThumb(thumb) ? thumb : sfApiUrl(thumb)) : EMPTY_IMG;
  }
  popEl.querySelector(".sf-ss-popname").textContent = item.label;
  const pos = popEl.querySelector(".sf-ss-poppos");
  const neg = popEl.querySelector(".sf-ss-popneg");
  pos.style.display = raw.prompt ? "" : "none";
  if (raw.prompt) pos.querySelector("span").textContent = raw.prompt;
  neg.style.display = raw.negative_prompt ? "" : "none";
  if (raw.negative_prompt) neg.querySelector("span").textContent = raw.negative_prompt;
  placePop(popEl, root, e);
  popEl.style.display = "block";
}

function hidePop(ctx) {
  ctx.popEl.style.display = "none";
}

function makeTag(item, ctx) {
  const { node } = ctx;
  const label = document.createElement("label");
  label.className = "sf-ss-tag" + (item.selected ? " sf-ss-sel" : "") + (item.hidden ? " sf-ss-hide" : "");
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
    toggleSelect(node, item.name);
    renderList(ctx);
  };

  label.onmouseenter = (e) => showPop(ctx, item, e);
  label.onmousemove = (e) => {
    if (ctx.popEl.style.display !== "none") placePop(ctx.popEl, ctx.root, e);
  };
  label.onmouseleave = () => hidePop(ctx);
  return label;
}

// Grid 视图卡片：缩略图 + 名字（loading="lazy" 避免 275 张远程图一次性拉取）
function makeCard(item, ctx) {
  const { node } = ctx;
  const card = document.createElement("div");
  card.className = "sf-ss-card" + (item.selected ? " sf-ss-cardsel" : "") + (item.hidden ? " sf-ss-hide" : "");
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
    toggleSelect(node, item.name);
    renderList(ctx);
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
  listEl.className = "sf-ss-list" + (viewMode(node) === "grid" ? " sf-ss-grid" : "");
  // 网格固定列数（全局设置）：内联 repeat(N, 1fr) 覆盖 CSS 自适应；
  // Auto（0）时清空内联回落到 auto-fill minmax(80px, 1fr)
  const cols = gridCols();
  listEl.style.gridTemplateColumns = cols > 0 ? `repeat(${cols}, 1fr)` : "";
  const names = readState(node);
  const items = lib.filterAndSort(styles, searchEl.value, names, isZh());
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
    return;
  }
  if (ctx.pending) {
    ctx.pending.then(() => {
      if (currentLibraryName(ctx.node) !== ctx.name) ensureLoaded(ctx); // 加载期间库被切换：重新加载
      else renderList(ctx);
    });
    return;
  }
  ctx.pending = fetchStylesList(name).then((data) => {
    ctx.pending = null;
    if (currentLibraryName(ctx.node) !== name) return; // 竞态：期间已切换库
    ctx.name = name;
    ctx.styles = data;
    renderList(ctx);
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
  }

  const root = document.createElement("div");
  root.className = "sf-ss-root";

  const tools = document.createElement("div");
  tools.className = "sf-ss-tools";
  const resetBtn = document.createElement("button");
  resetBtn.className = "sf-ss-reset";
  resetBtn.textContent = isZh() ? "重置" : "Reset";
  resetBtn.title = isZh() ? "清空所有已选样式" : "Reset all selected styles";
  resetBtn.onclick = () => {
    writeState(node, []);
    renderList(ctx);
  };
  const searchEl = document.createElement("textarea");
  searchEl.className = "sf-ss-search";
  searchEl.rows = 1;
  searchEl.placeholder = isZh() ? "🔎 搜索样式 ..." : "🔎 Search styles ...";
  searchEl.oninput = () => renderList(ctx);
  installWheelZoomPassthrough(searchEl);

  // Grid/List 显示模式切换（对齐 v2 stylesSelectorDisplay）
  const zh = isZh();
  const viewSeg = document.createElement("div");
  viewSeg.className = "sf-ss-viewseg";
  // 统一遍历同步全部按钮：单按钮闭包 sync 只更新自身，切换后另一按钮的
  // 高亮类残留 → 双高亮（"切换后不恢复不高亮"）
  const syncViewBtns = () => {
    const cur = viewMode(node);
    for (const btn of viewSeg.children) {
      btn.classList.toggle("sf-ss-viewon", btn.dataset.mode === cur);
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

  const listEl = document.createElement("div");
  listEl.className = "sf-ss-list";

  // hover 信息浮窗（图 + 名称 + 正/负提示词，对齐原节点 previewer）
  const popEl = document.createElement("div");
  popEl.className = "sf-ss-pop";
  const popImg = document.createElement("img");
  popImg.className = "sf-ss-popimg";
  popImg.src = EMPTY_IMG;
  popImg.onerror = () => {
    popImg.src = EMPTY_IMG;
  };
  const popName = document.createElement("span");
  popName.className = "sf-ss-popname";
  const popPos = document.createElement("div");
  popPos.className = "sf-ss-poppos";
  const popPosB = document.createElement("b");
  popPosB.textContent = "Positive: ";
  const popPosT = document.createElement("span");
  popPos.append(popPosB, popPosT);
  const popNeg = document.createElement("div");
  popNeg.className = "sf-ss-popneg";
  const popNegB = document.createElement("b");
  popNegB.textContent = "Negative: ";
  const popNegT = document.createElement("span");
  popNeg.append(popNegB, popNegT);
  popEl.append(popImg, popName, popPos, popNeg);

  root.append(tools, listEl, popEl);

  // 列表高度显式管理：不用 CSS 百分比（父容器高度不确定时 calc 失效），
  // 观察 root 实际高度后内联设置，min 150px 兜底（对齐原节点 min-height）。
  const fitListHeight = () => {
    if (!root.clientHeight) return;
    const toolsH = tools.offsetHeight || TOOLS_H - 6;
    const h = Math.max(150, root.clientHeight - toolsH - 6 - ROOT_PAD);
    listEl.style.height = h + "px";
  };
  const listRo = new ResizeObserver(fitListHeight);
  listRo.observe(root);

  const ctx = {
    node,
    root,
    listEl,
    searchEl,
    popEl,
    styles: null,
    name: null,
    pending: null,
    reRender: () => renderList(ctx),
  };

  // 内容高度测量（对齐 sf_load_image.js 的 measureH 模式）：工具条实测 +
  // 列表固定高 + root padding/gap。列表是固定高度滚动容器，内容高度恒定。
  // 不能读 root.offsetHeight（LiteGraph 拉伸后读数会形成反馈环）。
  const measureContentHeight = () => {
    const toolsH = tools.offsetHeight || TOOLS_H - 6;
    return ROOT_PAD + toolsH + 6 + LIST_H;
  };

  const widget = node.addDOMWidget(lib.DOM_WIDGET, lib.DOM_WIDGET, root, {
    serialize: false,
    getValue: () => null,
    setValue: () => {},
    // 对齐 Easy-Use v2 原节点：minHeight = 内容需求；maxHeight 动态跟随
    // 节点高度（节点拉高 → 列表区域变高 → 网格显示更多行）。Vue DOMWidget
    // 的默认 computeLayoutSize 读这两个 options，勿手动覆盖（覆盖会丢失
    // 动态 maxHeight）。
    getMinHeight: measureContentHeight,
    getMaxHeight: () => Math.max(measureContentHeight(), (node.size ? node.size[1] : WIDGET_H + 60) - 75),
    margin: 4,
  });
  applyAdaptiveCanvasOnly(widget);

  if (typeof node.setSize === "function") node.setSize([420, WIDGET_H + 60]);
  else { node.size[0] = 420; node.size[1] = WIDGET_H + 60; }

  // styles 库切换 → 重新拉取并渲染
  const stylesW = (node.widgets || []).find((x) => x.name === "styles");
  if (stylesW) {
    const origCb = stylesW.callback;
    stylesW.callback = function () {
      const r = origCb?.apply(this, arguments);
      ctx.styles = null; // 强制重拉
      ctx.name = null;
      ensureLoaded(ctx);
      return r;
    };
  }

  node._sfStylesCtx = ctx;
  ensureLoaded(ctx);
}

app.registerExtension({
  name: "sfnodes.StylesSelector",

  // 网格固定列数设置（sfnodes.StylesSelector.GridColumns）：ComfyUI 设置
  // 面板修改后重渲染全图节点。注册写法对齐 SFLoraStack 的 accent 设置
  // （onChange 在 store 更新前触发 → setTimeout(0)；设置值异步加载晚于
  // init → 轮询重试补刷新）。
  init() {
    try {
      app.ui.settings.addSetting({
        id: GRID_COLS_SETTING,
        name: "SF Styles Selector: grid columns",
        tooltip: "Grid 视图固定列数（少列=大缩略图，多列=小缩略图）；Auto 按节点宽度自适应",
        defaultValue: "Auto",
        type: "combo",
        options: ["Auto", "4", "5", "6", "8", "10", "12"],
        onChange: () => setTimeout(refreshAll, 0),
      });
      let tries = 12;
      const retry = () => {
        if (tries-- <= 0) return;
        setTimeout(() => {
          refreshAll();
          retry();
        }, 500);
      };
      retry();
    } catch (_e) { /* 设置系统不可用则保持 Auto */ }
  },

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
      const ctx = this._sfStylesCtx;
      if (ctx) ensureLoaded(ctx);
      return r;
    };

    // 自愈最小尺寸（与 getMinHeight 双保险，对齐 sf_prompt_list）。只抬升
    // 过小的尺寸，已保存（>= min）的尺寸永不变更 -> 不脏加载
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
