// ==========================================================================
// sf_prompt_list.js - SFPromptList 节点体：带行号的多行文本编辑器
// ==========================================================================
//
// 隐藏原生 multiline_text widget（值真源：graphToPrompt 只收集 widget.value，
// DOM widget 的 getValue 返回 null 不参与），替换为 DOM widget 编辑器：
// 左侧行号栏（从 0 开始）+ textarea，编辑时写回原生 widget.value。
//
// 行号 = 后端过滤后的输出 index：skip_empty 开启时空白行（trim 后为空）
// 跳过不占号（空行位置显示 · 占位符），关闭时按逻辑行编号。wrap 开启时
// 长行软换行走行高镜像测量对齐（mirror 与 textarea 同几何块级 div，行高
// 按行文本缓存、宽度变化清空）；渲染后强制重同步 gutter/hl scrollTop
// （resize/删文本后浏览器钳制 textarea.scrollTop 不触发 scroll 事件，
// 不同步则行号错位）。
// start_index/max_rows 切片范围高亮：仅当切片实际裁剪（start>0 或 max_rows
// 非默认值/截断）时，选中行文本区叠加半透明强调色背景块（hl 层 absolute
// 全局坐标 + scrollTop 同步裁切）+ 行号变强调色联动；wrap 开启时高亮随
// 镜像测量行高展开（与行号同源，scrollTop 重同步后两者一致对齐）。
// 行数超过 MAX_FULL_LINES 时切换可视区虚拟渲染（padding 占位），防极端
// 行数卡顿。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { applyAdaptiveCanvasOnly, isVueNodes } from "./sf_common.js";

const CLASS = "SFPromptList";
const WIDGET_TYPE = "sf_prompt_list_editor";

// 固定垂直预算（textarea 吸收节点拉伸，按最小值计入防 paint 膨胀，
// sf_prompt_reader 同款模式）
const PAD = 6;
const HDR_H = 24;
const EDITOR_MIN_H = 140;
const CORE_H = PAD + HDR_H + PAD + EDITOR_MIN_H + PAD;
const MIN_W = 340;

// 虚拟化阈值与行高——LINE_H 必须与 CSS 的 font:12px monospace line-height:1.4 一致
const MAX_FULL_LINES = 500;
const LINE_H = 12 * 1.4;

// 该行是否必须镜像测量：含 tab（等宽字体下宽度不可估）或
// 字符数 × 12px（等宽字体最大字符宽，CJK 全角）超过容器宽度
function needsMeasure(text, cw) {
  return text.includes("\t") || text.length * 12 > cw;
}

function injectCSS() {
  if (document.getElementById("sf-pl-css")) return;
  const s = document.createElement("style");
  s.id = "sf-pl-css";
  s.textContent = `
.sf-pl-root { position:relative; display:flex; flex-direction:column; flex:1 1 0;
  min-height:0; box-sizing:border-box; padding:${PAD}px; gap:${PAD}px;
  font:12px sans-serif; color:#ddd; overflow:hidden; background:transparent; }
.sf-pl-hdr { flex:0 0 auto; display:flex; align-items:center; gap:6px;
  padding:3px 6px 3px 9px; border:1px solid #333; border-radius:5px;
  background:rgba(255,255,255,0.02); }
.sf-pl-hlbl { font:10px 'Segoe UI',-apple-system,sans-serif; color:#8f8f8f; flex:1 1 0;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-pl-count { flex:0 0 auto; font-size:10px; color:#8f8f8f;
  white-space:nowrap; user-select:none; }
.sf-pl-editor { flex:1 1 0; min-height:0; display:flex;
  background:#1d1d1d; border:1px solid #333; border-radius:5px; overflow:hidden; }
.sf-pl-editor:focus-within { border-color:${"var(--sf-acc, #f66744)"}; }
.sf-pl-gutter { flex:0 0 auto; overflow:hidden; box-sizing:border-box;
  padding:6px 0 6px 8px; background:rgba(0,0,0,0.25);
  border-right:1px solid #2c2c2c; user-select:none; }
.sf-pl-gn { display:block; text-align:right; padding-right:8px;
  font:12px monospace; line-height:1.4; color:#777; white-space:pre; }
.sf-pl-gn.sf-pl-gap { color:#555; font-style:italic; }
.sf-pl-gn.sf-pl-on { color:${"var(--sf-acc, #f66744)"}; font-weight:bold; }
.sf-pl-tawrap { flex:1 1 0; min-height:0; position:relative; display:flex; }
.sf-pl-hl { position:absolute; inset:0; overflow:hidden; pointer-events:none; }
.sf-pl-hl-row { position:absolute; left:0; right:0; height:16.8px;
  background:${"var(--sf-acc, #f66744)"}; opacity:0.16; }
.sf-pl-ta { flex:1 1 0; min-height:0; width:100%; box-sizing:border-box;
  background:transparent; color:#e0e0e0; border:0; outline:none; resize:none;
  font:12px monospace; line-height:1.4; padding:6px 8px; }
.sf-pl-ta::placeholder { color:#5c5c5c; font-style:italic; }
`;
  document.head.appendChild(s);
}

// 隐藏原生 multiline_text widget（保留其 .value 作值真源）。
// hideNativeImageCombo 三件套：hidden + computeSize 归零 + element none，
// rAF 补刀覆盖 Vue 延迟 DOM 渲染。只针对 multiline_text，其余 widget 保留。
function hideNativeMultiline(node) {
  let target = null;
  for (const w of node.widgets || []) {
    if (w && w.name === "multiline_text") target = w;
  }
  if (!target) return null;
  target.hidden = true;
  target.computeSize = () => [0, -4];
  if (!target.options) target.options = {};
  target.options.canvasOnly = true;
  if (target.element) target.element.style.display = "none";
  requestAnimationFrame(() => {
    for (const w of node.widgets || []) {
      if (!w || w.name !== "multiline_text") continue;
      const el = w.element || w.inputEl;
      if (el) el.style.display = "none";
    }
  });
  return target;
}

function buildEditor(node, textWidget) {
  injectCSS();
  const root = document.createElement("div");
  root.className = "sf-pl-root";

  const hdr = document.createElement("div");
  hdr.className = "sf-pl-hdr";
  const hlbl = document.createElement("span");
  hlbl.className = "sf-pl-hlbl";
  hlbl.textContent = "multiline_text";
  hlbl.title = "每行将作为列表的一项；行号从 0 开始（仅编辑辅助，不影响输出）";
  const count = document.createElement("span");
  count.className = "sf-pl-count";
  hdr.append(hlbl, count);

  const editor = document.createElement("div");
  editor.className = "sf-pl-editor";
  const gutter = document.createElement("div");
  gutter.className = "sf-pl-gutter";
  const taWrap = document.createElement("div");
  taWrap.className = "sf-pl-tawrap";
  const hl = document.createElement("div");
  hl.className = "sf-pl-hl";
  const ta = document.createElement("textarea");
  ta.className = "sf-pl-ta";
  ta.spellcheck = false;
  taWrap.append(hl, ta);
  editor.append(gutter, taWrap);

  root.append(hdr, editor);

  const lineCount = () => ta.value.split("\n").length;

  // INT widget 读取（非数字兜底默认值）
  const intOf = (name, dflt) => {
    for (const w of node.widgets || []) {
      if (w && w.name === name && typeof w.value === "number" && Number.isFinite(w.value)) {
        return Math.max(0, Math.floor(w.value));
      }
    }
    return dflt;
  };

  // skip_empty 开关实时读取（找不到 widget 时默认 True，与后端默认一致）
  const skipEmptyOn = () => {
    for (const w of node.widgets || []) {
      if (w && w.name === "skip_empty") return !!w.value;
    }
    return true;
  };

  // wrap_text 开关实时读取（默认 False，与后端默认一致）：关闭时
  // wrap="off" 水平滚动不软换行（行号恒单行高，精确对齐）；打开时软换行
  // 走行高测量对齐
  const wrapOn = () => {
    for (const w of node.widgets || []) {
      if (w && w.name === "wrap_text") return !!w.value;
    }
    return false;
  };

  // ── 行高测量（wrap 开启时软换行精确对齐）──
  // mirror 与 textarea 同几何（同宽同 padding 同字体同换行参数）：每逻辑行
  // 一个**块级 div**（inline span 的行盒高度取字体度量而非 line-height，
  // 实测系统性偏小——必须用块级），div 的 getBoundingClientRect().height =
  // 该逻辑行在 textarea 中的视觉高度（同宽同字体下断行与 textarea 一致，
  // 实测行数吻合）。按行文本缓存，编辑只重测变化的行；宽度变化（换行
  // 重新分布）时清空缓存。**不做 scrollHeight 总高校准**——textarea 的
  // scrollHeight 含底部预留行（单行/空内容虚高 1-2 行实测），校准会把正确
  // 测量改错。
  const hCache = new Map();
  let measContainer = null;
  let measWidth = 0;
  const contentWidth = () => {
    const w = ta.clientWidth - 16; // padding 8×2（clientWidth 已扣滚动条）
    return Number.isFinite(w) && w > 0 ? w : 0;
  };
  function measureHeights(rows) {
    const cw = contentWidth();
    if (!wrapOn() || cw <= 0) return; // 关闭换行无软换行；未布局保持单行兜底
    if (measWidth !== cw) {
      hCache.clear();
      measWidth = cw;
    }
    if (!measContainer) {
      measContainer = document.createElement("div");
      measContainer.style.cssText =
        "position:absolute;visibility:hidden;pointer-events:none;left:-9999px;top:0;" +
        "font:12px monospace;line-height:1.4;white-space:pre-wrap;overflow-wrap:break-word;" +
        "box-sizing:border-box;padding:6px 8px;";
      root.appendChild(measContainer);
    }
    measContainer.style.width = ta.clientWidth + "px"; // 与 textarea 同宽（border-box）
    measContainer.innerHTML = "";
    const pending = [];
    for (const t of rows) {
      if (hCache.has(t)) continue;
      // 空白行（trim 后为空）通常固定单行——但超长纯空白行在 pre-wrap 下
      // 同样会软换行，长度判定（needsMeasure）前不能仅凭 trim 跳过
      if (!needsMeasure(t, cw) && t.trim()) {
        hCache.set(t, LINE_H);
        continue;
      }
      const d = document.createElement("div"); // 块级：行盒高度 = line-height
      d.textContent = t;
      measContainer.appendChild(d);
      pending.push([t, d]);
    }
    for (const [t, d] of pending) {
      const h = d.getBoundingClientRect().height;
      hCache.set(t, h >= LINE_H ? h : LINE_H); // 空 div 高 0 → 兜底单行高
    }
    measContainer.innerHTML = "";
  }
  const lineH = (t) => (wrapOn() ? hCache.get(t) ?? LINE_H : LINE_H);

  // 行号 = 后端过滤后的输出 index：skip_empty 开启时空白行（trim 后为空）
  // 跳过不占号，空行位置渲染 · 占位符；关闭时按逻辑行编号。
  // 切片范围（start_index/max_rows）高亮：仅当切片实际裁剪（start>0 或
  // max_rows 非默认值/实际截断）时，选中行行号加 .sf-pl-on + 文本区叠加
  // 背景块（hl 层 absolute 全局坐标 + scrollTop 同步裁切；wrap 开启时随
  // 镜像测量行高展开，与行号同源对齐）
  function renderGutter() {
    const rows = ta.value.split("\n");
    const skip = skipEmptyOn();
    // 过滤后 index 映射：逻辑行 i → 输出 index（skip 时空行 -1）
    const idxOf = new Array(rows.length);
    let valid = 0;
    for (let i = 0; i < rows.length; i++) {
      idxOf[i] = skip && !rows[i].trim() ? -1 : valid++;
    }
    // 切片范围（与后端语义一致：start clamp 到有效行末、end 按 max_rows 截断）
    const startRaw = Math.max(0, intOf("start_index", 0));
    const start = Math.min(startRaw, Math.max(0, valid - 1));
    const maxRows = Math.max(1, intOf("max_rows", 1000));
    const end = Math.min(start + maxRows, valid);
    // 仅裁剪时高亮：start 非 0、max_rows 非默认 1000（显式设置即裁剪意图，
    // 即使恰好覆盖全部行也高亮）、或 max_rows 实际截断（如默认 1000 但
    // 行数超 1000）
    const clipped = startRaw > 0 || maxRows !== 1000 || end < valid;
    const selected = (i) => clipped && idxOf[i] >= start && idxOf[i] < end;

    count.textContent = `${valid}/${rows.length} line${rows.length === 1 ? "" : "s"}`;
    const digits = Math.max(2, String(Math.max(0, valid - 1)).length);
    gutter.style.width = `calc(${digits}ch + 16px)`;
    const frag = document.createDocumentFragment();
    const hlFrag = document.createDocumentFragment();
    // 高亮与行号同源（同一份镜像测量行高 + y 累计）：wrap 开时行号已精确
    // 对齐，高亮块随之对齐，无需再门控
    const hlOn = (i) => selected(i);
    if (rows.length <= MAX_FULL_LINES) {
      gutter.style.paddingTop = "";
      gutter.style.paddingBottom = "";
      measureHeights(rows);
      let y = 6; // textarea padding-top，与首行基线对齐
      for (let i = 0; i < rows.length; i++) {
        const h = lineH(rows[i]);
        const s = document.createElement("span");
        s.className = "sf-pl-gn";
        if (idxOf[i] < 0) {
          s.classList.add("sf-pl-gap");
          s.textContent = "\u00B7";
        } else {
          s.textContent = String(idxOf[i]);
          if (selected(i)) s.classList.add("sf-pl-on");
        }
        s.style.height = h + "px";
        if (hlOn(i)) {
          const b = document.createElement("div");
          b.className = "sf-pl-hl-row";
          b.style.top = y + "px";
          b.style.height = h + "px";
          hlFrag.appendChild(b);
        }
        y += h;
        frag.appendChild(s);
      }
      gutter.replaceChildren(frag);
    } else {
      const first = Math.max(0, Math.floor(ta.scrollTop / LINE_H));
      const visible = Math.max(1, Math.ceil(gutter.clientHeight / LINE_H) + 2);
      const last = Math.min(rows.length, first + visible);
      gutter.style.paddingTop = `${first * LINE_H}px`;
      gutter.style.paddingBottom = `${Math.max(0, rows.length - last) * LINE_H}px`;
      for (let i = first; i < last; i++) {
        const s = document.createElement("span");
        s.className = "sf-pl-gn";
        if (idxOf[i] < 0) {
          s.classList.add("sf-pl-gap");
          s.textContent = "\u00B7";
        } else {
          s.textContent = String(idxOf[i]);
          if (selected(i)) s.classList.add("sf-pl-on");
        }
        if (hlOn(i)) {
          const b = document.createElement("div");
          b.className = "sf-pl-hl-row";
          b.style.top = `${6 + i * LINE_H}px`;
          hlFrag.appendChild(b);
        }
        frag.appendChild(s);
      }
      gutter.replaceChildren(frag);
    }
    hl.replaceChildren(hlFrag);
    // 渲染改变 gutter/hl 内容高度 → 浏览器可能钳制其 scrollTop 与 ta 失步
    // （resize/删文本后 ta 的 scrollTop 被钳制也不触发 scroll 事件）→ 强制重同步
    gutter.scrollTop = ta.scrollTop;
    hl.scrollTop = ta.scrollTop;
  }

  let renderTimer = null;
  function scheduleRender() {
    clearTimeout(renderTimer);
    renderTimer = setTimeout(renderGutter, 80);
    node._sfPromptListRenderTimer = renderTimer;
  }

  // 编辑器 → 原生 widget（值真源）。短文本即时渲染行号，长文本防抖
  ta.addEventListener("input", () => {
    if (textWidget) textWidget.value = ta.value;
    if (lineCount() <= MAX_FULL_LINES) renderGutter();
    else scheduleRender();
    node.setDirtyCanvas?.(true, true);
  });

  // 滚动同步：gutter/hl 为 overflow:hidden，scrollTop 仍可程序化设置（近似对齐）。
  // 虚拟化模式下重渲染窗口（防抖）
  ta.addEventListener("scroll", () => {
    gutter.scrollTop = ta.scrollTop;
    hl.scrollTop = ta.scrollTop;
    if (lineCount() > MAX_FULL_LINES) scheduleRender();
  });

  // 节点宽度变化 → 换行重新分布：清行高缓存并重渲染（软换行对齐跟随）。
  // 首次布局（nodeCreated 时 clientWidth=0）也由这里修正为精确高度。
  // 同时强制重同步 gutter/hl scrollTop——resize 后浏览器钳制 ta.scrollTop
  // 不触发 scroll 事件，不同步则行号/高亮与文字错位。80ms 防抖合并拖拽。
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(() => {
      gutter.scrollTop = ta.scrollTop;
      hl.scrollTop = ta.scrollTop;
      const w = contentWidth();
      if (w !== measWidth) {
        hCache.clear();
        measWidth = w;
      }
      if (lineCount() <= MAX_FULL_LINES) scheduleRender();
    }).observe(ta);
  }

  // 事件防护：防 canvas 拖拽/取消选中/快捷键；Ctrl+Enter 放行 run-workflow
  ta.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") return;
    e.stopPropagation();
  });
  ta.addEventListener("pointerdown", (e) => e.stopPropagation());
  ta.addEventListener("mousedown", (e) => e.stopPropagation());

  // 原生 widget 值 → DOM（configure 恢复 / callback 外部设置时同步）
  function syncFromWidget() {
    const v = textWidget ? textWidget.value : "";
    if (ta.value !== v) ta.value = v;
    ta.wrap = wrapOn() ? "soft" : "off";
    renderGutter();
  }

  // ── 切片/开关 widget 值轮询兜底 ──
  // callback 在部分前端路径（Vue 数字输入组件等）不触发——任何前端更新
  // widget.value 的方式都经此兜底：400ms 轻量比较四值快照，变化才重渲染。
  // 快照在构建时初始化（代表初始渲染状态）；onWidgetChanged 改值后经
  // _sfPlUpdateWatch 同步快照，避免轮询误判"无变化"。与 callback/
  // onWidgetChanged 触发时合并（scheduleRender 防抖天然去重）。
  let watchVals = null;
  function updateWatch() {
    watchVals = [intOf("start_index", 0), intOf("max_rows", 1000), skipEmptyOn(), wrapOn()].join("|");
  }
  function checkWatch() {
    const prev = watchVals;
    updateWatch();
    if (prev !== null && prev !== watchVals) syncFromWidget();
  }
  updateWatch();
  root._sfPlCheckWatch = checkWatch;
  root._sfPlUpdateWatch = updateWatch;
  const watchTimer = setInterval(checkWatch, 400);
  node._sfPromptListWatchTimer = watchTimer;

  root._sfPlSync = syncFromWidget;
  root._sfPlSchedule = scheduleRender;
  return root;
}

function setupNode(node) {
  const textWidget = hideNativeMultiline(node);
  const root = buildEditor(node, textWidget);
  const widget = node.addDOMWidget(WIDGET_TYPE, WIDGET_TYPE, root, {
    serialize: false,
    getValue: () => null,
    setValue: () => {},
    getMinHeight: () => CORE_H,
    margin: 4,
  });
  applyAdaptiveCanvasOnly(widget);

  // 新节点默认尺寸。configure() 在 onNodeCreated 之后运行并恢复已保存尺寸，
  // 所以这只对全新节点生效
  if (typeof node.setSize === "function") node.setSize([420, 320]);
  else { node.size[0] = 420; node.size[1] = 320; }

  // 外部写原生 widget 值时同步 DOM（粘贴/其他插件设置路径）
  if (textWidget) {
    const origCb = textWidget.callback;
    textWidget.callback = function () {
      const r = origCb?.apply(this, arguments);
      node._sfPromptListRoot?._sfPlSync();
      return r;
    };
  }

  // 切片/开关 widget 变化 → 重渲染（行号跳号 / 换行模式 / 高亮范围切换）；
  // configure 恢复已由 onConfigure → _sfPlSync 覆盖
  for (const w of node.widgets || []) {
    if (w && (w.name === "skip_empty" || w.name === "wrap_text" || w.name === "start_index" || w.name === "max_rows")) {
      const origCb = w.callback;
      w.callback = function () {
        const r = origCb?.apply(this, arguments);
        node._sfPromptListRoot?._sfPlSync();
        return r;
      };
    }
  }

  root._sfPlSync();
  node._sfPromptListRoot = root;
}

app.registerExtension({
  name: "sfnodes.PromptList",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CLASS) return;

    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origCreated?.apply(this, arguments);
      setupNode(this);
    };

    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = origConfigure?.apply(this, arguments);
      this._sfPromptListRoot?._sfPlSync();
      return r;
    };

    // onWidgetChanged：LiteGraph/ComfyUI 的 widget 值变化节点级回调——
    // 数字输入等不触发 callback 的路径经此刷新（防抖；与轮询兜底重叠无害）。
    // 同时更新轮询快照，防止 checkWatch 把已生效的变化误判为"无变化"
    const origWidgetChanged = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function (widget, value, prevValue) {
      const r = origWidgetChanged?.apply(this, arguments);
      if (widget && this._sfPromptListRoot
          && (widget.name === "skip_empty" || widget.name === "wrap_text"
              || widget.name === "start_index" || widget.name === "max_rows")) {
        this._sfPromptListRoot._sfPlSchedule();
        this._sfPromptListRoot._sfPlUpdateWatch();
      }
      return r;
    };

    // 自愈最小尺寸（与 getMinHeight 双保险）。只抬升过小的尺寸，
    // 已保存（>= min）的尺寸永不变更 -> 不脏加载
    const origResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      // LEGACY ONLY：Nodes 2.0 的渲染尺寸在 Vue 布局 store 里而非 node.size
      if (!isVueNodes()) {
        if (size[0] < MIN_W) size[0] = MIN_W;
        if (size[1] < CORE_H) size[1] = CORE_H;
      }
      return origResize?.apply(this, arguments);
    };

    const origRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this._sfPromptListRenderTimer) clearTimeout(this._sfPromptListRenderTimer);
      if (this._sfPromptListWatchTimer) clearInterval(this._sfPromptListWatchTimer);
      this._sfPromptListRoot = null;
      return origRemoved?.apply(this, arguments);
    };
  },
});
