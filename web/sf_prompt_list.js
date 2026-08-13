// ==========================================================================
// sf_prompt_list.js - SFPromptList 节点体：带行号的多行文本编辑器
// ==========================================================================
//
// 隐藏原生 multiline_text widget（值真源：graphToPrompt 只收集 widget.value，
// DOM widget 的 getValue 返回 null 不参与），替换为 DOM widget 编辑器：
// 左侧行号栏（从 0 开始）+ textarea，编辑时写回原生 widget.value。
//
// 行号 = 后端过滤后的输出 index：skip_empty 开启时空白行（trim 后为空）
// 跳过不占号（空行位置显示 · 占位符），关闭时按逻辑行编号。长行软换行时
// 行号与视觉行精确对齐：逐行镜像测量视觉高度（行高缓存 + 宽度变化失效），
// 仅 > MAX_FULL_LINES 的虚拟化模式保留固定行高近似（scrollTop 同步）。
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
  const ta = document.createElement("textarea");
  ta.className = "sf-pl-ta";
  ta.spellcheck = false;
  editor.append(gutter, ta);

  root.append(hdr, editor);

  const lineCount = () => ta.value.split("\n").length;

  // skip_empty 开关实时读取（找不到 widget 时默认 True，与后端默认一致）
  const skipEmptyOn = () => {
    for (const w of node.widgets || []) {
      if (w && w.name === "skip_empty") return !!w.value;
    }
    return true;
  };

  // ── 行高测量（软换行精确对齐）──
  // textarea 长行自动软换行时视觉行数大于逻辑行数，gutter 若按固定 LINE_H
  // 渲染会与视觉行逐行错位。每个逻辑行的视觉高度只取决于该行文本与容器
  // 宽度（pre-wrap 换行无上下文依赖）→ 按行缓存高度，编辑只重测变化的行；
  // 节点宽度变化（换行重新分布）时清空缓存。
  const hCache = new Map();
  let measContainer = null;
  let measWidth = 0;
  const contentWidth = () => {
    const w = ta.clientWidth - 16; // padding 8×2
    return Number.isFinite(w) && w > 0 ? w : 0;
  };
  // 批量测量缓存未命中行：一次性建镜像节点 + 批量读高度（防 layout thrash）。
  // 空行/纯空白行固定单行；needsMeasure 判定必不换行的行直接 LINE_H 跳过测量。
  function measureHeights(rows) {
    const cw = contentWidth();
    if (cw <= 0) return; // 未布局：保持单行兜底（ResizeObserver 布局后修正）
    if (measWidth !== cw) {
      hCache.clear();
      measWidth = cw;
    }
    if (!measContainer) {
      measContainer = document.createElement("div");
      measContainer.style.cssText =
        "position:absolute;visibility:hidden;pointer-events:none;left:-9999px;top:0;" +
        "font:12px monospace;line-height:1.4;white-space:pre-wrap;overflow-wrap:break-word;";
      root.appendChild(measContainer);
    }
    measContainer.style.width = cw + "px";
    measContainer.innerHTML = "";
    const pending = [];
    for (const t of rows) {
      if (hCache.has(t)) continue;
      // 空白行（trim 后为空）通常固定单行——但超长纯空白行在 pre-wrap 下
      // 同样会软换行，长度判定（needsMeasure）前不能仅凭 trim 跳过
      if (needsMeasure(t, cw)) {
        const d = document.createElement("div");
        d.textContent = t;
        measContainer.appendChild(d);
        pending.push([t, d]);
      } else if (t.trim()) {
        hCache.set(t, LINE_H);
      }
    }
    for (const [t, d] of pending) {
      const h = d.getBoundingClientRect().height;
      hCache.set(t, h >= LINE_H ? h : LINE_H);
    }
    measContainer.innerHTML = "";
  }
  const lineH = (t) => hCache.get(t) ?? LINE_H;

  // 行号 = 后端过滤后的输出 index：skip_empty 开启时空白行（trim 后为空）
  // 跳过不占号，空行位置渲染 · 占位符；关闭时按逻辑行编号
  function renderGutter() {
    const rows = ta.value.split("\n");
    const skip = skipEmptyOn();
    let valid = 0;
    for (const r of rows) {
      if (!(skip && !r.trim())) valid += 1;
    }
    count.textContent = `${valid}/${rows.length} line${rows.length === 1 ? "" : "s"}`;
    const digits = Math.max(2, String(Math.max(0, valid - 1)).length);
    gutter.style.width = `calc(${digits}ch + 16px)`;
    if (rows.length <= MAX_FULL_LINES) {
      gutter.style.paddingTop = "";
      gutter.style.paddingBottom = "";
      measureHeights(rows);
      const frag = document.createDocumentFragment();
      let idx = 0;
      for (const r of rows) {
        const s = document.createElement("span");
        s.className = "sf-pl-gn";
        if (skip && !r.trim()) {
          s.classList.add("sf-pl-gap");
          s.textContent = "\u00B7";
        } else {
          s.textContent = String(idx++);
        }
        s.style.height = lineH(r) + "px";
        frag.appendChild(s);
      }
      gutter.replaceChildren(frag);
    } else {
      const first = Math.max(0, Math.floor(ta.scrollTop / LINE_H));
      const visible = Math.max(1, Math.ceil(gutter.clientHeight / LINE_H) + 2);
      const last = Math.min(rows.length, first + visible);
      gutter.style.paddingTop = `${first * LINE_H}px`;
      gutter.style.paddingBottom = `${Math.max(0, rows.length - last) * LINE_H}px`;
      const frag = document.createDocumentFragment();
      let idx = 0;
      for (let i = 0; i < first; i++) {
        if (!(skip && !rows[i].trim())) idx += 1;
      }
      for (let i = first; i < last; i++) {
        const s = document.createElement("span");
        s.className = "sf-pl-gn";
        if (skip && !rows[i].trim()) {
          s.classList.add("sf-pl-gap");
          s.textContent = "\u00B7";
        } else {
          s.textContent = String(idx++);
        }
        frag.appendChild(s);
      }
      gutter.replaceChildren(frag);
    }
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

  // 滚动同步：gutter 为 overflow:hidden，scrollTop 仍可程序化设置（近似对齐）。
  // 虚拟化模式下重渲染窗口（防抖）
  ta.addEventListener("scroll", () => {
    gutter.scrollTop = ta.scrollTop;
    if (lineCount() > MAX_FULL_LINES) scheduleRender();
  });

  // 节点宽度变化 → 换行重新分布：清行高缓存并重渲染（软换行对齐跟随）。
  // 首次布局（nodeCreated 时 clientWidth=0）也由这里修正为精确高度。
  // 走 80ms 防抖：拖拽拉伸连续触发 RO 时合并渲染，避免每帧全量测量。
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(() => {
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
    renderGutter();
  }

  root._sfPlSync = syncFromWidget;
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

  // skip_empty 开关变化 → 重渲染行号（空行占号/跳号切换）；
  // configure 恢复已由 onConfigure → _sfPlSync 覆盖
  for (const w of node.widgets || []) {
    if (w && w.name === "skip_empty") {
      const origSkipCb = w.callback;
      w.callback = function () {
        const r = origSkipCb?.apply(this, arguments);
        node._sfPromptListRoot?._sfPlSync();
        return r;
      };
      break;
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
      this._sfPromptListRoot = null;
      return origRemoved?.apply(this, arguments);
    };
  },
});
