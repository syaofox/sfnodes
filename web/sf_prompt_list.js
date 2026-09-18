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
// hl 只含选中块、内容高远小于文本 → 用 ::before（--sf-pl-hl-pad 撑到
// textarea.scrollHeight）垫高滚动区，否则 scrollTop 被钳制、高亮滚出视口
// 后钉在视口边缘（幽灵高亮）。
// 行数超过 MAX_FULL_LINES 时切换可视区虚拟渲染（padding 占位，行窗口起点/
// padding 均计入 textarea 的 6px 顶部内边距保持同基线），防极端行数卡顿。
// 模式开关（头部 Edit/Select）：点选模式下 textarea readOnly，用原生文本
// 选择做行选择反馈（自带边缘自动滚动/Shift 与方向键扩选），selectionStart/
// End → 逻辑行区间 → selectionToRange 吸附空白行 → 写回 start_index/
// max_rows 原生 widget（现有回调包装链触发高亮重渲染）；行号栏支持单击选
// 单行/按住拖动选多行（window 捕获 pointermove + rAF 合并重渲染，span 带
// dataset.line）。模式状态存 node.properties.sfPromptListSelect（默认编辑）。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { applyAdaptiveCanvasOnly, injectCSSOnce, installWheelZoomPassthrough, isVueNodes } from "./sf_common.js";

const CLASS = "SFPromptList";
const WIDGET_TYPE = "sf_prompt_list_editor";

// 点选模式持久化键（node.properties，随工作流保存；缺省/删除 = 编辑模式）
const SELECT_KEY = "sfPromptListSelect";

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

// 点选模式：逻辑行区间 a..b（闭区间，允许乱序/越界）→ 输出索引切片
// {start, maxRows}。idxOf[i] = 逻辑行 i 的输出 index（-1 = skip_empty 过滤
// 掉的空白行）；区间内取首/末有效行，maxRows 含区间内被跳过的空白行对应的
// 输出行（与后端 start_index + max_rows 切片语义一致）。整段全为空白（或
// 单击空白行）时就近吸附：先向下找最近有效行，再向上；无有效行返回 null。
function selectionToRange(a, b, idxOf) {
  const n = idxOf.length;
  if (!n) return null;
  if (a > b) { const t = a; a = b; b = t; }
  a = Math.max(0, Math.min(a, n - 1));
  b = Math.max(0, Math.min(b, n - 1));
  let start = -1;
  let end = -1;
  for (let i = a; i <= b; i++) {
    const k = idxOf[i];
    if (k < 0) continue;
    if (start < 0) start = k;
    end = k;
  }
  if (start >= 0) return { start, maxRows: end - start + 1 };
  for (let i = b + 1; i < n; i++) if (idxOf[i] >= 0) return { start: idxOf[i], maxRows: 1 };
  for (let i = a - 1; i >= 0; i--) if (idxOf[i] >= 0) return { start: idxOf[i], maxRows: 1 };
  return null;
}

function injectCSS() {
  injectCSSOnce("sf-pl-css", `
.sf-pl-root { position:relative; display:flex; flex-direction:column; flex:1 1 0;
  min-height:0; box-sizing:border-box; padding:${PAD}px; gap:${PAD}px;
  font:12px sans-serif; color:var(--sf-text); overflow:hidden; background:transparent; }
.sf-pl-hdr { flex:0 0 auto; display:flex; align-items:center; gap:6px;
  padding:3px 6px 3px 9px; border:1px solid var(--sf-border-soft); border-radius:5px;
  background:var(--sf-surface); }
.sf-pl-hlbl { font:10px 'Segoe UI',-apple-system,sans-serif; color:var(--sf-text-faint); flex:1 1 0;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-pl-count { flex:0 0 auto; font-size:10px; color:var(--sf-text-faint);
  white-space:nowrap; user-select:none; }
.sf-pl-mode { flex:0 0 auto; box-sizing:border-box; display:inline-flex; align-items:center;
  background:var(--sf-surface); border:1px solid var(--sf-border-soft); border-radius:4px;
  color:var(--sf-text); cursor:pointer; font:10px 'Segoe UI',-apple-system,sans-serif;
  padding:2px 8px; transition:background .1s,color .1s,border-color .1s; }
.sf-pl-mode:hover { border-color:${"var(--sf-acc, #f66744)"}; color:var(--sf-text-strong); }
.sf-pl-mode.on { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"};
  color:var(--sf-text-strong); }
.sf-pl-editor { flex:1 1 0; min-height:0; display:flex;
  background:var(--sf-input-bg); border:1px solid var(--sf-border-soft); border-radius:5px; overflow:hidden; }
.sf-pl-editor:focus-within { border-color:${"var(--sf-acc, #f66744)"}; }
.sf-pl-gutter { flex:0 0 auto; overflow:hidden; box-sizing:border-box;
  padding:6px 0 6px 8px; background:var(--sf-surface);
  border-right:1px solid var(--sf-border-soft); user-select:none; }
.sf-pl-gn { display:block; text-align:right; padding-right:8px;
  font:12px monospace; line-height:1.4; color:var(--sf-text-faint); white-space:pre; }
.sf-pl-gn.sf-pl-gap { color:var(--sf-text-faint); font-style:italic; }
.sf-pl-gn.sf-pl-on { color:${"var(--sf-acc, #f66744)"}; font-weight:bold; }
.sf-pl-tawrap { flex:1 1 0; min-height:0; position:relative; display:flex; }
.sf-pl-hl { position:absolute; inset:0; overflow:hidden; pointer-events:none; }
.sf-pl-hl::before { content:""; display:block; height:var(--sf-pl-hl-pad, 0px); }
.sf-pl-hl-row { position:absolute; left:0; right:0; height:16.8px;
  background:${"var(--sf-acc, #f66744)"}; opacity:0.16; }
.sf-pl-ta { flex:1 1 0; min-height:0; width:100%; box-sizing:border-box;
  background:transparent; color:var(--sf-text); border:0; outline:none; resize:none;
  font:12px monospace; line-height:1.4; padding:6px 8px; }
.sf-pl-ta::placeholder { color:var(--sf-text-faint); font-style:italic; }
.sf-pl-ta.sf-pl-select { cursor:crosshair; }
.sf-pl-ta.sf-pl-select::selection { background:color-mix(in srgb, ${"var(--sf-acc, #f66744)"} 35%, transparent); }
.sf-pl-select-mode .sf-pl-gn { cursor:pointer; }
.sf-pl-select-mode .sf-pl-gn:hover { color:var(--sf-text-strong); background:var(--sf-surface-hover); }
`);
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
  const modeBtn = document.createElement("button");
  modeBtn.type = "button";
  modeBtn.className = "sf-pl-mode";
  const count = document.createElement("span");
  count.className = "sf-pl-count";
  hdr.append(hlbl, modeBtn, count);

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
  // 输入框滚轮透传：Ctrl+滚轮总缩放画布；普通滚轮在文本可滚动时滚动文本、
  // 否则缩放画布（对齐 ComfyUI 原生输入框；sf DOM widget 不在 Vue 转发路径内）
  installWheelZoomPassthrough(ta);

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

  // ── 点选模式（Edit/Select 开关；默认编辑，状态存 node.properties）──
  // readOnly 阻止编辑但保留原生文本选择：拖选自带边缘自动滚动、Shift+点击
  // 与方向键扩选——selectionStart/End 即选择范围，无需自绘指针几何/预览层
  let selectMode = !!(node.properties && node.properties[SELECT_KEY]);

  function applySelectMode(on) {
    selectMode = !!on;
    ta.readOnly = selectMode;
    ta.classList.toggle("sf-pl-select", selectMode);
    root.classList.toggle("sf-pl-select-mode", selectMode);
    modeBtn.classList.toggle("on", selectMode);
    modeBtn.textContent = selectMode ? "Select" : "Edit";
    modeBtn.title = selectMode
      ? "点选模式：单击选单行、拖选 / Shift+点击选多行，自动设置 start_index / max_rows；点击切回编辑模式"
      : "编辑模式；点击进入点选模式（鼠标选择行自动设置 start_index / max_rows）";
    node.properties = node.properties || {};
    if (selectMode) node.properties[SELECT_KEY] = true;
    else delete node.properties[SELECT_KEY];
    node.setDirtyCanvas?.(true, true);
  }
  modeBtn.addEventListener("pointerdown", (e) => e.stopPropagation?.());
  modeBtn.addEventListener("click", (e) => {
    e.preventDefault?.();
    e.stopPropagation?.();
    applySelectMode(!selectMode);
  });

  // ── 行高测量（wrap 开启时软换行精确对齐）──
  // mirror 与 textarea 同几何（同 padding 同字体同换行参数）且**同一布局
  // 容器内 left:0;right:0 拉伸**——宽度与 textarea 精确一致（含浮点），
  // 不用 clientWidth 取整值（亚像素宽度差在超长文本下会累积成 1 行差）。
  // 每逻辑行一个**块级 div**（inline span 的行盒高度取字体度量而非
  // line-height，实测系统性偏小——必须用块级），div 的
  // getBoundingClientRect().height = 该逻辑行在 textarea 中的视觉高度。
  // 按行文本缓存，编辑只重测变化的行；宽度变化（换行重新分布）时清空。
  // 不做 scrollHeight 总高校准——textarea 的 scrollHeight 含底部预留行
  // （超长文本实测虚高 1-2 行），校准会把正确测量改错。
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
        "position:absolute;left:0;top:0;visibility:hidden;pointer-events:none;" +
        "font:12px monospace;line-height:1.4;white-space:pre-wrap;overflow-wrap:break-word;" +
        "box-sizing:border-box;padding:6px 8px;overflow:hidden;";
      taWrap.appendChild(measContainer);
    }
    // 精确内容宽 = ta 布局宽（浮点） - 滚动条宽——与 textarea 内容区精确一致。
    // clientWidth 是取整值，亚像素差在超长文本（十几行）下累积成 1 行差；
    // 传统滚动条占宽 ~15px 需显式扣除（clientWidth 已扣，这里基于 rect 重算）
    const sb = Math.max(0, ta.offsetWidth - ta.clientWidth);
    measContainer.style.width = (ta.getBoundingClientRect().width - sb) + "px";
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
  //
  // lastIdxOf = 最近一次渲染的映射，点选提交直接复用（不重算）
  let lastIdxOf = [];
  function renderGutter() {
    const rows = ta.value.split("\n");
    const skip = skipEmptyOn();
    // 过滤后 index 映射：逻辑行 i → 输出 index（skip 时空行 -1）
    const idxOf = new Array(rows.length);
    let valid = 0;
    for (let i = 0; i < rows.length; i++) {
      idxOf[i] = skip && !rows[i].trim() ? -1 : valid++;
    }
    lastIdxOf = idxOf;
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
        s.dataset.line = String(i); // 点选模式：行号点击 → 逻辑行号
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
      // 行 i 正文顶边 = 6 + i*LINE_H。窗口起点须扣除该 6px：否则行边界后
      // 6px 区间内上一行仍有可见残段却不渲染（行号空档）。内联 paddingTop
      // 会覆盖 CSS 的 6px，故两处 padding 都补回 6px 保持与正文/高亮同基线
      const first = Math.max(0, Math.floor((ta.scrollTop - 6) / LINE_H));
      const visible = Math.max(1, Math.ceil(gutter.clientHeight / LINE_H) + 2);
      const last = Math.min(rows.length, first + visible);
      gutter.style.paddingTop = `${6 + first * LINE_H}px`;
      gutter.style.paddingBottom = `${Math.max(0, rows.length - last) * LINE_H + 6}px`;
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
        s.dataset.line = String(i); // 点选模式：行号点击 → 逻辑行号
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
    // hl 只含选中行高亮块，内容高远小于文本高 → scrollTop 同步会被浏览器
    // 钳制（高亮滚出视口后钉在视口边缘盖住无关文本，即"幽灵高亮"）。
    // ::before 垫高到文本全高（尾项补偿横向滚动条造成的两框 clientHeight
    // 差），使 hl 可滚范围 ≥ textarea，scrollTop 同步不被钳制
    hl.style.setProperty("--sf-pl-hl-pad",
      `${ta.scrollHeight + Math.max(0, hl.clientHeight - ta.clientHeight)}px`);
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

  // ── 点选提交：原生选择 → 逻辑行区间 → 输出索引写回 start_index/max_rows ──
  // 字符位置 → 逻辑行号（charCodeAt 10 = \n）；readOnly 下 selectionStart/End
  // 同样有效
  function lineOfChar(value, pos) {
    let line = 0;
    const end = Math.max(0, Math.min(pos, value.length));
    for (let i = 0; i < end; i++) if (value.charCodeAt(i) === 10) line++;
    return line;
  }

  // 写原生 widget（.value + callback 经 setupNode 已有包装链触发 _sfPlSync
  // 高亮重渲染）；按 widget min/max 钳制（超长文本行数可超 start_index 上限）
  function setNativeWidget(name, value) {
    const w = (node.widgets || []).find((x) => x && x.name === name);
    if (!w) return;
    const opt = w.options || {};
    let v = value;
    if (typeof opt.min === "number") v = Math.max(opt.min, v);
    if (typeof opt.max === "number") v = Math.min(opt.max, v);
    if (w.value === v) return;
    w.value = v;
    try { w.callback?.(v); } catch { /* 回调异常不阻断点选 */ }
  }

  function applyRange(range) {
    if (!range) return;
    setNativeWidget("start_index", range.start);
    setNativeWidget("max_rows", range.maxRows);
    root._sfPlUpdateWatch?.(); // 同步轮询快照，防 checkWatch 误判重渲染
    node.setDirtyCanvas?.(true, true);
  }

  function commitSelection() {
    if (!selectMode) return;
    const value = ta.value;
    const a = lineOfChar(value, ta.selectionStart ?? 0);
    const b = lineOfChar(value, ta.selectionEnd ?? 0);
    applyRange(selectionToRange(a, b, lastIdxOf));
  }

  // 行号点击：直接按逻辑行号提交（虚拟化窗口行同样带 dataset.line）
  function pickLine(line) {
    if (!selectMode || !Number.isFinite(line)) return;
    applyRange(selectionToRange(line, line, lastIdxOf));
  }

  // ── 行号栏拖动多选（点选模式）──
  // pointerdown 定锚点 → 实时扩展（rAF 合并重渲染）；重渲染会重建 span，故
  // pointermove/up 挂 window 捕获按命中元素 dataset.line 跟踪，不 setPointerCapture
  let gutterDrag = null;
  let gutterDragTimer = null;
  function lineFromTarget(t) {
    const span = t?.closest?.(".sf-pl-gn");
    const line = Number(span?.dataset?.line);
    return Number.isFinite(line) ? line : null;
  }
  function gutterDragStart(line) {
    if (!selectMode || !Number.isFinite(line)) return;
    gutterDrag = { anchor: line, focus: line };
    applyRange(selectionToRange(line, line, lastIdxOf));
  }
  function gutterDragMove(line) {
    if (!gutterDrag || !selectMode || !Number.isFinite(line)) return;
    gutterDrag.focus = line;
    if (gutterDragTimer) return; // 本帧已排队，合并一次重渲染
    const timer = requestAnimationFrame(() => {
      gutterDragTimer = null;
      if (gutterDrag) applyRange(selectionToRange(gutterDrag.anchor, gutterDrag.focus, lastIdxOf));
    });
    gutterDragTimer = timer || 0; // 测试 mock 同步执行且返回 undefined
  }
  function gutterDragEnd() {
    if (!gutterDrag) return;
    const { anchor, focus } = gutterDrag;
    gutterDrag = null;
    if (gutterDragTimer) {
      try { globalThis.cancelAnimationFrame?.(gutterDragTimer); } catch { /* ignore */ }
      gutterDragTimer = null;
    }
    applyRange(selectionToRange(anchor, focus, lastIdxOf));
  }
  function onGutterPointerMove(e) {
    if (e.buttons === 0) { onGutterPointerEnd(); return; } // 窗口外松手漏 pointerup 兜底
    const line = lineFromTarget(e.target);
    if (line !== null) gutterDragMove(line);
  }
  function onGutterPointerEnd() {
    window.removeEventListener("pointermove", onGutterPointerMove, true);
    window.removeEventListener("pointerup", onGutterPointerEnd, true);
    window.removeEventListener("pointercancel", onGutterPointerEnd, true);
    gutterDragEnd();
  }
  gutter.addEventListener("pointerdown", (e) => {
    if (!selectMode) return;
    if (gutterDrag) onGutterPointerEnd(); // 上次拖拽漏收尾（窗口外松手）自愈
    const line = lineFromTarget(e.target);
    if (line === null) return;
    e.preventDefault?.();
    e.stopPropagation?.();
    gutterDragStart(line);
    window.addEventListener("pointermove", onGutterPointerMove, true);
    window.addEventListener("pointerup", onGutterPointerEnd, true);
    window.addEventListener("pointercancel", onGutterPointerEnd, true);
  });

  // select 事件覆盖拖选；click/mouseup/keyup 兜底单击与键盘扩选；rAF 合并
  let commitScheduled = false;
  function scheduleCommit() {
    if (!selectMode || commitScheduled) return;
    commitScheduled = true;
    requestAnimationFrame(() => {
      commitScheduled = false;
      commitSelection();
    });
  }
  ta.addEventListener("select", scheduleCommit);
  ta.addEventListener("click", scheduleCommit);
  ta.addEventListener("mouseup", scheduleCommit);
  ta.addEventListener("keyup", scheduleCommit);

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

  // 事件防护：防 canvas 拖拽/取消选中/快捷键；放行所有修饰键组合（Ctrl+S
  // 保存工作流、Ctrl+C/V 复制粘贴、Ctrl+Enter 运行等——否则焦点在输入框时
  // Ctrl+S 会漏成浏览器"保存网页"）
  ta.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
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
  // 点选模式钩子（onConfigure 恢复 / 测试直调）
  root._sfPlSetSelectMode = applySelectMode;
  root._sfPlConfigMode = () => applySelectMode(!!(node.properties && node.properties[SELECT_KEY]));
  root._sfPlApplySelection = commitSelection;
  root._sfPlPickLine = pickLine;
  root._sfPlGutterDragStart = gutterDragStart;
  root._sfPlGutterDragMove = gutterDragMove;
  root._sfPlGutterDragEnd = gutterDragEnd;
  applySelectMode(selectMode);

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
      const root = this._sfPromptListRoot;
      root?._sfPlConfigMode?.(); // 点选模式随 properties 恢复
      root?._sfPlSync();
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
