// ==========================================================================
// sf_find_replace_ui.js - SFTextFindReplace 节点体 DOM widget 渲染 + 交互
// ==========================================================================
//
// 构建节点体：全局开关 pill 行、每条规则一行（拖拽把手 / ON-OFF / 查找 -> 替换
// textarea / 删除）、操作行（+ Add rule / ↺ Reset）、以及实时前后对比预览。
//
// 与 Pixaroma 原件的差异（已确认范围）：无 accent 颜色设置（固定强调色
// #f66744）、无注册帮助面板系统、无 resize floor / canvas zoom 穿透辅助；
// CSS 类名统一 sf-fr- 前缀（与 Pixaroma 共存时互不污染）。
//
// ==========================================================================

import {
    readState,
    applyRulesJS,
    diffTokens,
    escapeHtml,
    getPreviewInput,
    setFind,
    setReplace,
} from "./sf_find_replace_lib.js";
import { injectCSSOnce, installWheelZoomPassthrough } from "./sf_common.js";

const CSS_ID = "sf-find-replace-css";

const CSS = `
.sf-fr-root {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 6px 8px 8px 8px;
  box-sizing: border-box;
  font-family: inherit;
  color: #ddd;
  /* NO height:100% 与 NO min-height——刻意（Prompt Reader 模式）。Nodes 2.0 的
     宿主 wrapper 给此 root flex:1，它仍填满节点体、预览随节点增高；legacy 则由
     ComfyUI 尺寸化 widget 元素。关键是 root 的自然 flex min-content 高度（固定
     行 + 预览真实 min-height）就是 Nodes 2.0 缩放地板要测量的值（它把节点折叠
     到 --node-height:0 再读内容高度），所以节点不会被拖小到溢出——无需 JS。
     这里若设 height:100% 会在该测量下塌缩为 0 并破坏地板。 */
}

/* ---- 顶行：开关 pill（窄节点上它们之间自动换行） ---- */
.sf-fr-toprow { display: flex; align-items: flex-start; gap: 6px; flex: 0 0 auto; }
.sf-fr-toggles { display: flex; gap: 6px; flex-wrap: wrap; flex: 1 1 auto; min-width: 0; }
.sf-fr-tog {
  font-size: 10.5px;
  padding: 4px 10px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.05);
  color: rgba(255,255,255,0.68);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.sf-fr-tog:hover { border-color: ${"var(--sf-acc, #f66744)"}; color: #ddd; }
.sf-fr-tog.on { background: ${"var(--sf-acc, #f66744)"}; border-color: ${"var(--sf-acc, #f66744)"}; color: #fff; }
.sf-fr-tog.on:hover { filter: brightness(1.08); color: #fff; }
.sf-fr-tog.is-muted {
  opacity: 0.4;
  cursor: not-allowed;
  border-color: rgba(255,255,255,0.1);
}
.sf-fr-tog.is-muted:hover { border-color: rgba(255,255,255,0.1); color: rgba(255,255,255,0.68); }

/* ---- 规则行 ---- */
.sf-fr-row {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 6px;
  border-radius: 4px;
  /* 半透明叠加（非不透明深色）使行适应被用户改色的节点体，而非显示灰色块。 */
  background: rgba(0,0,0,0.18);
  border: 1px solid rgba(255,255,255,0.08);
  position: relative;
  transition: opacity 0.12s ease;
  flex: 0 0 auto;
}
.sf-fr-row.is-disabled { opacity: 0.45; }
.sf-fr-row.is-dragging { opacity: 0.4; }
.sf-fr-row.is-drop-target-above { box-shadow: 0 -2px 0 0 ${"var(--sf-acc, #f66744)"}; }
.sf-fr-row.is-drop-target-below { box-shadow: 0 2px 0 0 ${"var(--sf-acc, #f66744)"}; }

.sf-fr-handle {
  cursor: grab;
  color: #888;
  font-size: 14px;
  line-height: 22px;
  user-select: none;
  padding: 0 1px;
  letter-spacing: -2px;
  flex: none;
}
.sf-fr-handle:active { cursor: grabbing; }
.sf-fr-handle:hover { color: #ccc; }

.sf-fr-toggle {
  min-width: 30px;
  height: 18px;
  margin-top: 2px;
  border-radius: 9px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.15);
  cursor: pointer;
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  font-weight: 600;
  color: rgba(255,255,255,0.65);
  letter-spacing: 0.5px;
  user-select: none;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
.sf-fr-toggle:hover { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.35); color: #fff; }
.sf-fr-toggle.on { background: ${"var(--sf-acc, #f66744)"}; border-color: ${"var(--sf-acc, #f66744)"}; color: #fff; }
.sf-fr-toggle.on:hover { filter: brightness(1.08); color: #fff; }

/* find -> replace 字段 */
.sf-fr-field {
  flex: 1 1 0;
  /* align-self:flex-start + height（非 min-height）让 textarea 保持内容尺寸
     （一行直到你输入更多），所以它不会在节点很大时（如删除规则后）拉伸填满
     高行，也不会在节点被压缩时出现滚动条。autoGrow 对多行内容覆盖 height，
     上限 max-height。 */
  align-self: flex-start;
  min-width: 0;
  height: 30px;
  max-height: 120px;
  resize: none;
  background: #1d1d1d;
  border: 1px solid #333;
  border-radius: 4px;
  color: #e0e0e0;
  font: 12px monospace;
  padding: 6px 8px;
  outline: none;
  box-sizing: border-box;
  overflow-y: auto;
  line-height: 1.35;
}
.sf-fr-field:focus { border-color: ${"var(--sf-acc, #f66744)"}; }
.sf-fr-field::placeholder { color: rgba(255,255,255,0.32); font-style: italic; }
.sf-fr-field.is-delete::placeholder { color: rgba(255,150,160,0.55); }

.sf-fr-arrow { color: ${"var(--sf-acc, #f66744)"}; font-weight: 700; line-height: 30px; flex: none; }

.sf-fr-delete {
  width: 18px;
  height: 18px;
  margin-top: 2px;
  border-radius: 3px;
  background: transparent;
  border: none;
  color: #888;
  cursor: pointer;
  font-size: 14px;
  line-height: 14px;
  flex: none;
  padding: 0;
}
.sf-fr-delete:hover { color: ${"var(--sf-acc, #f66744)"}; background: color-mix(in srgb, ${"var(--sf-acc, #f66744)"} 12%, transparent); }
.sf-fr-delete:disabled { color: #444; cursor: not-allowed; background: transparent; }

/* ---- 操作行 ---- */
.sf-fr-actions { display: flex; flex-wrap: wrap; gap: 6px; align-self: flex-start; user-select: none; flex: 0 0 auto; }
.sf-fr-add, .sf-fr-reset {
  box-sizing: border-box;
  min-width: 92px;
  user-select: none;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 4px;
  color: rgba(255,255,255,0.85);
  cursor: pointer;
  font: 11px inherit;
  font-family: inherit;
  padding: 5px 12px;
  transition: background 0.1s, color 0.1s, border-color 0.1s;
}
.sf-fr-add { color: ${"var(--sf-acc, #f66744)"}; border-color: color-mix(in srgb, ${"var(--sf-acc, #f66744)"} 50%, transparent); }
.sf-fr-add:hover { background: ${"var(--sf-acc, #f66744)"}; border-color: ${"var(--sf-acc, #f66744)"}; color: #fff; }
.sf-fr-reset:hover { background: ${"var(--sf-acc, #f66744)"}; border-color: ${"var(--sf-acc, #f66744)"}; color: #fff; }
.sf-fr-reset:disabled {
  color: rgba(255,255,255,0.3);
  cursor: default;
  background: rgba(255,255,255,0.02);
  border-color: rgba(255,255,255,0.08);
}
.sf-fr-reset:disabled:hover {
  background: rgba(255,255,255,0.02);
  border-color: rgba(255,255,255,0.08);
  color: rgba(255,255,255,0.3);
}

/* ---- 实时预览（填满节点剩余高度） ---- */
.sf-fr-preview {
  border-top: 1px solid #3a3a3a;
  padding-top: 8px;
  flex: 1 1 0;
  /* 真实 min-height（非 0）：这是 flex 区域，其 min-height 就是让 root 在
     Nodes 2.0 缩放地板测量下不会塌缩到内容之下。它仍会增长填满额外节点高度。 */
  min-height: 100px;
  display: flex;
  flex-direction: column;
}
.sf-fr-prev-head {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.4px;
  color: #7fd18f; font-weight: 700; margin-bottom: 5px;
  flex: 0 0 auto;
}
.sf-fr-prev-note { color: #666; font-weight: 400; text-transform: none; letter-spacing: 0; font-size: 9.5px; }
.sf-fr-prev-body {
  background: #161616;
  border: 1px solid #2c3a2c;
  border-radius: 4px;
  padding: 7px 9px;
  font: 11px monospace;
  color: #cfcfcf;
  line-height: 1.5;
  flex: 1 1 0;
  min-height: 60px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
.sf-fr-before { color: #7d7d7d; margin-bottom: 5px; }
.sf-fr-before .o { background: #3a2026; color: #e2899a; text-decoration: line-through; border-radius: 2px; padding: 0 1px; }
.sf-fr-after .n { background: #1f4a2a; color: #9af0ad; border-radius: 2px; padding: 0 1px; }
.sf-fr-prev-empty { color: #777; font-style: italic; }
.sf-fr-prev-nochange { color: #888; font-size: 9.5px; margin-top: 4px; font-style: italic; }
.sf-fr-prev-trunc { color: #b89; font-size: 9.5px; margin-top: 5px; }
.sf-fr-warn { color: #e9b04a; font-size: 10px; margin-top: 6px; }

/* ---- 确认对话框 ---- */
.sf-fr-confirm-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.55);
  display: flex; align-items: center; justify-content: center;
  z-index: 10000; font-family: inherit; -webkit-font-smoothing: antialiased;
}
.sf-fr-confirm-box {
  background: #1d1d1d; border: 1px solid #2e2e2e; border-radius: 6px;
  min-width: 320px; max-width: 480px; padding: 18px 20px; color: #ddd;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
.sf-fr-confirm-title { font-size: 14px; font-weight: 600; color: #fff; margin: 0 0 8px 0; }
.sf-fr-confirm-msg { font-size: 13px; color: #bbb; margin: 0 0 16px 0; line-height: 1.4; }
.sf-fr-confirm-actions { display: flex; gap: 8px; justify-content: flex-end; }
.sf-fr-confirm-btn {
  background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 3px;
  color: #ddd; cursor: pointer; font-size: 12px; padding: 6px 14px; font-family: inherit;
}
.sf-fr-confirm-btn:hover { background: #333; border-color: #555; }
.sf-fr-confirm-btn.primary { background: ${"var(--sf-acc, #f66744)"}; border-color: ${"var(--sf-acc, #f66744)"}; color: #fff; }
.sf-fr-confirm-btn.primary:hover { background: #ff7a58; border-color: #ff7a58; }
`;

export function injectCSS() {
    injectCSSOnce(CSS_ID, CSS);
}

export function buildRoot() {
    const root = document.createElement("div");
    root.className = "sf-fr-root";
    return root;
}

// 预览块的最小高度（head + body 几行）。预览 flex 填满超过此地板之外的节点高度。
const PREVIEW_MIN = 100;

// 节点最小高度 = 固定部分（开关 + 规则行 + 操作，实时测量）+ 最小预览块。
// 不是完整预览——用户可以把节点拖高、预览填满新空间（无死区）。
// 累加子元素 offsetHeight（不是 root.scrollHeight，后者会被 ComfyUI 拉伸——
// 反馈回路），预览弹性子元素用 PREVIEW_MIN 代替。按 4px 网格取整，子像素/字体
// 抖动不会让 node.size 在每次工作流切换时变大（getMinHeight/computeLayoutSize
// 喂 Nodes 2.0 的 grow-to-content，它只增不减并会累积）。
export function measureMinHeight(root) {
    if (!root) return 180;
    let h = 0;
    let count = 0;
    for (const child of root.children) {
        if (child.offsetParent === null) continue;
        count += 1;
        if (child.classList.contains("sf-fr-preview")) h += PREVIEW_MIN;
        else h += child.offsetHeight;
    }
    const cs = getComputedStyle(root);
    const gap = parseFloat(cs.rowGap || cs.gap) || 0;
    if (count > 1) h += gap * (count - 1);
    h += parseFloat(cs.paddingTop) || 0;
    h += parseFloat(cs.paddingBottom) || 0;
    return Math.max(180, Math.round(h / 4) * 4);
}

const TOGGLE_DEFS = [
    { key: "caseSensitive", label: "Aa Case", title: "精确匹配大小写。关闭 = 忽略大小写。" },
    { key: "wholeWord", label: "Whole word", title: "只匹配整个单词，'art' 不会命中 'artist'。" },
    { key: "regex", label: ".* Regex", title: "把查找字段当作正则表达式。替换可用 \\1 反向引用。" },
    { key: "tidy", label: "✨ Tidy", title: "编辑完成后，折叠多余空格并修复孤立或重复的逗号。" },
];

// renderAll：清空 root 并重建整个节点体。
export function renderAll(node, root, handlers) {
    const state = readState(node);
    root.innerHTML = "";

    // -- 顶行：开关 --
    const toprow = document.createElement("div");
    toprow.className = "sf-fr-toprow";
    const toggles = document.createElement("div");
    toggles.className = "sf-fr-toggles";
    for (const def of TOGGLE_DEFS) {
        const pill = document.createElement("div");
        const muted = def.key === "wholeWord" && state.regex;
        pill.className = "sf-fr-tog" + (state[def.key] ? " on" : "") + (muted ? " is-muted" : "");
        pill.textContent = def.label;
        pill.title = muted ? "Regex 开启时 Whole word 被忽略（可在模式里加 \\b）。" : def.title;
        if (!muted) pill.addEventListener("click", () => handlers.onToggleGlobal(def.key));
        toggles.appendChild(pill);
    }
    toprow.appendChild(toggles);
    root.appendChild(toprow);

    // -- 规则行 --
    for (const rule of state.rules) {
        root.appendChild(buildRuleRow(node, state, rule, handlers));
    }

    // -- 操作 --
    const actions = document.createElement("div");
    actions.className = "sf-fr-actions";

    const add = document.createElement("button");
    add.className = "sf-fr-add";
    add.type = "button";
    add.textContent = "+ Add rule";
    add.title = "在底部添加一条空的查找/替换规则";
    add.addEventListener("click", () => handlers.onAdd());
    actions.appendChild(add);

    const reset = document.createElement("button");
    reset.className = "sf-fr-reset";
    reset.type = "button";
    reset.textContent = "↺ Reset";
    reset.title = "清除所有规则并把开关恢复默认";
    reset.addEventListener("click", () => handlers.onReset());
    actions.appendChild(reset);
    root.appendChild(actions);

    // -- 预览 --
    const preview = document.createElement("div");
    preview.className = "sf-fr-preview";
    const head = document.createElement("div");
    head.className = "sf-fr-prev-head";
    head.innerHTML = `<span>Live preview</span><span class="sf-fr-prev-note">last text that ran through</span>`;
    preview.appendChild(head);
    const body = document.createElement("div");
    body.className = "sf-fr-prev-body";
    preview.appendChild(body);
    root.appendChild(preview);

    refreshResetState(node, root);
    renderPreview(node, root);
}

function buildRuleRow(node, state, rule, handlers) {
    const rowEl = document.createElement("div");
    rowEl.className = "sf-fr-row" + (rule.enabled ? "" : " is-disabled");
    rowEl.dataset.id = rule.id;
    rowEl.draggable = false;

    const handle = document.createElement("span");
    handle.className = "sf-fr-handle";
    handle.draggable = true;
    handle.textContent = "⋮⋮";
    handle.title = "拖动排序";
    rowEl.appendChild(handle);

    const toggle = document.createElement("div");
    toggle.className = "sf-fr-toggle" + (rule.enabled ? " on" : "");
    toggle.textContent = rule.enabled ? "ON" : "OFF";
    toggle.title = rule.enabled ? "点击跳过此规则" : "点击应用此规则";
    toggle.addEventListener("click", () => handlers.onToggleRule(rule.id));
    rowEl.appendChild(toggle);

    const findTa = document.createElement("textarea");
    findTa.className = "sf-fr-field sf-fr-find";
    findTa.value = rule.find || "";
    findTa.rows = 1;
    findTa.placeholder = "find...";
    findTa.title = "要查找的文本" + (state.regex ? "（正则表达式）" : "");
    rowEl.appendChild(findTa);
    installWheelZoomPassthrough(findTa); // 输入框滚轮透传(缩放画布/滚动文本, 对齐原生)
    attachFieldEditor(node, findTa, rule.id, "find");

    const arrow = document.createElement("span");
    arrow.className = "sf-fr-arrow";
    arrow.textContent = "→";
    rowEl.appendChild(arrow);

    const replaceTa = document.createElement("textarea");
    replaceTa.className = "sf-fr-field sf-fr-replace" + ((rule.replace || "") ? "" : " is-delete");
    replaceTa.value = rule.replace || "";
    replaceTa.rows = 1;
    replaceTa.placeholder = "replace…";
    replaceTa.title = "替换成的文本。留空 = 删除查找到的文本。";
    rowEl.appendChild(replaceTa);
    installWheelZoomPassthrough(replaceTa); // 输入框滚轮透传(缩放画布/滚动文本, 对齐原生)
    attachFieldEditor(node, replaceTa, rule.id, "replace");

    const del = document.createElement("button");
    del.className = "sf-fr-delete";
    del.type = "button";
    del.textContent = "✕";
    del.title = "删除此规则";
    del.disabled = state.rules.length <= 1;
    del.addEventListener("click", () => handlers.onDelete(rule.id));
    rowEl.appendChild(del);

    attachDragHandlers(node, rowEl, rule.id, handlers.onDrop);
    return rowEl;
}

// 根据是否有任何非默认内容启用/禁用 Reset 按钮。
export function refreshResetState(node, root) {
    const reset = root.querySelector(".sf-fr-reset");
    if (!reset) return;
    const s = readState(node);
    const anyRuleContent = s.rules.some((r) => (r.find && r.find.trim()) || (r.replace && r.replace.trim()) || !r.enabled);
    const moreThanOne = s.rules.length !== 1;
    const nonDefaultToggles = s.caseSensitive || s.wholeWord || s.regex || s.tidy !== true;
    reset.disabled = !(anyRuleContent || moreThanOne || nonDefaultToggles);
}

// renderPreview：从持久化的上次运行输入 + 当前规则重算前后对比 diff，填入预览
// 体。每次编辑/开关调用都安全，无需完整重渲染。
export function renderPreview(node, root) {
    const body = root.querySelector(".sf-fr-prev-body");
    if (!body) return;
    const prev = getPreviewInput(node);
    if (!prev) {
        body.innerHTML = `<div class="sf-fr-prev-empty">Run the workflow once to preview the result.</div>`;
        return;
    }
    const state = readState(node);
    const { output, warnings } = applyRulesJS(prev.input, state);

    let html;
    if (output === prev.input) {
        html =
            `<div class="sf-fr-after">${escapeHtml(output) || '<span style="color:#666">(empty)</span>'}</div>` +
            `<div class="sf-fr-prev-nochange">no changes from your current rules</div>`;
    } else {
        const diff = diffTokens(prev.input, output);
        let beforeHtml = "";
        let afterHtml = "";
        for (const part of diff) {
            const esc = escapeHtml(part.s);
            if (part.t === "eq") {
                beforeHtml += esc;
                afterHtml += esc;
            } else if (part.t === "del") {
                beforeHtml += `<span class="o">${esc}</span>`;
            } else {
                afterHtml += `<span class="n">${esc}</span>`;
            }
        }
        html =
            `<div class="sf-fr-before">${beforeHtml}</div>` +
            `<div class="sf-fr-after">${afterHtml || '<span style="color:#666">(empty)</span>'}</div>`;
    }

    if (prev.truncated) {
        html += `<div class="sf-fr-prev-trunc">Preview sample shortened - the full text still passes through.</div>`;
    }
    if (warnings && warnings.length) {
        html += `<div class="sf-fr-warn">⚠ ${escapeHtml(warnings.join("; "))}</div>`;
    }
    body.innerHTML = html;
}

// ==========================================================================
// 交互：字段编辑器、拖拽排序、主题确认框
// ==========================================================================
//
// 所有输入事件用 stopImmediatePropagation 阻止逃逸进 ComfyUI canvas 快捷键。
// Enter 插入换行（find/replace 值可能合法含换行），因此不拦截。

function autoGrow(ta) {
    // 空字段：钉在一行。不要为换行的 PLACEHOLDER 增长——节点窄时 placeholder
    // （"find..."）会换行成多行、scrollHeight 暴涨，字段变高（且节点加宽后也
    // 不会缩回，因为 autoGrow 只在 input 时运行）。只有真实输入内容让字段增长。
    if (!ta.value) { ta.style.height = "30px"; return; }
    ta.style.height = "auto";
    ta.style.height = Math.max(30, Math.min(ta.scrollHeight, 120)) + "px";
}

// 重新测量 root 内每个 find/replace 字段。由宽度变化 ResizeObserver 调用，使
// 窄宽度下增长（内容换行）的字段在节点加宽时缩回。
export function autoGrowAllFields(root) {
    if (!root) return;
    root.querySelectorAll(".sf-fr-field").forEach((ta) => autoGrow(ta));
}

// which = "find" | "replace"
export function attachFieldEditor(node, taEl, ruleId, which) {
    taEl.dataset.committed = taEl.value;
    let pending = false;

    const commit = () => {
        if (taEl.value !== taEl.dataset.committed) {
            if (which === "find") setFind(node, ruleId, taEl.value);
            else setReplace(node, ruleId, taEl.value);
            taEl.dataset.committed = taEl.value;
        }
        pending = false;
    };

    taEl.addEventListener("input", (e) => {
        e.stopImmediatePropagation();
        autoGrow(taEl);
        // 同步提交按键到 state，使下次读取是最新的。
        if (which === "find") setFind(node, ruleId, taEl.value);
        else setReplace(node, ruleId, taEl.value);
        taEl.dataset.committed = taEl.value;
        if (which === "replace") taEl.classList.toggle("is-delete", !taEl.value);
        // 把（较重的）预览重算 + Reset 状态 + 节点增高合并为一个 rAF，按住按键
        // 时不会每次按键都重算词级 diff。
        if (!pending) {
            pending = true;
            requestAnimationFrame(() => {
                node._sfFrRefreshPreview?.();
                node._sfFrRefreshReset?.();
                node._sfFrRefit?.();
                pending = false;
            });
        }
    });

    taEl.addEventListener("keydown", (e) => {
        // 放行所有修饰键组合（Ctrl+S 保存工作流、Ctrl/Cmd+Enter 运行等）——
        // 否则焦点在输入框时 Ctrl+S 会漏成浏览器"保存网页"。
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.stopImmediatePropagation();
    });

    taEl.addEventListener("blur", commit);
    taEl.addEventListener("pointerdown", (e) => e.stopImmediatePropagation());
    taEl.addEventListener("mousedown", (e) => e.stopImmediatePropagation());

    // 延后初始 auto-grow 使 textarea 已在 DOM 中（未挂载元素 scrollHeight 为 0）。
    // 没有它，重渲染会让已增长的字段塌缩。
    requestAnimationFrame(() => autoGrow(taEl));
}

// sfConfirm：主题确认对话框（不用原生 window.confirm()）。返回 Promise<boolean>。
export function sfConfirm({ title, message, okText = "OK", cancelText = "Cancel" } = {}) {
    return new Promise((resolve) => {
        const backdrop = document.createElement("div");
        backdrop.className = "sf-fr-confirm-backdrop";

        const box = document.createElement("div");
        box.className = "sf-fr-confirm-box";

        const titleEl = document.createElement("div");
        titleEl.className = "sf-fr-confirm-title";
        titleEl.textContent = title || "Confirm";
        box.appendChild(titleEl);

        if (message) {
            const msgEl = document.createElement("div");
            msgEl.className = "sf-fr-confirm-msg";
            msgEl.textContent = message;
            box.appendChild(msgEl);
        }

        const actions = document.createElement("div");
        actions.className = "sf-fr-confirm-actions";

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "sf-fr-confirm-btn";
        cancelBtn.textContent = cancelText;
        actions.appendChild(cancelBtn);

        const okBtn = document.createElement("button");
        okBtn.type = "button";
        okBtn.className = "sf-fr-confirm-btn primary";
        okBtn.textContent = okText;
        actions.appendChild(okBtn);

        box.appendChild(actions);
        backdrop.appendChild(box);
        document.body.appendChild(backdrop);

        let done = false;
        const finish = (val) => {
            if (done) return;
            done = true;
            window.removeEventListener("keydown", onKey, true);
            backdrop.remove();
            resolve(val);
        };
        const onKey = (e) => {
            if (e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); finish(false); }
            else if (e.key === "Enter") { e.preventDefault(); e.stopImmediatePropagation(); finish(true); }
        };
        window.addEventListener("keydown", onKey, true);
        backdrop.addEventListener("mousedown", (e) => { if (e.target === backdrop) finish(false); });
        cancelBtn.addEventListener("click", () => finish(false));
        okBtn.addEventListener("click", () => finish(true));
        queueMicrotask(() => okBtn.focus());
    });
}

// 拖拽排序规则行。HANDLE 是拖拽源而非整行（在 textarea 内拖动可正常选择文本）。
const _drag = { id: null };

// 移除所有行上的拖拽/落点类（root 范围，非 parentElement 范围），使被放弃的
// 拖拽——落在非行区域，或行已重建后结束——不会留下过期的落点高亮线。
function clearDragClasses(root) {
    if (!root) return;
    root.querySelectorAll(".sf-fr-row").forEach((s) => {
        s.classList.remove("is-drop-target-above", "is-drop-target-below", "is-dragging");
    });
}

export function attachDragHandlers(node, rowEl, rowId, onDrop) {
    rowEl.addEventListener("dragstart", (e) => {
        if (!e.target.closest || !e.target.closest(".sf-fr-handle")) {
            e.preventDefault();
            return;
        }
        clearDragClasses(node._sfFrRoot); // 清除先前拖拽的残留
        _drag.id = rowId;
        rowEl.classList.add("is-dragging");
        try { e.dataTransfer.effectAllowed = "move"; } catch (_) {}
        try { e.dataTransfer.setData("text/plain", rowId); } catch (_) {}
    });

    rowEl.addEventListener("dragover", (e) => {
        if (!_drag.id || _drag.id === rowId) return;
        e.preventDefault();
        try { e.dataTransfer.dropEffect = "move"; } catch (_) {}
        const rect = rowEl.getBoundingClientRect();
        const isAbove = (e.clientY - rect.top) < rect.height / 2;
        rowEl.classList.toggle("is-drop-target-above", isAbove);
        rowEl.classList.toggle("is-drop-target-below", !isAbove);
    });

    rowEl.addEventListener("dragleave", () => {
        rowEl.classList.remove("is-drop-target-above");
        rowEl.classList.remove("is-drop-target-below");
    });

    rowEl.addEventListener("drop", (e) => {
        if (!_drag.id || _drag.id === rowId) return;
        e.preventDefault();
        const above = rowEl.classList.contains("is-drop-target-above");
        const fromId = _drag.id;
        _drag.id = null;
        clearDragClasses(node._sfFrRoot);
        onDrop(fromId, rowId, above);
    });

    rowEl.addEventListener("dragend", () => {
        _drag.id = null;
        clearDragClasses(node._sfFrRoot);
    });
}
