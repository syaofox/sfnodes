// ==========================================================================
// sf_prompt_stack.js - SFPromptStack 动态 Prompt 列表节点
// ==========================================================================
// 行数据存 node.properties.promptStackState（JSON，随工作流保存），由
// graphToPrompt 钩子注入隐藏 PromptStackState 输入（同 SFLoraStack 模式；
// buildIndex/findNode 复用 sf_lora_stack.js 导出——规则 14 不内联副本）。
// 行 UI：输出 index（从 0 起、跳过关闭行）+ 开关 + 多行输入 + ▲▼✕。
// 文本变化只写 state 不重渲染（防丢焦点）；结构变化（增删/排序/开关）
// 才重建行。Python 侧只过滤 enabled 且非空的行。
// ==========================================================================

import { app } from "/scripts/app.js";
import { applyAdaptiveCanvasOnly, isVueNodes } from "./sf_common.js";
import { buildIndex, findNode } from "./sf_lora_stack.js";
import {
    HIDDEN_INPUT, readState, writeState, promptState, activeRows, newId,
    MIN_ROW_H, MAX_ROW_H,
} from "./sf_prompt_stack_core.js";

const CLASS = "SFPromptStack";
const WIDGET_TYPE = "sf_prompt_stack_ui";

// 高度常量——与 CSS 锁步（SFLoraStack 同款）
const PAD = 9;
const ADD_H = 30;
const ROW_H = 52;   // 默认行高：多行输入 2 行 + 内边距（行可拖拽调节）
const ROW_GAP = 6;
const EMPTY_H = 30;
const MIN_W = 340;

// 行高（拖拽后存 state.rows[i].h；未设置用默认）
function rowHOf(row) {
    return typeof row.h === "number" && Number.isFinite(row.h) ? row.h : ROW_H;
}

function contentHeight(st) {
    const n = st.rows.length;
    if (!n) return PAD + ADD_H + ROW_GAP + EMPTY_H + PAD;
    let rowsH = 0;
    for (const r of st.rows) rowsH += rowHOf(r);
    rowsH += (n - 1) * ROW_GAP;
    return PAD + ADD_H + ROW_GAP + rowsH + PAD;
}

function injectCSS() {
    if (document.getElementById("sf-ps-css")) return;
    const s = document.createElement("style");
    s.id = "sf-ps-css";
    s.textContent = `
.sf-ps-root { box-sizing:border-box; width:100%; padding:${PAD}px;
  background:#1d1d1d; border-radius:4px; color:#ddd;
  font:12px sans-serif; position:relative; }
.sf-ps-add { box-sizing:border-box; width:100%; height:${ADD_H}px; border:0; border-radius:6px;
  background:${"var(--sf-acc, #f66744)"}; color:#fff; font:600 12px 'Segoe UI',sans-serif;
  cursor:pointer; display:flex; align-items:center; justify-content:center; gap:6px; }
.sf-ps-add:hover { filter:brightness(1.08); }
.sf-ps-empty { height:${EMPTY_H}px; display:flex; align-items:center; justify-content:center;
  color:#777; font-size:11px; user-select:none; }
.sf-ps-row { display:flex; align-items:stretch; gap:6px;
  margin-top:${ROW_GAP}px; }
.sf-ps-row:first-of-type { margin-top:0; }
.sf-ps-idx { flex:0 0 26px; display:flex; align-items:center; justify-content:center;
  font:11px monospace; color:#888; user-select:none; }
.sf-ps-idx.off { color:#555; }
.sf-ps-tg { flex:0 0 22px; display:flex; align-items:center; justify-content:center;
  cursor:pointer; user-select:none; font-size:14px; color:${"var(--sf-acc, #f66744)"}; }
.sf-ps-tg.off { color:#666; }
.sf-ps-tawrap { flex:1 1 0; min-width:0; position:relative; display:flex; }
.sf-ps-ta { flex:1 1 0; min-width:0; box-sizing:border-box; resize:none;
  background:rgba(255,255,255,0.04); color:#e0e0e0; border:1px solid #333;
  border-radius:5px; outline:none; font:12px monospace; line-height:1.4;
  padding:5px 7px; }
.sf-ps-ta:focus { border-color:${"var(--sf-acc, #f66744)"}; }
.sf-ps-ta.off { opacity:0.45; }
/* 右下角拖拽角标（调节行高）：hover 显现，拖动中强调色 */
.sf-ps-grip { position:absolute; right:0; bottom:0; width:14px; height:14px;
  cursor:row-resize; opacity:0; z-index:2; }
.sf-ps-grip::after { content:""; position:absolute; right:2px; bottom:2px;
  width:0; height:0; border-style:solid; border-width:0 0 7px 7px;
  border-color:transparent transparent #666 transparent; }
.sf-ps-tawrap:hover .sf-ps-grip, .sf-ps-grip.active { opacity:1; }
.sf-ps-grip.active::after { border-color:transparent transparent ${"var(--sf-acc, #f66744)"} transparent; }
.sf-ps-btns { flex:0 0 auto; display:flex; flex-direction:column; gap:3px; }
.sf-ps-btn { flex:1 1 0; min-height:0; border:1px solid rgba(255,255,255,0.14);
  background:rgba(255,255,255,0.05); color:#bbb; border-radius:4px;
  cursor:pointer; font-size:10px; line-height:1; padding:0 6px; }
.sf-ps-btn:hover:not(:disabled) { border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.sf-ps-btn:disabled { opacity:0.35; cursor:default; }
.sf-ps-del:hover { border-color:#d65; color:#f88; }
`;
    document.head.appendChild(s);
}

function buildRow(row, index, callbacks) {
    const wrap = document.createElement("div");
    wrap.className = "sf-ps-row";
    if (!row.enabled) wrap.classList.add("off");

    const idx = document.createElement("div");
    idx.className = "sf-ps-idx" + (row.enabled ? "" : " off");
    // 输出 index（从 0 起、跳过关闭行）；空文本/关闭行不参与输出 → 显示占位
    idx.textContent = row.enabled && index >= 0 ? String(index) : "\u2013";
    idx.title = row.enabled && index >= 0 ? `输出 index ${index}` : "当前不参与输出";

    const tg = document.createElement("div");
    tg.className = "sf-ps-tg" + (row.enabled ? "" : " off");
    tg.textContent = row.enabled ? "\u2611" : "\u2610"; // ☑ ☐
    tg.title = row.enabled ? "点击关闭（不输出）" : "点击开启";

    const taWrap = document.createElement("div");
    taWrap.className = "sf-ps-tawrap";
    const ta = document.createElement("textarea");
    ta.className = "sf-ps-ta" + (row.enabled ? "" : " off");
    ta.spellcheck = false;
    ta.rows = 2;
    ta.placeholder = "Type your prompt...";
    ta.value = row.text;
    const grip = document.createElement("div");
    grip.className = "sf-ps-grip";
    grip.title = "拖拽调节本行高度";
    taWrap.append(ta, grip);

    const btns = document.createElement("div");
    btns.className = "sf-ps-btns";
    const mkBtn = (label, act, title, dis) => {
        const b = document.createElement("button");
        b.className = "sf-ps-btn" + (act === "del" ? " sf-ps-del" : "");
        b.textContent = label;
        b.title = title;
        b.tabIndex = -1;
        if (dis) b.disabled = true;
        b.addEventListener("click", (e) => {
            e.stopPropagation();
            callbacks[act]();
        });
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        btns.appendChild(b);
        return b;
    };
    mkBtn("\u25B2", "up", "上移", callbacks.isFirst);
    mkBtn("\u25BC", "down", "下移", callbacks.isLast);
    mkBtn("\u2715", "del", "删除该行", false);

    // 文本变化只写 state（不重建行——防丢焦点/光标）；但行是否参与输出
    // 可能因非空/为空变化，轻量更新本行 index 显示（activeRows 语义）
    ta.addEventListener("input", () => callbacks.text(ta.value));
    ta.addEventListener("keydown", (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey) return; // 放行所有修饰键组合(保存/复制/运行等)
        e.stopPropagation();
    });
    ta.addEventListener("pointerdown", (e) => e.stopPropagation());
    ta.addEventListener("mousedown", (e) => e.stopPropagation());

    tg.addEventListener("click", (e) => {
        e.stopPropagation();
        callbacks.toggle();
    });
    tg.addEventListener("pointerdown", (e) => e.stopPropagation());

    // 右下角角标拖拽调行高：拖拽中只改行高（实时同步节点高度），
    // 结束写 state.rows[i].h（随工作流保存）
    let dragStartY = 0;
    let dragStartH = 0;
    grip.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        grip.setPointerCapture?.(e.pointerId);
        dragStartY = e.clientY;
        dragStartH = rowHOf(row);
        grip.classList.add("active");
    });
    grip.addEventListener("pointermove", (e) => {
        if (!grip.classList.contains("active")) return;
        e.stopPropagation();
        const h = Math.max(MIN_ROW_H, Math.min(MAX_ROW_H, dragStartH + (e.clientY - dragStartY)));
        wrap.style.height = h + "px";
        callbacks.heightDrag(h);
    });
    grip.addEventListener("pointerup", (e) => {
        if (!grip.classList.contains("active")) return;
        e.stopPropagation();
        grip.classList.remove("active");
        const h = parseFloat(wrap.style.height) || dragStartH;
        callbacks.height(h);
    });
    grip.addEventListener("pointercancel", () => {
        grip.classList.remove("active");
        const h = parseFloat(wrap.style.height) || dragStartH;
        callbacks.height(h);
    });

    wrap.append(idx, tg, taWrap, btns);
    callbacks.attachIdx?.(idx);
    return { wrap, ta };
}

function buildRoot(node) {
    injectCSS();
    const root = document.createElement("div");
    root.className = "sf-ps-root";

    const add = document.createElement("button");
    add.className = "sf-ps-add";
    add.textContent = "+ Add Prompt";
    add.addEventListener("click", (e) => {
        e.stopPropagation();
        const st = readState(node);
        st.rows.push({ id: newId(), enabled: true, label: "", text: "" });
        writeState(node, st);
        renderRows();
        // 聚焦新行输入
        const rows = root.querySelectorAll(".sf-ps-ta");
        rows[rows.length - 1]?.focus();
    });
    add.addEventListener("pointerdown", (e) => e.stopPropagation());

    const list = document.createElement("div");
    list.className = "sf-ps-list";
    root.append(add, list);

    function renderRows() {
        const st = readState(node);
        const active = activeRows(st);
        list.innerHTML = "";
        if (!st.rows.length) {
            const empty = document.createElement("div");
            empty.className = "sf-ps-empty";
            empty.textContent = "(no prompts - click + Add Prompt)";
            list.appendChild(empty);
            return;
        }
        const lastIdx = st.rows.length - 1;
        // 拖拽中/结束后同步节点高度：只增不减（内容不被裁；缩小由用户手动）
        function syncNodeHeight() {
            const h = contentHeight(readState(node));
            if (node.size[1] < h) node.size[1] = h;
            node.setDirtyCanvas?.(true, true);
        }
        st.rows.forEach((row, i) => {
            let idxEl = null;
            const { wrap, ta } = buildRow(row, active.indexOf(row), {
                attachIdx: (el) => { idxEl = el; },
                text: (v) => {
                    const s = readState(node);
                    s.rows[i].text = v;
                    writeState(node, s);
                    // 行是否参与输出随文本变化 → 轻量更新本行 index（不重建行）
                    if (idxEl) {
                        const pos = activeRows(s).indexOf(s.rows[i]);
                        idxEl.textContent = s.rows[i].enabled && pos >= 0 ? String(pos) : "\u2013";
                    }
                },
                toggle: () => {
                    const s = readState(node);
                    s.rows[i].enabled = !s.rows[i].enabled;
                    writeState(node, s);
                    renderRows();
                },
                up: () => moveRow(i, -1),
                down: () => moveRow(i, 1),
                del: () => {
                    const s = readState(node);
                    s.rows.splice(i, 1);
                    writeState(node, s);
                    renderRows();
                },
                heightDrag: (h) => {
                    // 拖拽中：实时同步节点高度（行高由 buildRow 内联样式驱动）
                    syncNodeHeight();
                },
                height: (h) => {
                    const s = readState(node);
                    s.rows[i].h = h;
                    writeState(node, s);
                    syncNodeHeight();
                },
                isFirst: i === 0,
                isLast: i === lastIdx,
            });
            wrap.style.height = rowHOf(row) + "px";
            list.appendChild(wrap);
        });
    }

    function moveRow(i, dir) {
        const j = i + dir;
        if (j < 0 || j >= readState(node).rows.length) return;
        const s = readState(node);
        [s.rows[i], s.rows[j]] = [s.rows[j], s.rows[i]];
        writeState(node, s);
        renderRows();
    }

    root._sfPsRender = renderRows;
    return root;
}

function setupNode(node) {
    const root = buildRoot(node);
    const widget = node.addDOMWidget(WIDGET_TYPE, WIDGET_TYPE, root, {
        serialize: false,
        getValue: () => null,
        setValue: () => {},
        getMinHeight: () => contentHeight(readState(node)),
        margin: 4,
    });
    applyAdaptiveCanvasOnly(widget);

    if (typeof node.setSize === "function") node.setSize([440, 220]);
    else { node.size[0] = 440; node.size[1] = 220; }

    root._sfPsRender();
    node._sfPsRoot = root;
}

app.registerExtension({
    name: "sfnodes.PromptStack",

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
            this._sfPsRoot?._sfPsRender();
            return r;
        };

        const origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            if (!isVueNodes()) {
                const minH = contentHeight(readState(this));
                if (size[0] < MIN_W) size[0] = MIN_W;
                if (size[1] < minH) size[1] = minH;
            }
            return origResize?.apply(this, arguments);
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._sfPsRoot = null;
            return origRemoved?.apply(this, arguments);
        };
    },
});

// ── graphToPrompt：注入每节点状态（只注入，从不剪枝；同 SFLoraStack）──
if (!app._sfPromptStackPatched) {
    app._sfPromptStackPatched = true;
    const _origGraphToPrompt = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...args) {
        const result = await _origGraphToPrompt(...args);
        try {
            const out = result?.output;
            if (out) {
                let index = null;
                for (const id in out) {
                    const entry = out[id];
                    if (!entry || entry.class_type !== CLASS) continue;
                    if (!index) index = buildIndex([CLASS]);
                    entry.inputs = entry.inputs || {};
                    const node = findNode(index, id);
                    const st = node ? readState(node) : { version: 1, rows: [] };
                    entry.inputs[HIDDEN_INPUT] = JSON.stringify(promptState(st));
                }
            }
        } catch (e) {
            console.warn("[SF Prompt Stack] could not inject state:", (e && e.message) || e);
        }
        return result;
    };
}
