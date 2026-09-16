// ==========================================================================
// sf_track_data_merge.js - SFTrackDataMerge 节点（逐槽 -/+ 模式单节点加减）
// ==========================================================================
//
// 一个基础 track_data + 动态槽 track_1..N（复用 sf_dynamic_slots），节点体
// DOM 行列表逐槽切换模式：`-` 逐对象相减（保留对象数）/ `+` 并集塌单身份。
//
// 模式状态存 node.properties.sfTrackMergeModes（随工作流保存），提交时经
// graphToPrompt 钩子注入 hidden STRING SlotModes（与 SFPromptStack 同模式）。
// 纯逻辑（解析/切换/序列化）在 sf_track_data_merge_lib.js，本文件只做 UI。
// ==========================================================================

import { app } from "/scripts/app.js";
import { applyAdaptiveCanvasOnly, el, injectCSSOnce } from "./sf_common.js";
import { installConfiguredSlotRecovery, installDynamicSlots } from "./sf_dynamic_slots.js";
import {
    MODE_ADD,
    SLOT_PREFIX,
    collectSlotNames,
    getMode,
    parseModes,
    serializeModes,
    setMode,
    toggleMode,
} from "./sf_track_data_merge_lib.js";

const CLASS = "SFTrackDataMerge";
const WIDGET_TYPE = "sf_track_data_merge_ui";
const HIDDEN_INPUT = "SlotModes";
const PROP = "sfTrackMergeModes";

const INITIAL_INPUTS = 4;
const MAX_INPUTS = 20;
const SLOT_TYPE = "MASK,SAM3_TRACK_DATA";
// 排除基础槽 `track_data`（同以 track_ 开头，不能用裸前缀匹配）
const SLOT_MATCH = (name) => /^track_\d+$/.test(name);

const ROW_H = 22;
const HINT_H = 16;
const PAD = 6;

function readState(node) {
    return parseModes(node && node.properties ? node.properties[PROP] : null);
}

function writeState(node, modes) {
    if (!node.properties) node.properties = {};
    node.properties[PROP] = parseModes(modes);
}

function contentHeight(node) {
    const n = collectSlotNames(node.inputs).length;
    return PAD * 2 + Math.max(n, 1) * ROW_H + HINT_H;
}

function injectCSS() {
    injectCSSOnce("sf-tdm-css", `
.sf-tdm-root { box-sizing:border-box; width:100%; padding:${PAD}px;
  background:var(--sf-panel-bg); border-radius:4px; color:var(--sf-text);
  font:12px sans-serif; }
.sf-tdm-row { display:flex; align-items:center; gap:6px; height:${ROW_H - 2}px; }
.sf-tdm-row + .sf-tdm-row { margin-top:2px; }
.sf-tdm-name { flex:1 1 auto; font:11px monospace; color:var(--sf-text-dim);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-tdm-btn { flex:0 0 26px; height:18px; border:0; border-radius:4px; cursor:pointer;
  font:700 13px/1 'Segoe UI',sans-serif; color:#fff; padding:0; }
.sf-tdm-btn.sub { background:var(--sf-negative, #e05a4a); }
.sf-tdm-btn.add { background:var(--sf-positive, #4caf50); }
.sf-tdm-btn:hover { filter:brightness(1.1); }
.sf-tdm-empty { height:${ROW_H}px; display:flex; align-items:center; justify-content:center;
  color:var(--sf-text-faint); font-size:11px; user-select:none; }
.sf-tdm-hint { margin-top:4px; font-size:10px; color:var(--sf-text-faint);
  user-select:none; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
`);
}

// prompt key（节点 id 字符串，子图可能为 "5:10"）→ 图节点。
function findNode(id) {
    const key = String(id);
    const byId = app.graph?.getNodeById?.(Number(key));
    if (byId) return byId;
    const nodes = app.graph?._nodes || [];
    return nodes.find((n) => n && String(n.id) === key)
        || nodes.find((n) => n && key.endsWith(":" + n.id)) || null;
}

function render(node) {
    const list = node._sfTdmList;
    if (!list) return;
    list.innerHTML = "";
    const modes = readState(node);
    const names = collectSlotNames(node.inputs);
    if (!names.length) {
        list.appendChild(el("div", "sf-tdm-empty", "无输入槽"));
        return;
    }
    for (const name of names) {
        const row = el("div", "sf-tdm-row");
        row.appendChild(el("span", "sf-tdm-name", name));
        const mode = getMode(modes, name);
        const btn = el("button", "sf-tdm-btn " + (mode === MODE_ADD ? "add" : "sub"),
            mode === MODE_ADD ? "+" : "-");
        btn.type = "button";
        btn.title = mode === MODE_ADD
            ? "当前：并集合并（单身份）· 点击改为逐对象排除"
            : "当前：逐对象排除（保留对象数）· 点击改为并集合并";
        btn.addEventListener("click", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            const current = getMode(readState(node), name);
            writeState(node, setMode(readState(node), name, toggleMode(current)));
            render(node);
        });
        row.appendChild(btn);
        list.appendChild(row);
    }
}

function setupNode(node) {
    injectCSS();
    installDynamicSlots(node, {
        inputPrefix: SLOT_PREFIX,
        inputStart: 1,
        inputCount: MAX_INPUTS,
        inputType: SLOT_TYPE,
        initialInputs: INITIAL_INPUTS,
        inputMatch: SLOT_MATCH,
    });
    installConfiguredSlotRecovery(node, {
        inputPrefix: SLOT_PREFIX,
        inputStart: 1,
        inputType: SLOT_TYPE,
        initialInputs: INITIAL_INPUTS,
        inputCount: MAX_INPUTS,
        inputMatch: SLOT_MATCH,
    });

    const root = el("div", "sf-tdm-root");
    const list = el("div", "sf-tdm-list");
    root.appendChild(list);
    root.appendChild(el("div", "sf-tdm-hint", "- 排除（保留对象数）  + 合并（单身份）"));
    node._sfTdmRoot = root;
    node._sfTdmList = list;

    if (typeof node.addDOMWidget === "function") {
        const widget = node.addDOMWidget(WIDGET_TYPE, WIDGET_TYPE, root, {
            serialize: false,
            getValue: () => null,
            setValue: () => {},
            getMinHeight: () => contentHeight(node),
            margin: 4,
        });
        applyAdaptiveCanvasOnly(widget);
    } else {
        root.remove();
        node._sfTdmList = null;
    }

    // 追加 DOM 行列表后让节点高度容纳（保留当前宽度；computeSize 早期可能不可用）
    try {
        const sz = node.computeSize?.();
        if (sz && node.size) {
            node.setSize([node.size[0] || sz[0], Math.max(node.size[1] || 0, sz[1])]);
        }
    } catch (e) {
        // 忽略：节点尺寸由前端后续布局主导
    }

    const wrap = (name) => {
        const orig = node[name];
        node[name] = function (...args) {
            const r = typeof orig === "function" ? orig.apply(this, args) : undefined;
            render(this);
            return r;
        };
    };
    wrap("onConnectionsChange");
    wrap("onAfterGraphConfigured");
    wrap("onConfigure");

    render(node);
}

app.registerExtension({
    name: "sfnodes.TrackDataMerge",

    nodeCreated(node) {
        if (node.comfyClass !== CLASS) return;
        try {
            setupNode(node);
        } catch (e) {
            console.warn("[SF Track Data Merge] setup failed:", (e && e.message) || e);
        }
    },
});

// ── graphToPrompt：注入每节点槽模式（只注入，从不剪枝；同 SFPromptStack）──
if (!app._sfTrackMergePatched) {
    app._sfTrackMergePatched = true;
    const _origGraphToPrompt = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...args) {
        const result = await _origGraphToPrompt(...args);
        try {
            const out = result?.output;
            if (out) {
                for (const id in out) {
                    const entry = out[id];
                    if (!entry || entry.class_type !== CLASS) continue;
                    const node = findNode(id);
                    const modes = node ? readState(node) : {};
                    entry.inputs = entry.inputs || {};
                    entry.inputs[HIDDEN_INPUT] = serializeModes(modes);
                }
            }
        } catch (e) {
            console.warn("[SF Track Data Merge] could not inject SlotModes:", (e && e.message) || e);
        }
        return result;
    };
}
