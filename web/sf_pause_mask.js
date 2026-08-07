// ==========================================================================
// sf_pause_mask.js - SFPauseMask 主扩展
// ==========================================================================
//
// 基于 SFPauseImage 的闸门模式扩展（MASK 类型，输入键 "mask"）：
//   - 节点体：Pause/Pass 切换 + 状态行 + Regenerate/Continue + Copy/Save
//     Disk/Save Output/Open + 遮罩灰度预览（web/sf_pause_mask_ui.js）
//   - 快照机制：Pause 时 Python 把首帧存成灰度 PNG（temp 目录按节点 id
//     命名），Continue 时前端把上游剪出 prompt、Python 读回快照
//   - 队列语义（双钩子）：graphToPrompt 只 INJECT {mode}；api.queuePrompt
//     提交时才 PRUNE（prune 复用 sf_pause_text_lib.js::applyGateMode，
//     inputKey "mask"）
//   - executed 事件接收 Python 的快照 frame（sf_pause_mask_frame）回填预览
//   - Save 链路复用 /api/sfnodes/preview/{save,prepare}（save 嵌 workflow/prompt）
//
// 与 SFPauseImage 的差异：CLASS/输入键/frame 键/state 与 ui 模块；其余机制
// （双钩子、Save、Copy/Open、一次性模式）同构。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { getState, setGate, STATE_PROP } from "./sf_pause_mask_lib.js";
import { applyGateMode } from "./sf_pause_text_lib.js";
import {
    buildPauseWidget, renderPause, showFrame, frameViewUrl, NODE_MIN_W, NODE_MIN_H,
} from "./sf_pause_mask_ui.js";

const CLASS = "SFPauseMask";
const HIDDEN_INPUT = "PauseState";
const WIDGET_TYPE = "sf_pause_mask_ui";

// ── 内联 shared 辅助 ────────────────────────────────────────────────────
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
                return !isVueNodes();
            },
        });
    } catch { /* ignore */ }
    return widget;
}

// ── 队列：带一次性提交模式跑一次 run ──
async function queueWithMode(node, mode) {
    const allNodes = app.graph?._nodes || app.graph?.nodes || [];
    for (const n of allNodes) {
        if (n !== node && n._sfPauseMaskSubmitMode) n._sfPauseMaskSubmitMode = null;
    }
    node._sfPauseMaskSubmitMode = mode;
    node._sfPauseMaskBusy = mode === "continue" ? "Continuing…" : "Regenerating…";
    renderPause(node);
    try {
        await app.queuePrompt(0, 1);
    } catch (err) {
        console.error("[SF Pause Mask] queue failed", err);
    } finally {
        node._sfPauseMaskSubmitMode = null;
        node._sfPauseMaskBusy = null;
        renderPause(node);
    }
}

function flash(node, msg) {
    node._sfPauseMaskFlash = msg;
    renderPause(node);
    clearTimeout(node._sfPauseMaskFlashTimer);
    node._sfPauseMaskFlashTimer = setTimeout(() => {
        node._sfPauseMaskFlash = null;
        renderPause(node);
    }, 2000);
}

// 把预览的快照复制到系统剪贴板（PNG）
async function copySnapshot(node) {
    const frame = getState(node).frame;
    if (!frame?.filename) { flash(node, "Run once to capture a mask"); return; }
    if (!navigator.clipboard?.write || typeof ClipboardItem === "undefined") {
        flash(node, "Clipboard not supported here");
        return;
    }
    try {
        const resp = await fetch(frameViewUrl(frame));
        if (!resp.ok) {
            flash(node, resp.status === 404 ? "Snapshot expired - run again" : "Copy failed");
            return;
        }
        const blob = await resp.blob();
        const png = blob.type === "image/png" ? blob : new Blob([blob], { type: "image/png" });
        await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
        flash(node, "Copied to clipboard");
    } catch (err) {
        if (err?.name === "NotAllowedError") { flash(node, "Click the page, then Copy again"); return; }
        flash(node, "Copy failed");
    }
}

// 在新标签页打开预览的快照（全屏查看）
function openSnapshot(node) {
    const frame = getState(node).frame;
    if (!frame?.filename) { flash(node, "Run once to capture a mask"); return; }
    const win = window.open(frameViewUrl(frame), "_blank", "noopener");
    if (!win) flash(node, "Popup blocked");
}

// ── Save（走 sfnodes 后端路由 /api/sfnodes/preview/*，复用 pause_image 路由）──
const SAVE_PREFIX = "PauseMask";

async function snapshotDataURL(node) {
    const frame = getState(node).frame;
    if (!frame?.filename) throw new Error("nosnap");
    const resp = await fetch(frameViewUrl(frame));
    if (!resp.ok) throw new Error(resp.status === 404 ? "expired" : "fetch");
    const blob = await resp.blob();
    return await new Promise((res, rej) => {
        const r = new FileReader();
        r.onload = () => res(r.result);
        r.onerror = () => rej(new Error("read"));
        r.readAsDataURL(blob);
    });
}

// 优先用快照捕获时的执行期工作流；无则回退活图
async function resolveSaveMeta(node) {
    const m = node._sfPauseMaskExecMeta;
    if (m && m.workflow) return { workflow: m.workflow, prompt: m.prompt };
    const { workflow, output } = await app.graphToPrompt();
    return { workflow, prompt: output };
}

function saveErr(node, err) {
    if (err?.message === "expired") flash(node, "Snapshot expired - run again");
    else if (err?.message === "nosnap") flash(node, "Run once to capture a mask");
    else flash(node, "Save failed");
}

// 存到 ComfyUI output/ 目录（嵌入工作流）
async function saveToOutput(node) {
    if (!node._sfPauseMaskHasSnapshot) { flash(node, "Run once to capture a mask"); return; }
    try {
        const image_b64 = await snapshotDataURL(node);
        const { workflow, prompt } = await resolveSaveMeta(node);
        const resp = await fetch(api.apiURL("/api/sfnodes/preview/save"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_b64, filename_prefix: SAVE_PREFIX, workflow, prompt }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) { flash(node, `Save failed: ${data.error || resp.status}`); return; }
        flash(node, `Saved: ${data.filename}`);
    } catch (err) { saveErr(node, err); }
}

// 存到用户选择的目录（showSaveFilePicker 优先，<a download> 回退）
async function saveToDisk(node) {
    if (!node._sfPauseMaskHasSnapshot) { flash(node, "Run once to capture a mask"); return; }
    let preparedBlob;
    let suggestedName = `${SAVE_PREFIX}.png`;
    try {
        const image_b64 = await snapshotDataURL(node);
        const { workflow, prompt } = await resolveSaveMeta(node);
        const resp = await fetch(api.apiURL("/api/sfnodes/preview/prepare"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_b64, filename_prefix: SAVE_PREFIX, workflow, prompt }),
        });
        if (!resp.ok) {
            const e = await resp.json().catch(() => ({}));
            flash(node, `Save failed: ${e.error || resp.status}`);
            return;
        }
        const data = await resp.json();
        if (data.suggested_filename) suggestedName = data.suggested_filename;
        preparedBlob = await (await fetch(data.image_b64)).blob();
    } catch (err) { saveErr(node, err); return; }

    if (typeof window.showSaveFilePicker === "function") {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName,
                types: [{ description: "PNG image", accept: { "image/png": [".png"] } }],
            });
            const w = await handle.createWritable();
            await w.write(preparedBlob);
            await w.close();
            flash(node, `Saved: ${handle.name}`);
        } catch (err) {
            if (err?.name === "AbortError") return;
            flash(node, "Save failed");
        }
        return;
    }
    const url = URL.createObjectURL(preparedBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = suggestedName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1500);
    flash(node, "Saved to Downloads");
}

function setupNode(node) {
    const root = buildPauseWidget(node, {
        onGate: (gate) => { setGate(node, gate); renderPause(node); },
        onContinue: () => queueWithMode(node, "continue"),
        onRegenerate: () => queueWithMode(node, "pause"),
        onCopy: () => copySnapshot(node),
        onSaveDisk: () => saveToDisk(node),
        onSaveOutput: () => saveToOutput(node),
        onOpen: () => openSnapshot(node),
    });
    const widget = node.addDOMWidget(WIDGET_TYPE, WIDGET_TYPE, root, {
        serialize: false,
        getMinHeight: () => NODE_MIN_H,
    });
    applyAdaptiveCanvasOnly(widget);

    // 新节点默认尺寸（configure 之后恢复已保存尺寸，只对全新节点生效）
    if (!node.size || node.size[0] < NODE_MIN_W) node.size[0] = 400;
    if (!node.size || node.size[1] < NODE_MIN_H) node.size[1] = 400;

    queueMicrotask(() => restore(node));
}

// 纯 DOM 恢复：重渲染控件 + 重载最近快照（如有）
function restore(node) {
    renderPause(node);
    const s = getState(node);
    if (s.frame) showFrame(node, s.frame);
}

app.registerExtension({
    name: "sfnodes.PauseMask",

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
            restore(this);
            return r;
        };

        const origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            if (size[0] < NODE_MIN_W) size[0] = NODE_MIN_W;
            if (size[1] < NODE_MIN_H) size[1] = NODE_MIN_H;
            return origResize?.apply(this, arguments);
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            clearTimeout(this._sfPauseMaskFlashTimer);
            this._sfPauseMaskEls = null;
            return origRemoved?.apply(this, arguments);
        };
    },
});

// ── executed 事件：接收 Python 的快照预览 frame，回填预览 ──
api.addEventListener("executed", (e) => {
    const d = e.detail;
    const frames = d?.output?.sf_pause_mask_frame;
    if (!frames || !frames.length) return;
    let node = app.graph.getNodeById(d.node);
    if (!node && typeof d.node === "string") node = app.graph.getNodeById(parseInt(d.node, 10));
    if (!node || node.comfyClass !== CLASS) return;
    const f = frames[0];
    // 捕获执行期工作流供 Save 按钮（仅运行时，绝不进 node.properties）
    if (f._sf_pause_meta) node._sfPauseMaskExecMeta = f._sf_pause_meta;
    const s = getState(node);
    s.frame = { filename: f.filename, subfolder: f.subfolder || "", type: f.type || "temp" };
    showFrame(node, s.frame);
});

// ── app.graphToPrompt hook：注入模式（Pattern #9，Switch 拆分）──
function buildNodeIndex() {
    const index = new Map();
    const visit = (graph) => {
        if (!graph) return;
        const nodes = graph._nodes || graph.nodes || [];
        for (const n of nodes) {
            if (!n) continue;
            if (n.comfyClass === CLASS || n.type === CLASS) index.set(String(n.id), n);
            const inner = n.subgraph || n.graph || n._graph;
            if (inner && inner !== graph) visit(inner);
        }
    };
    visit(app.graph);
    return index;
}

function findNode(index, promptId) {
    const sId = String(promptId);
    if (index.has(sId)) return index.get(sId);
    const tail = sId.includes(":") ? sId.slice(sId.lastIndexOf(":") + 1) : null;
    if (tail && index.has(tail)) return index.get(tail);
    return null;
}

function makeIsOutput() {
    const reg = window.LiteGraph?.registered_node_types;
    if (!reg) return null;
    return (classType) => !!(classType && reg[classType]?.nodeData?.output_node);
}

// continue 必须先于 pause/pass 处理（链式闸门）
const MODE_RANK = { continue: 0, pause: 1, pass: 2 };

function collectGates(out) {
    let index = null;
    const gates = [];
    for (const id in out) {
        const entry = out[id];
        if (!entry || entry.class_type !== CLASS) continue;
        if (!index) index = buildNodeIndex();
        const node = findNode(index, id);
        const submit = node?._sfPauseMaskSubmitMode;
        let mode;
        if (submit === "continue" || submit === "pause") {
            mode = submit;
        } else if (node) {
            mode = node.properties?.[STATE_PROP]?.gate === "pass" ? "pass" : "pause";
        } else {
            mode = "pass";   // 解析不到活节点：无害的 pass（不剪）
        }
        gates.push({ id, entry, mode });
    }
    return gates;
}

// ─────────────────────────────────────────────────────────────────────────
// 双钩子、两个职责（与 SFPauseImage 相同推理）：graphToPrompt 只 INJECT
// （Export/分享/保存按钮也会触发，剪枝会截断导出）；api.queuePrompt 提交时
// 才 PRUNE。一次性模式在 finally 里清除，剪枝时仍可读。
// ─────────────────────────────────────────────────────────────────────────
if (!app._sfPauseMaskPatched) {
    app._sfPauseMaskPatched = true;
    const _origGraphToPrompt = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...args) {
        const result = await _origGraphToPrompt(...args);
        // FAIL OPEN：这里抛错会弄坏整个工作流的 Run
        try {
            const out = result?.output;
            if (out) {
                for (const g of collectGates(out)) {
                    g.entry.inputs = g.entry.inputs || {};
                    g.entry.inputs[HIDDEN_INPUT] = JSON.stringify({ mode: g.mode });
                }
            }
        } catch (e) {
            console.error("[sfnodes] Pause Mask prompt injection failed; prompt sent unchanged", e);
        }
        return result;
    };
}

if (!api._sfPauseMaskQueueWrapped) {
    api._sfPauseMaskQueueWrapped = true;
    const _origQueuePrompt = api.queuePrompt.bind(api);
    api.queuePrompt = async function (...args) {
        try {
            const out = args[1]?.output;
            if (out) {
                const isOutput = makeIsOutput();
                const gates = collectGates(out);
                gates.sort((a, b) => MODE_RANK[a.mode] - MODE_RANK[b.mode]);
                for (const g of gates) {
                    if (!out[g.id]) continue;  // 已被更早的 continue 闸门剪掉
                    // prune 复用 sf_pause_text_lib.js::applyGateMode，输入键 "mask"
                    applyGateMode(out, g.id, g.entry, g.mode, isOutput, HIDDEN_INPUT, {
                        inputKey: "mask",
                        editedText: "",
                    });
                }
            }
        } catch (err) {
            console.error("[SF Pause Mask] submit-time prune failed", err);
        }
        return _origQueuePrompt(...args);
    };
}
