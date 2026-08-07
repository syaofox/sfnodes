// ==========================================================================
// sf_pause_text.js - SFPauseText 主扩展
// ==========================================================================
//
// 复刻 Pixaroma Pause Text（精简 shared 依赖版）：
//   - 节点体：状态条 + 可编辑文本框 + Pause/Pass/Keep 切换 + Copy/Revert +
//     Regenerate/Continue 按钮（web/sf_pause_text_ui.js）
//   - 状态存 node.properties.pauseTextState（随工作流保存，见 sf_pause_text_lib.js）
//   - 队列语义（双钩子，与 SFPromptTags 同款 Switch 拆分）：
//       * app.graphToPrompt 只 INJECT（Export/分享/保存按钮也会触发，删节点
//         会把导出静默截断）；注入 {mode, text} 到隐藏 PauseState
//       * api.queuePrompt 提交时才 PRUNE（单浏览器提交漏斗）：
//         pause 删下游；continue 跳上游模型链；pass 不剪
//   - 一次性提交模式 _sfPauseTextSubmitMode：Continue/Regenerate 按钮先挂模式
//     再 queuePrompt，剪枝钩子读取；finally 清除
//   - executed 事件接收 Python 的模型文本（sf_pause_text ui 键）回填盒子
//   - Regenerate：沿上游连线回溯，把每个数字型 seed widget 滚成新随机值
//
// 与原件差异（已确认范围）：无 accent 颜色设置、状态条不做 canvas 绘制/
// slot 行浮动（普通 DOM 行）、无 resize floor / canvas zoom 辅助。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import {
    getState, setGate, setText, setModelText, revertText, STATE_PROP,
} from "./sf_pause_text_lib.js";
import { applyGateMode } from "./sf_pause_text_lib.js";
import {
    buildPauseTextWidget, renderPause, syncText, flashIcon, NODE_MIN_W, nodeMinH,
} from "./sf_pause_text_ui.js";

const CLASS = "SFPauseText";
const HIDDEN_INPUT = "PauseState";
const WIDGET_TYPE = "sf_pause_text_ui";

// ── 内联 shared 辅助 ────────────────────────────────────────────────────
// Nodes 2.0（Vue）渲染器判定，由设置 Comfy.VueNodes.Enabled 驱动；实时读取，
// 运行时切换渲染器也尊重
function isVueNodes() {
    return !!window.LiteGraph?.vueNodesMode;
}
// adaptive canvasOnly：legacy 下 true（不进 Parameters tab），Nodes 2.0 下 false
// （否则 Vue 根本不渲染该 widget）。实时 getter，渲染时求值
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
// "continue" -> 剪掉上游，把编辑文本送下游；"pause" -> 剪掉下游（停在闸门），
// 重新捕获模型文本
async function queueWithMode(node, mode) {
    // 同一时刻只允许这一个闸门携带一次性提交模式
    const allNodes = app.graph?._nodes || app.graph?.nodes || [];
    for (const n of allNodes) {
        if (n !== node && n._sfPauseTextSubmitMode) n._sfPauseTextSubmitMode = null;
    }
    node._sfPauseTextSubmitMode = mode;
    node._sfPauseTextBusy = mode === "continue" ? "Continuing…" : "Regenerating…";
    renderPause(node);
    try {
        await app.queuePrompt(0, 1);
    } catch (err) {
        console.error("[SF Pause Text] queue failed", err);
    } finally {
        node._sfPauseTextSubmitMode = null;
        node._sfPauseTextBusy = null;
        renderPause(node);
    }
}

// 短暂状态消息，2.5s 后清除（Copy / 无种子反馈）
function flash(node, msg) {
    node._sfPauseTextFlash = msg;
    renderPause(node);
    clearTimeout(node._sfPauseTextFlashTimer);
    node._sfPauseTextFlashTimer = setTimeout(() => {
        node._sfPauseTextFlash = null;
        renderPause(node);
    }, 2500);
}

// ── Regenerate：滚上游种子，然后以 Pause 模式重跑 ──
function getLink(graph, linkId) {
    if (linkId == null) return null;
    let link = graph.links?.[linkId];
    if (!link && typeof graph.links?.get === "function") link = graph.links.get(linkId);
    return link;
}

// 名称含 "seed" 的数字 widget（跳过 "control_after_generate" combo，其值非数字）
function isSeedWidget(w) {
    return !!(w && typeof w.name === "string" && /seed/i.test(w.name) && typeof w.value === "number");
}

function setRandomSeed(node, w) {
    let max = 0xffffffff;
    if (w.options && Number.isFinite(w.options.max)) max = Math.min(w.options.max, Number.MAX_SAFE_INTEGER);
    let min = 0;
    if (w.options && Number.isFinite(w.options.min)) min = w.options.min;
    const span = Math.max(1, Math.min(max - min, 0xffffffff));
    const val = Math.floor(min + Math.random() * span);
    w.value = val;
    try { w.callback?.(val, app.canvas, node); } catch { /* ignore */ }
}

// 从本节点 text 输入沿活图回溯，随机化沿途每个 seed widget（visited 集 +
// 深度上限）。返回滚了多少个种子，调用方可提示"没有种子"。
function randomizeUpstreamSeeds(node) {
    const graph = node.graph;
    if (!graph) return 0;
    const seen = new Set();
    const stack = [];
    const MAX_DEPTH = 50;
    for (const inp of node.inputs || []) {
        if (inp.name !== "text" || inp.link == null) continue;
        const l = getLink(graph, inp.link);
        if (l && l.origin_id != null) stack.push({ id: l.origin_id, depth: 0 });
    }
    let count = 0;
    while (stack.length) {
        const { id, depth } = stack.pop();
        const key = String(id);
        if (seen.has(key) || depth > MAX_DEPTH) continue;
        seen.add(key);
        const n = graph.getNodeById(id);
        if (!n) continue;
        for (const w of n.widgets || []) {
            if (isSeedWidget(w)) { setRandomSeed(n, w); count++; }
        }
        for (const ni of n.inputs || []) {
            if (ni.link == null) continue;
            const l = getLink(graph, ni.link);
            if (l && l.origin_id != null) stack.push({ id: l.origin_id, depth: depth + 1 });
        }
    }
    if (count) graph.setDirtyCanvas?.(true, true);
    return count;
}

async function regenerate(node) {
    const rolled = randomizeUpstreamSeeds(node);
    if (!rolled) flash(node, "No seed found upstream - text may be unchanged");
    await queueWithMode(node, "pause");
}

// 复制盒子文本到系统剪贴板
async function copyText(node) {
    const txt = getState(node).text || "";
    if (!txt) { flash(node, "Nothing to copy"); return; }
    try {
        if (!navigator.clipboard?.writeText) throw new Error("no clipboard");
        await navigator.clipboard.writeText(txt);
        flashIcon(node._sfPauseTextEls?.copyBtn);
        flash(node, "Copied to clipboard");
    } catch {
        flash(node, "Could not copy to clipboard");
    }
}

function setupNode(node) {
    const root = buildPauseTextWidget(node, {
        onGate: (gate) => { setGate(node, gate); renderPause(node); },
        onInput: (val) => { setText(node, val); renderPause(node); },
        onContinue: () => queueWithMode(node, "continue"),
        onRegenerate: () => regenerate(node),
        onCopy: () => copyText(node),
        onRevert: () => { revertText(node); syncText(node); renderPause(node); flashIcon(node._sfPauseTextEls?.revertBtn); },
    });
    const widget = node.addDOMWidget(WIDGET_TYPE, WIDGET_TYPE, root, {
        serialize: false,
        getMinHeight: () => nodeMinH(),
    });
    applyAdaptiveCanvasOnly(widget);

    // 新节点默认尺寸——开大（通常装一段提示词）。用 setSize（非裸写 node.size）
    // 让 DOM widget 宽度真正传播。无条件设置：configure() 在 onNodeCreated 之后
    // 运行并恢复已保存尺寸，所以这只对全新节点生效。
    if (typeof node.setSize === "function") node.setSize([480, 520]);
    else { node.size[0] = 480; node.size[1] = 520; }

    // 延迟首次渲染直到 node.properties 恢复（与 SFPromptTags 同款）
    queueMicrotask(() => restore(node));
}

// 纯 DOM 恢复：重渲染控件 + 把存储文本推进盒子。绝不改动序列化状态，
// 加载路径安全
function restore(node) {
    renderPause(node);
    syncText(node);
}

app.registerExtension({
    name: "sfnodes.PauseText",

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

        // 自愈最小尺寸（与 getMinHeight 双保险）。只抬升过小的尺寸，
        // 已保存（>= min）的尺寸永不变更 -> 不脏加载
        const origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            // LEGACY ONLY：Nodes 2.0 的渲染尺寸在 Vue 布局 store 里而非 node.size，
            // 在那里钳 node.size 会失步（Vue 渲染拖拽尺寸而 node.size 是钳后值）
            if (!isVueNodes()) {
                if (size[0] < NODE_MIN_W) size[0] = NODE_MIN_W;
                if (size[1] < nodeMinH()) size[1] = nodeMinH();
            }
            return origResize?.apply(this, arguments);
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            clearTimeout(this._sfPauseTextFlashTimer);
            this._sfPauseTextEls = null;
            return origRemoved?.apply(this, arguments);
        };
    },
});

// ── executed 事件：接收 Python 的模型文本，填进盒子 ──
// Python 只在有线 Pause/Pass（一次新鲜模型捕获）时 emit sf_pause_text，
// 所以收到就代表"用新文本替换盒子"
api.addEventListener("executed", (e) => {
    const d = e.detail;
    const payload = d?.output?.sf_pause_text;
    if (!payload || !payload.length) return;
    let node = app.graph.getNodeById(d.node);
    if (!node && typeof d.node === "string") node = app.graph.getNodeById(parseInt(d.node, 10));
    if (!node || node.comfyClass !== CLASS) return;
    const text = typeof payload[0] === "string" ? payload[0] : String(payload[0] ?? "");
    setModelText(node, text);   // 替换盒子 + revert 基线
    syncText(node);
    renderPause(node);
});

// ── app.graphToPrompt hook：注入模式/文本（Pattern #9）──
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

// isOutput(classType)：class_type 是否为 OUTPUT_NODE。从实时节点定义读取。
// 注册表缺失 -> null -> applyGateMode 回退为删一切（安全：上游仍被跳过）
function makeIsOutput() {
    const reg = window.LiteGraph?.registered_node_types;
    if (!reg) return null;
    return (classType) => !!(classType && reg[classType]?.nodeData?.output_node);
}

// 当前编辑的盒子文本：优先活 textarea，否则存储状态
function editedTextOf(node) {
    const live = node._sfPauseTextEls?.ta?.value;
    if (typeof live === "string") return live;
    return getState(node).text;
}

// CONTINUE 闸门会把自己的下游分支剪掉，可能删掉位于其上游的另一个闸门——
// 所以 continue 必须先于 pause/pass 处理
const MODE_RANK = { continue: 0, pause: 1, pass: 2 };

// prompt 中每个 Pause Text 条目及其应生效的模式 + 编辑文本。两个钩子共用，
// INJECT 与 PRUNE 永远不会对闸门在做什么有分歧
function collectGates(out) {
    let index = null;
    const gates = [];
    for (const id in out) {
        const entry = out[id];
        if (!entry || entry.class_type !== CLASS) continue;
        if (!index) index = buildNodeIndex();
        const node = findNode(index, id);
        const submit = node?._sfPauseTextSubmitMode;
        let mode;
        if (submit === "continue" || submit === "pause") {
            mode = submit;
        } else if (node) {
            // 普通 Run：切换开关决定。Keep 每次 run 都像 Continue（跳模型、
            // 复用当前文本、让下游出图）——它是转成持久模式的 Continue
            const g = node.properties?.[STATE_PROP]?.gate;
            mode = g === "pass" ? "pass" : g === "keep" ? "continue" : "pause";
        } else {
            // 解析不到活节点：默认无害的 "pass"（不剪）而不是破坏性的
            // "pause"（会截断工作流）
            mode = "pass";
        }
        const editedText = node ? editedTextOf(node) : "";
        gates.push({ id, entry, mode, editedText });
    }
    return gates;
}

// ─────────────────────────────────────────────────────────────────────────
// 双钩子、两个职责——Switch 拆分（与 SFPromptTags 相同推理）：graphToPrompt 会
// 因 Export/分享/保存按钮运行，只能 INJECT；在那里删节点会把导出静默截断。
// 剪枝移到 api.queuePrompt——唯一的浏览器提交漏斗。
//
// _sfPauseTextSubmitMode 与活 textarea 在剪枝时仍可读：一次性模式在
// `await app.queuePrompt(...)` 解析后的 finally 里清除，而 api.queuePrompt
// 就发生在该 await 之内。
// ─────────────────────────────────────────────────────────────────────────
if (!app._sfPauseTextPatched) {
    app._sfPauseTextPatched = true;
    const _origGraphToPrompt = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...args) {
        const result = await _origGraphToPrompt(...args);
        // FAIL OPEN：这里抛错会拒绝 ComfyUI 自己的 graphToPrompt、弄坏整个
        // 工作流的 Run。绝不包住上面的 await；核心失败必须传播。
        try {
            const out = result?.output;
            if (out) {
                for (const g of collectGates(out)) {
                    g.entry.inputs = g.entry.inputs || {};
                    g.entry.inputs[HIDDEN_INPUT] = JSON.stringify({
                        mode: g.mode, text: g.editedText,
                    });
                }
            }
        } catch (e) {
            console.error("[sfnodes] Pause Text prompt injection failed; prompt sent unchanged", e);
        }
        return result;
    };
}

// 提交时剪枝。api.queuePrompt(number, {output, workflow}, options) 是所有浏览器
// run 的唯一漏斗（普通 Run、局部"执行节点"、批量队列）。原样转发 ...args，
// partialExecutionTargets 与未来选项都能存活。
if (!api._sfPauseTextQueueWrapped) {
    api._sfPauseTextQueueWrapped = true;
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
                    applyGateMode(out, g.id, g.entry, g.mode, isOutput, HIDDEN_INPUT, {
                        inputKey: "text",
                        editedText: g.editedText,
                    });
                }
            }
        } catch (err) {
            // 剪枝失败绝不能挡住用户的 run
            console.error("[SF Pause Text] submit-time prune failed", err);
        }
        return _origQueuePrompt(...args);
    };
}
