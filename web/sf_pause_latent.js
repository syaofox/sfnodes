// ==========================================================================
// sf_pause_latent.js - SFPauseLatent 主扩展
// ==========================================================================
//
// 基于 SFPauseImage 的闸门模式扩展（LATENT 类型，输入键 "latent"）：
//   - 节点体：Pause/Pass 切换 + 状态行 + Regenerate/Continue + Copy/Save
//     Disk/Save Output/Open + 预览图（web/sf_pause_latent_ui.js）
//   - 快照机制：Pause 时 Python 把中间态 latent 整 batch 存成 safetensors
//     （temp 目录按节点 id 命名），Continue 时前端把第一段采样（整条上游链）
//     剪出 prompt、Python 读回快照——只有下游运行，第二段采样器从暂停时那份
//     精确的中间态继续，第一段完全不重跑
//   - 预览：image 输入（VAEDecode 结果）只在 Pause 时在场；Continue 时前端
//     连同 latent 链接一并剪掉它（extraInputKeys），否则 VAEDecode 仍被消费
//     会把第一段采样器拉活；Python 回传预览 png 快照显示同一张图
//   - 队列语义（双钩子，同 SFPauseImage 同款 Switch 拆分）：graphToPrompt 只
//     INJECT {mode}；api.queuePrompt 提交时才 PRUNE（prune 复用
//     sf_pause_text_lib.js::applyGateMode，inputKey "latent"）
//   - executed 事件接收 Python 的预览 frame（sf_pause_latent_frame）回填
//   - Save 链路复用 /api/sfnodes/preview/{save,prepare}（save 嵌 workflow/prompt）
//
// 与 SFPauseImage 的差异：CLASS/输入键（latent + image 预览）/frame 键/
// state 与 ui 模块；其余机制（双钩子、Save、Copy/Open、一次性模式）同构。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { applyAdaptiveCanvasOnly } from "./sf_common.js";
import { api } from "/scripts/api.js";
import { getState, setGate, STATE_PROP } from "./sf_pause_latent_lib.js";
import { applyGateMode } from "./sf_pause_text_lib.js";
import {
    buildPauseWidget, renderPause, showFrame, frameViewUrl, NODE_MIN_W, NODE_MIN_H,
} from "./sf_pause_latent_ui.js";

const CLASS = "SFPauseLatent";
const HIDDEN_INPUT = "PauseState";
const WIDGET_TYPE = "sf_pause_latent_ui";

// ── 队列：带一次性提交模式跑一次 run ──
// "continue" -> 剪上游（跳过第一段采样），下游从 latent 快照继续；"pause" ->
// 剪下游（停在闸门），重新采样第一段 + 预览
async function queueWithMode(node, mode) {
    // 同一时刻只允许这一个闸门携带一次性提交模式（防两个闸门快速双击都
    // "continue" 进同一个 prompt）
    const allNodes = app.graph?._nodes || app.graph?.nodes || [];
    for (const n of allNodes) {
        if (n !== node && n._sfPauseLatentSubmitMode) n._sfPauseLatentSubmitMode = null;
    }
    node._sfPauseLatentSubmitMode = mode;
    node._sfPauseLatentBusy = mode === "continue" ? "Continuing…" : "Regenerating…";
    renderPause(node);
    try {
        // 转发正常 Run 签名 (number, batchCount)。app.queuePrompt 内部跑
        // app.graphToPrompt，我们的 hook 在那里读一次性模式
        await app.queuePrompt(0, 1);
    } catch (err) {
        console.error("[SF Pause Latent] queue failed", err);
    } finally {
        node._sfPauseLatentSubmitMode = null;
        node._sfPauseLatentBusy = null;
        renderPause(node);
    }
}

// 状态行短暂消息，2s 后清除（Copy / Open 反馈）
function flash(node, msg) {
    node._sfPauseLatentFlash = msg;
    renderPause(node);
    clearTimeout(node._sfPauseLatentFlashTimer);
    node._sfPauseLatentFlashTimer = setTimeout(() => {
        node._sfPauseLatentFlash = null;
        renderPause(node);
    }, 2000);
}

// 把预览的快照复制到系统剪贴板（PNG）
async function copySnapshot(node) {
    const frame = getState(node).frame;
    if (!frame?.filename) { flash(node, "Run once to capture an image"); return; }
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
        // 强制 image/png——部分服务器报 image/x-png，ClipboardItem 严格
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
    if (!frame?.filename) { flash(node, "Run once to capture an image"); return; }
    // noopener：新标签页无法回指 ComfyUI 窗口
    const win = window.open(frameViewUrl(frame), "_blank", "noopener");
    if (!win) flash(node, "Popup blocked");
}

// ── Save（走 sfnodes 后端路由 /api/sfnodes/preview/*）──
const SAVE_PREFIX = "PauseLatent";

// 取预览的快照并返回 PNG data URL。temp 文件消失（如重启）时抛 "expired"
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

// 优先用快照捕获时的执行期工作流（产生它的精确种子），保存的 PNG 拖回
// ComfyUI 是同一张图。没有捕获元数据时回退到活图。
async function resolveSaveMeta(node) {
    const m = node._sfPauseLatentExecMeta;
    if (m && m.workflow) return { workflow: m.workflow, prompt: m.prompt };
    const { workflow, output } = await app.graphToPrompt();
    return { workflow, prompt: output };
}

function saveErr(node, err) {
    if (err?.message === "expired") flash(node, "Snapshot expired - run again");
    else if (err?.message === "nosnap") flash(node, "Run once to capture an image");
    else flash(node, "Save failed");
}

// 存到 ComfyUI output/ 目录（嵌入工作流）
async function saveToOutput(node) {
    if (!node._sfPauseLatentHasSnapshot) { flash(node, "Run once to capture an image"); return; }
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

// 存到用户选择的目录（OS "Save as" 对话框；无 showSaveFilePicker 时回退浏览器
// Downloads 目录）
async function saveToDisk(node) {
    if (!node._sfPauseLatentHasSnapshot) { flash(node, "Run once to capture an image"); return; }
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
            if (err?.name === "AbortError") return; // 用户取消，静默
            flash(node, "Save failed");
        }
        return;
    }
    // 回退：<a download> 到浏览器 Downloads 目录
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
        getMinHeight: () => NODE_MIN_H,  // 常量
    });
    applyAdaptiveCanvasOnly(widget);

    // 新节点默认尺寸。configure() 在 onNodeCreated 之后运行并恢复已保存尺寸，
    // 所以这只对全新节点生效
    if (!node.size || node.size[0] < NODE_MIN_W) node.size[0] = 400;
    if (!node.size || node.size[1] < NODE_MIN_H) node.size[1] = 400;

    // 延迟首次渲染直到 node.properties 恢复
    queueMicrotask(() => restore(node));
}

// 纯 DOM 恢复：重渲染控件 + 重载最近快照（如有）。绝不改动序列化状态，
// 加载路径安全
function restore(node) {
    renderPause(node);
    const s = getState(node);
    if (s.frame) showFrame(node, s.frame);  // onerror 时禁用 Continue
}

app.registerExtension({
    name: "sfnodes.PauseLatent",

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

        // 自愈最小尺寸（与 getMinHeight 双保险）。只抬升过小的尺寸
        const origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            if (size[0] < NODE_MIN_W) size[0] = NODE_MIN_W;
            if (size[1] < NODE_MIN_H) size[1] = NODE_MIN_H;
            return origResize?.apply(this, arguments);
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            clearTimeout(this._sfPauseLatentFlashTimer);
            this._sfPauseLatentEls = null;
            return origRemoved?.apply(this, arguments);
        };
    },
});

// ── executed 事件：接收 Python 的预览 frame，回填预览 ──
api.addEventListener("executed", (e) => {
    const d = e.detail;
    const frames = d?.output?.sf_pause_latent_frame;
    if (!frames || !frames.length) return;
    // 节点 id 可能是数字（legacy）或字符串（Vue）——都试
    let node = app.graph.getNodeById(d.node);
    if (!node && typeof d.node === "string") node = app.graph.getNodeById(parseInt(d.node, 10));
    if (!node || node.comfyClass !== CLASS) return;
    const f = frames[0];
    // 捕获执行期工作流供 Save 按钮（仅运行时，绝不持久化到 node.properties——
    // 会撑爆已保存工作流）。只在新鲜 pause/pass 捕获时出现，所以即使之后
    // Continue（其 frame 无 meta）仍是生成工作流
    if (f._sf_pause_meta) node._sfPauseLatentExecMeta = f._sf_pause_meta;
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

// isOutput(classType)：class_type 是否为 OUTPUT_NODE。从实时节点定义读。
// 注册表缺失 -> null -> prune 回退删一切（安全：上游仍被跳过）
function makeIsOutput() {
    const reg = window.LiteGraph?.registered_node_types;
    if (!reg) return null;
    return (classType) => !!(classType && reg[classType]?.nodeData?.output_node);
}

// 处理顺序：continue 闸门会把自己的下游分支剪掉，可能删掉位于其上游的另一个
// 闸门——所以 continue 必须先于 pause/pass 处理（链式 Pause bug：后一个闸门
// Continue 时被前一个 Pause 闸门先剪掉分支，Continue 毫无效果）
const MODE_RANK = { continue: 0, pause: 1, pass: 2 };

// prompt 中每个 Pause Latent 条目及其应生效的模式。两个钩子共用，INJECT 与
// PRUNE 永不产生分歧
function collectGates(out) {
    let index = null;
    const gates = [];
    for (const id in out) {
        const entry = out[id];
        if (!entry || entry.class_type !== CLASS) continue;
        if (!index) index = buildNodeIndex();
        const node = findNode(index, id);
        // 生效模式：一次性按钮覆盖（Continue/Regenerate）优先，否则持久切换
        // （Pause 默认 / Pass）。解析不到活节点时默认无害的 "pass"（不剪）
        // 而非破坏性的 "pause"（会静默截断无法确认的工作流）
        const submit = node?._sfPauseLatentSubmitMode;
        let mode;
        if (submit === "continue" || submit === "pause") {
            mode = submit;
        } else if (node) {
            mode = node.properties?.[STATE_PROP]?.gate === "pass" ? "pass" : "pause";
        } else {
            mode = "pass";
        }
        gates.push({ id, entry, mode });
    }
    return gates;
}

// ─────────────────────────────────────────────────────────────────────────
// 双钩子、两个职责（Switch 拆分，与 SFPauseText 相同推理）：
//   graphToPrompt   -> 只 INJECT 模式。它也会因 "Export (API)"、工作流分享、
//                      保存按钮运行，在那里剪枝会把导出的工作流静默截断
//   api.queuePrompt -> PRUNE。只在 prompt 真正提交时运行
// 一次性 _sfPauseLatentSubmitMode 在剪枝时仍可读：queueWithMode 在
// `await app.queuePrompt(...)` 解析后的 finally 里清除，api.queuePrompt 就
// 发生在那次 await 之内。
// ─────────────────────────────────────────────────────────────────────────
if (!app._sfPauseLatentPatched) {
    app._sfPauseLatentPatched = true;
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
                    g.entry.inputs[HIDDEN_INPUT] = JSON.stringify({ mode: g.mode });
                }
            }
        } catch (e) {
            console.error("[sfnodes] Pause Latent prompt injection failed; prompt sent unchanged", e);
        }
        return result;
    };
}

// 提交时剪枝。api.queuePrompt(number, {output, workflow}, options) 是所有浏览器
// run 的唯一漏斗。原样转发 ...args
if (!api._sfPauseLatentQueueWrapped) {
    api._sfPauseLatentQueueWrapped = true;
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
                    // prune 复用 sf_pause_text_lib.js::applyGateMode，
                    // 以 "latent" 作为闸门输入键；预览 image 链接在 continue
                    // 时一并删除（否则 VAEDecode 仍被消费、拉活第一段采样器）
                    applyGateMode(out, g.id, g.entry, g.mode, isOutput, HIDDEN_INPUT, {
                        inputKey: "latent",
                        extraInputKeys: ["image"],
                        editedText: "",
                    });
                }
            }
        } catch (err) {
            // 剪枝失败绝不能挡住用户的 run
            console.error("[SF Pause Latent] submit-time prune failed", err);
        }
        return _origQueuePrompt(...args);
    };
}
