// ==========================================================================
// sf_pause_kit.js - 闸门家族共享引擎（image/mask/latent 三闸门收敛）
// ==========================================================================
//
// 四闸门（text/image/mask/latent）中结构同构的三份克隆收敛于此：
//   - state（makeGateState）：node.properties 随工作流保存，加载路径绝不重写
//     序列化状态（不误标工作流已修改）
//   - 节点体 UI（buildPauseBody）：Pause/Pass 切换 pill + 状态行 + 两行按钮
//     （Regenerate/Continue + Copy/Save Disk/Save Output/Open）+ 预览
//     （object-fit:contain）+ 尺寸行。CSS 类前缀逐字保留（sf-pi-/sf-pm-/
//     sf-pl-——类名前缀与既有插件隔离，见 experience/nodes-text.md §6.3）
//   - 主扩展（definePauseGate）：快照 Copy/Open/Save 链路、双钩子、executed
//     回填。队列语义（双钩子，与 SFPauseText 相同推理）：graphToPrompt 只
//     INJECT {mode}——它也会因 Export (API)/分享/保存运行，在那里剪枝会把
//     导出的工作流静默截断；api.queuePrompt 提交时才 PRUNE。
//   - 图索引助手（buildClassNodeIndex/findNodeByPromptId）：四闸门单源，
//     复合 id + seen 环守卫（子图引用循环防栈溢出）
//
// SFPauseText 不在本 kit：keep 三态 + editedText 注入 + textarea 编辑器，
// 结构不同；它通过 sf_pause_text_lib.js 共享 prune 实现（applyGateMode，
// 本 kit 的剪枝同样复用它）。
//
// ⚠ 兼容性红线（改动前先读）：
//   - 运行时属性名由 propPrefix 派生：_sfPauseXxxSubmitMode / _Busy / _Flash /
//     _FlashTimer / _ExecMeta / _HasSnapshot / _Els，以及双钩子守卫标志
//     app._sfPauseXxxPatched / api._sfPauseXxxQueueWrapped
//   - frameEventKey 逐字对应 Python 端 ui 键：image 闸门是历史遗留的
//     "sf_pause_frame"（无 _image_ 段！nodes/image/pause_image.py 硬编码）
//   - CSS style id / 类前缀 / widget type / save prefix / 扩展注册名
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { applyAdaptiveCanvasOnly, sfApiUrl, injectCSSOnce } from "./sf_common.js";
import { applyGateMode } from "./sf_pause_text_lib.js";

// ── state 工厂 ──────────────────────────────────────────────────────────
// 持久化形状：{ gate: "pause"（默认）| "pass", frame: {filename, subfolder,
// type} | null }。"hasSnapshot" 是运行时推导（frame 文件能否真实加载），
// 绝不住 properties——否则重启后 temp 快照消失时的加载解析会弄脏未编辑的工作流。
export function makeGateState(stateProp) {
    const getState = (node) => {
        node.properties = node.properties || {};
        let s = node.properties[stateProp];
        if (!s || typeof s !== "object") {
            s = { gate: "pause", frame: null };
            node.properties[stateProp] = s;
        }
        if (s.gate !== "pause" && s.gate !== "pass") s.gate = "pause";
        return s;
    };
    const setGate = (node, gate) => {
        const s = getState(node);
        s.gate = gate === "pass" ? "pass" : "pause";
    };
    return { STATE_PROP: stateProp, getState, setGate };
}

// ── 节点体 UI 工厂 ───────────────────────────────────────────────────────
// cfg: { cssId, cssPrefix, elsProp, flashProp, busyProp, hasSnapProp,
//        emptyText, contTitle, regenTitle, toolNoun, getState }
const HEADER_H = 130;       // 切换 + 状态 + 两行按钮
const PREVIEW_MIN_H = 150;  // 预览区最小高度
const DIMS_H = 16;          // 预览下的尺寸行

export function buildPauseBody(cfg) {
    const {
        cssId, cssPrefix, elsProp, flashProp, busyProp, hasSnapProp,
        emptyText, contTitle, regenTitle, toolNoun, getState,
    } = cfg;
    // NODE_MIN_W：4 按钮工具行容纳所需。NODE_MIN_H 用固定数字——每次保存/
    // 加载字节一致，node.size 不抖动，工作流不会被误标"已修改"
    const NODE_MIN_W = 300;
    const NODE_MIN_H = HEADER_H + PREVIEW_MIN_H + DIMS_H;

    function injectCSS() {
        injectCSSOnce(cssId, `
.${cssPrefix}root { display:flex; flex-direction:column; flex:1 1 0; min-height:0;
  box-sizing:border-box; padding:6px; gap:6px; font:12px sans-serif; color:#ddd;
  overflow:hidden; }
.${cssPrefix}toggle { display:flex; background:rgba(0,0,0,0.25); border-radius:6px; padding:2px; gap:2px; flex:0 0 auto; }
.${cssPrefix}seg { flex:1 1 0; text-align:center; padding:4px 0; border-radius:5px; cursor:pointer;
  color:rgba(255,255,255,0.6); user-select:none; border:1px solid transparent; }
.${cssPrefix}seg.active { background:${"var(--sf-acc, #f66744)"}; color:#fff; border-color:${"var(--sf-acc, #f66744)"}; }
.${cssPrefix}seg:not(.active):hover { border-color:${"var(--sf-acc, #f66744)"}; color:#ddd; }
.${cssPrefix}status { flex:0 0 auto; font-size:11px; color:rgba(255,255,255,0.7); min-height:14px; text-align:center; }
.${cssPrefix}btns { display:flex; gap:6px; flex:0 0 auto; }
.${cssPrefix}btn { flex:1 1 0; min-width:0; height:26px; line-height:24px; border-radius:4px;
  border:1px solid rgba(255,255,255,0.18); background:rgba(255,255,255,0.05);
  color:rgba(255,255,255,0.85); font:12px sans-serif; cursor:pointer; padding:0 6px;
  box-sizing:border-box; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; user-select:none; }
.${cssPrefix}btn:hover:not(:disabled) { border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.${cssPrefix}btn.primary:not(:disabled) { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.${cssPrefix}btn.primary:hover:not(:disabled) { background:#ff8a5e; border-color:#ff8a5e; }
.${cssPrefix}btn:disabled { opacity:0.45; cursor:default; }
.${cssPrefix}preview { flex:1 1 0; min-height:0; position:relative; background:#1d1d1d;
  border:1px solid #333; border-radius:4px; overflow:hidden; }
.${cssPrefix}img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:none; }
.${cssPrefix}empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  text-align:center; color:#777; font-size:11px; padding:8px; box-sizing:border-box; }
.${cssPrefix}dims { flex:0 0 auto; text-align:center; font-size:10px; color:#aaa;
  min-height:13px; line-height:13px; }
`);
    }

    // 构建 DOM widget。callbacks: { onGate(gate), onContinue(), onRegenerate(),
    // onCopy(), onSaveDisk(), onSaveOutput(), onOpen() }。元素引用缓存在
    // node[elsProp]，供 renderPause / showFrame 使用。
    function buildPauseWidget(node, callbacks) {
        injectCSS();
        const root = document.createElement("div");
        root.className = `${cssPrefix}root`;

        const toggle = document.createElement("div");
        toggle.className = `${cssPrefix}toggle`;
        const segPause = document.createElement("div");
        segPause.className = `${cssPrefix}seg`;
        segPause.textContent = "Pause";
        segPause.title = "Run 时停在此处，预览后再继续";
        const segPass = document.createElement("div");
        segPass.className = `${cssPrefix}seg`;
        segPass.textContent = "Pass";
        segPass.title = "直接放行；整条工作流一次跑完";
        toggle.append(segPause, segPass);
        segPause.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("pause"); });
        segPass.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("pass"); });

        const status = document.createElement("div");
        status.className = `${cssPrefix}status`;

        // 第一行：工作流决策
        const btns = document.createElement("div");
        btns.className = `${cssPrefix}btns`;
        const btnContinue = document.createElement("button");
        btnContinue.className = `${cssPrefix}btn primary`;
        btnContinue.textContent = "▶ Continue";
        btnContinue.title = contTitle;
        const btnRegen = document.createElement("button");
        btnRegen.className = `${cssPrefix}btn`;
        btnRegen.textContent = "⟳ Regenerate";
        btnRegen.title = regenTitle;
        // Regenerate 在左、Continue 在右（Continue 是主要"提交"动作）
        btns.append(btnRegen, btnContinue);

        // 第二行：作用于预览图的工具
        const btns2 = document.createElement("div");
        btns2.className = `${cssPrefix}btns`;
        const btnCopy = document.createElement("button");
        btnCopy.className = `${cssPrefix}btn`;
        btnCopy.textContent = "Copy";
        btnCopy.title = `把预览的${toolNoun}复制到剪贴板`;
        const btnSaveDisk = document.createElement("button");
        btnSaveDisk.className = `${cssPrefix}btn`;
        btnSaveDisk.textContent = "Save Disk";
        btnSaveDisk.title = `把预览的${toolNoun}保存到电脑上的文件夹`;
        const btnSaveOut = document.createElement("button");
        btnSaveOut.className = `${cssPrefix}btn`;
        btnSaveOut.textContent = "Save Output";
        btnSaveOut.title = `把预览的${toolNoun}保存到 ComfyUI 的 output 目录`;
        const btnOpen = document.createElement("button");
        btnOpen.className = `${cssPrefix}btn`;
        btnOpen.textContent = "Open";
        btnOpen.title = `在新标签页打开预览的${toolNoun}`;
        btns2.append(btnCopy, btnSaveDisk, btnSaveOut, btnOpen);

        // stopPropagation 防止点击到达 canvas（取消选中/拖拽）
        btnContinue.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onContinue(); });
        btnRegen.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onRegenerate(); });
        btnCopy.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onCopy(); });
        btnSaveDisk.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onSaveDisk(); });
        btnSaveOut.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onSaveOutput(); });
        btnOpen.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onOpen(); });

        const preview = document.createElement("div");
        preview.className = `${cssPrefix}preview`;
        const img = document.createElement("img");
        img.className = `${cssPrefix}img`;
        const empty = document.createElement("div");
        empty.className = `${cssPrefix}empty`;
        empty.textContent = emptyText;
        preview.append(img, empty);

        // 尺寸行在预览下方（自己的行），不压图
        const dims = document.createElement("div");
        dims.className = `${cssPrefix}dims`;

        root.append(toggle, status, btns, btns2, preview, dims);

        node[elsProp] = {
            segPause, segPass, status,
            btnContinue, btnRegen, btnCopy, btnSaveDisk, btnSaveOut, btnOpen,
            img, empty, dims,
        };
        return root;
    }

    // 从当前状态重渲染控件。纯 DOM（绝不碰 node.size / node.properties 值），
    // 加载路径上调用安全。
    function renderPause(node) {
        const els = node[elsProp];
        if (!els) return;
        const s = getState(node);
        const paused = s.gate === "pause";
        const hasSnap = !!node[hasSnapProp];  // 仅运行时（见 makeGateState 注释）
        els.segPause.classList.toggle("active", paused);
        els.segPass.classList.toggle("active", !paused);

        // Continue / Regenerate 只在 Pause 模式有意义；Copy / Open 只要有图就可用
        els.btnRegen.disabled = !paused;
        els.btnContinue.disabled = !paused || !hasSnap;
        els.btnCopy.disabled = !hasSnap;
        els.btnSaveDisk.disabled = !hasSnap;
        els.btnSaveOut.disabled = !hasSnap;
        els.btnOpen.disabled = !hasSnap;

        if (node[flashProp]) {
            els.status.textContent = node[flashProp];
        } else if (node[busyProp]) {
            els.status.textContent = node[busyProp];
        } else if (!paused) {
            els.status.textContent = "Passing through: whole workflow runs";
        } else if (hasSnap) {
            els.status.textContent = "Paused and ready. Continue to run the rest.";
        } else {
            els.status.textContent = "Paused. Press Run to preview.";
        }
    }

    // 构建快照 frame 的 /view URL。带缓存戳：快照文件名按节点确定性（每次 pause
    // run 覆盖写），不戳会命中浏览器缓存
    function frameViewUrl(frame) {
        const params = new URLSearchParams({
            filename: frame.filename,
            subfolder: frame.subfolder || "",
            type: frame.type || "temp",
            t: String(Date.now()),
        });
        return sfApiUrl(`/view?${params.toString()}`);
    }

    // 在预览中加载 + 显示快照 frame。frame = {filename, subfolder, type}。
    // 成功启用 Continue；失败（重启后 temp 快照被清）显示"expired"并禁用 Continue。
    function showFrame(node, frame) {
        const els = node[elsProp];
        if (!els || !frame || !frame.filename) return;
        const url = frameViewUrl(frame);
        const { img, empty, dims } = els;
        img.onload = () => {
            img.style.display = "block";
            empty.style.display = "none";
            dims.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
            // 仅运行时标志（非 node.properties）：加载时的图片解析绝不重写序列化
            // 状态、不弄脏工作流
            node[hasSnapProp] = true;
            renderPause(node);
        };
        img.onerror = () => {
            img.style.display = "none";
            empty.style.display = "flex";
            empty.textContent = "Preview expired. Press Run to pause again.";
            dims.textContent = "";
            node[hasSnapProp] = false;
            renderPause(node);
        };
        img.src = url;
    }

    return {
        NODE_MIN_W, NODE_MIN_H,
        injectCSS, buildPauseWidget, renderPause, frameViewUrl, showFrame,
    };
}

// ── 图索引助手（四闸门单源）────────────────────────────────────────────
// 按 comfyClass/type 收集全图（含子图）节点。seen 守卫防子图引用循环栈溢出；
// 复合 id（顶层 ""，子图内 "5:" 前缀风格）让子图节点精确匹配它的 "5:3"
// prompt id，且不与碰巧共享裸 id 的顶层节点冲突。
export function buildClassNodeIndex(classNames) {
    const names = Array.isArray(classNames) ? classNames : [classNames];
    const index = new Map();
    const visit = (graph, prefix, seen) => {
        if (!graph || seen.has(graph)) return;
        seen.add(graph);
        const nodes = graph._nodes || graph.nodes || [];
        for (const n of nodes) {
            if (!n) continue;
            const cid = String(prefix) + n.id;
            if (names.includes(n.comfyClass) || names.includes(n.type)) {
                index.set(cid, n);
                // 裸 id 兜底，first-write-wins（顶层先访问），子图节点不覆盖
                // 顶层节点的精确 id 解析
                if (!index.has(String(n.id))) index.set(String(n.id), n);
            }
            const inner = n.subgraph || n.graph || n._graph;
            if (inner && inner !== graph) visit(inner, cid + ":", seen);
        }
    };
    visit(app.graph, "", new Set());
    return index;
}

// prompt id -> 活节点。先精确（含复合 id），再退到冒号尾段（旧版 prompt 的
// 子图 id 形态）
export function findNodeByPromptId(index, promptId) {
    const sId = String(promptId);
    if (index.has(sId)) return index.get(sId);
    const tail = sId.includes(":") ? sId.slice(sId.lastIndexOf(":") + 1) : null;
    if (tail && index.has(tail)) return index.get(tail);
    return null;
}

// isOutput(classType)：class_type 是否为 OUTPUT_NODE。从实时节点定义读。
// 注册表缺失 -> null -> prune 回退删一切（安全：上游仍被跳过）
function makeIsOutputNode() {
    const reg = window.LiteGraph?.registered_node_types;
    if (!reg) return null;
    return (classType) => !!(classType && reg[classType]?.nodeData?.output_node);
}

// continue 必须先于 pause/pass 处理（链式闸门：continue 会剪掉自己的下游分支，
// 可能删掉位于其上游的另一个闸门）
const MODE_RANK = { continue: 0, pause: 1, pass: 2 };

// ── 主扩展工厂 ───────────────────────────────────────────────────────────
// cfg: { classy, extensionName, widgetType, savePrefix,
//        stateProp, propPrefix,
//        inputKey, extraInputKeys, frameEventKey,
//        logTag, injectName, captureMsg,
//        cssId, cssPrefix, emptyText, contTitle, regenTitle, toolNoun }
// 立即执行注册（app.registerExtension + executed 监听 + 双钩子安装），与原
// 单体模块顶层副作用一致。返回 { app: 扩展对象, ...内部件 }（供测试检视）。
export function definePauseGate(cfg) {
    const {
        classy, extensionName, widgetType, savePrefix,
        stateProp, propPrefix,
        hiddenInput = "PauseState",
        inputKey, extraInputKeys = null,
        frameEventKey,
        logTag, injectName, captureMsg,
        cssId, cssPrefix, emptyText, contTitle, regenTitle, toolNoun,
    } = cfg;

    const state = makeGateState(stateProp);
    const { getState, setGate, STATE_PROP } = state;

    const body = buildPauseBody({
        cssId, cssPrefix,
        elsProp: propPrefix + "Els",
        flashProp: propPrefix + "Flash",
        busyProp: propPrefix + "Busy",
        hasSnapProp: propPrefix + "HasSnapshot",
        emptyText, contTitle, regenTitle, toolNoun,
        getState,
    });
    const { buildPauseWidget, renderPause, showFrame, frameViewUrl, NODE_MIN_W, NODE_MIN_H } = body;

    const submitProp = propPrefix + "SubmitMode";
    const busyProp = propPrefix + "Busy";
    const flashProp = propPrefix + "Flash";
    const flashTimerProp = propPrefix + "FlashTimer";
    const execMetaProp = propPrefix + "ExecMeta";
    const hasSnapProp = propPrefix + "HasSnapshot";

    // ── 队列：带一次性提交模式跑一次 run ──
    // "continue" -> 剪上游（跳过它），下游从快照继续；"pause" -> 剪下游（停在
    // 闸门），重新捕获快照 + 预览
    async function queueWithMode(node, mode) {
        // 同一时刻只允许这一个闸门携带一次性提交模式（防两个闸门快速双击都
        // "continue" 进同一个 prompt）
        const allNodes = app.graph?._nodes || app.graph?.nodes || [];
        for (const n of allNodes) {
            if (n !== node && n[submitProp]) n[submitProp] = null;
        }
        node[submitProp] = mode;
        node[busyProp] = mode === "continue" ? "Continuing…" : "Regenerating…";
        renderPause(node);
        try {
            // 转发正常 Run 签名 (number, batchCount)。app.queuePrompt 内部跑
            // app.graphToPrompt，我们的 hook 在那里读一次性模式
            await app.queuePrompt(0, 1);
        } catch (err) {
            console.error(`[${logTag}] queue failed`, err);
        } finally {
            node[submitProp] = null;
            node[busyProp] = null;
            renderPause(node);
        }
    }

    // 状态行短暂消息，2s 后清除（Copy / Open 反馈）
    function flash(node, msg) {
        node[flashProp] = msg;
        renderPause(node);
        clearTimeout(node[flashTimerProp]);
        node[flashTimerProp] = setTimeout(() => {
            node[flashProp] = null;
            renderPause(node);
        }, 2000);
    }

    // 把预览的快照复制到系统剪贴板（PNG）
    async function copySnapshot(node) {
        const frame = getState(node).frame;
        if (!frame?.filename) { flash(node, captureMsg); return; }
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

    // 在新标签页打开预览的快照（全屏查看）。noopener：新标签页无法回指 ComfyUI
    function openSnapshot(node) {
        const frame = getState(node).frame;
        if (!frame?.filename) { flash(node, captureMsg); return; }
        const win = window.open(frameViewUrl(frame), "_blank", "noopener");
        if (!win) flash(node, "Popup blocked");
    }

    // ── Save（走 sfnodes 后端路由 /api/sfnodes/preview/*）──

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
        const m = node[execMetaProp];
        if (m && m.workflow) return { workflow: m.workflow, prompt: m.prompt };
        const { workflow, output } = await app.graphToPrompt();
        return { workflow, prompt: output };
    }

    function saveErr(node, err) {
        if (err?.message === "expired") flash(node, "Snapshot expired - run again");
        else if (err?.message === "nosnap") flash(node, captureMsg);
        else flash(node, "Save failed");
    }

    // 存到 ComfyUI output/ 目录（嵌入工作流）
    async function saveToOutput(node) {
        if (!node[hasSnapProp]) { flash(node, captureMsg); return; }
        try {
            const image_b64 = await snapshotDataURL(node);
            const { workflow, prompt } = await resolveSaveMeta(node);
            const resp = await fetch(api.apiURL("/api/sfnodes/preview/save"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_b64, filename_prefix: savePrefix, workflow, prompt }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) { flash(node, `Save failed: ${data.error || resp.status}`); return; }
            flash(node, `Saved: ${data.filename}`);
        } catch (err) { saveErr(node, err); }
    }

    // 存到用户选择的目录（OS "Save as" 对话框；无 showSaveFilePicker 时回退浏览器
    // Downloads 目录）
    async function saveToDisk(node) {
        if (!node[hasSnapProp]) { flash(node, captureMsg); return; }
        let preparedBlob;
        let suggestedName = `${savePrefix}.png`;
        try {
            const image_b64 = await snapshotDataURL(node);
            const { workflow, prompt } = await resolveSaveMeta(node);
            const resp = await fetch(api.apiURL("/api/sfnodes/preview/prepare"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_b64, filename_prefix: savePrefix, workflow, prompt }),
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
        const widget = node.addDOMWidget(widgetType, widgetType, root, {
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

    // prompt 中每个本类条目及其应生效的模式。两个钩子共用，INJECT 与 PRUNE
    // 永不产生分歧。生效模式：一次性按钮覆盖（Continue/Regenerate）优先，否则
    // 持久切换（Pause 默认 / Pass）。解析不到活节点时默认无害的 "pass"（不剪）
    // 而非破坏性的 "pause"（会静默截断无法确认的工作流）
    function collectGates(out) {
        let index = null;
        const gates = [];
        for (const id in out) {
            const entry = out[id];
            if (!entry || entry.class_type !== classy) continue;
            if (!index) index = buildClassNodeIndex(classy);
            const node = findNodeByPromptId(index, id);
            const submit = node?.[submitProp];
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

    const ext = {
        name: extensionName,

        beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== classy) return;

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
                clearTimeout(this[flashTimerProp]);
                this[propPrefix + "Els"] = null;
                return origRemoved?.apply(this, arguments);
            };
        },
    };
    app.registerExtension(ext);

    // ── executed 事件：接收 Python 的快照预览 frame，回填预览 ──
    api.addEventListener("executed", (e) => {
        const d = e.detail;
        const frames = d?.output?.[frameEventKey];
        if (!frames || !frames.length) return;
        // 节点 id 可能是数字（legacy）或字符串（Vue）——都试
        let node = app.graph.getNodeById(d.node);
        if (!node && typeof d.node === "string") node = app.graph.getNodeById(parseInt(d.node, 10));
        if (!node || node.comfyClass !== classy) return;
        const f = frames[0];
        // 捕获执行期工作流供 Save 按钮（仅运行时，绝不持久化到 node.properties——
        // 会撑爆已保存工作流）。只在新鲜 pause/pass 捕获时出现，所以即使之后
        // Continue（其 frame 无 meta）仍是生成工作流
        if (f._sf_pause_meta) node[execMetaProp] = f._sf_pause_meta;
        const s = getState(node);
        s.frame = { filename: f.filename, subfolder: f.subfolder || "", type: f.type || "temp" };
        showFrame(node, s.frame);
    });

    // ─────────────────────────────────────────────────────────────────────
    // 双钩子、两个职责（Switch 拆分，与 SFPauseText 相同推理）：见文件头注释。
    // 一次性 submitProp 在剪枝时仍可读：queueWithMode 在
    // `await app.queuePrompt(...)` 解析后的 finally 里清除，api.queuePrompt 就
    // 发生在那次 await 之内。
    // ─────────────────────────────────────────────────────────────────────
    if (!app[propPrefix + "Patched"]) {
        app[propPrefix + "Patched"] = true;
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
                        g.entry.inputs[hiddenInput] = JSON.stringify({ mode: g.mode });
                    }
                }
            } catch (e) {
                console.error(`[sfnodes] ${injectName} prompt injection failed; prompt sent unchanged`, e);
            }
            return result;
        };
    }

    // 提交时剪枝。api.queuePrompt(number, {output, workflow}, options) 是所有
    // 浏览器 run 的唯一漏斗。原样转发 ...args
    if (!api[propPrefix + "QueueWrapped"]) {
        api[propPrefix + "QueueWrapped"] = true;
        const _origQueuePrompt = api.queuePrompt.bind(api);
        api.queuePrompt = async function (...args) {
            try {
                const out = args[1]?.output;
                if (out) {
                    const isOutput = makeIsOutputNode();
                    const gates = collectGates(out);
                    gates.sort((a, b) => MODE_RANK[a.mode] - MODE_RANK[b.mode]);
                    for (const g of gates) {
                        if (!out[g.id]) continue;  // 已被更早的 continue 闸门剪掉
                        // prune 复用 sf_pause_text_lib.js::applyGateMode；
                        // extraInputKeys 用于 latent 闸门的预览 image 链接
                        // （continue 时一并删除，否则 VAEDecode 仍被消费、拉活
                        // 第一段采样器）
                        applyGateMode(out, g.id, g.entry, g.mode, isOutput, hiddenInput, {
                            inputKey,
                            extraInputKeys: extraInputKeys || [],
                            editedText: "",
                        });
                    }
                }
            } catch (err) {
                // 剪枝失败绝不能挡住用户的 run
                console.error(`[${logTag}] submit-time prune failed`, err);
            }
            return _origQueuePrompt(...args);
        };
    }

    return {
        ext, state, body,
        queueWithMode, flash, copySnapshot, openSnapshot,
        saveToOutput, saveToDisk,
        collectGates,
        props: { submitProp, busyProp, flashProp, flashTimerProp, execMetaProp, hasSnapProp },
    };
}
