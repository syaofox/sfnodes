// SFPauseText 主扩展端到端冒烟测试（Node 直接运行：node tests/test_pause_text_smoke.js）
// 用 mock DOM/app/api 真实加载模块，验证：
//   - 模块加载 / 双钩子包装（graphToPrompt 注入 + queuePrompt 剪枝）
//   - nodeCreated setupNode（DOM widget 构建）
//   - graphToPrompt：pause/continue/pass 注入 PauseState（含 keep -> continue 映射）
//   - queuePrompt：pause 删下游 / continue 跳上游模型链
//   - executed 事件回填模型文本（setModelText）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素）──
function makeEl() {
    const style = { setProperty() {}, getPropertyValue() { return ""; } };
    return {
        style, dataset: {}, children: [],
        className: "", textContent: "", innerHTML: "", value: "", placeholder: "",
        type: "", title: "", rows: 1, spellcheck: false, disabled: false, checked: false,
        draggable: false, isConnected: true, offsetWidth: 100, offsetHeight: 20,
        selectionStart: 0, selectionEnd: 0,
        classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        replaceWith() {}, replaceChildren(...kids) { this.children = kids; },
        remove() { this.removed = true; },
        contains() { return false; }, closest() { return null; },
        querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        addEventListener() {}, removeEventListener() {},
        focus() {}, blur() {}, select() {}, click() {},
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {}, setPointerCapture() {}, releasePointerCapture() {}, setSelectionRange() {},
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
    getElementById() { return null; },
    activeElement: makeEl(),
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
    app: null,
    LiteGraph: {
        vueNodesMode: false,
        registered_node_types: {
            "SaveImage": { nodeData: { output_node: true } },
            "SFPauseText": { nodeData: { output_node: true } },
            "SecondOut": { nodeData: { output_node: true } },
            "LLM": { nodeData: {} },
        },
    },
};

// ── app / api mock ──
let promptObj = { output: {} };
let queuedCalls = 0;
globalThis.app = {
    graph: {
        _nodes: [], links: {},
        getNodeById(id) { return this._nodes.find((n) => String(n.id) === String(id)); },
        setDirtyCanvas() {},
    },
    canvas: { setDirty() {}, ds: { scale: 1 } },
    extensionManager: { toast: { add() {} } },
    registerExtension(ext) { this._ext = ext; },
    graphToPrompt: async () => promptObj,
    queuePrompt: async () => { queuedCalls++; },
};
globalThis.window.app = globalThis.app;
const executedHandlers = [];
globalThis.api = {
    queuePrompt: async () => {},
    addEventListener(type, fn) { if (type === "executed") executedHandlers.push(fn); },
};

// ── 加载模块（替换 /scripts/* import，相对 import 改 .mjs 同 tmp）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ptx_"));
for (const n of ["sf_pause_text_lib.js", "sf_pause_text_ui.js", "sf_pause_text.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    await import(path.join(tmpDir, "sf_pause_text.mjs"));
    check("graphToPrompt 已包装", app._sfPauseTextPatched === true);
    check("queuePrompt 已包装", api._sfPauseTextQueueWrapped === true);
    check("executed 监听已注册", executedHandlers.length === 1);

    const ext = app._ext;
    check("扩展已注册", !!ext && ext.name === "sfnodes.PauseText");

    // ── beforeRegisterNodeDef / nodeCreated ──
    const proto = {};
    ext.beforeRegisterNodeDef({ name: "SFPauseText", prototype: proto }, { name: "SFPauseText" });
    check("原型钩子已安装", typeof proto.onNodeCreated === "function" &&
        typeof proto.onConfigure === "function" && typeof proto.onRemoved === "function");

    const gate = {
        id: 1, size: [300, 200], inputs: [], widgets: [],
        properties: { pauseTextState: { gate: "pause", text: "box text", original: "" } },
        addDOMWidget() { return { options: {} }; },
        setDirtyCanvas() {},
        setSize() {},
        type: "SFPauseText", comfyClass: "SFPauseText",
    };
    app.graph._nodes = [gate];
    proto.onNodeCreated.call(gate);   // setupNode 挂在原型 onNodeCreated（无 nodeCreated 钩子）
    check("setupNode 完成", !!gate._sfPauseTextEls && !!gate._sfPauseTextEls.ta);
    await new Promise((r) => setTimeout(r, 5));   // 让 queueMicrotask restore 跑完
    check("restore 已推文本进盒子", gate._sfPauseTextEls.ta.value === "box text");

    // ── graphToPrompt 注入：pause ──
    promptObj = { output: {
        "1": { class_type: "SFPauseText", inputs: { text: ["2", 0] } },
        "2": { class_type: "LLM", inputs: {} },
        "3": { class_type: "Process", inputs: { text: ["1", 0] } },
        "4": { class_type: "SaveImage", inputs: { img: ["3", 0] } },
    } };
    await app.graphToPrompt();
    let st = JSON.parse(promptObj.output["1"].inputs.PauseState);
    check("注入 pause 模式与文本", st.mode === "pause" && st.text === "box text");
    check("注入后不剪任何节点", promptObj.output["2"] && promptObj.output["3"] && promptObj.output["4"]);

    // ── queuePrompt：pause 剪枝删下游 ──
    await api.queuePrompt(0, promptObj, {});
    check("pause 剪枝删除下游 3/4", !promptObj.output["3"] && !promptObj.output["4"]);
    check("pause 剪枝保留上游 2", promptObj.output["2"] !== undefined);

    // ── keep -> continue 映射 + continue 剪枝 ──
    gate.properties.pauseTextState.gate = "keep";
    gate._sfPauseTextEls.ta.value = "kept edit";   // editedTextOf 优先读活 textarea
    promptObj = { output: {
        "1": { class_type: "SFPauseText", inputs: { text: ["2", 0] } },
        "2": { class_type: "LLM", inputs: {} },
        "3": { class_type: "Process", inputs: { text: ["1", 0] } },
        "4": { class_type: "SaveImage", inputs: { img: ["3", 0] } },
        "5": { class_type: "SecondOut", inputs: { text: ["2", 0] } },
    } };
    await app.graphToPrompt();
    st = JSON.parse(promptObj.output["1"].inputs.PauseState);
    check("keep 映射为 continue", st.mode === "continue" && st.text === "kept edit");
    await api.queuePrompt(0, promptObj, {});
    check("continue 剪枝删 text 链接", !("text" in promptObj.output["1"].inputs));
    check("continue 剪枝删拉活上游的 SecondOut", !promptObj.output["5"]);
    check("continue 剪枝保留下游 SaveImage", promptObj.output["4"] !== undefined);

    // ── pass：不剪 ──
    gate.properties.pauseTextState.gate = "pass";
    promptObj = { output: {
        "1": { class_type: "SFPauseText", inputs: { text: ["2", 0] } },
        "2": { class_type: "LLM", inputs: {} },
        "3": { class_type: "Process", inputs: { text: ["1", 0] } },
    } };
    await app.graphToPrompt();
    st = JSON.parse(promptObj.output["1"].inputs.PauseState);
    check("pass 注入", st.mode === "pass");
    await api.queuePrompt(0, promptObj, {});
    check("pass 不剪", promptObj.output["2"] && promptObj.output["3"]);

    // ── executed 事件回填 ──
    executedHandlers[0]({ detail: { node: 1, output: { sf_pause_text: ["model words"] } } });
    check("executed 回填模型文本", gate.properties.pauseTextState.text === "model words" &&
        gate.properties.pauseTextState.original === "model words");
    check("executed 推文本进盒子", gate._sfPauseTextEls.ta.value === "model words");

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
