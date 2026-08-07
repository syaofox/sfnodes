// SFPauseImage 主扩展端到端冒烟测试（Node 直接运行：node tests/test_pause_image_smoke.js）
// 用 mock DOM/app/api 真实加载模块，验证：
//   - 模块加载 / 双钩子包装（graphToPrompt 注入 + queuePrompt 剪枝）
//   - beforeRegisterNodeDef / onNodeCreated setupNode（DOM widget 构建）
//   - graphToPrompt：pause/pass 注入 PauseState
//   - queuePrompt：pause 删下游 / continue 跳上游模型链（inputKey image）
//   - executed 事件回填快照 frame + 捕获 exec meta
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
        selectionStart: 0, selectionEnd: 0, naturalWidth: 0, naturalHeight: 0,
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
    open() { return {}; },
    showSaveFilePicker: undefined,
    LiteGraph: {
        vueNodesMode: false,
        registered_node_types: {
            "SaveImage": { nodeData: { output_node: true } },
            "SFPauseImage": { nodeData: { output_node: true } },
            "SecondOut": { nodeData: { output_node: true } },
            "KSampler": { nodeData: {} },
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
    apiURL: (route) => route,
};

// ── 加载模块（替换 /scripts/* import，相对 import 改 .mjs 同 tmp）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pi_"));
for (const n of ["sf_pause_text_lib.js", "sf_pause_image_lib.js",
    "sf_pause_image_ui.js", "sf_pause_image.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    await import(path.join(tmpDir, "sf_pause_image.mjs"));
    check("graphToPrompt 已包装", app._sfPauseImagePatched === true);
    check("queuePrompt 已包装", api._sfPauseImageQueueWrapped === true);
    check("executed 监听已注册", executedHandlers.length === 1);
    check("扩展已注册", !!app._ext && app._ext.name === "sfnodes.PauseImage");

    const ext = app._ext;
    const proto = {};
    ext.beforeRegisterNodeDef({ name: "SFPauseImage", prototype: proto }, { name: "SFPauseImage" });
    check("原型钩子已安装", typeof proto.onNodeCreated === "function" &&
        typeof proto.onConfigure === "function" && typeof proto.onRemoved === "function");

    const gate = {
        id: 1, size: [300, 200], inputs: [], widgets: [],
        properties: { pauseImageState: { gate: "pause", frame: null } },
        addDOMWidget() { return { options: {} }; },
        setDirtyCanvas() {},
        setSize() {},
        type: "SFPauseImage", comfyClass: "SFPauseImage",
    };
    app.graph._nodes = [gate];
    proto.onNodeCreated.call(gate);
    check("setupNode 完成", !!gate._sfPauseImageEls && !!gate._sfPauseImageEls.img);
    await new Promise((r) => setTimeout(r, 5));   // queueMicrotask restore 跑完

    // ── graphToPrompt 注入：pause ──
    promptObj = { output: {
        "1": { class_type: "SFPauseImage", inputs: { image: ["2", 0] } },
        "2": { class_type: "VAEDecode", inputs: {} },
        "3": { class_type: "Upscale", inputs: { image: ["1", 0] } },
        "4": { class_type: "SaveImage", inputs: { images: ["3", 0] } },
    } };
    await app.graphToPrompt();
    let st = JSON.parse(promptObj.output["1"].inputs.PauseState);
    check("注入 pause 模式", st.mode === "pause");
    check("注入后不剪", promptObj.output["2"] && promptObj.output["3"] && promptObj.output["4"]);

    // ── queuePrompt：pause 剪枝删下游 ──
    await api.queuePrompt(0, promptObj, {});
    check("pause 剪枝删除下游 3/4", !promptObj.output["3"] && !promptObj.output["4"]);
    check("pause 剪枝保留上游 2", promptObj.output["2"] !== undefined);

    // ── pass：不剪 ──
    gate.properties.pauseImageState.gate = "pass";
    promptObj = { output: {
        "1": { class_type: "SFPauseImage", inputs: { image: ["2", 0] } },
        "2": { class_type: "VAEDecode", inputs: {} },
        "3": { class_type: "Upscale", inputs: { image: ["1", 0] } },
    } };
    await app.graphToPrompt();
    st = JSON.parse(promptObj.output["1"].inputs.PauseState);
    check("pass 注入", st.mode === "pass");
    await api.queuePrompt(0, promptObj, {});
    check("pass 不剪", promptObj.output["2"] && promptObj.output["3"]);

    // ── continue（一次性提交模式）：跳上游模型链 ──
    gate.id = 3;   // 与 continue 测试图的闸门 id 对齐（findNode 按 id 解析）
    gate.properties.pauseImageState.gate = "pause";
    gate._sfPauseImageSubmitMode = "continue";   // 模拟 queueWithMode 挂载
    promptObj = { output: {
        "0": { class_type: "EmptyImage", inputs: {} },
        "1": { class_type: "KSampler", inputs: { latent: ["0", 0] } },
        "2": { class_type: "VAEDecode", inputs: { samples: ["1", 0] } },
        "3": { class_type: "SFPauseImage", inputs: { image: ["2", 0] } },
        "4": { class_type: "Upscale", inputs: { image: ["3", 0] } },
        "5": { class_type: "SaveImage", inputs: { images: ["4", 0] } },
        "6": { class_type: "SecondOut", inputs: { images: ["2", 0] } },
    } };
    await app.graphToPrompt();
    st = JSON.parse(promptObj.output["3"].inputs.PauseState);
    check("一次性模式 continue 注入", st.mode === "continue");
    await api.queuePrompt(0, promptObj, {});
    check("continue 剪枝删 image 链接", !("image" in promptObj.output["3"].inputs));
    check("continue 剪枝删拉活上游的 SecondOut", !promptObj.output["6"]);
    check("continue 剪枝保留下游 SaveImage", promptObj.output["5"] !== undefined);
    gate._sfPauseImageSubmitMode = null;

    // ── executed 事件回填 ──
    executedHandlers[0]({ detail: {
        node: 3,
        output: { sf_pause_frame: [{ filename: "sf_pause_1.png", subfolder: "", type: "temp", _sf_pause_meta: { prompt: { x: 1 }, workflow: { w: 1 } } }] },
    } });
    const s = gate.properties.pauseImageState;
    check("executed 回填 frame", s.frame.filename === "sf_pause_1.png" && s.frame.type === "temp");
    check("executed 捕获 exec meta", gate._sfPauseImageExecMeta && gate._sfPauseImageExecMeta.workflow.w === 1);
    check("预览 img.src 已设置", typeof gate._sfPauseImageEls.img.src === "string" && gate._sfPauseImageEls.img.src.includes("sf_pause_1.png"));

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
