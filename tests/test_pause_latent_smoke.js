// SFPauseLatent 主扩展端到端冒烟测试（Node 直接运行：node tests/test_pause_latent_smoke.js）
// 用 mock DOM/app/api 真实加载模块，验证 kit 的 latent 配置路径：
//   - 模块加载 / 双钩子包装（graphToPrompt 注入 + queuePrompt 剪枝）
//   - beforeRegisterNodeDef / onNodeCreated setupNode（DOM widget 构建）
//   - graphToPrompt：pause/pass 注入 PauseState
//   - queuePrompt：pause 删下游但保留 image 预览链接；continue 删 latent+image
//     双链接（extraInputKeys）、跳第一段上游、保留下游
//   - executed 事件按 sf_pause_latent_frame 键回填快照 frame + 捕获 exec meta
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
            "SFPauseLatent": { nodeData: { output_node: true } },
            "SecondOut": { nodeData: { output_node: true } },
            "KSampler": { nodeData: {} },
            "VAEDecode": { nodeData: {} },
        },
    },
};

// ── app / api mock ──
let promptObj = { output: {} };
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
    queuePrompt: async () => {},
};
globalThis.window.app = globalThis.app;
const executedHandlers = [];
globalThis.api = {
    queuePrompt: async () => {},
    addEventListener(type, fn) { if (type === "executed") executedHandlers.push(fn); },
    apiURL: (route) => route,
};

// ── 加载模块（替换 /scripts/* import，相对 import 改 .mjs 同 tmp）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pl_"));
for (const n of ["sf_common.js", "sf_pause_text_lib.js", "sf_pause_kit.js",
    "sf_pause_latent.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    await import(path.join(tmpDir, "sf_pause_latent.mjs"));
    check("graphToPrompt 已包装", app._sfPauseLatentPatched === true);
    check("queuePrompt 已包装", api._sfPauseLatentQueueWrapped === true);
    check("executed 监听已注册", executedHandlers.length === 1);
    check("扩展已注册", !!app._ext && app._ext.name === "sfnodes.PauseLatent");

    const ext = app._ext;
    const proto = {};
    ext.beforeRegisterNodeDef({ name: "SFPauseLatent", prototype: proto }, { name: "SFPauseLatent" });
    check("原型钩子已安装", typeof proto.onNodeCreated === "function" &&
        typeof proto.onConfigure === "function" && typeof proto.onRemoved === "function");

    const gate = {
        id: 1, size: [300, 200], inputs: [], widgets: [],
        properties: { pauseLatentState: { gate: "pause", frame: null } },
        addDOMWidget() { return { options: {} }; },
        setDirtyCanvas() {},
        setSize() {},
        type: "SFPauseLatent", comfyClass: "SFPauseLatent",
    };
    app.graph._nodes = [gate];
    proto.onNodeCreated.call(gate);
    check("setupNode 完成", !!gate._sfPauseLatentEls && !!gate._sfPauseLatentEls.img);
    await new Promise((r) => setTimeout(r, 5));   // queueMicrotask restore 跑完

    // ── graphToPrompt 注入：pause ──
    // 真实拓扑："5" VAEDecode 解码第一段采样器 "0" 的输出供预览（image 链）。
    promptObj = { output: {
        "0": { class_type: "KSampler", inputs: {} },
        "5": { class_type: "VAEDecode", inputs: { samples: ["0", 0] } },
        "1": { class_type: "SFPauseLatent", inputs: { latent: ["0", 0], image: ["5", 0] } },
        "3": { class_type: "SaveImage", inputs: { images: ["1", 0] } },
    } };
    await app.graphToPrompt();
    let st = JSON.parse(promptObj.output["1"].inputs.PauseState);
    check("注入 pause 模式", st.mode === "pause");
    check("注入后不剪", promptObj.output["0"] && promptObj.output["3"]);

    // ── queuePrompt：pause 剪枝删下游，保留 image 预览链接 ──
    await api.queuePrompt(0, promptObj, {});
    check("pause 剪枝删除下游 3", !promptObj.output["3"]);
    check("pause 剪枝保留上游 0/5", promptObj.output["0"] && promptObj.output["5"]);
    check("pause 保留 image 预览链接（extraInputKeys 仅 continue 生效）",
        JSON.stringify(promptObj.output["1"].inputs.image) === JSON.stringify(["5", 0]));

    // ── pass：不剪 ──
    gate.properties.pauseLatentState.gate = "pass";
    promptObj = { output: {
        "0": { class_type: "KSampler", inputs: {} },
        "5": { class_type: "VAEDecode", inputs: { samples: ["0", 0] } },
        "1": { class_type: "SFPauseLatent", inputs: { latent: ["0", 0], image: ["5", 0] } },
        "3": { class_type: "SaveImage", inputs: { images: ["1", 0] } },
    } };
    await app.graphToPrompt();
    st = JSON.parse(promptObj.output["1"].inputs.PauseState);
    check("pass 注入", st.mode === "pass");
    await api.queuePrompt(0, promptObj, {});
    check("pass 不剪且双链接保留",
        promptObj.output["3"] !== undefined &&
        "latent" in promptObj.output["1"].inputs && "image" in promptObj.output["1"].inputs);

    // ── continue（一次性提交模式）：跳第一段上游 + 删双链接 ──
    gate.id = 1;
    gate.properties.pauseLatentState.gate = "pause";
    gate._sfPauseLatentSubmitMode = "continue";   // 模拟 queueWithMode 挂载
    promptObj = { output: {
        "0": { class_type: "KSampler", inputs: {} },
        "5": { class_type: "VAEDecode", inputs: { samples: ["0", 0] } },
        "1": { class_type: "SFPauseLatent", inputs: { latent: ["0", 0], image: ["5", 0] } },
        "3": { class_type: "SaveImage", inputs: { images: ["1", 0] } },
        "4": { class_type: "SecondOut", inputs: { images: ["5", 0] } },  // 会拉活 VAEDecode→第一段
    } };
    await app.graphToPrompt();
    st = JSON.parse(promptObj.output["1"].inputs.PauseState);
    check("一次性模式 continue 注入", st.mode === "continue");
    await api.queuePrompt(0, promptObj, {});
    check("continue 删 latent 链接", !("latent" in promptObj.output["1"].inputs));
    check("continue 删 image 预览链接（否则 VAEDecode 拉活第一段）",
        !("image" in promptObj.output["1"].inputs));
    check("continue 删拉活上游的 SecondOut", !promptObj.output["4"]);
    check("continue 保留下游 SaveImage", promptObj.output["3"] !== undefined);
    check("continue 上游留存为无害孤儿", promptObj.output["0"] !== undefined && promptObj.output["5"] !== undefined);
    gate._sfPauseLatentSubmitMode = null;

    // ── executed 事件回填（sf_pause_latent_frame 键）──
    executedHandlers[0]({ detail: {
        node: 1,
        output: { sf_pause_latent_frame: [{ filename: "sf_pause_latent_1.png", subfolder: "", type: "temp", _sf_pause_meta: { prompt: { x: 1 }, workflow: { w: 1 } } }] },
    } });
    const s = gate.properties.pauseLatentState;
    check("executed 回填 frame", s.frame.filename === "sf_pause_latent_1.png" && s.frame.type === "temp");
    check("executed 捕获 exec meta", gate._sfPauseLatentExecMeta && gate._sfPauseLatentExecMeta.workflow.w === 1);
    check("预览 img.src 已设置", typeof gate._sfPauseLatentEls.img.src === "string" && gate._sfPauseLatentEls.img.src.includes("sf_pause_latent_1.png"));

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
