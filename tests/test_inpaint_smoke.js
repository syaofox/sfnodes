// SFInpaintCrop 前端冒烟测试（Node 直接运行：node tests/test_inpaint_smoke.js）
// 用 mock DOM/app/api 真实加载 sf_inpaint 系列模块，验证：
//   - 模块加载 / import 链（core/paint/render + sf_crop 基础库）
//   - graphToPrompt + queuePrompt 双注入钩子包装
//   - beforeRegisterNodeDef 原型钩子（onExecuted/onConfigure）
//   - nodeCreated：隐藏 STRING widget（SFInpaintJson）+ DOM widget + 按钮 + 预览
//   - graphToPrompt：注入 SFInpaintJson 闭包值
//   - executed 事件缓存源 URL + 预览刷新
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
    devicePixelRatio: 1,
    requestAnimationFrame: (cb) => setTimeout(cb, 0),
    cancelAnimationFrame: (id) => clearTimeout(id),
    LiteGraph: { vueNodesMode: false, registered_node_types: {} },
};
globalThis.LiteGraph = globalThis.window.LiteGraph;
globalThis.Image = class { set src(v) { this._src = v; } get src() { return this._src; } };
globalThis.FileReader = class { readAsDataURL() {} };
globalThis.WheelEvent = class { constructor(t, o) { this.type = t; Object.assign(this, o); } };

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
    loadGraphData: async () => {},
    ui: { settings: { getSettingValue: () => "Red", setSettingValueAsync: async () => {} } },
};
globalThis.window.app = globalThis.app;
const executedHandlers = [];
globalThis.api = {
    queuePrompt: async () => {},
    fetchApi: async () => ({ json: async () => ({ path: "" }) }),
    addEventListener(type, fn) { if (type === "executed") executedHandlers.push(fn); },
    apiURL: (route) => route,
};

// ── 加载模块（替换 /scripts/* import，相对 import 改 .mjs 同 tmp）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_inpaint_"));
const files = ["sf_crop_framework.js", "sf_crop_preview.js", "sf_crop_undo_guard.js",
    "sf_inpaint_geometry.js", "sf_inpaint_core.js", "sf_inpaint_paint.js",
    "sf_inpaint_render.js", "sf_inpaint.js"];
for (const n of files) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"')
        .replace(/^import "\.\/([a-z_]+)\.js";/gm, 'import "./$1.mjs";');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    await import(path.join(tmpDir, "sf_inpaint.mjs"));
    check("loadGraphData 已包装", app._sfInpaintGraphLoadWrapped === true);
    check("queuePrompt 已包装", api._sfInpaintQueueWrapped === true);
    check("扩展已注册", !!app._ext && app._ext.name === "sfnodes.InpaintCrop");

    const ext = app._ext;
    const proto = {};
    ext.beforeRegisterNodeDef({ name: "SFInpaintCrop", prototype: proto }, { name: "SFInpaintCrop" });
    check("原型钩子已安装", typeof proto.onExecuted === "function" &&
        typeof proto.onConfigure === "function");

    // ── nodeCreated ──
    const node = {
        id: 1, size: [300, 400], inputs: [], widgets: [], properties: {},
        comfyClass: "SFInpaintCrop", type: "SFInpaintCrop",
        addInput(name, type) { this.inputs.push({ name, type, link: null }); },
        addWidget(type, name, val, cb) {
            const w = { type, name, value: val, options: {}, hidden: false, callback: cb };
            this.widgets.push(w);
            return w;
        },
        addDOMWidget(name, type, el, opts) {
            const w = { name, type, el, options: opts };
            this.widgets.push(w);
            return w;
        },
        setDirtyCanvas() {},
        setSize() {},
    };
    await ext.nodeCreated(node);
    check("image/mask 输入已加", node.inputs.some((i) => i.name === "image") &&
        node.inputs.some((i) => i.name === "mask"));
    const sfJson = node.widgets.find((w) => w.name === "SFInpaintJson");
    check("隐藏 STRING widget 已建", !!sfJson && sfJson.type === "STRING" && sfJson.value === "{}");
    const domW = node.widgets.find((w) => w.name === "InpaintCropWidget");
    check("DOM widget 已建", !!domW && typeof domW.options.getValue === "function");
    const btn = node.widgets.find((w) => w.name === "Open mask editor");
    check("按钮 widget 已建", !!btn && btn.type === "button");
    check("executed 监听已注册", executedHandlers.length === 1);

    // 闭包同步 API
    node._sfInpaintJsonSync('{"project_id": "p1", "src_path": "sfnodes_inpaint/inpaint_src_x.png"}');
    check("_sfInpaintJsonSync 同步隐藏 widget", sfJson.value.includes("project_id"));
    check("_sfInpaintJsonGet 闭包", node._sfInpaintJsonGet().includes("p1"));

    // ── graphToPrompt 注入 ──
    app.graph._nodes = [node];
    promptObj = { output: {
        "1": { class_type: "SFInpaintCrop", inputs: {} },
        "2": { class_type: "VAEDecode", inputs: {} },
    } };
    await app.graphToPrompt();
    check("graphToPrompt 注入 SFInpaintJson", promptObj.output["1"].inputs.SFInpaintJson.includes("project_id"));
    check("注入不破坏其他节点", promptObj.output["2"] !== undefined);

    // ── executed 事件缓存源 URL ──
    executedHandlers[0]({ detail: {
        node: 1,
        output: { sf_inpaint_source: [{ filename: "sf_inpaint_src_abc.png", subfolder: "", type: "temp" }] },
    } });
    check("executed 缓存源 URL", !!node._sfInpaintSourceURL &&
        node._sfInpaintSourceURL.includes("sf_inpaint_src_abc.png"));
    check("properties 持久化源", node.properties?.sfInpaintSource?.filename === "sf_inpaint_src_abc.png");

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
