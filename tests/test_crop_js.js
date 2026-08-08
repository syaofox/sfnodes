// SFImageCrop 前端测试（Node 直接运行：node tests/test_crop_js.js）
// 覆盖：
//   - 纯逻辑：alignments（computeAlignedXY / defaultAlignForMeta）、
//     core 常量（RATIOS / SNAPS）、undo_guard 安装
//   - 主扩展冒烟（mock DOM/app/api）：模块加载、扩展注册、原型钩子、
//     nodeCreated 不抛错、onExecuted 抑制原生预览
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素，先例 test_pause_image_smoke.js）──
function makeEl() {
    const style = { setProperty() {}, getPropertyValue() { return ""; } };
    return {
        style, dataset: {}, children: [],
        className: "", textContent: "", innerHTML: "", value: "", placeholder: "",
        type: "", title: "", rows: 1, spellcheck: false, disabled: false, checked: false,
        draggable: false, isConnected: true, offsetWidth: 100, offsetHeight: 20,
        naturalWidth: 0, naturalHeight: 0, scrollHeight: 0, scrollTop: 0, width: 0, height: 0,
        classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        replaceWith() {}, replaceChildren(...kids) { this.children = kids; },
        insertBefore(c, ref) { this.children.push(c); return c; },
        remove() { this.removed = true; },
        contains() { return false; }, closest() { return null; },
        querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        addEventListener() {}, removeEventListener() {},
        focus() {}, blur() {}, select() {}, click() {}, setAttribute() {},
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {}, setPointerCapture() {}, releasePointerCapture() {}, setSelectionRange() {},
        getContext() { return { drawImage() {}, fillRect() {}, beginPath() {}, stroke() {}, fill() {},
            measureText() { return { width: 10 }; }, arcTo() {}, moveTo() {}, lineTo() {}, closePath() {},
            rect() {}, roundRect() {}, fillText() {}, strokeRect() {}, strokeStyle: "", fillStyle: "",
            font: "", textAlign: "", textBaseline: "", lineWidth: 1, lineCap: "", lineJoin: "" }; },
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
    createElementNS() { const el = makeEl(); el.setAttribute = () => {}; return el; },
    createTextNode(t) { return { textContent: t, nodeType: 3 }; },
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
    getElementById() { return null; },
    activeElement: makeEl(),
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
    app: null, api: null,
    devicePixelRatio: 1,
    requestAnimationFrame(fn) { rafQueue.push(fn); return rafQueue.length; },
    cancelAnimationFrame() {},
    showSaveFilePicker: undefined,
    LiteGraph: { vueNodesMode: false },
};
let rafQueue = [];
globalThis.requestAnimationFrame = (fn) => { rafQueue.push(fn); return rafQueue.length; };
globalThis.cancelAnimationFrame = () => {};
function flushRaf() {
    const q = rafQueue; rafQueue = [];
    for (const fn of q) fn();
}
globalThis.ResizeObserver = class { constructor() {} observe() {} disconnect() {} };
class MockHTMLInput { constructor() { this.type = ""; this.closest = () => null; this.style = {}; } }
Object.defineProperty(MockHTMLInput.prototype, "value", { configurable: true, get() { return this._v; }, set(v) { this._v = v; } });
globalThis.HTMLInputElement = MockHTMLInput;

let promptObj = { output: {} };
globalThis.app = {
    graph: { _nodes: [], links: {}, setDirtyCanvas() {}, getNodeById() { return null; }, remove() {} },
    canvas: { canvas: { dispatchEvent() {} }, ds: { scale: 1 }, selected_nodes: {}, current_node: null, node_over: null },
    extensionManager: { toast: { add() {} } },
    registerExtension(ext) { this._ext = ext; },
    graphToPrompt: async () => promptObj,
    loadGraphData: async () => {},
};
globalThis.window.app = globalThis.app;
const executedHandlers = [];
globalThis.api = {
    fetchApi: async () => ({ json: async () => ({ path: "", composite_path: "" }) }),
    addEventListener(type, fn) { if (type === "executed") executedHandlers.push(fn); },
    removeEventListener() {},
    apiURL: (route) => route,
    queuePrompt: async function (...args) { globalThis._queueArgs = args; },
};

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_crop_"));
function stageJs(names) {
    for (const n of names) {
        const code = fs
            .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
            .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
            .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"')
            .replace(/import "\.\/([a-z_]+)\.js";/g, 'import "./$1.mjs";');
        fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
    }
}

(async () => {
    // ── 1. 纯逻辑：alignments ──
    stageJs(["sf_crop_alignments.js"]);
    const A = await import(path.join(tmpDir, "sf_crop_alignments.mjs"));
    const { ALIGNMENTS, computeAlignedXY, defaultAlignForMeta } = A;
    check("ALIGNMENTS 10 项", ALIGNMENTS.length === 10);
    check("computeAlignedXY center", JSON.stringify(computeAlignedXY("mc", 100, 50, { w: 300, h: 200 })) ===
        JSON.stringify({ x: 100, y: 75 }));
    check("computeAlignedXY top-left", JSON.stringify(computeAlignedXY("tl", 100, 50, { w: 300, h: 200 })) ===
        JSON.stringify({ x: 0, y: 0 }));
    check("computeAlignedXY bottom-right", JSON.stringify(computeAlignedXY("br", 100, 50, { w: 300, h: 200 })) ===
        JSON.stringify({ x: 200, y: 150 }));
    check("computeAlignedXY free 回 null", computeAlignedXY("free", 100, 50, { w: 300, h: 200 }) === null);
    check("computeAlignedXY 无 dims 回 null", computeAlignedXY("mc", 100, 50, null) === null);
    check("defaultAlignForMeta 新节点居中", defaultAlignForMeta({}) === "mc");
    check("defaultAlignForMeta 已有裁剪保持 free", defaultAlignForMeta({ crop_w: 100 }) === "free");

    // ── 2. core 常量 ──
    stageJs(["sf_crop_undo_guard.js", "sf_crop_alignments.js", "sf_crop_framework.js", "sf_crop_core.js"]);
    const C = await import(path.join(tmpDir, "sf_crop_core.mjs"));
    check("RATIOS 10 项", C.RATIOS.length === 10);
    check("SNAPS 5 项", C.SNAPS.length === 5);
    check("RATIOS[0] Free", C.RATIOS[0].label === "Free");
    check("RATIOS[1] 1:1", C.RATIOS[1].w === 1 && C.RATIOS[1].h === 1);
    check("CropAPI.saveComposite 走新路由", C.CropAPI.saveComposite.toString().includes("/api/sfnodes/crop/save"));
    check("saveComposite 不抛 ReferenceError", (async () => {
        try { await C.CropAPI.saveComposite("p1", "data:image/png;base64,x"); return true; }
        catch (e) { return !(e instanceof ReferenceError); }
    })());

    // ── 3. 主扩展冒烟 ──
    const ALL = ["sf_crop.js", "sf_crop_framework.js", "sf_crop_preview.js", "sf_crop_undo_guard.js",
        "sf_crop_alignments.js", "sf_crop_core.js", "sf_crop_panel.js", "sf_crop_interaction.js",
        "sf_crop_render.js"];
    stageJs(ALL);
    await import(path.join(tmpDir, "sf_crop.mjs"));
    check("扩展已注册", !!app._ext && app._ext.name === "sfnodes.ImageCrop");
    check("loadGraphData 已包装", app._sfCropGraphLoadWrapped === true);

    const ext = app._ext;
    const proto = {};
    ext.beforeRegisterNodeDef({ name: "SFImageCrop", prototype: proto }, { name: "SFImageCrop" });
    check("原型钩子已安装", typeof proto.onExecuted === "function" && typeof proto.onConfigure === "function");

    // nodeCreated：mock 节点走完整 setup 不抛错
    const node = {
        id: 5, type: "SFImageCrop", comfyClass: "SFImageCrop",
        size: [300, 380], inputs: [], outputs: [], widgets: [], flags: {},
        properties: {},
        imgs: null,
        graph: app.graph,
        addInput() { this.inputs.push({ name: arguments[0], type: arguments[1] }); },
        addWidget(type, name, value, cb) {
            const w = { type, name, value, options: {}, computeSize: () => [0, -4], callback: cb };
            this.widgets.push(w);
            return w;
        },
        addDOMWidget(name, type, el, opts = {}) {
            // 模拟 Vue DOMWidget：value getter 调 getValue、setter 调 setValue
            const w = { options: opts, name, element: el, computeLayoutSize() {} };
            Object.defineProperty(w, "value", {
                configurable: true,
                get() { return opts.getValue ? opts.getValue() : undefined; },
                set(v) { if (opts.setValue) opts.setValue(v); },
            });
            (this.domWidgets ||= []).push(w);
            return w;
        },
        setDirtyCanvas() {},
        setSize() {},
        disconnectInput() {},
        onRemoved: null,
    };
    let created = true;
    try { ext.nodeCreated(node); flushRaf(); } catch (e) { created = false; console.log("nodeCreated error:", e); }
    check("nodeCreated 不抛错", created);
    check("image 输入槽已加", node.inputs.some((i) => i.name === "image" && i.type === "IMAGE"));
    check("mask 输入槽已加", node.inputs.some((i) => i.name === "mask" && i.type === "MASK"));
    check("Open Crop 按钮已加", node.widgets.length >= 1);
    check("executed 监听已注册", executedHandlers.length === 1);

    // 隐藏状态 widget（SFCropJson）：数据载体随 workflow 保存/加载
    const jsonWidget = node.widgets.find((w) => w.type === "STRING" && w.name === "SFCropJson");
    check("SFCropJson 隐藏 widget 已加", !!jsonWidget);
    check("_sfCropJsonGet 暴露闭包", typeof node._sfCropJsonGet === "function" && node._sfCropJsonGet() === "{}");

    // graphToPrompt 注入（回归：CropWidget 不在 schema 会被前端剥离，
    // SFCropJson 在 Python hidden 声明内 → 不被剥离）
    app.graph._nodes = [node];
    node._sfCropJsonSync('{"crop_x":10,"crop_y":20,"crop_w":100,"crop_h":50,"project_id":"p1"}');
    promptObj = { output: { "5": { class_type: "SFImageCrop", inputs: { image: "img.png" } } } };
    const injected = await app.graphToPrompt();
    const ci = injected.output["5"].inputs.SFCropJson;
    check("graphToPrompt 注入 SFCropJson", typeof ci === "string");
    check("注入的 crop_json 是当前值", JSON.parse(ci).crop_w === 100);
    check("SFCropJson widget 值同步", jsonWidget.value === '{"crop_x":10,"crop_y":20,"crop_w":100,"crop_h":50,"project_id":"p1"}');

    // 回归：_sfCropJsonSync 不得写 DOM widget.value（Vue setter → setValue →
    // _sfCropJsonSync 无限递归，Maximum call stack size exceeded）
    let recursive = false;
    try {
        node._sfCropJsonSync('{"crop_w":1}');
        // 模拟 Vue 加载恢复 / 保存路径的 setter 链：写 DOM widget.value
        for (const dw of node.domWidgets) dw.value = { crop_json: '{"crop_w":2}' };
    } catch (e) {
        recursive = e instanceof RangeError || /stack/i.test(String(e));
    }
    check("DOM widget setter 链无递归", !recursive);
    check("setter 链后 cropJson 更新", node._sfCropJsonGet().includes('"crop_w":2'));

    // queuePrompt 提交时注入（回归：graphToPrompt 注入在部分前端 Run 路径被
    // 丢弃，后端收到 CropWidget=None → 透传原图。queuePrompt 是唯一漏斗）
    globalThis._queueArgs = null;
    await api.queuePrompt(1, { output: { "5": { class_type: "SFImageCrop", inputs: {} } }, workflow: {} }, {});
    const qEntry = globalThis._queueArgs?.[1]?.output?.["5"];
    check("queuePrompt 注入 SFCropJson", typeof qEntry?.inputs?.SFCropJson === "string" &&
        JSON.parse(qEntry.inputs.SFCropJson).crop_w === 2);
    check("queuePrompt 包装幂等", api._sfCropQueueWrapped === true);

    // 点击 Open Crop：完整编辑器构建链路（回归：CANVAS_RATIOS 漏提取曾导致
    // createCanvasSettings ReferenceError，编辑器不弹出）
    const openBtn = node.widgets.find((w) => w.name === "Open Crop");
    let openOk = true;
    try { openBtn.callback(); flushRaf(); } catch (e) { openOk = false; console.log("Open Crop error:", e.message); }
    check("Open Crop 点击不抛错", openOk);
    check("编辑器实例已创建", !!node._sfCropEditor);
    check("编辑器 overlay 已挂载", node._sfCropEditor?.el?.overlay?.isConnected === true);

    // onExecuted 抑制原生预览
    const execNode = { imgs: [{ src: "x" }] };
    proto.onExecuted.call(execNode, {});
    check("onExecuted 清空 imgs", execNode.imgs === null);

    // CropAPI 保存走 sfnodes 路由
    const saveBody = C.CropAPI.saveComposite.toString();
    check("路由已替换", !saveBody.includes("/pixaroma/") && saveBody.includes("/api/sfnodes/crop/save"));

    console.log();
    if (failures.length) {
        console.log(failures.length + " FAILURES:", failures);
        process.exit(1);
    }
    console.log("ALL PASS");
    process.exit(0);
})().catch((e) => { console.error("LOAD ERROR:", e); process.exit(1); });
