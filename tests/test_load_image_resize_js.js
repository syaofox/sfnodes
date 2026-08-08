// SFLoadImageResize 前端测试（Node 直接运行：node tests/test_load_image_resize_js.js）
// 覆盖：
//   - 纯函数（sf_load_image_resize.js）：previewResize 与 Python 数学对齐、
//     safeMathEval、roundToStep
//   - api 拆分函数（sf_load_image_api.js）：splitFilenameSubfolder /
//     splitTypeAnnotation / previewMatches
//   - 主扩展冒烟（mock DOM/app/api）：模块加载、扩展注册、原型钩子安装、
//     readState/writeState round-trip、graphToPrompt 状态注入 + orig_name
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
        naturalWidth: 0, naturalHeight: 0, scrollHeight: 0, scrollTop: 0,
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
    app: null, api: null,
    devicePixelRatio: 1,
    requestAnimationFrame(fn) { rafQueue.push(fn); return rafQueue.length; },
    cancelAnimationFrame() {},
    LiteGraph: { vueNodesMode: false },
};
let rafQueue = [];
globalThis.requestAnimationFrame = (fn) => { rafQueue.push(fn); return rafQueue.length; };
globalThis.cancelAnimationFrame = () => {};
function flushRaf() {
    const q = rafQueue; rafQueue = [];
    for (const fn of q) fn();
}

// ── app / api mock ──
let promptObj = { output: {} };
globalThis.app = {
    graph: { _nodes: [], links: {}, setDirtyCanvas() {} },
    canvas: { canvas: { dispatchEvent() {} }, ds: { scale: 1 } },
    extensionManager: { toast: { add() {} }, workflow: { activeWorkflow: { changeTracker: { captureCanvasState() {} } } } },
    ui: { settings: { getSettingValue() { return null; }, addSetting(cfg) { (this._added ||= []).push(cfg); } } },
    registerExtension(ext) { this._ext = ext; },
    graphToPrompt: async () => promptObj,
    loadGraphData: async (data) => { globalThis._loaded = data; },
};
globalThis.window.app = globalThis.app;
globalThis.api = { apiURL: (route) => route };

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_li_"));
function stageJs(names) {
    for (const n of names) {
        const code = fs
            .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
            .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
            .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
        fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
    }
}

(async () => {
    // ── 1. 纯函数：resize 面板引擎 ──
    stageJs(["sf_load_image_resize.js"]);
    const R = await import(path.join(tmpDir, "sf_load_image_resize.mjs"));
    const { previewResize, formatMP, safeMathEval } = R;

    const st = (p) => ({ mode: "off", max_mp: 1.0, longest_side: 1024, scale_factor: 1.0,
        fit_w: 1024, fit_h: 1024, cover_w: 1024, cover_h: 1024, ratio_w: 1, ratio_h: 1,
        ratio_action: "crop", pad_left: 0, pad_right: 0, pad_top: 0, pad_bottom: 0,
        snap: 0, resample: "auto", allow_upscale: true, ...p });

    let r = previewResize(100, 50, st({}));
    check("preview off 直通", r.w === 100 && r.h === 50);
    r = previewResize(1024, 1024, st({ mode: "max_mp", max_mp: 0.5 }));
    check("preview max_mp 0.5 -> 724²（与 Python 对齐）", r.w === 724 && r.h === 724);
    r = previewResize(100, 50, st({ mode: "fit_inside", fit_w: 1000, fit_h: 40 }));
    check("preview fit_inside -> 80x40（与 Python 对齐）", r.w === 80 && r.h === 40);
    r = previewResize(100, 50, st({ mode: "match_ratio", ratio_w: 4, ratio_h: 3 }));
    check("preview match_ratio 4:3 -> 67x50（与 Python 对齐）", r.w === 67 && r.h === 50);
    r = previewResize(724, 724, st({ mode: "max_mp", max_mp: 0.5, snap: 64 }));
    check("preview snap floor 724->704", r.w === 704 && r.h === 704);
    r = previewResize(100, 50, st({ mode: "pad", pad_left: 8, pad_right: 8, pad_top: 4, pad_bottom: 4 }));
    check("preview pad -> 116x58", r.w === 116 && r.h === 58);
    r = previewResize(100, 50, st({ mode: "cover", cover_w: 200, cover_h: 100 }));
    check("preview cover fill -> 200x100", r.w === 200 && r.h === 100);
    r = previewResize(100, 50, st({ mode: "longest_side", longest_side: 1024, allow_upscale: false }));
    check("preview longest_side 禁放大直通", r.w === 100 && r.h === 50);
    check("formatMP 1.0MP", formatMP(1024, 1024) === "1.05");

    check("safeMathEval 1024+64", safeMathEval("1024+64") === 1088);
    check("safeMathEval (1024+128)/2", safeMathEval("(1024+128)/2") === 576);
    check("safeMathEval 非法字符 -> NaN", Number.isNaN(safeMathEval("alert(1)")));
    check("safeMathEval 空 -> NaN", Number.isNaN(safeMathEval("")));
    check("safeMathEval 除零 -> NaN", Number.isNaN(safeMathEval("1/0")));

    // ── 2. api 拆分函数 ──
    stageJs(["sf_load_image_api.js"]);
    const A = await import(path.join(tmpDir, "sf_load_image_api.mjs"));
    const { splitFilenameSubfolder, splitTypeAnnotation, previewMatches } = A;
    check("split 子文件夹", JSON.stringify(splitFilenameSubfolder("Studio1/cat.png")) ===
        JSON.stringify({ subfolder: "Studio1", filename: "cat.png" }));
    check("split 根目录", JSON.stringify(splitFilenameSubfolder("cat.png")) ===
        JSON.stringify({ subfolder: "", filename: "cat.png" }));
    check("split 反斜杠归一", JSON.stringify(splitFilenameSubfolder("Studio1\\cat.png")) ===
        JSON.stringify({ subfolder: "Studio1", filename: "cat.png" }));
    check("注解剥离 [input]", JSON.stringify(splitTypeAnnotation("clipspace-x.png [input]")) ===
        JSON.stringify({ name: "clipspace-x.png", type: "input" }));
    check("注解剥离 [output]", JSON.stringify(splitTypeAnnotation("out.png [output]")) ===
        JSON.stringify({ name: "out.png", type: "output" }));
    check("无注解默认 input", splitTypeAnnotation("cat.png").type === "input");

    const mockImg = { src: "http://localhost/view?filename=cat.png&subfolder=Studio1&type=input" };
    check("previewMatches 匹配", previewMatches({ imgs: [mockImg] }, "Studio1/cat.png"));
    check("previewMatches 不匹配", !previewMatches({ imgs: [mockImg] }, "dog.png"));
    check("previewMatches 无 imgs", !previewMatches({ imgs: [] }, "cat.png"));

    // ── 3. 主扩展冒烟 ──
    stageJs(["sf_load_image.js", "sf_load_image_ui.js", "sf_load_image_api.js", "sf_load_image_resize.js"]);
    await import(path.join(tmpDir, "sf_load_image.mjs"));
    check("扩展已注册", !!app._ext && app._ext.name === "sfnodes.LoadImageResize");
    check("graphToPrompt 已包装", typeof app.graphToPrompt === "function");
    check("loadGraphData 已包装", app._sfLiGraphLoadWrapped === true);

    const ext = app._ext;
    const proto = {};
    ext.beforeRegisterNodeDef({ name: "SFLoadImageResize", prototype: proto }, { name: "SFLoadImageResize" });
    check("原型钩子已安装", typeof proto.onConnectInput === "function" &&
        typeof proto.onConnectionsChange === "function" &&
        typeof proto.onDrawForeground === "function" &&
        typeof proto.onConfigure === "function" &&
        typeof proto.onRemoved === "function");
    check("onConnectInput 拒绝输入", proto.onConnectInput.call({}) === false);

    // hideNativeImageCombo：不得误杀主 DOM widget（回归：名字未替换曾导致
    // addDOMWidget 的元素被 display:none，整个面板不可见）。真实调用顺序：
    // setup 时 widgets 尚无 sf_load_image_ui（addDOMWidget 在其后），rAF 阶段
    // 再遍历时按名字跳过。
    const U = await import(path.join(tmpDir, "sf_load_image_ui.mjs"));
    const uiNode = { widgets: [
        { name: "image", options: {}, element: { style: {} }, computeSize: () => {} },
        { name: "SFLoadImageResizeState", options: {}, element: { style: {} }, computeSize: () => {} },
    ] };
    const imgW = U.hideNativeImageCombo(uiNode);
    check("hideNativeImageCombo 返回 image widget", imgW && imgW.name === "image");
    check("image widget 元素被隐藏", uiNode.widgets[0].element.style.display === "none");
    check("image widget canvasOnly", uiNode.widgets[0].options.canvasOnly === true);
    check("state widget 元素被隐藏", uiNode.widgets[1].element.style.display === "none");
    // rAF 阶段：模拟 addDOMWidget 之后 widgets 含主 DOM widget，不得被隐藏
    const domWidget = { name: "sf_load_image_ui", options: {}, element: { style: {} }, computeSize: () => {} };
    uiNode.widgets.push(domWidget);
    flushRaf();
    check("主 DOM widget 元素未被误隐藏", domWidget.element.style.display !== "none");

    // 设置注册：ThumbSize + 文件夹侧栏宽度两项
    const addedIds = (app.ui.settings._added || []).map((c) => c.id);
    check("ThumbSize 设置已注册", addedIds.includes("sfnodes.LoadImage.ThumbSize"));
    check("侧栏宽度设置已注册", addedIds.includes("sfnodes.LoadImage.BrowserFolderWidth"));
    const fwCfg = (app.ui.settings._added || []).find((c) => c.id === "sfnodes.LoadImage.BrowserFolderWidth");
    check("侧栏宽度默认 104 / slider", fwCfg && fwCfg.defaultValue === 104 && fwCfg.type === "slider" &&
        fwCfg.attrs && fwCfg.attrs.min === 60 && fwCfg.attrs.max === 240);

    // graphToPrompt 注入：模拟节点 + prompt
    const node = {
        id: 7, type: "SFLoadImageResize", comfyClass: "SFLoadImageResize",
        properties: { sfLoadImageResizeState: JSON.stringify({ mode: "max_mp", max_mp: 0.5 }) },
        _sfLiSelectedFilename: "Studio1/cat.png",
        _sfLiOrigName: "Studio1/cat.png",
        _sfLiImageWidget: { value: "Studio1/cat.png" },
    };
    app.graph._nodes = [node];
    promptObj = { output: { "7": { class_type: "SFLoadImageResize", inputs: { image: "Studio1/cat.png" } } } };
    const out = await app.graphToPrompt();
    const entry = out.output["7"];
    check("graphToPrompt 注入状态", JSON.parse(entry.inputs.SFLoadImageResizeState).mode === "max_mp");
    check("graphToPrompt 注入 orig_name", JSON.parse(entry.inputs.SFLoadImageResizeState).orig_name === "Studio1/cat.png");
    check("graphToPrompt image 保持", entry.inputs.image === "Studio1/cat.png");

    // readState/writeState round-trip（DEFAULT_STATE 与 Python 侧一致）
    const stateStr = node.properties.sfLoadImageResizeState;
    check("DEFAULT_STATE pad_color 对齐 Python", JSON.parse(stateStr).mode === "max_mp");

    console.log();
    if (failures.length) {
        console.log(failures.length + " FAILURES:", failures);
        process.exit(1);
    }
    console.log("ALL PASS");
})().catch((e) => { console.error("LOAD ERROR:", e); process.exit(1); });
