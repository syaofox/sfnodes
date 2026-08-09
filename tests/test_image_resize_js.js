// SFImageResize 前端测试（Node 直接运行：node tests/test_image_resize_js.js）
// 覆盖：
//   - 纯函数（sf_image_resize_lib.js）：effectiveWiredState 与 Python
//     _apply_wired_size 镜像一致（longest_side 优先 / 单轴 / 双轴 / 0 值直通）、
//     readWiredInt 读取规则、getReadoutInfo 分支
//   - 主扩展冒烟（mock DOM/app/api）：模块加载、扩展注册、原型钩子安装、
//     graphToPrompt 隐藏状态注入
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素，先例 test_load_image_resize_js.js）──
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
    extensionManager: { toast: { add() {} } },
    registerExtension(ext) { this._ext = ext; },
    graphToPrompt: async () => promptObj,
    loadGraphData: async (data) => { globalThis._loaded = data; },
};
globalThis.window.app = globalThis.app;
globalThis.api = {
    apiURL: (route) => route,
    addEventListener(type, fn) { (this._execHandlers ||= []).push([type, fn]); },
};

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ir_"));
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

// FakeNode：wired 输入模拟（inputs/graph.links/getNodeById/widgets）
function makeGraph() {
    const nodes = [];
    return {
        nodes,
        _nodes: nodes,
        links: {},
        getNodeById(id) { return nodes.find((n) => n.id === id) || null; },
    };
}
let _upId = 10;
function makeUpstream(widgets, id = null) {
    const nid = id != null ? id : _upId++;
    return { id: nid, widgets: widgets || [] };
}
function wireNode(node, name, up, slot = 0) {
    const inp = node.inputs.find((i) => i.name === name);
    const linkId = Math.random();
    inp.link = linkId;
    node.graph.links[linkId] = { origin_id: up.id, origin_slot: slot };
    node.graph.nodes.push(up);
}
function makeNode(graph) {
    const node = {
        id: 1, graph,
        properties: {},
        inputs: [
            { name: "image", type: "IMAGE", link: null },
            { name: "mask", type: "MASK", link: null },
            { name: "width", type: "INT", link: null },
            { name: "height", type: "INT", link: null },
            { name: "longest_side", type: "INT", link: null },
        ],
        widgets: [],
        setDirtyCanvas() {}, computeSize() { return [360, 340]; },
        size: [360, 340],
    };
    graph.nodes.push(node);
    return node;
}

(async () => {
    // ── 1. lib 纯函数 ──
    stageJs(["sf_image_resize_lib.js", "sf_load_image_resize.js"]);
    const L = await import(path.join(tmpDir, "sf_image_resize_lib.mjs"));
    const { effectiveWiredState, wireInfo, readWiredInt, isWired, readState, writeState, getReadoutInfo } = L;

    const DEFAULTS = { mode: "off", max_mp: 1.0, longest_side: 1024, scale_factor: 1.0,
        fit_w: 1024, fit_h: 1024, cover_w: 1024, cover_h: 1024, ratio_w: 1, ratio_h: 1,
        ratio_action: "crop", pad_left: 0, pad_right: 0, pad_top: 0, pad_bottom: 0,
        snap: 0, resample: "auto", allow_upscale: true };

    // readState / writeState round-trip
    const n1 = { properties: {} };
    writeState(n1, "sfImageResizeState", { ...DEFAULTS, mode: "pad" });
    const s1 = readState(n1, "sfImageResizeState", DEFAULTS);
    check("read/writeState round-trip", s1.mode === "pad");
    check("readState 坏 JSON -> 默认", readState({ properties: { sfImageResizeState: "{" } }, "sfImageResizeState", DEFAULTS).mode === "off");
    check("readState 空 -> 默认", readState({}, "sfImageResizeState", DEFAULTS).mode === "off");

    // isWired
    const g = makeGraph();
    const node = makeNode(g);
    check("isWired 未接", isWired(node, "width") === false);
    wireNode(node, "width", makeUpstream([{ value: 512 }]));
    check("isWired 已接", isWired(node, "width") === true);

    // readWiredInt：单数值 widget 信任
    check("readWiredInt 单数值", readWiredInt(node, "width") === 512);
    check("readWiredInt 未接 -> null", readWiredInt(node, "height") === null);
    wireNode(node, "height", makeUpstream([{ value: 100 }, { value: 200 }]));
    check("readWiredInt 多数值 widget -> null", readWiredInt(node, "height") === null);
    wireNode(node, "longest_side", makeUpstream([{ value: "combo" }]));
    check("readWiredInt 字符串 widget -> null", readWiredInt(node, "longest_side") === null);

    // wireInfo
    const info = wireInfo(node);
    check("wireInfo 计数（count 只统计 width/height）", info.count === 2 && info.wiredW && info.wiredH && info.wiredLongest);
    check("wireInfo valW", info.valW === 512);

    // effectiveWiredState —— 镜像 Python _apply_wired_size
    const st = (p) => ({ ...DEFAULTS, ...p });
    let e = effectiveWiredState(st({}), { wiredW: false, wiredH: false, wiredLongest: false, count: 0, valW: null, valH: null, valLongest: null }, 100, 50);
    check("无 wired -> 原 state", e.mode === "off");

    e = effectiveWiredState(st({}), { wiredW: false, wiredH: false, wiredLongest: true, count: 0, valW: null, valH: null, valLongest: 1024 }, 100, 50);
    check("longest_side -> mode=longest_side", e.mode === "longest_side" && e.longest_side === 1024);

    e = effectiveWiredState(st({}), { wiredW: false, wiredH: false, wiredLongest: true, count: 0, valW: null, valH: null, valLongest: 0 }, 100, 50);
    check("longest_side 0 -> off", e.mode === "off");

    e = effectiveWiredState(st({}), { wiredW: true, wiredH: false, wiredLongest: false, count: 1, valW: 200, valH: null, valLongest: null }, 100, 50);
    check("只接 width -> scale_factor 2.0", e.mode === "scale_factor" && Math.abs(e.scale_factor - 2.0) < 1e-9);

    e = effectiveWiredState(st({}), { wiredW: false, wiredH: true, wiredLongest: false, count: 1, valW: null, valH: 100, valLongest: null }, 100, 50);
    check("只接 height -> scale_factor 2.0", e.mode === "scale_factor" && Math.abs(e.scale_factor - 2.0) < 1e-9);

    e = effectiveWiredState(st({}), { wiredW: true, wiredH: false, wiredLongest: false, count: 1, valW: 0, valH: null, valLongest: null }, 100, 50);
    check("只接 width 0 -> off", e.mode === "off");

    e = effectiveWiredState(st({}), { wiredW: true, wiredH: true, wiredLongest: false, count: 2, valW: 50, valH: 50, valLongest: null }, 100, 50);
    check("双接 -> cover 精确盒", e.mode === "cover" && e.cover_w === 50 && e.cover_h === 50);

    e = effectiveWiredState(st({ mode: "fit_inside" }), { wiredW: true, wiredH: true, wiredLongest: false, count: 2, valW: 50, valH: 50, valLongest: null }, 100, 50);
    check("双接 + fit_inside -> 保持 fit_inside", e.mode === "fit_inside" && e.fit_w === 50 && e.fit_h === 50);

    e = effectiveWiredState(st({ mode: "pad" }), { wiredW: true, wiredH: true, wiredLongest: false, count: 2, valW: 100, valH: null, valLongest: null }, 100, 50);
    check("双接但 val 不可读 -> 原 state", e.mode === "pad");

    // getReadoutInfo
    const baseWi = { wiredW: false, wiredH: false, wiredLongest: false, count: 0, valW: null, valH: null, valLongest: null };
    let r = getReadoutInfo(st({ mode: "scale_factor", scale_factor: 2 }), null, { w: 100, h: 50 }, baseWi);
    check("readout live 计算", r.mode === "dual" && r.inW === 100 && r.outW === 200);
    r = getReadoutInfo(st({}), null, null, baseWi);
    check("readout 无 live 无缓存 -> null", r === null);
    r = getReadoutInfo(st({}), { in_w: 10, in_h: 10, out_w: 20, out_h: 20 }, null, baseWi);
    check("readout 缓存回退", r.mode === "dual" && r.outW === 20);
    r = getReadoutInfo(st({}), null, { w: 100, h: 50 },
        { wiredW: true, wiredH: true, wiredLongest: false, count: 2, valW: null, valH: null, valLongest: null });
    check("readout wired 不可读 -> msg", r.mode === "msg");
    r = getReadoutInfo(st({}), null, { w: 100, h: 50 },
        { wiredW: false, wiredH: false, wiredLongest: true, count: 0, valW: null, valH: null, valLongest: 512 });
    check("readout wired longest 可读 -> 计算", r.mode === "dual" && r.outW === 512 && r.outH === 256);

    // ── 2. 主扩展冒烟 ──
    stageJs(["sf_common.js", "sf_image_resize.js", "sf_image_resize_ui.js", "sf_image_resize_lib.js",
        "sf_load_image_resize.js", "sf_load_image_ui.js", "sf_load_image_api.js"]);
    const M = await import(path.join(tmpDir, "sf_image_resize.mjs"));
    check("扩展已注册", globalThis.app._ext?.name === "sfnodes.ImageResize");
    const ext = globalThis.app._ext;

    // 原型钩子安装：注册一个伪节点类型
    const proto = {};
    LiteGraph = { INPUT: 1, NODE_SLOT_HEIGHT: 20 };
    globalThis.LiteGraph = LiteGraph;
    globalThis.window.LiteGraph = LiteGraph;
    ext.beforeRegisterNodeDef({ prototype: proto }, { name: "SFImageResize" });
    check("onNodeCreated 已安装", typeof proto.onNodeCreated === "function");
    check("onConfigure 已安装", typeof proto.onConfigure === "function");
    check("onConnectionsChange 已安装", typeof proto.onConnectionsChange === "function");
    check("onRemoved 已安装", typeof proto.onRemoved === "function");
    check("onResize 已安装", typeof proto.onResize === "function");
    check("onDrawForeground 已安装", typeof proto.onDrawForeground === "function");

    // 非目标节点不安装
    ext.beforeRegisterNodeDef({ prototype: {} }, { name: "OtherNode" });
    check("非目标节点不安装", true);

    // graphToPrompt 注入
    promptObj = { output: { "1": { class_type: "SFImageResize", inputs: {} } } };
    const up2 = makeUpstream([], 1);
    up2.comfyClass = "SFImageResize";
    up2.properties = { sfImageResizeState: JSON.stringify({ ...DEFAULTS, mode: "max_mp" }) };
    globalThis.app.graph = { _nodes: [up2], links: {}, getNodeById() { return null; } };
    const res = await globalThis.app.graphToPrompt();
    check("graphToPrompt 注入隐藏状态", res.output["1"].inputs.SFImageResizeState === up2.properties.sfImageResizeState);

    // executed 回填
    const execNode = { id: 5, comfyClass: "SFImageResize", properties: {}, setDirtyCanvas() {} };
    globalThis.app.graph.getNodeById = (id) => (id === 5 ? execNode : null);
    globalThis.api._execHandlers ||= [];
    // api.addEventListener mock 在 stageJs 后覆盖了全局 api —— 重新捕获
    const execHandler = globalThis.api._execHandlers?.find(([e]) => e === "executed");
    if (execHandler) {
        execHandler[1]({ detail: { node: 5, output: { sf_image_resize: [{ in_w: 10, in_h: 10, out_w: 20, out_h: 20 }] } } });
        check("executed 回填 dims", execNode.properties.sfIrDims?.out_w === 20);
    } else {
        check("executed 回填 dims", false);
    }

    console.log();
    if (failures.length) {
        console.log(failures.length + " FAILURES: " + failures.join(", "));
        process.exit(1);
    }
    console.log("ALL PASS");
})();
