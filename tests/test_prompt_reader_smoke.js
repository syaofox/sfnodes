// SFPromptReader 主扩展端到端冒烟测试（Node 直接运行：node tests/test_prompt_reader_smoke.js）
// 用 mock DOM/app/api 真实加载模块，验证：
//   - 模块加载 / 扩展注册
//   - beforeRegisterNodeDef 包装 prototype（onConfigure / onConnectionsChange /
//     onRemoved / onSelected / onDeselected）
//   - nodeCreated setupNode：DOM widget 构建、image widget 隐藏、node.imgs 抑制、
//     image widget callback 包装、初始 microtask 提取
//   - extract 请求打到 /api/sfnodes/prompt_reader/extract + 状态持久化
//   - filename 接线 → refreshWiredState（wired UI + 跟随 timer）
//   - onRemoved 清理
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
    return {
        style: {}, dataset: {}, children: [], _handlers: {},
        className: "", textContent: "", innerHTML: "", value: "", placeholder: "",
        type: "", title: "", readOnly: false, disabled: false, isConnected: true,
        offsetWidth: 100, offsetHeight: 20,
        classList: {
            _s: new Set(),
            add(...c) { c.forEach((x) => this._s.add(x)); },
            remove(...c) { c.forEach((x) => this._s.delete(x)); },
            toggle(c, force) { if (force === undefined) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); } else { force ? this._s.add(c) : this._s.delete(c); } },
            contains(c) { return this._s.has(c); },
        },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        remove() { this.removed = true; },
        contains() { return false; },
        // 按选择器缓存返回同一元素，便于测试拿到 setupNode 挂的事件 handler
        querySelector(sel) {
            if (!this._qsCache) this._qsCache = {};
            if (!this._qsCache[sel]) this._qsCache[sel] = makeEl();
            return this._qsCache[sel];
        },
        querySelectorAll() { return []; },
        addEventListener(name, fn) { this._handlers[name] = fn; },
        removeEventListener() {},
        select() {}, click() {},
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {},
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
    getElementById() { return null; },
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    activeElement: makeEl(),
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    getComputedStyle() { return { position: "static", display: "block" }; },
    innerWidth: 1280, innerHeight: 720,
    LiteGraph: { vueNodesMode: false },
};
globalThis.LiteGraph = { INPUT: 1, OUTPUT: 2 };
globalThis.requestAnimationFrame = (fn) => fn();
globalThis.setInterval = () => 0;   // 跟随 timer 保持 no-op，不挂住进程
globalThis.clearInterval = () => {};
globalThis.navigator = { clipboard: { writeText: async () => {} } };
globalThis.FormData = class { append() {} };

// ── app / api mock ──
let extractCalls = [];
let listCalls = [];
globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/sfnodes/prompt_reader/extract")) {
        extractCalls.push(u);
        return { ok: true, status: 200, json: async () => ({ found: true, text: "smoke prompt", source: "comfyui" }) };
    }
    if (u.includes("/api/sfnodes/prompt_reader/list")) {
        const type = new URL(u, "http://localhost").searchParams.get("type");
        listCalls.push(type);
        return { ok: true, json: async () => (type === "output" ? ["out1.png", "out2.mp4"] : ["smoke.png", "uploaded.png"]) };
    }
    if (u.startsWith("/upload/image")) {
        return { ok: true, json: async () => ({ name: "uploaded.png" }) };
    }
    throw new Error("unexpected fetch: " + u);
};
globalThis.app = {
    graph: { _nodes: [], links: {}, getNodeById() { return null; }, setDirtyCanvas() {} },
    registerExtension(ext) { this._ext = ext; },
    loadGraphData: async () => {},
};

function makeNode() {
    const imageWidget = {
        name: "image", value: "smoke.png", hidden: false,
        options: { values: ["smoke.png"] },
        computeSize: () => [0, 0], element: null, callback: null,
    };
    return {
        id: "1", comfyClass: "SFPromptReader", type: "SFPromptReader",
        widgets: [imageWidget], inputs: [], outputs: [], properties: {},
        size: [400, 300],
        graph: { setDirtyCanvas() {}, links: {}, getNodeById() { return null; } },
        addDOMWidget(name, type, el, opts) {
            const w = { name, type, options: opts || {}, element: el, value: null };
            this.widgets.push(w);
            return w;
        },
        // 模拟真实 LiteGraph：断开触发 onConnectionsChange 级联（unwire →
        // refreshWiredState → onImageChanged），drop/手动接管依赖这条链刷新。
        disconnectInput(i) {
            const inp = this.inputs[i];
            if (inp && inp.link != null) {
                inp.link = null;
                _fakeType.prototype.onConnectionsChange.call(this, 1, i, false, null, { name: inp.name });
            }
        },
    };
}

// makeNode 定义在顶层，FakeType 在 async IIFE 内创建：经模块级变量桥接
let _fakeType = null;

// ── 加载模块（替换 /scripts/* import，相对 import 改 .mjs 同 tmp）──
(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pr_smoke_"));
    for (const n of ["sf_common.js", "sf_prompt_reader.js"]) {
        const code = fs.readFileSync(path.join(__dirname, "..", "web", n), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
            .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
            .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
        fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
    }
    await import(path.join(tmpDir, "sf_prompt_reader.mjs"));

    check("扩展已注册", app._ext?.name === "sfnodes.PromptReader");

    // beforeRegisterNodeDef 包装
    const FakeType = function () {};
    _fakeType = FakeType;
    app._ext.beforeRegisterNodeDef(FakeType, { name: "SFPromptReader" });
    check("onConfigure 已包装", typeof FakeType.prototype.onConfigure === "function" && FakeType.prototype.onConfigure !== Object.getPrototypeOf(FakeType).onConfigure);
    check("onConnectionsChange 已包装", typeof FakeType.prototype.onConnectionsChange === "function");
    check("onRemoved 已包装", typeof FakeType.prototype.onRemoved === "function");
    check("onSelected 已包装", typeof FakeType.prototype.onSelected === "function");

    // nodeCreated → setupNode
    const node = makeNode();
    app._ext.nodeCreated(node);
    check("DOM widget 已添加", node.widgets.some((w) => w.name === "sf_prompt_reader_ui"));
    check("_sfPrRoot 已构建", !!node._sfPrRoot);
    check("image widget 引用", node._sfPrImageWidget?.name === "image");
    check("image widget 已隐藏", node._sfPrImageWidget.hidden === true);
    check("node.imgs 被抑制", Array.isArray(node.imgs) && node.imgs.length === 0);
    check("image widget callback 已包装", typeof node._sfPrImageWidget.callback === "function");
    check("默认尺寸", node.size[0] === 400 && node.size[1] === 300);
    check("_sfPrSelectedFilename 已种子", node._sfPrSelectedFilename === "smoke.png");

    // 初始 microtask：widget 有值 → onImageChanged → extract
    await new Promise((r) => setTimeout(r, 20));
    check("初始提取已请求", extractCalls.length === 1 && extractCalls[0].includes("/api/sfnodes/prompt_reader/extract"));
    check("提取带 filename 参数", extractCalls[0].includes("smoke.png"));
    check("状态已持久化", node.properties?.promptReaderState?.found === true
        && node.properties.promptReaderState.text === "smoke prompt"
        && node.properties.promptReaderState.filename === "smoke.png");

    // filename 接线 → refreshWiredState（wired UI + 跟随 timer，不炸）
    node.inputs.push({ name: "filename", link: 42 });
    try {
        FakeType.prototype.onConnectionsChange.call(node, 1, 1, true, { origin_id: 9, origin_slot: 0 }, { name: "filename" });
        check("filename 接线刷新不炸", true);
    } catch (e) {
        check("filename 接线刷新不炸", false);
    }
    // 其他输入/输出变化不触发 refreshWiredState（不炸即可）
    try {
        FakeType.prototype.onConnectionsChange.call(node, 1, 1, true, { origin_id: 9, origin_slot: 0 }, { name: "text" });
        FakeType.prototype.onConnectionsChange.call(node, 2, 1, true, { origin_id: 9, origin_slot: 0 }, { name: "text" });
        check("其他槽变化不炸", true);
    } catch (e) {
        check("其他槽变化不炸", false);
    }

    // 拖拽 drop：video/mp4 放行并上传+提取；非媒体类型拒绝；空 type 放行
    const root = node._sfPrRoot;
    const dropHandler = root?._handlers?.drop;
    check("drop handler 已挂", typeof dropHandler === "function");
    const uploadsBefore = extractCalls.length;
    const ev = (type) => ({ preventDefault() {}, stopPropagation() {}, dataTransfer: { files: [{ type, name: "clip.mp4" }] } });
    if (typeof dropHandler === "function") {
        await dropHandler(ev("video/mp4"));
        await new Promise((r) => setTimeout(r, 20));
        check("drop mp4: 提取已触发", extractCalls.length === uploadsBefore + 1);
        check("drop mp4: 新文件进了 combo options", node._sfPrImageWidget.options.values.includes("uploaded.png"));

        const before = extractCalls.length;
        await dropHandler(ev("text/plain"));
        await new Promise((r) => setTimeout(r, 20));
        check("drop 非媒体类型被拒", extractCalls.length === before);

        await dropHandler(ev(""));
        await new Promise((r) => setTimeout(r, 20));
        check("drop 空 type 放行", extractCalls.length === before + 1);
    }

    // ── 目录切换（IN/OUT 按钮）──
    const srcBtn = root._qsCache?.['[data-role="source"]'];
    const srcClick = srcBtn?._handlers?.click;
    check("source 按钮已挂", typeof srcClick === "function");
    if (typeof srcClick === "function") {
        const before = extractCalls.length;
        await srcClick({ stopPropagation() {} });
        await new Promise((r) => setTimeout(r, 20));
        check("切到 output: 已拉取列表", listCalls.at(-1) === "output");
        const vals = node._sfPrImageWidget.options.values;
        check("切到 output: options 带 [output] 注解",
            vals.length === 2 && vals.every((v) => v.endsWith(" [output]")) && vals[0] === "out1.png [output]");
        check("切到 output: 选中第一项", node._sfPrImageWidget.value === "out1.png [output]");
        check("切到 output: 自动提取", extractCalls.length === before + 1
            && extractCalls.at(-1).includes("out1.png%20%5Boutput%5D"));
        check("切到 output: 状态持久化", node.properties.promptReaderState.folder === "output");
        check("按钮显示 OUT", srcBtn.textContent === "OUT");

        // output 模式下 drop 上传 → 自动切回 input 并选中新文件
        const before2 = extractCalls.length;
        await root._handlers.drop(ev("video/mp4"));
        await new Promise((r) => setTimeout(r, 20));
        check("output 模式 drop: 切回 input", node.properties.promptReaderState.folder === "input");
        check("output 模式 drop: 选中上传文件", node._sfPrImageWidget.value === "uploaded.png");
        check("output 模式 drop: 已提取", extractCalls.length === before2 + 1);
        check("按钮显示 IN", srcBtn.textContent === "IN");
    }

    // ── 加载恢复：state.source=output 的节点 → 拉 output 列表 ──
    const node2 = makeNode();
    node2.properties = { promptReaderState: { folder: "output", filename: "smoke.png" } };
    app._ext.nodeCreated(node2);
    await new Promise((r) => setTimeout(r, 20));
    check("恢复 output 源: 拉取 output 列表", listCalls.at(-1) === "output");
    check("恢复 output 源: 值带 [output] 注解", node2._sfPrImageWidget.value === "out1.png [output]");

    // onRemoved 清理
    FakeType.prototype.onRemoved.call(node);
    check("onRemoved 清空 root", node._sfPrRoot === null);
    check("onRemoved 清空 widget 引用", node._sfPrImageWidget === null);

    console.log();
    if (failures.length) {
        console.log(failures.length + " FAILED: " + failures.join("; "));
        process.exit(1);
    }
    console.log("ALL PASS");
})().catch((e) => { console.error("CRASH:", e); process.exit(1); });
