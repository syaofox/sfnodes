// SFLoadImagesPath 渐进式目录浏览主扩展冒烟测试（Node 直接运行：node tests/test_load_images_path_smoke.js）
// 用 mock DOM/app/api 真实加载模块，验证：
//   - 模块加载 / 扩展注册
//   - nodeCreated：folder combo 隐藏（值仍是数据通道）、DOM widget 添加
//   - 源切换三档（input/output/images）写 folder 值 + 按需 fetch 当前层
//   - 下拉选择子目录 = 进入（面包屑前进 + 值更新 + fetch 下一层）
//   - 左右快速步进：当前层子目录循环（进入所选）
//   - 面包屑回退（祖先段点击）
//   - 模式切换：直接输入路径模式
//   - onConfigure 恢复：DOM 状态同步当前值
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素；innerHTML 赋值清空 children，模拟真实 DOM）──
function makeEl() {
    return {
        style: {}, dataset: {}, children: [], _handlers: {},
        className: "", textContent: "", value: "", placeholder: "",
        type: "", title: "", readOnly: false, disabled: false, isConnected: true,
        offsetWidth: 100, offsetHeight: 20,
        _innerHTML: "",
        get innerHTML() { return this._innerHTML; },
        set innerHTML(v) { this._innerHTML = v; this.children = []; },
        classList: {
            _s: new Set(),
            add(...c) { c.forEach((x) => this._s.add(x)); },
            remove(...c) { c.forEach((x) => this._s.delete(x)); },
            toggle(c, force) {
                if (force === undefined) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); }
                else { force ? this._s.add(c) : this._s.delete(c); }
            },
            contains(c) { return this._s.has(c); },
        },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        remove() { this.removed = true; },
        contains() { return false; },
        // 真实 DOM 语义：按 data-role 在子树中查找
        querySelector(sel) {
            const role = (sel.match(/\[data-role='([^']+)'\]/) || [])[1];
            const find = (el) => {
                if (el.dataset?.role === role) return el;
                for (const c of el.children || []) {
                    const hit = find(c);
                    if (hit) return hit;
                }
                return null;
            };
            return find(this);
        },
        querySelectorAll(sel) {
            const role = (sel.match(/\[data-role='([^']+)'\]/) || [])[1];
            const out = [];
            const walk = (el) => {
                if (el.dataset?.role === role) out.push(el);
                for (const c of el.children || []) walk(c);
            };
            walk(this);
            return out;
        },
        addEventListener(name, fn) { this._handlers[name] = fn; },
        removeEventListener() {},
        select() {}, click() {}, focus() {}, blur() {},
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {},
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
    getElementById() { return null; },
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
globalThis.requestAnimationFrame = (fn) => fn();
globalThis.queueMicrotask = (fn) => fn();
globalThis.navigator = {};

// ── 目录树 mock：按 folder 值返回下一级子目录 ──
const SUBDIR_TREE = {
    "input": ["faces", "empty"],
    "input/faces": ["sub1", "sub2"],
    "output": ["render"],
    "images": ["anime", "default"],
};
let subdirCalls = [];
globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/sfnodes/images_path/subdirs")) {
        const folder = new URL(u, "http://localhost").searchParams.get("folder");
        subdirCalls.push(folder);
        return { ok: true, json: async () => ({ subdirs: SUBDIR_TREE[folder] || [] }) };
    }
    throw new Error("unexpected fetch: " + u);
};
let _docHandlers = {};
let _bodyAppends = [];
globalThis._bodyAppends = _bodyAppends;
globalThis.app = {
    graph: { _nodes: [], links: {}, getNodeById() { return null; }, setDirtyCanvas() {} },
    registerExtension(ext) { this._ext = ext; },
    loadGraphData: async () => {},
};
globalThis.api = {
    apiURL: (route) => route,
    fetchApi: async () => ({ json: async () => ({}) }),
};
// 文档级监听（popup 关闭机制）与 body 挂载（拿 popup 元素）需要记录
globalThis.document.addEventListener = (name, fn) => { _docHandlers[name] = fn; };
globalThis.document.removeEventListener = (name) => { delete _docHandlers[name]; };
globalThis.document.body.appendChild = (c) => { _bodyAppends.push(c); return c; };

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_lip_"));
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

function makeNode() {
    const folderWidget = {
        name: "folder", value: "default", hidden: false,
        options: { values: ["default"] }, computeSize: null, element: null, inputEl: null,
    };
    return {
        comfyClass: "SFLoadImagesPath",
        widgets: [folderWidget],
        addDOMWidget(name, type, el, opts) {
            const w = { name, type, element: el, options: opts || {}, value: null, computeSize() {} };
            this.widgets.push(w);
            return w;
        },
        setDirtyCanvas() {},
        onConfigure: null,
        properties: {},
        graph: { setDirtyCanvas() {} },
    };
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
    stageJs(["sf_common.js", "load_images_path.js"]);
    await import(path.join(tmpDir, "load_images_path.mjs"));
    check("扩展已注册", globalThis.app._ext?.name === "sfnodes.load_images_path");

    const ext = globalThis.app._ext;
    const node = makeNode();
    ext.nodeCreated(node);

    check("folder combo 已隐藏", node.widgets[0].hidden === true);
    check("DOM widget 已添加", node.widgets.some((w) => w.name === "lip_ui"));

    const root = node.widgets.find((w) => w.name === "lip_ui").element;
    const folderWidget = node.widgets[0];

    // 初始默认值 "default" → fetch 根层
    await wait(20);
    check("默认值 fetch default", subdirCalls.includes("default"));

    // ── 源切换：点 input → 回根 + fetch input ──
    root.children[0].children[0]._handlers.click();
    check("源切换 input 写值", folderWidget.value === "input");
    await wait(20);
    check("进入 input 层 fetch", subdirCalls.includes("input"));

    // ── 下拉 popup（SFLoadImageResize 风格）：打开列当前层子目录，点击进入 ──
    _bodyAppends.length = 0;
    root.querySelector("[data-role='dir-trigger']")._handlers.click();
    await wait(20);
    const popup = _bodyAppends[_bodyAppends.length - 1];
    check("popup 已打开", !!popup && popup.className.includes("sf-lip-popup"));
    const listEl = popup.querySelector("[data-role='pop-list']");
    check("popup 列出子目录", listEl.children.length === 2);
    check("popup 头部显示当前路径", typeof popup.children[0].textContent === "string" && popup.children[0].textContent.includes("input"));
    listEl.children[0]._handlers.click();   // 📁 faces
    check("popup 点击进入", folderWidget.value === "input/faces");
    check("点击后 popup 已关闭", popup.removed === true);
    await wait(20);
    check("进入后 fetch 下一层", subdirCalls.includes("input/faces"));
    const crumbs = root.querySelector("[data-role='crumbs']");
    check("面包屑含 source 与层级", crumbs.children.length === 3 &&
        crumbs.children[0].textContent === "input" && crumbs.children[2].textContent === "faces");
    // 下拉按钮显示当前目录名（末段）
    const triggerName = root.querySelector("[data-role='dir-trigger']").children[0];
    check("下拉按钮显示当前目录名", triggerName.textContent === "faces");
    check("子目录计数显示", root.querySelector("[data-role='dir-count']").textContent.includes("2"));

    // ── Esc 关闭 popup ──
    _bodyAppends.length = 0;
    root.querySelector("[data-role='dir-trigger']")._handlers.click();
    await wait(20);
    const popupEsc = _bodyAppends[_bodyAppends.length - 1];
    check("popup 再次打开", !!popupEsc);
    if (typeof _docHandlers.keydown === "function") _docHandlers.keydown({ key: "Escape" });
    check("Esc 关闭 popup", popupEsc.removed === true);

    // ── 同值缓存：值未变重复渲染（onConfigure 恢复等）不重复请求 ──
    subdirCalls = subdirCalls.filter((f) => f !== "input/faces");
    if (node.onConfigure) node.onConfigure({});
    await wait(20);
    check("同值恢复不重复 fetch", !subdirCalls.includes("input/faces"));

    // ── 刷新按钮：强制重新加载当前层 ──
    const refreshBtn = root.children[5].children[0];
    refreshBtn._handlers.click();
    await wait(20);
    check("刷新强制重新 fetch 当前层", subdirCalls.includes("input/faces"));

    // ── 左右快速切换：同级目录循环步进（不改变层级深度）──
    // 当前 input/faces：父层 input 的子目录 = [faces, empty]；▶ → empty，◀ → faces
    await root.querySelector("[data-role='dir-next']")._handlers.click();
    check("▶ 同级切换", folderWidget.value === "input/empty");
    await wait(20);
    await root.querySelector("[data-role='dir-prev']")._handlers.click();
    check("◀ 同级切换回", folderWidget.value === "input/faces");
    await wait(20);

    // 进入 sub1 后 ▶：父层 faces 的子目录 = [sub1, sub2] → sub2（层级深度不变）
    _bodyAppends.length = 0;
    root.querySelector("[data-role='dir-trigger']")._handlers.click();
    await wait(20);
    const popup2 = _bodyAppends[_bodyAppends.length - 1];
    const list2 = popup2.querySelector("[data-role='pop-list']");
    list2.children[0]._handlers.click();   // 📁 sub1
    check("进入 sub1", folderWidget.value === "input/faces/sub1");
    await wait(20);
    await root.querySelector("[data-role='dir-next']")._handlers.click();
    check("▶ 深层同级切换", folderWidget.value === "input/faces/sub2");
    await wait(20);
    await root.querySelector("[data-role='dir-prev']")._handlers.click();
    check("◀ 深层同级切换回", folderWidget.value === "input/faces/sub1");
    await wait(20);

    // ── 根层：无同级 → 按钮禁用 ──
    const crumbsR = root.querySelector("[data-role='crumbs']");
    crumbsR.children[0]._handlers.click();   // source 段回根
    check("面包屑回根", folderWidget.value === "input");
    await wait(20);
    check("根层 ◀ 禁用", root.querySelector("[data-role='dir-prev']").disabled === true);
    check("根层 ▶ 禁用", root.querySelector("[data-role='dir-next']").disabled === true);

    // ── 模式切换：直接输入路径 ──
    root.children[1].children[1]._handlers.click();
    check("切到路径模式（值保持）", folderWidget.value === "input");
    const input = root.querySelector("[data-role='path-input']");
    check("路径输入框预填当前值", input.value === "input");
    input.value = "/data/images/custom";
    root.children[4].children[1]._handlers.click();
    check("路径输入应用写值", folderWidget.value === "/data/images/custom");

    // 切回目录模式：路径值不在目录列表 → 回默认源根
    root.children[1].children[0]._handlers.click();
    check("切回目录模式回源根", folderWidget.value === "images");
    await wait(20);

    // ── onConfigure 恢复：外部改值（如工作流加载）→ DOM 状态同步 + fetch 当前层 ──
    folderWidget.value = "output/render";
    if (node.onConfigure) node.onConfigure({});
    await wait(20);
    const cur = root.querySelector("[data-role='current']");
    check("onConfigure 同步当前值显示", typeof cur.textContent === "string" && cur.textContent.includes("output/render"));
    check("恢复后 fetch 当前层", subdirCalls.includes("output/render"));

    console.log();
    if (failures.length) {
        console.log(failures.length + " FAILURES:", failures);
        process.exit(1);
    }
    console.log("ALL PASS");
    process.exit(0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
