// SFLoadImagesPath 渐进式目录浏览主扩展冒烟测试（Node 直接运行：node tests/test_load_images_path_smoke.js）
// 用 mock DOM/app/api 真实加载模块，验证：
//   - 模块加载 / 扩展注册
//   - nodeCreated：folder combo 隐藏（值仍是数据通道）、DOM widget 添加
//   - 源切换两档（input/output）写 folder 值 + 按需 fetch 当前层
//   - 下拉选择子目录 = 进入（面包屑前进 + 值更新 + fetch 下一层）
//   - 左右快速步进：当前层子目录循环（进入所选）
//   - 面包屑回退（祖先段点击）
//   - 模式切换：直接输入路径模式
//   - popup 定位：视口钳位 / 方向翻转 / 限高（LoRA Stack 同款）
//   - onConfigure 恢复：DOM 状态同步当前值
//   - SFLoadImagesCursor 宿主：共用目录浏览（源切换/进入子目录/Path Mode）、Auto total 隐藏且早退
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

// ── 目录树 mock：按 folder 值返回下一级子目录 + 一级图片文件数 ──
const SUBDIR_TREE = {
    "input": { dirs: ["faces", "empty"], files: 5 },
    "input/faces": { dirs: ["sub1", "sub2"], files: 3 },
    "output": { dirs: ["render"], files: 0 },
    "output/render": { dirs: [], files: 7 },
};
let subdirCalls = [];
globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/sfnodes/images_path/subdirs")) {
        const folder = new URL(u, "http://localhost").searchParams.get("folder");
        subdirCalls.push(folder);
        const entry = SUBDIR_TREE[folder] || { dirs: [], files: 0 };
        return { ok: true, json: async () => ({ subdirs: entry.dirs, file_count: entry.files }) };
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

function makeNode(cls = "SFLoadImagesPath") {
    const folderWidget = {
        name: "folder", value: "default", hidden: false,
        options: { values: ["default"] }, computeSize: null, element: null, inputEl: null,
    };
    return {
        comfyClass: cls,
        widgets: [folderWidget],
        inputs: [],
        addDOMWidget(name, type, el, opts) {
            const w = { name, type, element: el, options: opts || {}, value: null, computeSize() {} };
            this.widgets.push(w);
            return w;
        },
        setDirtyCanvas() {},
        onConfigure: null,
        onRemoved: null,
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
    check("源切换两档（input/output）", root.children[0].children.length === 2);

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
    // 定位（mock：锚点 left=0/bottom=20，window 1280×720，popup offsetWidth/Height=100/20）：
    // 下方空间充足 → 向下展开 top=bottom+4；left 钳到 8；限高 min(60vh=432, downSpace=688)
    check("popup 左侧视口钳位", popup.style.left === "8px");
    check("popup 向下展开", popup.style.top === "24px");
    check("popup 限高取方向空间", popup.style.maxHeight === "432px");
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
    check("计数显示目录数+文件数", root.querySelector("[data-role='dir-count']").textContent === "2 目录 · 3 文件");

    // ── Esc 关闭 popup ──
    _bodyAppends.length = 0;
    root.querySelector("[data-role='dir-trigger']")._handlers.click();
    await wait(20);
    const popupEsc = _bodyAppends[_bodyAppends.length - 1];
    check("popup 再次打开", !!popupEsc);
    if (typeof _docHandlers.keydown === "function") _docHandlers.keydown({ key: "Escape" });
    check("Esc 关闭 popup", popupEsc.removed === true);

    // ── popup 边缘定位：下方空间不足 → 向上展开；右侧越界 → left 钳位 ──
    const triggerBtn = root.querySelector("[data-role='dir-trigger']");
    const origRect = triggerBtn.getBoundingClientRect;
    triggerBtn.getBoundingClientRect = () => ({ left: 1200, top: 700, right: 1300, bottom: 720, width: 100, height: 20 });
    _bodyAppends.length = 0;
    triggerBtn._handlers.click();
    await wait(20);
    const popupEdge = _bodyAppends[_bodyAppends.length - 1];
    check("popup 下方不足向上展开", popupEdge.style.top === "676px");   // max(8, 700-4-20)
    check("popup 右侧越界钳位", popupEdge.style.left === "1172px");     // 1280-100-8
    check("popup 限高取上方空间", popupEdge.style.maxHeight === "432px"); // min(432, upSpace=696)
    if (typeof _docHandlers.keydown === "function") _docHandlers.keydown({ key: "Escape" });
    check("边缘 popup 已关闭", popupEdge.removed === true);
    triggerBtn.getBoundingClientRect = origRect;

    // ── 同值缓存：值未变重复渲染（onConfigure 恢复等）不重复请求 ──
    subdirCalls = subdirCalls.filter((f) => f !== "input/faces");
    if (node.onConfigure) node.onConfigure({});
    await wait(20);
    check("同值恢复不重复 fetch", !subdirCalls.includes("input/faces"));

    // ── 刷新按钮：强制重新加载当前层 ──
    const refreshBtn = root.children[5].children[1];
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
    check("路径模式清空计数", root.querySelector("[data-role='dir-count']").textContent === "");
    const input = root.querySelector("[data-role='path-input']");
    check("路径输入框预填当前值", input.value === "input");
    input.value = "/data/images/custom";
    root.children[4].children[1]._handlers.click();
    check("路径输入应用写值", folderWidget.value === "/data/images/custom");

    // 切回目录模式：路径值不在目录列表 → 回默认源根（input）
    root.children[1].children[0]._handlers.click();
    check("切回目录模式回源根", folderWidget.value === "input");
    await wait(20);
    // 同值缓存命中也要重绘计数（否则停在路径模式的空计数）
    check("切回目录模式计数恢复", root.querySelector("[data-role='dir-count']").textContent === "2 目录 · 5 文件");

    // ── onConfigure 恢复：外部改值（如工作流加载）→ DOM 状态同步 + fetch 当前层 ──
    // 当前路径由面包屑承载（commit 2491fbb 移除了独立的 current 行）——
    // "output/render" → [src=output, ▸, render] 三段，末段为当前目录名。
    folderWidget.value = "output/render";
    if (node.onConfigure) node.onConfigure({});
    await wait(20);
    const crumbsRestored = root.querySelector("[data-role='crumbs']");
    check("onConfigure 同步当前值显示", crumbsRestored.children.length === 3
        && crumbsRestored.children[0].textContent === "output" && crumbsRestored.children[2].textContent === "render");
    check("恢复后 fetch 当前层", subdirCalls.includes("output/render"));

    // ── 自动 total：当前目录图片数反向写所驱动 SFForLoopStart.total（默认关）──
    const autoTotalBtn = root.children[5].children[0];
    check("auto total 按钮默认关", autoTotalBtn.textContent === "Auto total" && !autoTotalBtn.classList.contains("on"));
    check("auto total 默认不写 properties", node.properties.sfLoadImagesPathAutoTotal === undefined);

    // 构造循环节点 + skip_first_images 连线（link 5 → loop 的 index 输出 slot 1）
    const totalWidget = { name: "total", value: 9, options: { min: 1, max: 100000 }, callback: null };
    const loop = {
        id: "L1", comfyClass: "SFForLoopStart", type: "SFForLoopStart",
        outputs: [{ name: "flow" }, { name: "index" }, { name: "value1" }],
        inputs: [{ name: "total", link: null }],
        widgets: [totalWidget], properties: {},
        setDirtyCanvas() { this._dirty = true; },
    };
    node.inputs = [{ name: "skip_first_images", link: 5 }];
    node.graph = {
        links: { 5: { origin_id: "L1", origin_slot: 1 } },
        _nodes: [node, loop],
        getNodeById(id) { return String(id) === "L1" ? loop : null; },
        setDirtyCanvas() {},
    };
    folderWidget.value = "input";
    if (node.onConfigure) node.onConfigure({});
    await wait(20);
    check("auto total 默认关不写", totalWidget.value === 9);

    autoTotalBtn._handlers.click({ preventDefault() {}, stopPropagation() {} });
    check("auto total 开启同步图片数", totalWidget.value === 5 && autoTotalBtn.classList.contains("on") === true);
    check("auto total 持久化", node.properties.sfLoadImagesPathAutoTotal === true);

    // 目录切换 → fetch 后实时更新
    folderWidget.value = "input/faces";
    if (node.onConfigure) node.onConfigure({});
    await wait(20);
    check("auto total 目录变化更新", totalWidget.value === 3);

    // total 已转输入（有连线）→ 不写，连线优先
    loop.inputs = [{ name: "total", link: 77 }];
    totalWidget.value = 8;
    root._sfLipSyncLoopTotal();
    check("auto total 已接线的 total 不写", totalWidget.value === 8);
    loop.inputs = [{ name: "total", link: null }];

    // 非 index 输出（flow slot 0）→ 不写
    node.graph.links[5] = { origin_id: "L1", origin_slot: 0 };
    totalWidget.value = 7;
    root._sfLipSyncLoopTotal();
    check("auto total 非 index 输出不写", totalWidget.value === 7);
    node.graph.links[5] = { origin_id: "L1", origin_slot: 1 };

    // Path Mode 无目录计数 → 不写
    root.children[1].children[1]._handlers.click();
    totalWidget.value = 6;
    root._sfLipSyncLoopTotal();
    check("auto total 路径模式不写", totalWidget.value === 6);
    root.children[1].children[0]._handlers.click();
    await wait(20);

    // 多 LIP 驱动同一循环 → 取最大图片数
    const node2 = makeNode();
    node2.widgets[0].value = "output/render";   // 7 张
    ext.nodeCreated(node2);
    node2.graph = node.graph;
    node2.inputs = [{ name: "skip_first_images", link: 6 }];
    node.graph.links[6] = { origin_id: "L1", origin_slot: 1 };
    node.graph._nodes.push(node2);
    await wait(20);
    root._sfLipSyncLoopTotal();
    check("auto total 多节点取 max", totalWidget.value === 7);
    node.graph._nodes.pop();
    delete node.graph.links[6];

    // 未连线 → 不写（轮询兜底路径也不炸）
    node.inputs = [];
    totalWidget.value = 4;
    root._sfLipCheckWatch();
    check("auto total 未连线不写", totalWidget.value === 4);

    // 关闭开关 → 不再写且清除 properties
    autoTotalBtn._handlers.click({ preventDefault() {}, stopPropagation() {} });
    check("auto total 关闭清除持久化", node.properties.sfLoadImagesPathAutoTotal === undefined
        && autoTotalBtn.classList.contains("on") === false);

    // onConfigure 恢复 properties → 重新开启并同步（模拟工作流加载）
    node.properties.sfLoadImagesPathAutoTotal = true;
    node.inputs = [{ name: "skip_first_images", link: 5 }];
    if (node.onConfigure) node.onConfigure({});
    await wait(20);
    check("auto total 恢复后同步", totalWidget.value === 3 && autoTotalBtn.classList.contains("on") === true);

    // ── 游标节点（SFLoadImagesCursor）：共用目录浏览 UI，Auto total 隐藏且早退 ──
    const cursor = makeNode("SFLoadImagesCursor");
    subdirCalls = [];
    ext.nodeCreated(cursor);
    check("cursor: folder widget 已隐藏", cursor.widgets[0].hidden === true);
    check("cursor: DOM widget 已添加", cursor.widgets.some((w) => w.name === "lip_ui"));
    const cRoot = cursor.widgets.find((w) => w.name === "lip_ui").element;
    await wait(20);
    check("cursor: 初始 fetch 当前目录", subdirCalls.includes("default"));
    cRoot.children[0].children[1]._handlers.click();   // OUT · output
    check("cursor: 源切换写值", cursor.widgets[0].value === "output");
    await wait(20);
    _bodyAppends.length = 0;
    cRoot.querySelector("[data-role='dir-trigger']")._handlers.click();
    await wait(20);
    const cPopup = _bodyAppends[_bodyAppends.length - 1];
    const cList = cPopup.querySelector("[data-role='pop-list']");
    cList.children[0]._handlers.click();   // 📁 render
    check("cursor: popup 进入子目录", cursor.widgets[0].value === "output/render");
    await wait(20);
    check("cursor: 子目录计数显示", cRoot.querySelector("[data-role='dir-count']").textContent === "7 文件");
    const cAutoBtn = cRoot.children[5].children[0];
    check("cursor: Auto total 按钮隐藏", cAutoBtn.style.display === "none");
    cAutoBtn._handlers.click({ preventDefault() {}, stopPropagation() {} });
    check("cursor: Auto total 不写 properties", cursor.properties.sfLoadImagesPathAutoTotal === undefined);
    cRoot.children[1].children[1]._handlers.click();   // Path Mode
    const cInput = cRoot.querySelector("[data-role='path-input']");
    cInput.value = "/data/batch";
    cRoot.children[4].children[1]._handlers.click();
    check("cursor: 路径模式写值", cursor.widgets[0].value === "/data/batch");

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
