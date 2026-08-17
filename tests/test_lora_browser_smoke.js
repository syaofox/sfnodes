// SF LoRA 浏览器主扩展冒烟测试（Node 直接运行：node tests/test_lora_browser_smoke.js）
// mock DOM/app/fetch 真实加载 sf_lora_browser.js 全依赖链，验证：
//   - 扩展注册（name/keybinding/command）
//   - 工具栏按钮挂载 + 点击打开窗口
//   - 文件夹模式（默认）：根层显示立即子文件夹 + 当前层文件；下钻/面包屑返回
//   - 平面模式：seg 切换、面包屑隐藏、分批渲染（FLAT_STEP=60）+ 滚动动态加载
//   - 搜索两种模式都扁平匹配；模式记忆写设置
//   - 无缩略图占位；点击文件卡片 -> LoRA Stack 同款信息面板（浏览器 ctx 路径）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}
const tick = () => new Promise((r) => setTimeout(r, 0));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── mock DOM ──
function makeEl(tag) {
    const el = {
        tagName: String(tag || "div").toUpperCase(),
        style: { setProperty() {}, getPropertyValue() { return ""; } },
        dataset: {}, children: [], listeners: {}, _text: "",
        _cls: new Set(), _attrs: {},
        value: "", placeholder: "", title: "", type: "", loading: "",
        disabled: false, isConnected: true, src: "", alt: "",
        offsetWidth: 100, offsetHeight: 20,
        scrollTop: 0, clientHeight: 600, scrollHeight: 800,
        classList: {
            add(...c) { c.forEach((x) => el._cls.add(x)); sync(); },
            remove(...c) { c.forEach((x) => el._cls.delete(x)); sync(); },
            toggle(c, force) {
                if (force === undefined) { el._cls.has(c) ? el._cls.delete(c) : el._cls.add(c); }
                else { force ? el._cls.add(c) : el._cls.delete(c); }
                sync();
            },
            contains(c) { return el._cls.has(c); },
        },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        before(c) { if (c) el._before = c; },
        after(c) { if (c) el._after = c; },
        remove() { this.removed = true; },
        contains(t) {
            if (this === t) return true;
            return (this.children || []).some((c) => c === t || (c.contains && c.contains(t)));
        },
        focus() {}, blur() {}, select() {},
        setPointerCapture() {}, releasePointerCapture() {},
        closest() { return null; },
        setAttribute(k, v) { this._attrs[k] = v; },
        removeAttribute(k) { delete this._attrs[k]; },
        dispatchEvent(ev) { this.emit(ev.type, ev); },
        querySelector(sel) {
            const parts = sel.trim().split(/\s+/).filter(Boolean);
            const match = (el2, i) => {
                if (i >= parts.length) return el2;
                const p = parts[i];
                const wantCls = p.startsWith(".") ? p.slice(1) : null;
                const wantTag = p.toUpperCase();
                for (const c of el2.children || []) {
                    const hasCls = wantCls == null
                        || String(c.className || "").split(/\s+/).includes(wantCls);
                    const hasTag = wantCls != null || c.tagName === wantTag;
                    if (hasCls && hasTag) {
                        const hit = match(c, i + 1);
                        if (hit) return hit;
                    }
                }
                return null;
            };
            return match(this, 0);
        },
        querySelectorAll(sel) {
            const out = [];
            const matchOne = (el2) => {
                if (sel.startsWith("[") && sel.endsWith("]")) {
                    const key = sel.slice(1, -1).replace(/^data-/, "");
                    return el2.dataset && key in el2.dataset;
                }
                const wantCls = sel.startsWith(".") ? sel.slice(1) : null;
                if (wantCls) return String(el2.className || "").split(/\s+/).includes(wantCls);
                return el2.tagName === sel.toUpperCase();
            };
            const walk = (el2) => {
                for (const c of el2.children || []) {
                    if (matchOne(c)) out.push(c);
                    walk(c);
                }
            };
            walk(this);
            return out;
        },
        addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
        removeEventListener(type, fn) {
            const a = this.listeners[type];
            if (a) { const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); }
        },
        emit(type, evt) {
            const e = evt || { target: this };
            e.stopPropagation ||= () => {};
            e.preventDefault ||= () => {};
            for (const fn of [...(this.listeners[type] || [])]) fn(e);
        },
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {},
    };
    function sync() { el.className = [...el._cls].join(" "); }
    Object.defineProperty(el, "className", {
        get() { return [...el._cls].join(" "); },
        set(v) { el._cls = new Set(String(v).split(/\s+/).filter(Boolean)); },
    });
    Object.defineProperty(el, "textContent", {
        get() { return el._text; },
        set(v) { el._text = v; el.children = []; },
    });
    Object.defineProperty(el, "innerHTML", {
        get() { return ""; },
        set() { el.children = []; },
    });
    return el;
}

const bodyChildren = [];
const settingsGroupEl = makeEl("div");
// 设置存储（模式/位置/窗口几何记忆）
const settingStore = {};
globalThis.document = {
    createElement(tag) { return makeEl(tag); },
    createTextNode(t) { return { textContent: t }; },
    body: { appendChild(c) { bodyChildren.push(c); return c; }, contains() { return false; } },
    head: { appendChild() {} },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    _listeners: {},
    addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); },
    removeEventListener(type, fn) {
        const a = this._listeners[type];
        if (a) { const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); }
    },
    emit(type, evt) { for (const fn of [...(this._listeners[type] || [])]) fn(evt); },
    getElementById() { return null; },
    activeElement: makeEl(),
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
};
globalThis.navigator = { clipboard: { writeText: async () => {} } };
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.queueMicrotask = (fn) => Promise.resolve().then(fn);
globalThis.Event = class { constructor(type, opts) { this.type = type; this.bubbles = opts?.bubbles; } };
globalThis.CustomEvent = globalThis.Event;

const registered = [];
const cmdCalls = [];
function fakeLiteNode(type) {
    return {
        id: 900 + Math.floor(Math.random() * 1000),
        type, comfyClass: type,
        pos: [0, 0], size: [336, 60],
        properties: {}, widgets: [], inputs: [], outputs: [], flags: {},
        setDirtyCanvas() {}, setSize(s) { this.size = s; }, computeSize() { return [336, 100]; },
        addDOMWidget() { return { element: makeEl("div"), options: {}, computeLayoutSize() { return { minHeight: 60, minWidth: 1 }; } }; },
    };
}
globalThis.app = {
    graph: {
        _nodes: [],
        setDirtyCanvas() {},
        add(node) { this._nodes.push(node); },   // 极简 mock：不触发扩展 nodeCreated
        remove(node) { const i = this._nodes.indexOf(node); if (i >= 0) this._nodes.splice(i, 1); },
    },
    canvas: { ds: { scale: 1, offset: [0, 0] }, selected_nodes: {}, current_node: null, node_over: null, selectNode() {} },
    api: { fetchApi: async () => ({ ok: false, json: async () => ({}) }) },
    menu: { settingsGroup: { element: settingsGroupEl } },
    ui: {
        settings: {
            getSettingValue(key) { return settingStore[key]; },
            setSettingValueAsync(key, val) { settingStore[key] = val; },
            setSettingValue(key, val) { settingStore[key] = val; },
        },
    },
    registerExtension(ext) { registered.push(ext); },
    extensionManager: {
        command: {
            execute: async (id, args) => {
                cmdCalls.push({ id, args });
                if (id === "Comfy.AddNode") {
                    const n = fakeLiteNode(args?.type || "SFLoraStack");
                    n.pos = [0, 0];
                    app.graph.add(n);
                    return n;
                }
                return null;
            },
        },
    },
    _sfLoraBrowserRegistered: false,
};
globalThis.window.LiteGraph = {
    createNode(type) { return fakeLiteNode(type); },
    ds: null, vueNodesMode: false,
};
globalThis.api = { apiURL: (r) => r };

// ── fetch mock：66 项（4 个层级 + 62 个 batch 平铺，超过 FLAT_STEP=60 验证滚动加载）──
const LIST_MOCK = [
    "a.safetensors",
    "characters/xiangling.safetensors",
    "style/watercolor.safetensors",
    "style/lineart/ink.safetensors",
    ...Array.from({ length: 62 }, (_, i) => "batch/b" + String(i).padStart(2, "0") + ".safetensors"),
];
globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/sfnodes/lora_list")) {
        return { ok: true, json: async () => ({ loras: LIST_MOCK }) };
    }
    if (u.includes("/api/sfnodes/lora_info?name=")) {
        return { ok: true, json: async () => ({
            ok: true,
            info: {
                title: decodeURIComponent(u.split("name=")[1] || "x"),
                source: "file",
                description: "file desc",
                triggers: ["aa", "bb"],
                file_triggers: ["aa", "bb"],
                sidecar_triggers: [],
                custom_triggers: [],
                custom_description: "",
                has_preview: false,
                custom_preview: false,
                preview_v: 0,
                orphan_key: "",
                civitai_host: "com",
            },
        }) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
};

// ── 复制全部依赖链为 .mjs ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_lb_"));
const MODS = [
    "sf_lora_browser.js", "sf_lora_browser_ui.js", "sf_lora_browser_lib.js",
    "sf_lora_stack_core.js", "sf_lora_stack_api.js",
    "sf_lora_stack_settings.js", "sf_lora_stack_info.js",
    "sf_common.js", "sf_markdown.js", "sf_lora_info.js",
    "sf_workflows_ui.js", "sf_workflows_lib.js",
];
for (const n of MODS) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

function hasClass(el, cls) {
    return String(el?.className || "").split(/\s+/).includes(cls);
}
function findByClass(root, cls) {
    if (!root) return null;
    for (const c of root.children || []) {
        if (hasClass(c, cls)) return c;
        const f = findByClass(c, cls);
        if (f) return f;
    }
    return null;
}
function findByText(root, text) {
    if (!root) return null;
    if (root._text === text) return root;
    for (const c of root.children || []) {
        const f = findByText(c, text);
        if (f) return f;
    }
    return null;
}
function countCards(root) {
    let folders = 0, files = 0;
    (function walk(r) {
        if (!r) return;
        if (hasClass(r, "sf-lb-card")) {
            if (hasClass(r, "folder")) folders++;
            else files++;
        }
        for (const c of r.children || []) walk(c);
    })(root);
    return { folders, files };
}
function currentCrumbText(win) {
    const path = findByClass(win, "sf-lb-path");
    const crumb = (path?.querySelectorAll(".sf-lb-crumb") || []).find((c) => hasClass(c, "cur"));
    return crumb?._text ?? null;
}
// seg 按钮在 bar 内（窗口 api 的 segButtons 在闭包里，不在 DOM 元素上）
function segButton(win, mode) {
    return (win.querySelectorAll(".sf-lb-segb") || []).find((b) => b.dataset?.mode === mode);
}
function pathEl(win) {
    return findByClass(win, "sf-lb-path");
}

(async () => {
    const B = await import(path.join(tmpDir, "sf_lora_browser.mjs"));
    const I = await import(path.join(tmpDir, "sf_lora_stack_info.mjs"));
    check("主扩展模块加载", typeof B === "object");

    // ── 注册 ──
    const ext = registered.find((e) => e.name === "sfnodes.LoraBrowser");
    check("扩展已注册（sfnodes 前缀）", !!ext);
    check("命令 id", ext?.commands?.[0]?.id === "sfnodes.OpenLoraBrowser");
    check("热键 Alt+Shift+L", ext?.keybindings?.[0]?.combo?.key === "l"
        && ext?.keybindings?.[0]?.combo?.alt === true
        && ext?.keybindings?.[0]?.combo?.shift === true);
    check("canvas 菜单项", ext?.getCanvasMenuItems?.().length === 1);
    check("openInfoPanelFor 已导出", typeof I.openInfoPanelFor === "function");
    check("openInfoPanel 兼容入口仍在", typeof I.openInfoPanel === "function");

    // ── 工具栏按钮挂载 ──
    await ext.setup();
    check("按钮已插入 settingsGroup 前", !!settingsGroupEl._before);
    check("按钮 class", hasClass(settingsGroupEl._before?.children?.[0], "sf-lb-btn"));

    // ── 打开窗口（默认文件夹模式）──
    const btn = settingsGroupEl._before.children.find((c) => hasClass(c, "sf-lb-btn"));
    btn.emit("click");
    await tick(); await tick(); await tick();
    const win = bodyChildren.find((c) => hasClass(c, "sf-lb-win"));
    check("窗口已打开", !!win && win.style.display === "flex");
    check("计数 66", findByClass(win, "sf-lb-count")?._text === "66");
    check("根层面包屑", currentCrumbText(win) === "All LoRAs");
    let cards = countCards(win);
    check("根层 3 文件夹（batch/characters/style）", cards.folders === 3);
    check("根层 1 文件", cards.files === 1);
    check("seg 切换控件存在", !!segButton(win, "folder") && !!segButton(win, "flat"));
    check("默认文件夹模式高亮", hasClass(segButton(win, "folder"), "on") && !hasClass(segButton(win, "flat"), "on"));
    check("面包屑可见", pathEl(win).style.display !== "none");

    // ── 文件夹下钻 characters ──
    const folderCards = (win.querySelectorAll(".sf-lb-card") || []).filter((c) => hasClass(c, "folder"));
    const chars = folderCards.find((c) => c.dataset.folderName === "characters");
    chars.emit("click");
    await tick();
    check("下钻后面包屑", currentCrumbText(win) === "characters");
    cards = countCards(win);
    check("characters 层 0 文件夹", cards.folders === 0);
    check("characters 层 1 文件", cards.files === 1);
    check("characters 层文件名", (win.querySelectorAll(".sf-lb-card") || []).find((c) => !hasClass(c, "folder"))?.dataset?.name === "characters/xiangling.safetensors");

    // ── 面包屑返回根 ──
    const rootCrumb = findByClass(win, "sf-lb-path").querySelectorAll("[data-folder]")
        .find((c) => c.dataset.folder === "");
    rootCrumb.emit("click");
    await tick();
    check("返回根层面包屑", currentCrumbText(win) === "All LoRAs");
    cards = countCards(win);
    check("回根后 3 文件夹", cards.folders === 3);
    check("回根后 1 文件", cards.files === 1);

    // ── 搜索（文件夹模式扁平匹配）──
    const searchWrap = findByClass(win, "sf-lb-search");
    const searchInput = searchWrap?.children?.find((c) => c.tagName === "INPUT");
    check("搜索框存在", !!searchInput);
    searchInput.value = "xiang";
    searchInput.emit("input");
    await sleep(250);
    cards = countCards(win);
    check("搜索扁平 1 文件", cards.files === 1 && cards.folders === 0);
    check("搜索计数 1 / 66", findByClass(win, "sf-lb-count")?._text === "1 / 66");
    searchInput.value = "";
    searchInput.emit("input");
    await sleep(250);
    cards = countCards(win);
    check("清空搜索回根层", cards.folders === 3 && cards.files === 1);

    // ── 切换平面模式 ──
    segButton(win, "flat").emit("click");
    await tick();
    check("flat 高亮", hasClass(segButton(win, "flat"), "on"));
    check("平面模式面包屑隐藏", pathEl(win).style.display === "none");
    check("平面首屏 60 项（分批）", countCards(win).files === 60);
    check("平面 loadmore 哨兵", !!findByClass(win, "sf-lb-loadmore"));
    check("平面计数 60 / 66", findByClass(win, "sf-lb-count")?._text === "60 / 66");
    check("模式已记忆", settingStore["sfnodes.LoraBrowser.Mode"] === "flat");

    // ── 平面滚动加载 → 66 ──
    const mainEl = findByClass(win, "sf-lb-main");
    mainEl.scrollTop = 5000;
    mainEl.scrollHeight = 5000;
    mainEl.emit("scroll");
    await tick(); await tick();
    check("滚动后 66 项全载", countCards(win).files === 66);
    check("loadmore 消失", findByClass(win, "sf-lb-loadmore") === null);
    check("平面计数 66", findByClass(win, "sf-lb-count")?._text === "66");

    // ── 平面搜索（扁平）──
    searchInput.value = "watercolor";
    searchInput.emit("input");
    await sleep(250);
    check("平面搜索 1 项", countCards(win).files === 1);
    searchInput.value = "";
    searchInput.emit("input");
    await sleep(250);

    // ── 切回文件夹模式 ──
    segButton(win, "folder").emit("click");
    await tick();
    check("切回后面包屑显示", pathEl(win).style.display !== "none");
    cards = countCards(win);
    check("切回后 3 文件夹 1 文件", cards.folders === 3 && cards.files === 1);
    check("模式已记忆 folder", settingStore["sfnodes.LoraBrowser.Mode"] === "folder");

    // ── 无缩略图占位 ──
    const rootFile = (win.querySelectorAll(".sf-lb-card") || []).find((c) => !hasClass(c, "folder"));
    const th = rootFile?.querySelector(".sf-lb-thumb");
    check("文件卡片缩略图存在", !!th);
    if (th) {
        th.src = "/api/sfnodes/lora_thumb?name=a.safetensors";
        th.emit("error");
        check("error 后换占位图", String(th.src).startsWith("data:image/svg+xml"));
        check("error 后 noimg class", hasClass(th, "noimg"));
        th.emit("error");
        check("二次 error 占位不变", String(th.src).startsWith("data:image/svg+xml"));
    }

    // ── 双击文件卡片 -> 用 SF LoRA Stack 加载到当前工作流 ──
    const fileCard = (win.querySelectorAll(".sf-lb-card") || []).find((c) => !hasClass(c, "folder"));
    const nodesBefore = app.graph._nodes.length;
    fileCard.emit("dblclick");
    await sleep(300);   // 给双击判定留足时间（单击已被取消）
    check("双击创建节点", app.graph._nodes.length === nodesBefore + 1);
    check("走官方 AddNode 命令", cmdCalls.some((c) => c.id === "Comfy.AddNode" && c.args?.type === "SFLoraStack"));
    const addedNode = app.graph._nodes[nodesBefore];
    check("节点类型 SFLoraStack", addedNode?.comfyClass === "SFLoraStack");
    check("节点位置在视口中心附近", addedNode?.pos?.[0] > 0 && addedNode?.pos?.[1] > 0);
    const st = JSON.parse(addedNode?.properties?.loraStackState || "{}");
    check("行已预置 LoRA", st.loras?.[0]?.name === "a.safetensors");
    check("双击不打开信息面板", ![...bodyChildren].some((c) => hasClass(c, "sf-ls-info-p")));

    // ── 单击文件卡片 -> 250ms 延迟后打开信息面板（等双击判定）──
    fileCard.emit("click");
    await tick();
    check("单击 250ms 内不开面板（等待双击判定）", ![...bodyChildren].some((c) => hasClass(c, "sf-ls-info-p")));
    await sleep(300);
    let panel = [...bodyChildren].reverse().find((c) => hasClass(c, "sf-ls-info-p"));
    check("信息面板已打开", !!panel);
    check("面板标题 = LoRA 名", findByClass(panel, "sf-ls-info-h")?.children?.[0]?._text === "a.safetensors");
    check("面板内 chips 存在（aa/bb）", (() => { let n = 0; (function walk(r){ if(!r) return; if(hasClass(r,"sf-ls-chip")) n++; for(const c of r.children||[]) walk(c); })(panel); return n >= 2; })());

    // ── 平面模式双击也可添加（batch/b00）──
    segButton(win, "flat").emit("click");
    await tick();
    const flatFile = (win.querySelectorAll(".sf-lb-card") || []).find((c) => !hasClass(c, "folder"));
    const flatName = flatFile?.dataset?.name;
    const n2 = app.graph._nodes.length;
    flatFile.emit("dblclick");
    await sleep(300);
    check("平面模式双击也创建节点", app.graph._nodes.length === n2 + 1);
    const st2 = JSON.parse(app.graph._nodes[n2]?.properties?.loraStackState || "{}");
    check("平面双击行名匹配卡片", !!flatName && st2.loras?.[0]?.name === flatName);
    segButton(win, "folder").emit("click");
    await tick();

    // ── 关闭窗口 ──
    const closeBtn = findByText(win, "✕");
    closeBtn.emit("click");
    await tick();
    check("窗口已关闭", win.style.display === "none");

    if (failures.length) {
        console.log("FAILURES:", failures.join(", "));
        process.exit(1);
    }
    console.log("ALL PASS");
})();
