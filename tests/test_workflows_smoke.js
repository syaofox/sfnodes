// SF Workflows 主扩展冒烟测试（Node 直接运行：node tests/test_workflows_smoke.js）
// 用 mock DOM/app/store/fetch 真实加载模块，验证：
//   - 模块加载 / 扩展注册 / setup 挂工具栏按钮
//   - toggle 打开面板：buildBar+buildFooter+loadData（fetch 契约）→ render 全链路不抛错
//   - fetchIndex/fetchMeta 端点与参数正确
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素 + canvas 2d ctx）──
function makeCtx() {
    return new Proxy({}, {
        get(_t, k) {
            if (k === "canvas") return {};
            return () => {};
        },
        set() { return true; },
    });
}
function makeEl() {
    const style = { setProperty() {}, getPropertyValue() { return ""; } };
    return {
        style, dataset: {}, children: [],
        className: "", textContent: "", innerHTML: "", value: "", placeholder: "",
        type: "", title: "", rows: 1, spellcheck: false, disabled: false, checked: false,
        draggable: false, isConnected: true, offsetWidth: 100, offsetHeight: 20,
        selectionStart: 0, selectionEnd: 0, naturalWidth: 0, naturalHeight: 0,
        tabIndex: 0, loading: "", src: "", alt: "",
        classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        replaceWith() {}, replaceChildren(...kids) { this.children = kids; },
        before() {}, append() {},
        remove() { this.removed = true; },
        contains() { return false; }, closest() { return null; },
        querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        addEventListener() {}, removeEventListener() {},
        focus() {}, blur() {}, select() {}, click() {},
        dispatchEvent() {}, setSelectionRange() {},
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        getContext() { return makeCtx(); },
        setPointerCapture() {}, releasePointerCapture() {},
        scrollIntoView() {}, getComputedStyle() { return {}; },
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
    body: { appendChild() {}, append() {} },
    head: { appendChild(el) { if (el?.id === "sf-wb-css") globalThis.__sfwbStyleEl = el; } },
    addEventListener() {}, removeEventListener() {},
    getElementById() { return null; },
    createTextNode: (t) => ({ textContent: t }),
    execCommand: () => true,
    querySelector: (sel) => (sel === "#vue-app" ? vueAppEl : null),
    activeElement: makeEl(),
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
    app: null,
    open() { return {}; },
    devicePixelRatio: 1,
    requestAnimationFrame: (fn) => { fn(); return 0; },
    getComputedStyle: () => ({ gridTemplateColumns: "150px 150px" }),
    LiteGraph: {
        vueNodesMode: false,
        registered_node_types: { "KSampler": {}, "Note": {} },
    },
};
globalThis.requestAnimationFrame = globalThis.window.requestAnimationFrame;
globalThis.getComputedStyle = globalThis.window.getComputedStyle;
globalThis.URL = { createObjectURL: () => "blob:x", revokeObjectURL: () => {} };
globalThis.ClipboardItem = function () {};
globalThis.navigator = { clipboard: { writeText: async () => true } };
globalThis.performance = { now: () => 0 };

// ── mock app / workflow store / fetch ──
const fetchCalls = [];
globalThis.fetch = async (url, opts) => {
    fetchCalls.push({ url: String(url), opts });
    if (String(url).includes("/api/sfnodes/workflows/index")) {
        return { ok: true, json: async () => ({ ok: true, entries: [
            { rel: "a.json", name: "a", folder: "", size: 10, modified: 100, node_count: 2,
              class_types: ["KSampler"], models: ["m.safetensors"], loras: [], text: "x", map: [[0,0,1,1,"#123456"]], fingerprint: "fp1", error: null },
            { rel: "sub/b.json", name: "b", folder: "sub", size: 20, modified: 200, node_count: 1,
              class_types: ["Note"], models: [], loras: [], text: "", map: [], fingerprint: "", error: null },
        ], folders: ["sub"], collections: [], issues: { unsaved_names: [], duplicates: [], missing_nodes: [] } }) };
    }
    if (String(url).includes("/api/sfnodes/workflows/meta")) {
        return { ok: true, json: async () => ({ ok: true, meta: { notes: {}, covers: {}, folderColors: {} } }) };
    }
    if (String(url).includes("/api/userdata/")) {
        return { ok: true, text: async () => "{}" };
    }
    return { ok: false, status: 404, json: async () => ({}) };
};

const workflows = [
    { path: "workflows/a.json", isModified: false, isTemporary: false, activeState: { nodes: [] } },
    { path: "workflows/sub/b.json", isModified: false, isTemporary: false, activeState: { nodes: [] } },
];

// ── mock pinia 书签 store（模拟真实语义：ComfyUI 启动时不读收藏文件，
//    bookmarkedWorkflows 初始 null，直到 loadBookmarks() 才从磁盘读入）──
const diskBookmarks = [{ path: "workflows/a.json" }];
const bookmarkCalls = { load: 0, toggle: 0 };
const bookmarkStoreMock = {
    bookmarkedWorkflows: null,
    async loadBookmarks() { bookmarkCalls.load++; this.bookmarkedWorkflows = [...diskBookmarks]; },
    async toggleBookmarked(p) { bookmarkCalls.toggle++; this.bookmarkedWorkflows = [...(this.bookmarkedWorkflows || []), { path: p }]; },
};
const vueAppEl = {
    __vue_app__: {
        config: {
            globalProperties: {
                $pinia: { _s: { get: (id) => (id === "workflowBookmark" ? bookmarkStoreMock : null) } },
            },
        },
    },
};

globalThis.app = {
    graph: { _nodes: [], links: {} },
    canvas: { setDirty() {}, ds: { scale: 1 } },
    ui: {
        settings: {
            getSettingValue: () => null,
            setSettingValueAsync: async () => {},
            setSettingValue: () => {},
        },
    },
    extensionManager: {
        workflow: {
            activeWorkflow: workflows[0],
            openWorkflows: [],
            bookmarkedWorkflows: [],
            getWorkflowByPath: (p) => workflows.find((w) => w.path === p) || null,
            syncWorkflows: async () => {},
            renameWorkflow: async () => {},
        },
    },
    menu: { settingsGroup: { element: makeEl() } },
    registerExtension(ext) { this._ext = ext; },
    loadGraphData: async () => {},
    graphToPrompt: async () => ({ workflow: {}, output: {} }),
};
globalThis.window.app = globalThis.app;
globalThis.window.sfnodesGetSetting = (k, d) => null;
globalThis.window.sfnodesSetSetting = () => {};

// ── 加载模块 ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_wf_"));
for (const n of ["sf_common.js", "sf_workflows_lib.js", "sf_workflows_ui.js", "sf_workflows.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}
globalThis.api = {
    queuePrompt: async () => {},
    addEventListener() {},
    apiURL: (r) => r,
};

(async () => {
    await import(path.join(tmpDir, "sf_workflows.mjs"));
    check("扩展已注册", !!app._ext && app._ext.name === "sfnodes.WorkflowBrowser");
    check("命令已注册", Array.isArray(app._ext.commands) && app._ext.commands[0].id === "sfnodes.OpenWorkflowBrowser");
    check("快捷键已注册", Array.isArray(app._ext.keybindings) && app._ext.keybindings[0].combo.key === "w");

    // setup：挂工具栏按钮
    await app._ext.setup();
    check("setup 不抛错", true);

    // 触发面板打开（toggle 经命令回调）
    const cmd = app._ext.commands[0];
    await cmd.function();
    await new Promise((r) => setTimeout(r, 50));   // loadData 异步
    check("面板已打开", fetchCalls.some((c) => String(c.url).includes("/api/sfnodes/workflows/index")));
    check("meta 已获取", fetchCalls.some((c) => String(c.url).includes("/api/sfnodes/workflows/meta")));
    check("index 请求 no-store", fetchCalls.some((c) => String(c.url).includes("index") && c.opts?.cache === "no-store"));
    // 收藏加载必须在读列表前完成（ComfyUI 启动不读收藏文件；原版同款守卫
    // 在 loadData 的 Promise.all 里。修复前 loadBookmarks 永不被调用 → 重启后
    // 收藏恒为空，直到用户再收藏一次才触发加载）
    check("收藏已加载（loadData 触发 loadBookmarks）", bookmarkCalls.load >= 1);
    check("收藏读自磁盘", JSON.stringify((bookmarkStoreMock.bookmarkedWorkflows || [])) === JSON.stringify(diskBookmarks));

    // ── CSS 密度插值验证：注入的样式应含展开的 calc(...*var(--sfwb-k))，
    //    且无字面 ${z(...)} 残留（运行时插值失败的证据）──
    const cssText = (globalThis.__sfwbStyleEl?.textContent) || "";
    check("CSS 已注入", cssText.includes("--sfwb-k"));
    check("z() 已展开为 calc", cssText.includes("calc(") && cssText.includes("* var(--sfwb-k"));
    check("无字面插值残留", !cssText.includes("${z("));

    // 再开一次：toggle 关闭（幂等）
    await cmd.function();
    check("再次 toggle 关闭", true);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
