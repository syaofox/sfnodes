// SF 画布聚合菜单测试（Node 直接运行：node tests/test_canvas_menu_js.js）
// 覆盖（.mjs 拷贝链真实加载，test_lora_browser_smoke.js 同款手法）：
// - 全包唯一画布入口：有 getCanvasMenuItems 的扩展仅 sfnodes.CanvasMenu
// - 顶层唯一项 "📦 SF Menu"（has_submenu）；子菜单含浏览器/工作流/便签/内存
// - 对齐门槛：0 选中无 SF Align；2 选中出现 SF Align ▶ Width/Height/Size 三组
// - 驱动 Add SF Note 回调 → 落图（LiteGraph 兜底 + 视口中心落点）
// - 驱动 Align Width→Widest → 两节点同宽
// - 驱动 Free VRAM → POST /free{unload_models,free_memory} + 成功 toast
// - 驱动 Free RAM → POST /api/sfnodes/memory/ram + toast 含释放量
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// ── DOM stubs（sf_note 顶层补丁 + 各模块按需）──
function makeEl(tag) {
    return {
        tag, children: [], style: {}, dataset: {}, _handlers: {}, id: "",
        textContent: "", className: "",
        setAttribute() {}, removeAttribute() {},
        addEventListener(t, fn) { ((this._handlers[t] = this._handlers[t] || []).push(fn)); },
        removeEventListener() {},
        appendChild(c) { this.children.push(c); return c; },
        append(...cs) { for (const c of cs) this.children.push(c); },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        getBoundingClientRect() { return { left: 0, top: 0 }; },
        focus() {}, blur() {}, select() {},
    };
}
globalThis.document = {
    createElement: (t) => makeEl(t),
    createTextNode: (t) => ({ text: t }),
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    head: { appendChild() {} },
    body: { appendChild() {} },
    hidden: false,
    addEventListener() {},
    removeEventListener() {},
};
globalThis.window = {
    innerWidth: 1280, innerHeight: 800,
    addEventListener() {}, removeEventListener() {},
    open() {},
};
// LiteGraph 全套（sf_note 注册/补丁/兜底建节点用）
const registeredTypes = {};
globalThis.LiteGraph = {
    registerNodeType(type, cls) { registeredTypes[type] = cls; },
    createNode(type) {
        const C = registeredTypes[type];
        return C ? new C() : null;
    },
};
globalThis.LGraphNode = class LGraphNode {
    constructor(title) {
        this.title = title;
        this.properties = {};
        this.flags = {};
        this.size = [100, 60];
        this.pos = [0, 0];
    }
    serialize() { return { type: this.type }; }
    setDirtyCanvas() {}
};
globalThis.LGraphCanvas = {
    active_canvas: null,
    prototype: { drawNode() { return "orig"; }, processMouseDown() { return "orig-pm"; } },
};
globalThis.LGraph = { prototype: { getNodeOnPos() { return "orig-node"; } } };

// ── app/fetch stubs ──
const graphNodes = [];
const fakeGraph = {
    _nodes: graphNodes,
    add(n) { graphNodes.push(n); n.graph = this; },
    getNodeById(id) { return graphNodes.find((n) => n.id === id) || null; },
    setDirtyCanvas() {},
};
const toasts = [];
const fetchCalls = [];
const registered = [];
globalThis.app = {
    canvas: {
        canvas: { clientWidth: 800, clientHeight: 600 },
        ds: { offset: [0, 0], scale: 1 },
        selected_nodes: [],
    },
    graph: fakeGraph,
    ui: { settings: { getSettingValue: () => undefined } },
    extensionManager: { toast: { add: (t) => toasts.push(t) } }, // 无 command → 走 LiteGraph 兜底
    registerExtension: (e) => { registered.push(e); },
};
globalThis.api = { apiURL: (r) => r, fetchApi: async () => ({ ok: false }) };
globalThis.fetch = async (url, opts) => {
    fetchCalls.push({ url: String(url), opts: opts || {} });
    const u = String(url);
    if (u === "/free") return { ok: true, status: 200 };
    if (u.includes("/api/sfnodes/memory/ram")) {
        return {
            ok: true,
            json: async () => ({ ok: true, before_usage: 60.0, after_usage: 50.0, freed_mb: 1000 }),
        };
    }
    return { ok: false, status: 404, json: async () => ({}) };
};

// ── 复制依赖链为 .mjs 并真实加载 ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_cm_"));
const MODS = [
    "sf_canvas_menu.js", "sf_canvas_align.js", "sf_canvas_align_lib.js",
    "sf_memory_menu.js", "sf_note.js", "sf_note_lib.js",
    "sf_workflows.js", "sf_workflows_ui.js", "sf_workflows_lib.js",
    "sf_lora_browser.js", "sf_lora_browser_ui.js", "sf_lora_browser_lib.js",
    "sf_lora_stack_core.js", "sf_lora_stack_api.js",
    "sf_lora_stack_settings.js", "sf_lora_stack_info.js",
    "sf_common.js", "sf_markdown.js", "sf_lora_shared_info.js", "sf_lora_info.js",
];
for (const n of MODS) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    await import(path.join(tmpDir, "sf_canvas_menu.mjs"));

    // ── 唯一入口 ──
    const menuOwners = registered.filter((e) => typeof e.getCanvasMenuItems === "function");
    check("仅聚合器提供画布菜单", menuOwners.length === 1 && menuOwners[0].name === "sfnodes.CanvasMenu");
    check("分散扩展不再注册菜单",
        !registered.some((e) => e.name === "sfnodes.CanvasAlign" || e.name === "sfnodes.MemoryMenu"));
    const menuExt = menuOwners[0];

    const sub = () => menuExt.getCanvasMenuItems()[0].submenu.options;
    const byContent = (arr, c) => arr.find((o) => o && o.content === c);

    // ── 0 选中 ──
    globalThis.app.canvas.selected_nodes = [];
    let items = menuExt.getCanvasMenuItems();
    check("顶层唯一项 📦 SF Menu", items.length === 1
        && items[0].content === "📦 SF Menu" && items[0].has_submenu === true);
    let opts = sub();
    check("0 选中含浏览器/工作流/便签/内存",
        !!byContent(opts, "📚 SF LoRA Browser") && !!byContent(opts, "🎞 SF Workflows")
        && !!byContent(opts, "Add SF Note") && !!byContent(opts, "SF Memory"));
    check("0 选中无 SF Align", !byContent(opts, "SF Align"));

    // ── 2 选中：对齐出现 ──
    const n1 = { size: [100, 60], pos: [0, 0], setDirtyCanvas() {} };
    const n2 = { size: [150, 80], pos: [0, 0], setDirtyCanvas() {} };
    globalThis.app.canvas.selected_nodes = [n1, n2];
    opts = sub();
    const align = byContent(opts, "SF Align");
    check("2 选中出现 SF Align", !!align && align.has_submenu === true);
    const groups = align ? align.submenu.options.map((o) => o.content) : [];
    check("Align 含宽/高/等大三组",
        groups.includes("SF Align Width") && groups.includes("SF Align Height")
        && groups.includes("SF Align Size"));
    // 驱动 Width→Widest：两节点同宽（取最宽 150）
    const wGroup = byContent(align.submenu.options, "SF Align Width");
    const widest = byContent(wGroup.submenu.options, "Width \u2192 Widest");
    widest.callback();
    check("Widest 对齐同宽", n1.size[0] === 150 && n2.size[0] === 150);
    globalThis.app.canvas.selected_nodes = [];

    // ── 驱动 Add SF Note（LiteGraph 兜底）──
    const noteExt = registered.find((e) => e.name === "sfnodes.Note");
    noteExt.registerCustomNodes();
    const before = graphNodes.length;
    opts = sub();
    byContent(opts, "Add SF Note").callback();
    await tick(); await tick();
    const added = graphNodes[graphNodes.length - 1];
    check("菜单建便签落图", graphNodes.length === before + 1 && !!added);
    check("落点在视口中心附近",
        Math.abs(added.pos[0] - 400) < 100 && Math.abs(added.pos[1] - 300) < 100);

    // ── 驱动 Free VRAM ──
    fetchCalls.length = 0; toasts.length = 0;
    const mem = byContent(sub(), "SF Memory");
    check("SF Memory 子菜单两项", mem.submenu.options.length === 2);
    await byContent(mem.submenu.options, "Free VRAM").callback();
    const freeCall = fetchCalls.find((c) => c.url === "/free");
    check("VRAM 调原生 /free", !!freeCall && freeCall.opts.method === "POST");
    const body = JSON.parse(freeCall.opts.body || "{}");
    check("/free 参数全开", body.unload_models === true && body.free_memory === true);
    check("VRAM 成功 toast", toasts.some((t) => t.severity === "success"));

    // ── 驱动 Free RAM ──
    fetchCalls.length = 0; toasts.length = 0;
    await byContent(mem.submenu.options, "Free RAM").callback();
    check("RAM 调自建路由",
        fetchCalls.some((c) => c.url.includes("/api/sfnodes/memory/ram") && c.opts.method === "POST"));
    check("RAM toast 含释放量", toasts.some((t) => t.severity === "success" && /1000MB/.test(t.detail || "")));

    if (failures.length) {
        console.log(`\n${failures.length} FAILED: ${failures}`);
        process.exit(1);
    }
    console.log("\nALL PASSED");
})().catch((e) => { console.error("FATAL:", e); process.exit(1); });
