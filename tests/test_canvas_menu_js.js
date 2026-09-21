// SF 画布聚合菜单测试（Node 直接运行：node tests/test_canvas_menu_js.js）
// 覆盖（.mjs 拷贝链真实加载，test_lora_browser_smoke.js 同款手法）：
// - 全包唯一画布入口：有 getCanvasMenuItems 的扩展仅 sfnodes.CanvasMenu
// - 画布入口带前导 null 分隔线；节点入口无（Vue 单选会变菜单顶部横线）
// - 顶层唯一项 "📦 SF Menu"（has_submenu）；子菜单含浏览器/预设/工作流/内存/便签
// - 节点右键入口：getNodeMenuItems(node) 返回同一菜单（门槛与画布一致，任意节点可用）
// - 对齐门槛：<2 选中出 disabled 提示行（无 callback）；2 选中出 SF Align
//   （画布入口 6 动作，节点入口 +3 项 Mouse Node）；排序 Align 最前
// - 驱动 Align Width: Widest → 两节点同宽
// - 驱动 Add SF Note → 便签落图（右键 pointerdown 捕获位置优先，缺省中心兜底）
// - 驱动 SF LoRA Presets → 独立面板挂载
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
        tag, children: [], dataset: {}, _handlers: {}, id: "",
        style: { setProperty() {}, removeProperty() {}, getPropertyValue: () => "" },
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
const bodyAppends = [];
globalThis.document = {
    createElement: (t) => makeEl(t),
    createTextNode: (t) => ({ text: t }),
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    head: { appendChild() {} },
    body: { appendChild(el) { bodyAppends.push(el); } },
    hidden: false,
    addEventListener() {},
    removeEventListener() {},
};
const windowListeners = {};
globalThis.window = {
    innerWidth: 1280, innerHeight: 800,
    addEventListener(t, fn) { (windowListeners[t] = windowListeners[t] || []).push(fn); },
    removeEventListener() {},
    open() {},
};
function fireWindow(type, ev) {
    for (const fn of windowListeners[type] || []) fn(ev);
}
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
        canvas: {
            clientWidth: 800, clientHeight: 600,
            getBoundingClientRect: () => ({ left: 0, top: 0 }),
        },
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
    "sf_node_color.js", "sf_node_color_lib.js", "sf_popup.js",
    "sf_memory_menu.js", "sf_note.js", "sf_note_lib.js",
    "sf_workflows.js", "sf_workflows_ui.js", "sf_workflows_lib.js",
    "sf_lora_browser.js", "sf_lora_browser_ui.js", "sf_lora_browser_lib.js",
    "sf_lora_stack_core.js", "sf_lora_stack_api.js",
    "sf_lora_stack_settings.js", "sf_lora_stack_info.js",
    "sf_lora_preset_manager.js", "sf_lora_preset_filter.js",
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

    const canvasItems = () => menuExt.getCanvasMenuItems();
    const sub = () => canvasItems()[1].submenu.options;
    const byContent = (arr, c) => arr.find((o) => o && o.content === c);

    // ── 0 选中 ──
    globalThis.app.canvas.selected_nodes = [];
    let items = canvasItems();
    check("画布入口前导分隔线 + 📦 SF Menu", items.length === 2 && items[0] === null
        && items[1].content === "📦 SF Menu" && items[1].has_submenu === true);
    let opts = sub();
    check("0 选中含浏览器/预设/工作流/内存/便签",
        !!byContent(opts, "SF LoRA Browser") && !!byContent(opts, "SF LoRA Presets")
        && !!byContent(opts, "SF Workflows") && !!byContent(opts, "SF Memory")
        && !!byContent(opts, "Add SF Note"));
    const hint = byContent(opts, "SF Align (select ≥2 nodes)");
    check("0 选中 Align 提示行 disabled 无 callback",
        !!hint && hint.disabled === true && typeof hint.callback !== "function");
    check("0 选中无 SF Align 子菜单", !byContent(opts, "SF Align"));
    check("0 选中无 SF Node Color", !byContent(opts, "SF Node Color…"));
    const order0 = opts.map((o) => o && o.content);
    check("排序：Align 提示最前、工具项 Browser→Presets→Workflows→Memory→Note",
        order0[0] === "SF Align (select ≥2 nodes)"
        && order0.indexOf("SF LoRA Browser") < order0.indexOf("SF LoRA Presets")
        && order0.indexOf("SF LoRA Presets") < order0.indexOf("SF Workflows")
        && order0.indexOf("SF Workflows") < order0.indexOf("SF Memory")
        && order0.indexOf("SF Memory") < order0.indexOf("Add SF Note"));

    // ── 节点右键入口（getNodeMenuItems）──
    check("聚合器提供节点菜单入口", typeof menuExt.getNodeMenuItems === "function");
    const nodeItems = () => menuExt.getNodeMenuItems({ id: 1 });
    let nItems = nodeItems();
    check("节点右键同一顶层项 📦 SF Menu", nItems.length === 1
        && nItems[0].content === "📦 SF Menu" && nItems[0].has_submenu === true);
    let nOpts = nItems[0].submenu.options;
    check("节点菜单 0 选中含浏览器/预设/工作流/内存/便签",
        !!byContent(nOpts, "SF LoRA Browser") && !!byContent(nOpts, "SF LoRA Presets")
        && !!byContent(nOpts, "SF Workflows") && !!byContent(nOpts, "SF Memory")
        && !!byContent(nOpts, "Add SF Note"));
    check("节点菜单 0 选中 Align 提示行、无子菜单/Node Color",
        !!byContent(nOpts, "SF Align (select ≥2 nodes)")
        && !byContent(nOpts, "SF Align") && !byContent(nOpts, "SF Node Color…"));

    // ── 2 选中：对齐出现 ──
    const n1 = { size: [100, 60], pos: [0, 0], setDirtyCanvas() {} };
    const n2 = { size: [150, 80], pos: [0, 0], setDirtyCanvas() {} };
    globalThis.app.canvas.selected_nodes = [n1, n2];
    opts = sub();
    const align = byContent(opts, "SF Align");
    check("2 选中出现 SF Align", !!align && align.has_submenu === true);
    check("2 选中提示行消失且 Align 居首",
        !byContent(opts, "SF Align (select ≥2 nodes)") && opts[0] === align);
    const nodeColor = byContent(opts, "SF Node Color…");
    check("2 选中出现 SF Node Color", !!nodeColor && typeof nodeColor.callback === "function");
    nOpts = nodeItems()[0].submenu.options;
    check("节点菜单 2 选中出现 SF Align/SF Node Color",
        !!byContent(nOpts, "SF Align") && !!byContent(nOpts, "SF Node Color…"));
    const flat = align ? align.submenu.options.map((o) => o.content) : [];
    check("Align 画布入口 6 动作且无 Mouse Node",
        ["Width: Widest", "Width: Narrowest",
            "Height: Tallest", "Height: Shortest",
            "Size: Widest & Tallest", "Size: Narrowest & Shortest"]
            .every((c) => flat.includes(c))
        && flat.length === 9 && flat.every((c) => !String(c).includes("Mouse Node")));
    check("Align 无三级嵌套", align.submenu.options.every((o) => !o.submenu));
    check("分组头 Width/Height/Size 置灰不可点",
        ["Width", "Height", "Size"].every((c) => {
            const h = byContent(align.submenu.options, c);
            return h && h.disabled === true && typeof h.callback !== "function";
        }));
    // 节点入口：node 作为 Mouse Node 基准传入，三项出现
    const nAlign = byContent(menuExt.getNodeMenuItems(n2)[0].submenu.options, "SF Align");
    const nodeFlat = nAlign ? nAlign.submenu.options.map((o) => o.content) : [];
    check("Align 节点入口 9 动作（含 Mouse Node）",
        ["Width: Mouse Node", "Height: Mouse Node", "Size: Mouse Node"]
            .every((c) => nodeFlat.includes(c)) && nodeFlat.length === 12);
    // 驱动 Width: Widest：两节点同宽（取最宽 150）
    const widest = byContent(align.submenu.options, "Width: Widest");
    widest.callback();
    check("Widest 对齐同宽", n1.size[0] === 150 && n2.size[0] === 150);
    // 驱动节点入口 Width: Mouse Node：以右键节点 n2（150）为基准，n1 从 300 回缩
    n1.size[0] = 300;
    byContent(nAlign.submenu.options, "Width: Mouse Node").callback();
    check("Mouse Node 以基准节点为准", n1.size[0] === 150 && n2.size[0] === 150);
    globalThis.app.canvas.selected_nodes = [];

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

    // ── 驱动 Add SF Note（先注册自定义节点类型，test_note_js 先例）──
    const noteExt = registered.find((e) => e.name === "sfnodes.Note");
    noteExt?.registerCustomNodes?.();
    let beforeNotes = graphNodes.length;
    byContent(sub(), "Add SF Note").callback();
    await tick();
    check("Add SF Note 便签落图（无右键位置走视口中心兜底）",
        graphNodes.length === beforeNotes + 1
        && graphNodes[graphNodes.length - 1]?.type === "SF Note"
        && Array.isArray(graphNodes[graphNodes.length - 1]?.pos));

    // ── 模拟右键 pointerdown（capture 记录）→ 落点=鼠标位置 ──
    fireWindow("pointerdown", { button: 2, clientX: 500, clientY: 400 });
    beforeNotes = graphNodes.length;
    byContent(sub(), "Add SF Note").callback();
    await tick();
    const atNote = graphNodes[graphNodes.length - 1];
    check("Add SF Note 落点=右键画布坐标",
        graphNodes.length === beforeNotes + 1
        && atNote?.pos?.[0] === 500 && atNote?.pos?.[1] === 400);

    // ── 驱动 SF LoRA Presets（独立模式：无 node 也挂载面板）──
    bodyAppends.length = 0;
    byContent(sub(), "SF LoRA Presets").callback();
    await tick();
    check("SF LoRA Presets 面板挂载", bodyAppends.some((el) => el && el.id === "sf-lpm-overlay"));

    if (failures.length) {
        console.log(`\n${failures.length} FAILED: ${failures}`);
        process.exit(1);
    }
    console.log("\nALL PASSED");
})().catch((e) => { console.error("FATAL:", e); process.exit(1); });
