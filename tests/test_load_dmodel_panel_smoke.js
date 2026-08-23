// SF Load Diffusion Model 信息面板集成冒烟测试（Node 直接运行：
// node tests/test_load_dmodel_panel_smoke.js）
// 真实加载 sf_lora_stack_info.js + sf_dmodel_api.js，以节点模块同款 ctx
// 打开面板，验证 dmodel 域四件套与路由束契约：
//   - api 束键名与面板 A.* 逐一对应（键错名会静默回退 LoRA 路由——本测试
//     存在的首要理由）
//   - hideTriggers: 面板不渲染触发词区块
//   - info 走 /dmodel_info；描述三档读 custom_description
//   - autoCivitai: 打开即查 /dmodel/civitai（侧车无 id 时）
//   - samplesKind: samples URL 带 kind=diffusion_models
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// ── mock DOM（test_lora_stack_info_desc_smoke 同款惰性元素）─────────────────
function makeEl(tag) {
    const el = {
        tagName: String(tag || "div").toUpperCase(),
        style: { setProperty() {}, getPropertyValue() { return ""; } },
        dataset: {}, children: [], listeners: {}, _text: "",
        _cls: new Set(),
        value: "", placeholder: "", title: "", type: "",
        disabled: false, isConnected: true,
        offsetWidth: 100, offsetHeight: 20,
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
        remove() { this.removed = true; },
        contains(t) {
            if (this === t) return true;
            return (this.children || []).some((c) => c === t || (c.contains && c.contains(t)));
        },
        focus() {}, blur() {}, select() {},
        setPointerCapture() {}, releasePointerCapture() {},
        querySelector(sel) {
            const parts = sel.trim().split(/\s+/).filter(Boolean);
            const match = (el, i) => {
                if (i >= parts.length) return el;
                const p = parts[i];
                const wantCls = p.startsWith(".") ? p.slice(1) : null;
                const wantTag = p.toUpperCase();
                for (const c of el.children || []) {
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
            // 仅 h3 a 悬停挂接需要；返回空数组即可（无标题链接场景）
            return [];
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
    function sync() {
        el.className = [...el._cls].join(" ");
    }
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
globalThis.document = {
    createElement(tag) { return makeEl(tag); },
    createTextNode(t) { return { textContent: t }; },
    body: { appendChild(c) { bodyChildren.push(c); return c; }, contains() { return false; } },
    head: { appendChild() {} },
    querySelector() { return null; },
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
globalThis.LiteGraph = { WIDGET_TEXT_COLOR: "#fff" };

// ── fetch 记录器：断言路由域（必须在模块 import 前就位——模块顶层
//    const app = globalThis.app 绑定一次）──
const fetched = [];
let civitaiCalls = 0;
let samplesUrls = [];
globalThis.fetch = async (url) => {
    const u = String(url);
    fetched.push(u.split("?")[0] + (u.includes("kind=") ? "?kind" : ""));
    if (u.includes("/api/sfnodes/dmodel_info?name=")) {
        return { ok: true, json: async () => ({
            ok: true,
            info: {
                title: "Test DiT",
                base_model: "qwen-image",
                size: "6.6 GB",
                triggers: [], file_triggers: [], sidecar_triggers: [],
                custom_triggers: [],
                custom_description: "my custom note",
                description: "file desc",
                file_description: "file desc",
                civitai_host: "com",
                source: "file",
                preview_v: 0,
            },
        }) };
    }
    if (u.includes("/api/sfnodes/dmodel/civitai")) {
        civitaiCalls++;
        return { ok: true, json: async () => ({ ok: true, found: false, reason: "notfound" }) };
    }
    if (u.includes("/api/sfnodes/lora_samples?filename=")) {
        return { ok: true, json: async () => ({ images: ["sample/a.png"], sample_dir: "sample" }) };
    }
    return { ok: true, json: async () => ({ ok: true }) };
};
globalThis.app = {
    graph: { _nodes: [], setDirtyCanvas() {} },
    canvas: {
        ds: { scale: 1, offset: [0, 0] },
        canvas: { getBoundingClientRect: () => ({ left: 0, top: 0, right: 800, bottom: 600 }) },
    },
    api: {
        fetchApi: async (url, opts) => {
            const u = String(url);
            fetched.push(u.split("?")[0] + (u.includes("kind=") ? "?kind" : ""));
            if (u.includes("lora_samples")) samplesUrls.push(u);
            return { ok: true, json: async () => ({ ok: true }), status: 200 };
        },
    },
    ui: { settings: { getSettingValue: () => false } },
};

// ── 加载模块（/scripts/app.js -> globalThis；相对 import 改 .mjs）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_dmp_"));
for (const n of ["sf_lora_stack_core.js", "sf_lora_stack_api.js",
    "sf_lora_stack_settings.js", "sf_common.js", "sf_markdown.js",
    "sf_lora_shared_info.js", "sf_lora_info.js", "sf_lora_stack_info.js", "sf_workflows_ui.js",
    "sf_workflows_lib.js", "sf_lora_stack_dropdown.js",
    "sf_lora_stack_render.js", "sf_lora_stack_interaction.js", "sf_dmodel_api.js"]) {
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
function findByText(root, text) {
    if (!root) return null;
    if (root._text === text) return root;
    for (const c of root.children || []) {
        const f = findByText(c, text);
        if (f) return f;
    }
    return null;
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
const lastPanel = () => [...bodyChildren].reverse().find((c) => hasClass(c, "sf-ls-info-p"));

(async () => {
    const API = await import(path.join(tmpDir, "sf_dmodel_api.mjs"));
    const I = await import(path.join(tmpDir, "sf_lora_stack_info.mjs"));

    // ── 契约 1：api 束键名与面板 A.* 一致（错名 = 静默回退 LoRA 域）──
    const NEED = ["info", "thumbUrl", "civitai", "invalidate", "delCivitai",
        "saveDescription", "savePreview", "deletePreview", "saveCivitaiThumb",
        "migrate", "merge"];
    check("api 束键齐全", NEED.every((k) => typeof API.dmodelApi[k] === "function"));

    // ── 节点模块同款 ctx（hideTriggers/samplesKind/autoCivitai/api）──
    const NAME = "sub/test_dit.safetensors";
    const row = { id: NAME, name: NAME, triggers: [], custom: [] };
    const fakeNode = { id: 42, widgets: [], setDirtyCanvas() {}, pos: [100, 100], size: [220, 100] };
    const ctx = {
        key: "dmodel:42:" + NAME,
        node: fakeNode,
        anchorRect: () => ({ left: 10, top: 10, right: 30, bottom: 30, width: 20, height: 20 }),
        getRow: () => row,
        patchRow: (patch) => Object.assign(row, patch),
        accent: "#f66744",
        prefs: () => ({ civitai: true, thumbs: true }),
        refresh: () => {},
        api: API.dmodelApi,
        hideTriggers: true,
        samplesKind: "diffusion_models",
        autoCivitai: true,
    };

    await I.openInfoPanelFor(ctx, NAME);
    // loadInfo -> renderBody -> autoCivitai -> renderBody 全部落地
    for (let i = 0; i < 12; i++) await tick();

    const panel = lastPanel();
    check("面板已打开", !!panel);

    // ── 契约 2：info 走 dmodel 域 ──
    check("info 请求命中 /dmodel_info",
        fetched.some((u) => u.startsWith("/api/sfnodes/dmodel_info")));
    check("绝无 /lora_info 回退", !fetched.some((u) => u.includes("/api/sfnodes/lora_info")));
    check("绝无 /lora/civitai 回退", !fetched.some((u) => u.includes("/api/sfnodes/lora/civitai")));

    // ── 契约 3：hideTriggers —— 触发词区块整块缺席 ──
    check("无 Trigger words 区块", !findByText(panel, "Trigger words"));
    check("无触发词输入占位", !findByText(panel, "add your own trigger word… (comma or Enter for batch)"));

    // ── 契约 4：custom_description 到达面板（"Custom" 档按钮仅在有
    //    custom_description 或编辑中才渲染——行为级代理断言，不耦合
    //    renderMarkdown 的 mock 内部结构）──
    const dsecEl = findByClass(panel, "sf-ls-desc");
    check("描述区存在", !!dsecEl);
    const hasCustomTab = findByText(dsecEl, "Custom");
    check("Custom 描述档出现（custom_description 已送达）", !!hasCustomTab);
    check("File/Civitai 档齐备", !!findByText(dsecEl, "File") && !!findByText(dsecEl, "Civitai"));

    // ── 契约 5：autoCivitai 自动查询一次 ──
    check("打开即自动匹配 Civitai", civitaiCalls === 1);

    // ── 契约 6：samplesKind 注入 samples URL ──
    check("samples 列表带 kind=diffusion_models",
        samplesUrls.length > 0 && samplesUrls.every((u) => u.includes("kind=diffusion_models")));

    console.log("");
    if (failures.length) {
        console.log("FAILED:", failures.length, "check(s)");
        process.exit(1);
    }
    console.log("ALL PASS");
    process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
