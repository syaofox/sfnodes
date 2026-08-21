// SFLoraStack 信息面板 Description dirty 流程冒烟测试（Node 直接运行：
// node tests/test_lora_stack_info_desc_smoke.js）
// mock DOM/app/api/fetch 真实加载 sf_lora_stack_info.js，验证：
//   - 进入编辑出现 textarea；改动草稿 → Save 按钮 dirty 高亮；改回基准不高亮
//   - textarea 内 Esc：dirty 时弹确认框（误按保护），取消保留草稿
//   - ✕ 关闭：dirty 时弹确认框；Discard 后面板关闭、编辑状态重置
//   - 重开面板不残留上一行的编辑态（防草稿泄漏）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// ── mock DOM（惰性元素；tagName 区分 textarea/h4；querySelector 递归
//    按 className/tagName 匹配；className 与 classList 双向同步——
//    单向会复制出真实 DOM 没有的 bug：el.className="qa" 后 classList._s
//    为空，第一次 toggle 的 sync 用 _s 覆盖 className 把 "qa" 冲掉）──
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
        querySelectorAll() { return []; },
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
    // className 与 _cls 双向同步（真实 DOM 语义：classList 是 className 的视图）
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
globalThis.app = {
    graph: { _nodes: [], setDirtyCanvas() {} },
    api: { fetchApi: async () => ({ ok: false, json: async () => ({}) }) },
};
globalThis.api = { apiURL: (r) => r };

// ── fetch mock：lora_list + lora_info（file 源描述）──
globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/sfnodes/lora_list")) {
        return { ok: true, json: async () => ({ loras: ["a.safetensors", "b.safetensors"] }) };
    }
    if (u.includes("/api/sfnodes/lora_info?name=")) {
        return { ok: true, json: async () => ({
            ok: true,
            info: {
                source: "file",
                description: "original desc",
                triggers: [],
                custom_triggers: [],
                custom_description: "",
                civitai_host: "com",
            },
        }) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
};

// ── 加载模块（/scripts/app|api.js -> globalThis；相对 import 改 .mjs）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_lsid_"));
for (const n of ["sf_lora_stack_core.js", "sf_lora_stack_api.js",
    "sf_lora_stack_settings.js", "sf_common.js", "sf_markdown.js",
    "sf_lora_shared_info.js", "sf_lora_info.js", "sf_lora_stack_info.js", "sf_workflows_ui.js",
    "sf_workflows_lib.js", "sf_lora_stack_dropdown.js",
    "sf_lora_stack_render.js", "sf_lora_stack_interaction.js"]) {
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
const lastMask = () => [...bodyChildren].reverse().find((c) => hasClass(c, "sf-ls-confirm-mask"));
const lastPanel = () => [...bodyChildren].reverse().find((c) => hasClass(c, "sf-ls-info-p"));

(async () => {
    const I = await import(path.join(tmpDir, "sf_lora_stack_info.mjs"));
    check("info 模块加载", typeof I.openInfoPanel === "function");

    const node = {
        id: 1, comfyClass: "SFLoraStack", type: "SFLoraStack",
        properties: {
            loraStackState: JSON.stringify({
                version: 1, sep: ", ", step: 0.05, defStrength: 1.0,
                linkStrength: true, civitai: false, thumbs: false, hideExt: true, cacheMode: "last",
                loras: [
                    { id: "l1", name: "a.safetensors", on: true, sm: 1, sc: 1, triggers: [], custom: [] },
                    { id: "l2", name: "b.safetensors", on: true, sm: 1, sc: 1, triggers: [], custom: [] },
                ],
            }),
        },
        widgets: [],
        size: [336, 200],
        pos: [100, 100],
    };

    // ── 打开面板 → 进入 Description 编辑 ──
    await I.openInfoPanel(node, "l1", () => {});
    let panel = lastPanel();
    check("面板已打开", !!panel);
    check("浏览态无 textarea", panel.querySelector(".sf-ls-desc textarea") === null);

    findByText(panel, "✏️").emit("click");
    let ta = panel.querySelector(".sf-ls-desc textarea");
    check("进入编辑出现 textarea", !!ta);
    check("textarea 预填基准值", ta.value === "original desc");
    let save = findByText(panel, "Save");
    check("初始 Save 无高亮", !!save && !hasClass(save, "dirty"));

    // ── 改动 → 高亮；改回基准 → 不高亮 ──
    ta.value = "changed desc";
    ta.emit("input");
    check("改动后 Save 高亮", hasClass(findByText(panel, "Save"), "dirty"));
    ta.value = "original desc";
    ta.emit("input");
    check("改回基准不高亮", !hasClass(findByText(panel, "Save"), "dirty"));
    ta.value = "changed again";
    ta.emit("input");
    check("再次改动高亮", hasClass(findByText(panel, "Save"), "dirty"));

    // ── textarea 内 Esc：dirty 时弹确认框；取消保留草稿 ──
    ta.emit("keydown", { key: "Escape", target: ta });
    let mask = lastMask();
    check("Esc 弹确认框", !!mask);
    check("确认框标题", findByClass(mask, "sf-ls-confirm-t")._text === "Discard description changes?");
    findByText(mask, "Keep editing").emit("click");
    await tick();
    check("Esc 取消后草稿保留", panel.querySelector(".sf-ls-desc textarea") !== null
        && panel.querySelector(".sf-ls-desc textarea").value === "changed again");
    check("Esc 取消后仍高亮", hasClass(findByText(panel, "Save"), "dirty"));

    // ── ✕ 关闭：dirty 时确认；Discard 后关闭 ──
    findByClass(panel, "sf-ls-info-x").emit("click");
    mask = lastMask();
    check("✕ 弹确认框", !!mask);
    findByText(mask, "Discard").emit("click");
    await tick();
    check("确认丢弃后面板关闭", panel.removed === true);

    // ── 重开：编辑态已重置（草稿不泄漏到下一面板）──
    await I.openInfoPanel(node, "l1", () => {});
    panel = lastPanel();
    check("重开面板无残留 textarea", panel.querySelector(".sf-ls-desc textarea") === null);
    check("重开无 Save 按钮（浏览态）", findByText(panel, "Save") === null);

    // ── 外部点击关闭：查看态关闭；编辑态（dirty）不关；拖动/面板内/确认框豁免 ──
    // document 监听在面板打开的 setTimeout(0) 后挂载——每次打开面板后必须先
    // await tick()（宏任务）再发事件，否则监听器还没挂上（假阳性）。
    // click 前需先 pointerdown 记录按下位置（位移判定区分点击与拖动）。
    const docClick = (target, dx, dy, cx, cy) => {
        document.emit("pointerdown", { clientX: dx ?? 10, clientY: dy ?? 10 });
        document.emit("click", {
            clientX: cx ?? 10, clientY: cy ?? 10,
            target: target ?? { closest: () => null },
        });
    };

    // 查看态：点击面板外 → 关闭
    await tick();
    docClick({ closest: () => null });
    check("查看态外部点击关闭", panel.removed === true);

    // 重开 → 编辑 → dirty → 外部点击不关、草稿保留
    await I.openInfoPanel(node, "l1", () => {});
    panel = lastPanel();
    await tick();
    findByText(panel, "✏️").emit("click");
    ta = panel.querySelector(".sf-ls-desc textarea");
    ta.value = "draft keep";
    ta.emit("input");
    docClick({ closest: () => null });
    check("编辑态外部点击不关闭", panel.removed !== true);
    check("编辑态外部点击草稿保留", panel.querySelector(".sf-ls-desc textarea").value === "draft keep");

    // 拖动（位移 > 6px）不算点击 → 不关闭
    docClick({ closest: () => null }, 10, 10, 200, 200);
    check("拖动不关闭", panel.removed !== true);

    // 面板内点击 → 不关闭
    docClick(panel);
    check("面板内点击不关闭", panel.removed !== true);

    // 确认框豁免：dirty Esc 弹框后点击框内元素不关闭（框挂在 body 不在面板内）
    ta.emit("keydown", { key: "Escape", target: ta });
    mask = lastMask();
    check("外部点击用例确认框弹出", !!mask);
    docClick({ closest: () => mask });
    check("确认框点击不关闭", panel.removed !== true);
    findByText(mask, "Keep editing").emit("click");
    await tick();

    // 监听器解绑：关闭后面板的 document 监听不再触发（防孤儿监听器）
    const snap = { keydown: document._listeners.keydown?.length || 0,
                   click: document._listeners.click?.length || 0,
                   pointerdown: document._listeners.pointerdown?.length || 0 };
    findByClass(panel, "sf-ls-info-x").emit("click");
    mask = lastMask();
    findByText(mask, "Discard").emit("click");
    await tick();
    check("关闭后解绑 document 监听", panel.removed === true
        && (document._listeners.click?.length || 0) < snap.click
        && (document._listeners.pointerdown?.length || 0) < snap.pointerdown
        && (document._listeners.keydown?.length || 0) < snap.keydown);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(1);
});
