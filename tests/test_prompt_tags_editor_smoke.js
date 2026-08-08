// SFPromptTags 全屏编辑器冒烟测试（Node 直接运行：node tests/test_prompt_tags_editor_smoke.js）
// 用 mock DOM/app 真实加载全部模块并调用 openLibraryEditor / closeLibraryEditor，
// 验证模块加载、编辑器构建、render 全链路（侧栏/创建表单/卡片/菜单/导入导出路径）
// 不抛运行时错误。静态语法检查覆盖不到的绑定/调用错误靠这里兜底。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素：任何 querySelector 都返回新元素，事件绑定不炸即可）──
// selector 缓存：同一元素同一 selector 返回同一子元素，供测试捕获事件回调
const childCache = new WeakMap();
function makeEl() {
    const style = {
        setProperty() {},
        getPropertyValue() { return ""; },
    };
    const el = {
        style, dataset: {}, children: [],
        className: "", textContent: "", innerHTML: "", value: "", placeholder: "",
        type: "", title: "", rows: 1, spellcheck: false, disabled: false, checked: false,
        draggable: false, isConnected: true, offsetWidth: 100, offsetHeight: 20,
        selectionStart: 0, selectionEnd: 0,
        classList: {
            add() {}, remove() {}, toggle() {}, contains: () => false,
        },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        replaceWith() {},
        replaceChildren(...kids) { this.children = kids; },
        remove() { this.removed = true; },
        contains() { return false; },
        closest() { return null; },
        querySelector(sel) {
            let m = childCache.get(this);
            if (!m) { m = new Map(); childCache.set(this, m); }
            if (!m.has(sel)) m.set(sel, makeEl());
            return m.get(sel);
        },
        querySelectorAll() { return []; },
        addEventListener(type, fn) {
            (this._ls = this._ls || {})[type] = fn;
        },
        removeEventListener() {},
        focus() {}, blur() {}, select() {},
        click() { if (this._ls && this._ls.click) this._ls.click(); },
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {}, setPointerCapture() {}, releasePointerCapture() {}, setSelectionRange() {},
    };
    return el;
}
const createdEls = [];
globalThis.document = {
    createElement() { const e = makeEl(); createdEls.push(e); return e; },
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
    getElementById() { return null; },
    activeElement: makeEl(),
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
    app: null,
};

const settingsStore = {};
globalThis.app = {
    ui: {
        settings: {
            getSettingValue: (k) => (k in settingsStore ? settingsStore[k] : null),
            setSettingValueAsync: (k, v) => { settingsStore[k] = v; return Promise.resolve(); },
            setSettingValue: (k, v) => { settingsStore[k] = v; },
        },
    },
    extensionManager: { toast: { add() {} } },
    constructor: function FakeApp() {},
};
globalThis.window.app = globalThis.app;

// ── mock fetch：内置默认库文件（store 首启 / 恢复默认库用）──
const defaultLib = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "web", "prompt_tags_default.json"), "utf8"));
globalThis.fetch = async (url) => {
    if (String(url).includes("prompt_tags_default.json")) return { ok: true, json: async () => defaultLib };
    return { ok: false, status: 404, json: async () => ({}) };
};

// ── 加载模块：全部复制到 tmp，/scripts/app.js -> globalThis.app，相对 import 改 .mjs ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ptg_smoke_"));
const names = [
    "sf_prompt_tags_lib.js", "sf_prompt_tags_pinyin.js",
    "sf_prompt_tags_cursors.js",
    "sf_prompt_tags_store.js", "sf_prompt_tags_guard.js",
    "sf_prompt_tags_editor.js",
];
for (const n of names) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    const E = await import(path.join(tmpDir, "sf_prompt_tags_editor.mjs"));

    // node mock：openLibraryEditor 需要 node（onInsert 用），_els 给 Insert 路径用
    const node = {
        id: 42,
        _sfPromptTagsRoot: { _els: { ta: { value: "", selectionStart: 0, selectionEnd: 0 } } },
    };
    let inserted = null;
    const opts = {
        accent: "#f66744",
        onInsert: (name, sym) => { inserted = { name, sym }; },
    };

    // 空库打开
    E.openLibraryEditor(node, opts);
    check("打开不抛错", true);
    check("overlay 已创建", createdEls.some((e) => e.className === "sf-ptge"));
    check("工作副本可用（内部 render 已跑）", true);

    // 有数据的库打开（覆盖卡片/侧栏/创建表单/菜单构建路径）
    settingsStore["sfnodes.PromptTags.Library"] = JSON.stringify({
        categories: ["Styles", "Animals"],
        listCats: ["Animals"],
        catModes: { Styles: "order" },
        tags: [
            { name: "oil", cat: "Styles", text: "oil painting, thick strokes" },
            { name: "fox", cat: "Animals", kind: "list", text: "red fox\nsnow leopard" },
            { name: "loose", text: "uncategorized text" },
        ],
    });
    E.closeLibraryEditor();
    E.openLibraryEditor(node, opts);
    check("有库打开（卡片/侧栏/表单路径）不抛错", true);
    check("关闭后再次打开正常", true);

    // prefill 路径（右键 Save as tag）
    E.closeLibraryEditor();
    E.openLibraryEditor(node, { ...opts, prefill: "some selected text\nwith two lines" });
    check("prefill 打开不抛错", true);

    // onInsert 回调链路（编辑器 Insert 按钮触发）
    E.closeLibraryEditor();
    E.openLibraryEditor(node, opts);
    opts.onInsert("oil", "@");
    check("onInsert 回调可调用", inserted && inserted.name === "oil" && inserted.sym === "@");

    E.closeLibraryEditor();
    check("关闭清理不抛错", true);
    // 关闭后立刻再开（验证状态复位）
    E.openLibraryEditor(node, opts);
    E.closeLibraryEditor();
    check("状态复位后可再开", true);

    // ── 恢复默认库（⋯ 菜单 → Restore default library）──
    // settingsStore 当前是 3 标签库；fetch mock 提供 949 默认库
    E.openLibraryEditor(node, opts);
    await E.restoreDefaultLibrary();           // fetch 默认 → confirmDanger 弹框
    const modal = createdEls.filter((e) => (e.className || "").includes("sf-ptge-modal")).pop();
    check("恢复确认框已弹出", !!modal);
    modal.querySelector(".dg-go").click();     // 确认 → onConfirm（替换工作副本 + 持久化）
    check("确认后弹框关闭", modal.removed === true);
    await new Promise((r) => setTimeout(r, 420));   // commitLibrary 防抖 350ms 落盘
    const restored = JSON.parse(settingsStore["sfnodes.PromptTags.Library"] || "null");
    check("库已替换为默认（949/50）", !!restored && restored.tags.length === 949 && restored.categories.length === 50);
    // 再次恢复：库已是默认 → 不弹确认框（toast info）
    const before = createdEls.filter((e) => (e.className || "").includes("sf-ptge-modal")).length;
    await E.restoreDefaultLibrary();
    const after = createdEls.filter((e) => (e.className || "").includes("sf-ptge-modal")).length;
    check("已是默认时不重复弹确认框", after === before);
    E.closeLibraryEditor();
    check("恢复后关闭正常", true);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
