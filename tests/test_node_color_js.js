// SF Node Color 前端冒烟测试（Node 直接运行：node tests/test_node_color_js.js）
// 覆盖（.mjs 拷贝链真实加载，test_canvas_menu_js.js 同款手法）：
// - buildNodeColorMenuItem：0 选中返回 null；≥1 选中返回 "SF Node Color…"
// - 驱动菜单项 → 取色面板挂载；应用 → 规范化 hex 写入 node.color/bgcolor + 撤销钩子
// - 最近色 localStorage 持久化 + swatch 渲染
// - 清除颜色 → 置 undefined；overlay 自点击关闭
// 复用真源 sf_node_color.js / sf_node_color_lib.js / sf_canvas_align_lib.js /
// sf_popup.js / sf_common.js（仅替换绝对 import 与相对 .js 扩展名）。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const RECENT_KEY = "sfnodes.node_color.recent";
const store = {};

function makeEl(tag, cls, text) {
    return {
        tag: tag || "div", tagName: (tag || "div").toUpperCase(), className: cls || "", textContent: text || "",
        value: "", type: "", spellcheck: true, id: "", title: "",
        children: [], _parent: null, style: {}, dataset: {}, _handlers: {},
        setAttribute() {}, removeAttribute() {},
        addEventListener(t, fn) { (this._handlers[t] = this._handlers[t] || []).push(fn); },
        removeEventListener() {},
        appendChild(c) { c._parent = this; this.children.push(c); return c; },
        append(...cs) { for (const c of cs) { c._parent = this; this.children.push(c); } },
        remove() {
            if (this._parent) {
                const i = this._parent.children.indexOf(this);
                if (i >= 0) this._parent.children.splice(i, 1);
                this._parent = null;
            }
            this._removed = true;
        },
        getBoundingClientRect() { return { left: 0, top: 0, width: 280, height: 220 }; },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        focus() {}, blur() {}, select() {}, click() {},
    };
}

const docHandlers = {};
globalThis.document = {
    createElement: (t) => makeEl(t),
    createTextNode: (t) => ({ text: t }),
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    head: { appendChild() {} },
    body: { children: [], appendChild(c) { c._parent = this; this.children.push(c); return c; } },
    addEventListener(t, fn) { (docHandlers[t] = docHandlers[t] || []).push(fn); },
    removeEventListener() {},
};
globalThis.window = {
    innerWidth: 1280, innerHeight: 800, devicePixelRatio: 1,
    addEventListener() {}, removeEventListener() {},
};
globalThis.localStorage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
};

// ── app stubs ──
const graphCalls = { before: 0, after: 0 };
const fakeGraph = {
    beforeChange() { graphCalls.before++; },
    afterChange() { graphCalls.after++; },
};
const toasts = [];
globalThis.app = {
    canvas: { ds: { scale: 1 }, setDirty() {} },
    graph: fakeGraph,
    extensionManager: { toast: { add: (t) => toasts.push(t) } },
};
globalThis.api = { apiURL: (r) => r };

// ── 复制依赖链为 .mjs 并真实加载 ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_nc_"));
const MODS = ["sf_node_color.js", "sf_node_color_lib.js", "sf_canvas_align_lib.js", "sf_popup.js", "sf_common.js"];
for (const n of MODS) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

function all(root, pred, out = []) {
    if (pred(root)) out.push(root);
    for (const c of root.children || []) all(c, pred, out);
    return out;
}
function fire(e, type, evt) {
    for (const fn of (e._handlers[type] || [])) fn(evt || {});
}

(async () => {
    const mod = await import(path.join(tmpDir, "sf_node_color.mjs"));

    // ── 0 选中 ──
    globalThis.app.canvas.selected_nodes = [];
    check("0 选中返回 null", mod.buildNodeColorMenuItem() === null);

    // ── 1 选中：出现菜单项 ──
    const n1 = { color: undefined, bgcolor: undefined };
    const n2 = { color: undefined, bgcolor: undefined };
    globalThis.app.canvas.selected_nodes = [n1, n2];
    const item = mod.buildNodeColorMenuItem();
    check("选中出现菜单项", !!item && item.content === "SF Node Color…" && typeof item.callback === "function");

    // ── 驱动：面板挂载 ──
    item.callback();
    check("面板挂载到 body", document.body.children.length === 1);
    const overlay = document.body.children[0];
    const hex = all(overlay, (e) => e.className === "sf-nc-hex")[0];
    const colorInput = all(overlay, (e) => e.className === "sf-nc-color")[0];
    const applyBtn = all(overlay, (e) => e.tagName === "BUTTON" && e.textContent === "应用")[0];
    const clearBtn = all(overlay, (e) => e.tagName === "BUTTON" && e.textContent === "清除颜色")[0];
    check("面板含 hex/取色器/按钮", !!(hex && colorInput && applyBtn && clearBtn));

    // ── 应用（大写 + 无 # → 规范化） ──
    hex.value = "A1B2C3";
    fire(applyBtn, "click");
    check("应用写规范化 hex 到标题与节点体",
        n1.color === "#a1b2c3" && n1.bgcolor === "#a1b2c3" && n2.color === "#a1b2c3");
    check("应用走 beforeChange/afterChange", graphCalls.before === 1 && graphCalls.after === 1);
    check("应用成功 toast", toasts.some((t) => t.severity === "success"));
    check("最近色持久化", JSON.parse(store[RECENT_KEY] || "[]")[0] === "#a1b2c3");
    check("最近色 swatch 渲染", all(overlay, (e) => e.className === "sf-nc-swatch").length === 1);

    // ── 非法 hex 不改色 ──
    hex.value = "zzz";
    const before = n1.color;
    fire(applyBtn, "click");
    check("非法 hex 不改色 + 错误 toast",
        n1.color === before && toasts.some((t) => t.severity === "error"));

    // ── 最近色 swatch 点击可直接应用 ──
    const sw = all(overlay, (e) => e.className === "sf-nc-swatch")[0];
    fire(sw, "click");
    check("点击最近色应用", n1.color === "#a1b2c3");

    // ── 清除颜色 ──
    fire(clearBtn, "click");
    check("清除颜色置 undefined", n1.color === undefined && n1.bgcolor === undefined && n2.color === undefined);

    // ── overlay 自点击关闭 ──
    fire(overlay, "mousedown", { target: overlay, preventDefault() {}, stopPropagation() {} });
    check("overlay 自点击关闭", document.body.children.length === 0 && overlay._removed);

    // ── Esc 关闭（attachPopupDismiss 真源） ──
    item.callback();
    check("再次打开面板", document.body.children.length === 1);
    (docHandlers.keydown || []).forEach((fn) => fn({ key: "Escape" }));
    check("Esc 关闭面板", document.body.children.length === 0);

    if (failures.length) {
        console.log(`\n${failures.length} FAILED: ${failures}`);
        process.exit(1);
    }
    console.log("\nALL PASSED");
})().catch((e) => { console.error("FATAL:", e); process.exit(1); });
