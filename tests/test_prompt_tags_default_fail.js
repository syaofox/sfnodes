// SFPromptTags 默认库失败路径冒烟测试（Node 直接运行）
// 覆盖语法检查漏不掉的防御逻辑：
//   - fetch 内置默认库失败（404/断网）→ 空库保留、不落盘、会话内不重试
//   - restoreDefaultLibrary 在编辑器未打开时静默返回（不抛错）
// 需要独立模块实例（_defaultPromise 缓存成功/失败结果，与成功路径测试互斥）。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

function makeEl() {
    const style = { setProperty() {}, getPropertyValue() { return ""; } };
    return {
        style, dataset: {}, children: [], className: "", textContent: "", innerHTML: "",
        value: "", placeholder: "", type: "", title: "", rows: 1, spellcheck: false,
        disabled: false, checked: false, draggable: false, isConnected: true,
        offsetWidth: 100, offsetHeight: 20, selectionStart: 0, selectionEnd: 0,
        classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        append(...k) { this.children.push(...k); }, appendChild(c) { this.children.push(c); return c; },
        prepend(...k) { this.children.unshift(...k); }, replaceWith() {},
        replaceChildren(...k) { this.children = k; }, remove() { this.removed = true; },
        contains() { return false; }, closest() { return null; },
        querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        addEventListener() {}, removeEventListener() {}, focus() {}, blur() {}, select() {}, click() {},
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {}, setPointerCapture() {}, releasePointerCapture() {}, setSelectionRange() {},
    };
}
globalThis.document = {
    createElement() { return makeEl(); }, body: { appendChild() {} }, head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {}, getElementById() { return null; },
    activeElement: makeEl(),
};
globalThis.window = { addEventListener() {}, removeEventListener() {}, innerWidth: 1280, innerHeight: 720, app: null };

const settingsStore = {};
let fetchCalls = 0;
globalThis.fetch = async () => { fetchCalls++; return { ok: false, status: 404, json: async () => ({}) }; };
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

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ptg_fail_"));
for (const n of ["sf_prompt_tags_lib.js", "sf_prompt_tags_pinyin.js",
    "sf_prompt_tags_cursors.js", "sf_prompt_tags_store.js", "sf_prompt_tags_guard.js",
    "sf_prompt_tags_editor.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    const S = await import(path.join(tmpDir, "sf_prompt_tags_store.mjs"));
    const E = await import(path.join(tmpDir, "sf_prompt_tags_editor.mjs"));

    // ── 场景 1：fetch 失败 → 空库保留、不落盘、不重试 ──
    delete settingsStore["sfnodes.PromptTags.Library"];
    S.reloadLibrary();
    check("fetch 失败时首读为空库", S.getLibrary().tags.length === 0);
    await new Promise((r) => setTimeout(r, 30));
    check("失败后不写 settings", !("sfnodes.PromptTags.Library" in settingsStore));
    check("内存库保持空", S.getLibrary().tags.length === 0);
    // 会话内不重试：再次 reload + getLibrary 不发起新请求
    S.reloadLibrary();
    S.getLibrary();
    await new Promise((r) => setTimeout(r, 30));
    check("失败结果缓存（fetch 仅 1 次）", fetchCalls === 1);

    // ── 场景 2：编辑器未打开时 restoreDefaultLibrary 静默返回 ──
    let threw = false;
    try { await E.restoreDefaultLibrary(); } catch { threw = true; }
    check("编辑器未打开时 restore 不抛错", !threw);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
