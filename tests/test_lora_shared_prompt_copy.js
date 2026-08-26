// sf_lora_shared_info.attachSamplePromptCopyButtons 冒烟测试（Node 直接运行：
// node tests/test_lora_shared_prompt_copy.js）
// mock DOM/clipboard 真实加载 sf_lora_shared_info.js + sf_common.js，验证：
//   - civitai 标题紧邻的 pre 注入复制按钮（常驻半透明样式类），非紧邻/普通
//     代码块不注入
//   - 点击按钮：剪贴板收到围栏原文；notify 成功文案
//   - clipboard 失败（含 execCommand 回退失败）：notify 失败文案
//   - 同一容器重复 attach 不产生重复按钮/CSS
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// ── mock DOM ──────────────────────────────────────────────────────────────
function makeEl(tag) {
    const el = {
        tagName: String(tag || "div").toUpperCase(),
        style: {},
        dataset: {}, children: [], listeners: {}, _ownText: "",
        title: "", className: "",
        disabled: false, isConnected: true,
        classList: {
            add(...c) { c.forEach((x) => { el.className = [...new Set(String(el.className).split(/\s+/).filter(Boolean).concat(x))].join(" "); }); },
            contains(c) { return String(el.className).split(/\s+/).includes(c); },
        },
        append(...kids) { el.children.push(...kids); },
        appendChild(c) { el.children.push(c); return c; },
        remove() {},
        focus() {}, blur() {}, select() {},
        querySelector(sel) {
            const wantCls = sel.startsWith(".") ? sel.slice(1) : null;
            const walk = (e) => {
                for (const c of e.children || []) {
                    if (wantCls && String(c.className || "").split(/\s+/).includes(wantCls)) return c;
                    const f = walk(c);
                    if (f) return f;
                }
                return null;
            };
            return walk(el);
        },
        addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
        emit(type, evt) {
            const e = evt || { target: el };
            e.stopPropagation ||= () => {};
            e.preventDefault ||= () => {};
            for (const fn of [...(el.listeners[type] || [])]) fn(e);
        },
    };
    Object.defineProperty(el, "textContent", {
        get() {
            return el._ownText + (el.children || []).map((c) => String(c.textContent ?? "")).join("");
        },
        set(v) { el._ownText = String(v); el.children = []; },
    });
    return el;
}

const injectedStyles = new Map();
globalThis.document = {
    createElement(tag) { return makeEl(tag); },
    getElementById(id) { return injectedStyles.get(id) || null; },
    execCommand() { return false; }, // 剪贴板回退路径恒失败，锁定 notify 文案
    head: { appendChild(c) { if (c.id) injectedStyles.set(c.id, c); } },
    body: { appendChild() {}, append() {} },
};
globalThis.window = { innerWidth: 1280, innerHeight: 720 };
let clipImpl = async () => {};
// Node ≥21 的 navigator 是 getter-only 全局，直接赋值静默失效——必须 defineProperty 覆盖
Object.defineProperty(globalThis, "navigator", {
    value: { clipboard: { writeText: (t) => clipImpl(t) } },
    writable: true, configurable: true,
});
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.queueMicrotask = (fn) => Promise.resolve().then(fn);
globalThis.app = {};
globalThis.api = {};

// ── 加载模块（/scripts/* -> globalThis；相对 import 改 .mjs）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_lspc_"));
for (const n of ["sf_common.js", "sf_lora_shared_info.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

// 工具：按后端 _format_sample_prompts 渲染产物搭描述容器
function h3(text) { const e = makeEl("h3"); e.textContent = text; return e; }
function pre(prompt) {
    const e = makeEl("pre");
    const code = makeEl("code");
    code.textContent = prompt;
    e.appendChild(code);
    return e;
}
function plain(tag, text) { const e = makeEl(tag); e.textContent = text ?? ""; return e; }
function collectButtons(root, out = []) {
    for (const c of root.children || []) {
        if (String(c.className || "").split(/\s+/).includes("sf-ls-desc-copybtn")) out.push(c);
        collectButtons(c, out);
    }
    return out;
}

(async () => {
    const I = await import(path.join(tmpDir, "sf_lora_shared_info.mjs"));
    check("导出存在", typeof I.attachSamplePromptCopyButtons === "function"
        && typeof I.SAMPLE_ICON_PROMPT === "string" && typeof I.makeSampleIcon === "function");

    const PROMPT = "masterpiece, best quality, 1girl\n// resource comment line";
    const db = makeEl("div");
    db.children.push(
        h3("civitai_00_1a2b3c4d — 140313761"),
        pre(PROMPT),
        plain("em", "*Steps: 12, CFG: 1*"),
        h3("About this model"),
        pre("plain code block"),
        h3("civitai_05_99887766 — 140313799"),
        plain("p", "no prompt for this one"),
    );
    const msgs = [];
    I.attachSamplePromptCopyButtons(db, (m) => msgs.push(m));

    let btns = collectButtons(db);
    check("仅 civitai 紧邻代码块出按钮（共 1 个）", btns.length === 1);
    check("按钮落在第一个 pre 内", btns[0] && db.children[1].children.includes(btns[0]));
    check("pre 设为 relative 定位", db.children[1].style.position === "relative");
    check("按钮 tooltip", btns[0]?.title === "Copy this prompt to clipboard");
    check("CSS 已注入", injectedStyles.has("sf-lora-desc-copybtn"));

    // 点击 → 剪贴板内容 = 围栏原文；notify 成功文案
    let clipText = "";
    clipImpl = async (t) => { clipText = t; };
    btns[0].emit("click");
    await tick();
    check("剪贴板收到 prompt 原文", clipText === PROMPT);
    check("成功消息", msgs[msgs.length - 1] === "Prompt copied to clipboard.");
    check("点击后透明度恢复 CSS 接管", btns[0].style.opacity === "");

    // 重复 attach 幂等：不重复注入按钮与 CSS
    I.attachSamplePromptCopyButtons(db, (m) => msgs.push(m));
    btns = collectButtons(db);
    check("重复 attach 不出重复按钮", btns.length === 1);

    // clipboard 失败（execCommand 回退也失败）→ 失败文案
    const db2 = makeEl("div");
    db2.children.push(h3("civitai_01_deadbeef — 99"), pre("another prompt"));
    clipImpl = async () => { throw new Error("denied"); };
    I.attachSamplePromptCopyButtons(db2, (m) => msgs.push(m));
    const btn2 = collectButtons(db2)[0];
    check("第二个容器出按钮", !!btn2);
    btn2.emit("click");
    await tick();
    check("失败消息", msgs[msgs.length - 1] === "Could not copy to clipboard.");

    // 容器为空/null 安全
    I.attachSamplePromptCopyButtons(null, (m) => msgs.push(m));
    check("null 容器安全", true);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(1);
});
