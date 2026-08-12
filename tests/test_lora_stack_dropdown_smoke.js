// SF LoRA Stack 下拉弹窗定位冒烟测试（Node 直接运行：node tests/test_lora_stack_dropdown_smoke.js）
// mock DOM/app/api/fetch 真实加载 sf_lora_stack_dropdown.js，验证：
//   - 打开时按空内容高度定方向（向下）
//   - 目录导航后弹窗变高 → 重新锚定翻到上方（place 随 renderList 重跑）
//   - ‹ back 回退高度恢复 → 翻回下方
//   - 搜索过滤同样触发重锚
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素 + 事件记录/分发；offsetHeight/offsetWidth 可写以
//    模拟弹窗高度随列表变化；innerHTML/textContent 赋值清空子节点）──
function makeEl() {
    const el = {
        style: { setProperty() {}, getPropertyValue() { return ""; } }, dataset: {}, children: [], listeners: {},
        className: "", value: "", placeholder: "", title: "", type: "",
        disabled: false, isConnected: true,
        offsetWidth: 100, offsetHeight: 20, _text: "",
        classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        remove() { this.removed = true; },
        contains() { return false; },
        querySelector() { return null; }, querySelectorAll() { return []; },
        addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
        removeEventListener(type, fn) {
            const a = this.listeners[type];
            if (a) { const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); }
        },
        emit(type, evt) {
            const e = evt || { target: this, key: "" };
            e.stopPropagation ||= () => {};
            for (const fn of [...(this.listeners[type] || [])]) fn(e);
        },
        focus() {}, blur() {}, select() {}, click() { this.emit("click", { target: this }); },
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
    };
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
    createElement() { return makeEl(); },
    createTextNode(t) { return { textContent: t }; },
    body: { appendChild(c) { bodyChildren.push(c); return c; }, contains() { return false; } },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
    getElementById() { return null; },
    activeElement: makeEl(),
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
};
globalThis.app = { graph: {}, ui: { settings: { getSettingValue: () => null } } };
globalThis.window.app = globalThis.app;
globalThis.api = { apiURL: (r) => r };

// ── fetch mock：/api/sfnodes/lora_list 返回两级目录 ──
const LORAS = [
    "root.safetensors",
    "dirA/a1.safetensors", "dirA/a2.safetensors", "dirA/a3.safetensors",
    "dirB/b1.safetensors",
];
globalThis.fetch = async (url) => {
    if (String(url).includes("/api/sfnodes/lora_list")) {
        return { ok: true, json: async () => ({ loras: LORAS }) };
    }
    throw new Error("unexpected fetch: " + url);
};

// ── 加载模块（/scripts/app|api.js -> globalThis；相对 import 改 .mjs）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_lsd_"));
for (const n of ["sf_lora_stack_core.js", "sf_lora_stack_api.js",
    "sf_common.js", "sf_lora_stack_dropdown.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    const D = await import(path.join(tmpDir, "sf_lora_stack_dropdown.mjs"));
    check("dropdown 模块加载", typeof D.openLoraDropdown === "function");

    // 锚点：下方空间随弹窗高度变化（rect 为视口坐标）。
    // innerHeight=720，下边距 8 → 可用底部 = 712。
    //   h=20 :  580+4+20 = 604 ≤ 712 → 向下 top=584
    //   h=520: 580+4+520 = 1104 > 712 → 上翻 top=max(8, 560-4-520)=36
    const anchor = makeEl();
    anchor.getBoundingClientRect = () =>
        ({ left: 100, top: 560, right: 300, bottom: 580, width: 200, height: 20 });

    let picked = null;
    await D.openLoraDropdown(anchor, { current: "", onPick: (v) => { picked = v; } });
    const pop = bodyChildren[bodyChildren.length - 1];
    const list = pop.children[2]; // srch, crumb, list
    const input = pop.children[0].children[1];

    check("popup 已打开", !!pop && pop.className === "sf-ls-dd");
    check("初始（小高度）向下展开", pop.style.top === "584px");
    check("根层列出文件夹与文件", list.children.length === 3
        && list.children[0].children[1]._text === "dirA"
        && list.children[2].children[0].textContent === "root.safetensors");

    // 导航进大目录：弹窗高度增长 → 必须重新锚定翻到上方
    pop.offsetHeight = 520;
    list.children[0].emit("click"); // 📁 dirA
    check("导航后（大高度）翻到上方", pop.style.top === "36px");
    check("进入 dirA 渲染 back + 3 文件", list.children.length === 4
        && list.children[0].className === "sf-ls-dd-back"
        && list.children[3].children[0].textContent === "a3.safetensors");

    // 搜索过滤同样重锚（保持大高度 → 仍向上）
    input.value = "a2";
    input.emit("input");
    check("搜索渲染命中行", list.children.length === 1
        && list.children[0].children[0].textContent === "a2.safetensors");
    check("搜索后保持向上（大高度）", pop.style.top === "36px");

    // 搜索无匹配 → 空态，重锚不崩
    input.value = "zzz";
    input.emit("input");
    check("搜索无匹配空态", list.children.length === 1 && list.children[0].className === "sf-ls-dd-empty");

    // 清空搜索回到目录视图，高度恢复 → 翻回下方
    input.value = "";
    input.emit("input");
    pop.offsetHeight = 20;
    list.children[0].emit("click"); // ‹ back
    check("back 回根层", list.children.length === 3 && list.children[0].children[1]._text === "dirA");
    check("高度恢复后翻回下方", pop.style.top === "584px");

    // 选择文件 → onPick 回调 + 弹窗关闭
    list.children[2].emit("click"); // root.safetensors
    check("点击文件回调 onPick", picked === "root.safetensors");
    check("选择后弹窗已关闭", pop.removed === true);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(1);
});
