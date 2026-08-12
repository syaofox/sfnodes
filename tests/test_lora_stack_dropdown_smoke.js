// SF LoRA Stack 下拉弹窗定位冒烟测试（Node 直接运行：node tests/test_lora_stack_dropdown_smoke.js）
// mock DOM/app/api/fetch 真实加载 sf_lora_stack_dropdown.js，验证：
//   - 方向在打开时定一次：比较上下可用空间选大者（空间大 → 向上展开）
//   - 展开期间方向永不翻转（导航变高/back 变矮只更新 top/maxHeight）
//   - 下方空间大 → 向下展开，top 恒定
//   - maxHeight 按所选方向空间钳制（≤ 60vh）：小空间下内容超高也不越界
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

// 视口 innerHeight=720、底部边距 8 → 底部可用 = 712。
function anchor(rect) {
    const a = makeEl();
    a.getBoundingClientRect = () => rect;
    return a;
}

(async () => {
    const D = await import(path.join(tmpDir, "sf_lora_stack_dropdown.mjs"));
    check("dropdown 模块加载", typeof D.openLoraDropdown === "function");

    // ── Case 1：上方空间大（up=556 > down=128）→ 打开即向上，方向永不翻转 ──
    //   rect: top=560, bottom=580
    //   upSpace = 560-4 = 556, downSpace = 712-580-4 = 128
    //   maxHeight = min(60vh=432, 556) = 432
    //   h=20  → top = 560-4-20 = 536；h=520 → top = 560-4-520 = 36
    let picked = null;
    await D.openLoraDropdown(anchor({ left: 100, top: 560, right: 300, bottom: 580, width: 200, height: 20 }), {
        current: "", onPick: (v) => { picked = v; },
    });
    let pop = bodyChildren[bodyChildren.length - 1];
    let list = pop.children[2];
    let input = pop.children[0].children[1];

    check("popup 已打开", !!pop && pop.className === "sf-ls-dd");
    check("上方空间大 → 打开即向上", pop.style.top === "536px");
    check("maxHeight 钳到方向空间（60vh 内）", pop.style.maxHeight === "432px");
    check("根层列出文件夹与文件", list.children.length === 3
        && list.children[0].children[1]._text === "dirA"
        && list.children[2].children[0].textContent === "root.safetensors");

    // 导航进大目录：高度增长 → 方向不变（仍向上），顶边延伸
    pop.offsetHeight = 520;
    list.children[0].emit("click"); // 📁 dirA
    check("导航后（大高度）方向不变仍向上", pop.style.top === "36px");
    check("导航后 maxHeight 不变", pop.style.maxHeight === "432px");
    check("进入 dirA 渲染 back + 3 文件", list.children.length === 4
        && list.children[0].className === "sf-ls-dd-back"
        && list.children[3].children[0].textContent === "a3.safetensors");

    // 搜索过滤：方向仍不变
    input.value = "a2";
    input.emit("input");
    check("搜索渲染命中行", list.children.length === 1
        && list.children[0].children[0].textContent === "a2.safetensors");
    check("搜索后仍向上（不跳变）", pop.style.top === "36px");

    // 高度恢复 → 仍向上（不回跳为向下）
    input.value = "";
    input.emit("input");
    pop.offsetHeight = 20;
    list.children[0].emit("click"); // ‹ back
    check("back 回根层", list.children.length === 3);
    check("高度恢复后仍向上（方向从未翻转）", pop.style.top === "536px");

    list.children[2].emit("click"); // root.safetensors
    check("点击文件回调 onPick", picked === "root.safetensors");
    check("选择后弹窗已关闭", pop.removed === true);

    // ── Case 2：下方空间大（down=588 > up=96）→ 向下展开，top 恒定 ──
    //   rect: top=100, bottom=120；downSpace = 712-120-4 = 588
    await D.openLoraDropdown(anchor({ left: 100, top: 100, right: 300, bottom: 120, width: 200, height: 20 }), {
        current: "", onPick: () => {},
    });
    pop = bodyChildren[bodyChildren.length - 1];
    list = pop.children[2];
    check("下方空间大 → 打开即向下", pop.style.top === "124px");
    pop.offsetHeight = 520;
    list.children[0].emit("click"); // 📁 dirA（大高度）
    check("导航后 top 恒定（仍向下）", pop.style.top === "124px");
    check("下方大空间 maxHeight 取 60vh", pop.style.maxHeight === "432px");

    D.closeLoraDropdown();

    // ── Case 3：所选方向空间小于 60vh → maxHeight 钳到方向空间，永不越界 ──
    //   rect: top=360, bottom=380；upSpace = 356, downSpace = 328
    //   向上展开，maxHeight = min(432, 356) = 356（钳制生效）
    //   h=20  → top = 360-4-20 = 336；h=520（实际被钳 356）→ top = 8
    await D.openLoraDropdown(anchor({ left: 100, top: 360, right: 300, bottom: 380, width: 200, height: 20 }), {
        current: "", onPick: () => {},
    });
    pop = bodyChildren[bodyChildren.length - 1];
    list = pop.children[2];
    check("方向空间比较选上方", pop.style.top === "336px");
    check("maxHeight 钳到方向空间（356 < 60vh）", pop.style.maxHeight === "356px");
    pop.offsetHeight = 520;
    list.children[0].emit("click");
    check("内容超高 top 钳到 8（顶边不越界）", pop.style.top === "8px");
    check("内容超高 maxHeight 不变（底边贴锚点）", pop.style.maxHeight === "356px");

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(1);
});
