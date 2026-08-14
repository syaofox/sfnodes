// SFPromptStack 前端冒烟测试（Node 直接运行：node tests/test_prompt_stack_smoke.js）
// 用 mock DOM/app 真实加载模块（sf_lora_stack.js 依赖替换为 mock buildIndex/
// findNode，避免其扩展副作用），验证：
//   - 模块加载 / 扩展注册 / prototype 包装
//   - nodeCreated setupNode：DOM widget、空状态提示、默认尺寸
//   - Add 按钮 → 行添加 + state 写入 + 聚焦新行
//   - textarea input → state 更新（不重建行）
//   - 开关 toggle → enabled 翻转 + 重渲染（index 语义：跳过关闭行）
//   - ▲▼ 排序 / ✕ 删除
//   - graphToPrompt 注入隐藏 PromptStackState
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

function makeEl() {
    let _inner = "";
    const el = {
        style: {}, dataset: {}, children: [], _handlers: {},
        className: "", textContent: "", value: "", placeholder: "",
        type: "", title: "", spellcheck: true, id: "", tabIndex: 0,
        disabled: false, rows: 2,
        classList: {
            _s: new Set(),
            add(...c) { c.forEach((x) => this._s.add(x)); },
            remove(...c) { c.forEach((x) => this._s.delete(x)); },
            toggle(c, force) { if (force === undefined) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); } else { force ? this._s.add(c) : this._s.delete(c); } },
            contains(c) { return this._s.has(c); },
        },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        replaceChildren(...kids) { this.children = kids; },
        remove() { this.removed = true; },
        addEventListener(name, fn) { this._handlers[name] = fn; },
        removeEventListener() {},
        focus() { this.focused = true; },
        // 递归找 className 含选择器子串的元素（聚焦新行输入用）
        querySelectorAll(sel) {
            const out = [];
            const walk = (el) => {
                for (const c of el.children || []) {
                    if (c.className && c.className.includes(sel.replace(".", ""))) out.push(c);
                    walk(c);
                }
            };
            walk(this);
            return out;
        },
    };
    // innerHTML 赋值 "" 时清空 children（代码用 list.innerHTML = "" 清列表）
    Object.defineProperty(el, "innerHTML", {
        configurable: true,
        get() { return _inner; },
        set(v) { _inner = v; if (v === "") this.children = []; },
    });
    return el;
}
globalThis.document = {
    createElement() { return makeEl(); },
    getElementById() { return null; },
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
};
globalThis.window = { addEventListener() {}, removeEventListener() {}, LiteGraph: { vueNodesMode: false } };
globalThis.crypto = { randomUUID: () => "uuid-" + Math.random().toString(36).slice(2, 10) };

// ── app mock：graphToPrompt 可被包装，注入钩子测试用它 ──
globalThis.app = {
    graph: { _nodes: [], links: {}, getNodeById() { return null; }, setDirtyCanvas() {} },
    registerExtension(ext) { this._ext = ext; },
    loadGraphData: async () => {},
    graphToPrompt: async () => ({ output: { "1": { class_type: "SFPromptStack", inputs: {} } } }),
};

// buildIndex/findNode 替换（sf_lora_stack.js 副作用大，mock 掉）
globalThis.__buildIndex = () => new Map([["1", globalThis.__testNode]]);
globalThis.__findNode = (index, id) => index.get(String(id));

function makeNode() {
    return {
        id: "1", comfyClass: "SFPromptStack", type: "SFPromptStack",
        widgets: [], inputs: [], outputs: [], properties: {},
        size: [440, 220],
        graph: { setDirtyCanvas() {} },
        setDirtyCanvas() {},
        addDOMWidget(name, type, el, opts) {
            const w = { name, type, options: opts || {}, element: el, value: null };
            this.widgets.push(w);
            return w;
        },
    };
}

(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ps_smoke_"));
    for (const n of ["sf_common.js", "sf_prompt_stack_core.js", "sf_prompt_stack.js"]) {
        let code = fs.readFileSync(path.join(__dirname, "..", "web", n), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
            .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;");
        // 先替换 sf_lora_stack 依赖（避免其扩展副作用），再统一转 .mjs
        code = code.replace(
            'import { buildIndex, findNode } from "./sf_lora_stack.js";',
            "const buildIndex = globalThis.__buildIndex; const findNode = globalThis.__findNode;");
        code = code.replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
        fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
    }
    await import(path.join(tmpDir, "sf_prompt_stack.mjs"));

    check("扩展已注册", app._ext?.name === "sfnodes.PromptStack");
    check("graphToPrompt 已包装", app.graphToPrompt._sfPsTestWrapped === undefined); // 包装为函数替换

    // beforeRegisterNodeDef 包装
    const FakeType = function () {};
    app._ext.beforeRegisterNodeDef(FakeType, { name: "SFPromptStack" });
    check("onNodeCreated 已包装", typeof FakeType.prototype.onNodeCreated === "function");
    check("onConfigure 已包装", typeof FakeType.prototype.onConfigure === "function");
    check("onResize 已包装", typeof FakeType.prototype.onResize === "function");
    check("onRemoved 已包装", typeof FakeType.prototype.onRemoved === "function");

    // nodeCreated → setupNode
    const node = makeNode();
    globalThis.__testNode = node;
    FakeType.prototype.onNodeCreated.call(node);
    const root = node._sfPsRoot;
    check("DOM widget 已添加", node.widgets.some((w) => w.name === "sf_prompt_stack_ui"));
    check("默认尺寸", node.size[0] === 440 && node.size[1] === 220);
    check("空状态提示", root.children[1].children[0]?.className?.includes("sf-ps-empty") === true);

    const addBtn = root.children[0];
    const list = root.children[1];

    // Add → 行添加 + state 写入 + 聚焦新行
    addBtn._handlers.click({ stopPropagation() {} });
    check("Add 添加一行", list.children[0]?.className?.includes("sf-ps-row") === true);
    const st1 = JSON.parse(node.properties.promptStackState);
    check("state 写入", st1.rows.length === 1 && st1.rows[0].enabled === true && st1.rows[0].text === "");
    // 行结构：[idx, tg, tawrap[ta, grip], btns]
    const row0 = list.children[0];
    const ta0 = row0.children[2].children[0];
    const grip0 = row0.children[2].children[1];
    check("行结构", row0.children.length === 4 && ta0?.className?.includes("sf-ps-ta") && grip0?.className?.includes("sf-ps-grip"));
    check("聚焦新行", ta0?.focused === true);
    check("空文本行显示占位", row0.children[0]?.textContent === "\u2013");
    check("默认行高", row0.style.height === "52px");

    // 输入文本 → state 更新（不重建行）+ index 出现
    const listRef = list.children;
    ta0.value = "hello world";
    ta0._handlers.input();
    const st2 = JSON.parse(node.properties.promptStackState);
    check("输入写 state", st2.rows[0].text === "hello world");
    check("输入不重建行", list.children === listRef);
    check("输入后 index 0", row0.children[0]?.textContent === "0");

    // Add 第二行 + 关闭第一行 → index 语义（跳过关闭行）
    addBtn._handlers.click({ stopPropagation() {} });
    list.children[1].children[2].children[0].value = "second";
    list.children[1].children[2].children[0]._handlers.input();
    list.children[0].children[1]._handlers.click({ stopPropagation() {} }); // 关闭第一行
    // renderRows 重建行 → 重新取引用
    const rowAfterToggle = list.children[1];
    check("关闭后行1 index 变 0", rowAfterToggle?.children[0]?.textContent === "0");
    check("关闭行显示占位", list.children[0]?.children[0]?.textContent === "\u2013");
    check("关闭行 tg 状态", list.children[0]?.children[1]?.className?.includes("off") === true);
    const st3 = JSON.parse(node.properties.promptStackState);
    check("关闭写 state", st3.rows[0].enabled === false);

    // 重新开启 → index 恢复
    list.children[0].children[1]._handlers.click({ stopPropagation() {} });
    check("开启后 index 恢复", list.children[0]?.children[0]?.textContent === "0"
        && list.children[1]?.children[0]?.textContent === "1");

    // ▲ 排序：第二行上移
    const btns1 = list.children[1].children[3].children; // [up, down, del]
    btns1[0]._handlers.click({ stopPropagation() {} });
    const st4 = JSON.parse(node.properties.promptStackState);
    check("上移交换", st4.rows[0].text === "second" && st4.rows[1].text === "hello world");

    // ✕ 删除：删第一行
    const btns0 = list.children[0].children[3].children;
    btns0[2]._handlers.click({ stopPropagation() {} });
    const st5 = JSON.parse(node.properties.promptStackState);
    check("删除行", st5.rows.length === 1 && st5.rows[0].text === "hello world");

    // 首行 ▲ / 末行 ▼ 禁用
    const btnsA = list.children[0].children[3].children;
    check("首行 ▲ 禁用", btnsA[0].disabled === true);
    check("末行 ▼ 禁用", btnsA[1].disabled === true);

    // 右下角角标拖拽调行高：down → move（+50）→ up → state.h 写入 + 节点高度同步
    const gripA = list.children[0].children[2].children[1];
    const rowHBefore = parseFloat(list.children[0].style.height);
    gripA._handlers.pointerdown({ stopPropagation() {}, pointerId: 1, clientY: 100 });
    gripA._handlers.pointermove({ stopPropagation() {}, clientY: 150 });
    check("拖拽中行高+50", parseFloat(list.children[0].style.height) === rowHBefore + 50);
    gripA._handlers.pointerup({ stopPropagation() {}, clientY: 150 });
    const stH = JSON.parse(node.properties.promptStackState);
    check("拖拽结束写 state.h", stH.rows[0].h === rowHBefore + 50);
    // clamp：拖到超出上限 → 上限；节点高度同步（单行 300 → contentHeight 354 > 初始 220）
    gripA._handlers.pointerdown({ stopPropagation() {}, pointerId: 1, clientY: 0 });
    gripA._handlers.pointermove({ stopPropagation() {}, clientY: 10000 });
    gripA._handlers.pointerup({ stopPropagation() {}, clientY: 10000 });
    const stH2 = JSON.parse(node.properties.promptStackState);
    check("拖拽上限 clamp", stH2.rows[0].h === 300);
    check("节点高度同步", node.size[1] === 354);
    // 重渲染后行高保持（h 从 state 恢复）
    FakeType.prototype.onConfigure.call(node);
    check("重渲染行高保持", parseFloat(list.children[0].style.height) === 300);

    // graphToPrompt 注入
    const out = await app.graphToPrompt();
    const inj = JSON.parse(out.output["1"].inputs.PromptStackState);
    check("注入形状", inj.version === 1 && inj.rows.length === 1
        && JSON.stringify(inj.rows[0]) === JSON.stringify({ enabled: true, text: "hello world" }));

    // onConfigure 恢复
    node.properties.promptStackState = JSON.stringify({ version: 1, rows: [
        { id: "x1", enabled: false, label: "", text: "cfg row" },
    ] });
    FakeType.prototype.onConfigure.call(node);
    check("onConfigure 恢复", list.children[0]?.children[2]?.children[0]?.value === "cfg row"
        && list.children[0]?.children[0]?.textContent === "\u2013");

    // onRemoved 清理
    FakeType.prototype.onRemoved.call(node);
    check("onRemoved 清理", node._sfPsRoot === null);

    if (failures.length) {
        console.log(`\n${failures.length} FAILED`);
        process.exit(1);
    }
    console.log("\nALL PASS");
})();
