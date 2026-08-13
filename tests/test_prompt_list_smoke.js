// SFPromptList 前端行号编辑器冒烟测试（Node 直接运行：node tests/test_prompt_list_smoke.js）
// 用 mock DOM/app 真实加载模块，验证：
//   - 模块加载 / 扩展注册
//   - beforeRegisterNodeDef 包装 prototype（onNodeCreated / onConfigure /
//     onResize / onRemoved）
//   - nodeCreated setupNode：multiline_text widget 隐藏（值真源保留）、
//     DOM widget 构建、初始值同步、默认尺寸
//   - 输入回写 widget.value + 行号渲染（从 0 起）+ 行数计数
//   - 外部写 widget 值（callback / _sfPlSync）→ DOM 同步
//   - 虚拟化分支（> 500 行）防抖渲染不炸
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素）──
function makeEl() {
    return {
        style: {}, dataset: {}, children: [], _handlers: {},
        className: "", textContent: "", innerHTML: "", value: "", placeholder: "",
        type: "", title: "", spellcheck: true, id: "",
        clientHeight: 200, scrollTop: 0,
        classList: {
            _s: new Set(),
            add(...c) { c.forEach((x) => this._s.add(x)); },
            remove(...c) { c.forEach((x) => this._s.delete(x)); },
            toggle(c, force) { if (force === undefined) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); } else { force ? this._s.add(c) : this._s.delete(c); } },
            contains(c) { return this._s.has(c); },
        },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        replaceChildren(...kids) { this.children = kids; },
        remove() { this.removed = true; },
        addEventListener(name, fn) { this._handlers[name] = fn; },
        removeEventListener() {},
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
    createDocumentFragment() { return makeEl(); },
    getElementById() { return null; },
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    LiteGraph: { vueNodesMode: false },
};
globalThis.requestAnimationFrame = (fn) => fn();

// ── app mock ──
globalThis.app = {
    graph: { _nodes: [], links: {}, getNodeById() { return null; }, setDirtyCanvas() {} },
    registerExtension(ext) { this._ext = ext; },
    loadGraphData: async () => {},
};

function makeNode() {
    const textWidget = {
        name: "multiline_text", value: "body_text", hidden: false,
        options: {}, computeSize: () => [0, 0], element: null, callback: null,
    };
    const skipWidget = {
        name: "skip_empty", value: true, hidden: false,
        options: {}, computeSize: () => [0, 0], element: null, callback: null,
    };
    const wrapWidget = {
        name: "wrap_text", value: false, hidden: false,
        options: {}, computeSize: () => [0, 0], element: null, callback: null,
    };
    return {
        id: "1", comfyClass: "SFPromptList", type: "SFPromptList",
        widgets: [textWidget, skipWidget, wrapWidget], inputs: [], outputs: [], properties: {},
        size: [400, 300],
        graph: { setDirtyCanvas() {} },
        setDirtyCanvas() {},
        addDOMWidget(name, type, el, opts) {
            const w = { name, type, options: opts || {}, element: el, value: null };
            this.widgets.push(w);
            return w;
        },
    };
}

// ── 加载模块（替换 /scripts/* import，相对 import 改 .mjs 同 tmp）──
(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pl_smoke_"));
    for (const n of ["sf_common.js", "sf_prompt_list.js"]) {
        const code = fs.readFileSync(path.join(__dirname, "..", "web", n), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
            .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
            .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
        fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
    }
    await import(path.join(tmpDir, "sf_prompt_list.mjs"));

    check("扩展已注册", app._ext?.name === "sfnodes.PromptList");

    // beforeRegisterNodeDef 包装
    const FakeType = function () {};
    app._ext.beforeRegisterNodeDef(FakeType, { name: "SFPromptList" });
    check("onNodeCreated 已包装", typeof FakeType.prototype.onNodeCreated === "function");
    check("onConfigure 已包装", typeof FakeType.prototype.onConfigure === "function");
    check("onResize 已包装", typeof FakeType.prototype.onResize === "function");
    check("onRemoved 已包装", typeof FakeType.prototype.onRemoved === "function");

    // nodeCreated → setupNode
    const node = makeNode();
    const textWidget = node.widgets[0];
    const skipWidget = node.widgets[1];
    const wrapWidget = node.widgets[2];
    FakeType.prototype.onNodeCreated.call(node);
    const root = node._sfPromptListRoot;
    check("DOM widget 已添加", node.widgets.some((w) => w.name === "sf_prompt_list_editor"));
    check("multiline_text 已隐藏", textWidget.hidden === true);
    check("multiline_text 零高度", textWidget.computeSize()[0] === 0);
    check("值真源保留", textWidget.value === "body_text");
    check("初始值已同步 DOM", root.children[1].children[1].value === "body_text");
    check("默认尺寸", node.size[0] === 420 && node.size[1] === 320);
    check("callback 已包装", typeof textWidget.callback === "function");
    check("skip_empty callback 已包装", typeof skipWidget.callback === "function");
    check("wrap_text callback 已包装", typeof wrapWidget.callback === "function");

    // 初始行号：1 行（"body_text"）→ 行号 0
    const ta = root.children[1].children[1];
    const gutter = root.children[1].children[0];
    const count = root.children[0].children[1];
    check("初始 1 行", gutter.children[0]?.children?.length === 1);
    check("初始行号 0", gutter.children[0]?.children?.[0]?.textContent === "0");
    check("初始计数", count.textContent === "1/1 line");

    // wrap 默认关闭 → textarea wrap="off"（水平滚动不换行）
    check("wrap 默认关闭", ta.wrap === "off");

    // 打开 wrap → textarea wrap="soft"（自动换行）+ 行号重渲染
    wrapWidget.value = true;
    wrapWidget.callback();
    check("wrap 打开后 ta.wrap=soft", ta.wrap === "soft");
    check("wrap 打开后行号仍正确", gutter.children[0]?.children?.map((s) => s.textContent).join(",") === "0");
    wrapWidget.value = false;
    wrapWidget.callback();
    check("wrap 关闭后 ta.wrap=off", ta.wrap === "off");

    // 输入 → 回写 widget.value + 行号从 0 起
    ta.value = "row0\nrow1\nrow2";
    ta._handlers.input();
    check("输入回写值真源", textWidget.value === "row0\nrow1\nrow2");
    check("3 行渲染", gutter.children[0]?.children?.length === 3);
    check("行号从 0 起", gutter.children[0]?.children?.map((s) => s.textContent).join(",") === "0,1,2");
    check("计数更新", count.textContent === "3/3 lines");

    // 空文本 = 1 个逻辑空行、0 个有效行（占位符）
    ta.value = "";
    ta._handlers.input();
    check("空文本 0/1", count.textContent === "0/1 line" && gutter.children[0]?.children?.length === 1);
    check("空文本占位符", gutter.children[0]?.children?.[0]?.textContent === "\u00B7");

    // skip_empty 开启：空白行跳过不占号，行号对齐输出 index
    ta.value = "row0\n\nrow2";
    ta._handlers.input();
    check("空行跳过 2/3", count.textContent === "2/3 lines");
    check("空行跳过行号", gutter.children[0]?.children?.map((s) => s.textContent).join(",") === "0,\u00B7,1");
    check("空行占位符样式", gutter.children[0]?.children?.[1]?.classList.contains("sf-pl-gap") === true);

    // 全空格行（trim 后为空）同样跳过
    ta.value = "a\n   \nb";
    ta._handlers.input();
    check("全空格行跳过", gutter.children[0]?.children?.map((s) => s.textContent).join(",") === "0,\u00B7,1");

    // 开头空行：第一个有效行 index 从 0 开始
    ta.value = "\nfirst";
    ta._handlers.input();
    check("开头空行", gutter.children[0]?.children?.map((s) => s.textContent).join(",") === "\u00B7,0");

    // 关闭 skip_empty → 恢复逻辑行编号
    skipWidget.value = false;
    skipWidget.callback();
    check("关闭过滤行号", gutter.children[0]?.children?.map((s) => s.textContent).join(",") === "0,1");
    check("关闭过滤计数", count.textContent === "2/2 lines");
    skipWidget.value = true;
    skipWidget.callback();
    check("重新开启恢复跳号", gutter.children[0]?.children?.map((s) => s.textContent).join(",") === "\u00B7,0");

    // 外部写 widget 值 → callback → DOM 同步
    textWidget.value = "x\ny";
    textWidget.callback("x\ny");
    check("callback 同步 DOM", ta.value === "x\ny");
    check("callback 同步行号", gutter.children[0]?.children?.map((s) => s.textContent).join(",") === "0,1");

    // _sfPlSync 直接同步（configure 路径）
    textWidget.value = "only";
    root._sfPlSync();
    check("_sfPlSync 同步", ta.value === "only" && count.textContent === "1/1 line");

    // onConfigure 兜底同步
    textWidget.value = "cfg";
    FakeType.prototype.onConfigure.call(node);
    check("onConfigure 同步", ta.value === "cfg");

    // onResize 自愈（Classic：isVueNodes() false）
    const size = [300, 100];
    FakeType.prototype.onResize.call(node, size);
    check("onResize 抬升最小尺寸", size[0] >= 340 && size[1] >= 182);

    // 虚拟化分支（> 500 行，防抖 80ms）：滚动顶部时渲染窗口行号
    ta.value = Array.from({ length: 601 }, (_, i) => "line" + i).join("\n");
    ta._handlers.input();
    await new Promise((r) => setTimeout(r, 120));
    check("虚拟化渲染不炸", true);
    check("虚拟化计数", count.textContent === "601/601 lines");
    const vRows = gutter.children[0]?.children || [];
    check("虚拟化窗口行号", vRows.length >= 10 && vRows[0].textContent === "0" && vRows[vRows.length - 1].textContent === String(vRows.length - 1));
    check("虚拟化底部占位", parseFloat(gutter.style.paddingBottom) > 0);

    // 虚拟化滚动到中部 → 窗口跟随（防抖后重渲染）
    ta.scrollTop = 3000;
    ta._handlers.scroll();
    await new Promise((r) => setTimeout(r, 120));
    const midRows = gutter.children[0]?.children || [];
    check("虚拟化滚动跟随", midRows.length >= 10 && midRows[0].textContent !== "0");

    // 虚拟化含空行：601 行中间（第 300 位置）插空行 → 窗口内 · 占位 + index 连续
    const rows601 = Array.from({ length: 600 }, (_, i) => "line" + i);
    rows601.splice(300, 0, "");
    ta.value = rows601.join("\n");
    ta.scrollTop = 300 * (12 * 1.4);
    ta._handlers.scroll();
    await new Promise((r) => setTimeout(r, 120));
    const gapRows = gutter.children[0]?.children || [];
    check("虚拟化含空行占位", gapRows[0]?.textContent === "\u00B7");
    check("虚拟化含空行 index 连续", gapRows[1]?.textContent === "300" && gapRows[gapRows.length - 1]?.textContent === String(300 + gapRows.length - 2));

    // 行数缩回 ≤500 → 恢复全渲染（padding 占位清空）
    ta.value = "a\nb";
    ta._handlers.input();
    check("缩回全渲染", gutter.style.paddingBottom === "" && gutter.children[0]?.children?.length === 2);

    // onRemoved 清理
    FakeType.prototype.onRemoved.call(node);
    check("onRemoved 清理", node._sfPromptListRoot === null);

    if (failures.length) {
        console.log(`\n${failures.length} FAILED`);
        process.exit(1);
    }
    console.log("\nALL PASS");
})();
