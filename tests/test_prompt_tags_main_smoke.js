// SFPromptTags 主扩展端到端冒烟测试（Node 直接运行：node tests/test_prompt_tags_main_smoke.js）
// 用 mock DOM/app/api 真实加载全部模块，验证：
//   - 扩展注册 / beforeRegisterNodeDef 原型安装 / nodeCreated setupNode
//   - graphToPrompt 包装后：@tag 展开 + *wildcard/#list 游标选择 + PromptState 注入
//   - queuePrompt 包装后：commitPicks 仅消耗被入队 build 的选择
//   - 右键菜单 / 编辑器导入链路的模块级完整性（模块加载即验证 import 链）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（同 editor smoke 的惰性元素）──
function makeEl() {
    const style = { setProperty() {}, getPropertyValue() { return ""; } };
    return {
        style, dataset: {}, children: [],
        className: "", textContent: "", innerHTML: "", value: "", placeholder: "",
        type: "", title: "", rows: 1, spellcheck: false, disabled: false, checked: false,
        draggable: false, isConnected: true, offsetWidth: 100, offsetHeight: 20,
        selectionStart: 0, selectionEnd: 0,
        classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        replaceWith() {}, replaceChildren(...kids) { this.children = kids; },
        remove() { this.removed = true; },
        contains() { return false; }, closest() { return null; },
        querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        addEventListener() {}, removeEventListener() {},
        focus() {}, blur() {}, select() {}, click() {},
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {}, setPointerCapture() {}, releasePointerCapture() {}, setSelectionRange() {},
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
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
globalThis.getComputedStyle = () => ({ paddingLeft: "0px", paddingRight: "0px", paddingTop: "0px", paddingBottom: "0px" });

// ── app / api mock ──
const settingsStore = {};
let promptObj = { output: {} };   // 闭包引用：包装后的 graphToPrompt 每次读取最新值
globalThis.app = {
    graph: { _nodes: [], links: {} },
    canvas: { setDirty() {} },
    ui: {
        settings: {
            getSettingValue: (k) => (k in settingsStore ? settingsStore[k] : null),
            setSettingValueAsync: (k, v) => { settingsStore[k] = v; return Promise.resolve(); },
            setSettingValue: (k, v) => { settingsStore[k] = v; },
        },
    },
    extensionManager: { toast: { add() {} } },
    constructor: function FakeApp() {},
    registerExtension(ext) { this._ext = ext; },
    graphToPrompt: async () => promptObj,
};
globalThis.window.app = globalThis.app;
let queuedCalls = 0;
globalThis.api = {
    queuePrompt: async () => { queuedCalls++; return Promise.resolve(); },
};

// ── 加载全部 6 个模块（/scripts/app.js、/scripts/api.js -> globalThis；相对 import 改 .mjs）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ptg_main_"));
for (const n of ["sf_prompt_tags_lib.js", "sf_prompt_tags_pinyin.js",
    "sf_prompt_tags_cursors.js",
    "sf_prompt_tags_store.js", "sf_prompt_tags_guard.js",
    "sf_prompt_tags_editor.js", "sf_prompt_tags.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    const Main = await import(path.join(tmpDir, "sf_prompt_tags.mjs"));
    const Store = await import(path.join(tmpDir, "sf_prompt_tags_store.mjs"));
    check("主模块加载（import 链完整）", true);
    check("扩展已注册", !!app._ext && app._ext.name === "sfnodes.PromptTags");
    check("graphToPrompt 已包装", app._sfPromptTagsPatched === true);
    check("queuePrompt 已包装", app._sfPromptTagsQueuePatched === true && queuedCalls === 0);

    const ext = app._ext;

    // ── beforeRegisterNodeDef：原型安装 ──
    const proto = {};
    ext.beforeRegisterNodeDef({ name: "SFPromptTags", prototype: proto }, { name: "SFPromptTags" });
    check("onConfigure/onConnectionsChange/onRemoved 已安装",
        typeof proto.onConfigure === "function" &&
        typeof proto.onConnectionsChange === "function" &&
        typeof proto.onRemoved === "function");

    // ── nodeCreated：setupNode（DOM widget 构建）──
    const node = {
        id: 1, size: [300, 160], inputs: [], widgets: [],
        properties: { promptState: { text: "", order: "mine", sep: ", ", showExpanded: true } },
        addDOMWidget() { return { onResize() {} }; },
        setDirtyCanvas() {},
        type: "SFPromptTags", comfyClass: "SFPromptTags",
    };
    app.graph._nodes = [node];
    ext.nodeCreated(node);
    check("setupNode 完成（DOM widget 已建）", !!node._sfPromptTagsRoot && !!node._sfPromptTagsRoot._els);
    check("尺寸下限已应用", node.size[0] >= 440 && node.size[1] >= 172);

    // ── 库数据 + 输入框内容 ──
    settingsStore["sfnodes.PromptTags.Library"] = JSON.stringify({
        categories: ["Styles", "Animals"],
        listCats: ["Animals"],
        catModes: {},
        tags: [
            { name: "oil", cat: "Styles", text: "oil painting, thick strokes" },
            { name: "animal", cat: "Animals", kind: "list", text: "red fox\nsnow leopard" },
        ],
    });
    node.properties.promptState.text = "a portrait, @oil, #animal, *Styles";

    // ── graphToPrompt 端到端：展开 + 注入 ──
    promptObj = { output: { "1": { class_type: "SFPromptTags", inputs: {} } } };
    const result = await app.graphToPrompt();
    const entry = result.output["1"];
    const state = JSON.parse(entry.inputs.PromptState);
    check("@tag 已展开", state.text.includes("oil painting, thick strokes"));
    check("#list 已掷一行", /red fox|snow leopard/.test(state.text));
    check("*Styles 已掷一标签", state.text.includes("oil painting, thick strokes, ") || state.text.startsWith("a portrait, oil painting"));
    check("order/sep 注入", state.order === "mine" && state.sep === ", ");
    check("lastRun 已记录", node._sfPromptTagsLastRun && node._sfPromptTagsLastRun.src === node.properties.promptState.text);

    // ── 中文标签端到端：中文库 + 中文 token 展开 ──
    settingsStore["sfnodes.PromptTags.Library"] = JSON.stringify({
        categories: ["风格"],
        tags: [{ name: "油画", cat: "风格", text: "油画, 厚涂笔触" }],
    });
    Store.reloadLibrary();   // 直改 settingsStore 需显式刷新 store 缓存（真实场景改库走 store API）
    node.properties.promptState.text = "画@油画";
    promptObj = { output: { "1": { class_type: "SFPromptTags", inputs: {} } } };
    const r2 = await app.graphToPrompt();
    const s2 = JSON.parse(r2.output["1"].inputs.PromptState);
    check("中文 @tag 端到端展开", s2.text === "画油画, 厚涂笔触");

    // ── queuePrompt：入队后 commitPicks（order 游标推进）──
    settingsStore["sfnodes.PromptTags.Cursors"] = JSON.stringify({});
    node.properties.promptState.text = "#animal";
    await app.graphToPrompt();                     // 掷出（order 模式默认 shuffle，位置写入需入队）
    await globalThis.api.queuePrompt({ output: promptObj.output });
    check("queuePrompt 原逻辑被调用", queuedCalls === 1);
    // shuffle 的 state 在 _pending，入队后应已写入 Cursors 设置
    check("入队后游标已提交", "sfnodes.PromptTags.Cursors" in settingsStore);

    // ── 第二次 queuePrompt：幂等（commitPicks 空 _pending 不写）──
    const before = settingsStore["sfnodes.PromptTags.Cursors"];
    await globalThis.api.queuePrompt({ output: promptObj.output });
    check("空 _pending 时入队不重复写", settingsStore["sfnodes.PromptTags.Cursors"] === before);

    // ── onRemoved 清理不抛错 ──
    proto.onRemoved.call(node);
    check("onRemoved 清理不抛错", node._sfPromptTagsRoot === null);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
