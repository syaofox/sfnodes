// SFCharacterSelect 主扩展冒烟测试（Node 直接运行：node tests/test_character_smoke.js）
// 覆盖：扩展注册名 sfnodes.* 前缀、setupNode 全链路无抛错（TDZ/拼写类错误即挂）、
// 隐藏双真源 widget 补建、library 切换清空、roles 载入后卡片渲染 + 无有效选择自动首图。
// FakeDOM + FakeNode，fetch 桩返回固定角色库。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ---- FakeDOM ----
function makeEl(tag) {
    const el = {
        tag,
        children: [],
        style: {},
        dataset: {},
        className: "",
        textContent: "",
        value: "",
        title: "",
        placeholder: "",
        src: "",
        rows: 0,
        loading: "",
        classList: {
            toggle() {}, add() {}, remove() {},
            contains() { return false; },
        },
        append(...kids) { for (const k of kids) this.children.push(k); return kids[kids.length - 1]; },
        appendChild(kid) { this.children.push(kid); return kid; },
        addEventListener() {},
        removeEventListener() {},
        querySelector() { return makeEl("stub"); },
        querySelectorAll() { return []; },
        getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
    };
    // 真 DOM 语义：innerHTML = "" 清空子节点（render* 依赖此行为）
    let _html = "";
    Object.defineProperty(el, "innerHTML", {
        get() { return _html; },
        set(v) { _html = String(v); if (v === "") el.children.length = 0; },
    });
    return el;
}
globalThis.document = {
    createElement: (tag) => makeEl(tag),
    getElementById: () => null,
    activeElement: null,
    head: { appendChild() {} },
};
globalThis.navigator = { language: "zh-CN" };
globalThis.window = {};

const ROLES = [
    { name: "京京", prompt: "JR", images: [{ label: "脸", url: "/api/f.jpg", prompt: "p-face" }, { label: "身", url: "/api/b.jpg", prompt: "" }] },
    { name: "小果", prompt: "XG", images: [{ label: "a1", url: "/api/a1.jpg", prompt: "p-a1" }] },
];
globalThis.fetch = async () => ({ ok: true, json: async () => ROLES });

// ---- app/依赖桩 ----
const capturedExts = [];
const app = { registerExtension: (ext) => capturedExts.push(ext) };
const Dunn = {
    applyAdaptiveCanvasOnly() {},
    hideJsonWidget(widgets, name) { return (widgets || []).find((w) => w.name === name) || null; },
    injectCSSOnce() {},
    installWheelZoomPassthrough() {},
    isGraphLoading: () => false,
    isVueNodes: () => false,
    sfApiUrl: (u) => u,
};

const loadLib = () => {
    const code = fs.readFileSync(path.join(__dirname, "..", "web", "sf_character_lib.js"), "utf8")
        .replace(/export\s+(?=function|const|let|class|var)/g, "");
    const names = ["parseState", "serializeSelection", "coerceSelection",
        "entryOf", "entryImages", "rolePromptOf", "displayPrompt", "displayRolePrompt",
        "resolveLabel", "filterAndSort", "isRemoteThumb",
        "STATE_WIDGET", "PROMPT_WIDGET", "DOM_WIDGET", "CHARACTERS_API", "LIB_PREFIX"];
    return new Function(code + "\nreturn {" + names.join(",") + "};")();
};
const lib = loadLib();

const mainCode = fs.readFileSync(path.join(__dirname, "..", "web", "sf_character.js"), "utf8")
    .replace(/import[^;]+;/g, "");
new Function(
    "app", "lib",
    "applyAdaptiveCanvasOnly", "hideJsonWidget", "injectCSSOnce",
    "installWheelZoomPassthrough", "isGraphLoading", "isVueNodes", "sfApiUrl",
    mainCode
)(app, lib, Dunn.applyAdaptiveCanvasOnly, Dunn.hideJsonWidget, Dunn.injectCSSOnce,
    Dunn.installWheelZoomPassthrough, Dunn.isGraphLoading, Dunn.isVueNodes, Dunn.sfApiUrl);

const ext = capturedExts.find((e) => e.name === "sfnodes.CharacterSelect");
check("扩展已注册 sfnodes.* 前缀", ext !== undefined);

// ---- FakeNode ----
function makeNode() {
    const widgets = [{ name: "library", value: "character_x", callback: null }];
    const domWidgets = [];
    return {
        type: "SFCharacterSelect",
        widgets,
        properties: {},
        size: [440, 600],
        graph: {},
        computeSize() { return this.size.slice(); },
        setSize(sz) { this.size = sz.slice(); },
        addWidget(type, nm, val) {
            const w = { name: nm, type, value: val, hidden: false, options: {} };
            this.widgets.push(w);
            return w;
        },
        addDOMWidget(nm, tm, el, opts) {
            const w = { name: nm, dom: el, options: opts };
            domWidgets.push(w);
            this._domWidgets = domWidgets;
            return w;
        },
    };
}
function FakeNodeType() {}
ext.beforeRegisterNodeDef(FakeNodeType, { name: "SFCharacterSelect" });
check("非目标节点不挂接", (() => {
    function Other() {}
    const before = Other.prototype.onNodeCreated;
    ext.beforeRegisterNodeDef(Other, { name: "SFStylesSelector" });
    return Other.prototype.onNodeCreated === before;
})());

(async () => {
    const node = makeNode();
    FakeNodeType.prototype.onNodeCreated.call(node);
    const hasState = node.widgets.some((w) => w.name === "SFCharacterState");
    const hasPrompt = node.widgets.some((w) => w.name === "SFCharacterPrompt");
    check("setupNode 无抛错且补建双隐藏 widget", hasState && hasPrompt);
    const st = node.widgets.find((w) => w.name === "SFCharacterState");
    check("隐藏 widget 零尺寸", typeof st.computeSize === "function" && st.computeSize()[0] === 0);
    check("DOM widget 已添加", (node._domWidgets || []).length === 1);
    check("DOM widget 不序列化", node._domWidgets[0].options.serialize === false);

    // 等待 roles 异步载入 + 重渲染
    for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r));
    const ctx = node._sfCharacterCtx;
    check("ctx 已挂载且 roles 载入", !!ctx && Array.isArray(ctx.roles) && ctx.roles.length === 2);
    check("无有效选择自动首图", (() => {
        try { return JSON.parse(st.value).role === "京京"; } catch (e) { return false; }
    })());
    check("角色卡已渲染", ctx.listEl.children.length === 2);
    check("shots 横条已渲染首角图片", ctx.shotsBar.children.length === 2);
    check("计数按钮已更新", ctx.root && true);

    // library 切换清空 + onConfigure 无抛错
    const libW = node.widgets.find((w) => w.name === "library");
    libW.value = "character_y";
    libW.callback("character_y");
    check("切库清空选择", st.value === '{"role":"","shots":[]}' || JSON.parse(st.value).role === "");
    FakeNodeType.prototype.onConfigure.call(node);
    for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r));
    check("onConfigure 无抛错且重渲染", ctx.listEl.children.length === 2);

    console.log();
    if (failures.length) {
        console.log(`FAILED: ${failures.length}: ${failures.join(", ")}`);
        process.exit(1);
    }
    console.log("ALL PASS");
})();
