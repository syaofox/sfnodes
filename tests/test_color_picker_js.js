// sf_color_picker.js COLOR 值归一冒烟测试（Node 直接运行：node tests/test_color_picker_js.js）
// 背景：新版 Vue 前端内置 ColorWidget 接管 COLOR 类型（core 覆盖同名自定义注册），
// 其 value 必须是 hex 字符串；旧工作流遗留数组值显示为 "0,0,0" 且无色块。
// 验证：
//   - nodeCreated / loadedGraphNode / configure 包装：数组值 → hex（SFImageResizePlus / SFMaskFill）
//   - 非 COLOR 类节点不受影响；hex 字符串原样保留
//   - getCustomWidgets 工厂接受 hex 字符串默认值（旧前端自定义 widget 路径兼容）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM/app（仅注册期需要）──
globalThis.document = {
    createElement() { return { style: {}, addEventListener() {}, click() {}, remove() {} }; },
    body: { appendChild() {} },
};
globalThis.app = {
    registerExtension(ext) { this._ext = ext; },
};

function stageModule() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sf_color_picker_"));
    const file = path.join(tmp, "sf_color_picker.js");
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", "sf_color_picker.js"), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;");
    fs.writeFileSync(file, code);
    return require(file);
}

function makeNode(comfyClass, widgetType, widgetValue) {
    return {
        comfyClass,
        widgets: [{ name: "pad_color", type: widgetType, value: widgetValue }],
    };
}

const mod = stageModule();
const ext = globalThis.app._ext;

// ── 1. nodeCreated：目标节点数组值归一 ──
{
    const n = makeNode("SFImageResizePlus", "color", [0, 0, 0]);
    ext.nodeCreated(n);
    check("nodeCreated 数组→hex (SFImageResizePlus)", n.widgets[0].value === "#000000");

    const n2 = makeNode("SFMaskFill", "COLOR", [255, 255, 255]);
    ext.nodeCreated(n2);
    check("nodeCreated 数组→hex (SFMaskFill, 大写 type)", n2.widgets[0].value === "#ffffff");
}

// ── 2. hex 字符串与非目标节点不动 ──
{
    const n = makeNode("SFImageResizePlus", "color", "#808080");
    ext.nodeCreated(n);
    check("hex 字符串保留", n.widgets[0].value === "#808080");

    const n2 = makeNode("SFOther", "color", [0, 0, 0]);
    ext.nodeCreated(n2);
    check("非目标类节点不动", Array.isArray(n2.widgets[0].value));
}

// ── 3. configure 包装：恢复值后再归一 ──
{
    const n = makeNode("SFImageResizePlus", "color", "#ffffff");
    n.configure = function () { this.widgets[0].value = [10, 20, 30]; };
    ext.nodeCreated(n);
    n.configure({});
    check("configure 后数组→hex", n.widgets[0].value === "#0a141e");
}

// ── 4. loadedGraphNode：旧工作流遗留数组归一 ──
{
    const n = makeNode("SFMaskFill", "color", [1, 2, 3]);
    ext.loadedGraphNode(n);
    check("loadedGraphNode 数组→hex", n.widgets[0].value === "#010203");
}

// ── 5. getCustomWidgets 工厂：hex 字符串默认值（旧前端自定义 widget 路径）──
{
    const widgets = ext.getCustomWidgets();
    const fakeNode = { addCustomWidget: (w) => w };
    const { widget } = widgets.COLOR(fakeNode, "pad_color", ["COLOR", { default: "#808080" }]);
    check("工厂 hex 默认值→#808080", widget.value === "#808080");
    check("工厂 type=COLOR", widget.type === "COLOR");
    const { widget: w2 } = widgets.COLOR(fakeNode, "pad_color", ["COLOR", { default: [1, 2, 3] }]);
    check("工厂数组默认值仍兼容", w2.value === "#010203");
}

if (failures.length) {
    console.log(`\n${failures.length} FAILED`);
    process.exit(1);
}
console.log("\nall passed");
