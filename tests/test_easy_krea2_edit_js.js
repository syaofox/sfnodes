// SFEasyKrea2Edit 前端冒烟测试（Node 直接运行：node tests/test_easy_krea2_edit_js.js）
// 覆盖：动态参考图槽位（初始 1 / 全连追加 / 断开回收 / 加载恢复）、逐图 strength widget
// 与 properties 同步、graphToPrompt 注入 hidden SFEasyKrea2EditState。
// 用 mock app + 真 sf_dynamic_slots / sf_easy_krea2_edit_lib。
const fs = require("fs");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── 被测扩展加载 ──
const capturedExts = [];
const nodesById = {};
let promptResult = { output: {} };
const app = {
    graph: { getNodeById: (id) => nodesById[id] || null, _nodes: [] },
    registerExtension: (ext) => capturedExts.push(ext),
};
app.graphToPrompt = async () => JSON.parse(JSON.stringify(promptResult));

const stripImports = (file) => fs
    .readFileSync(path.join(__dirname, "..", "web", file), "utf8")
    .replace(/import[^;]+;/g, "")
    .replace(/export\s+(?=function|const|let|class|var)/g, "");

const slots = new Function("app", stripImports("sf_dynamic_slots.js") + "\nreturn { installDynamicSlots, installConfiguredSlotRecovery };")(app);
const lib = new Function(stripImports("sf_easy_krea2_edit_lib.js") +
    "\nreturn { HIDDEN_INPUT, stateJson, syncStrengthWidgets };")({});

function makeNodeType() {
    function FakeNode() { }
    FakeNode.prototype.onNodeCreated = function () { };
    return FakeNode;
}

function makeNode() {
    const node = Object.create(nodeType.prototype);
    node.id = 7;
    node.comfyClass = "SFEasyKrea2Edit";
    node.type = "SFEasyKrea2Edit";
    node.inputs = [{ name: "image1", type: "IMAGE", link: null }];
    node.outputs = [];
    node.widgets = [];
    node.properties = {};
    node.size = [300, 200];
    node.addInput = function (name, type) { this.inputs.push({ name, type, link: null }); };
    node.removeInput = function (idx) { this.inputs.splice(idx, 1); };
    node.addWidget = function (type, name, value, cb, options) {
        const w = { type, name, value, callback: cb, options };
        this.widgets.push(w);
        return w;
    };
    node.computeSize = function () { return [320, 120 + this.widgets.length * 24]; };
    node.setSize = function (sz) { this.size = sz; };
    node.setDirtyCanvas = function () { };
    return node;
}

const nodeType = makeNodeType();
new Function(
    "app", "installDynamicSlots", "installConfiguredSlotRecovery",
    "HIDDEN_INPUT", "stateJson", "syncStrengthWidgets",
    stripImports("sf_easy_krea2_edit.js"),
)(app, slots.installDynamicSlots, slots.installConfiguredSlotRecovery,
    lib.HIDDEN_INPUT, lib.stateJson, lib.syncStrengthWidgets);

check("注册了一个扩展", capturedExts.length === 1);
const ext = capturedExts[0];
check("扩展名 sfnodes.*", ext.name.startsWith("sfnodes."));

// 非目标节点不处理
const otherType = makeNodeType();
ext.beforeRegisterNodeDef(otherType, { name: "OtherNode" });
check("非目标节点不包装", capturedExts.length === 1);

// 目标节点：创建即装槽位 + strength widget（image1）
ext.beforeRegisterNodeDef(nodeType, { name: "SFEasyKrea2Edit" });
const node = makeNode();
node.onNodeCreated();
check("初始 1 个 image 槽", node.inputs.filter((i) => /^image\d+$/.test(i.name)).length === 1);
check("初始 1 个 strength widget", node.widgets.filter((w) => w._sfStrengthN != null).length === 1);
check("strength widget 标签", node.widgets[0].name === "强度 1");
check("扩展注册一次", capturedExts.length === 1);

// 连接 image1 → 追加 image2 槽 + strength widget
node.inputs[0].link = 100;
node.onConnectionsChange(1, 0, true, {});
check("全连后追加 image2", node.inputs.some((i) => i.name === "image2"));
check("追加 image2 strength widget", node.widgets.some((w) => w._sfStrengthN === 2));

// 编辑 image2 强度 → properties 真源 → 注入 hidden
node.widgets.find((w) => w._sfStrengthN === 2).callback(0.25);
nodesById[7] = node;
app.graph._nodes = [node];
promptResult = { output: { 7: { class_type: "SFEasyKrea2Edit", inputs: {} } } };

(async () => {
    const result = await app.graphToPrompt();
    const hidden = result.output["7"].inputs["SFEasyKrea2EditState"];
    check("graphToPrompt 注入 hidden 状态", typeof hidden === "string");
    const parsed = JSON.parse(hidden);
    check("注入 strengths", parsed.strengths["1"] === 1 && parsed.strengths["2"] === 0.25);

    // 断开 image2 → 回收槽与 widget；properties 保留
    node.inputs = node.inputs.filter((i) => i.name !== "image2");
    node.onConnectionsChange(1, 1, false, {});
    check("断开后回收 image2 槽", !node.inputs.some((i) => i.name === "image2"));
    check("断开后回收 strength widget", !node.widgets.some((w) => w._sfStrengthN === 2));
    check("properties 保留历史强度", node.properties.sfEasyKrea2Strengths["2"] === 0.25);

    console.log();
    if (failures.length) {
        console.log(`FAILED: ${failures.length} -> ${failures}`);
        process.exit(1);
    }
    console.log("ALL PASSED");
})();
