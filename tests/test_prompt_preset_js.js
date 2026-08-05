// SFPromptPreset 前端逻辑测试（Node 直接运行：node tests/test_prompt_preset_js.js）
// 覆盖：pose/couple 互斥联动、悬浮卡片 widget 命中检测
const fs = require("fs");
const path = require("path");

const code = fs
    .readFileSync(path.join(__dirname, "..", "web", "prompt_preset.js"), "utf8")
    .replace('import { app } from "/scripts/app.js";', "")
    .replace('import { api } from "/scripts/api.js";', "");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const capturedExts = [];
const app = { registerExtension: (ext) => capturedExts.push(ext) };
const api = { fetchApi: async () => ({ ok: false }) };
new Function("app", "api", code)(app);

const captured = capturedExts.find((ext) => typeof ext.beforeRegisterNodeDef === "function");
check("存在节点扩展", captured !== undefined);

// --- 互斥联动 ---
const nodeType = { prototype: {} };
captured.beforeRegisterNodeDef(nodeType, { name: "SFPromptPreset" });
const otherType = { prototype: {} };
captured.beforeRegisterNodeDef(otherType, { name: "OtherNode" });
check("其他节点不受影响", otherType.prototype.onNodeCreated === undefined);

function fakeWidget(name, initial = "禁用") {
    return { name, value: initial, callback: null };
}
function makeNode() {
    const n = { widgets: [fakeWidget("pose_preset"), fakeWidget("couple_preset")] };
    nodeType.prototype.onNodeCreated.call(n);
    return n;
}

const n1 = makeNode();
const pose1 = n1.widgets.find(w => w.name === "pose_preset");
const couple1 = n1.widgets.find(w => w.name === "couple_preset");
check("pose 与 couple 均有 callback", typeof pose1.callback === "function" && typeof couple1.callback === "function");
pose1.value = "回眸";
pose1.callback();
check("选 pose 后 couple 置禁用", couple1.value === "禁用");
couple1.value = "公主抱";
couple1.callback();
check("选 couple 后 pose 置禁用", pose1.value === "禁用");

const n3 = makeNode();
const pose3 = n3.widgets.find(w => w.name === "pose_preset");
const couple3 = n3.widgets.find(w => w.name === "couple_preset");
pose3.value = "随机";
pose3.callback();
check("选随机后 couple 置禁用", couple3.value === "禁用");

const n4 = makeNode();
const pose4 = n4.widgets.find(w => w.name === "pose_preset");
const couple4 = n4.widgets.find(w => w.name === "couple_preset");
couple4.value = "拥抱";
couple4.callback();
pose4.value = "禁用";
pose4.callback();
check("pose 置禁用不动 couple", couple4.value === "拥抱");

// --- 悬浮卡片命中检测（widgetAt）---
// widgetAt 是扩展文件内局部函数，测试代码需与其同作用域执行
globalThis.__sf_failures = failures;
const cardTestCode = code + `
const check2 = (n, c) => {
    if (c) console.log("PASS:", n);
    else { globalThis.__sf_failures.push(n); console.log("FAIL:", n); }
};
const fakeCanvas = (nodes, mx, my) => ({ graph_mouse: [mx, my], graph: { _nodes: nodes } });
const fakeComboWidget = (name, x, y, w, h) => ({
    name, type: "combo", hidden: false, pos: [x, y], size: [w, h], value: "回眸",
});
const fakeNode = (wx, wy, widgets) => ({ pos: [100, 200], widgets, _dummy: wx + wy });

// 节点 (100,200) + widget pos (0,30) size (150,20) → 画布坐标 (100,250)-(250,270)
const nodes = [fakeNode(0, 0, [
    fakeComboWidget("outfit_preset", 0, 8, 150, 20),
    fakeComboWidget("pose_preset", 0, 30, 150, 20),
    fakeComboWidget("environment_preset", 0, 52, 150, 20),
])];
const hitO = widgetAt(fakeCanvas(nodes, 150, 218), 150, 218);
check2("命中 outfit_preset", hitO !== null && hitO.widget.name === "outfit_preset" && hitO.category === "Outfit");
let hit = widgetAt(fakeCanvas(nodes, 150, 240), 150, 240);
check2("命中 pose_preset", hit !== null && hit.widget.name === "pose_preset" && hit.category === "Pose");
hit = widgetAt(fakeCanvas(nodes, 150, 250), 150, 250);
check2("边界命中", hit !== null && hit.widget.name === "pose_preset");
hit = widgetAt(fakeCanvas(nodes, 200, 262), 200, 262);
check2("命中 environment_preset", hit !== null && hit.widget.name === "environment_preset" && hit.category === "Environment");
hit = widgetAt(fakeCanvas(nodes, 50, 240), 50, 240);
check2("widget 左侧外不命中", hit === null);
hit = widgetAt(fakeCanvas(nodes, 150, 400), 150, 400);
check2("节点下方不命中", hit === null);

const nodesHidden = [fakeNode(0, 0, [Object.assign(fakeComboWidget("pose_preset", 0, 30, 150, 20), { hidden: true })])];
hit = widgetAt(fakeCanvas(nodesHidden, 150, 240), 150, 240);
check2("隐藏 widget 不命中", hit === null);

const nodesOther = [fakeNode(0, 0, [
    fakeComboWidget("input_text", 0, 30, 150, 20),
    Object.assign(fakeComboWidget("seed", 0, 52, 150, 20), { type: "number" }),
])];
hit = widgetAt(fakeCanvas(nodesOther, 150, 240), 150, 240);
check2("非预设 widget 不命中", hit === null);

check2("无 graph_mouse 不崩", widgetAt({}, 0, 0) === null);
`;
new Function("app", "api", cardTestCode)(app, api);

console.log("\nFAILURES:", failures.length);
process.exit(failures.length ? 1 : 0);
