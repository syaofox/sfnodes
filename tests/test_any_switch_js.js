// SFAnySwitch 前端逻辑测试（Node 直接运行：node tests/test_any_switch_js.js）
// 覆盖：
// - nodeCreated：初始 4 个 any_XX 输入槽（后端灵活 schema 无具体输入）
// - 动态槽位：全连 → 追加 any_05、断开尾部空槽回收（复用 installDynamicSlots 真实现）
// - 类型着色（复用 any_pack 真 slotLinkTypes/unionType/setSlotType）：输入槽跟随自身连线、
//   输出跟随第一个非 "*" 输入并同步 label、断开回退 "*"/"value"
// - workflow 恢复：onAfterGraphConfigured 重算
// - rgthree 简化差异锁定：经 reroute（link.type="*"）保持 "*" 不着色
const fs = require("fs");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ---- mocks（与 test_any_pack_js.js 同款） ----
let canvasDirty = 0;
const capturedExts = [];
const app = {
    graph: { _nodes: [], links: {} },
    canvas: { setDirty: () => { canvasDirty++; } },
    registerExtension: (ext) => capturedExts.push(ext),
};

const load = (file, returnExpr) => {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", file), "utf8")
        .replace(/import[^;]+;/g, "")
        .replace(/export\s+(?=function|const|let|class|var)/g, "");
    return new Function("app", code + "\n" + returnExpr)(app);
};

// 真实现：sf_dynamic_slots 的 installDynamicSlots + any_pack 的槽位改型三件套
const { installDynamicSlots } = load("sf_dynamic_slots.js", "return { installDynamicSlots }");
const packFns = load("any_pack.js", "return { setSlotType, slotLinkTypes, unionType }");

// 被测模块：注入真实依赖
const switchCode = fs
    .readFileSync(path.join(__dirname, "..", "web", "sf_any_switch.js"), "utf8")
    .replace(/import[^;]+;/g, "")
    .replace(/export\s+(?=function|const|let|class|var)/g, "");
new Function(
    "app", "installDynamicSlots", "setSlotType", "slotLinkTypes", "unionType",
    switchCode
)(app, installDynamicSlots, packFns.setSlotType, packFns.slotLinkTypes, packFns.unionType);

const ext = capturedExts.find((e) => e.name === "sfnodes.AnySwitch");
check("扩展已注册", ext !== undefined);

// ---- FakeNode ----
const makeSlot = (name, type, isOut) => ({
    name, type,
    localized_name: name, // 模拟 addInputSocket：初始槽带显示名
    label: null,
    link: null,
    links: isOut ? [] : null,
});
const makeGraph = () => ({
    _nodes: [],
    links: {},
    getNodeById(id) { return this._nodes.find((n) => n.id === id) || null; },
});
function makeNode(graph) {
    const node = {
        id: 1,
        comfyClass: "SFAnySwitch",
        graph,
        inputs: [],
        outputs: [],
        size: [200, 100],
        computeSize() { return [this.size[0], this.size[1]]; },
        setSize(sz) { this.size = sz; },
        addInput(name, type) { this.inputs.push(makeSlot(name, type, false)); },
        addOutput(name, type) { this.outputs.push(makeSlot(name, type, true)); },
        removeInput(i) { this.inputs.splice(i, 1); },
    };
    node.outputs.push(makeSlot("value", "*", true));
    return node;
}

let nextLinkId = 1;
function connect(node, inputIndex, type) {
    const linkId = nextLinkId++;
    node.inputs[inputIndex].link = linkId;
    node.inputs[inputIndex].type = type; // LiteGraph 连线时按 commonType 写入槽类型
    graph.links[linkId] = { id: linkId, type, origin_id: 9, target_id: node.id };
    node.onConnectionsChange(1, inputIndex, true, { id: linkId }, node.inputs[inputIndex]);
}
function disconnect(node, inputIndex) {
    const linkId = node.inputs[inputIndex].link;
    delete graph.links[linkId];
    node.inputs[inputIndex].link = null;
    node.inputs[inputIndex].type = "*";
    node.onConnectionsChange(1, inputIndex, false, null, node.inputs[inputIndex]);
}

const graph = makeGraph();
const node = makeNode(graph);
ext.nodeCreated(node);

check("初始 4 个输入槽 any_01..any_04", node.inputs.length === 4
    && node.inputs.map((s) => s.name).join(",") === "any_01,any_02,any_03,any_04");
check("初始槽与输出均为 * 且 label 未着类型", node.inputs.every((s) => s.type === "*")
    && node.outputs[0].type === "*" && node.outputs[0].label !== "IMAGE"
    && node.outputs[0].label !== "MASK");

// 连入 IMAGE：该输入与输出着色，输出 label = 类型名
connect(node, 0, "IMAGE");
check("输入槽着色 IMAGE", node.inputs[0].type === "IMAGE");
check("输出跟随第一个非 * 输入", node.outputs[0].type === "IMAGE" && node.outputs[0].label === "IMAGE");

// 4 槽全连 → 追加 any_05
connect(node, 1, "IMAGE");
connect(node, 2, "MASK");
connect(node, 3, "MASK");
check("全连追加 any_05", node.inputs.length === 5 && node.inputs[4].name === "any_05");
check("混合类型输出取第一个输入类型", node.outputs[0].type === "IMAGE");

// 第二输入改为不同类型：各输入独立着色
disconnect(node, 1);
connect(node, 1, "MASK");
check("输入槽独立着色", node.inputs[0].type === "IMAGE" && node.inputs[1].type === "MASK");

// 断开尾部空槽回收（any_05 空槽被移除，回到 4 槽）
disconnect(node, 1);
check("断开回收尾部空槽", node.inputs.length === 4 && node.inputs[3].name === "any_04");

// 全部断开：输入回 *，输出回 * 且 label 复位 value
disconnect(node, 0);
disconnect(node, 2);
disconnect(node, 3);
check("全断开输入回 *", node.inputs.every((s) => s.type === "*"));
check("全断开输出回 * 且 label 复位", node.outputs[0].type === "*" && node.outputs[0].label === "value");

// 经 reroute（link.type = "*"）：不着色
connect(node, 0, "*");
check("经 reroute 保持 * 不着色", node.inputs[0].type === "*" && node.outputs[0].type === "*");
disconnect(node, 0);

// workflow 恢复：configure 不触发 onConnectionsChange，onAfterGraphConfigured 重算
const restoredLinkId = nextLinkId++;
node.inputs[0].link = restoredLinkId;
node.inputs[0].type = "LATENT";
graph.links[restoredLinkId] = { id: restoredLinkId, type: "LATENT", origin_id: 9, target_id: 1 };
node.onAfterGraphConfigured();
check("configure 恢复重算输入着色", node.inputs[0].type === "LATENT");
check("configure 恢复重算输出着色", node.outputs[0].type === "LATENT" && node.outputs[0].label === "LATENT");

if (failures.length) {
    console.log(`\n${failures}`);
    process.exit(1);
}
console.log("\nALL PASS");
