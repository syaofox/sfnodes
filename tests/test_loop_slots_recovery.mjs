// SF 循环节点动态槽「加载恢复」测试（Node 直接运行：node tests/test_loop_slots_recovery.mjs）
// 覆盖 sf_dynamic_slots.js::installConfiguredSlotRecovery 的输入/输出两侧恢复：
//   - 加载既有工作流：按实际已连接槽数补齐到 linked+1，回收多余尾部空槽
//   - While（start=0）/ For（start=1）两套配置
//   - 与 installDynamicSlots 协同（nodeCreated → configure → onAfterGraphConfigured）
import { installConfiguredSlotRecovery, installDynamicSlots } from "../web/sf_dynamic_slots.js";

let failures = 0;
const check = (name, cond) => {
    if (cond) console.log("PASS:", name);
    else { failures += 1; console.log("FAIL:", name); }
};

class FakeNode {
    constructor(inputs = [], outputs = []) {
        this.inputs = inputs.map((s) => ({ ...s }));
        this.outputs = outputs.map((s) => ({ ...s }));
        this.size = [200, 100];
    }
    addInput(name, type) { this.inputs.push({ name, type, link: null }); }
    removeInput(idx) { this.inputs.splice(idx, 1); }
    addOutput(name, type) { this.outputs.push({ name, type, links: null }); }
    removeOutput(idx) { this.outputs.splice(idx, 1); }
    computeSize() { return [200, 100]; }
    setSize(size) { this.size = size; }
}

const names = (slots) => slots.map((s) => s.name).join(",");

const forCfg = {
    inputPrefix: "initial_value", inputStart: 1, inputCount: 19, inputType: "*", initialInputs: 1,
    outputPrefix: "value", outputStart: 1, outputCount: 19, outputType: "*", initialOutputs: 1,
};
const whileCfg = {
    inputPrefix: "initial_value", inputStart: 0, inputCount: 20, inputType: "*", initialInputs: 1,
    outputPrefix: "value", outputStart: 0, outputCount: 20, outputType: "*", initialOutputs: 1,
};

// 1. 两个已连接输入 + 一个空尾 → 保持 3；输出补齐到 3
let node = new FakeNode(
    [
        { name: "flow", type: "FLOW_CONTROL", link: 1 },
        { name: "initial_value1", type: "*", link: 2 },
        { name: "initial_value2", type: "*", link: 3 },
    ],
    [
        { name: "value1", type: "*", links: [4] },
        { name: "value2", type: "*", links: [5] },
    ],
);
installConfiguredSlotRecovery(node, forCfg);
node.onAfterGraphConfigured();
check("输入补齐 initial_value3（2 连接 + 1 空位）", names(node.inputs) === "flow,initial_value1,initial_value2,initial_value3");
check("输出补齐 value3（2 连接 + 1 空位）", names(node.outputs) === "value1,value2,value3");

// 2. 多余尾部空槽回收（加载旧工作流时槽位多于实际连接）
node = new FakeNode(
    [
        { name: "initial_value1", link: 2 },
        { name: "initial_value2", link: 3 },
        { name: "initial_value3", link: null },
        { name: "initial_value4", link: null },
    ],
    [{ name: "value1", links: [4] }],
);
installConfiguredSlotRecovery(node, forCfg);
node.onAfterGraphConfigured();
check("回收多余输入空槽到 3", names(node.inputs) === "initial_value1,initial_value2,initial_value3");
check("输出保持 2（1 连接 + 1 空位）", names(node.outputs) === "value1,value2");

// 3. 全部未连接 → 裁到初始 1 个
node = new FakeNode(
    [{ name: "initial_value1", link: null }, { name: "initial_value2", link: null }],
    [{ name: "value1", links: [] }, { name: "value2", links: null }],
);
installConfiguredSlotRecovery(node, forCfg);
node.onAfterGraphConfigured();
check("全空输入裁到 1", names(node.inputs) === "initial_value1");
check("全空输出裁到 1", names(node.outputs) === "value1");

// 4. While 循环配置（start=0；condition 不参与匹配）
node = new FakeNode(
    [
        { name: "condition", type: "BOOLEAN", link: 1 },
        { name: "initial_value0", type: "*", link: 2 },
        { name: "initial_value1", type: "*", link: 3 },
    ],
    [{ name: "value0", type: "*", links: [4] }],
);
installConfiguredSlotRecovery(node, whileCfg);
node.onAfterGraphConfigured();
check("While 输入补齐 initial_value2", names(node.inputs) === "condition,initial_value0,initial_value1,initial_value2");
check("While 输出补齐 value1", names(node.outputs) === "value0,value1");

// 5. installDynamicSlots 协同：nodeCreated（空槽）→ configure 重建槽位与链接 → 恢复
node = new FakeNode([], []);
installDynamicSlots(node, forCfg);
installConfiguredSlotRecovery(node, forCfg);
node.inputs = [
    { name: "total", type: "INT", link: 1 },
    { name: "initial_value1", type: "*", link: 2 },
    { name: "initial_value2", type: "*", link: 3 },
];
node.outputs = [
    { name: "flow", type: "FLOW_CONTROL", links: [4] },
    { name: "index", type: "INT", links: [5] },
    { name: "value1", type: "*", links: [6] },
];
node.onAfterGraphConfigured();
check("协同后输入 3 + 空位", names(node.inputs) === "total,initial_value1,initial_value2,initial_value3");
check("协同后输出 flow/index/value1 + value2 空位", names(node.outputs) === "flow,index,value1,value2");

// 6. 连接事件仍会动态加槽（installDynamicSlots 行为未被恢复逻辑破坏）
node = new FakeNode([], []);
installDynamicSlots(node, forCfg);
installConfiguredSlotRecovery(node, forCfg);
node.addInput("initial_value1", "*");
node.addOutput("value1", "*");
// LiteGraph 时序：连线建立（slot 已置 link/links）后才触发 onConnectionsChange
node.inputs[0].link = 1;
node.outputs[0].links = [2];
node.onConnectionsChange(1, 0, true, {}, null);
node.onConnectionsChange(2, 0, true, {}, null);
check("连接后输入追加 initial_value2", names(node.inputs) === "initial_value1,initial_value2");
check("连接后输出追加 value2", names(node.outputs) === "value1,value2");

console.log();
if (failures) {
    console.log(`FAILED: ${failures}`);
    process.exit(1);
}
console.log("test_loop_slots_recovery: all assertions passed");
