// SFTrackDataSubtract / SFTrackDataAdd 前端动态槽位测试
// （Node 直接运行：node tests/test_track_data_slots_js.js）
// 覆盖：
// - 初始 4 个槽（后端静态 20 槽 schema，前端裁剪）
// - 动态槽位：全连 → 追加、断开尾部空槽回收（复用 sf_dynamic_slots 真实现）
// - installConfiguredSlotRecovery：加载/粘贴恢复按链接数补齐 + 回收尾部空槽
const fs = require("fs");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const capturedExts = [];
const app = {
    graph: { _nodes: [], links: {} },
    registerExtension: (ext) => capturedExts.push(ext),
};

const load = (file, returnExpr) => {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", file), "utf8")
        .replace(/import[^;]+;/g, "")
        .replace(/export\s+(?=function|const|let|class|var)/g, "");
    return new Function("app", code + "\n" + returnExpr)(app);
};

// 真实现：sf_dynamic_slots 的 installDynamicSlots + installConfiguredSlotRecovery
const slots = load("sf_dynamic_slots.js",
    "return { installDynamicSlots, installConfiguredSlotRecovery }");

// 被测扩展：注入真实依赖
const extCode = fs
    .readFileSync(path.join(__dirname, "..", "web", "sf_track_data_slots.js"), "utf8")
    .replace(/import[^;]+;/g, "");
new Function("app", "installDynamicSlots", "installConfiguredSlotRecovery", extCode)(
    app, slots.installDynamicSlots, slots.installConfiguredSlotRecovery
);

const ext = capturedExts.find((e) => e.name === "sfnodes.TrackDataSlots");
check("扩展已注册", ext !== undefined);

const makeSlot = (name, type) => ({ name, type, localized_name: name, label: null, link: null });

// 模拟后端静态 schema 建出的 20 个槽
function makeNode(comfyClass, prefix) {
    const node = {
        id: 1,
        comfyClass,
        inputs: [],
        size: [220, 100],
        computeSize() { return [this.size[0], this.size[1]]; },
        setSize(sz) { this.size = sz; },
        addInput(name, type) { this.inputs.push(makeSlot(name, type)); },
        removeInput(i) { this.inputs.splice(i, 1); },
    };
    for (let i = 1; i <= 20; i++) node.inputs.push(makeSlot(prefix + i, "MASK,SAM3_TRACK_DATA"));
    return node;
}

let nextLinkId = 1;
// 带回调连线（触发动态槽增删）
function connect(node, index) {
    node.inputs[index].link = nextLinkId++;
    node.onConnectionsChange(1, index, true, { id: node.inputs[index].link }, node.inputs[index]);
}
// 静默连线（模拟 configure 直赋 links，不触发 onConnectionsChange）
function connectSilent(node, index) {
    node.inputs[index].link = nextLinkId++;
}

for (const [cls, prefix] of [["SFTrackDataSubtract", "exclude_"], ["SFTrackDataAdd", "add_"]]) {
    const node = makeNode(cls, prefix);
    ext.nodeCreated(node);

    check(`${prefix} 裁剪到初始 4 槽`, node.inputs.length === 4
        && node.inputs.map((s) => s.name).join(",") === `${prefix}1,${prefix}2,${prefix}3,${prefix}4`);
    check(`${prefix} 槽类型多类型`, node.inputs.every((s) => s.type === "MASK,SAM3_TRACK_DATA"));

    // 全连 4 槽 → 追加第 5
    for (let i = 0; i < 4; i++) connect(node, i);
    check(`${prefix} 全连追加第 5`, node.inputs.length === 5 && node.inputs[4].name === `${prefix}5`);

    // 断开尾部空槽 → 回收
    node.inputs[4].link = null;
    node.onConnectionsChange(1, 4, false, null, node.inputs[4]);
    check(`${prefix} 断开回收尾部空槽`, node.inputs.length === 4
        && node.inputs.every((s) => s.link !== null));
}

// 非目标 class 不处理
const other = makeNode("SFConditioningCombine", "conditioning_");
ext.nodeCreated(other);
check("非目标 class 保持 20 槽", other.inputs.length === 20);

// 加载/粘贴恢复：configure 静默赋 links 不触发 onConnectionsChange
for (const [cls, prefix] of [["SFTrackDataSubtract", "exclude_"], ["SFTrackDataAdd", "add_"]]) {
    const node = makeNode(cls, prefix);
    ext.nodeCreated(node);
    node.inputs.push(makeSlot(`${prefix}5`, "MASK,SAM3_TRACK_DATA")); // configure 恢复的已保存槽
    for (let i = 0; i < 5; i++) connectSilent(node, i); // 5 槽全连，无空闲
    node.onAfterGraphConfigured();
    check(`${prefix} 恢复补齐 linked+1`, node.inputs.length === 6
        && node.inputs[5].name === `${prefix}6`);

    // 尾部多余空槽 → 回收到 linked+1
    node.inputs.push(makeSlot(`${prefix}7`, "MASK,SAM3_TRACK_DATA"));
    node.onAfterGraphConfigured();
    check(`${prefix} 恢复回收多余尾槽`, node.inputs.length === 6);
}

console.log();
if (failures.length) {
    console.log(`FAILED: ${failures.length} -> ${failures}`);
    process.exit(1);
}
console.log("ALL PASS");
