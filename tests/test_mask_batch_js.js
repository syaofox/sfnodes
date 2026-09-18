// SFMaskBatch 前端动态槽位测试（Node 直接运行：node tests/test_mask_batch_js.js）
// 覆盖：
// - 初始 2 个槽（后端静态 16 槽 schema，前端裁剪）
// - 动态槽位：全连 → 追加、断开尾部空槽回收（复用 sf_dynamic_slots 真实现）
// - 非目标 class 不处理
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

// 真实现：sf_dynamic_slots 的 installDynamicSlots
const slots = load("sf_dynamic_slots.js", "return { installDynamicSlots }");

// 被测扩展：注入真实依赖
const extCode = fs
    .readFileSync(path.join(__dirname, "..", "web", "mask_batch.js"), "utf8")
    .replace(/import[^;]+;/g, "");
new Function("app", "installDynamicSlots", extCode)(app, slots.installDynamicSlots);

const ext = capturedExts.find((e) => e.name === "sfnodes.MaskBatch");
check("扩展已注册", ext !== undefined);

const makeSlot = (name, type) => ({ name, type, localized_name: name, label: null, link: null });

// 模拟后端静态 schema 建出的 16 个槽
function makeNode(comfyClass, prefix = "mask_") {
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
    for (let i = 1; i <= 16; i++) node.inputs.push(makeSlot(prefix + i, "MASK"));
    return node;
}

const node = makeNode("SFMaskBatch");
ext.nodeCreated(node);

check("裁剪到初始 2 槽", node.inputs.length === 2
    && node.inputs.map((s) => s.name).join(",") === "mask_1,mask_2");
check("槽类型 MASK", node.inputs.every((s) => s.type === "MASK"));

// 全连 2 槽 → 追加第 3
let nextLinkId = 1;
const connect = (index) => {
    node.inputs[index].link = nextLinkId++;
    node.onConnectionsChange(1, index, true, { id: node.inputs[index].link }, node.inputs[index]);
};
connect(0);
connect(1);
check("全连追加第 3 槽", node.inputs.length === 3 && node.inputs[2].name === "mask_3");

// 断开尾部空槽 → 回收
node.inputs[2].link = null;
node.onConnectionsChange(1, 2, false, null, node.inputs[2]);
check("断开回收尾部空槽", node.inputs.length === 2
    && node.inputs.every((s) => s.link !== null));

// 非目标 class 不处理
const other = makeNode("SFImageBatch", "image_");
ext.nodeCreated(other);
check("非目标 class 保持 16 槽", other.inputs.length === 16);

console.log();
if (failures.length) {
    console.log(`FAILED: ${failures.length} -> ${failures}`);
    process.exit(1);
}
console.log("ALL PASS");
