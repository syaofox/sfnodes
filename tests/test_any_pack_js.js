// SFAnyPack / SFAnyUnpack 前端逻辑测试（Node 直接运行：node tests/test_any_pack_js.js）
// 覆盖：
// - 新增：按连接数据类型自动着色（pack 输入 / unpack 输出）、union 累积/去重、断开恢复 "*"、
//   workflow 恢复（onAfterGraphConfigured）重算、源为 "*" 时不着色
// - 回归：动态槽位自动增删、自动命名、手动改名保护、名称传播（pack → unpack）、prompt 键映射
const fs = require("fs");
const path = require("path");

const code = fs
    .readFileSync(path.join(__dirname, "..", "web", "any_pack.js"), "utf8")
    .replace(/import[^;]+;/g, "");

// ---- 与 web/sf_dynamic_slots.js 一致的工具函数（测试独立，行为保持一致） ----
const isSlotConnected = (slot) => {
    if (!slot) return false;
    if (slot.link !== null && slot.link !== undefined && slot.link !== -1) return true;
    return Array.isArray(slot.links) && slot.links.length > 0;
};
const uniqueName = (slots, selfIndex, base) => {
    if (!Array.isArray(slots) || slots.length === 0) return base;
    let name = base;
    let i = 2;
    while (slots.some((s, idx) => idx !== selfIndex && s && s.name === name)) {
        name = base + "_" + i++;
    }
    return name;
};

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ---- mocks ----
let canvasDirty = 0;
const capturedExts = [];
const app = {
    graph: { _nodes: [], links: {} },
    canvas: { setDirty: () => { canvasDirty++; } },
    registerExtension: (ext) => capturedExts.push(ext),
    graphToPrompt: (result) => result,
};

new Function("app", "isSlotConnected", "uniqueName", code)(app, isSlotConnected, uniqueName);

const ext = capturedExts.find((e) => e.name === "sfnodes.AnyPack");
check("扩展已注册", ext !== undefined);

const makeGraph = () => ({
    _nodes: [],
    links: {},
    getNodeById(id) { return this._nodes.find((n) => n.id === id) || null; },
});

const makeSlot = (name, type, isOut) => ({
    name, type,
    link: null,
    links: isOut ? [] : null,
    sfManualName: false,
});

function makeNode(comfyClass, graph, id) {
    const isPack = comfyClass === "SFAnyPack";
    const node = {
        id,
        comfyClass,
        graph,
        inputs: [],
        outputs: [],
        size: [200, 100],
        computeSize() { return [this.size[0], this.size[1]]; },
        setSize(sz) { this.size = sz; },
        addInput(name, type) { this.inputs.push(makeSlot(name, type, false)); },
        addOutput(name, type) { this.outputs.push(makeSlot(name, type, true)); },
        removeInput(i) { this.inputs.splice(i, 1); },
        removeOutput(i) { this.outputs.splice(i, 1); },
        getExtraMenuOptions() {},
    };
    if (isPack) {
        node.outputs.push(makeSlot("pack", "SF_PACK", true));
        for (let i = 0; i < 20; i++) node.addInput("value" + i, "*");
    } else {
        node.inputs.push(makeSlot("pack", "SF_PACK", false));
        for (let i = 0; i < 20; i++) node.addOutput("out" + i, "*");
    }
    return node;
}

// 数据源节点（pack 输入侧）
const makeSource = (id, outputs) => ({
    id,
    comfyClass: "TestSource",
    graph: null,
    inputs: [],
    outputs: outputs.map(([name, type]) => makeSlot(name, type, true)),
});

// 数据目标节点（unpack 输出侧）
const makeTarget = (id, inputs) => ({
    id,
    comfyClass: "TestTarget",
    graph: null,
    inputs: inputs.map(([name, type]) => makeSlot(name, type, false)),
    outputs: [],
});

// ---- 事件模拟：真实前端在调用 onConnectionsChange 前已更新 slot.link / slot.links ----
let linkSeq = 1;
const makeLink = (type, origin, originSlot, target, targetSlot) => {
    const id = linkSeq++;
    const link = { id, type, origin_id: origin.id, origin_slot: originSlot, target_id: target.id, target_slot: targetSlot };
    return link;
};

function connectInput(graph, target, inputIndex, source, sourceSlot) {
    const link = makeLink(source.outputs[sourceSlot].type, source, sourceSlot, target, inputIndex);
    graph.links[link.id] = link;
    target.inputs[inputIndex].link = link.id;
    const so = source.outputs[sourceSlot];
    so.links = [...(so.links || []), link.id];
    target.onConnectionsChange(1, inputIndex, true, link, target.inputs[inputIndex]);
    return link;
}

function disconnectInput(graph, target, inputIndex) {
    const id = target.inputs[inputIndex].link;
    const link = graph.links[id];
    delete graph.links[id];
    target.inputs[inputIndex].link = null;
    const source = graph.getNodeById(link.origin_id);
    if (source && Array.isArray(source.outputs[link.origin_slot].links)) {
        source.outputs[link.origin_slot].links = source.outputs[link.origin_slot].links.filter((x) => x !== id);
    }
    target.onConnectionsChange(1, inputIndex, false, link, target.inputs[inputIndex]);
}

function connectOutput(graph, node, outputIndex, target, targetSlot) {
    const link = makeLink(target.inputs[targetSlot].type, node, outputIndex, target, targetSlot);
    graph.links[link.id] = link;
    node.outputs[outputIndex].links = [...(node.outputs[outputIndex].links || []), link.id];
    target.inputs[targetSlot].link = link.id;
    node.onConnectionsChange(2, outputIndex, true, link, node.outputs[outputIndex]);
    return link;
}

function disconnectOutput(graph, node, outputIndex, linkId) {
    const link = graph.links[linkId];
    delete graph.links[linkId];
    node.outputs[outputIndex].links = (node.outputs[outputIndex].links || []).filter((x) => x !== linkId);
    const target = graph.getNodeById(link.target_id);
    if (target && target.inputs[link.target_slot] && target.inputs[link.target_slot].link === linkId) {
        target.inputs[link.target_slot].link = null;
    }
    node.onConnectionsChange(2, outputIndex, false, link, node.outputs[outputIndex]);
}

// ===========================================================================
const graph = makeGraph();
const pack = makeNode("SFAnyPack", graph, 5);
ext.nodeCreated(pack);
graph._nodes.push(pack);
check("pack 创建后裁剪到 2 输入", pack.inputs.length === 2);
check("pack 初始槽位均为 *", pack.inputs.every((s) => s.type === "*"));

// 其他节点不受影响
const other = makeNode("SomeOtherNode", graph, 50);
ext.nodeCreated(other);
check("其他节点不安装 hook", other.onConnectionsChange === undefined && other.onAfterGraphConfigured === undefined);

// ---------- 着色：pack 输入 ----------
const imgSrc = makeSource(1, [["image", "IMAGE"]]);
graph._nodes.push(imgSrc);
const slot0Before = pack.inputs[0];
const dirtyBefore = canvasDirty;
connectInput(graph, pack, 0, imgSrc, 0);
check("连接 IMAGE 后输入槽着色为 IMAGE", pack.inputs[0].type === "IMAGE");
check("着色通过替换数组元素触发重渲染", pack.inputs[0] !== slot0Before);
check("替换元素保留 link", pack.inputs[0].link === slot0Before.link);
check("自动命名沿用源输出名（回归）", pack.inputs[0].name === "image");
check("着色触发画布重绘", canvasDirty > dirtyBefore);

// 源为 "*" 时保持 *
const anySrc = makeSource(2, [["any", "*"]]);
graph._nodes.push(anySrc);
const slot1Before = pack.inputs[1];
const dirtyBefore2 = canvasDirty;
connectInput(graph, pack, 1, anySrc, 0);
check("源为 * 时槽位保持 *", pack.inputs[1].type === "*");
check("无类型变化时不替换元素", pack.inputs[1] === slot1Before);
check("无类型变化不触发重绘", canvasDirty === dirtyBefore2);
check("全连接后自动追加 value2（回归）", pack.inputs.length === 3 && pack.inputs[2].name === "value2" && pack.inputs[2].type === "*");

// 断开：回收尾部空槽 + 恢复 *
disconnectInput(graph, pack, 1);
check("断开后尾部空槽回收（回归）", pack.inputs.length === 2);
check("断开后类型恢复 *", pack.inputs[1].type === "*");
disconnectInput(graph, pack, 0);
check("全部断开后输入槽恢复 *", pack.inputs[0].type === "*");

// ---------- 名称传播（回归）：pack → unpack ----------
const unpack = makeNode("SFAnyUnpack", graph, 4);
ext.nodeCreated(unpack);
graph._nodes.push(unpack);
check("unpack 创建后裁剪到 1 输出", unpack.outputs.length === 1);
check("unpack 初始输出槽为 *", unpack.outputs[0].type === "*");

connectInput(graph, pack, 0, imgSrc, 0); // 重新连接 IMAGE（自动命名 image）
const plink = makeLink("SF_PACK", pack, 0, unpack, 0);
graph.links[plink.id] = plink;
pack.outputs[0].links = [plink.id];
unpack.inputs[0].link = plink.id;
unpack.onConnectionsChange(1, 0, true, plink, unpack.inputs[0]);
check("unpack 输出名跟随 pack 输入名（传播）", unpack.outputs[0].name === "image");

// ---------- 着色：unpack 输出（union） ----------
const imgTarget = makeTarget(7, [["image", "IMAGE"]]);
const imgTarget2 = makeTarget(9, [["image2", "IMAGE"]]);
const maskTarget = makeTarget(8, [["mask", "MASK"]]);
graph._nodes.push(imgTarget, imgTarget2, maskTarget);

const lImg = connectOutput(graph, unpack, 0, imgTarget, 0);
check("连接 IMAGE 目标后输出槽着色", unpack.outputs[0].type === "IMAGE");
check("unpack 自动追加 out1（回归）", unpack.outputs.length === 2 && unpack.outputs[1].name === "out1");
const lMask = connectOutput(graph, unpack, 0, maskTarget, 0);
check("第二个类型形成 union", unpack.outputs[0].type === "IMAGE,MASK");
const lImg2 = connectOutput(graph, unpack, 0, imgTarget2, 0);
check("union 去重", unpack.outputs[0].type === "IMAGE,MASK");
check("未连接槽保持 *", unpack.outputs[1].type === "*");

disconnectOutput(graph, unpack, 0, lMask.id);
check("断开一个后 union 收缩", unpack.outputs[0].type === "IMAGE");
disconnectOutput(graph, unpack, 0, lImg2.id);
disconnectOutput(graph, unpack, 0, lImg.id);
check("全部断开后输出槽恢复 *", unpack.outputs[0].type === "*");
check("断开后尾部空槽回收（回归）", unpack.outputs.length === 1);

// ---------- 手动改名保护 + 仍着色 ----------
pack.inputs[0].name = "myimage";
pack.inputs[0].sfManualName = true;
connectInput(graph, pack, 0, imgSrc, 0);
check("手动改名不被自动命名覆盖", pack.inputs[0].name === "myimage");
check("手动改名槽位仍着色", pack.inputs[0].type === "IMAGE");

// ---------- union 源类型展平 ----------
const unionSrc = makeSource(13, [["img_mask", "IMAGE,MASK"]]);
graph._nodes.push(unionSrc);
const pack3 = makeNode("SFAnyPack", graph, 14);
ext.nodeCreated(pack3);
graph._nodes.push(pack3);
connectInput(graph, pack3, 0, unionSrc, 0);
check("union 源类型展平到槽位", pack3.inputs[0].type === "IMAGE,MASK");

// ---------- workflow 恢复（onAfterGraphConfigured） ----------
const pack2 = makeNode("SFAnyPack", graph, 6);
ext.nodeCreated(pack2);
graph._nodes.push(pack2);
pack2.inputs[0].link = 77;
graph.links[77] = { id: 77, type: "LATENT", origin_id: 10, origin_slot: 0, target_id: 6, target_slot: 0 };
pack2.inputs[1].link = 78;
graph.links[78] = { id: 78, type: "CLIP", origin_id: 11, origin_slot: 0, target_id: 6, target_slot: 1 };
pack2.onAfterGraphConfigured.call(pack2);
check("workflow 恢复后 pack 按 links 着色", pack2.inputs[0].type === "LATENT" && pack2.inputs[1].type === "CLIP");

const unpack2 = makeNode("SFAnyUnpack", graph, 16);
ext.nodeCreated(unpack2);
graph._nodes.push(unpack2);
unpack2.outputs[0].links = [79];
graph.links[79] = { id: 79, type: "MODEL", origin_id: 16, origin_slot: 0, target_id: 12, target_slot: 0 };
unpack2.onAfterGraphConfigured.call(unpack2);
check("workflow 恢复后 unpack 按 links 着色", unpack2.outputs[0].type === "MODEL");

// ---------- prompt 键映射（回归） ----------
ext.setup();
check("setup 安装 prompt 映射", typeof app.graphToPrompt === "function" && app.graphToPrompt.sfAnyPackMapped === true);
app.graph = graph;
pack.inputs[0].name = "foo";
const gtp = app.graphToPrompt({ output: { 5: { inputs: { foo: [3, 0] } } } });
check("重命名输入映射回 value0", gtp.output[5].inputs.value0 !== undefined && gtp.output[5].inputs.foo === undefined);

// ===========================================================================
if (failures.length) {
    console.log("\n" + failures.length + " 项失败: " + failures.join(", "));
    process.exit(1);
}
console.log("\n全部通过");
