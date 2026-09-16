// SFTrackDataMerge 前端冒烟测试（Node 直接运行：node tests/test_track_data_merge_js.js）
// 覆盖：动态槽安装（初始 4 / 全连追加 / 断开回收）、DOM 行模式渲染 + 点击切换
// 写入 node.properties、graphToPrompt 钩子注入 hidden SlotModes。
// 纯逻辑断言见 test_track_data_merge_lib.mjs；此处用 mock DOM + 真 sf_dynamic_slots。
const fs = require("fs");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM ──
function makeEl(tag, cls, text) {
    const e = {
        tagName: tag, className: cls || "", textContent: text != null ? text : "", type: "", title: "",
        children: [], _listeners: {},
        appendChild(c) { this.children.push(c); return c; },
        addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); },
        click() { (this._listeners.click || []).forEach((fn) => fn({ preventDefault() {}, stopPropagation() {} })); },
        remove() {},
    };
    Object.defineProperty(e, "innerHTML", { set() { this.children = []; }, get() { return ""; } });
    return e;
}

// ── 被测扩展加载 ──
const capturedExts = [];
const nodesById = {};
let promptResult = { output: {} };
const app = {
    graph: { getNodeById: (id) => nodesById[id] || null },
    registerExtension: (ext) => capturedExts.push(ext),
};
// 基线 graphToPrompt 必须在模块加载前就位（模块会包装它）
app.graphToPrompt = async () => JSON.parse(JSON.stringify(promptResult));

const stripImports = (file) => fs
    .readFileSync(path.join(__dirname, "..", "web", file), "utf8")
    .replace(/import[^;]+;/g, "")
    .replace(/export\s+(?=function|const|let|class|var)/g, "");

const slots = new Function("app", stripImports("sf_dynamic_slots.js") + "\nreturn { installDynamicSlots, installConfiguredSlotRecovery };")(app);
const L = new Function(stripImports("sf_track_data_merge_lib.js") + "\nreturn { MODE_ADD, MODE_SUB, SLOT_PREFIX, collectSlotNames, getMode, parseModes, serializeModes, setMode, toggleMode };")({});

new Function(
    "app", "el", "injectCSSOnce", "applyAdaptiveCanvasOnly",
    "installDynamicSlots", "installConfiguredSlotRecovery",
    "MODE_ADD", "SLOT_PREFIX", "collectSlotNames", "getMode", "parseModes",
    "serializeModes", "setMode", "toggleMode",
    stripImports("sf_track_data_merge.js")
)(app, makeEl, () => {}, (w) => w,
    slots.installDynamicSlots, slots.installConfiguredSlotRecovery,
    L.MODE_ADD, L.SLOT_PREFIX, L.collectSlotNames, L.getMode, L.parseModes,
    L.serializeModes, L.setMode, L.toggleMode);

const ext = capturedExts.find((e) => e.name === "sfnodes.TrackDataMerge");
check("扩展已注册", ext !== undefined);

const makeSlot = (name, type) => ({ name, type, localized_name: name, label: null, link: null });

function makeNode() {
    const node = {
        id: 1,
        comfyClass: "SFTrackDataMerge",
        inputs: [],
        properties: {},
        size: [260, 100],
        computeSize() { return [this.size[0], this.size[1]]; },
        setSize(sz) { this.size = sz; },
        addInput(name, type) { this.inputs.push(makeSlot(name, type)); },
        removeInput(i) { this.inputs.splice(i, 1); },
        addDOMWidget(type, name, root) { this._dom = { type, name, root }; return { options: {} }; },
    };
    // 真实 schema：required track_data 在前，optional track_1..20 在后
    node.inputs.push(makeSlot("track_data", "SAM3_TRACK_DATA"));
    for (let i = 1; i <= 20; i++) node.inputs.push(makeSlot("track_" + i, "MASK,SAM3_TRACK_DATA"));
    nodesById[1] = node;
    return node;
}

const isDyn = (s) => /^track_\d+$/.test(s.name);
const dynNames = (n) => n.inputs.filter(isDyn).map((s) => s.name);
const dynIdx = (n) => n.inputs.map((s, i) => [s, i]).filter(([s]) => isDyn(s)).map(([, i]) => i);

const node = makeNode();
ext.nodeCreated(node);

check("裁剪到初始 4 动态槽", dynNames(node).join(",") === "track_1,track_2,track_3,track_4");
check("基础槽 track_data 保留且不在末位被回收", node.inputs[0].name === "track_data" && node.inputs.length === 5);
check("DOM widget 已挂载", node._dom && node._dom.name === "sf_track_data_merge_ui");

const list = node._sfTdmList;
check("渲染 4 行（不含基础槽）", list.children.length === 4);
check("默认模式为 sub（按钮 -）", list.children.every((row) => {
    const btn = row.children[1];
    return btn.textContent === "-" && btn.className.includes("sub");
}));

// 点击第 2 行切换为 add
list.children[1].children[1].click();
check("点击切换写入 properties", node.properties.sfTrackMergeModes
    && node.properties.sfTrackMergeModes.track_2 === "add"
    && node.properties.sfTrackMergeModes.track_1 === undefined);
const row2 = node._sfTdmList.children[1];
check("重渲染后按钮为 +", row2.children[1].textContent === "+"
    && row2.children[1].className.includes("add"));
// 再点回 sub
node._sfTdmList.children[1].children[1].click();
check("再点回 sub", node.properties.sfTrackMergeModes.track_2 === "sub");

// 全连 4 动态槽 → 追加第 5，并渲染第 5 行
let nextLinkId = 1;
for (const i of dynIdx(node)) {
    node.inputs[i].link = nextLinkId++;
    node.onConnectionsChange(1, i, true, { id: node.inputs[i].link }, node.inputs[i]);
}
check("全连追加第 5 槽", dynNames(node).join(",") === "track_1,track_2,track_3,track_4,track_5");
check("渲染 5 行", node._sfTdmList.children.length === 5);

// 断开尾部空槽 → 回收并重渲染
const tail = node.inputs.length - 1;
node.inputs[tail].link = null;
node.onConnectionsChange(1, tail, false, null, node.inputs[tail]);
check("断开回收第 5 槽", dynNames(node).join(",") === "track_1,track_2,track_3,track_4");
check("渲染回 4 行", node._sfTdmList.children.length === 4);

// 加载恢复：基础槽 track_data 已连不算动态链接数（inputMatch 生效）
const node2 = makeNode();
ext.nodeCreated(node2);
node2.inputs[0].link = 999; // track_data 已连
// configure 恢复出第 5 个动态槽（重载时才存在）
node2.inputs.push(makeSlot("track_5", "MASK,SAM3_TRACK_DATA"));
for (const i of dynIdx(node2)) node2.inputs[i].link = nextLinkId++;
node2.onAfterGraphConfigured();
check("恢复补齐到 linked+1 且基础槽不干扰", dynNames(node2).join(",")
    === "track_1,track_2,track_3,track_4,track_5,track_6");

// graphToPrompt 注入 SlotModes（track_2 显式 add）
nodesById[1] = node;
node.properties.sfTrackMergeModes = { track_2: "add" };
(async () => {
    promptResult = { output: { 1: { class_type: "SFTrackDataMerge", inputs: { track_data: ["400", 0] } } } };
    const res = await app.graphToPrompt();
    const injected = res.output["1"].inputs.SlotModes;
    check("注入 SlotModes", injected === '{"track_2":"add"}');
    check("保留原 inputs", JSON.stringify(res.output["1"].inputs.track_data) === '["400",0]');

    // 非目标 class 不注入
    promptResult = { output: { 1: { class_type: "OtherNode", inputs: {} } } };
    const res2 = await app.graphToPrompt();
    check("非目标 class 不注入", res2.output["1"].inputs.SlotModes === undefined);

    console.log();
    if (failures.length) {
        console.log(`FAILED: ${failures.length} -> ${failures}`);
        process.exit(1);
    }
    console.log("ALL PASS");
})();
