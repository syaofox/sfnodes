// SF SCAIL-2 前端逻辑测试（Node 直接运行：node tests/test_scail2_js.js）
// 覆盖：
// - 扩展注册名 sfnodes.scail2 + beforeRegisterNodeDef
// - Reference Pack：数量驱动槽位期望列表、动态重建、旧版 subject_N_image_M 迁移
// - Fit Video：resolution=custom 显示 custom_width/height，否则隐藏
// - Simple Video：widget 排序、widgets_values 位置错位修复
// - setWidgetVisible 隐藏/恢复
const fs = require("fs");
const path = require("path");

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

// ---- mocks ----
const capturedExts = [];
const app = {
  graph: { _nodes: [], links: {}, setDirtyCanvas: () => {} },
  registerExtension: (ext) => capturedExts.push(ext),
};

function fakeWidget(name, value, type = "number") {
  return { name, value, type, hidden: false, options: {}, computeSize: () => [100, 20], _state: {} };
}

class FakeNode {
  constructor(widgets = [], inputs = []) {
    this.widgets = widgets;
    this._widgets = widgets;
    this.inputs = inputs;
    this.outputs = [];
    this.size = [300, 200];
    this.flags = {};
    this.graph = app.graph;
    this._widgetSlotsDirty = false;
  }
  addInput(name, type) { this.inputs.push({ name, type, link: null }); }
  removeInput(index) { this.inputs.splice(index, 1); }
  addOutput(name, type) { this.outputs.push({ name, type, links: [] }); }
  removeOutput(index) { this.outputs.splice(index, 1); }
  disconnectInput() {}
  disconnectOutput() {}
  computeSize() { return [300, 200]; }
  setSize() {}
  setDirtyCanvas() {}
}

// ---- 加载被测模块（去 import / export）----
const raw = fs.readFileSync(path.join(__dirname, "..", "web", "sf_scail2.js"), "utf8");
const code = raw
  .replace(/import[^;]+;/g, "")
  .replace(/export\s*\{[^}]*\}\s*;?/g, "")
  .replace(/export\s+(?=function|const|let|class|var)/g, "");
const names = [
  "desiredReferencePackInputNames", "rebuildReferencePackInputs", "migrateLegacyReferencePackInputs",
  "setWidgetVisible", "isWidgetVisible", "updateFitVideoWidgets", "updateSimpleVideoWidgets",
  "reorderSimpleVideoWidgets", "repairSimpleVideoWidgetOrder", "trimReferencePackNodeDataInputs",
  "SCAIL2_LABELS", "EXT_NAME",
];
const exported = new Function("app", code + "\nreturn {" + names.join(",") + "};")(app);

// 1. 扩展注册
check("ext registered", capturedExts.length === 1 && capturedExts[0].name === "sfnodes.scail2");
check("beforeRegisterNodeDef is fn", typeof capturedExts[0].beforeRegisterNodeDef === "function");
check("EXT_NAME const", exported.EXT_NAME === "sfnodes.scail2");

// 2. 期望槽位（数量驱动）
check("desired slots", JSON.stringify(exported.desiredReferencePackInputNames(new FakeNode([
  fakeWidget("subject_count", 2), fakeWidget("reference_count", 1),
]))) === JSON.stringify(["subject_1_image", "subject_2_image", "reference_1", "scene_image"]));

// 3. 动态重建（新增 subject_2 / reference_1）
const rpNode = new FakeNode(
  [fakeWidget("subject_count", 2), fakeWidget("reference_count", 1)],
  [{ name: "subject_1_image", type: "IMAGE", link: null }, { name: "scene_image", type: "IMAGE", link: null }],
);
exported.rebuildReferencePackInputs(rpNode, exported.desiredReferencePackInputNames(rpNode));
check("rebuild adds slots", JSON.stringify(rpNode.inputs.map((i) => i.name)) ===
  JSON.stringify(["subject_1_image", "subject_2_image", "reference_1", "scene_image"]));
// 缩回：数量回到 1/0
rpNode.widgets[0].value = 1;
rpNode.widgets[1].value = 0;
exported.rebuildReferencePackInputs(rpNode, exported.desiredReferencePackInputNames(rpNode));
check("rebuild removes slots", JSON.stringify(rpNode.inputs.map((i) => i.name)) ===
  JSON.stringify(["subject_1_image", "scene_image"]));

// 4. 旧版 subject_N_image_M 迁移
const legacyNode = new FakeNode(
  [fakeWidget("subject_count", 1), fakeWidget("reference_count", 0)],
  [
    { name: "subject_1_image", type: "IMAGE", link: 7 },
    { name: "subject_1_image_2", type: "IMAGE", link: null },
  ],
);
const migrated = exported.migrateLegacyReferencePackInputs(legacyNode);
check("legacy migrated", migrated === true);
check("legacy renamed to reference_1", legacyNode.inputs.some((i) => i.name === "reference_1"));

// 5. Fit Video 显隐
const fitNode = new FakeNode([
  fakeWidget("resolution", "512p", "combo"),
  fakeWidget("custom_width", 832), fakeWidget("custom_height", 480),
]);
exported.updateFitVideoWidgets(fitNode);
check("fit 512p hides width", exported.isWidgetVisible(fitNode.widgets[1]) === false);
check("fit 512p hides height", exported.isWidgetVisible(fitNode.widgets[2]) === false);
fitNode.widgets[0].value = "custom";
exported.updateFitVideoWidgets(fitNode);
check("fit custom shows width", exported.isWidgetVisible(fitNode.widgets[1]) === true);
check("fit custom shows height", exported.isWidgetVisible(fitNode.widgets[2]) === true);

// 6. Simple Video widget 排序（高级项靠后）
const simpleNode = new FakeNode([
  fakeWidget("seed", 1), fakeWidget("advanced", false, "toggle"), fakeWidget("mode", "replacement", "combo"),
  fakeWidget("chunk_frames", 81), fakeWidget("long_video_mode", "chunk", "combo"),
]);
exported.reorderSimpleVideoWidgets(simpleNode);
const order = simpleNode.widgets.map((w) => w.name);
check("reorder advanced after others", order.indexOf("advanced") > order.indexOf("mode"));
check("reorder long_video_mode after advanced", order.indexOf("long_video_mode") > order.indexOf("advanced"));
check("reorder chunk_frames last", order[order.length - 1] === "chunk_frames");

// 6b. Simple Video 高级项显隐（tiled_decode 随 advanced 显示，chunk/context 均可见）
const getW = (node, name) => node.widgets.find((w) => w.name === name);
const advNode = new FakeNode([
  fakeWidget("advanced", false, "toggle"), fakeWidget("long_video_mode", "chunk", "combo"),
  fakeWidget("max_frames", 0), fakeWidget("chunk_frames", 81), fakeWidget("overlap_frames", 5),
  fakeWidget("color_correction", false, "toggle"), fakeWidget("context_frames", 81),
  fakeWidget("context_overlap_frames", 20), fakeWidget("tiled_decode", false, "toggle"),
]);
exported.updateSimpleVideoWidgets(advNode);
check("advanced off hides tiled_decode", exported.isWidgetVisible(getW(advNode, "tiled_decode")) === false);
getW(advNode, "advanced").value = true;
exported.updateSimpleVideoWidgets(advNode);
check("advanced on shows tiled_decode (chunk)", exported.isWidgetVisible(getW(advNode, "tiled_decode")) === true);
check("chunk shows chunk_frames", exported.isWidgetVisible(getW(advNode, "chunk_frames")) === true);
check("chunk hides context_frames", exported.isWidgetVisible(getW(advNode, "context_frames")) === false);
getW(advNode, "long_video_mode").value = "context_sampling";
exported.updateSimpleVideoWidgets(advNode);
check("context shows tiled_decode", exported.isWidgetVisible(getW(advNode, "tiled_decode")) === true);
check("context shows context_frames", exported.isWidgetVisible(getW(advNode, "context_frames")) === true);
check("context hides chunk_frames", exported.isWidgetVisible(getW(advNode, "chunk_frames")) === false);

// 7. widgets_values 位置错位修复（advanced + long_video_mode 顺序颠倒）
const repairNode = new FakeNode([
  fakeWidget("advanced", false, "toggle"), fakeWidget("long_video_mode", "chunk", "combo"),
  fakeWidget("max_frames", 0), fakeWidget("chunk_frames", 81), fakeWidget("overlap_frames", 5),
]);
exported.repairSimpleVideoWidgetOrder(repairNode, {
  widgets_values: [1, "replacement", "context_sampling", true],
});
check("repair long_video_mode", repairNode.widgets[1].value === "context_sampling");
check("repair advanced", repairNode.widgets[0].value === true);

// 8. setWidgetVisible 往返
const w = fakeWidget("x", 1);
check("visible initially", exported.isWidgetVisible(w) === true);
exported.setWidgetVisible(w, false);
check("hidden after false", exported.isWidgetVisible(w) === false && w.type === "hidden");
exported.setWidgetVisible(w, true);
check("restored after true", exported.isWidgetVisible(w) === true && w.type === "number");

// 9. nodeData 裁剪：schema 外可选输入删除，仅留 subject_1_image/scene_image + 计数
const nodeData = {
  name: "SFSCAIL2ReferencePack",
  input: { required: {}, optional: {
    subject_count: ["INT", {}], subject_1_image: ["IMAGE", {}], subject_2_image: ["IMAGE", {}],
    subject_2_image_2: ["IMAGE", {}], reference_1: ["IMAGE", {}], scene_image: ["IMAGE", {}],
  } },
  input_order: { optional: ["subject_count", "subject_1_image", "subject_2_image", "reference_1", "scene_image"] },
};
exported.trimReferencePackNodeDataInputs(nodeData);
check("trim keeps subject_1_image", "subject_1_image" in nodeData.input.optional);
check("trim keeps scene_image", "scene_image" in nodeData.input.optional);
check("trim drops subject_2_image", !("subject_2_image" in nodeData.input.optional));
check("trim keeps count widgets", "subject_count" in nodeData.input.optional);

if (failures.length) {
  console.log("\n" + failures.length + " FAILURES");
  process.exit(1);
}
console.log("\ntest_scail2_js: all assertions passed");
