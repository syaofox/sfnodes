// SFQwenImage21PromptEnhancer 前端逻辑测试（Node 直接运行：node tests/test_qwen21_enhancer_js.js）
// 覆盖：扩展注册名；模式->widget 显隐表；模式->源输入增删（clip/llama_model，已连线保留）；
// 中间槽移除后的 link.target_slot 位移修正；mode widget callback 与 onAfterGraphConfigured 重放；
// 动态图片槽初始裁剪（installDynamicSlots 集成）。
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
  return {
    name, value, type, hidden: false, options: {}, computeSize: () => [100, 20],
    _state: {}, triggerDraw: () => {},
  };
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
    this.comfyClass = "SFQwenImage21PromptEnhancer";
    this._widgetSlotsDirty = false;
  }
  addInput(name, type) { this.inputs.push({ name, type, link: null }); }
  removeInput(index) { this.inputs.splice(index, 1); }
  addOutput(name, type) { this.outputs.push({ name, type, links: [] }); }
  removeOutput(index) { this.outputs.splice(index, 1); }
  disconnectInput() {}
  computeSize() { return [300, 200]; }
  setSize() {}
  setDirtyCanvas() {}
}

// ---- 加载被测模块（去 import / export；依赖库同样剥壳注入）----
function loadStripped(file) {
  const raw = fs.readFileSync(path.join(__dirname, "..", "web", file), "utf8");
  return raw
    .replace(/import[^;]+;/g, "")
    .replace(/export\s*\{[^}]*\}\s*;?/g, "")
    .replace(/export\s+(?=function|const|let|class|var)/g, "");
}

const dynCode = loadStripped("sf_dynamic_slots.js");
const dyn = new Function(
  dynCode + "\nreturn { installDynamicSlots, installConfiguredSlotRecovery, removeInputAt, syncInputLinkTargets };"
)();

const libCode = loadStripped("sf_widget_visibility_lib.js");
const lib = new Function(
  libCode + "\nreturn { setWidgetVisible, isWidgetVisible, refreshWidgetSnapshot };"
)();

const names = [
  "EXT_NAME", "MODE_WIDGETS", "currentMode", "modeWidgetNames", "desiredSourceInputs",
  "applyModeWidgets", "syncSourceInputs", "applyModeVisibility",
];
const exported = new Function(
  "app", "installDynamicSlots", "installConfiguredSlotRecovery", "removeInputAt", "syncInputLinkTargets",
  "setWidgetVisible", "isWidgetVisible", "refreshWidgetSnapshot",
  loadStripped("sf_qwen21_prompt_enhancer.js") + "\nreturn {" + names.join(",") + "};"
)(
  app,
  dyn.installDynamicSlots, dyn.installConfiguredSlotRecovery, dyn.removeInputAt, dyn.syncInputLinkTargets,
  lib.setWidgetVisible, lib.isWidgetVisible, lib.refreshWidgetSnapshot
);

const REQUIRED_WIDGETS = [
  "mode", "task", "prompt", "output_language", "max_tokens", "temperature", "top_k",
  "top_p", "min_p", "repetition_penalty", "seed", "thinking", "vision_megapixels",
  "detail", "unload_after",
];

function makeNode({ mode = "本地官方PE", inputs = [] } = {}) {
  const widgets = REQUIRED_WIDGETS.map((name) => fakeWidget(name, name === "mode" ? mode : 0));
  return new FakeNode(widgets, inputs);
}

function widgetVisible(node, name) {
  return lib.isWidgetVisible(node.widgets.find((w) => w.name === name));
}

function inputNames(node) {
  return node.inputs.map((input) => input.name);
}

// ---- 1. 扩展注册 ----
check("ext registered", capturedExts.length === 1 && capturedExts[0].name === "sfnodes.QwenImage21PromptEnhancer");
check("nodeCreated/loadedGraphNode 均为函数",
  typeof capturedExts[0].nodeCreated === "function" && typeof capturedExts[0].loadedGraphNode === "function");

// ---- 2. 模式 -> widget 表 ----
check("PE/LLM 模式 widget 集合一致（thinking 生效）",
  JSON.stringify(exported.MODE_WIDGETS["本地官方PE"]) === JSON.stringify(exported.MODE_WIDGETS["本地LLM"])
  && exported.MODE_WIDGETS["本地官方PE"].includes("thinking")
  && !exported.MODE_WIDGETS["本地官方PE"].includes("detail"));
check("LLaMA 无 thinking、有 max_tokens",
  !exported.MODE_WIDGETS["本地LLaMA"].includes("thinking")
  && exported.MODE_WIDGETS["本地LLaMA"].includes("max_tokens"));
check("API 仅 temperature/seed/vision/detail",
  JSON.stringify(exported.MODE_WIDGETS["API"]) === JSON.stringify(["temperature", "seed", "vision_megapixels", "detail"]));
check("公共项常显", exported.modeWidgetNames("API").includes("prompt")
  && exported.modeWidgetNames("API").includes("task"));

// ---- 3. 模式 -> 源输入 ----
check("PE/LLM -> clip", JSON.stringify(exported.desiredSourceInputs("本地官方PE")) === '["clip"]'
  && JSON.stringify(exported.desiredSourceInputs("本地LLM")) === '["clip"]');
check("LLaMA -> llama_model", JSON.stringify(exported.desiredSourceInputs("本地LLaMA")) === '["llama_model"]');
check("API -> 无源输入", exported.desiredSourceInputs("API").length === 0);

// ---- 4. applyModeWidgets 显隐 ----
let node = makeNode({ mode: "API" });
exported.applyModeWidgets(node, "API");
check("API 隐藏本地专属 widget", !widgetVisible(node, "max_tokens") && !widgetVisible(node, "top_k")
  && !widgetVisible(node, "top_p") && !widgetVisible(node, "min_p") && !widgetVisible(node, "repetition_penalty")
  && !widgetVisible(node, "thinking") && !widgetVisible(node, "unload_after"));
check("API 显示 detail/temperature/seed/vision", widgetVisible(node, "detail") && widgetVisible(node, "temperature")
  && widgetVisible(node, "seed") && widgetVisible(node, "vision_megapixels"));
exported.applyModeWidgets(node, "本地LLaMA");
check("LLaMA 显示 max_tokens、隐藏 thinking/detail", widgetVisible(node, "max_tokens")
  && !widgetVisible(node, "thinking") && !widgetVisible(node, "detail"));
exported.applyModeWidgets(node, "本地官方PE");
check("PE 显示 thinking、隐藏 detail", widgetVisible(node, "thinking") && !widgetVisible(node, "detail"));

// ---- 5. syncSourceInputs 增删 ----
node = makeNode({ inputs: [{ name: "clip", type: "CLIP", link: null }, { name: "image_1", type: "IMAGE", link: null },
  { name: "system_prompt", type: "STRING", link: null }, { name: "llama_model", type: "LLAMACPPMODEL", link: null }] });
exported.syncSourceInputs(node, "本地官方PE", app.graph);
check("PE 移除未连线 llama_model、保留 clip",
  JSON.stringify(inputNames(node)) === '["clip","image_1","system_prompt"]');
exported.syncSourceInputs(node, "本地LLaMA", app.graph);
check("LLaMA 移除 clip、补回 llama_model",
  inputNames(node).includes("llama_model") && !inputNames(node).includes("clip"));

node = makeNode({ inputs: [{ name: "clip", type: "CLIP", link: 5 }, { name: "llama_model", type: "LLAMACPPMODEL", link: null }] });
exported.syncSourceInputs(node, "API", app.graph);
check("API 不静默断已连线源输入（clip 保留）", inputNames(node).includes("clip"));

// ---- 6. link.target_slot 位移修正（移除中间未连线槽）----
const links = { 7: { target_slot: 1 } };
const graph2 = { links, setDirtyCanvas: () => {} };
node = makeNode({ inputs: [{ name: "clip", type: "CLIP", link: null }, { name: "image_1", type: "IMAGE", link: 7 }] });
exported.syncSourceInputs(node, "本地LLaMA", graph2);
check("移除未连线 clip 后 image_1 索引前移", inputNames(node)[0] === "image_1");
check("已连线输入 target_slot 同步", links[7].target_slot === 0);

// ---- 7. 扩展 setup：mode callback + 配置完成重放 + 图片槽裁剪 ----
node = makeNode({ inputs: [{ name: "clip", type: "CLIP", link: null }, { name: "llama_model", type: "LLAMACPPMODEL", link: null }] });
node.inputs.push({ name: "image_1", type: "IMAGE", link: null }, { name: "image_2", type: "IMAGE", link: null });
capturedExts[0].nodeCreated(node);
check("setup 裁剪图片槽到初始 1 个", inputNames(node).filter((n) => n.startsWith("image_")).length === 1);
check("setup 后 PE 模式移除 llama_model", !inputNames(node).includes("llama_model"));

const modeWidget = node.widgets.find((w) => w.name === "mode");
modeWidget.value = "API";
modeWidget.callback();
check("mode callback 切换后 clip 被移除", !inputNames(node).includes("clip"));
check("mode callback 切换后本地 widget 隐藏", !widgetVisible(node, "max_tokens") && widgetVisible(node, "detail"));

modeWidget.value = "本地LLaMA";
node.onAfterGraphConfigured();
check("onAfterGraphConfigured 重放（补回 llama_model）", inputNames(node).includes("llama_model")
  && !inputNames(node).includes("clip") && widgetVisible(node, "max_tokens") && !widgetVisible(node, "thinking"));

console.log();
if (failures.length) {
  console.log(`${failures.length} FAILED: ${failures.join(", ")}`);
  process.exit(1);
}
console.log("test_qwen21_enhancer_js: all assertions passed");
