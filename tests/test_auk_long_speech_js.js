// SFAuKLongSpeech mode 联动显隐测试（Node 直接运行：node tests/test_auk_long_speech_js.js）
// 覆盖：扩展注册名；模式->widget 显隐（参考音色隐藏 voice_description / 声音描述隐藏
// reference_seconds）；input_audio 插槽增删（未连线移除、已连线保留）；切回恢复；
// 隐藏 widget 值不丢；mode callback 与 onAfterGraphConfigured 双路重放；显隐变化后节点自适应尺寸。
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
        this.inputs = inputs;
        this.outputs = [];
        this.size = [300, 300];
        this.flags = {};
        this.graph = app.graph;
        this.comfyClass = "SFAuKLongSpeech";
        this._widgetSlotsDirty = false;
        this.sizeCalls = 0;
    }
    addInput(name, type) { this.inputs.push({ name, type, link: null }); }
    removeInput(index) { this.inputs.splice(index, 1); }
    addOutput(name, type) { this.outputs.push({ name, type, links: [] }); }
    removeOutput(index) { this.outputs.splice(index, 1); }
    disconnectInput() {}
    computeSize() { return [300, 300]; }
    setSize() { this.sizeCalls += 1; }
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
const dyn = new Function(dynCode + "\nreturn { removeInputAt, syncInputLinkTargets };")();

const libCode = loadStripped("sf_widget_visibility_lib.js");
const lib = new Function(
    libCode + "\nreturn { setWidgetVisible, isWidgetVisible, refreshWidgetSnapshot, fitNodeToContent };"
)();

const names = ["currentMode", "desiredSourceInputs", "applyModeWidgets", "syncSourceInputs", "applyModeVisibility"];
const mod = new Function(
    "app", "removeInputAt", "syncInputLinkTargets",
    "setWidgetVisible", "isWidgetVisible", "refreshWidgetSnapshot", "fitNodeToContent",
    loadStripped("sf_auk_long_speech.js") + "\nreturn {" + names.join(",") + "};"
)(
    app, dyn.removeInputAt, dyn.syncInputLinkTargets,
    lib.setWidgetVisible, lib.isWidgetVisible, lib.refreshWidgetSnapshot, lib.fitNodeToContent
);

// ---- 节点工厂 ----
function makeNode(modeValue = "参考音色 TTS", audioConnected = false) {
    const widgets = [
        fakeWidget("engine", "engine"),
        fakeWidget("text", "", "string"),
        fakeWidget("mode", modeValue, "combo"),
        fakeWidget("reference_seconds", 10),
        fakeWidget("voice_description", "温柔女声", "string"),
        fakeWidget("speech_rate", 4.15),
    ];
    const inputs = [
        { name: "engine", type: "SF_AUK_ENGINE", link: 1 },
        { name: "input_audio", type: "AUDIO", link: audioConnected ? 2 : null },
    ];
    return new FakeNode(widgets, inputs);
}

function widgetOf(node, name) {
    return node.widgets.find((widget) => widget.name === name);
}

function inputOf(node, name) {
    return node.inputs.find((input) => input.name === name);
}

// ---- 注册与初始状态 ----
check("扩展已注册", capturedExts.length === 1 && capturedExts[0].name === "sfnodes.auk_long_speech");
const ext = capturedExts[0];

const node = makeNode();
ext.nodeCreated(node);
check("参考音色：reference_seconds 显示", lib.isWidgetVisible(widgetOf(node, "reference_seconds")));
check("参考音色：voice_description 隐藏", !lib.isWidgetVisible(widgetOf(node, "voice_description")));
check("参考音色：input_audio 插槽存在", !!inputOf(node, "input_audio"));
check("隐藏 widget 值保留", widgetOf(node, "voice_description").value === "温柔女声");
check("显隐后节点自适应尺寸", node.sizeCalls > 0);

// ---- 切到声音描述 TTS（未连线 → 移除插槽）----
const modeWidget = widgetOf(node, "mode");
modeWidget.value = "声音描述 TTS";
modeWidget.callback("声音描述 TTS");
check("声音描述：voice_description 显示", lib.isWidgetVisible(widgetOf(node, "voice_description")));
check("声音描述：reference_seconds 隐藏", !lib.isWidgetVisible(widgetOf(node, "reference_seconds")));
check("声音描述：未连线 input_audio 插槽移除", !inputOf(node, "input_audio"));
check("移除插槽后 engine 保留", !!inputOf(node, "engine"));

// ---- 已连线的 input_audio 在声音描述模式下保留（不断线）----
const linkedNode = makeNode("参考音色 TTS", true);
ext.nodeCreated(linkedNode);
widgetOf(linkedNode, "mode").value = "声音描述 TTS";
widgetOf(linkedNode, "mode").callback("声音描述 TTS");
check("已连线插槽保留", !!inputOf(linkedNode, "input_audio")
    && lib.isWidgetVisible(widgetOf(linkedNode, "voice_description")));

// ---- 切回参考音色（补回插槽）----
widgetOf(node, "mode").value = "参考音色 TTS";
widgetOf(node, "mode").callback("参考音色 TTS");
check("切回：input_audio 补回且类型 AUDIO", inputOf(node, "input_audio")?.type === "AUDIO");
check("切回：reference_seconds 显示", lib.isWidgetVisible(widgetOf(node, "reference_seconds")));
check("切回：voice_description 隐藏", !lib.isWidgetVisible(widgetOf(node, "voice_description")));

// ---- onAfterGraphConfigured 重放（configure 直赋值不触发 callback）----
const loaded = makeNode("参考音色 TTS");
ext.nodeCreated(loaded);
widgetOf(loaded, "mode").value = "声音描述 TTS";
loaded.onAfterGraphConfigured();
check("配置完成后重放显隐", lib.isWidgetVisible(widgetOf(loaded, "voice_description"))
    && !lib.isWidgetVisible(widgetOf(loaded, "reference_seconds"))
    && !inputOf(loaded, "input_audio"));
check("currentMode 读取", mod.currentMode(loaded) === "声音描述 TTS");
check("desiredSourceInputs", JSON.stringify(mod.desiredSourceInputs("参考音色 TTS")) === '["input_audio"]'
    && mod.desiredSourceInputs("声音描述 TTS").length === 0);

if (failures.length) {
    console.log(`\n${failures.length} 项失败：`);
    for (const name of failures) console.log("  -", name);
    process.exit(1);
}
console.log("\nOK");
