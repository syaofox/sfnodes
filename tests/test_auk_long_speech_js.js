// SFAuKLongSpeech mode 联动显隐测试（Node 直接运行：node tests/test_auk_long_speech_js.js）
// 覆盖：扩展注册名；三模式 widget 显隐（TTS 互斥项、处理模式 instruction/duration_mode/
// speed_multiplier/预设下拉）；input_audio 插槽增删（未连线移除、已连线保留）；切回恢复；
// 隐藏 widget 值不丢；预设下拉填入 instruction 与 properties 恢复；mode callback 与
// onAfterGraphConfigured 双路重放；显隐变化后节点自适应尺寸。
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
        this.properties = {};
        this._widgetSlotsDirty = false;
        this.sizeCalls = 0;
    }
    addWidget(type, name, value, callback, options) {
        const widget = {
            name, value, type, callback, options: options || {},
            hidden: false, computeSize: () => [100, 20], _state: {}, triggerDraw: () => {},
        };
        if (widget.options.serialize !== undefined) widget.serialize = widget.options.serialize;
        this.widgets.push(widget);
        return widget;
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

const presetCode = loadStripped("sf_auk_presets_lib.js");
const presets = new Function(
    presetCode + "\nreturn { DEFAULT_GROUP, DEFAULT_TEMPLATE, SCOPE_PROCESS, groupNames, itemsOf, templateText };"
)();

const names = ["currentMode", "desiredSourceInputs", "applyModeWidgets", "syncSourceInputs", "applyModeVisibility"];
const mod = new Function(
    "app", "removeInputAt", "syncInputLinkTargets",
    "setWidgetVisible", "isWidgetVisible", "refreshWidgetSnapshot", "fitNodeToContent",
    "DEFAULT_GROUP", "SCOPE_PROCESS", "groupNames", "itemsOf", "templateText",
    loadStripped("sf_auk_long_speech.js") + "\nreturn {" + names.join(",") + "};"
)(
    app, dyn.removeInputAt, dyn.syncInputLinkTargets,
    lib.setWidgetVisible, lib.isWidgetVisible, lib.refreshWidgetSnapshot, lib.fitNodeToContent,
    presets.DEFAULT_GROUP, presets.SCOPE_PROCESS, presets.groupNames, presets.itemsOf, presets.templateText
);

// ---- 节点工厂 ----
function makeNode(modeValue = "参考音色 TTS", audioConnected = false) {
    const widgets = [
        fakeWidget("engine", "engine"),
        fakeWidget("text", "", "string"),
        fakeWidget("mode", modeValue, "combo"),
        fakeWidget("max_chunk_seconds", 24),
        fakeWidget("reference_seconds", 10),
        fakeWidget("voice_description", "温柔女声", "string"),
        fakeWidget("speech_rate", 4.15),
        fakeWidget("instruction", "", "string"),
        fakeWidget("duration_mode", "等长", "combo"),
        fakeWidget("speed_multiplier", 1.0),
        fakeWidget("ref_tail_seconds", 4.0),
        fakeWidget("pause_seconds", 0.1),
        fakeWidget("continuity", "滚动参考", "combo"),
        fakeWidget("trim_trailing_silence", true, "boolean"),
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

// ---- 长音频处理模式：显隐 + 预设下拉 ----
const process = makeNode("长音频处理（编辑/增强）");
ext.nodeCreated(process);
const pText = widgetOf(process, "text");
const pInstruction = widgetOf(process, "instruction");
const pDuration = widgetOf(process, "duration_mode");
const pSpeed = widgetOf(process, "speed_multiplier");
const pPresetGroup = widgetOf(process, "预设分类");
const pPresetTemplate = widgetOf(process, "提示词模板");
const pPresetApply = widgetOf(process, "填入 instruction");
check("处理模式：instruction/duration/speed 显示", lib.isWidgetVisible(pInstruction)
    && lib.isWidgetVisible(pDuration) && lib.isWidgetVisible(pSpeed));
check("处理模式：预设下拉已挂载且显示", !!pPresetGroup && !!pPresetTemplate && !!pPresetApply
    && lib.isWidgetVisible(pPresetApply));
check("处理模式：TTS 参数隐藏", !lib.isWidgetVisible(pText)
    && !lib.isWidgetVisible(widgetOf(process, "speech_rate"))
    && !lib.isWidgetVisible(widgetOf(process, "ref_tail_seconds"))
    && !lib.isWidgetVisible(widgetOf(process, "pause_seconds"))
    && !lib.isWidgetVisible(widgetOf(process, "continuity"))
    && !lib.isWidgetVisible(widgetOf(process, "trim_trailing_silence"))
    && !lib.isWidgetVisible(widgetOf(process, "reference_seconds"))
    && !lib.isWidgetVisible(widgetOf(process, "voice_description")));
check("处理模式：input_audio 插槽存在", !!inputOf(process, "input_audio"));
check("处理模式：max_chunk_seconds 常显", lib.isWidgetVisible(widgetOf(process, "max_chunk_seconds")));
check("预设下拉置于 instruction 之前", process.widgets.indexOf(pPresetApply) < process.widgets.indexOf(pInstruction));
const processGroupValues = pPresetGroup.options.values;
check("处理模式预设不含 TTS 两组", !processGroupValues.includes("1. 参考音色 TTS")
    && !processGroupValues.includes("2. 声音描述 TTS"));
check("处理模式预设不含长模式不适用组", !processGroupValues.includes("3. 语音内容编辑（替换、增添、删除）")
    && !processGroupValues.includes("4. 歌词编辑")
    && !processGroupValues.includes("14. 多人语音分离")
    && !processGroupValues.includes("16. 按说话内容提取目标说话人"));
check("处理模式预设含增强/语速/分离等", ["6. 语速调整", "13. 语音增强（降噪、去混响、修复）", "15. 音乐人声分离"]
    .every((name) => processGroupValues.includes(name)));

// 选模板 → 填入 instruction；选择存 properties
pPresetGroup.value = "13. 语音增强（降噪、去混响、修复）";
pPresetGroup.callback(pPresetGroup.value);
pPresetTemplate.value = "Denoise · CN";
pPresetTemplate.callback(pPresetTemplate.value);
check("预设填入 instruction", pInstruction.value === "请只去除背景噪声，保留其他内容，输出等长结果。");
check("预设选择存 properties", process.properties.sfAukPresetGroup === "13. 语音增强（降噪、去混响、修复）"
    && process.properties.sfAukPresetTemplate === "Denoise · CN");

// configure 重放：properties 恢复 + 模式显隐
const processLoaded = makeNode("参考音色 TTS");
ext.nodeCreated(processLoaded);
processLoaded.properties = { sfAukPresetGroup: "5. 音高调整", sfAukPresetTemplate: "Lower · CN" };
widgetOf(processLoaded, "mode").value = "长音频处理（编辑/增强）";
processLoaded.onAfterGraphConfigured();
check("加载恢复预设选择", widgetOf(processLoaded, "预设分类").value === "5. 音高调整"
    && widgetOf(processLoaded, "提示词模板").value === "Lower · CN");
check("加载恢复后处理模式显隐", lib.isWidgetVisible(widgetOf(processLoaded, "instruction"))
    && !lib.isWidgetVisible(widgetOf(processLoaded, "text")));

// 处理模式 → 声音描述：插槽移除（未连线）、处理参数隐藏
widgetOf(process, "mode").value = "声音描述 TTS";
widgetOf(process, "mode").callback("声音描述 TTS");
check("处理→声音描述：处理参数隐藏", !lib.isWidgetVisible(pInstruction)
    && !lib.isWidgetVisible(pPresetApply) && lib.isWidgetVisible(widgetOf(process, "voice_description")));
check("处理→声音描述：input_audio 移除", !inputOf(process, "input_audio"));
check("desiredSourceInputs 处理模式", JSON.stringify(mod.desiredSourceInputs("长音频处理（编辑/增强）")) === '["input_audio"]');

if (failures.length) {
    console.log(`\n${failures.length} 项失败：`);
    for (const name of failures) console.log("  -", name);
    process.exit(1);
}
console.log("\nOK");
