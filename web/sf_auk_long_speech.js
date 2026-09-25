// SFAuKLongSpeech mode 联动显隐 + 处理模式预设下拉（前端）
//
// 参考音色 TTS：显示 text/reference_seconds 等 TTS 参数，隐藏 voice_description 与处理模式参数；
// 声音描述 TTS：显示 text/voice_description，隐藏 reference_seconds 与处理模式参数；
// 长音频处理：显示 instruction（+ 官方模板预设下拉）/duration_mode/speed_multiplier，隐藏 TTS 参数。
// input_audio 插槽：TTS 参考音色与处理模式需要，声音描述模式未连线时移除（已连线保留，不静默断线）。
//
// 复用公共库：sf_widget_visibility_lib（widget 显隐，隐藏只影响渲染、值仍随工作流保存）
// + sf_dynamic_slots（插槽增删与 link.target_slot 位移修正）+ sf_auk_presets_lib（官方模板数据）。
// mode callback 与 onAfterGraphConfigured 双路重放（加载/粘贴恢复时也生效）。

import { app } from "/scripts/app.js";

import { removeInputAt, syncInputLinkTargets } from "./sf_dynamic_slots.js";
import {
    DEFAULT_GROUP,
    SCOPE_PROCESS,
    groupNames,
    itemsOf,
    templateText,
} from "./sf_auk_presets_lib.js";
import {
    fitNodeToContent,
    isWidgetVisible,
    refreshWidgetSnapshot,
    setWidgetVisible,
} from "./sf_widget_visibility_lib.js";

const EXT_NAME = "sfnodes.auk_long_speech";

const MODE_REFERENCE = "参考音色 TTS";
const MODE_DESCRIPTION = "声音描述 TTS";
const MODE_PROCESS = "长音频处理（编辑/增强）";

const PRESET_GROUP = "预设分类";
const PRESET_TEMPLATE = "提示词模板";
const PRESET_APPLY = "填入 instruction";
const PROP_GROUP = "sfAukPresetGroup";
const PROP_TEMPLATE = "sfAukPresetTemplate";

const TTS_WIDGETS = [
    "text", "max_chunk_seconds", "speech_rate", "ref_tail_seconds", "pause_seconds",
    "continuity", "trim_trailing_silence", "seed", "nfe_steps", "cfg_strength", "sway_sampling_coef",
];
const PROCESS_WIDGETS = [
    "instruction", "duration_mode", "speed_multiplier",
    "max_chunk_seconds", "seed", "nfe_steps", "cfg_strength", "sway_sampling_coef",
    PRESET_GROUP, PRESET_TEMPLATE, PRESET_APPLY,
];

// 模式 -> 该模式生效的 widget（未列出的受管 widget 一律隐藏）
const MODE_WIDGETS = {
    [MODE_REFERENCE]: [...TTS_WIDGETS, "reference_seconds"],
    [MODE_DESCRIPTION]: [...TTS_WIDGETS, "voice_description"],
    [MODE_PROCESS]: PROCESS_WIDGETS,
};
const MANAGED_WIDGETS = [...new Set(Object.values(MODE_WIDGETS).flat())];

// 模式 -> 生效的源输入（AUDIO 插槽）
const SOURCE_INPUTS = { input_audio: "AUDIO" };
const MODE_SOURCE_INPUTS = {
    [MODE_REFERENCE]: ["input_audio"],
    [MODE_DESCRIPTION]: [],
    [MODE_PROCESS]: ["input_audio"],
};

const SETUP = new WeakSet();

function findWidget(node, name) {
    return (node?.widgets || []).find((widget) => widget?.name === name);
}

export function currentMode(node) {
    return findWidget(node, "mode")?.value ?? MODE_REFERENCE;
}

export function desiredSourceInputs(mode) {
    return [...(MODE_SOURCE_INPUTS[mode] ?? MODE_SOURCE_INPUTS[MODE_REFERENCE])];
}

function isInputConnected(input) {
    return !!input && input.link != null && input.link !== -1;
}

function setComboValues(widget, values) {
    if (!widget.options) widget.options = {};
    widget.options.values = values;
}

// 在 instruction 之前插入官方模板预设三件套（serialize:false，不占 widgets_values 位）
function installPresetWidgets(node) {
    if (findWidget(node, PRESET_APPLY)) return;
    const instruction = findWidget(node, "instruction");
    if (!instruction) return;

    const applyTemplate = () => {
        const text = templateText(groupWidget.value, templateWidget.value);
        if (!text) return;
        instruction.value = text;
        node.setDirtyCanvas?.(true, true);
    };
    const remember = () => {
        node.properties[PROP_GROUP] = groupWidget.value;
        node.properties[PROP_TEMPLATE] = templateWidget.value;
    };
    // 处理模式只列 process scope 的模板（不含 TTS/内容编辑/多人分离等长模式不适用项）
    const processGroups = groupNames(SCOPE_PROCESS);
    const initialGroup = processGroups[0] ?? DEFAULT_GROUP;
    const initialTemplate = itemsOf(initialGroup, SCOPE_PROCESS)[0]?.label ?? "";
    const syncTemplates = (keepValue) => {
        const labels = itemsOf(groupWidget.value, SCOPE_PROCESS).map((item) => item.label);
        setComboValues(templateWidget, labels);
        if (!keepValue || !labels.includes(templateWidget.value)) {
            templateWidget.value = labels[0] ?? "";
        }
    };

    const groupWidget = node.addWidget("combo", PRESET_GROUP, initialGroup, () => {
        syncTemplates(false);
        remember();
        node.setDirtyCanvas?.(true, true);
    }, { values: processGroups, serialize: false });
    const templateWidget = node.addWidget("combo", PRESET_TEMPLATE, initialTemplate, () => {
        applyTemplate();
        remember();
    }, { values: itemsOf(initialGroup, SCOPE_PROCESS).map((item) => item.label), serialize: false });
    const applyWidget = node.addWidget("button", PRESET_APPLY, null, () => applyTemplate(), { serialize: false });

    const added = [groupWidget, templateWidget, applyWidget];
    const rest = node.widgets.filter((widget) => !added.includes(widget));
    const index = rest.indexOf(instruction);
    node.widgets.length = 0;
    node.widgets.push(...rest.slice(0, index), ...added, ...rest.slice(index));

    node._sfAukPresetRestore = () => {
        const savedGroup = node.properties?.[PROP_GROUP];
        groupWidget.value = processGroups.includes(savedGroup) ? savedGroup : initialGroup;
        syncTemplates(false);
        const labels = itemsOf(groupWidget.value, SCOPE_PROCESS).map((item) => item.label);
        const savedTemplate = node.properties?.[PROP_TEMPLATE];
        templateWidget.value = labels.includes(savedTemplate) ? savedTemplate : (labels[0] ?? "");
    };
}

// 按模式显隐 widget（隐藏只影响渲染，值仍随工作流保存/提交）。返回是否有变化。
export function applyModeWidgets(node, mode) {
    const visibleNames = new Set(MODE_WIDGETS[mode] ?? MODE_WIDGETS[MODE_REFERENCE]);
    let changed = false;
    for (const name of MANAGED_WIDGETS) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        if (isWidgetVisible(widget) !== visibleNames.has(name)) {
            changed = setWidgetVisible(widget, visibleNames.has(name)) || changed;
        }
    }
    if (changed) refreshWidgetSnapshot(node);
    return changed;
}

// 按模式增删源输入槽：不需要的移除（已连线的保留），需要的补回。返回是否有变化。
export function syncSourceInputs(node, mode, graph) {
    const desired = new Set(desiredSourceInputs(mode));
    let changed = false;
    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index--) {
        const input = node.inputs[index];
        if (!input || !(input.name in SOURCE_INPUTS)) continue;
        if (desired.has(input.name) || isInputConnected(input)) continue;
        removeInputAt(node, index);
        changed = true;
    }
    for (const name of desired) {
        if (node.inputs?.some((input) => input?.name === name)) continue;
        node.addInput?.(name, SOURCE_INPUTS[name]);
        changed = true;
    }
    if (changed) {
        syncInputLinkTargets(node, graph);
        node._widgetSlotsDirty = true;
        node.setDirtyCanvas?.(true, true);
    }
    return changed;
}

export function applyModeVisibility(node, mode, graph) {
    const widgetsChanged = applyModeWidgets(node, mode);
    const inputsChanged = syncSourceInputs(node, mode, graph);
    if (widgetsChanged) fitNodeToContent(node, graph);
    return widgetsChanged || inputsChanged;
}

function installModeWatch(node) {
    const widget = findWidget(node, "mode");
    if (!widget || widget._sfAukLongSpeechModeWatch) return;
    widget._sfAukLongSpeechModeWatch = true;
    const original = widget.callback;
    widget.callback = function () {
        const result = original?.apply(this, arguments);
        applyModeVisibility(node, currentMode(node), app.graph);
        return result;
    };
}

function setupNode(node) {
    if (SETUP.has(node)) {
        applyModeVisibility(node, currentMode(node), app.graph);
        return;
    }
    SETUP.add(node);
    installPresetWidgets(node);
    installModeWatch(node);
    applyModeVisibility(node, currentMode(node), app.graph);

    // 加载/粘贴恢复时 configure 直赋 widget 值不触发 callback，配置完成后重放一次
    const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
    node.onAfterGraphConfigured = function () {
        if (originalOnAfterGraphConfigured) {
            originalOnAfterGraphConfigured.apply(this, arguments);
        }
        installPresetWidgets(this);
        installModeWatch(this);
        this._sfAukPresetRestore?.();
        applyModeVisibility(this, currentMode(this), app.graph);
    };
}

app.registerExtension({
    name: EXT_NAME,

    nodeCreated(node) {
        if (node.comfyClass !== "SFAuKLongSpeech") return;
        setupNode(node);
    },

    loadedGraphNode(node) {
        if (node.comfyClass !== "SFAuKLongSpeech") return;
        setupNode(node);
    },
});
