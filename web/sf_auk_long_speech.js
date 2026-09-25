// SFAuKLongSpeech mode 联动显隐（前端）
//
// 参考音色 TTS：显示 input_audio 插槽与 reference_seconds，隐藏 voice_description；
// 声音描述 TTS：显示 voice_description，隐藏 reference_seconds；input_audio 未连线时移除插槽
// （已连线保留——不静默断用户的线，后端按模式忽略该输入）。
//
// 复用公共库：sf_widget_visibility_lib（widget 显隐，隐藏只影响渲染、值仍随工作流保存）
// + sf_dynamic_slots（插槽增删与 link.target_slot 位移修正）。机制与 §126.4 同款。
// mode callback 与 onAfterGraphConfigured 双路重放（加载/粘贴恢复时也生效）。

import { app } from "/scripts/app.js";

import { removeInputAt, syncInputLinkTargets } from "./sf_dynamic_slots.js";
import {
    fitNodeToContent,
    isWidgetVisible,
    refreshWidgetSnapshot,
    setWidgetVisible,
} from "./sf_widget_visibility_lib.js";

const EXT_NAME = "sfnodes.auk_long_speech";

const MODE_REFERENCE = "参考音色 TTS";
const MODE_DESCRIPTION = "声音描述 TTS";

// 模式 -> 该模式生效的专属 widget（未列出的模式专属 widget 一律隐藏）
const MODE_WIDGETS = {
    [MODE_REFERENCE]: ["reference_seconds"],
    [MODE_DESCRIPTION]: ["voice_description"],
};
const MANAGED_WIDGETS = ["reference_seconds", "voice_description"];

// 模式 -> 生效的源输入（AUDIO 插槽；声音描述模式不使用）
const SOURCE_INPUTS = { input_audio: "AUDIO" };
const MODE_SOURCE_INPUTS = {
    [MODE_REFERENCE]: ["input_audio"],
    [MODE_DESCRIPTION]: [],
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
    installModeWatch(node);
    applyModeVisibility(node, currentMode(node), app.graph);

    // 加载/粘贴恢复时 configure 直赋 widget 值不触发 callback，配置完成后重放一次
    const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
    node.onAfterGraphConfigured = function () {
        if (originalOnAfterGraphConfigured) {
            originalOnAfterGraphConfigured.apply(this, arguments);
        }
        installModeWatch(this);
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
