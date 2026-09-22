import { app } from "/scripts/app.js";
import {
    installConfiguredSlotRecovery,
    installDynamicSlots,
    removeInputAt,
    syncInputLinkTargets,
} from "./sf_dynamic_slots.js";
import { isWidgetVisible, refreshWidgetSnapshot, setWidgetVisible } from "./sf_widget_visibility_lib.js";

const EXT_NAME = "sfnodes.QwenImage21PromptEnhancer";

const MODE_OFFICIAL_PE = "本地官方PE";
const MODE_LOCAL_LLM = "本地LLM";
const MODE_LLAMA = "本地LLaMA";
const MODE_API = "API";

const SLOT_CONFIG = {
    inputPrefix: "image_",
    inputStart: 1,
    inputCount: 8,
    inputType: "IMAGE",
    initialInputs: 1,
};

// 模式 -> 该模式生效的专属 widget（未列出的模式专属 widget 一律隐藏）。
// mode/task/prompt/output_language 为全模式公共项，常显不参与显隐。
const MODE_WIDGETS = {
    [MODE_OFFICIAL_PE]: [
        "max_tokens", "temperature", "top_k", "top_p", "min_p", "repetition_penalty",
        "seed", "thinking", "vision_megapixels", "unload_after",
    ],
    [MODE_LOCAL_LLM]: [
        "max_tokens", "temperature", "top_k", "top_p", "min_p", "repetition_penalty",
        "seed", "thinking", "vision_megapixels", "unload_after",
    ],
    // 本地LLaMA：thinking 由插件 chat_handler 决定，不生效；其余同 CLIP 本地模式
    [MODE_LLAMA]: [
        "max_tokens", "temperature", "top_k", "top_p", "min_p", "repetition_penalty",
        "seed", "vision_megapixels", "unload_after",
    ],
    // API：llm_client 仅透传 temperature/seed（其余采样参数不生效），
    // max_tokens 不发送、thinking 由 llm_client 处理、无本地模型可卸载。
    [MODE_API]: ["temperature", "seed", "vision_megapixels", "detail"],
};
const ALWAYS_VISIBLE_WIDGETS = ["mode", "task", "prompt", "output_language"];
const MANAGED_WIDGETS = [...new Set(Object.values(MODE_WIDGETS).flat())];

// 模式 -> 生效的本地源输入（clip 或 llama_model；API 两者都不用）
const SOURCE_INPUT_TYPES = { clip: "CLIP", llama_model: "LLAMACPPMODEL" };
const MODE_SOURCE_INPUTS = {
    [MODE_OFFICIAL_PE]: ["clip"],
    [MODE_LOCAL_LLM]: ["clip"],
    [MODE_LLAMA]: ["llama_model"],
    [MODE_API]: [],
};

const SETUP = new WeakSet();

function findWidget(node, name) {
    return (node?.widgets || []).find((widget) => widget?.name === name);
}

export function currentMode(node) {
    return findWidget(node, "mode")?.value ?? MODE_OFFICIAL_PE;
}

export function modeWidgetNames(mode) {
    return [...ALWAYS_VISIBLE_WIDGETS, ...(MODE_WIDGETS[mode] ?? MODE_WIDGETS[MODE_OFFICIAL_PE])];
}

export function desiredSourceInputs(mode) {
    return [...(MODE_SOURCE_INPUTS[mode] ?? [])];
}

function isInputConnected(input) {
    return !!input && input.link != null && input.link !== -1;
}

// 按模式显隐 widget（隐藏只影响渲染，值仍随工作流保存/提交）。返回是否有变化。
export function applyModeWidgets(node, mode) {
    const visibleNames = new Set(MODE_WIDGETS[mode] ?? MODE_WIDGETS[MODE_OFFICIAL_PE]);
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

// 按模式增删源输入槽：不需要的移除（已连线的保留——不静默断用户的线，后端按模式忽略），
// 需要的补回。返回是否有变化。
export function syncSourceInputs(node, mode, graph) {
    const desired = new Set(desiredSourceInputs(mode));
    let changed = false;
    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index--) {
        const input = node.inputs[index];
        if (!input || !(input.name in SOURCE_INPUT_TYPES)) continue;
        if (desired.has(input.name) || isInputConnected(input)) continue;
        removeInputAt(node, index);
        changed = true;
    }
    for (const name of desired) {
        if (node.inputs?.some((input) => input?.name === name)) continue;
        node.addInput?.(name, SOURCE_INPUT_TYPES[name]);
        changed = true;
    }
    if (changed) {
        syncInputLinkTargets(node, graph);
        node._widgetSlotsDirty = true;
        node.setDirtyCanvas?.(true, true);
    }
    return changed;
}

function fitNodeHeight(node, graph) {
    if (!node || node.flags?.collapsed) return;
    const width = Math.max(node.size?.[0] || 300, 300);
    const size = node.computeSize?.([width, node.size?.[1] || 0]);
    if (!size) return;
    node.setSize?.([Math.max(width, size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
    (graph || node.graph || app.graph)?.setDirtyCanvas?.(true, true);
}

export function applyModeVisibility(node, mode, graph) {
    const widgetsChanged = applyModeWidgets(node, mode);
    const inputsChanged = syncSourceInputs(node, mode, graph);
    if (widgetsChanged) fitNodeHeight(node, graph);
    return widgetsChanged || inputsChanged;
}

function installModeWatch(node) {
    const widget = findWidget(node, "mode");
    if (!widget || widget._sfQwen21ModeWatch) return;
    widget._sfQwen21ModeWatch = true;
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

    installDynamicSlots(node, SLOT_CONFIG);
    installConfiguredSlotRecovery(node, SLOT_CONFIG);
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
        if (node.comfyClass !== "SFQwenImage21PromptEnhancer") return;
        setupNode(node);
    },

    loadedGraphNode(node) {
        if (node.comfyClass !== "SFQwenImage21PromptEnhancer") return;
        setupNode(node);
    },
});
