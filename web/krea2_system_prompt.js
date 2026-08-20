// SF Krea2 System Prompt 预设联动：切换 preset 时自动把对应预设文本填入 text widget，
// 之后仍可手动编辑；选 "none" 不覆盖当前内容。提供"管理预设"入口（新增/修改/删除/复位）。
// 预设数据唯一来源为后端（sf_utils/krea2_presets.py，合并内置+用户覆盖，经 API 获取），
// 前端不内嵌副本，避免双份维护。combo 选项由用户预设动态重建（VALIDATE_INPUTS 兜底）。
//
// 预设文本以字面量随工作流保存（数据载体模式），后端改进预设措辞后旧工作流不生效。
// 对策：选择预设时在 node.properties.krea2PresetName 记下"文本由哪个预设派生"；
// 工作流加载/复制时若标记与当前 preset 一致，自动刷新为最新预设文本——用户手动
// 编辑过即清除标记，绝不覆盖手动内容（旧工作流无标记，保持原样）。

import { app } from "/scripts/app.js";
import {
  fetchPresets,
  addManageButton,
  setPresetOptions,
  reloadNodes,
  presetsChangedEvent,
  nodesOfClass,
} from "./sf_krea2_presets.js";

const KIND = "krea2";
const COMIFY_CLASS = "SFKrea2SystemPrompt";
const PRESET_PROP = "krea2PresetName";
const MAX_ATTEMPTS = 10;

let presets = null;          // 合并预设缓存：{key: text}
let failedAttempts = 0;
let gaveUp = false;
const pendingNodes = [];     // 缓存就绪前创建的节点挂载队列

async function loadPresets() {
    if (presets || gaveUp) return;
    failedAttempts += 1;
    try {
        const data = await fetchPresets(KIND);
        presets = data.presets;
        for (const n of nodesOfClass(COMIFY_CLASS)) {
            n.properties = n.properties || {};
            n.properties._krea2PresetData = presets;
            setPresetOptions(n, presets);
        }
        attachPending();
        return;
    } catch (e) {
        if (failedAttempts === 1 || failedAttempts === MAX_ATTEMPTS) {
            console.error(`[SFKrea2SystemPrompt] 预设加载失败（第 ${failedAttempts}/${MAX_ATTEMPTS} 次）:`, e);
        }
    }
    if (failedAttempts >= MAX_ATTEMPTS) {
        gaveUp = true;
        console.error("[SFKrea2SystemPrompt] 预设加载已放弃（后端路由未生效，请重启容器）。预设联动与管理不可用，可手动编辑文本");
    } else {
        setTimeout(loadPresets, 3000);
    }
}

function attachPending() {
    while (pendingNodes.length > 0) {
        pendingNodes.shift()(presets);
    }
}

function syncFromPreset(node, data) {
    const presetWidget = node.widgets?.find((w) => w.name === "preset");
    const textWidget = node.widgets?.find((w) => w.name === "text");
    if (!presetWidget || !textWidget) return;
    const preset = presetWidget.value;
    if (preset === "none" || data[preset] === undefined) return;
    // 仅当文本确认为"预设派生"（标记与当前 preset 一致）时刷新为最新措辞；
    // 标记缺失（旧工作流/手动编辑过）一律不动，宁可不更新也不覆盖用户内容。
    if (node.properties?.[PRESET_PROP] === preset) {
        textWidget.value = data[preset];
        node.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "sfnodes.krea2_system_prompt",
    setup() {
        loadPresets();
        document.addEventListener(presetsChangedEvent(KIND), async () => {
            const data = await reloadNodes(KIND, COMIFY_CLASS);
            if (data && data.presets) presets = data.presets;
        });
    },
    nodeCreated(node) {
        if (node?.comfyClass !== COMIFY_CLASS) return;

        addManageButton(node, KIND);

        const presetWidget = node.widgets?.find((w) => w.name === "preset");
        const textWidget = node.widgets?.find((w) => w.name === "text");
        if (!presetWidget || !textWidget) return;

        // 立即包装所有 callback（在 Vue 首次渲染前），确保 combo/text 值变化时能正确联动。
        // 预设数据通过 node.properties 间接引用，不受 widget 重建影响。
        const origPresetCallback = presetWidget.callback;
        presetWidget.callback = function (value) {
            const r = origPresetCallback ? origPresetCallback.call(this, value) : undefined;
            const cur = presets || node.properties._krea2PresetData;
            if (value !== "none" && cur && cur[value] !== undefined) {
                textWidget.value = cur[value];
                node.properties[PRESET_PROP] = value;
                node.setDirtyCanvas(true, true);
            }
            return r;
        };

        const origTextCallback = textWidget.callback;
        textWidget.callback = function (value) {
            const r = origTextCallback ? origTextCallback.call(this, value) : undefined;
            delete node.properties[PRESET_PROP];
            return r;
        };

        const origConfigure = node.configure;
        node.configure = function (info) {
            const r = origConfigure ? origConfigure.call(this, info) : undefined;
            syncFromPreset(node, node.properties._krea2PresetData || {});
            return r;
        };
        syncFromPreset(node, node.properties._krea2PresetData || {});

        const init = (data) => {
            node.properties._krea2PresetData = data;
            setPresetOptions(node, data);
            syncFromPreset(node, data);
        };

        if (presets) {
            init(presets);
        } else {
            pendingNodes.push(init);
        }
    },
});
