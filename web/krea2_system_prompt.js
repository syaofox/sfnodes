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
        for (const n of nodesOfClass(COMIFY_CLASS)) setPresetOptions(n, presets);
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

        const init = (data) => {
            setPresetOptions(node, data);
            const origPresetCallback = presetWidget.callback;
            presetWidget.callback = function (value) {
                const r = origPresetCallback ? origPresetCallback.call(this, value) : undefined;
                if (value !== "none" && data[value] !== undefined) {
                    textWidget.value = data[value];
                    node.properties[PRESET_PROP] = value; // 标记预设派生，加载时可自动同步最新措辞
                    node.setDirtyCanvas(true, true);
                }
                return r;
            };

            // 用户手动编辑 = 接管文本控制权：清除派生标记，防止加载时被自动覆盖。
            const origTextCallback = textWidget.callback;
            textWidget.callback = function (value) {
                const r = origTextCallback ? origTextCallback.call(this, value) : undefined;
                delete node.properties[PRESET_PROP];
                return r;
            };

            // widget 值在 configure 时才恢复（工作流加载/复制路径），包装后在值恢复处同步。
            const origConfigure = node.configure;
            node.configure = function (info) {
                const r = origConfigure ? origConfigure.call(this, info) : undefined;
                syncFromPreset(node, data);
                return r;
            };
            syncFromPreset(node, data);
        };

        if (presets) {
            init(presets);
        } else {
            pendingNodes.push(init);
        }
    },
});
