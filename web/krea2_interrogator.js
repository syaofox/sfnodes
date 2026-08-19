// SF Image Interrogator 预设联动：切换 preset 时自动把对应预设文本填入 prompt widget，
// 之后仍可手动编辑；提供"管理预设"入口（新增/修改/删除/复位）。
// 预设数据唯一来源为后端（sf_utils/krea2_presets.py，合并内置+用户覆盖，经 API 获取），
// 前端不内嵌副本，避免双份维护。combo 选项由用户预设动态重建（VALIDATE_INPUTS 兜底）。

import { app } from "/scripts/app.js";
import {
  fetchPresets,
  addManageButton,
  setPresetOptions,
  reloadNodes,
  presetsChangedEvent,
  nodesOfClass,
} from "./sf_krea2_presets.js";

const KIND = "interrogator";
const COMIFY_CLASS = "SFImageInterrogator";
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
        // 重建所有已存在节点的 combo options（用户新增的预设要出现在下拉里）
        for (const n of nodesOfClass(COMIFY_CLASS)) setPresetOptions(n, presets);
        attachPending();
        return;
    } catch (e) {
        if (failedAttempts === 1 || failedAttempts === MAX_ATTEMPTS) {
            console.error(`[SFImageInterrogator] 预设加载失败（第 ${failedAttempts}/${MAX_ATTEMPTS} 次）:`, e);
        }
    }
    if (failedAttempts >= MAX_ATTEMPTS) {
        gaveUp = true;
        console.error("[SFImageInterrogator] 预设加载已放弃（后端路由未生效，请重启容器）。预设联动与管理不可用");
    } else {
        setTimeout(loadPresets, 3000);
    }
}

function attachPending() {
    while (pendingNodes.length > 0) {
        pendingNodes.shift()(presets);
    }
}

app.registerExtension({
    name: "sfnodes.krea2_interrogator",
    setup() {
        loadPresets();
        // 其他窗口/节点改了预设 → 重拉并重建（本地管理 popup 已直接重建，此为跨端兜底；
        // 用 reloadNodes 不再广播，避免监听到自身广播后无限循环）
        document.addEventListener(presetsChangedEvent(KIND), async () => {
            const data = await reloadNodes(KIND, COMIFY_CLASS);
            if (data && data.presets) presets = data.presets;
        });
    },
    // 防御：ComfyUI 前端按 widget 数组索引恢复旧工作流的值（widgets_values 位置敏感）。
    // 若旧版保存的工作流因 widget 顺序变化发生值错位（如 vision_megapixels 的数值 1
    // 落入 user_prompt、或 user_prompt 文本落入 vision_megapixels），图加载完成后按
    // widget 名自愈：user_prompt 必须是字符串，vision_megapixels 必须是数字。
    afterConfigureGraph() {
        const nodes = app.graph?._nodes?.filter((n) => n?.comfyClass === COMIFY_CLASS) ?? [];
        for (const node of nodes) {
            const userPrompt = node.widgets?.find((w) => w.name === "user_prompt");
            if (userPrompt && typeof userPrompt.value !== "string") {
                userPrompt.value = "";
                node.setDirtyCanvas?.(true, true);
            }
            const megapixels = node.widgets?.find((w) => w.name === "vision_megapixels");
            if (megapixels && typeof megapixels.value !== "number") {
                const parsed = parseFloat(megapixels.value);
                megapixels.value = Number.isFinite(parsed) ? parsed : 1.0;
                node.setDirtyCanvas?.(true, true);
            }
        }
    },
    nodeCreated(node) {
        if (node?.comfyClass !== COMIFY_CLASS) return;

        addManageButton(node, KIND);

        const presetWidget = node.widgets?.find((w) => w.name === "preset");
        const promptWidget = node.widgets?.find((w) => w.name === "prompt");
        if (!presetWidget || !promptWidget) return;

        // 立即包装 callback（在 Vue 首次渲染前），确保 combo 值变化时能正确联动。
        // 预设数据通过 node.properties 间接引用，不受 widget 重建影响。
        const origCallback = presetWidget.callback;
        presetWidget.callback = function (value) {
            const r = origCallback ? origCallback.call(this, value) : undefined;
            const data = node.properties._krea2PresetData;
            if (data && data[value] !== undefined) {
                promptWidget.value = data[value];
                node.setDirtyCanvas(true, true);
            }
            return r;
        };

        const init = (data) => {
            node.properties._krea2PresetData = data;
            setPresetOptions(node, data);
        };

        if (presets) {
            init(presets);
        } else {
            pendingNodes.push(init);
        }
    },
});
