// SF Image Interrogator 预设联动：切换 preset 时自动把对应预设文本填入 prompt widget，
// 之后仍可手动编辑。
// 预设数据唯一来源为后端 nodes/model/krea2.py 的 INTERROGATOR_PRESETS（经 API 获取），
// 前端不内嵌副本，避免双份维护。

import { app } from "/scripts/app.js";

const PRESETS_API = "/api/sfnodes/interrogator_presets";

let presets = null;          // 预设缓存：{key: text}
let retryLogged = false;
const pendingNodes = [];     // 缓存就绪前创建的节点挂载队列

async function loadPresets() {
    if (presets) return;
    try {
        const resp = await fetch(PRESETS_API);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (!data || typeof data !== "object" || Object.keys(data).length === 0) {
            throw new Error("empty presets");
        }
        presets = data;
        attachPending();
        return;
    } catch (e) {
        if (!retryLogged) {
            retryLogged = true;
            console.error("[SFImageInterrogator] 预设加载失败，每 3 秒自动重试:", e);
        }
    }
    setTimeout(loadPresets, 3000);
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
    },
    // 防御：ComfyUI 前端按 widget 数组索引恢复旧工作流的值（widgets_values 位置敏感）。
    // 若旧版保存的工作流因 widget 顺序变化发生值错位（如 vision_megapixels 的数值 1
    // 落入 user_prompt、或 user_prompt 文本落入 vision_megapixels），图加载完成后按
    // widget 名自愈：user_prompt 必须是字符串，vision_megapixels 必须是数字。
    afterConfigureGraph() {
        const nodes = app.graph?._nodes?.filter((n) => n?.comfyClass === "SFImageInterrogator") ?? [];
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
        if (node?.comfyClass !== "SFImageInterrogator") return;

        const presetWidget = node.widgets?.find((w) => w.name === "preset");
        const promptWidget = node.widgets?.find((w) => w.name === "prompt");
        if (!presetWidget || !promptWidget) return;

        const init = (data) => {
            const origCallback = presetWidget.callback;
            presetWidget.callback = function (value) {
                const r = origCallback ? origCallback.call(this, value) : undefined;
                if (data[value] !== undefined) {
                    promptWidget.value = data[value];
                    node.setDirtyCanvas(true, true);
                }
                return r;
            };
        };

        if (presets) {
            init(presets);
        } else {
            pendingNodes.push(init);
        }
    },
});
