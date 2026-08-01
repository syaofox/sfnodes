// SF Krea2 System Prompt 预设联动：切换 preset 时自动把对应预设文本填入 text widget，
// 之后仍可手动编辑；选 "none" 不覆盖当前内容。
// 预设数据唯一来源为后端 nodes/model/krea2.py 的 KREA2_PRESETS（经 API 获取），
// 前端不内嵌副本，避免双份维护。

import { app } from "/scripts/app.js";

const PRESETS_API = "/api/sfnodes/krea2_presets";

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
            console.error("[SFKrea2SystemPrompt] 预设加载失败，每 3 秒自动重试:", e);
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
    name: "sfnodes.krea2_system_prompt",
    setup() {
        loadPresets();
    },
    nodeCreated(node) {
        if (node?.comfyClass !== "SFKrea2SystemPrompt") return;

        const presetWidget = node.widgets?.find((w) => w.name === "preset");
        const textWidget = node.widgets?.find((w) => w.name === "text");
        if (!presetWidget || !textWidget) return;

        const init = (data) => {
            const origCallback = presetWidget.callback;
            presetWidget.callback = function (value) {
                const r = origCallback ? origCallback.call(this, value) : undefined;
                if (value !== "none" && data[value] !== undefined) {
                    textWidget.value = data[value];
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
