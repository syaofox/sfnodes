import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const DISABLED = "禁用";

const WIDGET_CATEGORY = {
    celebrity_preset: "Celebrity",
    outfit_preset: "Outfit",
    pose_preset: "Pose",
    couple_preset: "Couple Pose",
    environment_preset: "Environment",
    lighting_preset: "Lighting",
    style_preset: "Style",
    camera_angle_preset: "Camera Angle",
    camera_distance_preset: "Camera Distance",
    camera_lens_preset: "Camera Lens",
};

let presetDescriptions = null;

// 把当前选中预设的英文 description 写入 widget.tooltip，
// ComfyUI（新旧前端）均优先显示 widget.tooltip，随选中值变化自动更新
function syncWidgetTooltip(w) {
    const category = WIDGET_CATEGORY[w?.name];
    if (!category) return;
    const desc = presetDescriptions?.[category]?.[w.value];
    w.tooltip = desc || null;
}

app.registerExtension({
    name: "sfnodes.prompt_preset",

    async setup() {
        try {
            const resp = await api.fetchApi("/api/sfnodes/prompt_presets");
            if (resp.ok) {
                presetDescriptions = await resp.json();
                // 工作流恢复后节点已存在的情况：加载完成后统一同步一次
                for (const node of app.graph?._nodes ?? []) {
                    if (node?.type !== "SFPromptPreset" || !node.widgets) continue;
                    for (const w of node.widgets) {
                        if (WIDGET_CATEGORY[w.name]) {
                            syncWidgetTooltip(w);
                        }
                    }
                }
            }
        } catch (err) {
            console.log("[SFPromptPreset] 描述加载失败，预设说明不可用:", err);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SFPromptPreset") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const widgets = this.widgets ?? [];

            // 单人/双人动作互斥：选择其一自动将另一个置为"禁用"
            const poseWidget = widgets.find((w) => w.name === "pose_preset");
            const coupleWidget = widgets.find((w) => w.name === "couple_preset");
            const linkMutualExclusion = (w1, w2) => {
                const originalCallback = w1.callback;
                w1.callback = function (...args) {
                    if (this.value !== DISABLED) {
                        w2.value = DISABLED;
                    }
                    if (typeof originalCallback === "function") {
                        return originalCallback.apply(this, args);
                    }
                };
            };
            if (poseWidget && coupleWidget) {
                linkMutualExclusion(poseWidget, coupleWidget);
                linkMutualExclusion(coupleWidget, poseWidget);
            }

            // 预设 widget：选中值变化时把 description 同步到 tooltip
            for (const w of widgets) {
                if (!WIDGET_CATEGORY[w.name]) continue;
                const originalCallback = w.callback;
                w.callback = function (...args) {
                    syncWidgetTooltip(this);
                    if (typeof originalCallback === "function") {
                        return originalCallback.apply(this, args);
                    }
                };
                if (presetDescriptions) {
                    syncWidgetTooltip(w);
                }
            }
        };
    },
});
