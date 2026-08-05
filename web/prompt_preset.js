import { app } from "/scripts/app.js";

const DISABLED = "禁用";

app.registerExtension({
    name: "sfnodes.prompt_preset",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SFPromptPreset") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const poseWidget = this.widgets?.find((w) => w.name === "pose_preset");
            const coupleWidget = this.widgets?.find((w) => w.name === "couple_preset");
            if (!poseWidget || !coupleWidget) return;

            // 单人/双人动作互斥：选择其一自动将另一个置为"禁用"
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
            linkMutualExclusion(poseWidget, coupleWidget);
            linkMutualExclusion(coupleWidget, poseWidget);
        };
    },
});
