import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const DISABLED = "禁用";

const WIDGET_CATEGORY = {
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
let cardEl = null;
let mouseInsideCanvas = false;

function showDescriptionCard(text, x, y) {
    if (!cardEl) {
        cardEl = document.createElement("div");
        cardEl.style.cssText = [
            "position:fixed",
            "z-index:99999",
            "max-width:320px",
            "background:rgba(28,28,28,0.95)",
            "color:#ddd",
            "padding:6px 10px",
            "border:1px solid #555",
            "border-radius:4px",
            "font-size:12px",
            "line-height:1.5",
            "pointer-events:none",
            "display:none",
        ].join(";");
        document.body.appendChild(cardEl);
    }
    cardEl.textContent = text;
    cardEl.style.display = "block";
    cardEl.style.left = Math.min(x + 14, window.innerWidth - cardEl.offsetWidth - 8) + "px";
    cardEl.style.top = Math.min(y + 14, window.innerHeight - cardEl.offsetHeight - 8) + "px";
}

function hideDescriptionCard() {
    if (cardEl) cardEl.style.display = "none";
}

function widgetAt(canvas, mx, my) {
    if (!canvas?.graph_mouse) return null;
    for (const node of canvas.graph?._nodes ?? []) {
        if (!node.widgets) continue;
        for (const w of node.widgets) {
            const category = WIDGET_CATEGORY[w.name];
            if (!category || w.type !== "combo" || w.hidden) continue;
            if (!w.pos || !w.size) continue;
            const px = node.pos[0] + w.pos[0];
            const py = node.pos[1] + w.pos[1];
            if (mx >= px && mx <= px + w.size[0] && my >= py && my <= py + w.size[1]) {
                return { widget: w, category };
            }
        }
    }
    return null;
}

function onCanvasMouseMove(e) {
    const canvas = app.canvas;
    if (!canvas?.graph_mouse) return;
    const hit = widgetAt(canvas, canvas.graph_mouse[0], canvas.graph_mouse[1]);
    if (!hit) {
        hideDescriptionCard();
        return;
    }
    const desc = presetDescriptions?.[hit.category]?.[hit.widget.value];
    if (desc) {
        showDescriptionCard(desc, e.clientX, e.clientY);
    } else {
        hideDescriptionCard();
    }
}

app.registerExtension({
    name: "sfnodes.prompt_preset",

    async setup() {
        try {
            const resp = await api.fetchApi("/api/sfnodes/prompt_presets");
            if (resp.ok) {
                presetDescriptions = await resp.json();
            }
        } catch (err) {
            console.log("[SFPromptPreset] 描述加载失败，悬浮卡片不可用:", err);
        }
    },

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

// 悬浮描述卡片：canvas mousemove 命中检测（仅挂载一次）
app.registerExtension({
    name: "sfnodes.prompt_preset.desc_card",
    async setup() {
        const canvas = app.canvas;
        if (!canvas) return;
        canvas.addEventListener("mousemove", onCanvasMouseMove);
        canvas.addEventListener("mouseleave", hideDescriptionCard);
        window.addEventListener("scroll", hideDescriptionCard, true);
        window.addEventListener("dragstart", hideDescriptionCard, true);
    },
});
