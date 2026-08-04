// SF Power Lora Preset - 预设选择节点：combo 刷新按钮
import { app } from "/scripts/app.js";

const NODE_TYPE = "SFPowerLoraPreset";
const API = "/api/sfnodes/lora_presets";

async function fetchPresetNames() {
    try {
        const r = await fetch(API);
        if (!r.ok) throw new Error(`load presets failed: ${r.status}`);
        const res = await r.json();
        return ["None", ...Object.keys(res?.presets || {}).sort()];
    } catch (e) {
        console.error("[SFPowerLoraPreset]", e);
        return null;
    }
}

function refreshCombo(node, widget) {
    fetchPresetNames().then(names => {
        if (!names || !widget?.options) return;
        const cur = widget.value;
        widget.options.values = names;
        if (names.includes(cur)) {
            widget.value = cur;
        } else {
            widget.value = names[0];
        }
        node.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "sfnodes.SFPowerLoraPreset",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;

        const widget = node.widgets?.find(w => w.name === "preset");
        if (!widget) return;

        // 刷新已保存预设列表（新增/删除预设后无需重载工作流）
        refreshCombo(node, widget);

        const btn = node.addWidget("button", "\u21BB Refresh", null, () => {
            refreshCombo(node, widget);
        });
        btn.serialize = false;
    },
});
