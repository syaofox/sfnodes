// SF LoRA Preset - 预设选择节点：combo 刷新按钮（原 SF Power Lora Preset 改名）
// 2026-08 扩展：预设含可选 positive 提示词（与 triggers 分离），经额外 STRING 输出流通
import { app } from "/scripts/app.js";

const NODE_TYPE = "SFLoraPreset";
const API = "/api/sfnodes/lora_presets";

async function fetchPresetNames() {
    try {
        const r = await fetch(API);
        if (!r.ok) throw new Error(`load presets failed: ${r.status}`);
        const res = await r.json();
        return ["None", ...Object.keys(res?.presets || {}).sort()];
    } catch (e) {
        console.error("[SFLoraPreset]", e);
        return null;
    }
}

async function fetchPresetsMap() {
    try {
        const r = await fetch(API, { cache: "no-store" });
        if (!r.ok) return {};
        const j = await r.json();
        return j?.presets && typeof j.presets === "object" ? j.presets : {};
    } catch { return {}; }
}

function sanitizePositive(v) {
    if (typeof v !== "string") return "";
    const s = v.trim();
    return s.length > 8000 ? s.slice(0, 8000) : s;
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
        // 同步刷新 tooltip（选中预设的 positive 预览）
        refreshTooltip(node, widget);
    });
}

function refreshTooltip(node, widget) {
    const v = widget?.value;
    if (!v || v === "None") {
        widget.tooltip = "选择预设；None 表示不使用预设。预设含 LoRA 顺序/强度与可选正向提示词（positive 输出）。";
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
        return;
    }
    fetchPresetsMap().then((map) => {
        const p = map[v];
        const pos = sanitizePositive(p?.positive);
        if (pos) {
            const pv = pos.length > 120 ? pos.slice(0, 120) + "…" : pos;
            widget.tooltip = `预设 "${v}" 的正向提示词（positive 输出）：${pv}`;
        } else {
            widget.tooltip = `预设 "${v}"（无 positive 提示词，仅 LoRA 顺序/强度）`;
        }
        // LiteGraph 的 widget tooltip 需重绘才可见
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "sfnodes.SFLoraPreset",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        const orig = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = orig?.apply(this, arguments);
            const w = this.widgets?.find((x) => x.name === "preset");
            if (w) refreshTooltip(this, w);
            return r;
        };
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;

        const widget = node.widgets?.find(w => w.name === "preset");
        if (!widget) return;

        // 刷新已保存预设列表（新增/删除预设后无需重载工作流）
        refreshCombo(node, widget);

        // 选择变化即刷新 tooltip 预览
        const origCb = widget.callback;
        widget.callback = function (v) {
            const res = origCb ? origCb.apply(this, arguments) : undefined;
            refreshTooltip(node, widget);
            return res;
        };

        const btn = node.addWidget("button", "\u21BB Refresh", null, () => {
            refreshCombo(node, widget);
        });
        btn.serialize = false;
    },
});
