// SF LoRA Preset - 预设选择节点：combo 刷新按钮（原 SF Power Lora Preset 改名）
// 2026-08 扩展：预设含可选 positive 提示词（与 triggers 分离），经额外 STRING 输出流通
// 2026-08 管理：独立 Manage 按钮 + 弹窗编辑/删除（改名+positive 原子化）
import { app } from "/scripts/app.js";
import { loadPresets, deletePreset, renamePreset } from "./sf_lora_stack_api.js";
import { confirmDialog } from "./sf_lora_stack_info.js";
import { injectCSSOnce, installWheelZoomPassthrough } from "./sf_common.js";

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

function injectMgrCSS() {
    injectCSSOnce("sf-lora-preset-mgr-css", `
    .sf-ls-menu .edit { flex:none; color:#8cc8ff; background:rgba(70,130,220,0.18); border:1px solid rgba(70,130,220,0.38); border-radius:4px; padding:2px 6px; font-size:10px; cursor:pointer; opacity:0.92; }
    .sf-ls-menu .it:hover .edit { opacity:1; background:rgba(70,130,220,0.28); border-color:rgba(70,130,220,0.55); }
    .sf-ls-menu .it .edit:hover { color:#fff; background:rgba(70,130,220,0.38); }
    .sf-ls-menu .del { margin-left:auto; flex:none; color:#ff9a8a; background:rgba(220,70,50,0.18); border:1px solid rgba(220,70,50,0.38); border-radius:4px; padding:2px 6px; font-size:10px; cursor:pointer; opacity:0.92; }
    .sf-ls-menu .it:hover .del { opacity:1; background:rgba(220,70,50,0.28); border-color:rgba(220,70,50,0.55); }
    .sf-ls-menu .it .del:hover { color:#fff; background:rgba(220,70,50,0.38); }
    .sf-ls-save { display:flex; flex-direction:column; gap:6px; padding:6px 8px; }
    .sf-ls-save .lab { font:10px 'Segoe UI'; color:#8a8a8a; letter-spacing:.04em; text-transform:uppercase; }
    .sf-ls-save textarea { width:100%; box-sizing:border-box; min-height:58px; max-height:120px; resize:vertical; background:#161616; border:1px solid #4a4a4a; border-radius:5px; color:#fff; font:11px 'Segoe UI',sans-serif; padding:5px 7px; outline:none; }
    .sf-ls-save textarea:focus { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-save .hint { font:10px 'Segoe UI'; color:#6f6f6f; }
    .sf-ls-preset-pos { flex:1; min-width:0; font:10px 'Segoe UI'; color:#7a9a7a; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-left:8px; }
    .sf-ls-menu .it .l.has-pos { flex:0 1 auto; }
    `);
}

async function openPresetManager(node, widget) {
    const { openLoraPresetManager } = await import("./sf_lora_preset_manager.js");
    return openLoraPresetManager({
        node,
        widget,
        canSave: false,
        getActive: () => widget.value,
        onSelect: (name) => {
            widget.value = name;
            try { widget.callback?.(name); } catch {}
            refreshTooltip(node, widget);
            try { node.setDirtyCanvas?.(true, true); } catch {}
        },
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
        const mgr = node.addWidget("button", "Manage Presets", null, () => {
            openPresetManager(node, widget);
        });
        mgr.serialize = false;
    },
});
