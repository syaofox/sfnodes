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

let _mgrMenu = null;
let _mgrCleanup = null;
function closeMgr() {
    if (_mgrCleanup) { try { _mgrCleanup(); } catch {} }
    _mgrCleanup = null;
    if (_mgrMenu) { try { _mgrMenu.remove(); } catch {} }
    _mgrMenu = null;
}
function showMgr(menu) {
    document.body.appendChild(menu);
    // 居中或靠近视口中央（无锚点坐标时）
    const mw = menu.offsetWidth || 360, mh = menu.offsetHeight || 200;
    menu.style.left = Math.max(6, (window.innerWidth - mw) / 2) + "px";
    menu.style.top = Math.max(6, (window.innerHeight - mh) / 2) + "px";
    menu.style.position = "fixed";
    menu.style.zIndex = "10030";
    const onDown = (ev) => { if (!menu.contains(ev.target)) closeMgr(); };
    const onKey = (ev) => { if (ev.key === "Escape") closeMgr(); };
    _mgrMenu = menu;
    setTimeout(() => {
        if (_mgrMenu !== menu) return;
        document.addEventListener("pointerdown", onDown, true);
        document.addEventListener("keydown", onKey, true);
    }, 0);
    _mgrCleanup = () => {
        document.removeEventListener("pointerdown", onDown, true);
        document.removeEventListener("keydown", onKey, true);
    };
}

async function openPresetManager(node, widget) {
    closeMgr();
    injectMgrCSS();
    // 复用与栈内一致的菜单样式（需先注入 .sf-ls-menu）
    const { injectMenuCSS } = await import("./sf_lora_stack_interaction.js");
    injectMenuCSS();
    const menu = document.createElement("div");
    menu.className = "sf-ls-menu";
    menu.style.minWidth = "320px";
    menu.style.maxWidth = "420px";
    menu.style.setProperty("--acc", "#f66744");
    const loading = document.createElement("div");
    loading.className = "it dis";
    loading.textContent = "⏳ Loading presets…";
    menu.appendChild(loading);
    showMgr(menu);
    const res = await loadPresets();
    if (!menu.isConnected) return;
    let presets = res.ok ? res.presets : {};
    let msg = res.ok ? "" : (res.message || "Could not load presets.");
    render();

    function render(extraMsg) {
        if (extraMsg) msg = extraMsg;
        menu.textContent = "";
        if (msg) {
            const m = document.createElement("div");
            m.className = "msg";
            m.textContent = msg;
            m.style.padding = "6px 12px";
            m.style.color = "#c98a6a";
            m.style.fontSize = "11px";
            menu.appendChild(m);
        }
        const names = Object.keys(presets).sort();
        if (!names.length) {
            const it = document.createElement("div");
            it.className = "it dis";
            it.textContent = "(no presets yet)";
            menu.appendChild(it);
            const foot = document.createElement("div");
            foot.className = "it";
            foot.textContent = "Close";
            foot.style.justifyContent = "center";
            foot.addEventListener("click", closeMgr);
            menu.appendChild(foot);
            return;
        }
        for (const nm of names) {
            const it = document.createElement("div");
            it.className = "it";
            const k = document.createElement("span");
            k.className = "k";
            k.textContent = "📚";
            const l = document.createElement("span");
            l.className = "l";
            l.textContent = nm;
            it.append(k, l);
            const pos = sanitizePositive(presets[nm]?.positive);
            if (pos) {
                const pv = pos.length > 60 ? pos.slice(0, 60) + "…" : pos;
                const sub = document.createElement("span");
                sub.className = "sf-ls-preset-pos";
                sub.textContent = pv;
                sub.title = pos;
                l.classList.add("has-pos");
                it.appendChild(sub);
            }
            const edit = document.createElement("span");
            edit.className = "edit";
            edit.textContent = "✎";
            edit.title = "Edit (rename / positive)";
            edit.addEventListener("click", (ev) => {
                ev.stopPropagation();
                enterEdit(nm);
            });
            const del = document.createElement("span");
            del.className = "del";
            del.textContent = "✕";
            del.title = "Delete";
            del.addEventListener("click", async (ev) => {
                ev.stopPropagation();
                const ok = await confirmDialog({
                    title: "Delete preset?",
                    message: `Delete preset "${nm}"? This cannot be undone.`,
                    okLabel: "Delete",
                    cancelLabel: "Cancel",
                    accent: "#f66744",
                });
                if (!ok) return;
                const r = await deletePreset(nm);
                if (!r?.ok && r?.error) {
                    render(r.message || "Could not delete.");
                    return;
                }
                delete presets[nm];
                // 若当前选中被删，切回 None
                if (widget.value === nm) {
                    widget.value = "None";
                    refreshTooltip(node, widget);
                    node.setDirtyCanvas?.(true, true);
                }
                render();
            });
            it.append(edit, del);
            // 点击行选中该预设（仅预设节点）
            it.addEventListener("click", () => {
                widget.value = nm;
                widget.callback?.(nm);
                refreshTooltip(node, widget);
                node.setDirtyCanvas?.(true, true);
                closeMgr();
            });
            menu.appendChild(it);
        }
        const foot = document.createElement("div");
        foot.className = "it";
        foot.style.justifyContent = "center";
        foot.style.color = "#8a8a8a";
        foot.textContent = "Close";
        foot.addEventListener("click", closeMgr);
        menu.appendChild(foot);
    }

    function enterEdit(oldName) {
        const old = presets[oldName];
        if (!old) return;
        menu.textContent = "";
        const wrap = document.createElement("div");
        wrap.className = "sf-ls-save";
        wrap.style.padding = "6px 8px";
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.gap = "6px";
        row.style.alignItems = "center";
        const inp = document.createElement("input");
        inp.type = "text";
        inp.value = oldName;
        inp.maxLength = 64;
        inp.placeholder = "Preset name…";
        inp.style.flex = "1";
        inp.style.background = "#161616";
        inp.style.border = "1px solid #4a4a4a";
        inp.style.borderRadius = "5px";
        inp.style.color = "#fff";
        inp.style.padding = "5px 7px";
        inp.style.font = "11px 'Segoe UI',sans-serif";
        installWheelZoomPassthrough(inp);
        row.appendChild(inp);
        wrap.appendChild(row);
        const lab = document.createElement("div");
        lab.className = "lab";
        lab.textContent = "Positive prompt (optional)";
        const ta = document.createElement("textarea");
        ta.value = old.positive || "";
        ta.placeholder = "masterpiece, 1girl, ...";
        ta.maxLength = 8000;
        installWheelZoomPassthrough(ta);
        const hint = document.createElement("div");
        hint.className = "hint";
        hint.textContent = "Rename and/or edit positive prompt.";
        const acts = document.createElement("div");
        acts.style.display = "flex";
        acts.style.gap = "6px";
        acts.style.justifyContent = "flex-end";
        acts.style.marginTop = "6px";
        const ok = document.createElement("span");
        ok.textContent = "Save";
        ok.style.padding = "4px 8px";
        ok.style.borderRadius = "4px";
        ok.style.background = "#f66744";
        ok.style.color = "#fff";
        ok.style.cursor = "pointer";
        ok.style.fontSize = "11px";
        const no = document.createElement("span");
        no.textContent = "Cancel";
        no.style.padding = "4px 8px";
        no.style.cursor = "pointer";
        no.style.fontSize = "11px";
        no.style.color = "#ccc";
        acts.append(no, ok);
        wrap.append(lab, ta, hint, acts);
        menu.appendChild(wrap);
        const commit = async () => {
            const newName = inp.value.trim();
            if (!newName) return;
            const newPos = sanitizePositive(ta.value);
            if (newName !== oldName && presets[newName]) {
                render(`A preset named "${newName}" already exists.`);
                // 需重建编辑态，故重新进入
                setTimeout(() => enterEdit(oldName), 0);
                return;
            }
            const r = await renamePreset(oldName, newName, newPos);
            if (!r?.ok) {
                const m = r?.error === "already exists" ? `A preset named "${newName}" already exists.` : (r?.message || r?.error || "Could not save.");
                render(m);
                setTimeout(() => enterEdit(oldName), 0);
                return;
            }
            const updated = { ...old };
            if (newPos) updated.positive = newPos;
            else delete updated.positive;
            if (oldName !== newName) delete presets[oldName];
            presets[newName] = updated;
            // 同步 combo 选中
            if (widget.value === oldName) {
                widget.value = newName;
                // 重建 options 以含新名
                fetchPresetNames().then(names => {
                    if (names && widget.options) {
                        widget.options.values = names;
                        if (!names.includes(newName)) widget.options.values = ["None", ...Object.keys(presets).sort()];
                        widget.value = newName;
                    }
                    refreshTooltip(node, widget);
                });
            } else {
                // 刷新下拉列表
                fetchPresetNames().then(names => {
                    if (names && widget.options) widget.options.values = names;
                });
            }
            render();
        };
        const cancel = () => render();
        ok.addEventListener("click", commit);
        no.addEventListener("click", cancel);
        const kd = (ev) => {
            ev.stopPropagation();
            if (ev.key === "Escape") { ev.preventDefault(); cancel(); }
            if (ev.target === inp && ev.key === "Enter") { ev.preventDefault(); commit(); }
        };
        inp.addEventListener("keydown", kd);
        ta.addEventListener("keydown", kd);
        inp.focus();
        inp.select();
    }
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
        const mgr = node.addWidget("button", "⚙ Manage Presets", null, () => {
            openPresetManager(node, widget);
        });
        mgr.serialize = false;
    },
});
