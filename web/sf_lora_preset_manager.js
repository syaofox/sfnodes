// SF LoRA Preset Manager - 独立大面板（栈与预设节点共用）
// 搜预设名 + 关联 LoRA 文件名，高亮命中，n/total，底色高亮当前项
import { app } from "/scripts/app.js";
import { attachPopupDismiss, clampToViewport } from "./sf_popup.js";
import { el, injectCSSOnce, installWheelZoomPassthrough } from "./sf_common.js";
import { filterPresets, highlight } from "./sf_lora_preset_filter.js";
import { loadPresets, deletePreset, renamePreset } from "./sf_lora_stack_api.js";
import { confirmDialog } from "./sf_lora_stack_info.js";
import { sanitizePositive } from "./sf_lora_stack_core.js";

function canvasScale() {
    try { return app.canvas?.ds?.scale ?? 1; } catch { return 1; }
}

function injectCSS() {
    injectCSSOnce("sf-lora-preset-manager-css", `
    .sf-lpm-overlay{position:fixed;inset:0;z-index:10020;background:rgba(0,0,0,0.45);display:flex;align-items:center;justify-content:center;}
    .sf-lpm{width:560px;max-width:92vw;max-height:78vh;background:#222;border:1px solid #444;border-radius:10px;box-shadow:0 12px 40px rgba(0,0,0,0.6);display:flex;flex-direction:column;overflow:hidden;}
    .sf-lpm-head{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid #333;background:#1e1e1e;}
    .sf-lpm-title{font:600 13px 'Segoe UI',sans-serif;color:var(--acc,var(--sf-acc,#f66744));}
    .sf-lpm-x{cursor:pointer;color:#999;padding:0 6px;font-size:14px;}
    .sf-lpm-x:hover{color:#fff;}
    .sf-lpm-search{display:flex;align-items:center;gap:8px;padding:8px 12px;border-bottom:1px solid #2a2a2a;background:#1a1a1a;}
    .sf-lpm-search input{flex:1;min-width:0;background:#111;border:1px solid #444;border-radius:6px;color:#eee;padding:6px 8px;font:12px 'Segoe UI',sans-serif;outline:none;}
    .sf-lpm-search input:focus{border-color:var(--acc,var(--sf-acc,#f66744));}
    .sf-lpm-cnt{font:11px 'Segoe UI',sans-serif;color:#888;white-space:nowrap;}
    .sf-lpm-clear{cursor:pointer;color:#aaa;font-size:11px;padding:4px 8px;border-radius:4px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);}
    .sf-lpm-clear:hover{color:#fff;border-color:var(--acc,var(--sf-acc,#f66744));}
    .sf-lpm-save{padding:8px 12px;border-bottom:1px solid #2a2a2a;background:#1e1e1e;display:flex;flex-direction:column;gap:8px;}
    .sf-lpm-save-head{display:flex;align-items:center;justify-content:space-between;}
    .sf-lpm-save-title{font:600 12px 'Segoe UI',sans-serif;color:#ccc;}
    .sf-lpm-save-form{display:flex;flex-direction:column;gap:8px;}
    .sf-lpm-save-row{display:flex;gap:8px;align-items:center;}
    .sf-lpm-save-row input{flex:1;}
    .sf-lpm-autocomplete{max-height:120px;overflow-y:auto;border:1px solid #444;border-radius:6px;background:#111;margin-top:4px;display:none;}
    .sf-lpm-autocomplete.show{display:block;}
    .sf-lpm-autocomplete div{padding:6px 8px;cursor:pointer;font:12px 'Segoe UI',sans-serif;color:#ccc;}
    .sf-lpm-autocomplete div:hover{background:rgba(255,255,255,0.08);color:#fff;}
    .sf-lpm-autocomplete div.active{background:color-mix(in srgb, var(--acc,var(--sf-acc,#f66744)) 18%, #111);color:#fff;}
    .sf-lpm-list{flex:1;overflow-y:auto;padding:8px 10px;display:flex;flex-direction:column;gap:6px;}
    .sf-lpm-row{border:1px solid #3a3a3a;border-radius:8px;padding:8px 10px;display:flex;gap:10px;align-items:flex-start;background:#1e1e1e;cursor:pointer;}
    .sf-lpm-row:hover{border-color:#555;background:#252525;}
    .sf-lpm-row.active{border-color:var(--acc,var(--sf-acc,#f66744));background:color-mix(in srgb, var(--acc,var(--sf-acc,#f66744)) 18%, #1e1e1e);}
    .sf-lpm-info{flex:1;min-width:0;}
    .sf-lpm-name{font:600 12px 'Segoe UI',sans-serif;color:#e8e8e8;word-break:break-word;}
    .sf-lpm-name mark{background:color-mix(in srgb, var(--acc,var(--sf-acc,#f66744)) 35%, transparent);color:#fff;border-radius:2px;padding:0 2px;}
    .sf-lpm-loras{font:11px 'Segoe UI',sans-serif;color:#aaa;margin-top:4px;word-break:break-word;white-space:pre-wrap;}
    .sf-lpm-loras mark{background:color-mix(in srgb, var(--acc,var(--sf-acc,#f66744)) 35%, transparent);color:#fff;border-radius:2px;padding:0 2px;}
    .sf-lpm-positive{font:11px 'Segoe UI',sans-serif;color:#7a9a7a;margin-top:4px;white-space:pre-wrap;word-break:break-word;max-height:60px;overflow:hidden;}
    .sf-lpm-ops{display:flex;gap:6px;flex-shrink:0;align-items:center;}
    .sf-lpm-btn{cursor:pointer;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);color:#ccc;border-radius:5px;padding:4px 8px;font:11px 'Segoe UI',sans-serif;}
    .sf-lpm-btn:hover{border-color:var(--acc,var(--sf-acc,#f66744));color:#fff;}
    .sf-lpm-btn.edit{color:#8cc8ff;border-color:rgba(70,130,220,0.3);background:rgba(70,130,220,0.14);}
    .sf-lpm-btn.edit:hover{background:rgba(70,130,220,0.22);}
    .sf-lpm-btn.del{color:#ff9a8a;border-color:rgba(220,70,50,0.35);background:rgba(220,70,50,0.14);}
    .sf-lpm-btn.del:hover{background:rgba(220,70,50,0.22);}
    .sf-lpm-empty{color:#777;text-align:center;padding:20px;font:12px 'Segoe UI',sans-serif;}
    .sf-lpm-foot{display:flex;justify-content:flex-end;padding:8px 12px;border-top:1px solid #333;background:#1e1e1e;}
    .sf-lpm-foot .close{cursor:pointer;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);color:#ccc;border-radius:5px;padding:5px 10px;font:11px 'Segoe UI',sans-serif;}
    .sf-lpm-foot .close:hover{border-color:var(--acc,var(--sf-acc,#f66744));color:#fff;}
    .sf-lpm-form{padding:12px;display:flex;flex-direction:column;gap:8px;}
    .sf-lpm-form label{font:11px 'Segoe UI',sans-serif;color:#aaa;}
    .sf-lpm-form input,.sf-lpm-form textarea{width:100%;box-sizing:border-box;background:#111;border:1px solid #444;border-radius:6px;color:#eee;padding:6px 8px;font:12px 'Segoe UI',sans-serif;outline:none;}
    .sf-lpm-form input:focus,.sf-lpm-form textarea:focus{border-color:var(--acc,var(--sf-acc,#f66744));}
    .sf-lpm-form textarea{min-height:80px;resize:vertical;}
    .sf-lpm-form .hint{font:11px 'Segoe UI',sans-serif;color:#666;}
    .sf-lpm-form .acts{display:flex;gap:8px;justify-content:flex-end;margin-top:4px;}
    `);
}

function displayLoraName(full) {
    // 去目录，只取文件名，去扩展名前的 basename 更易搜
    const base = String(full || "").split("/").pop().split("\\").pop();
    return base;
}

export async function openLoraPresetManager(ctx = {}) {
    // ctx: { anchor?, node?, widget?, getActive?:()=>string, onSelect?:(name)=>void, presets?:object }
    injectCSS();
    // 若已存在则先移除
    const old = document.getElementById("sf-lpm-overlay");
    if (old) old.remove();

    const overlay = el("div", "sf-lpm-overlay");
    overlay.id = "sf-lpm-overlay";
    const panel = el("div", "sf-lpm");
    panel.style.setProperty("--acc", "var(--sf-acc, #f66744)");
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    // 头部
    const head = el("div", "sf-lpm-head");
    head.append(el("span", "sf-lpm-title", "Preset Manager"));
    const x = el("span", "sf-lpm-x", "✕");
    x.title = "Close";
    x.addEventListener("click", () => overlay.remove());
    head.appendChild(x);
    panel.appendChild(head);

    // 搜索区
    const searchBar = el("div", "sf-lpm-search");
    const inp = el("input", "");
    inp.type = "text";
    inp.placeholder = "Search presets… (name / lora)";
    inp.autocomplete = "off";
    inp.spellcheck = false;
    installWheelZoomPassthrough(inp);
    const cnt = el("span", "sf-lpm-cnt", "");
    const clear = el("span", "sf-lpm-clear", "Clear");
    clear.title = "Clear search";
    clear.addEventListener("click", () => { inp.value = ""; inp.dispatchEvent(new Event("input")); inp.focus(); });
    searchBar.append(inp, cnt, clear);
    panel.appendChild(searchBar);

    // 保存当前为预设（仅栈上下文，canSave !== false 且有 node）
    const canSave = ctx.canSave !== false && !!ctx.node;
    let saveBar = null;
    let saveNameInp = null;
    let savePosTa = null;
    let saveAuto = null;
    if (canSave) {
        saveBar = el("div", "sf-lpm-save");
        const saveHead = el("div", "sf-lpm-save-head");
        saveHead.append(el("span", "sf-lpm-save-title", "Save current as preset"));
        const saveToggle = el("span", "sf-lpm-btn", "Save");
        saveToggle.title = "Save current stack as preset";
        saveHead.appendChild(saveToggle);
        saveBar.appendChild(saveHead);
        const saveForm = el("div", "sf-lpm-save-form");
        saveForm.style.display = "none";
        const row = el("div", "sf-lpm-save-row");
        saveNameInp = el("input", "");
        saveNameInp.type = "text";
        saveNameInp.placeholder = "Preset name…";
        saveNameInp.maxLength = 64;
        saveNameInp.autocomplete = "off";
        installWheelZoomPassthrough(saveNameInp);
        row.appendChild(saveNameInp);
        const saveBtn = el("span", "sf-lpm-btn edit", "Save");
        const cancelBtn = el("span", "sf-lpm-btn", "Cancel");
        row.append(saveBtn, cancelBtn);
        saveForm.appendChild(row);
        saveAuto = el("div", "sf-lpm-autocomplete");
        saveForm.appendChild(saveAuto);
        const posLabel = el("label", "", "Positive prompt (optional)");
        posLabel.style.fontSize = "11px";
        posLabel.style.color = "#aaa";
        savePosTa = el("textarea", "");
        savePosTa.placeholder = "masterpiece, 1girl, ...";
        savePosTa.maxLength = 8000;
        try {
            const { readState } = await import("./sf_lora_stack_core.js");
            const cur = readState(ctx.node);
            savePosTa.value = cur.positive || "";
        } catch {}
        installWheelZoomPassthrough(savePosTa);
        const hint = el("div", "hint", "Select an existing preset below to overwrite, or type a new name.");
        hint.style.fontSize = "11px";
        hint.style.color = "#666";
        saveForm.append(posLabel, savePosTa, hint);
        // 切换展开/收起
        saveToggle.addEventListener("click", () => {
            const isHidden = saveForm.style.display === "none";
            saveForm.style.display = isHidden ? "flex" : "none";
            if (isHidden) {
                saveNameInp.focus();
                // 预填当前 positive 已在上方
                updateAuto();
            }
        });
        const updateAuto = () => {
            const q = (saveNameInp.value || "").trim().toLowerCase();
            const names = Object.keys(allPresets).sort();
            const matches = q ? names.filter(n => n.toLowerCase().includes(q)) : names.slice(0, 8);
            saveAuto.textContent = "";
            if (!matches.length || !q) {
                saveAuto.classList.remove("show");
                return;
            }
            for (const nm of matches.slice(0, 8)) {
                const it = el("div", "");
                it.textContent = nm;
                if (nm.toLowerCase() === q) it.classList.add("active");
                it.addEventListener("click", () => {
                    saveNameInp.value = nm;
                    saveAuto.classList.remove("show");
                    // 预填该预设的 positive 供参考（但保存时仍以当前 textarea 值为准）
                    const p = allPresets[nm];
                    if (p && typeof p.positive === "string") {
                        // 不自动覆盖当前输入的 positive，保持用户当前编辑的
                    }
                    saveNameInp.focus();
                });
                saveAuto.appendChild(it);
            }
            saveAuto.classList.add("show");
        };
        saveNameInp.addEventListener("input", updateAuto);
        saveNameInp.addEventListener("focus", updateAuto);
        saveNameInp.addEventListener("blur", () => setTimeout(() => saveAuto.classList.remove("show"), 150));
        const doSave = async () => {
            const nm = saveNameInp.value.trim();
            if (!nm) return;
            let pos = "";
            try {
                const { sanitizePositive } = await import("./sf_lora_stack_core.js");
                pos = sanitizePositive(savePosTa.value);
            } catch { pos = (savePosTa.value || "").trim().slice(0, 8000); }
            let data;
            try {
                const { readState, rowsToPreset } = await import("./sf_lora_stack_core.js");
                const cur = readState(ctx.node);
                data = rowsToPreset(cur, pos);
            } catch { return; }
            if (!data.loras.length) {
                hint.textContent = "Nothing to save - add a LoRA first.";
                hint.style.color = "#c98a6a";
                return;
            }
            if (allPresets[nm]) {
                const ok = await confirmDialog({
                    title: "Replace preset?",
                    message: `A preset named "${nm}" already exists. Replace it?`,
                    okLabel: "Replace",
                    cancelLabel: "Cancel",
                    accent: "#f66744",
                });
                if (!ok) return;
            }
            const { savePreset } = await import("./sf_lora_stack_api.js");
            const r = await savePreset(nm, data);
            if (!r?.ok) {
                hint.textContent = r?.message || "Could not save.";
                hint.style.color = "#c98a6a";
                return;
            }
            allPresets[nm] = data;
            // 同步栈的 activePreset/positive
            try {
                const { readState: rs, writeState } = await import("./sf_lora_stack_core.js");
                const cur = rs(ctx.node);
                writeState(ctx.node, { ...cur, positive: pos, activePreset: nm });
                try { ctx.node.setDirtyCanvas?.(true, true); } catch {}
                if (typeof ctx.onSelect === "function") {
                    // 保持与 onSelect 一致，刷新外部
                    try { ctx.onSelect(nm); } catch {}
                }
            } catch {}
            saveForm.style.display = "none";
            saveNameInp.value = "";
            savePosTa.value = "";
            try {
                const { readState: rs2 } = await import("./sf_lora_stack_core.js");
                try { savePosTa.value = rs2(ctx.node).positive || ""; } catch {}
            } catch {}
            render();
        };
        const doCancel = () => {
            saveForm.style.display = "none";
            saveNameInp.value = "";
            saveAuto.classList.remove("show");
        };
        saveBtn.addEventListener("click", doSave);
        cancelBtn.addEventListener("click", doCancel);
        saveNameInp.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Enter") { ev.preventDefault(); doSave(); }
            if (ev.key === "Escape") { ev.preventDefault(); doCancel(); }
        });
        savePosTa.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Escape") { ev.preventDefault(); doCancel(); }
        });
        saveBar.appendChild(saveForm);
        panel.appendChild(saveBar);
    }

    const list = el("div", "sf-lpm-list");
    panel.appendChild(list);
    const foot = el("div", "sf-lpm-foot");
    const closeBtn = el("span", "close", "Close");
    closeBtn.addEventListener("click", () => overlay.remove());
    foot.appendChild(closeBtn);
    panel.appendChild(foot);

    attachPopupDismiss(overlay, { exempt: (e) => panel.contains(e.target), onClose: () => overlay.remove() });

    let presets = ctx.presets || null;
    let allPresets = {};
    let q = "";
    let activeName = null;
    try {
        if (typeof ctx.getActive === "function") activeName = ctx.getActive();
        else if (ctx.widget) activeName = ctx.widget.value;
        else if (ctx.node) {
            try {
                const { readState } = await import("./sf_lora_stack_core.js");
                activeName = readState(ctx.node).activePreset;
            } catch {}
        }
    } catch {}

    async function load() {
        if (presets) {
            allPresets = presets;
            return;
        }
        try {
            const res = await loadPresets();
            allPresets = res.ok ? res.presets : {};
        } catch {
            allPresets = {};
        }
    }
    await load();

    let debounce = null;
    function scheduleRender() {
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(render, 150);
    }
    inp.addEventListener("input", scheduleRender);
    inp.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); inp.value = ""; q = ""; render(); }
        if (ev.key === "Enter" && Object.keys(filtered()).length === 1) {
            const sole = Object.keys(filtered())[0];
            handleSelect(sole);
        }
    });

    function filtered() {
        q = inp.value || "";
        return filterPresets(allPresets, q);
    }

    function handleSelect(name) {
        if (typeof ctx.onSelect === "function") ctx.onSelect(name);
        else if (ctx.widget) {
            ctx.widget.value = name;
            ctx.widget.callback?.(name);
            try { ctx.node?.setDirtyCanvas?.(true, true); } catch {}
        }
        overlay.remove();
    }

    async function handleEdit(oldName) {
        const old = allPresets[oldName];
        if (!old) return;
        // 进入编辑表单（复用面板内替换）
        const form = el("div", "sf-lpm-form");
        const nameLabel = el("label", "", "Preset name");
        const nameInp = el("input", "");
        nameInp.type = "text";
        nameInp.maxLength = 64;
        nameInp.value = oldName;
        installWheelZoomPassthrough(nameInp);
        const posLabel = el("label", "", "Positive prompt (optional)");
        const ta = el("textarea", "");
        ta.maxLength = 8000;
        ta.value = old.positive || "";
        ta.placeholder = "masterpiece, 1girl, ...";
        installWheelZoomPassthrough(ta);
        const hint = el("div", "hint", "Edit preset name and positive prompt.");
        const acts = el("div", "acts");
        const ok = el("span", "sf-lpm-btn edit", "Save");
        const no = el("span", "sf-lpm-btn", "Cancel");
        acts.append(no, ok);
        form.append(nameLabel, nameInp, posLabel, ta, hint, acts);
        list.textContent = "";
        list.appendChild(form);
        cnt.textContent = "";
        const doSave = async () => {
            const newName = nameInp.value.trim();
            if (!newName) return;
            const newPos = sanitizePositive(ta.value);
            if (newName !== oldName && allPresets[newName]) {
                // 简单提示
                hint.textContent = `A preset named "${newName}" already exists.`;
                hint.style.color = "#c98a6a";
                return;
            }
            const r = await renamePreset(oldName, newName, newPos);
            if (!r?.ok) {
                hint.textContent = r?.message || r?.error || "Could not save.";
                hint.style.color = "#c98a6a";
                return;
            }
            // 更新本地
            const updated = { ...old };
            if (newPos) updated.positive = newPos;
            else delete updated.positive;
            if (oldName !== newName) delete allPresets[oldName];
            allPresets[newName] = updated;
            // 同步调用方的 active 映射
            if (activeName === oldName) activeName = newName;
            if (ctx.widget && ctx.widget.value === oldName) {
                ctx.widget.value = newName;
                try { ctx.widget.callback?.(newName); } catch {}
            }
            if (ctx.node) {
                try {
                    const { readState, writeState } = await import("./sf_lora_stack_core.js");
                    const cur = readState(ctx.node);
                    if (cur.activePreset === oldName) {
                        writeState(ctx.node, { ...cur, activePreset: newName, positive: newPos });
                        try { ctx.node.setDirtyCanvas?.(true, true); } catch {}
                    }
                } catch {}
            }
            render();
        };
        const doCancel = () => render();
        ok.addEventListener("click", doSave);
        no.addEventListener("click", doCancel);
        nameInp.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Enter") { ev.preventDefault(); doSave(); }
            if (ev.key === "Escape") { ev.preventDefault(); doCancel(); }
        });
        ta.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Escape") { ev.preventDefault(); doCancel(); }
        });
        nameInp.focus();
        nameInp.select();
    }

    async function handleDelete(name) {
        const ok = await confirmDialog({
            title: "Delete preset?",
            message: `Delete preset "${name}"? This cannot be undone.`,
            okLabel: "Delete",
            cancelLabel: "Cancel",
            accent: "#f66744",
        });
        if (!ok) return;
        const r = await deletePreset(name);
        if (!r?.ok && r?.error) {
            // 显示错误
            const err = el("div", "sf-lpm-empty", r.message || "Could not delete.");
            err.style.color = "#c98a6a";
            list.prepend(err);
            setTimeout(() => err.remove(), 2500);
            return;
        }
        delete allPresets[name];
        if (activeName === name) activeName = null;
        if (ctx.widget && ctx.widget.value === name) {
            ctx.widget.value = "None";
            try { ctx.widget.callback?.("None"); } catch {}
        }
        if (ctx.node) {
            try {
                const { readState, writeState } = await import("./sf_lora_stack_core.js");
                const cur = readState(ctx.node);
                if (cur.activePreset === name) {
                    writeState(ctx.node, { ...cur, activePreset: "", positive: "" });
                    try { ctx.node.setDirtyCanvas?.(true, true); } catch {}
                }
            } catch {}
        }
        render();
    }

    function render() {
        q = inp.value || "";
        const filt = filtered();
        const names = Object.keys(filt).sort();
        const total = Object.keys(allPresets).length;
        cnt.textContent = `${names.length} / ${total}`;
        list.textContent = "";
        if (!names.length) {
            const empty = el("div", "sf-lpm-empty", q ? "(no matches)" : "(no presets yet)");
            list.appendChild(empty);
            return;
        }
        for (const nm of names) {
            const data = filt[nm];
            const row = el("div", "sf-lpm-row" + (nm === activeName ? " active" : ""));
            row.title = nm;
            const info = el("div", "sf-lpm-info");
            const nameEl = el("div", "sf-lpm-name");
            // 高亮 name
            nameEl.innerHTML = highlight(nm, q);
            info.appendChild(nameEl);
            const loraNames = (data.loras || []).map((l) => displayLoraName(l.lora)).join(", ");
            if (loraNames) {
                const lEl = el("div", "sf-lpm-loras");
                // 对 lora 名做高亮（需要对每个 lora 名单独高亮后 join）
                const parts = (data.loras || []).map((l) => {
                    const dn = displayLoraName(l.lora);
                    return highlight(dn, q);
                });
                lEl.innerHTML = parts.join(", ");
                lEl.title = (data.loras || []).map((l) => l.lora).join(", ");
                info.appendChild(lEl);
            }
            if (data.positive) {
                const p = el("div", "sf-lpm-positive");
                p.textContent = data.positive.length > 120 ? data.positive.slice(0, 120) + "…" : data.positive;
                p.title = data.positive;
                info.appendChild(p);
            }
            row.appendChild(info);
            const ops = el("div", "sf-lpm-ops");
            const edit = el("span", "sf-lpm-btn edit", "Edit");
            edit.title = "Edit (rename / positive)";
            edit.addEventListener("click", (ev) => { ev.stopPropagation(); handleEdit(nm); });
            const del = el("span", "sf-lpm-btn del", "Delete");
            del.title = "Delete";
            del.addEventListener("click", (ev) => { ev.stopPropagation(); handleDelete(nm); });
            ops.append(edit, del);
            row.appendChild(ops);
            row.addEventListener("click", () => handleSelect(nm));
            list.appendChild(row);
        }
        // 滚动当前项可见
        try {
            const doScroll = () => {
                const a = list.querySelector(".sf-lpm-row.active");
                if (a?.scrollIntoView) a.scrollIntoView({ block: "nearest" });
            };
            if (typeof requestAnimationFrame === "function") requestAnimationFrame(doScroll);
            else setTimeout(doScroll, 0);
        } catch {}
    }

    render();
    clampToViewport(panel, { scale: canvasScale() });
    inp.focus();
    return { overlay, panel, render };
}
