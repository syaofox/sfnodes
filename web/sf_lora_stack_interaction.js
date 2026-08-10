// ==========================================================================
// SF LoRA Stack - 节点主体全部事件。widget 元素上一组委托监听器，按 render
// 模块盖的 data-act 属性分发。`refresh(structural)`（来自主扩展）重渲染并在
// structural 时重适配节点高度。
// ==========================================================================
import {
    readState, writeState, patchLora, addLora, removeLora, duplicateLora, moveLora,
    setAllOn, countOn, accentOf, MAX_LORAS, rowsToPreset, presetToRows,
} from "./sf_lora_stack_core.js";
import { openLoraDropdown } from "./sf_lora_stack_dropdown.js";
import { openInfoPanel, confirmDialog } from "./sf_lora_stack_info.js";
import { openLoraPanel } from "./sf_lora_stack_settings.js";
import { loadPresets, savePreset, deletePreset } from "./sf_lora_stack_api.js";

let _menu = null;
let _menuCleanup = null;

function closeRowMenu() {
    if (_menuCleanup) { try { _menuCleanup(); } catch { /* 忽略 */ } }
    _menuCleanup = null;
    if (_menu) { try { _menu.remove(); } catch { /* 忽略 */ } }
    _menu = null;
}

export function injectMenuCSS() {
    if (document.getElementById("sf-ls-menu-css")) return;
    const s = document.createElement("style");
    s.id = "sf-ls-menu-css";
    s.textContent = `
    .sf-ls-menu { position:fixed; z-index:10030; min-width:178px; max-width:300px;
      background:#2b2b2b; border:1px solid #4a4a4a; border-radius:8px;
      box-shadow:0 12px 34px rgba(0,0,0,0.65); overflow:hidden;
      font:12px 'Segoe UI',system-ui,sans-serif; color:#e0e0e0; padding:3px 0; }
    .sf-ls-menu .it { display:flex; align-items:center; gap:9px; padding:7px 12px; cursor:pointer; }
    .sf-ls-menu .it .k { width:14px; text-align:center; color:#8a8a8a; flex:none; }
    .sf-ls-menu .it .l { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-ls-menu .it:hover { background:var(--acc, var(--sf-acc, #f66744)); color:#fff; } .sf-ls-menu .it:hover .k { color:#fff; }
    .sf-ls-menu .it.danger:hover { background:#e2504a; }
    .sf-ls-menu .it.dis { opacity:.35; pointer-events:none; }
    .sf-ls-menu .sep { height:1px; background:#1b1b1b; margin:3px 0; }
    /* 预设菜单：行尾删除 ✕（hover 显示）、保存命名输入行、状态消息 */
    .sf-ls-menu .del { margin-left:auto; flex:none; color:#c9736a; opacity:0; padding:0 2px;
      font-size:10px; cursor:pointer; }
    .sf-ls-menu .it:hover .del { opacity:1; }
    .sf-ls-menu .it .del:hover { color:#fff; }
    .sf-ls-menu .in { display:flex; align-items:center; gap:6px; padding:6px 8px; }
    .sf-ls-menu .in input { flex:1; min-width:0; box-sizing:border-box; background:#161616;
      border:1px solid #4a4a4a; border-radius:5px; color:#fff; font:11px 'Segoe UI',sans-serif;
      padding:5px 7px; outline:none; }
    .sf-ls-menu .in input:focus { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-menu .in .ok { flex:0 0 auto; padding:4px 8px; border-radius:4px; font-size:11px;
      color:#ccc; cursor:pointer; user-select:none; }
    .sf-ls-menu .in .ok:hover { color:#fff; background:rgba(255,255,255,0.08); }
    .sf-ls-menu .in .ok.pri { background:var(--acc, var(--sf-acc, #f66744)); color:#fff; font-weight:600; }
    .sf-ls-menu .in .ok.pri:hover { filter:brightness(1.1); }
    .sf-ls-menu .msg { padding:6px 12px; font-size:11px; color:#c98a6a; }
  `;
    document.head.appendChild(s);
}

// 菜单条目（共享：行菜单 + 预设菜单）。点击后先关菜单再回调。
export function makeMenuItem(k, label, cb, { danger = false, dis = false } = {}) {
    const it = document.createElement("div");
    it.className = "it" + (danger ? " danger" : "") + (dis ? " dis" : "");
    const ks = document.createElement("span"); ks.className = "k"; ks.textContent = k;
    const ls = document.createElement("span"); ls.className = "l"; ls.textContent = label;
    it.append(ks, ls);
    if (!dis && cb) it.addEventListener("click", () => { closeRowMenu(); cb(); });
    return it;
}

export function menuSep() { const d = document.createElement("div"); d.className = "sep"; return d; }

// 把菜单挂到 body：fixed 定位在点击处（越界钳制）+ 外部点击/Esc 关闭。
// 调用方负责先 closeRowMenu()（本函数假设 _menu 已被清空）。
export function showMenu(menu, x, y) {
    document.body.appendChild(menu);
    const mw = menu.offsetWidth, mh = menu.offsetHeight;
    menu.style.left = Math.max(6, Math.min(x, window.innerWidth - mw - 6)) + "px";
    menu.style.top = Math.max(6, Math.min(y, window.innerHeight - mh - 6)) + "px";

    const onDown = (ev) => { if (!menu.contains(ev.target)) closeRowMenu(); };
    const onKey = (ev) => { if (ev.key === "Escape") closeRowMenu(); };
    _menu = menu; // 先于定时器赋值，让下方守卫能看到
    setTimeout(() => {
        // 与其他弹窗同款守卫：菜单若在同一 tick 被关闭/替换，_menuCleanup
        // 已跑过（无害的）移除，现在挂监听会把捕获期监听器永远滞留——滞留
        // 的 onDown 会在每次后续行菜单的点击前用 pointerdown 关掉它，让条目
        // 点不动。
        if (_menu !== menu) return;
        document.addEventListener("pointerdown", onDown, true);
        document.addEventListener("keydown", onKey, true);
    }, 0);
    _menuCleanup = () => {
        document.removeEventListener("pointerdown", onDown, true);
        document.removeEventListener("keydown", onKey, true);
    };
}

function openRowMenu(node, id, x, y, refresh) {
    closeRowMenu();
    injectMenuCSS();
    const st = readState(node);
    const idx = st.loras.findIndex((e) => e.id === id);
    if (idx < 0) return;
    const e = st.loras[idx];

    const menu = document.createElement("div");
    menu.className = "sf-ls-menu";
    // 菜单 fixed 定位在 <body> 上，不继承任何东西——显式把本节点强调色交给
    // 它，否则 hover 保持品牌橙。
    menu.style.setProperty("--acc", accentOf(node));

    menu.append(
        makeMenuItem("i", "More info", () => openInfoPanel(node, id, refresh)),
        menuSep(),
        makeMenuItem("↑", "Move up", () => { moveLora(node, id, -1); refresh(true); }, { dis: idx === 0 }),
        makeMenuItem("↓", "Move down", () => { moveLora(node, id, +1); refresh(true); }, { dis: idx === st.loras.length - 1 }),
        makeMenuItem("⧉", "Duplicate", () => { duplicateLora(node, id); refresh(true); },
            { dis: st.loras.length >= MAX_LORAS }),
        makeMenuItem(e.on ? "◉" : "○", e.on ? "Disable" : "Enable",
            () => {
                const cur = readState(node).loras.find((x) => x.id === id); // 点击时重读
                patchLora(node, id, { on: !cur?.on });
                refresh(false);
            }),
        menuSep(),
        makeMenuItem("⌫", "Remove", () => { removeLora(node, id); refresh(true); }, { danger: true }),
    );

    showMenu(menu, x, y);
}

// ── 预设菜单（参考 SFPowerLoraLoader：存/取整个栈，机器级存储）────────────
// 与行菜单同 DOM 骨架（.sf-ls-menu），共享 closeRowMenu/showMenu。列表异步
// 加载（GET 预设），失败显示占位消息。保存命名在菜单内联输入（无
// app.canvas.prompt 依赖，Vue/Classic 双环境可用）。
async function openPresetsMenu(node, x, y, refresh) {
    closeRowMenu();
    injectMenuCSS();
    const menu = document.createElement("div");
    menu.className = "sf-ls-menu";
    menu.style.setProperty("--acc", accentOf(node));
    let presets = {};
    let msg = "";

    menu.appendChild(makeMenuItem("⏳", "Loading presets…", null, { dis: true }));
    showMenu(menu, x, y);
    const res = await loadPresets();
    if (!menu.isConnected) return;
    if (res.ok) presets = res.presets;
    else msg = res.message || "Could not load presets.";
    renderPresetsMenu();

    function renderPresetsMenu(extraMsg) {
        if (extraMsg) msg = extraMsg;
        menu.textContent = "";
        menu.append(
            makeMenuItem("💾", "Save current as preset…", enterPresetName),
            menuSep(),
        );
        if (msg) menu.appendChild(makeMenuItem("", msg, null, { dis: true }));
        const names = Object.keys(presets).sort();
        if (!names.length) {
            menu.appendChild(makeMenuItem("", "(no presets yet)", null, { dis: true }));
            return;
        }
        for (const nm of names) {
            const it = makeMenuItem("📚", nm, () => applyPreset(nm));
            const del = document.createElement("span");
            del.className = "del";
            del.textContent = "✕";
            del.title = "Delete this preset";
            del.addEventListener("click", (ev) => {
                ev.stopPropagation();
                deletePreset(nm).then(() => {
                    delete presets[nm];
                    if (menu.isConnected) renderPresetsMenu();
                });
            });
            it.appendChild(del);
            menu.appendChild(it);
        }
    }

    // 保存命名输入模式：菜单内容换成 input + Save/Cancel。Enter 提交、
    // Esc 取消。同设置面板 key 编辑器的交互。
    function enterPresetName() {
        menu.textContent = "";
        const row = document.createElement("div");
        row.className = "in";
        const inp = document.createElement("input");
        inp.type = "text";
        inp.placeholder = "Preset name…";
        inp.maxLength = 64;
        const ok = document.createElement("span");
        ok.className = "ok pri";
        ok.textContent = "Save";
        const no = document.createElement("span");
        no.className = "ok";
        no.textContent = "Cancel";
        const commit = async () => {
            const nm = inp.value.trim();
            if (!nm) return;
            const data = rowsToPreset(readState(node));
            if (!data.loras.length) {
                renderPresetsMenu("Nothing to save - add a LoRA first.");
                return;
            }
            if (presets[nm]) {
                const replace = await confirmDialog({
                    title: "Replace preset?",
                    message: `A preset named "${nm}" already exists. Replace it?`,
                    okLabel: "Replace",
                    cancelLabel: "Cancel",
                    accent: accentOf(node),
                });
                if (!replace) return;
            }
            const r = await savePreset(nm, data);
            if (!r?.ok) { renderPresetsMenu((r && r.message) || "Could not save."); return; }
            presets[nm] = data;
            msg = "";
            renderPresetsMenu();
        };
        const cancel = () => renderPresetsMenu();
        ok.addEventListener("click", commit);
        no.addEventListener("click", cancel);
        inp.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Enter") { ev.preventDefault(); commit(); }
            if (ev.key === "Escape") { ev.preventDefault(); cancel(); }
        });
        row.append(inp, ok, no);
        menu.appendChild(row);
        inp.focus();
        inp.select();
    }

    // 载入 = 替换整个栈（触发词随旧行丢弃，词本身在文件级存储，可重勾）。
    // 确认框防误点丢掉精心调好的配置。
    async function applyPreset(nm) {
        const preset = presets[nm];
        if (!preset) return;
        const st = readState(node);
        const rows = presetToRows(preset);
        if (!rows.length) {
            // 预设里没有合法行（如所有 lora 名为空/缺失）——菜单已关，静默
            // 关闭会让用户以为载入失败，这里至少留一条可查的日志。
            console.warn("[SF LoRA Stack] preset has no valid LoRA rows:", nm, JSON.stringify(preset).slice(0, 200));
            closeRowMenu();
            return;
        }
        const ok = await confirmDialog({
            title: "Load preset?",
            message: `Load "${nm}"? This replaces the current stack.`,
            okLabel: "Load",
            cancelLabel: "Cancel",
            accent: accentOf(node),
        });
        if (!ok) return;
        writeState(node, { ...st, loras: rows });
        refresh(true);
        closeRowMenu();
    }
}

function rowIdOf(target) {
    const row = target.closest?.(".sf-ls-row");
    return row?.dataset?.id || null;
}

function stepWeight(node, id, dir, which, refresh) {
    const st = readState(node);
    const e = st.loras.find((x) => x.id === id);
    if (!e) return;
    if (which === "c" && st.linkStrength) return; // 联动时 clip 跟随 model（防御）
    if (which === "c") patchLora(node, id, { sc: e.sc + dir * st.step });
    else patchLora(node, id, { sm: e.sm + dir * st.step });
    refresh(false);
}

export function attachInteractions(node, widgetEl, refresh) {
    attachDragSort(node, widgetEl, refresh);

    widgetEl.addEventListener("click", (ev) => {
        const t = ev.target;
        if (t?.dataset?.act === "wval" || t?.dataset?.act === "wcval") return; // 让权重框聚焦
        const act = t.closest?.("[data-act]")?.dataset?.act;
        if (!act) return;
        ev.stopPropagation();

        if (act === "add") {
            const res = addLora(node, "");
            refresh(true);
            // 立刻在新行上打开选择器，让添加-选择一键完成。
            if (res.ok) {
                requestAnimationFrame(() => {
                    const rowEl = widgetEl.querySelector(`.sf-ls-row[data-id="${res.state.loras[res.index].id}"] .sf-ls-name`);
                    if (rowEl) openNamePicker(node, res.state.loras[res.index].id, rowEl, refresh);
                });
            }
            return;
        }
        if (act === "allToggle") {
            const st = readState(node);
            setAllOn(node, !(st.loras.length && countOn(st) === st.loras.length));
            refresh(false);
            return;
        }
        if (act === "gear") { openLoraPanel(node, refresh); return; }
        if (act === "presets") { openPresetsMenu(node, ev.clientX, ev.clientY, refresh); return; }

        const id = rowIdOf(t);
        if (!id) return;
        if (act === "name") { openNamePicker(node, id, t.closest(".sf-ls-name"), refresh); return; }
        if (act === "info") { openInfoPanel(node, id, refresh); return; }
        if (act === "toggle") {
            const e = readState(node).loras.find((x) => x.id === id);
            patchLora(node, id, { on: !e?.on });
            refresh(false);
            return;
        }
        if (act === "winc") { stepWeight(node, id, +1, "m", refresh); return; }
        if (act === "wdec") { stepWeight(node, id, -1, "m", refresh); return; }
        if (act === "wcinc") { stepWeight(node, id, +1, "c", refresh); return; }
        if (act === "wcdec") { stepWeight(node, id, -1, "c", refresh); return; }
    });

    // 输入的权重在 change（blur / Enter）时提交。
    widgetEl.addEventListener("change", (ev) => {
        const act = ev.target?.dataset?.act;
        if (act !== "wval" && act !== "wcval") return;
        const id = rowIdOf(ev.target);
        if (!id) return;
        const raw = parseFloat(ev.target.value);
        if (!Number.isFinite(raw)) { refresh(false); return; } // 垃圾输入 -> 回到存储值
        if (act === "wcval" && readState(node).linkStrength) { refresh(false); return; } // 联动：clip 跟随 model
        patchLora(node, id, act === "wcval" ? { sc: raw } : { sm: raw });
        refresh(false);
    });

    // 聚焦权重框 -> 全选文本便于快速覆盖。
    widgetEl.addEventListener("focusin", (ev) => {
        const act = ev.target?.dataset?.act;
        if (act === "wval" || act === "wcval") ev.target.select?.();
    });

    // 权重框内打字不触发画布快捷键；Enter 提交。
    widgetEl.addEventListener("keydown", (ev) => {
        const act = ev.target?.dataset?.act;
        if (act !== "wval" && act !== "wcval") return;
        ev.stopPropagation();
        if (ev.key === "Enter") { ev.preventDefault(); ev.target.blur(); }
    });

    // 右键行 -> 行菜单。
    widgetEl.addEventListener("contextmenu", (ev) => {
        const id = rowIdOf(ev.target);
        if (!id) return;
        ev.preventDefault();
        ev.stopPropagation();
        openRowMenu(node, id, ev.clientX, ev.clientY, refresh);
    });
}

function openNamePicker(node, id, anchorEl, refresh) {
    const e = readState(node).loras.find((x) => x.id === id);
    openLoraDropdown(anchorEl, {
        current: e?.name || "",
        accent: accentOf(node),
        onPick: (name) => { patchLora(node, id, { name }); refresh(false); },
    });
}

// ── 拖拽排序（行左侧 ⋮ 手柄）───────────────────────────────────────────────
// 拖拽中只移动 DOM 行（即时视觉），绝不写 node.properties（避免每帧脏
// 标记）；pointerup 一次性提交 reorderLora + refresh(true)。document 级
// 监听：指针可移出行区域仍跟随。行元素若在拖拽中被 renderNode 重建
// （异步重绘），isConnected 检查放弃提交。

let _drag = null;   // { node, row, rows, from }

function dragIndex(row) {
    return row.parentElement ? [...row.parentElement.children].indexOf(row) : -1;
}

function clearDragMarks(rows) {
    for (const c of rows.children) {
        c.classList.remove("drag-before", "drag-after");
    }
}

function attachDragSort(node, widgetEl, refresh) {
    widgetEl.addEventListener("pointerdown", (ev) => {
        const grip = ev.target.closest?.(".sf-ls-grip");
        if (!grip) return;
        if (_drag) return;                     // 已在拖拽中
        const row = grip.closest(".sf-ls-row");
        if (!row) return;
        ev.preventDefault();
        ev.stopPropagation();
        const rows = row.parentElement;
        _drag = { node, row, rows, from: dragIndex(row) };
        row.classList.add("dragging");

        const onMove = (e) => {
            const d = _drag;
            if (!d || !d.row.isConnected) return;
            const y = e.clientY;
            // 找插入锚点：目标行 rect 中点与指针比较，跨过中点即插到该行前。
            let anchor = null;
            for (const c of d.rows.children) {
                if (c === d.row) continue;
                const r = c.getBoundingClientRect();
                if (y < r.top + r.height / 2) { anchor = c; break; }
            }
            if (anchor && anchor !== d.row.nextSibling) d.rows.insertBefore(d.row, anchor);
            else if (!anchor && d.row.nextSibling) d.rows.appendChild(d.row);
            // 目标行高亮（插入位置视觉反馈）
            clearDragMarks(d.rows);
            if (anchor) {
                const prev = anchor.previousSibling;
                if (prev === d.row) anchor.classList.add("drag-before");
                else if (prev) prev.classList.add("drag-after");
            } else {
                const last = d.rows.lastElementChild;
                if (last && last !== d.row) last.classList.add("drag-after");
            }
        };

        const onUp = () => {
            const d = _drag;
            _drag = null;
            document.removeEventListener("pointermove", onMove, true);
            document.removeEventListener("pointerup", onUp, true);
            document.removeEventListener("pointercancel", onUp, true);
            if (!d) return;
            d.row.classList.remove("dragging");
            clearDragMarks(d.rows);
            if (!d.row.isConnected) return;    // 拖拽中被重建——放弃
            const to = dragIndex(d.row);
            if (to === d.from || to < 0) return;
            reorderLora(d.node, d.from, to);
            refresh(true);
        };

        document.addEventListener("pointermove", onMove, true);
        document.addEventListener("pointerup", onUp, true);
        document.addEventListener("pointercancel", onUp, true);
    });
}

export { closeRowMenu };

// ── preset 输入（SF_LORA_PRESET，来自 SF Power Lora Preset）───────────────
// 连接后自动把预设加载到行（"连接即刷新"）；上游切换预设名时跟随重载。
// 加载路径（工作流恢复连接）只 watch 不写状态——writeState 会把干净的工作流
// 标成 modified（Vue Compat #18）。断开保留已加载的行（用户可继续编辑）。
// 执行语义在 Python：preset 优先覆盖行，行状态仅继承同名行的触发词勾选。

const PRESET_SLOT = 2;   // inputs: [model, clip, preset]

export function presetUpstream(node) {
    const slot = node?.inputs?.[PRESET_SLOT];
    if (!slot || slot.link == null) return null;
    const link = node.graph?.links?.[slot.link];
    if (!link) return null;
    const up = node.graph?.getNodeById?.(link.origin_id);
    return up && (up.comfyClass === "SFPowerLoraPreset" || up.type === "SFPowerLoraPreset") ? up : null;
}

// 读上游 combo 名 -> fetch 预设 -> 加载到行并刷新。加载路径不执行。
export async function loadPresetInto(node, refresh) {
    const up = presetUpstream(node);
    if (!up) return false;
    const combo = up.widgets?.find((w) => w.name === "preset");
    const name = combo?.value;
    if (!name || name === "None") return false;
    const res = await loadPresets();
    if (!res.ok || !res.presets[name]) return false;
    const rows = presetToRows(res.presets[name]);
    if (!rows.length) return false;
    const st = readState(node);
    writeState(node, { ...st, loras: rows });
    if (refresh) refresh(true);
    return true;
}

// 包装上游 preset combo 的 callback：切换预设时自动重载。幂等（每个 combo
// 只包一次）。上游 configure 重建 widget 后包装丢失——主扩展在
// onAfterGraphConfigured 重新调用本函数补上。
export function watchPresetUpstream(node, refresh) {
    const up = presetUpstream(node);
    if (!up) return;
    const combo = up.widgets?.find((w) => w.name === "preset");
    if (!combo || combo._sfLsPresetWatched) return;
    combo._sfLsPresetWatched = true;
    const orig = combo.callback;
    combo.callback = function (v) {
        const r = orig ? orig.apply(this, arguments) : undefined;
        loadPresetInto(node, refresh);
        return r;
    };
}
