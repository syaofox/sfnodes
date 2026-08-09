// ==========================================================================
// SF LoRA Stack - 节点主体全部事件。widget 元素上一组委托监听器，按 render
// 模块盖的 data-act 属性分发。`refresh(structural)`（来自主扩展）重渲染并在
// structural 时重适配节点高度。
// ==========================================================================
import {
    readState, patchLora, addLora, removeLora, duplicateLora, moveLora,
    setAllOn, countOn, accentOf, MAX_LORAS,
} from "./sf_lora_stack_core.js";
import { openLoraDropdown } from "./sf_lora_stack_dropdown.js";
import { openInfoPanel } from "./sf_lora_stack_info.js";
import { openLoraPanel } from "./sf_lora_stack_settings.js";

let _menu = null;
let _menuCleanup = null;

function closeRowMenu() {
    if (_menuCleanup) { try { _menuCleanup(); } catch { /* 忽略 */ } }
    _menuCleanup = null;
    if (_menu) { try { _menu.remove(); } catch { /* 忽略 */ } }
    _menu = null;
}

function injectMenuCSS() {
    if (document.getElementById("sf-ls-menu-css")) return;
    const s = document.createElement("style");
    s.id = "sf-ls-menu-css";
    s.textContent = `
    .sf-ls-menu { position:fixed; z-index:10030; width:168px; background:#2b2b2b; border:1px solid #4a4a4a;
      border-radius:8px; box-shadow:0 12px 34px rgba(0,0,0,0.65); overflow:hidden;
      font:12px 'Segoe UI',system-ui,sans-serif; color:#e0e0e0; padding:3px 0; }
    .sf-ls-menu .it { display:flex; align-items:center; gap:9px; padding:7px 12px; cursor:pointer; }
    .sf-ls-menu .it .k { width:14px; text-align:center; color:#8a8a8a; }
    .sf-ls-menu .it:hover { background:var(--acc,#f66744); color:#fff; } .sf-ls-menu .it:hover .k { color:#fff; }
    .sf-ls-menu .it.danger:hover { background:#e2504a; }
    .sf-ls-menu .it.dis { opacity:.35; pointer-events:none; }
    .sf-ls-menu .sep { height:1px; background:#1b1b1b; margin:3px 0; }
  `;
    document.head.appendChild(s);
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
    const item = (k, label, cb, { danger = false, dis = false } = {}) => {
        const it = document.createElement("div");
        it.className = "it" + (danger ? " danger" : "") + (dis ? " dis" : "");
        const ks = document.createElement("span"); ks.className = "k"; ks.textContent = k;
        const ls = document.createElement("span"); ls.textContent = label;
        it.append(ks, ls);
        if (!dis) it.addEventListener("click", () => { closeRowMenu(); cb(); });
        return it;
    };
    const sep = () => { const d = document.createElement("div"); d.className = "sep"; return d; };

    menu.append(
        item("i", "More info", () => openInfoPanel(node, id, refresh)),
        sep(),
        item("↑", "Move up", () => { moveLora(node, id, -1); refresh(true); }, { dis: idx === 0 }),
        item("↓", "Move down", () => { moveLora(node, id, +1); refresh(true); }, { dis: idx === st.loras.length - 1 }),
        item("⧉", "Duplicate", () => { duplicateLora(node, id); refresh(true); },
            { dis: st.loras.length >= MAX_LORAS }),
        item(e.on ? "◉" : "○", e.on ? "Disable" : "Enable",
            () => {
                const cur = readState(node).loras.find((x) => x.id === id); // 点击时重读
                patchLora(node, id, { on: !cur?.on });
                refresh(false);
            }),
        sep(),
        item("⌫", "Remove", () => { removeLora(node, id); refresh(true); }, { danger: true }),
    );

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
