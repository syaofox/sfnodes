// ==========================================================================
// SF LoRA Plot - 主扩展 + 行 UI。一个 DOM widget（Add LoRA / 全部开关 +
// 每行 [开关][LoRA 名][强度][✕]），固定 MODEL + CLIP 输入和 MODEL + CLIP +
// metadata 列表输出。右键行：上移/下移/复制/开关/删除。
//
// 架构镜像 LoRA Stack（Vue Compat #9）：状态在 node.properties.loraStackState，
// 由下面 graphToPrompt 钩子注入隐藏 LoraLoaderState 输入。状态模型、LoRA
// 列表 API、选择器弹窗、行样式、菜单骨架全部复用 sf_lora_stack_* 模块
// （AGENTS.md 规则 14——零内联副本）；Python 侧与 SFLoraStack 共用同一
// parse_state 契约。
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    applyAdaptiveCanvasOnly,
    installCanvasZoomPassthrough,
    isGraphLoading,
    isVueNodes,
} from "./sf_common.js";
import { listLoras, invalidateList, hasLora } from "./sf_lora_stack_api.js";
import {
    HIDDEN_INPUT, DEFAULT_STATE, MAX_LORAS,
    readState, patchLora, addLora, removeLora, duplicateLora, moveLora,
    setAllOn, countOn, promptState, accentOf, loadDefaults,
} from "./sf_lora_stack_core.js";
import { injectCSS, weightBox, displayName, ROW_H } from "./sf_lora_stack_render.js";
import { openLoraDropdown, closeLoraDropdown } from "./sf_lora_stack_dropdown.js";
import {
    closeRowMenu, injectMenuCSS, makeMenuItem, menuSep, showMenu,
} from "./sf_lora_stack_interaction.js";
import { buildIndex, findNode, hideJsonWidget } from "./sf_lora_stack.js";

const CLASS = "SFLoraPlot";

const MIN_W = 300;
const CHROME = 66;      // legacy 回退：标题 + 输入槽 + 输出槽行
const VUE_CHROME = 96;  // Nodes 2.0 回退（槽带绝对定位，不占 widget 区高度）

// 高度常量与 sf_lora_stack_render.js 锁步（行高直接复用其 ROW_H）：
// band 单行（Add + All）+ 行列表。
const PAD = 9;
const BAND_H = 28;
const BAND_GAP = 5;
const ROW_GAP = 6;
const EMPTY_H = 46;

function widgetH(node) { return contentHeight(readState(node)); }

function contentHeight(state) {
    const n = state.loras.length;
    const rowsH = n ? n * ROW_H + (n - 1) * ROW_GAP : EMPTY_H;
    return PAD + BAND_H + BAND_GAP + rowsH + PAD;
}

// 无滚动条显示所有行所需的节点高度。chrome（标题 + 输入/输出槽行）委托给
// LiteGraph 的 computeSize；不可用时才退回常量估算。
function fitNodeH(node) {
    try {
        const cs = node.computeSize?.();
        if (cs && cs[1] > 0) return Math.round(cs[1]);
    } catch (_e) { /* 走回退 */ }
    return widgetH(node) + (isVueNodes() ? VUE_CHROME : CHROME);
}

// 节点高度贴合内容。仅用户动作（加载路径绝不执行，否则保存的尺寸被改写、
// 干净的工作流打开即 "modified"——Vue Compat #18）。宽度保留手调值。
function fitToContent(node) {
    if (isGraphLoading()) return;
    const w = Math.max(node.size?.[0] || MIN_W, MIN_W);
    const h = fitNodeH(node);
    if (node.setSize) node.setSize([w, h]);
    else node.size = [w, h];
}

function refreshNode(node, structural) {
    renderNode(node);
    if (structural) fitToContent(node);
    node.setDirtyCanvas?.(true, true);
}

// 重绘图中全部 Plot 节点（含子图嵌套）。R 键刷新列表后调用。
function renderAllPlots() {
    const walk = (g) => {
        for (const n of (g?._nodes || [])) {
            if ((n.comfyClass === CLASS || n.type === CLASS) && n._sfPlotRoot) renderNode(n);
            const sub = n.subgraph || n.graph || n._graph;
            if (sub && sub !== g) walk(sub);
        }
    };
    walk(app.graph);
}

// plot 专用 CSS（行主体全部复用 .sf-ls-*；这里只补 band 布局与 ✕ 删除钮）。
function injectPlotCSS() {
    if (document.getElementById("sf-plot-css")) return;
    const s = document.createElement("style");
    s.id = "sf-plot-css";
    s.textContent = `
    .sf-plot-band { display:flex; align-items:stretch; gap:6px; height:${BAND_H}px; }
    .sf-plot-band .sf-ls-add { flex:1; width:auto; }
    .sf-plot-del { flex:0 0 auto; width:22px; height:22px; border-radius:5px;
      border:1px solid rgba(255,255,255,0.12); background:rgba(255,255,255,0.05);
      color:#a8a8a8; cursor:pointer; display:flex; align-items:center;
      justify-content:center; font-size:10px; user-select:none; }
    .sf-plot-del:hover { border-color:#e2504a; color:#fff; background:rgba(226,80,74,0.15); }
  `;
    document.head.appendChild(s);
}

function ensureRoot(node) {
    const held = node._sfPlotRoot;
    if (held && held.isConnected) { node._sfPlotRootMounted = true; return held; }
    const w = (node.widgets || []).find((x) => x.name === "plot_ui");
    const el = w?.element;
    const elRoot = el?.classList?.contains?.("sf-ls-root") ? el : el?.querySelector?.(".sf-ls-root");
    if (elRoot) { node._sfPlotRoot = elRoot; node._sfPlotRootMounted = true; return elRoot; }
    return node._sfPlotRootMounted ? null : (held || null);
}

// ── 行构建（纯 DOM，事件直接绑定——行数少，无需委托分发）──────────────
function rowIdOf(target) {
    const row = target.closest?.(".sf-ls-row");
    return row?.dataset?.id || null;
}

function rowMenu(node, id, x, y) {
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
        makeMenuItem("↑", "Move up", () => { moveLora(node, id, -1); refreshNode(node, true); }, { dis: idx === 0 }),
        makeMenuItem("↓", "Move down", () => { moveLora(node, id, +1); refreshNode(node, true); }, { dis: idx === st.loras.length - 1 }),
        makeMenuItem("⧉", "Duplicate", () => { duplicateLora(node, id); refreshNode(node, true); },
            { dis: st.loras.length >= MAX_LORAS }),
        makeMenuItem(e.on ? "◉" : "○", e.on ? "Disable" : "Enable",
            () => { patchLora(node, id, { on: !e.on }); refreshNode(node, false); }),
        menuSep(),
        makeMenuItem("⌫", "Remove", () => { removeLora(node, id); refreshNode(node, true); }, { danger: true }),
    );
    showMenu(menu, x, y);
}

function renderNode(node) {
    const root = ensureRoot(node);
    if (!root) return;
    let inner = root.querySelector(".sf-ls-inner");
    if (!inner) {
        inner = document.createElement("div");
        inner.className = "sf-ls-inner";
        root.appendChild(inner);
    }
    inner.textContent = "";
    const st = readState(node);

    // ── band：Add LoRA + 全部开/关 ──
    const band = document.createElement("div");
    band.className = "sf-plot-band";

    const add = document.createElement("button");
    add.className = "sf-ls-add";
    add.textContent = "＋ Add LoRA";
    add.title = "Add a row — each enabled row produces one model/clip pair";
    add.addEventListener("click", () => {
        const res = addLora(node, "");
        if (!res.ok) return;
        refreshNode(node, true);
        // 立刻在新行上打开选择器，让添加-选择一键完成。
        requestAnimationFrame(() => {
            const rowEl = inner.querySelector(`.sf-ls-row[data-id="${res.state.loras[res.index].id}"] .sf-ls-name`);
            if (rowEl) openNamePicker(node, res.state.loras[res.index].id, rowEl);
        });
    });
    band.appendChild(add);

    const all = document.createElement("div");
    all.className = "sf-ls-all";
    const allOn = st.loras.length > 0 && countOn(st) === st.loras.length;
    const cnt = document.createElement("span");
    cnt.className = "cnt";
    cnt.textContent = `${countOn(st)}/${st.loras.length} on`;
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    lbl.textContent = allOn ? "All off" : "All on";
    all.append(lbl, cnt);
    all.title = allOn ? "Turn every row off" : "Turn every row on";
    all.addEventListener("click", () => {
        setAllOn(node, !allOn);
        refreshNode(node, false);
    });
    band.appendChild(all);
    inner.appendChild(band);

    // ── 行列表 ──
    if (!st.loras.length) {
        const empty = document.createElement("div");
        empty.className = "sf-ls-empty";
        empty.textContent = "No LoRA rows — click Add LoRA to start.";
        inner.appendChild(empty);
        return;
    }
    const rows = document.createElement("div");
    rows.className = "sf-ls-rows";
    for (const e of st.loras) {
        rows.appendChild(buildRow(node, e, st));
    }
    inner.appendChild(rows);
}

function buildRow(node, e, st) {
    const row = document.createElement("div");
    row.className = "sf-ls-row" + (e.on ? "" : " off");
    row.dataset.id = e.id;
    row.title = e.name || "No LoRA selected";

    // 开关
    const sw = document.createElement("div");
    sw.className = "sf-ls-sw" + (e.on ? " on" : "");
    sw.title = e.on ? "On — this row is applied" : "Off — this row is skipped";
    sw.addEventListener("click", () => { patchLora(node, e.id, { on: !e.on }); refreshNode(node, false); });

    // 名称（点击弹 LoRA 选择器）
    const name = document.createElement("div");
    const missing = e.name ? hasLora(e.name) === false : false;
    name.className = "sf-ls-name" + (e.name ? "" : " empty") + (missing ? " missing" : "");
    name.title = e.name || "Click to choose a LoRA";
    const nm = document.createElement("span");
    nm.className = "nm";
    nm.textContent = e.name ? displayName(e.name, st.hideExt) : "(none)";
    const car = document.createElement("span");
    car.className = "car";
    car.textContent = "⌄";
    name.append(nm, car);
    name.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openNamePicker(node, e.id, name);
    });

    // 强度（输入 + ▲▼ 步进，复用 sf-ls-w 组件样式）
    const w = weightBox(e.sm, "m");
    w.title = "Strength — type a value or use the arrows";
    w.querySelector("input").addEventListener("change", (ev) => {
        const raw = parseFloat(ev.target.value);
        if (!Number.isFinite(raw)) { refreshNode(node, false); return; } // 垃圾输入 -> 回到存储值
        patchLora(node, e.id, { sm: raw });
        refreshNode(node, false);
    });
    w.querySelector("input").addEventListener("focusin", (ev) => ev.target.select?.());
    w.querySelector("input").addEventListener("keydown", (ev) => {
        ev.stopPropagation();
        if (ev.key === "Enter") { ev.preventDefault(); ev.target.blur(); }
    });
    const btns = w.querySelectorAll(".sf-ls-wbtn");
    btns[0].addEventListener("click", () => { patchLora(node, e.id, { sm: e.sm + st.step }); refreshNode(node, false); });
    btns[1].addEventListener("click", () => { patchLora(node, e.id, { sm: e.sm - st.step }); refreshNode(node, false); });

    // 删除
    const del = document.createElement("div");
    del.className = "sf-plot-del";
    del.textContent = "✕";
    del.title = "Remove this row";
    del.addEventListener("click", () => { removeLora(node, e.id); refreshNode(node, true); });

    row.append(sw, name, w, del);
    row.addEventListener("contextmenu", (ev) => {
        const id = rowIdOf(ev.target);
        if (!id) return;
        ev.preventDefault();
        ev.stopPropagation();
        rowMenu(node, id, ev.clientX, ev.clientY);
    });
    return row;
}

function openNamePicker(node, id, anchorEl) {
    const e = readState(node).loras.find((x) => x.id === id);
    openLoraDropdown(anchorEl, {
        current: e?.name || "",
        accent: accentOf(node),
        onPick: (picked) => { patchLora(node, id, { name: picked }); refreshNode(node, false); },
    });
}

function setupNode(node) {
    hideJsonWidget(node);
    injectPlotCSS();

    const root = document.createElement("div");
    root.className = "sf-ls-root";
    const inner = document.createElement("div");
    inner.className = "sf-ls-inner";
    root.appendChild(inner);

    const widget = node.addDOMWidget("plot_ui", "sf_lora_plot", root, {
        getValue: () => readState(node),
        setValue: () => {},
        getMinHeight: () => widgetH(node),
        getMaxHeight: () => widgetH(node),
        margin: 4,
        serialize: false,
    });
    widget.computeLayoutSize = () => ({ minHeight: widgetH(node), minWidth: 1 });
    applyAdaptiveCanvasOnly(widget);
    installCanvasZoomPassthrough(root);

    node._sfPlotRoot = root;
    node._sfPlotInner = inner;

    // 新节点的默认尺寸（configure() 会为加载的节点覆盖它，Vue Compat #8）。
    if (!Array.isArray(node.size)) node.size = [336, 0];
    node.size[0] = Math.max(node.size[0] || 0, 336);
    node.size[1] = fitNodeH(node);

    // 首次渲染推迟到 configure() 之后，让恢复的工作流渲染已存行而非默认
    // （Vue Compat #8）。fitToContent 在加载路径上让位。
    queueMicrotask(() => { renderNode(node); fitToContent(node); });

    // 预热列表让 missing 标记无需打开选择器即可显示。
    listLoras().then(() => { if (node._sfPlotRoot) renderNode(node); });
}

app.registerExtension({
    name: "sfnodes.LoraPlot",

    // 全局 LoRA 显示名设置（sfnodes.PowerLoraLoader.DisplayName）变化时经
    // 事件桥通知（LoRA Stack 同款）——行名随设置即时重绘（DOM 行，
    // setDirtyCanvas 管不到 widget DOM）。setTimeout(0) 推迟到设置 store
    // 更新后（同 Accent 时序教训）。
    init() {
        document.addEventListener("sfnodes.lora-display-mode-changed", () => {
            setTimeout(renderAllPlots, 0);
        });
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== CLASS) return;
        if (nodeType.prototype._sfPlotPatched) return;
        nodeType.prototype._sfPlotPatched = true;

        injectCSS();
        injectPlotCSS();

        // ComfyUI 按 R 时会对每个图节点调 node.refreshComboInNode(defs)——
        // 把它接进缓存失效与重渲染（LoRA Stack 同款先例）。
        const _origRefresh = nodeType.prototype.refreshComboInNode;
        nodeType.prototype.refreshComboInNode = function () {
            invalidateList();
            listLoras().then(() => renderAllPlots());
            if (_origRefresh) return _origRefresh.apply(this, arguments);
        };

        const _origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = _origConfigure?.apply(this, arguments);
            if (this._sfPlotRoot) { renderNode(this); fitToContent(this); }
            return r;
        };

        const _origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            // Legacy ONLY：Nodes 2.0 的渲染尺寸活在 Vue 布局 store，
            // getMinHeight/computeLayoutSize 已锁定高度——这里钳 node.size
            // 会失步并在切换工作流标签时弹跳（Nodes 2.0 resize 规则）。
            if (!isVueNodes()) {
                if (this.size[0] < MIN_W) this.size[0] = MIN_W;
                this.size[1] = fitNodeH(this);
            }
            if (_origResize) return _origResize.call(this, size);
        };

        // 同一钳制的保险带（节点 UI 约定 #7）：onResize 不覆盖每条 legacy
        // resize 路径，先增后减的循环可能让节点停在 MIN_W 之下。Legacy only。
        const _origDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (!isVueNodes() && !isGraphLoading() && this.size[0] < MIN_W) this.size[0] = MIN_W;
            return _origDrawFg?.apply(this, arguments);
        };

        const _origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            closeLoraDropdown(); // 瞬态——删除节点的画布点击也会自动关
            closeRowMenu();
            return _origRemoved?.apply(this, arguments);
        };
    },

    nodeCreated(node) {
        if (node.comfyClass !== CLASS) return;
        setupNode(node);
    },
});

// ── graphToPrompt：注入每节点状态（只注入，从不剪枝）────────────────────────
// 复用 LoRA Stack 的 buildIndex/findNode（已参数化导出）。
const _origGraphToPrompt = app.graphToPrompt.bind(app);
app.graphToPrompt = async function (...args) {
    const result = await _origGraphToPrompt(...args);
    try {
        const out = result?.output;
        if (out) {
            let index = null;
            for (const id in out) {
                const entry = out[id];
                if (!entry || entry.class_type !== CLASS) continue;
                if (!index) index = buildIndex([CLASS]);
                entry.inputs = entry.inputs || {};
                const node = findNode(index, id);
                const st = node ? readState(node) : { ...DEFAULT_STATE, ...loadDefaults(), loras: [] };
                entry.inputs[HIDDEN_INPUT] = JSON.stringify(promptState(st));
            }
        }
    } catch (e) {
        console.warn("[SF LoRA Plot] could not inject state:", (e && e.message) || e);
    }
    return result;
};
