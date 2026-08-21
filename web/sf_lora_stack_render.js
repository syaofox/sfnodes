// ==========================================================================
// SF LoRA Stack - DOM 构建（纯）+ CSS + 高度数学。这里没有事件监听器；
// interaction 模块在 widget 元素上挂一个委托处理器，按本模块盖的 data-act
// 属性分发。主扩展拥有尺寸计算并调用 renderNode()。
// ==========================================================================
import { app } from "/scripts/app.js";
import { isVueNodes, loraRowLabel } from "./sf_common.js";
import { BRAND, readState, accentOf, countOn, MAX_LORAS } from "./sf_lora_stack_core.js";
import { hasLora } from "./sf_lora_stack_api.js";
import { getLoraMetadata } from "./sf_lora_info.js";

// 重渲染图中全部 LoRA Stack 节点（含子图嵌套）。全局设置（sfnodes.Accent）
// 变化后调用，让新色立即生效。重绘纯 DOM，不触碰序列化状态。
export function repaintAll() {
    const walk = (g) => {
        for (const n of (g?._nodes || [])) {
            if ((n.comfyClass === "SFLoraStack" || n.type === "SFLoraStack") && n._sfLsRoot) renderNode(n);
            const sub = n.subgraph || n.graph || n._graph;
            if (sub && sub !== g) walk(sub);
        }
    };
    walk(app.graph);
    app.graph?.setDirtyCanvas?.(true, true);
}

// 高度常量——与 CSS 锁步，让节点贴合内容无底隙无滚动条（主扩展
// getMinHeight 读 contentHeight）。
const PAD = 9;
const ADD_H = 28;
const TOP_GAP = 5;
const TOPROW_H = 26;
const AFTER_TOP = 9;
export const ROW_H = 32;
const ROW_GAP = 6;
const EMPTY_H = 46;

// Add/All/gear 群组（"band"）。CLASSIC 下它浮出文档流，落在输入点（左）与
// 输出点（右）之间的空带上——见下方 offset + 预留，节点高度零成本。
// Nodes 2.0（子元素浮到 widget 顶部以上会被节点体裁切）保持正常流顶部。
// 分支在 renderNode。
const BAND_H = ADD_H + TOP_GAP + TOPROW_H; // 59
// Classic 浮动：节点局部 px。widget 体从节点顶下方 ~66px 开始；3 输出槽带
// 约 4..64，抬 ~62px 让它落在槽带里。校准于本节点的槽位布局（3 输入 /
// 3 输出）——注意：输入槽带画在 widget 区左侧、**不移动 widget 体顶**，
// 加第 3 输入（preset）后 band 仍保持 -62；preset 槽与 band 的垂直重叠靠
// CLASSIC_RSV_L（左 64px）水平避让。增删输出槽才需重调这里。
const CLASSIC_BAND_TOP = -62;
const CLASSIC_RSV_L = 64;   // 避开左侧 model / clip 标签
const CLASSIC_RSV_R = 80;   // 避开右侧 MODEL / CLIP / triggers 标签

export function contentHeight(state) {
    const n = state.loras.length;
    const rowsH = n ? n * ROW_H + (n - 1) * ROW_GAP : EMPTY_H;
    const bandInFlow = isVueNodes() ? BAND_H + AFTER_TOP : 0; // Classic 浮动（免费）
    return PAD + bandInFlow + rowsH + PAD;
}

const NO_LORAS = "(put LoRAs in models/loras)";
// 行名显示收敛于 sf_common.js::loraRowLabel（单一真源）：全局设置
// sfnodes.Lora.DisplayName ≠ full 时设置语义优先，full（默认）
// 回退每节点 hideExt（basename + 白名单剥模型扩展名）。SFLoraPlot 复用。
// 仅用于显示——行 title 保留真实文件名。
export function displayName(name, hideExt) {
    return loraRowLabel(name, hideExt);
}

// 一个权重框：可输入值 + ▲▼ 步进。`which` 是 "m"（model）或 "c"（clip）；
// data-act 值让委托处理器知道改哪个强度。
export function weightBox(value, which) {
    const w = document.createElement("div");
    w.className = "sf-ls-w";
    const val = document.createElement("input");
    val.className = "sf-ls-wval";
    val.dataset.act = which === "c" ? "wcval" : "wval";
    val.type = "text";
    val.value = Number(value).toFixed(2);
    val.title = which === "c" ? "Clip strength" : "Strength - type a value or use the arrows";
    const spin = document.createElement("div");
    spin.className = "sf-ls-wspin";
    const up = document.createElement("button");
    up.className = "sf-ls-wbtn"; up.dataset.act = which === "c" ? "wcinc" : "winc"; up.textContent = "▲"; up.tabIndex = -1;
    const dn = document.createElement("button");
    dn.className = "sf-ls-wbtn"; dn.dataset.act = which === "c" ? "wcdec" : "wdec"; dn.textContent = "▼"; dn.tabIndex = -1;
    spin.append(up, dn);
    w.append(val, spin);
    return w;
}

// 内联齿轮 SVG data URI（CSS mask），避免 emoji 跨平台形状不一致。
const GEAR_SVG = "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M6.8 1.2l.4-1h1.6l.4 1a6 6 0 0 1 1.5.9l1-.4 1.1 1.1-.4 1a6 6 0 0 1 .9 1.5l1 .4v1.6l-1 .4a6 6 0 0 1-.9 1.5l.4 1-1.1 1.1-1-.4a6 6 0 0 1-1.5.9l-.4 1H7.2l-.4-1a6 6 0 0 1-1.5-.9l-1 .4-1.1-1.1.4-1a6 6 0 0 1-.9-1.5l-1-.4V7.2l1-.4a6 6 0 0 1 .9-1.5l-.4-1 1.1-1.1 1 .4a6 6 0 0 1 1.5-.9zM8 5.5A2.5 2.5 0 1 0 8 10.5 2.5 2.5 0 0 0 8 5.5z"/></svg>'
);

export function injectCSS() {
    if (document.getElementById("sf-ls-css")) return;
    const s = document.createElement("style");
    s.id = "sf-ls-css";
    s.textContent = `
    .sf-ls-root { width:100%; box-sizing:border-box; background:#1d1d1d; border-radius:4px;
      color:#ddd; font-family:ui-sans-serif,system-ui,sans-serif; font-size:11px; position:relative; }
    /* 普通块流（非 flex、非 absolute），列表永不被压扁。每个子元素自然高度。 */
    .sf-ls-inner { box-sizing:border-box; padding:${PAD}px; }

    .sf-ls-band { display:flex; flex-direction:column; gap:${TOP_GAP}px; }
    .sf-ls-band:not(.floated) { margin-bottom:${AFTER_TOP}px; }
    .sf-ls-band.floated { position:absolute; pointer-events:none; z-index:2; }
    .sf-ls-band.floated > * { pointer-events:auto; }

    .sf-ls-add { box-sizing:border-box; width:100%; height:${ADD_H}px; border:0; border-radius:6px;
      background:var(--acc, var(--sf-acc, #f66744)); color:#fff; font:600 12px 'Segoe UI',sans-serif; cursor:pointer;
      display:flex; align-items:center; justify-content:center; gap:6px; }
    .sf-ls-add:hover { filter:brightness(1.08); }
    .sf-ls-add:disabled { opacity:.4; cursor:default; filter:none; }

    .sf-ls-toprow { display:flex; align-items:stretch; gap:6px; height:${TOPROW_H}px; }
    .sf-ls-all { flex:1; min-width:0; display:flex; align-items:center; gap:8px;
      background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.14); border-radius:5px;
      padding:0 9px; color:#a8a8a8; cursor:pointer; user-select:none; }
    .sf-ls-all:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#ddd; }
    .sf-ls-all .cnt { font-size:11px; white-space:nowrap; }
    .sf-ls-gear { flex:0 0 auto; width:32px; display:flex; align-items:center; justify-content:center;
      background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.14); border-radius:5px;
      cursor:pointer; user-select:none; }
    .sf-ls-gear::before { content:""; display:block; width:14px; height:14px; background:#bbb;
      -webkit-mask:url("${GEAR_SVG}") center/contain no-repeat;
      mask:url("${GEAR_SVG}") center/contain no-repeat; }
    .sf-ls-gear:hover { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-gear:hover::before { background:var(--acc, var(--sf-acc, #f66744)); }

    /* 预设按钮：存/取整个栈。All 有 min-width:0，宽节点下名称显示正常，
       窄节点里 All 内容被裁剪也不破坏布局。 */
    .sf-ls-presets { flex:0 0 auto; display:flex; align-items:center; justify-content:center;
      padding:0 8px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.14);
      border-radius:5px; cursor:pointer; user-select:none; font:11px 'Segoe UI',sans-serif;
      color:#a8a8a8; }
    .sf-ls-presets:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#ddd; }

    .sf-ls-rows { display:flex; flex-direction:column; gap:${ROW_GAP}px; }
    .sf-ls-row { box-sizing:border-box; height:${ROW_H}px; display:flex; align-items:center; gap:6px;
      background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12); border-radius:6px;
      padding:0 6px; position:relative; }
    .sf-ls-row.off { opacity:.42; }
    .sf-ls-row.dragging { opacity:.45; }
    .sf-ls-row.drag-before { box-shadow: inset 0 3px 0 var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-row.drag-after { box-shadow: inset 0 -3px 0 var(--acc, var(--sf-acc, #f66744)); }

    /* 拖拽排序手柄（行最左，⋮ 三点） */
    .sf-ls-grip { flex:0 0 auto; width:14px; height:100%; display:flex; align-items:center;
      justify-content:center; cursor:grab; color:#5a5a5a; user-select:none; touch-action:none; }
    .sf-ls-grip::before { content:""; width:3px; height:3px; border-radius:50%; background:currentColor;
      box-shadow:0 -5px 0 currentColor, 0 5px 0 currentColor; }
    .sf-ls-grip:hover { color:var(--acc, var(--sf-acc, #f66744)); }

    .sf-ls-name { flex:1; min-width:0; height:24px; display:flex; align-items:center; gap:5px;
      background:#161616; border:1px solid #3a3a3a; border-radius:5px; padding:0 8px;
      font:11px monospace; color:#ddd; cursor:pointer; overflow:hidden; }
    .sf-ls-name:hover { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-name .nm { flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .sf-ls-name.empty .nm { color:#777; }
    .sf-ls-name.missing .nm { color:#e05555; }
    .sf-ls-name.missing::before { content:"⚠"; flex:none; color:#e05555; font-size:11px; }
    .sf-ls-name .car { flex:none; color:#777; font-size:9px; }

    .sf-ls-w { flex:0 0 auto; display:flex; align-items:center; height:24px; width:56px;
      background:#161616; border:1px solid #3a3a3a; border-radius:5px; overflow:hidden; }
    .sf-ls-w:focus-within { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-wval { flex:1; min-width:0; width:100%; background:transparent; border:0; outline:none;
      color:#fff; text-align:center; font:11px monospace; padding:0; }
    .sf-ls-wval::-webkit-outer-spin-button,.sf-ls-wval::-webkit-inner-spin-button { -webkit-appearance:none; margin:0; }
    .sf-ls-wspin { flex:0 0 auto; display:flex; flex-direction:column; width:15px; height:100%;
      border-left:1px solid #3a3a3a; }
    .sf-ls-wbtn { flex:1; border:0; background:transparent; color:#9a9a9a; cursor:pointer;
      font-size:7px; line-height:1; display:flex; align-items:center; justify-content:center; padding:0; }
    .sf-ls-wbtn:hover { color:var(--acc, var(--sf-acc, #f66744)); background:rgba(255,255,255,0.06); }

    .sf-ls-info { flex:0 0 auto; width:22px; height:22px; border-radius:5px;
      border:1px solid rgba(255,255,255,0.12); background:rgba(255,255,255,0.05); color:#a8a8a8;
      cursor:pointer; display:flex; align-items:center; justify-content:center;
      font:italic 12px Georgia,serif; }
    .sf-ls-info:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    /* 高亮：该 LoRA 有用户编辑过的信息（_has_custom——与 Power 系 i 图标
       同语义同蓝色系：统一存储有词/描述或 .civitai.info 侧车有词/描述） */
    .sf-ls-info.net { border-color:rgba(79,195,247,0.7); color:#79c3f7;
      background:rgba(79,195,247,0.15); }

    .sf-ls-sw { flex:0 0 auto; width:30px; height:16px; border-radius:99px; background:#3a3a3a;
      position:relative; cursor:pointer; border:1px solid #000; }
    .sf-ls-sw::after { content:""; position:absolute; top:1px; left:1px; width:12px; height:12px;
      border-radius:50%; background:#8a8a8a; transition:left .14s, background .14s; }
    .sf-ls-sw.on { background:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-sw.on::after { left:15px; background:#fff; }

    .sf-ls-empty { box-sizing:border-box; height:${EMPTY_H}px;
      display:flex; align-items:center; justify-content:center; text-align:center; color:#777;
      font-size:11px; background:rgba(0,0,0,0.2); border:1px dashed #3a3a3a; border-radius:6px; padding:0 10px; }
  `;
    document.head.appendChild(s);
}

export function ensureRoot(node) {
    const held = node._sfLsRoot;
    if (held && held.isConnected) { node._sfLsRootMounted = true; return held; }
    const w = (node.widgets || []).find((x) => x.name === "loras_ui");
    const el = w?.element;
    const elRoot = el?.classList?.contains?.("sf-ls-root") ? el : el?.querySelector?.(".sf-ls-root");
    if (elRoot) { node._sfLsRoot = elRoot; node._sfLsRootMounted = true; return elRoot; }
    // 只在首次绘制时画进未连接的根（挂载前画好，挂上即见）。若曾挂载、现已
    // 丢失且无法重新解析，返回 null 让 renderNode 空操作而不是画一具分离的
    // 尸体——下一次事件/轮询会重新解析。
    return node._sfLsRootMounted ? null : (held || null);
}

export function renderNode(node) {
    const root = ensureRoot(node);
    if (!root) return;
    let inner = root.querySelector(".sf-ls-inner");
    if (!inner) {
        inner = document.createElement("div");
        inner.className = "sf-ls-inner";
        root.appendChild(inner);
    }
    node._sfLsInner = inner;

    const st = readState(node);
    const acc = accentOf(node);
    inner.style.setProperty("--acc", acc);
    inner.innerHTML = "";

    // ── Add / All / gear 群组（Classic 浮进槽位死带）────────────────────
    const band = document.createElement("div");
    band.className = "sf-ls-band";
    const classic = !isVueNodes();
    if (classic) {
        band.classList.add("floated");
        band.style.top = CLASSIC_BAND_TOP + "px";
        band.style.left = CLASSIC_RSV_L + "px";
        band.style.right = CLASSIC_RSV_R + "px";
    }

    const add = document.createElement("button");
    add.className = "sf-ls-add";
    add.dataset.act = "add";
    add.textContent = "＋ Add LoRA";
    add.disabled = st.loras.length >= MAX_LORAS;
    add.title = st.loras.length >= MAX_LORAS ? `Up to ${MAX_LORAS} LoRAs per node` : "Add a LoRA row";
    band.appendChild(add);

    // ── 全部开/关 + 计数，和齿轮 ─────────────────────────────────────────
    const on = countOn(st), total = st.loras.length;
    const toprow = document.createElement("div");
    toprow.className = "sf-ls-toprow";
    const all = document.createElement("div");
    all.className = "sf-ls-all";
    all.dataset.act = "allToggle";
    all.title = "Turn every LoRA on or off";
    const asw = document.createElement("span");
    asw.className = "sf-ls-sw" + (total && on === total ? " on" : "");
    const cnt = document.createElement("span");
    cnt.className = "cnt";
    cnt.textContent = total ? `${on} / ${total} on` : "no LoRAs";
    all.append(asw, cnt);
    const gear = document.createElement("div");
    gear.className = "sf-ls-gear";
    gear.dataset.act = "gear";
    // 无 textContent：图标由 ::before mask 绘制。
    gear.title = "LoRA Stack settings";
    const presets = document.createElement("div");
    presets.className = "sf-ls-presets";
    presets.dataset.act = "presets";
    presets.textContent = "Presets";
    presets.title = "Save the current stack as a preset, or load one";
    toprow.append(all, presets, gear);
    band.appendChild(toprow);
    inner.appendChild(band);

    // ── 行，或空状态 ──────────────────────────────────────────────────────
    if (!st.loras.length) {
        const empty = document.createElement("div");
        empty.className = "sf-ls-empty";
        empty.textContent = "No LoRAs yet — click ＋ Add LoRA to stack your first one.";
        inner.appendChild(empty);
        return;
    }

    const rows = document.createElement("div");
    rows.className = "sf-ls-rows";
    for (const e of st.loras) {
        const row = document.createElement("div");
        row.className = "sf-ls-row" + (e.on ? "" : " off");
        row.dataset.id = e.id;

        // 拖拽排序手柄（最左）。事件走 interaction 的 pointerdown 委托——
        // 行随 renderNode 重建，元素级监听会丢。
        const grip = document.createElement("div");
        grip.className = "sf-ls-grip";
        grip.title = "Drag to reorder";

        const name = document.createElement("div");
        // hasLora 在列表未取时返回 null，慢 fetch 不会闪假 "missing" 标记；
        // setup/刷新重绘会在列表落地后再渲染一次。文件已消失（改名/删除）
        // 的行运行时被跳过只有一行 console 日志——这个标记是用户唯一能
        // 看见"该 LoRA 没被应用"的地方。
        const missing = e.name ? hasLora(e.name) === false : false;
        name.className = "sf-ls-name" + (e.name ? "" : " empty") + (missing ? " missing" : "");
        name.dataset.act = "name";
        const nm = document.createElement("span");
        nm.className = "nm";
        nm.textContent = e.name ? displayName(e.name, st.hideExt) : NO_LORAS;
        nm.title = missing
            ? e.name + " - file not found (renamed or removed?). This row is skipped; pick the file again."
            : (e.name || "Pick a LoRA");
        const car = document.createElement("span"); car.className = "car"; car.textContent = "▾";
        name.append(nm, car);

        const wm = weightBox(e.sm, "m");

        const info = document.createElement("div");
        info.className = "sf-ls-info";
        info.dataset.act = "info";
        info.textContent = "i";
        info.title = "Info + pick trigger words";
        // 高亮 = 该 LoRA 有用户编辑过的信息（_has_custom：统一存储有词/描述，
        // 或 .civitai.info 侧车有词/描述——与 Power 系对话框 i 图标同一判定
        // 源，见 lora_notes 网关）。缓存命中即时；未命中 fetch 落地后碰活
        // 元素（renderNode 重建会重跑）。行已重建（isConnected=false）时
        // 丢弃——新行自己会查。
        if (e.name) {
            getLoraMetadata(e.name).then((meta) => {
                if (!info.isConnected) return;
                // "net" 是单个类名（CSS 选择器 .sf-ls-info.net = 两个类）
                info.classList.toggle("net", !!(meta && meta._has_custom));
            });
        }

        const sw = document.createElement("div");
        sw.className = "sf-ls-sw" + (e.on ? " on" : "");
        sw.dataset.act = "toggle";
        sw.title = e.on ? "On - click to turn off" : "Off - click to turn on";

        row.append(grip, name, wm);
        if (!st.linkStrength) row.appendChild(weightBox(e.sc, "c")); // 分离 model/clip
        row.append(info, sw);
        rows.appendChild(row);
    }
    inner.appendChild(rows);
}
