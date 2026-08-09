// ==========================================================================
// sf_dropdown_settings.js - SFValueDropdown 浮动设置面板
// ==========================================================================
//
// 与 Sizes / Sliders / Run Timer 同款壳：主题化面板贴在节点旁，头部可拖拽，
// 外部点击或 Esc 关闭。列表真正住在这里——节点面刻意只有一行。
//
// 复刻范围（与 sf_pause_text 同口径）：无 accent 颜色设置；无 pixaroma 的
// registerNodeSettings 中央注册（入口 = 节点行上的齿轮 + 右键菜单，见
// sf_dropdown.js）；Export/Import/Clear 全保留。
//
// 内联 shared 辅助：isGraphLoading（app.loadGraphData 包装，加载窗口防断线）、
// dropIncompatibleLinks（切换类型时剪掉不再兼容的线）。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { isGraphLoading } from "./sf_common.js";
import { isVueNodes } from "./sf_dropdown_ui.js";
import {
    readState, writeState, syncOutput, slotAccepts,
    MODES, MODE_LETTERS, MODE_LABELS,
} from "./sf_dropdown_lib.js";
import { visibleOptions } from "./sf_dropdown_lib.js";
import { TYPES, TYPE_LABELS, readable, previewText } from "./sf_dropdown_lib.js";

// ── 内联 shared：isGraphLoading ──────────────────────────────────────────
// "是否正在加载工作流？"守卫。LiteGraph 在节点 onConfigure 返回之后才在图
// 级别恢复已保存的线（此时任何节点级 configuring 标志早已清除），所以在
// onConnectionsChange 里改动序列化状态的节点会被那次连接回放覆盖——除非
// 抑制回放。包装 app.loadGraphData（工作流打开/切页/undo 的唯一漏斗）由
// sf_common.js 顶层统一安装（幂等单例），此处仅 import 判定函数。

// ── 内联 shared：dropIncompatibleLinks ───────────────────────────────────
// 剪掉新类型喂不了的线。只限真实用户动作。返回剪掉的线数让调用方能说出来——
// 静默切断用户看不见丢失的连接，是工作流悄悄停转的方式。加载期间永不运行：
// 已保存的图定义上自洽，在那里剪等于打开文件就损坏它。
function dropIncompatibleLinks(node) {
    if (!node?.outputs?.length || isGraphLoading()) return 0;
    const out = node.outputs[0];
    const links = Array.isArray(out.links) ? out.links.slice() : [];
    if (!links.length) return 0;

    const graph = node.graph;
    if (!graph) return 0;
    const want = out.type;
    let cut = 0;

    for (const id of links) {
        let link = graph.links?.[id];
        // graph.links 在新前端上可以是 Map（Vue Compat #3）。
        if (!link && typeof graph.links?.get === "function") link = graph.links.get(id);
        if (!link) continue;
        const target = graph.getNodeById?.(link.target_id);
        const slot = target?.inputs?.[link.target_slot];
        if (!slot) continue;
        const accepts = slot.type;
        // slotAccepts 而非 "==="：覆盖任一侧的 "*" 通配（Reroute、Set/Get、
        // Preview Any）以及 ComfyUI 多类型输入（到达时是逗号连接的
        // "FLOAT,INT,BOOLEAN"，如核心的 Math Expression）。相等测试把它读成
        // 一个未知名字，剪掉用户刚画的线。
        if (slotAccepts(accepts, want)) continue;
        target.disconnectInput?.(link.target_slot);
        cut++;
    }
    return cut;
}

// 与节点面相同的固定强调色。

let _panel = null;
let _panelNode = null;
let _onChange = null;
let _followRaf = null;  // canvas 跟随循环，见 startFollowing()
let _userMoved = false; // 用户是否故意拖走了面板？

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

// 所选类型的真实合法输入示例，而非复述列头。一眼说明节点的用途：短名代表
// 更长的、你不想重打的东西。按类型不同，因为"warm light"放在步数列表上方
// 是胡话。
const PLACEHOLDERS = {
    text:  { name: "warm light", value: "warm golden hour light, long soft shadows" },
    int:   { name: "square",     value: "1024" },
    float: { name: "gentle",     value: "0.35" },
    bool:  { name: "detail on",  value: "true" },
};

function toast(msg, severity = "info") {
    const t = app?.extensionManager?.toast;
    if (t?.add) t.add({ severity, summary: "SF Value Dropdown", detail: msg, life: 3200 });
    else console.warn("[SF Value Dropdown]", msg);
}

function injectCSS() {
    if (document.getElementById("sf-ddp-css")) return;
    const s = document.createElement("style");
    s.id = "sf-ddp-css";
    s.textContent = `
    .sf-ddp { position:fixed; z-index:10010; width:430px; max-width:94vw; background:#1a1a1a;
      border:1px solid #4a4a4a; border-radius:10px; box-shadow:0 18px 50px rgba(0,0,0,0.6);
      color:#d8d8d8; font:12px 'Segoe UI',-apple-system,sans-serif; overflow:hidden; }
    .sf-ddp-t { display:flex; align-items:center; gap:8px; padding:10px 12px; background:#232323;
      border-bottom:1px solid #333; cursor:grab; user-select:none; color:${"var(--sf-acc, #f66744)"}; }
    .sf-ddp-t .x { margin-left:auto; color:#8a8a8a; cursor:pointer; padding:0 4px; }
    .sf-ddp-t .x:hover { color:#fff; }
    .sf-ddp-b { padding:12px; display:flex; flex-direction:column; gap:12px; max-height:64vh; overflow-y:auto; }

    .sf-ddp-lab { font-size:11px; color:${"var(--sf-acc, #f66744)"}; letter-spacing:.04em; }
    .sf-ddp-sub { font-size:11px; color:#777; line-height:1.5; }

    .sf-ddp-catrow { display:flex; align-items:center; gap:5px; }
    .sf-ddp-catbtn { flex:0 1 auto; min-width:0; max-width:220px; overflow:hidden;
      text-overflow:ellipsis; white-space:nowrap; text-align:left; padding:6px 8px; border-radius:5px;
      background:#1d1d1d; border:1px solid #444; color:#ccc;
      font:11px 'Segoe UI',sans-serif; cursor:pointer; }
    .sf-ddp-catbtn:hover { border-color:${"var(--sf-acc, #f66744)"}; color:#ddd; }
    .sf-ddp-catbtn.on { border-color:${"var(--sf-acc, #f66744)"}; }
    .sf-ddp-catbtn:disabled { opacity:.4; cursor:default; }
    .sf-ddp-catbtn:disabled:hover { border-color:#444; color:#ccc; }
    .sf-ddp-btn:disabled { opacity:.4; cursor:default; }
    .sf-ddp-catpop { position:absolute; z-index:6; min-width:160px; max-height:220px; overflow-y:auto;
      background:#232323; border:1px solid #4a4a4a; border-radius:6px; box-shadow:0 10px 30px rgba(0,0,0,.5); padding:4px; }
    .sf-ddp-catopt { padding:6px 10px; border-radius:4px; cursor:pointer; color:#bbb; font-size:11.5px;
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-ddp-catopt:hover { background:#2e2e2e; color:#fff; }
    .sf-ddp-catopt.on { background:${"var(--sf-acc, #f66744)"}; color:#fff; }

    .sf-ddp-seg { display:flex; gap:5px; flex-wrap:wrap; }
    .sf-ddp-seg button { flex:1 1 auto; min-width:78px; text-align:center; padding:6px 8px; border-radius:5px;
      background:#1d1d1d; border:1px solid #444; color:#aaa;
      font:11px 'Segoe UI',sans-serif; cursor:pointer; }
    .sf-ddp-seg button:hover { border-color:${"var(--sf-acc, #f66744)"}; color:#ddd; }
    .sf-ddp-seg button.on { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }

    .sf-ddp-modes { display:flex; gap:5px; }
    .sf-ddp-modes button { width:34px; text-align:center; padding:6px 0; border-radius:5px;
      background:#1d1d1d; border:1px solid #444; color:#aaa;
      font:12px 'Segoe UI',sans-serif; cursor:pointer; }
    .sf-ddp-modes button:hover { border-color:${"var(--sf-acc, #f66744)"}; color:#ddd; }
    .sf-ddp-modes button.on { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }

    .sf-ddp-head { display:flex; align-items:center; justify-content:space-between; }
    .sf-ddp-count { font-size:11px; color:#666; }

    .sf-ddp-cols { display:flex; gap:6px; padding:0 0 4px 22px; }
    .sf-ddp-cols .a { width:118px; flex:none; font-size:11px; color:#777; }
    .sf-ddp-cols .b { flex:1; font-size:11px; color:#777; }

    .sf-ddp-list { background:rgba(0,0,0,0.28); border-radius:6px; padding:4px;
      display:flex; flex-direction:column; gap:3px; }
    /* 当前选中行无高亮。它曾带强调色水洗，读作"这行很特别"却不说为什么——
       节点面与打开的列表都已显示哪条生效，面板在发明第三个说法的地方。
       面板 BUILD 列表；节点从它 PICK。 */
    .sf-ddp-row { display:flex; align-items:flex-start; gap:6px; padding:4px;
      border-radius:5px; background:rgba(255,255,255,0.02); }
    .sf-ddp-row.drop-above { box-shadow:inset 0 2px 0 ${"var(--sf-acc, #f66744)"}; }
    .sf-ddp-row.drop-below { box-shadow:inset 0 -2px 0 ${"var(--sf-acc, #f66744)"}; }
    .sf-ddp-row .grip { color:${"var(--sf-acc, #f66744)"}; cursor:grab; flex:none; font-size:12px;
      line-height:1; padding:6px 2px 0; opacity:.8; }
    .sf-ddp-row .grip:hover { opacity:1; }
    .sf-ddp-nm { width:118px; flex:none; box-sizing:border-box; background:#1d1d1d;
      border:1px solid #444; border-radius:4px; color:#ddd; font:11px 'Segoe UI',sans-serif;
      padding:5px 7px; outline:none; }
    .sf-ddp-nm:focus { border-color:${"var(--sf-acc, #f66744)"}; }
    .sf-ddp-vl { flex:1 1 auto; min-width:0; box-sizing:border-box; background:#1d1d1d;
      border:1px solid #444; border-radius:4px; color:#ddd; font:11px 'Segoe UI',sans-serif;
      padding:5px 7px; outline:none; resize:none; overflow:hidden; line-height:1.45; }
    .sf-ddp-vl:focus { border-color:${"var(--sf-acc, #f66744)"}; }
    .sf-ddp-vl.bad { border-color:#a8552f; }
    .sf-ddp-warn { flex:none; width:13px; text-align:center; padding-top:5px;
      color:#e0703a; font-size:11px; cursor:default; }
    .sf-ddp-warn.hide { display:none; }
    /* + 与 ✕ 紧贴值框：曾经的间隙是值字段可用的死空间。+ 是填充芯片，读作
       两者中的主操作，不被 ✕ 压小。BOTH 对齐行第一行（值框单行高），不是行：
       值框随文本长高，飘在六行段落中间的按钮读作不属于任何东西。 */
    .sf-ddp-ins, .sf-ddp-del { flex:none; height:27px; padding:0;
      display:flex; align-items:center; justify-content:center;
      background:none; border:none; cursor:pointer; }
    .sf-ddp-ins { width:21px; }
    .sf-ddp-ins::before { content:"+"; display:flex; align-items:center; justify-content:center;
      width:19px; height:19px; border-radius:4px;
      background:color-mix(in srgb, ${"var(--sf-acc, #f66744)"} 22%, transparent);
      color:${"var(--sf-acc, #f66744)"}; font:15px/1 'Segoe UI',sans-serif; }
    .sf-ddp-ins:hover::before { background:${"var(--sf-acc, #f66744)"}; color:#fff; }
    .sf-ddp-del { width:17px; color:#777; font:13px/1 'Segoe UI',sans-serif; }
    .sf-ddp-del:hover { color:#e0604a; }

    .sf-ddp-empty { padding:14px 10px; text-align:center; }
    .sf-ddp-empty p { margin:0 0 10px; color:#777; font-size:11px; font-style:italic; }
    .sf-ddp-emptybtn { background:${"var(--sf-acc, #f66744)"}; color:#fff; border:0; border-radius:5px;
      padding:7px 16px; font:12px 'Segoe UI',sans-serif; cursor:pointer; }
    .sf-ddp-emptybtn:hover { filter:brightness(1.1); }

    .sf-ddp-f { display:flex; gap:8px; flex-wrap:wrap; padding:10px 12px; border-top:1px solid #333; background:#1f1f1f; }
    .sf-ddp-btn { border:1px solid rgba(255,255,255,0.14); background:rgba(255,255,255,0.04); color:rgba(255,255,255,0.65);
      border-radius:5px; padding:6px 12px; font:12px 'Segoe UI',sans-serif; cursor:pointer; }
    .sf-ddp-btn:hover { border-color:${"var(--sf-acc, #f66744)"}; background:${"var(--sf-acc, #f66744)"}; color:#fff; }
    .sf-ddp-btn.primary { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
    .sf-ddp-btn.primary:hover { filter:brightness(1.1); }
    .sf-ddp-push { margin-left:auto; }

    /* Clear-list 确认框。它住在面板元素内部：放在 document.body 上会落在面板
       外，外部点击关闭器第一击就把整个设置面板带走。 */
    .sf-ddp-ask { position:absolute; inset:0; z-index:5; background:rgba(0,0,0,0.55);
      display:flex; align-items:center; justify-content:center; }
    .sf-ddp-askbox { background:#232323; border:1px solid #4a4a4a; border-radius:8px;
      padding:14px 16px; width:min(320px,86%); display:flex; flex-direction:column; gap:10px;
      box-shadow:0 10px 30px rgba(0,0,0,0.5); }
    .sf-ddp-asktitle { color:${"var(--sf-acc, #f66744)"}; font-size:12px; }
    .sf-ddp-askmsg { color:#bbb; font-size:11.5px; line-height:1.5; }
    .sf-ddp-askrow { display:flex; gap:8px; justify-content:flex-end; }
  `;
    document.head.appendChild(s);
}

function getNodeScreenRect(node) {
    if (isVueNodes() && node && node.id != null) {
        const e = document.querySelector(`[data-node-id="${node.id}"]`);
        if (e) return e.getBoundingClientRect();
    }
    const c = app.canvas;
    const ds = c && c.ds;
    const cv = c && c.canvas;
    if (!ds || !cv || !node?.pos || !node?.size) return null;
    const cr = cv.getBoundingClientRect();
    const titleH = window.LiteGraph?.NODE_TITLE_HEIGHT || 30;
    const sc = ds.scale || 1;
    const off = ds.offset || [0, 0];
    const left = cr.left + (node.pos[0] + off[0]) * sc;
    const top = cr.top + (node.pos[1] - titleH + off[1]) * sc;
    return { left, top, right: left + node.size[0] * sc, bottom: top + (node.size[1] + titleH) * sc,
             width: node.size[0] * sc, height: (node.size[1] + titleH) * sc };
}

function placeBeside(panel, rect) {
    const vw = window.innerWidth, vh = window.innerHeight;
    const mw = panel.offsetWidth, mh = panel.offsetHeight;
    const gap = 12, pad = 8;
    if (!rect) {
        panel.style.left = Math.max(pad, (vw - mw) / 2) + "px";
        panel.style.top = Math.max(pad, (vh - mh) / 2) + "px";
        return;
    }
    let left = rect.right + gap;
    if (left + mw > vw - pad) left = rect.left - gap - mw;
    if (left < pad) left = Math.max(pad, vw - mw - pad);
    let top = rect.top;
    if (top + mh > vh - pad) top = vh - mh - pad;
    if (top < pad) top = pad;
    panel.style.left = left + "px";
    panel.style.top = top + "px";
}

/**
 * 画布移动时让面板贴着自己的节点。
 *
 * 没有这个，面板只写一次固定屏幕位置然后留在原地：缩放或平移后它被搁浅在
 * 别处，画布上两个 Dropdown 时无法分辨在编辑哪个。
 *
 * rAF 循环而非事件：LiteGraph 对变换变化不发任何事件，而缩放必须平滑跟随
 * 而非 350ms 后追上。每帧比较三个数字即返回，空闲成本为零，且只在面板打开
 * 时运行。
 *
 * 用户一拖面板就停止跟随：那时他们已把它放到刻意位置，再挪走比留在原地
 * 更糟。
 */
function startFollowing(panel, node) {
    let lastScale = null, lastX = null, lastY = null;
    const tick = () => {
        if (!_panel || _panel !== panel || !panel.isConnected) { _followRaf = null; return; }
        _followRaf = requestAnimationFrame(tick);
        if (_userMoved) return;
        const ds = app.canvas?.ds;
        if (!ds) return;
        const sc = ds.scale || 1;
        const ox = ds.offset?.[0] ?? 0, oy = ds.offset?.[1] ?? 0;
        if (sc === lastScale && ox === lastX && oy === lastY) return;
        lastScale = sc; lastX = ox; lastY = oy;
        placeBeside(panel, getNodeScreenRect(node));
    };
    _followRaf = requestAnimationFrame(tick);
}

function stopFollowing() {
    if (_followRaf != null) cancelAnimationFrame(_followRaf);
    _followRaf = null;
}

function makeDraggable(panel, handle) {
    handle.addEventListener("pointerdown", (e) => {
        if (e.target.closest(".x")) return;
        e.preventDefault();
        const r = panel.getBoundingClientRect();
        const ox = e.clientX - r.left, oy = e.clientY - r.top;

        // 两道防拖拽粘在光标上的防线，因为 pointerup 真的会丢：在窗口外释放、
        // 第二显示器、或上游吞掉。合成事件无法复现，绿色脚本测试毫无意义——
        // 这是从 Help 窗口人工报告换来的家规。
        try { handle.setPointerCapture(e.pointerId); } catch { /* not capturable */ }

        const move = (ev) => {
            if (!panel.isConnected) return up();
            // 按钮已松开：释放丢了，结束拖拽。
            if (!(ev.buttons & 1)) return up();
            // 从此面板在用户放置处，停止跟随。
            _userMoved = true;
            panel.style.left = Math.max(0, Math.min(window.innerWidth - panel.offsetWidth, ev.clientX - ox)) + "px";
            panel.style.top = Math.max(0, Math.min(window.innerHeight - panel.offsetHeight, ev.clientY - oy)) + "px";
        };
        // 幂等：上面的按钮守卫和真实释放都能调用它，lostpointercapture 在我们
        // 自己释放后也会触发。
        let done = false;
        const up = () => {
            if (done) return;
            done = true;
            try { handle.releasePointerCapture(e.pointerId); } catch { /* already gone */ }
            handle.removeEventListener("pointermove", move, true);
            handle.removeEventListener("pointerup", up, true);
            handle.removeEventListener("pointercancel", up, true);
            handle.removeEventListener("lostpointercapture", up, true);
            window.removeEventListener("pointermove", move, true);
            window.removeEventListener("pointerup", up, true);
        };
        handle.addEventListener("pointermove", move, true);
        handle.addEventListener("pointerup", up, true);
        handle.addEventListener("pointercancel", up, true);
        handle.addEventListener("lostpointercapture", up, true);
        // window 腰带：capture 拿不到时 move 事件仍到这里。
        window.addEventListener("pointermove", move, true);
        window.addEventListener("pointerup", up, true);
    });
}

function outsideClose(e) {
    if (!_panel) return;
    if (_panel.contains(e.target)) return;
    closeDropdownPanel();
}
function escClose(e) {
    if (e.key === "Escape" && _panel) {
        e.stopPropagation();
        closeDropdownPanel();
    }
}

export function closeDropdownPanel() {
    stopFollowing();
    _userMoved = false;
    if (_panel) { try { _panel.remove(); } catch {} }
    _panel = null;
    _panelNode = null;
    _onChange = null;
    document.removeEventListener("pointerdown", outsideClose, true);
    document.removeEventListener("keydown", escClose, true);
}

export function closeDropdownPanelFor(node) {
    if (_panelNode === node) closeDropdownPanel();
}

// 值框按内容长高。空框钉死单行：窄宽下包裹的 placeholder 否则把 scrollHeight
// 吹大，框长高后永远缩不回去（Nodes 2.0 recipe #7）。
const VALUE_MAX_H = 160;

function autoGrow(ta) {
    if (!ta.value) { ta.style.height = "27px"; ta.style.overflowY = "hidden"; return; }
    ta.style.height = "auto";
    const want = Math.max(27, ta.scrollHeight);
    ta.style.height = Math.min(VALUE_MAX_H, want) + "px";
    // 停止长高后其余文本必须仍可达。框在所有高度都 overflow:hidden，所以
    // 粘贴超过约九行包裹的样式段落完全看不见也滚不到——在一个以容纳长文本为
    // 卖点的节点里不能这样。
    ta.style.overflowY = want > VALUE_MAX_H ? "auto" : "hidden";
}

export function openDropdownPanel(node, onChange) {
    closeDropdownPanel();
    injectCSS();
    _onChange = onChange || null;
    _panelNode = node;

    const panel = el("div", "sf-ddp");

    const title = el("div", "sf-ddp-t");
    title.append(el("span", null, "⚙"), el("span", null, "Value Dropdown settings"));
    const x = el("span", "x", "✕");
    x.addEventListener("click", closeDropdownPanel);
    title.appendChild(x);

    const body = el("div", "sf-ddp-b");
    const foot = el("div", "sf-ddp-f");

    const fire = () => { _onChange?.(node); };

    // ── 分类 ────────────────────────────────────────────────────────────
    // 分类是节点状态的一部分（categories/category/行 category）。行归属 =
    // 添加时的当前分类，行级不改分类；新建/重命名/删除只在这里做（节点面的
    // 分类按钮只管切换）。
    const catSec = el("div");
    catSec.append(el("div", "sf-ddp-lab", "CATEGORY"));
    const catRow = el("div", "sf-ddp-catrow");
    const catBtn = el("button", "sf-ddp-catbtn");
    catBtn.title = "Switch category";
    catBtn.addEventListener("click", () => toggleCatPop());
    const bNewCat = el("button", "sf-ddp-btn", "+ New");
    bNewCat.title = "Create a new category and switch to it";
    bNewCat.addEventListener("click", newCategory);
    const bRenCat = el("button", "sf-ddp-btn", "Rename");
    bRenCat.title = "Rename the current category";
    bRenCat.addEventListener("click", renameCategory);
    const bDelCat = el("button", "sf-ddp-btn", "Delete");
    bDelCat.title = "Delete the current category; its entries move to default";
    bDelCat.addEventListener("click", deleteCategory);
    catRow.append(catBtn, bNewCat, bRenCat, bDelCat);
    catSec.appendChild(catRow);

    // 分类弹层：absolute 定位在面板内（面板是 fixed 容器）。
    let catPop = null;
    function closeCatPop() {
        if (catPop) { catPop.remove(); catPop = null; }
        catBtn.classList.remove("on");
    }
    function renderCatPop() {
        if (!catPop) return;
        catPop.textContent = "";
        const st = readState(node);
        for (const c of st.categories) {
            const opt = el("div", "sf-ddp-catopt" + (c === st.category ? " on" : ""), c);
            opt.addEventListener("click", () => selectCategory(c));
            catPop.appendChild(opt);
        }
    }
    function toggleCatPop() {
        if (catPop) { closeCatPop(); return; }
        catPop = el("div", "sf-ddp-catpop");
        renderCatPop();
        panel.appendChild(catPop);
        catPop.style.top = (catBtn.offsetTop + catBtn.offsetHeight + 4) + "px";
        catPop.style.left = "0px";
        catBtn.classList.add("on");
    }
    // 面板内点击弹层外（含新/改名/删除按钮）即关闭。弹层在面板内，面板的
    // document 级 outsideClose 豁免整个面板，管不到这里。
    panel.addEventListener("pointerdown", (e) => {
        if (!catPop) return;
        if (catPop.contains(e.target) || e.target.closest(".sf-ddp-catbtn")) return;
        closeCatPop();
    });

    function renderCatBtn() {
        const st = readState(node);
        catBtn.textContent = "▣ " + (st.category || "default") + " ▾";
        catBtn.title = `Category: ${st.category}\nClick to switch`;
        const single = st.categories.length <= 1;
        catBtn.classList.toggle("dim", single);
        bDelCat.classList.toggle("dim", single);
        catBtn.disabled = single;
        bDelCat.disabled = single;
        renderCatPop();
    }

    function selectCategory(c) {
        const st = readState(node);
        if (st.category === c) { closeCatPop(); return; }
        // 与节点面同规则：index 从头开始，游标丢弃。
        writeState(node, { category: c, index: 0 });
        node._sfDropdownPending = null;
        node._sfDropdownCursor = null;
        renderCatBtn();
        renderList();
        fire();
        closeCatPop();
    }

    function newCategory() {
        const name = prompt("New category name:");
        if (!name || !name.trim()) return;
        const c = name.trim();
        const st = readState(node);
        if (st.categories.includes(c)) { toast("That category already exists.", "warn"); return; }
        writeState(node, { categories: [...st.categories, c], category: c, index: 0 });
        node._sfDropdownPending = null;
        node._sfDropdownCursor = null;
        renderCatBtn();
        renderList();
        fire();
    }

    function renameCategory() {
        const st = readState(node);
        const old = st.category;
        const name = prompt(`Rename category "${old}" to:`, old);
        if (!name || !name.trim()) return;
        const c = name.trim();
        if (c === old) return;
        if (st.categories.includes(c)) { toast("That category already exists.", "warn"); return; }
        const categories = st.categories.map((x) => (x === old ? c : x));
        // 行归属跟随改名。
        const options = st.options.map((o) => ({
            ...o,
            category: o.category === old ? c : o.category,
        }));
        writeState(node, { categories, category: c, options });
        renderCatBtn();
        renderList();
        fire();
    }

    function deleteCategory() {
        const st = readState(node);
        if (st.categories.length <= 1) return;
        const old = st.category;
        if (!confirm(`Delete category "${old}"? Its entries move to "default".`)) return;
        const categories = st.categories.filter((x) => x !== old);
        const options = st.options.map((o) => ({
            ...o,
            category: o.category === old ? "default" : o.category,
        }));
        writeState(node, { categories, category: "default", index: 0 });
        node._sfDropdownPending = null;
        node._sfDropdownCursor = null;
        renderCatBtn();
        renderList();
        fire();
    }

    // 画在 THIS 面板上的是非问题。两个陷阱让它留在面板内而非 document.body
    // 对话框：(a) 面板在任何外部 pointerdown 上关闭，body 级 backdrop 就是
    // 外部，回答提问会同时关掉设置；(b) 面板的 Esc 关闭器是 document 级
    // capture 监听器，所以这里监听 WINDOW capture（先运行）——Esc 回答提问
    // 而不是关掉它下面的面板。Enter 是 OK 键。
    function askInPanel({ title: t, message, okText }) {
        return new Promise((resolve) => {
            const back = el("div", "sf-ddp-ask");
            const box = el("div", "sf-ddp-askbox");
            box.appendChild(el("div", "sf-ddp-asktitle", t));
            if (message) box.appendChild(el("div", "sf-ddp-askmsg", message));
            const row = el("div", "sf-ddp-askrow");
            const no = el("button", "sf-ddp-btn", "Cancel");
            const ok = el("button", "sf-ddp-btn primary", okText || "OK");
            row.append(no, ok);
            box.appendChild(row);
            back.appendChild(box);
            panel.appendChild(back);

            let done = false;
            const finish = (v) => {
                if (done) return;
                done = true;
                window.removeEventListener("keydown", onKey, true);
                back.remove();
                resolve(v);
            };
            const onKey = (e) => {
                if (e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); finish(false); }
                else if (e.key === "Enter") { e.preventDefault(); e.stopImmediatePropagation(); finish(true); }
            };
            window.addEventListener("keydown", onKey, true);
            back.addEventListener("pointerdown", (e) => { if (e.target === back) finish(false); });
            no.addEventListener("click", () => finish(false));
            ok.addEventListener("click", () => finish(true));
            queueMicrotask(() => ok.focus());
        });
    }

    // ── 会发出什么 ──────────────────────────────────────────────────────
    const typeSec = el("div");
    typeSec.append(el("div", "sf-ddp-lab", "WHAT COMES OUT"));
    const seg = el("div", "sf-ddp-seg");
    typeSec.appendChild(seg);
    const typeHint = el("div", "sf-ddp-sub");
    typeSec.appendChild(typeHint);

    // ── 列表 ────────────────────────────────────────────────────────────
    // ── 每次运行发生什么 ────────────────────────────────────────────────
    const runSec = el("div");
    runSec.append(el("div", "sf-ddp-lab", "EACH TIME YOU RUN"));
    const modeRow = el("div", "sf-ddp-modes");
    runSec.appendChild(modeRow);
    const modeHint = el("div", "sf-ddp-sub");
    runSec.appendChild(modeHint);

    const listSec = el("div");
    const head = el("div", "sf-ddp-head");
    head.append(el("span", "sf-ddp-lab", "THE LIST"));
    const count = el("span", "sf-ddp-count");
    head.appendChild(count);
    listSec.appendChild(head);
    const cols = el("div", "sf-ddp-cols");
    const ca = el("span", "a", "Name in the list");
    const cb = el("span", "b", "What it sends out");
    cols.append(ca, cb);
    listSec.appendChild(cols);
    const list = el("div", "sf-ddp-list");
    listSec.appendChild(list);

    body.append(catSec, typeSec, runSec, listSec);

    // ── 渲染 ────────────────────────────────────────────────────────────
    let dragFrom = -1;

    function renderModes() {
        const st = readState(node);
        modeRow.textContent = "";
        for (const m of MODES) {
            const b = el("button", st.mode === m ? "on" : null, MODE_LETTERS[m]);
            b.title = MODE_LABELS[m];
            b.addEventListener("click", () => {
                if (readState(node).mode === m) return;
                writeState(node, { mode: m });
                // 丢掉持有的或已花的位置，切换模式从节点显示的条目干净开始。
                node._sfDropdownPending = null;
                node._sfDropdownCursor = null;
                renderModes();
                fire();
            });
            modeRow.appendChild(b);
        }
        const n = visibleOptions(readState(node)).length;
        modeHint.textContent = st.mode === "fixed"
            ? "Always sends the entry you picked."
            : (n < 2
                ? (st.mode === "increment" ? "Steps to the next entry each run. Add more entries for this to do anything."
                                           : "Picks any entry each run. Add more entries for this to do anything.")
                : (st.mode === "increment" ? "Steps to the next entry each run and wraps at the end."
                                           : "Picks a different entry at random each run."));
    }

    function renderTypes() {
        const st = readState(node);
        seg.textContent = "";
        for (const t of TYPES) {
            const b = el("button", st.type === t ? "on" : null, TYPE_LABELS[t]);
            b.title = `Send ${TYPE_LABELS[t].toLowerCase()} out of this node`;
            b.addEventListener("click", () => setType(t));
            seg.appendChild(b);
        }
        const st2 = readState(node);
        const vis = visibleOptions(st2);
        const bad = vis.filter((o) => !readable(o.value, st2.type)).length;
        typeHint.textContent = bad
            ? `${bad} of ${vis.length} ${bad === 1 ? "entry does" : "entries do"} not read as ${TYPE_LABELS[st2.type].toLowerCase()}. They are kept, and send the fallback until you change them.`
            : "Changing this renames the output and unplugs anything that no longer fits. Your text is always kept.";
    }

    function setType(t) {
        const st = readState(node);
        if (st.type === t) return;
        writeState(node, { type: t });
        syncOutput(node);
        const cut = dropIncompatibleLinks(node);
        const vis = visibleOptions(readState(node));
        const bad = vis.filter((o) => !readable(o.value, t)).length;

        // 说明发生了什么。静默警告标记对改变节点输出的事太轻，静默剪线更糟。
        const bits = [];
        if (cut) bits.push(`${cut} ${cut === 1 ? "wire was" : "wires were"} unplugged`);
        if (bad) bits.push(`${bad} ${bad === 1 ? "entry does" : "entries do"} not read as ${TYPE_LABELS[t].toLowerCase()} and will send the fallback`);
        if (bits.length) toast(bits.join("; ") + ". Your text is kept.", "warn");

        renderTypes();
        renderList();
        fire();
    }

    function commit(patch) {
        writeState(node, patch);
        renderTypes();
        renderModes();
        renderCatBtn();   // Import/Clear 会改 categories/category，分类区必须跟随
        renderList();
        fire();
    }

    // 追加一个条目并把光标放进它的名字框，直接开打。footer 按钮与空态按钮
    // 共用，永不漂移。行归属 = 当前分类。
    function addRow() {
        const cur = readState(node);
        const vis = visibleOptions(cur);
        vis.push({ name: "", value: "", category: cur.category });
        commit({ options: mergeBack(cur, vis), index: cur.index });
        const boxes = list.querySelectorAll(".sf-ddp-nm");
        boxes[boxes.length - 1]?.focus();
    }

    // 当前分类行的全局索引。
    function globalsOf(cur) {
        const g = [];
        cur.options.forEach((o, i) => { if (o.category === cur.category) g.push(i); });
        return g;
    }

    // 把编辑后的当前分类行（vis）写回全量 options：其他分类的行保持相对顺序。
    function mergeBack(cur, vis) {
        const g = globalsOf(cur);
        const others = cur.options.filter((o, i) => !g.includes(i));
        return [...others, ...vis];
    }

    function renderList() {
        const st = readState(node);
        const vis = visibleOptions(st);
        count.textContent = vis.length === 1 ? "1 option" : `${vis.length} options`;
        list.textContent = "";

        if (!vis.length) {
            // 按钮属于这里——列表应在的位置、目光已在此处——而不是只在 footer
            // 里配一行指向它的散文。
            const box = el("div", "sf-ddp-empty");
            box.appendChild(el("p", null, "Nothing here yet."));
            const first = el("button", "sf-ddp-emptybtn", "Add your first entry");
            first.addEventListener("click", () => addRow());
            box.appendChild(first);
            list.appendChild(box);
            return;
        }

        vis.forEach((o, i) => {
            const row = el("div", "sf-ddp-row");

            // GRIP 是可拖拽元素，不是行。行 draggable 会让 e.target 是行，
            // 下面的守卫永远不匹配，重排静默无效，而且值框内拖拽劫持文本选择
            // 而不是选择文本（UI convention #11）。
            const grip = el("span", "grip", "⋮⋮");
            grip.draggable = true;
            grip.title = "Drag to reorder";

            const nm = el("input", "sf-ddp-nm");
            nm.value = o.name;
            nm.placeholder = PLACEHOLDERS[st.type].name;
            nm.title = "The short name you pick from the dropdown";

            const vl = el("textarea", "sf-ddp-vl");
            vl.value = o.value;
            vl.rows = 1;
            vl.placeholder = PLACEHOLDERS[st.type].value;
            vl.title = "The value this entry sends out. It can run to several lines.";
            if (!readable(o.value, st.type)) vl.classList.add("bad");

            const warn = el("span", "sf-ddp-warn" + (readable(o.value, st.type) ? " hide" : ""), "⚠");
            warn.title = `This does not read as ${TYPE_LABELS[st.type].toLowerCase()}. It is kept as you typed it, and sends ${JSON.stringify(previewText(o.value, st.type))} until you change it.`;

            // 字形由 ::before 芯片绘制，按钮本身为空。
            const ins = el("button", "sf-ddp-ins");
            ins.title = "Add a row below this one";
            const del = el("button", "sf-ddp-del", "✕");
            del.title = "Delete this row";

            row.append(grip, nm, vl, warn, ins, del);
            list.appendChild(row);
            autoGrow(vl);

            // 实时编辑直写；每次按键重渲染会毁掉正在输入的字段。vis 里的行与
            // cur.options 是同一对象引用，改字段即改状态。
            nm.addEventListener("input", () => {
                const cur = readState(node);
                if (!visibleOptions(cur)[i]) return;
                cur.options[globalsOf(cur)[i]].name = nm.value;
                writeState(node, { options: cur.options });
                fire();
            });
            vl.addEventListener("input", () => {
                const cur = readState(node);
                if (!visibleOptions(cur)[i]) return;
                cur.options[globalsOf(cur)[i]].value = vl.value;
                writeState(node, { options: cur.options });
                autoGrow(vl);
                const ok = readable(vl.value, readState(node).type);
                vl.classList.toggle("bad", !ok);
                warn.classList.toggle("hide", ok);
                renderTypes();
                fire();
            });

            ins.addEventListener("click", () => {
                const cur = readState(node);
                const vis = visibleOptions(cur);
                vis.splice(i + 1, 0, { name: "", value: "", category: cur.category });
                // 选中保持在同一 OPTION 上：上方插入会移位。
                commit({ options: mergeBack(cur, vis), index: cur.index > i ? cur.index + 1 : cur.index });
            });

            del.addEventListener("click", () => {
                const cur = readState(node);
                const vis = visibleOptions(cur);
                vis.splice(i, 1);
                // 删除选中行把选中移到接替它的位置（或最后一行）。搞错这点
                // 会让节点静默发出与面上名字不同的值。
                let idx = cur.index;
                if (i < idx) idx -= 1;
                else if (i === idx) idx = Math.min(i, vis.length - 1);
                commit({ options: mergeBack(cur, vis), index: Math.max(0, idx) });
            });

            grip.addEventListener("dragstart", (e) => {
                dragFrom = i;
                e.dataTransfer.effectAllowed = "move";
                try { e.dataTransfer.setData("text/plain", String(i)); } catch {}
            });
            // 无论拖拽如何结束都清掉。只有 drop 不够：释放在行间隙、列表内边距
            // 或面板外都不会触发它，dragFrom 残留指向某行——下次任何东西落到
            // 行上（文件、文本选择、任何拖拽，因为 dragover/drop 挂在 ROW 上
            // 对任何拖拽都触发）都会像那次残留的 grip 拖拽完成一样重排列表。
            // dragend 在 drop 之后触发，真实重排先拿到它的值。
            grip.addEventListener("dragend", () => {
                dragFrom = -1;
                for (const el2 of list.querySelectorAll(".drop-above, .drop-below")) {
                    el2.classList.remove("drop-above", "drop-below");
                }
            });
            row.addEventListener("dragover", (e) => {
                if (dragFrom < 0) return;
                e.preventDefault();
                const r = row.getBoundingClientRect();
                const below = e.clientY > r.top + r.height / 2;
                row.classList.toggle("drop-below", below);
                row.classList.toggle("drop-above", !below);
            });
            row.addEventListener("dragleave", () => {
                row.classList.remove("drop-above", "drop-below");
            });
            row.addEventListener("drop", (e) => {
                e.preventDefault();
                row.classList.remove("drop-above", "drop-below");
                if (dragFrom < 0 || dragFrom === i) { dragFrom = -1; return; }
                const r = row.getBoundingClientRect();
                let to = e.clientY > r.top + r.height / 2 ? i + 1 : i;
                const cur = readState(node);
                const vis = visibleOptions(cur);
                const moved = vis[dragFrom];
                // 按身份跨移动跟踪选中 OPTION，重排永不改变节点在发的条目。
                const selected = vis[cur.index];
                vis.splice(dragFrom, 1);
                if (dragFrom < to) to -= 1;
                vis.splice(to, 0, moved);
                dragFrom = -1;
                commit({ options: mergeBack(cur, vis), index: Math.max(0, vis.indexOf(selected)) });
            });
        });
    }

    // ── Footer ──────────────────────────────────────────────────────────
    // 没有 "Add option"。添加住在列表所在处：行上的 + 在其下插入，空列表自带
    // 按钮。第三条通向同一动作的路、停在它所作用的东西之外，只是噪音。

    const bExp = el("button", "sf-ddp-btn", "Export");
    bExp.title = "Save this list to a file you can load into another workflow";
    bExp.addEventListener("click", () => {
        const st = readState(node);
        const blob = new Blob([JSON.stringify(
            {
                sfnodes: "value_dropdown", version: 1, type: st.type,
                categories: st.categories, category: st.category,
                options: st.options.map((o) => ({ name: o.name, value: o.value, category: o.category })),
            }, null, 2)],
            { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "value-dropdown-list.json";
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    });

    const bImp = el("button", "sf-ddp-btn", "Import");
    bImp.title = "Load a list from a file. It replaces what is here.";
    bImp.addEventListener("click", () => {
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept = "application/json,.json";
        inp.addEventListener("change", async () => {
            const file = inp.files?.[0];
            if (!file) return;
            try {
                const data = JSON.parse(await file.text());
                const opts = Array.isArray(data?.options) ? data.options : null;
                if (!opts) { toast("That file does not hold a Dropdown list.", "error"); return; }

                // 分类：文件显式声明的列表权威；缺省（旧格式/手写文件）从行的
                // category 收集，再保证 default 恒在。行 category 不在列表中的
                // 补进去而非丢数据（与 readState 同规则）。
                const cats = [];
                for (const c of Array.isArray(data?.categories) ? data.categories : []) {
                    const s = typeof c === "string" ? c.trim() : "";
                    if (s && !cats.includes(s)) cats.push(s);
                }
                const clean = opts
                    .filter((o) => o && typeof o === "object" && !Array.isArray(o))
                    .map((o) => {
                        let c = (typeof o.category === "string" ? o.category.trim() : "") || "default";
                        if (!cats.includes(c)) cats.push(c);
                        return {
                            name: typeof o.name === "string" ? o.name : "",
                            value: typeof o.value === "string" ? o.value : (o.value == null ? "" : String(o.value)),
                            category: c,
                        };
                    });
                if (!clean.length) { toast("That file has no entries in it.", "error"); return; }
                if (!cats.includes("default")) cats.unshift("default");
                let cat = typeof data?.category === "string" ? data.category.trim() : "";
                if (!cats.includes(cat)) cat = cats[0] || "default";

                // 导出的文件总带着导出时的类型，所以 Import 可以改变本节点的
                // 类型——以及它的输出槽——与类型按钮一样。它也必须剪掉不再合适
                // 的线，否则节点保留着槽位不再支持的连接，不匹配只到运行时才
                // 浮出，远离引发它的 Import。
                const wasType = readState(node).type;
                commit({ options: clean, index: 0, type: data.type || wasType, categories: cats, category: cat });
                node._sfDropdownPending = null;
                node._sfDropdownCursor = null;
                syncOutput(node);
                const nowType = readState(node).type;
                // 只在类型真变了时剪线。剪线是破坏性的，导入同类型列表绝不能
                // 碰任何连接。
                const cut = nowType !== wasType ? dropIncompatibleLinks(node) : 0;

                const bits = [`Loaded ${clean.length} ${clean.length === 1 ? "entry" : "entries"} in ${cats.length} ${cats.length === 1 ? "category" : "categories"}`];
                if (nowType !== wasType) bits.push(`and switched this node to ${TYPE_LABELS[nowType].toLowerCase()}`);
                if (cut) bits.push(`- ${cut} ${cut === 1 ? "wire that no longer fits was" : "wires that no longer fit were"} unplugged`);
                toast(bits.join(" ") + ".", cut ? "warn" : "info");
            } catch {
                toast("That file could not be read.", "error");
            }
        });
        inp.click();
    });

    const bClr = el("button", "sf-ddp-btn", "Clear list");
    bClr.title = "Remove every entry from this category at once";
    bClr.addEventListener("click", async () => {
        const cur = readState(node);
        const vis = visibleOptions(cur);
        const n = vis.length;
        if (!n) { toast("The list is already empty."); return; }
        const ok = await askInPanel({
            title: "Clear the whole list?",
            message: `This removes ${n === 1 ? "the only entry" : `all ${n} entries`} from category "${cur.category}". `
                + "If you might want them back, Export first - Import brings the file straight back in.",
            okText: "Clear the list",
        });
        if (!ok) return;
        // 与手工选择相同的复位：持有的或已花的 In-order/Random 位置指向已不
        // 存在的列表。
        node._sfDropdownPending = null;
        node._sfDropdownCursor = null;
        commit({ options: mergeBack(cur, []), index: 0 });
    });

    const bDone = el("button", "sf-ddp-btn sf-ddp-push", "Done");
    bDone.addEventListener("click", closeDropdownPanel);

    foot.append(bExp, bImp, bClr, bDone);

    panel.append(title, body, foot);
    document.body.appendChild(panel);
    // 必须记录，而且这里漏过一次。没有它 _panel 保持 null，closeDropdownPanel
    // 什么也不移除，outsideClose 与 escClose 都早退，每次打开都在页面上再叠
    // 一个面板：四次打开后四个活面板共存，各自 handler 绑着各自陈旧的行动
    // 下标，一次点击能删掉没人看得见的面板里的行。单次打开看着完全正常，
    // 只有打开两次才会暴露。
    _panel = panel;
    renderCatBtn();
    renderTypes();
    renderModes();
    renderList();
    placeBeside(panel, getNodeScreenRect(node));
    makeDraggable(panel, title);
    startFollowing(panel, node);

    // 延迟注册，否则打开面板的那次点击立即关掉它。
    setTimeout(() => {
        document.addEventListener("pointerdown", outsideClose, true);
        document.addEventListener("keydown", escClose, true);
    }, 0);

    return panel;
}
