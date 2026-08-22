// ==========================================================================
// sf_workflows_ui.js - SF Workflows DOM 层
// ==========================================================================
//
// 浮动面板的纯 DOM 部分：窗口框架（拖拽/缩放/rect 记忆）、右键菜单、
// 封面（节点图绘制/手选/输出捕获）、拖拽守卫、网格卡片、文件夹侧栏、CSS。
// 无 app 依赖——需要时由主扩展注入（installOutputCoverCapture 的
// getActiveRel/saveMeta）。类名前缀 sf-wb-（源插件 pixwb-，防 CSS 互踩）。
//
// ==========================================================================

import { api } from "/scripts/api.js";
import {
    ancestorsOf, hasChildren, openSet, folderColor,
} from "./sf_workflows_lib.js";
import { copyText, el, injectCSSOnce, installWheelZoomPassthrough, sfApiUrl } from "./sf_common.js";

/** 微型 DOM 助手。每个面板都恰好想要这个。 */
// el 收敛于 sf_common（re-export 维持 sf_workflows.js 的既有导入路径）
export { el } from "./sf_common.js";

// ── "面板正在自我重绘吗？" ──────────────────────────────────────────────
export { sfApiUrl };
// 打开的改名框必须分清两件事：用户点走（提交输入）与重渲染把框从脚下拆走
// （保留并在之后放回）。两者都以普通 blur 到达。`input.isConnected` 看似
// 可行其实不行（Chrome 实测：移除聚焦元素时 blur 仍在元素已附加时触发，
// isConnected 在处理器内为 true、返回后才翻 false）。
// 由真正知道的渲染器回答。计数而非布尔：render() 重绘三列，嵌套调用不能
// 为外层清标志。
let renderDepth = 0;

export function markRendering(fn) {
    renderDepth++;
    try { return fn(); } finally { renderDepth--; }
}

export const isRendering = () => renderDepth > 0;

/**
 * 复制文本到剪贴板，成功返回 true。实现收敛于 sf_common.js（clipboard +
 * textarea 兜底双回退，只有一个副本）；此处 import 绑定后 re-export 透传，
 * 本模块内部（copyList）也直接使用。
 * navigator.clipboard 需要安全上下文，ComfyUI 常在 LAN 明文 http 上访问，
 * 整个 API 缺席——所以旧 textarea 技巧是兜底而非事后想。
 */
export { copyText };

// ── 拖拽与 rect（shared/floating_window 内联）────────────────────────────

/**
 * 在 `handle` 上开始指针拖拽。两道防线：
 *   1. setPointerCapture——该指针的每个事件都到本元素直到放手，哪怕在窗外
 *   2. buttons 抬起守卫：无按键的 move 意味着漏掉了 release，立即结束
 * `end` 幂等（守卫与真实释放都会调它）。
 */
export function startDrag(handle, e, onMove, onEnd) {
    if (e.button !== 0) return false;
    let done = false;
    const end = () => {
        if (done) return;
        done = true;
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", end);
        handle.removeEventListener("pointercancel", end);
        handle.removeEventListener("lostpointercapture", end);
        try { handle.releasePointerCapture(e.pointerId); } catch { /* 已释放 */ }
        onEnd?.();
    };
    const move = (ev) => {
        if (!(ev.buttons & 1)) { end(); return; }   // 释放丢了
        onMove(ev);
    };
    try { handle.setPointerCapture(e.pointerId); } catch { /* 旧构建：守卫仍覆盖 */ }
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
    handle.addEventListener("lostpointercapture", end);
    e.preventDefault();
    return true;
}

// ComfyUI 的浮动操作条（Run/Manager 那行）。面板在 y=60 打开会压住它，
// 藏起能再次关闭面板的按钮。每次打开重新测量（行可移动、高度不是我们的
// 硬编码）。
const TOOLBAR_GAP = 10;
const TOOLBAR_MAX_TOP = 220;

function toolbarFloor() {
    try {
        const bar = document.querySelector(".actionbar-container")
            || document.querySelector(".sf-wb-btn, .sf-wb-cmd")?.closest(".comfyui-button-group");
        if (!bar) return 0;
        const b = bar.getBoundingClientRect();
        if (!b.height || b.bottom > TOOLBAR_MAX_TOP) return 0;
        return Math.round(b.bottom + TOOLBAR_GAP);
    } catch {
        return 0;   // 面板打开绝不因找不到工具栏而失败
    }
}

/**
 * 面板的尺寸/位置持久化。`settingKey` 每面板唯一。
 */
export function makeRect({
    settingKey,
    minW = 420, minH = 280,
    prefW = 980, prefH = 756,
    edge = 24, homeX = 60, homeY = 70,
    sideDef = 204, sideMin = 130, sideMaxFrac = 0.55,
    saveDelay = 350,
    clearToolbar = true,
} = {}) {
    const sideMax = (winW) => Math.max(sideMin, Math.round(winW * sideMaxFrac));
    const floorY = () => (clearToolbar ? toolbarFloor() : 0);

    // 每次打开按视口计算而非烘焙——同一个人明天可能用笔记本开 ComfyUI
    function defaultRect() {
        const vw = window.innerWidth, vh = window.innerHeight;
        const top = Math.max(edge, floorY());
        const w = Math.max(minW, Math.min(prefW, vw - edge * 2));
        const h = Math.max(minH, Math.min(prefH, vh - top - edge));
        return {
            x: Math.max(edge, Math.min(homeX, vw - w - edge)),
            y: Math.max(top, Math.min(Math.max(homeY, top), vh - h - edge)),
            w, h, sw: sideDef,
        };
    }

    function clampRect(r) {
        const d = defaultRect();
        const vw = window.innerWidth, vh = window.innerHeight;
        // 工具栏地板也适用于已保存 rect：在此功能存在前压在顶栏上的面板
        // 否则会永远重新打开在自己的开关上面
        const top = Math.max(0, floorY());
        const w = Math.round(Math.max(minW, Math.min(r?.w ?? d.w, vw - edge)));
        const h = Math.round(Math.max(minH, Math.min(r?.h ?? d.h, vh - top - edge)));
        // 按当前宽度重钳：大窗口上加宽的侧栏不能在小窗口缩小后吞掉内容
        const sw = Math.round(Math.max(sideMin, Math.min(r?.sw ?? d.sw, sideMax(w))));
        return {
            ...(r && typeof r === "object" ? r : {}),
            x: Math.round(Math.max(0, Math.min(r?.x ?? d.x, vw - w))),
            y: Math.round(Math.max(top, Math.min(r?.y ?? d.y, Math.max(top, vh - h)))),
            w, h, sw,
        };
    }

    function readRect() {
        const raw = window.sfnodesGetSetting?.(settingKey, null);
        if (raw && typeof raw === "object") return clampRect(raw);
        if (typeof raw === "string") {
            try { return clampRect(JSON.parse(raw)); } catch { /* 落到默认 */ }
        }
        return defaultRect();
    }

    // 防抖：拖拽不会在每次 pointermove 都写设置
    let saveTimer = null;
    function saveRect(rect) {
        clearTimeout(saveTimer);
        saveTimer = setTimeout(() => {
            try { window.sfnodesSetSetting?.(settingKey, rect); } catch { /* 保存 rect 从不弄坏 UI */ }
        }, saveDelay);
    }

    return { defaultRect, clampRect, readRect, saveRect, sideMax, floorY, minW, minH };
}

// ── CSS（阶段 1：窗口/侧栏/网格/菜单/toast/封面/拖拽高亮）──────────────────

// ── 密度缩放 ──────────────────────────────────────────────────────────────
// 每个能被感知为"大或小"的尺寸都经 z() 乘以 --sfwb-k。刻意不用 CSS zoom 或
// transform：窗口在拖拽/缩放时以像素写自己的 left/top/width/height，缩放该
// 元素会让它按非写入数字渲染、resize 数学自我打架。calc() 让代码里每个
// 测量（getBoundingClientRect、gridTemplateColumns、拖拽命中测试）都按它
// 已假设的单位工作。变量设在 document.documentElement（右键菜单与 toast
// 是 <body> 的 fixed 子元素，不会从面板继承）。
const z = (n) => `calc(${n}px * var(--sfwb-k, 1))`;

export function injectWorkflowCSS() {
    injectCSSOnce("sf-wb-css", `
:root { --sfwb-k:1; --sfwb-acc:var(--sf-acc, #f66744); }
.sf-wb-win { position:fixed; z-index:9980; background:#1b1a19; border:1px solid #3d3936;
  border-radius:${z(10)}; box-shadow:0 20px 60px rgba(0,0,0,.6); flex-direction:column;
  color:#ddd; font:12px 'Segoe UI',sans-serif; overflow:hidden; display:none;
  min-width:560px; min-height:340px; }
.sf-wb-win * { box-sizing:border-box; }
.sf-wb-title { display:flex; align-items:center; gap:${z(8)}; padding:${z(8)} ${z(12)}; cursor:grab;
  background:#201f1e; border-bottom:1px solid #33302e; user-select:none; flex:0 0 auto; }
.sf-wb-title.sf-wb-dragging { cursor:grabbing; }
.sf-wb-name { font-weight:600; font-size:${z(13)}; color:#fff; display:flex; align-items:center; gap:${z(8)}; }
.sf-wb-logo { width:12px; height:12px; border-radius:3px; background:var(--sfwb-acc); display:inline-block; }
.sf-wb-count { color:#9a938f; font-weight:400; font-size:${z(11)}; }
.sf-wb-sp { flex:1; }
.sf-wb-wbtn { background:none; border:0; color:#aaa; font-size:${z(15)}; cursor:pointer; padding:${z(2)} ${z(8)}; border-radius:${z(4)}; }
.sf-wb-wbtn:hover { background:rgba(255,255,255,.1); color:#fff; }
.sf-wb-bar { display:flex; align-items:center; gap:${z(8)}; padding:${z(7)} ${z(10)}; border-bottom:1px solid #33302e; flex:0 0 auto; flex-wrap:wrap; }
.sf-wb-search { flex:1 1 ${z(180)}; display:flex; align-items:center; gap:${z(6)}; background:#141312;
  border:1px solid #3d3936; border-radius:${z(6)}; padding:${z(5)} ${z(9)}; min-width:${z(140)}; }
.sf-wb-search input { flex:1; background:transparent; border:0; outline:none; color:#e6e6e6; font:${z(12)} 'Segoe UI',sans-serif; }
.sf-wb-tbtn { background:rgba(255,255,255,.05); border:1px solid #4a4542; color:#cfcfcf; border-radius:${z(5)};
  padding:${z(5)} ${z(11)}; font:${z(12)} 'Segoe UI',sans-serif; cursor:pointer; white-space:nowrap; }
.sf-wb-tbtn:hover:not(:disabled) { border-color:var(--sfwb-acc); color:#fff; }
.sf-wb-tbtn:disabled { opacity:.45; cursor:default; }
.sf-wb-tbtn.sf-wb-primary { background:var(--sfwb-acc); border-color:var(--sfwb-acc); color:#fff; }
.sf-wb-tbtn.sf-wb-danger { border-color:#a8543f; color:#ff8d7d; background:rgba(168,84,63,.15); }
.sf-wb-seg { display:flex; border:1px solid #4a4542; border-radius:${z(5)}; overflow:hidden; flex:0 0 auto; }
.sf-wb-seg button { background:transparent; border:0; color:#a8a29e; padding:${z(5)} ${z(10)}; cursor:pointer; font:${z(12)} 'Segoe UI',sans-serif; }
.sf-wb-seg button.on { background:var(--sfwb-acc); color:#fff; }
.sf-wb-seg button + button { border-left:1px solid #4a4542; }
.sf-wb-sizeseg button { font-size:${z(11)}; min-width:${z(26)}; }
.sf-wb-body { flex:1; display:flex; min-height:0; position:relative; }
.sf-wb-side { width:190px; flex:none; background:#161514; border-right:1px solid #2e2b29; overflow-y:auto; padding:${z(8)} ${z(6)}; }
.sf-wb-sidegrip { flex:none; width:6px; cursor:col-resize; margin:0 -3px; z-index:2; background:transparent; }
.sf-wb-sidegrip:hover { background:var(--sfwb-acc); }
.sf-wb-main { flex:1; min-width:0; overflow-y:auto; padding:${z(10)}; background:#1e1d1c; }
.sf-wb-detail { flex:none; width:208px; background:#171615; border-left:1px solid #2e2b29; overflow-y:auto; }
.sf-wb-detail.hidden { display:none; }
.sf-wb-detgrip { flex:none; width:6px; cursor:col-resize; margin:0 -3px; z-index:2; background:transparent; }
.sf-wb-detgrip.hidden { display:none; }
.sf-wb-grip { position:absolute; right:0; bottom:0; width:18px; height:18px; cursor:nwse-resize; z-index:3; }
.sf-wb-foot { display:flex; align-items:center; gap:${z(8)}; padding:${z(6)} ${z(12)}; border-top:1px solid #33302e;
  background:#201f1e; font-size:${z(10.5)}; color:#8f8f8f; flex:0 0 auto; flex-wrap:wrap; }
.sf-wb-foot b { color:#c9c9c9; font-weight:600; }
.sf-wb-footsp { flex:1; }
.sf-wb-grouphead { font:600 ${z(9.5)} 'Segoe UI',sans-serif; letter-spacing:.1em; text-transform:uppercase;
  color:#6f6a66; padding:${z(8)} ${z(8)} ${z(4)}; }
.sf-wb-fold { display:flex; align-items:center; gap:${z(6)}; width:100%; text-align:left; padding:${z(5)} ${z(8)};
  border-radius:${z(5)}; cursor:pointer; background:transparent; border:0; color:#c9c5c2; font:${z(12)} 'Segoe UI',sans-serif; }
.sf-wb-fold:hover { background:rgba(255,255,255,.06); color:#fff; }
.sf-wb-fold.on { background:color-mix(in srgb, var(--sfwb-acc) 20%, transparent); color:#fff; }
.sf-wb-foldlbl { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-wb-cnt { font-size:${z(10.5)}; color:#8a8581; }
.sf-wb-fold.on .sf-wb-cnt { color:rgba(255,255,255,.75); }
.sf-wb-chev { font-size:${z(9)}; color:#8a8581; cursor:pointer; width:${z(12)}; text-align:center; flex:none; }
.sf-wb-chev-open { transform:rotate(90deg); }
.sf-wb-chevpad { width:${z(12)}; flex:none; }
.sf-wb-dot { width:8px; height:8px; border-radius:50%; flex:none; }
.sf-wb-favstar { color:#e0894b; flex:none; }
.sf-wb-nest { flex:none; }
.sf-wb-fold.sf-wb-dragging-me { opacity:.45; }
.sf-wb-fold.sf-wb-insert-above { box-shadow: inset 0 2px 0 0 var(--sfwb-acc); }
.sf-wb-fold.sf-wb-insert-below { box-shadow: inset 0 -2px 0 0 var(--sfwb-acc); }
.sf-wb-fold.sf-wb-droptarget { background:color-mix(in srgb, var(--sfwb-acc) 16%, transparent); }
.sf-wb-foldrename { flex:1; min-width:0; background:#141312; border:1px solid var(--sfwb-acc); border-radius:${z(4)};
  color:#e6e6e6; font:${z(12)} monospace; padding:${z(3)} ${z(6)}; outline:none; }
.sf-wb-grid { display:grid;
  grid-template-columns:repeat(auto-fill, minmax(${z(150)}, 1fr)); gap:${z(10)}; align-content:start; }
.sf-wb-list { display:flex; flex-direction:column; }
.sf-wb-card { background:#232120; border:1px solid #34312f; border-radius:${z(8)}; padding:${z(8)}; cursor:pointer;
  display:flex; flex-direction:column; gap:${z(6)}; min-width:0; }
.sf-wb-card:hover { border-color:#4c4744; }
.sf-wb-card.sel { border-color:var(--sfwb-acc); box-shadow:0 0 0 1px var(--sfwb-acc); }
.sf-wb-card.kbd { outline:2px solid var(--sfwb-acc); outline-offset:1px; }
.sf-wb-cov { width:100%; aspect-ratio:16/9; border-radius:${z(4)}; background:#141414; object-fit:cover; display:block; }
.sf-wb-cardname { font-size:${z(11.5)}; color:#e6e6e6; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-wb-cardmeta { font-size:${z(10)}; color:#8a8581; }
.sf-wb-row { display:flex; align-items:center; gap:${z(10)}; padding:${z(6)} ${z(10)}; border-radius:${z(6)}; cursor:pointer; }
.sf-wb-row:hover { background:rgba(255,255,255,.05); }
.sf-wb-row.sel { background:color-mix(in srgb, var(--sfwb-acc) 18%, transparent); }
.sf-wb-row.kbd { outline:2px solid var(--sfwb-acc); outline-offset:1px; }
.sf-wb-rowcov { width:${z(44)}; height:${z(28)}; border-radius:${z(3)}; background:#141414; object-fit:cover; flex:none; }
.sf-wb-rowname { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-wb-rowfold { color:#8a8581; font-size:${z(10.5)}; }
.sf-wb-openmark { position:absolute; top:6px; right:6px; width:8px; height:8px; border-radius:50%;
  background:#3ec371; }
.sf-wb-card { position:relative; }
.sf-wb-star { position:absolute; top:${z(14)}; right:${z(14)}; width:${z(20)}; height:${z(20)}; display:flex; align-items:center;
  justify-content:center; border-radius:50%; background:rgba(0,0,0,.55); color:#888; cursor:pointer;
  font-size:${z(12)}; z-index:2; }
.sf-wb-star.on { color:#e0894b; }
.sf-wb-rowstar { position:static; background:transparent; }
.sf-wb-rename { flex:1; min-width:0; background:#141312; border:1px solid var(--sfwb-acc); border-radius:${z(4)};
  color:#e6e6e6; font:${z(12)} monospace; padding:${z(3)} ${z(6)}; outline:none; }
.sf-wb-empty { padding:${z(30)}; text-align:center; color:#8a8581; }
.sf-wb-toast { position:absolute; left:50%; bottom:${z(10)}; transform:translateX(-50%); background:#2a2725;
  border:1px solid #4c4744; border-radius:${z(6)}; padding:${z(7)} ${z(14)}; font-size:${z(11.5)}; color:#eee;
  box-shadow:0 6px 18px rgba(0,0,0,.5); z-index:9; display:none; max-width:70%; }
.sf-wb-menu { position:fixed; z-index:9999; background:#232120; border:1px solid #4c4744; border-radius:${z(7)};
  padding:${z(4)}; box-shadow:0 12px 30px rgba(0,0,0,.6); min-width:${z(180)}; }
.sf-wb-menu button { display:block; width:100%; text-align:left; background:none; border:0; color:#cfcfcf;
  padding:${z(7)} ${z(12)}; border-radius:${z(5)}; cursor:pointer; font:${z(12)} 'Segoe UI',sans-serif; }
.sf-wb-menu button:hover:not(:disabled), .sf-wb-menu button:focus-visible { background:rgba(255,255,255,.08); color:#fff; }
.sf-wb-menu button:disabled { opacity:.45; cursor:default; }
.sf-wb-menu .sf-wb-menudanger { color:#ff8d7d; }
.sf-wb-menu .sf-wb-menudanger:hover { background:rgba(226,85,74,.18); color:#ffb0a5; }
.sf-wb-menusep { height:1px; background:#353230; margin:4px 2px; }
.sf-wb-btn { width:${z(36)}; height:${z(36)}; border-radius:${z(8)}; background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.12); display:flex; align-items:center; justify-content:center;
  cursor:pointer; color:#ccc; }
.sf-wb-btn:hover { background:var(--sfwb-acc); border-color:var(--sfwb-acc); color:#fff; }
.sf-wb-btn.sf-wb-btn-open { background:var(--sfwb-acc); border-color:var(--sfwb-acc); color:#fff; }
.sf-wb-btn-icon { width:${z(16)}; height:${z(16)}; border-radius:${z(3)}; background:currentColor;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 4h6v6H4zm10 0h6v6h-6zM4 14h6v6H4zm10 0h6v6h-6z'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 4h6v6H4zm10 0h6v6h-6zM4 14h6v6H4zm10 0h6v6h-6z'/%3E%3C/svg%3E");
  -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat; -webkit-mask-position:center; mask-position:center;
  -webkit-mask-size:contain; mask-size:contain; }
/* ── 详情面板 ── */
.sf-wb-detname { font-weight:600; font-size:${z(12.5)}; color:#fff; padding:${z(8)} ${z(10)} ${z(2)}; word-break:break-word; }
.sf-wb-detpath { font-size:${z(10.5)}; color:#8a8581; padding:0 ${z(10)} ${z(6)}; word-break:break-all; }
.sf-wb-detcov { width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:${z(4)}; background:#141414; display:block; }
.sf-wb-kv { display:flex; justify-content:space-between; gap:${z(8)}; padding:${z(2)} ${z(10)}; font-size:${z(11.5)}; color:#aaa; }
.sf-wb-kv b { color:#ddd; font-weight:600; text-align:right; }
.sf-wb-kv.sf-wb-warn { color:#ff8d7d; }
.sf-wb-kv.sf-wb-warn b { color:#ff8d7d; }
.sf-wb-modlist { display:flex; flex-direction:column; gap:${z(3)}; padding:0 ${z(10)} ${z(8)}; }
.sf-wb-mod { font:${z(10.5)}/1.4 monospace; color:#c9c5c2; word-break:break-all; }
.sf-wb-moddir { color:#8a8581; }
.sf-wb-modsep { color:#6a6561; }
.sf-wb-modname { color:#e6e6e6; }
.sf-wb-modext { color:var(--sfwb-acc); }
.sf-wb-note { width:100%; min-height:${z(70)}; resize:vertical; box-sizing:border-box; padding:${z(6)} ${z(8)};
  background:#141312; border:1px solid #3d3936; border-radius:${z(5)}; color:#e6e6e6;
  font:${z(11.5)}/1.5 monospace; outline:none; }
.sf-wb-note:focus { border-color:var(--sfwb-acc); }
.sf-wb-headrow { display:flex; align-items:center; gap:${z(6)}; padding:${z(8)} ${z(10)} ${z(4)}; }
.sf-wb-headrow .sf-wb-grouphead { padding:0; }
.sf-wb-copybtn { background:none; border:0; color:var(--sfwb-acc); font:11px 'Segoe UI',sans-serif; cursor:pointer; padding:1px 4px; }
.sf-wb-copybtn:hover { text-decoration:underline; }
.sf-wb-copybtn.done { color:#3ec371; }
.sf-wb-acts { display:flex; flex-wrap:wrap; gap:${z(5)}; padding:${z(8)} ${z(10)}; }
.sf-wb-btnstar { color:#8a8581; }
.sf-wb-btnstar.on { color:#e0894b; }
/* ── tidy 屏 ── */
.sf-wb-tidy { padding:${z(10)} ${z(14)}; }
.sf-wb-tdintro { font-size:${z(11)}; color:#8a8581; padding:0 ${z(2)} ${z(10)}; }
.sf-wb-tdsec { margin-bottom:14px; }
.sf-wb-tdhead { display:flex; align-items:baseline; gap:${z(8)}; padding:${z(6)} ${z(2)}; border-bottom:1px solid #33302e; }
.sf-wb-tdtitle { font-weight:600; font-size:${z(12.5)}; color:#fff; }
.sf-wb-tdcount { font-size:${z(10.5)}; color:#8a8581; }
.sf-wb-tdblurb { font-size:${z(10.5)}; color:#8a8581; padding:${z(4)} ${z(2)} ${z(8)}; }
.sf-wb-tdrow { display:flex; align-items:center; gap:${z(10)}; padding:${z(6)} ${z(8)}; border-radius:${z(6)}; cursor:pointer; }
.sf-wb-tdrow:hover { background:rgba(255,255,255,.05); }
.sf-wb-tdrow.sel { background:color-mix(in srgb, var(--sfwb-acc) 18%, transparent); }
.sf-wb-tdrow.kbd { outline:2px solid var(--sfwb-acc); outline-offset:1px; }
.sf-wb-tdrow.sf-wb-tddimmed { opacity:.5; }
.sf-wb-tdmid { flex:1; min-width:0; display:flex; flex-direction:column; gap:1px; }
.sf-wb-tdsub { font-size:${z(10)}; color:#8a8581; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-wb-tdfold { font-size:${z(10)}; color:#8a8581; flex:none; }
.sf-wb-tdacts { display:flex; gap:${z(5)}; flex:none; }
.sf-wb-tdbtn { background:rgba(255,255,255,.05); border:1px solid #4a4542; color:#cfcfcf; border-radius:${z(4)};
  padding:${z(4)} ${z(9)}; font:${z(11)} 'Segoe UI',sans-serif; cursor:pointer; white-space:nowrap; }
.sf-wb-tdbtn:hover { border-color:var(--sfwb-acc); color:#fff; }
.sf-wb-tdbtn.primary { background:var(--sfwb-acc); border-color:var(--sfwb-acc); color:#fff; }
.sf-wb-tdbtn.danger { border-color:#a8543f; color:#ff8d7d; background:rgba(168,84,63,.15); }
.sf-wb-tdgroup { border:1px solid #33302e; border-radius:${z(6)}; padding:${z(4)}; margin-bottom:${z(6)}; }
`);
}

// ── 窗口 ──────────────────────────────────────────────────────────────────

const RECT_SETTING = "sfnodes.Workflows.Rect";
const MIN_W = 560;
const MIN_H = 340;
const PREF_W = 1040;
const PREF_H = 720;
const EDGE = 24;
const HOME_X = 80;
const HOME_Y = 60;
const SIDE_DEF = 190;
const SIDE_MIN = 120;
const SIDE_MAX_FRAC = 0.45;
const DET_DEF = 208;
const DET_MIN = 150;
const DET_MAX_FRAC = 0.5;
const detMax = (winW) => Math.max(DET_MIN, Math.round(winW * DET_MAX_FRAC));

const RECT = makeRect({
    settingKey: RECT_SETTING,
    minW: MIN_W, minH: MIN_H, prefW: PREF_W, prefH: PREF_H,
    edge: EDGE, homeX: HOME_X, homeY: HOME_Y,
    sideDef: SIDE_DEF, sideMin: SIDE_MIN, sideMaxFrac: SIDE_MAX_FRAC,
});
const { clampRect, readRect, saveRect, sideMax, floorY } = RECT;

export function createWorkflowWindow({ onRender, onClose }) {
    injectWorkflowCSS();
    installDropGuard();

    const win = el("div", "sf-wb-win");
    win.style.display = "none";

    const title = el("div", "sf-wb-title");
    const name = el("div", "sf-wb-name");
    const count = el("span", "sf-wb-count", "");
    name.append(el("span", "sf-wb-logo"), el("span", null, "Workflows"), count);
    const closeBtn = el("button", "sf-wb-wbtn", "✕");
    closeBtn.type = "button";
    closeBtn.title = "Close (Esc)";
    title.append(name, el("div", "sf-wb-sp"), closeBtn);

    const bar = el("div", "sf-wb-bar");
    const body = el("div", "sf-wb-body");
    const side = el("div", "sf-wb-side");
    const sideGrip = el("div", "sf-wb-sidegrip");
    sideGrip.title = "Drag to resize the list. Double-click to reset.";
    const main = el("div", "sf-wb-main");
    const detail = el("div", "sf-wb-detail");
    const detGrip = el("div", "sf-wb-detgrip");
    detGrip.title = "Drag to resize. Double-click to reset.";
    body.append(side, sideGrip, main, detGrip, detail);

    const foot = el("div", "sf-wb-foot");
    const grip = el("div", "sf-wb-grip");
    win.append(title, bar, body, foot, grip);
    document.body.appendChild(win);

    let rect = readRect();
    let wasNarrow = null;
    const applyRect = () => {
        win.style.left = rect.x + "px";
        win.style.top = rect.y + "px";
        win.style.width = rect.w + "px";
        win.style.height = rect.h + "px";
        rect.sw = Math.max(SIDE_MIN, Math.min(rect.sw ?? SIDE_DEF, sideMax(rect.w)));
        side.style.width = rect.sw + "px";
        rect.dw = Math.max(DET_MIN, Math.min(rect.dw ?? DET_DEF, detMax(rect.w)));
        detail.style.width = rect.dw + "px";
        // 详情面板是窄窗口上最先消失的：三列在 560px 里留给网格的太少。
        // 其 grip 跟着走，否则会有一个看不见东西的把手
        const narrow = rect.w < 760;
        detail.classList.toggle("hidden", narrow);
        detGrip.classList.toggle("hidden", narrow);
        // 加宽越过阈值会揭示详情面板，但缩放路径刻意跳过重渲染——面板
        // 出现却空着直到别的事触发重绘。在可见性实际变化的帧上请求 REPAINT
        if (wasNarrow !== null && wasNarrow !== narrow && !narrow) onRender?.({ repaintOnly: true });
        wasNarrow = narrow;
    };
    applyRect();

    const onDragEnd = () => {
        title.classList.remove("sf-wb-dragging");
        saveRect(rect);
    };

    title.addEventListener("pointerdown", (e) => {
        if (e.target.closest(".sf-wb-wbtn")) return;
        const ox = e.clientX - win.offsetLeft;
        const oy = e.clientY - win.offsetTop;
        if (!startDrag(title, e, (ev) => {
            rect.x = Math.max(0, Math.min(ev.clientX - ox, window.innerWidth - Math.min(rect.w, 160)));
            rect.y = Math.max(floorY(), Math.min(ev.clientY - oy, window.innerHeight - 40));
            applyRect();
        }, onDragEnd)) return;
        title.classList.add("sf-wb-dragging");
    });

    grip.addEventListener("pointerdown", (e) => {
        const left = win.offsetLeft, top = win.offsetTop;
        const ox = e.clientX - (left + win.offsetWidth);
        const oy = e.clientY - (top + win.offsetHeight);
        startDrag(grip, e, (ev) => {
            rect.w = Math.max(MIN_W, Math.min(ev.clientX - ox - left, window.innerWidth - left));
            rect.h = Math.max(MIN_H, Math.min(ev.clientY - oy - top, window.innerHeight - top));
            applyRect();
            onRender?.({ resizeOnly: true });
        }, onDragEnd);
        e.stopPropagation();
    });

    sideGrip.addEventListener("pointerdown", (e) => {
        const bodyLeft = body.getBoundingClientRect().left;
        startDrag(sideGrip, e, (ev) => {
            rect.sw = Math.round(Math.max(SIDE_MIN, Math.min(ev.clientX - bodyLeft, sideMax(rect.w))));
            side.style.width = rect.sw + "px";
        }, onDragEnd);
        sideGrip.classList.add("sf-wb-dragging");
        e.stopPropagation();
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach((t) =>
        sideGrip.addEventListener(t, () => sideGrip.classList.remove("sf-wb-dragging")));
    sideGrip.addEventListener("dblclick", () => {
        rect.sw = SIDE_DEF;
        applyRect();
        saveRect(rect);
    });

    // 详情面板也可拖（模型文件名很长，固定 208px 列把它们都折成三行）
    detGrip.addEventListener("pointerdown", (e) => {
        const bodyRight = body.getBoundingClientRect().right;
        startDrag(detGrip, e, (ev) => {
            rect.dw = Math.round(Math.max(DET_MIN, Math.min(bodyRight - ev.clientX, detMax(rect.w))));
            detail.style.width = rect.dw + "px";
        }, onDragEnd);
        detGrip.classList.add("sf-wb-dragging");
        e.stopPropagation();
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach((t) =>
        detGrip.addEventListener(t, () => detGrip.classList.remove("sf-wb-dragging")));
    detGrip.addEventListener("dblclick", () => {
        rect.dw = DET_DEF;
        applyRect();
        saveRect(rect);
    });

    window.addEventListener("resize", () => {
        if (win.style.display === "none") return;
        rect = clampRect(rect);
        applyRect();
    });

    // 面板内点击不得到达 canvas，浏览不会取消选择面板打开前选中的东西
    win.addEventListener("pointerdown", (e) => e.stopPropagation());

    // ── 保持键盘活着 ──
    // 规则：除非在编辑什么，否则搜索框持有焦点。打字保持过滤、箭头持续
    // 工作，无论最后点在哪里。
    win.addEventListener("mousedown", (e) => {
        if (e.target.closest("input, textarea, select, [contenteditable]")) return;
        setTimeout(() => {
            const a = document.activeElement;
            if (a && win.contains(a) && a.matches("input, textarea, [contenteditable]")) return;
            // 右键菜单在 document.body、本窗外，聚焦在其中是刻意的
            if (a && a.closest(".sf-wb-menu")) return;
            bar.querySelector("input")?.focus({ preventScroll: true });
        }, 0);
    });

    // ── toast ──
    let toastEl = null, toastTimer = null;
    function toast(message) {
        // 空消息隐藏它。显示空框从不是调用者的意思
        if (!message) {
            if (toastEl) toastEl.style.display = "none";
            clearTimeout(toastTimer);
            return;
        }
        if (!toastEl) {
            toastEl = el("div", "sf-wb-toast");
            body.appendChild(toastEl);
        }
        toastEl.textContent = message;
        toastEl.style.display = "block";
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { if (toastEl) toastEl.style.display = "none"; }, 2600);
    }

    const api = {
        el: win, bar, side, main, detail, foot, title, count,
        isOpen: () => win.style.display !== "none",
        toast,
        setCount: (text) => { count.textContent = text; },
        isDetailVisible: () => !detail.classList.contains("hidden"),
        focusSearch: () => bar.querySelector("input")?.focus({ preventScroll: true }),
        open() {
            rect = clampRect(rect);
            applyRect();
            win.style.display = "flex";
            onRender?.();
            setTimeout(() => bar.querySelector("input")?.focus(), 20);
        },
        close() {
            // 隐藏面板会 blur 其中的焦点，若那是打开的改名框，blur 会提交
            // 半输入的名字。关闭不是"点走"，所以用渲染标志说"这不是用户的
            // 回答"。
            markRendering(() => { win.style.display = "none"; });
            const q = bar.querySelector("input");
            // 清空盒子 ≠ 清空搜索：过滤器在面板自己的状态里，关在搜索中的
            // 面板再开，看似未过滤却仍藏着列表的大部分
            if (q && q.value) { q.value = ""; q.dispatchEvent(new Event("input", { bubbles: true })); }
            if (toastEl) toastEl.style.display = "none";
            onClose?.();
        },
        toggle() { api.isOpen() ? api.close() : api.open(); },
        destroy() { win.remove(); },
    };

    // Esc 关闭，但仅当焦点在内，否则吞掉整个应用的 Escape
    win.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            const q = bar.querySelector("input");
            // Esc 先清搜索：一个键同时丢查询与窗口是错误量的撤销
            if (q && q.value && document.activeElement === q) {
                q.value = "";
                q.dispatchEvent(new Event("input", { bubbles: true }));
                e.stopPropagation();
                return;
            }
            e.stopPropagation();
            api.close();
        }
    });

    closeBtn.addEventListener("click", () => api.close());
    return api;
}

// ── 右键菜单 ──────────────────────────────────────────────────────────────

let menuEl = null;
let cleanup = null;
let returnFocus = null;
let focusHome = null;

export function setMenuFocusHome(fn) { focusHome = fn; }

export function closeContextMenu() {
    if (menuEl) { menuEl.remove(); menuEl = null; }
    if (cleanup) { cleanup(); cleanup = null; }
    // 把焦点还给面板。回调而非打开时聚焦的元素：到那时已太迟读——mousedown
    // 在 contextmenu 前运行并 blur 原焦点
    const back = returnFocus;
    returnFocus = null;
    try { back?.(); } catch { /* 面板已消失 */ }
}

/**
 * @param items [{label, fn, disabled, danger}] - null 项画分隔线
 */
export function openContextMenu(x, y, items, onClose) {
    closeContextMenu();
    menuEl = el("div", "sf-wb-menu");
    for (const it of items) {
        if (!it) { menuEl.append(el("div", "sf-wb-menusep")); continue; }
        const b = el("button", it.danger ? "sf-wb-menudanger" : null, it.label);
        b.type = "button";
        if (it.disabled) b.disabled = true;
        else b.addEventListener("click", () => { closeContextMenu(); it.fn(); });
        menuEl.append(b);
    }
    document.body.append(menuEl);

    // 保持在屏内
    const r = menuEl.getBoundingClientRect();
    menuEl.style.left = Math.round(Math.max(6, Math.min(x, window.innerWidth - r.width - 8))) + "px";
    menuEl.style.top = Math.round(Math.max(6, Math.min(y, window.innerHeight - r.height - 8))) + "px";

    // 捕获阶段：菜单动作重渲染面板、拆掉被点元素，冒泡阶段的"是否在菜单内"
    // 测试已读 false
    const away = (e) => { if (menuEl && !menuEl.contains(e.target)) closeContextMenu(); };

    // ── 键盘 ──
    const options = () => [...menuEl.querySelectorAll("button:not(:disabled)")];
    const step = (delta) => {
        const list = options();
        if (!list.length) return;
        const at = list.indexOf(document.activeElement);
        const next = at < 0 ? (delta > 0 ? 0 : list.length - 1)
                            : (at + delta + list.length) % list.length;
        list[next].focus();
    };
    const keys = (e) => {
        if (!menuEl) return;
        switch (e.key) {
            case "Escape":   e.stopPropagation(); closeContextMenu(); break;
            case "ArrowDown": e.preventDefault(); e.stopPropagation(); step(1); break;
            case "ArrowUp":   e.preventDefault(); e.stopPropagation(); step(-1); break;
            case "Home":      e.preventDefault(); e.stopPropagation(); options()[0]?.focus(); break;
            case "End":       e.preventDefault(); e.stopPropagation(); options().pop()?.focus(); break;
            // Tab 会把焦点移出仍打开的菜单
            case "Tab":       e.preventDefault(); e.stopPropagation(); step(e.shiftKey ? -1 : 1); break;
            default: break;
        }
    };

    // 延迟，否则打开菜单的那次 pointerdown 又把它关了
    const armed = setTimeout(() => {
        document.addEventListener("pointerdown", away, true);
        document.addEventListener("keydown", keys, true);
    }, 0);
    cleanup = () => {
        clearTimeout(armed);
        document.removeEventListener("pointerdown", away, true);
        document.removeEventListener("keydown", keys, true);
    };

    returnFocus = onClose || focusHome;
    options()[0]?.focus();
}

// ── 拖拽守卫 ──────────────────────────────────────────────────────────────
// 带 text/plain 的拖拽使页面上每个文本框都是该字符串的原生 drop 目标。
// 守卫只取消我们的拖拽落空的情况；普通文本拖拽不带我们的类型，不受影响。

/** 一个文件夹被拖来重排。 */
export const FOLDER_MIME = "application/x-sfnodes-workflows-folder";
/** 一个或多个工作流卡片被拖入文件夹。 */
export const CARD_MIME = "application/x-sfnodes-workflows-card";

export const DROP_TARGET_ATTR = "wfdrop";
const VALID_TARGET = "[data-wfdrop]";

function hasType(e, mime) {
    if (!e.dataTransfer) return false;
    try { return [...e.dataTransfer.types].includes(mime); } catch { return false; }
}

export const isFolderDrag = (e) => hasType(e, FOLDER_MIME);
export const isOurDrag = (e) => hasType(e, FOLDER_MIME) || hasType(e, CARD_MIME);

function isStrayDrop(e) {
    if (!isOurDrag(e)) return false;
    const t = e.target;
    if (t && typeof t.closest === "function" && t.closest(VALID_TARGET)) return false;
    return true;
}

let installed = false;

/** 取消我们的拖拽落在任何非真实文件夹行处。捕获阶段、幂等。 */
export function installDropGuard() {
    if (installed) return;
    installed = true;

    document.addEventListener("dragover", (e) => {
        if (!isStrayDrop(e)) return;
        try { e.dataTransfer.dropEffect = "none"; } catch { /* 某些状态只读 */ }
    }, true);

    document.addEventListener("drop", (e) => {
        if (!isStrayDrop(e)) return;
        e.preventDefault();
        e.stopPropagation();
    }, true);
}

// ── 封面 ──────────────────────────────────────────────────────────────────

const ACHROMATIC = 0.06;
const LIFT_L = 0.62;
const GREY_L = 0.42;
const LIFT_S = 0.45;
const NO_COLOUR = "#57534f";
const _liftCache = new Map();

function lift(hex) {
    // 颜色来自网上下载的工作流文件，数字/对象曾在此 .slice 上抛错
    if (!hex || typeof hex !== "string") return NO_COLOUR;
    const hit = _liftCache.get(hex);
    if (hit) return hit;

    let h = hex.slice(1);
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    if (h.length !== 6) return NO_COLOUR;
    const r = parseInt(h.slice(0, 2), 16) / 255;
    const g = parseInt(h.slice(2, 4), 16) / 255;
    const b = parseInt(h.slice(4, 6), 16) / 255;

    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let hue = 0, sat = 0;
    const l0 = (mx + mn) / 2;
    if (d) {
        sat = l0 > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
        hue = mx === r ? ((g - b) / d + (g < b ? 6 : 0))
            : mx === g ? ((b - r) / d + 2)
            : ((r - g) / d + 4);
        hue /= 6;
    }
    // 灰进灰出。只有真有色相的颜色才被提饱和
    const grey = sat < ACHROMATIC;
    const s = grey ? 0 : Math.max(sat, LIFT_S);
    const l = grey ? GREY_L : LIFT_L;
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const ch = (t) => {
        t = (t + 1) % 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 0.5) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
    };
    const to = (v) => Math.round(v * 255).toString(16).padStart(2, "0");
    const out = "#" + to(ch(hue + 1 / 3)) + to(ch(hue)) + to(ch(hue - 1 / 3));
    _liftCache.set(hex, out);
    return out;
}

/** 绘制图映射。按元素真实盒的设备像素缩放，否则高 DPI 屏上发虚。 */
export function drawMap(canvas, map) {
    const w = canvas.clientWidth || 120;
    const h = canvas.clientHeight || 64;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = "#141414";
    ctx.fillRect(0, 0, w, h);

    if (!Array.isArray(map) || !map.length) {
        // 不可读或空工作流仍得到诚实可见的东西而非网格里的空洞
        ctx.strokeStyle = "#2e2e2e";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(w * 0.3, h * 0.5); ctx.lineTo(w * 0.7, h * 0.5);
        ctx.stroke();
        return;
    }

    // 条目来自不可信文件。null 或短的曾以 e[0] 在 requestAnimationFrame 内
    // 抛错，无人捕获，卡片留下空白封面与一条 console 错误
    const boxes = map.filter((e) => Array.isArray(e) && e.length >= 4
        && Number.isFinite(+e[0]) && Number.isFinite(+e[1])
        && Number.isFinite(+e[2]) && Number.isFinite(+e[3]));
    if (!boxes.length) return;

    const pad = 6;
    const iw = Math.max(1, w - pad * 2);
    const ih = Math.max(1, h - pad * 2);

    // 先画连线、盒子压在上面。按阅读顺序近似为盒心之间的线：真实链接表
    // 不随映射携带，120x64 下图的印象就是全部可读的
    ctx.strokeStyle = "rgba(120,150,180,.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < boxes.length; i++) {
        const a = boxes[i - 1], b = boxes[i];
        ctx.moveTo(pad + (a[0] + a[2] / 2) * iw, pad + (a[1] + a[3] / 2) * ih);
        ctx.lineTo(pad + (b[0] + b[2] / 2) * iw, pad + (b[1] + b[3] / 2) * ih);
    }
    ctx.stroke();

    for (const e of boxes) {
        const x = pad + e[0] * iw;
        const y = pad + e[1] * ih;
        const bw = Math.max(2, e[2] * iw);
        const bh = Math.max(2, e[3] * ih);
        const col = e[4];
        ctx.fillStyle = lift(col);
        ctx.globalAlpha = col ? 0.95 : 0.5;
        const r = Math.min(2, bw / 2, bh / 2);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, y, bw, bh, r);
        else ctx.rect(x, y, bw, bh);
        ctx.fill();
    }
    ctx.globalAlpha = 1;
}

const PICTURE_EXTS = ["png", "jpg", "jpeg", "webp", "gif", "bmp", "avif"];

/** 这个文件名能放进 <img> 吗？只看扩展名。 */
export function isPictureName(filename) {
    const name = String(filename || "").split("?")[0];
    const dot = name.lastIndexOf(".");
    if (dot < 0) return false;
    return PICTURE_EXTS.includes(name.slice(dot + 1).toLowerCase());
}

/** 一张卡片的图片该来自哪里，若有。 */
export function coverFor(entry, meta) {
    const hand = meta?.covers?.[entry.rel];
    if (hand && hand.kind === "file" && hand.file) {
        // URL 里的版本号让图片可硬缓存、替换瞬间更新（文件名永不改变）
        return { kind: "image", url: sfApiUrl(`/api/sfnodes/workflows/cover/${encodeURIComponent(hand.file)}?v=${hand.v || 1}`) };
    }
    if (hand && hand.kind === "file" && hand.url) return { kind: "image", url: hand.url };
    if (hand && hand.kind === "output" && hand.filename && isPictureName(hand.filename)) {
        const p = new URLSearchParams({
            filename: hand.filename,
            subfolder: hand.subfolder || "",
            type: hand.type || "output",
        });
        return { kind: "image", url: sfApiUrl(`/view?${p.toString()}`) };
    }
    return { kind: "map" };
}

/** 这个工作流有没有用户手选的图片？ */
export function hasHandCover(entry, meta) {
    const hand = meta?.covers?.[entry.rel];
    return !!(hand && hand.kind === "file" && (hand.file || hand.url));
}

/** 卡片封面元素：有真图用图，否则画图映射。 */
export function coverEl(entry, state, cls) {
    const c = coverFor(entry, state.meta);
    if (c.kind === "image") {
        const img = el("img", cls);
        img.loading = "lazy";
        img.src = c.url;
        img.alt = "";
        // 已删除输出记录的封面不得在网格里留破图图标：回退到绘制图
        img.addEventListener("error", () => {
            const cv = el("canvas", cls);
            img.replaceWith(cv);
            requestAnimationFrame(() => drawMap(cv, entry.map));
        }, { once: true });
        return img;
    }
    const cv = el("canvas", cls);
    requestAnimationFrame(() => drawMap(cv, entry.map));
    return cv;
}

// ── 记住工作流产出（输出封面捕获）────────────────────────────────────────
// run 结束我们已知哪个工作流开着，事件带着它写的图片。记录这一对就够封面
// 随用户工作出现，无需回填、无需扫描输出目录。
let _capInstalled = false;

export function installOutputCoverCapture({ getActiveRel, saveMeta }) {
    if (_capInstalled) return;
    _capInstalled = true;

    api.addEventListener("executed", (ev) => {
        try {
            const images = ev?.detail?.output?.images;
            if (!Array.isArray(images) || !images.length) return;
            const rel = getActiveRel();
            if (!rel) return;   // 未保存工作流没有可钉住的文件
            const img = images.find((i) => i && i.filename && (i.type || "output") !== "temp"
                                             && isPictureName(i.filename));
            if (!img) return;
            const pending = {};
            pending[rel] = { kind: "output", filename: img.filename, subfolder: img.subfolder || "", type: img.type || "output" };
            // 防抖：批量 run 每个输出节点触发一次，各自写一次是浪费
            clearTimeout(_capTimer);
            _capTimer = setTimeout(() => flushOutputCovers(pending, saveMeta), 1200);
        } catch {
            // 封面缩略图绝不抛进 ComfyUI 的事件循环
        }
    });
}

let _capTimer = null;

async function flushOutputCovers(batch, saveMeta) {
    try { await saveMeta({ covers: batch }); } catch { /* 错过一张封面不值得一条消息 */ }
}

// ── 网格卡片 ──────────────────────────────────────────────────────────────

const fmtWhen = (secs) => {
    if (!secs) return "";
    const d = (Date.now() - secs * 1000) / 1000;
    if (d < 90) return "just now";
    if (d < 3600) return Math.round(d / 60) + " min ago";
    if (d < 86400) return Math.round(d / 3600) + "h ago";
    if (d < 86400 * 7) return Math.round(d / 86400) + " days ago";
    return new Date(secs * 1000).toLocaleDateString();
};

export function renderGrid(main, state, H) {
    // 清空会同步触发打开的改名框的 blur——标志必须恰好盖住这条语句
    markRendering(() => { main.textContent = ""; });
    const list = state.visible;

    if (!list.length) {
        main.append(el("div", "sf-wb-empty", state.query
            ? `Nothing matches "${state.query}".`
            : "Nothing in here yet."));
        dropRename(true);
        return;
    }

    const wrap = el("div", state.view === "list" ? "sf-wb-list" : "sf-wb-grid");
    const openNow = new Set(state.openPaths);

    for (const entry of list) {
        const card = el("div", state.view === "list" ? "sf-wb-row" : "sf-wb-card");
        card.dataset.rel = entry.rel;
        if (state.selected.has(entry.rel)) card.classList.add("sel");
        if (state.kbdRel === entry.rel) card.classList.add("kbd");
        card.title = entry.error ? `${entry.name}\n${entry.error}` : entry.name;

        if (state.view === "list") {
            card.append(coverEl(entry, state, "sf-wb-rowcov"));
            card.append(el("span", "sf-wb-rowname", entry.name));
            const right = el("span", "sf-wb-rowfold",
                `${entry.folder || ""}  ${fmtWhen(entry.modified)}`.trim());
            card.append(right);
        } else {
            card.append(coverEl(entry, state, "sf-wb-cov"));
            card.append(el("div", "sf-wb-cardname", entry.name));
            card.append(el("div", "sf-wb-cardmeta",
                entry.error ? "unreadable" : `${fmtWhen(entry.modified)} · ${entry.node_count} nodes`));
        }

        if (openNow.has(entry.rel)) {
            const mark = el("div", "sf-wb-openmark");
            mark.title = "Open right now";
            card.append(mark);
        }

        const fav = state.favourites.has(entry.rel);
        const star = el("div", "sf-wb-star" + (state.view === "list" ? " sf-wb-rowstar" : "") + (fav ? " on" : ""),
                        fav ? "★" : "☆");
        star.title = fav ? "Remove from favourites" : "Add to favourites";
        star.addEventListener("click", (e) => { e.stopPropagation(); H.onStar(entry); });
        // 星星上的两次快点击会到达卡片的 dblclick 打开工作流
        star.addEventListener("dblclick", (e) => e.stopPropagation());
        card.append(star);

        card.addEventListener("click", (e) => H.onSelect(entry, e));
        card.addEventListener("dblclick", () => H.onOpen(entry));
        card.addEventListener("contextmenu", (e) => { e.preventDefault(); H.onContext(entry, e); });

        // ── 拖入文件夹 ──
        card.draggable = true;
        card.addEventListener("dragstart", (e) => {
            // 绝不劫持对星星或改名框的点击
            const t = e.target;
            const tag = (t.tagName || "").toLowerCase();
            if (tag === "input" || tag === "textarea" || t.classList?.contains("sf-wb-star")) {
                e.preventDefault();
                return;
            }
            H.onDragStart(entry, e);
        });

        wrap.append(card);
    }
    main.append(wrap);
    // 最后、所有卡片入 DOM 后：重渲染前打开的改名框回到原位，输入仍在
    restoreRename(main);
}

// 正在进行的改名，能挺过网格在脚下重建。任何东西都可能触发重渲染——后台
// run 完成刷新封面、详情面板切星标——每次都在未告知的情况下扔掉半输入名
let activeRename = null;
let onRenameLost = null;

export function setRenameLostNotifier(fn) { onRenameLost = fn; }

/** 忘记进行中的改名，不提交。面板关闭与行消失时调用。 */
export function dropRename(tell) {
    if (!activeRename) return;
    const { currentName } = activeRename;
    activeRename = null;
    if (tell) { try { onRenameLost?.(currentName); } catch { /* 无面板 */ } }
}

/** 重渲染后放回改名框。每次网格渲染末尾调用；无进行中改名时不做。 */
export function restoreRename(main) {
    if (!activeRename) return;
    const { rel, value, currentName, commit } = activeRename;
    if (!rowFor(main, rel)) { dropRename(true); return; }
    beginRename(main, rel, currentName, commit, value);
}

function rowFor(main, rel) {
    const sel = `[data-rel="${CSS.escape(rel)}"]`;
    // tidy 屏一行可出现多次（每问题一节），只有标记 data-rename 的行提供
    // Rename——否则编辑框会落在先渲染的那节
    return main.querySelector(sel + "[data-rename]") || main.querySelector(sel) || null;
}

/** 把卡片名字换成输入框。Enter 提交、Escape 取消。 */
export function beginRename(main, rel, currentName, commit, startValue) {
    const card = rowFor(main, rel);
    if (!card) return;
    if (card.querySelector("input")) return;
    const nameEl = card.querySelector(".sf-wb-cardname") || card.querySelector(".sf-wb-rowname");
    if (!nameEl) return;

    const input = el("input", "sf-wb-rename");
    const resuming = startValue !== undefined;
    input.value = resuming ? startValue : currentName;
    const restore = () => { input.replaceWith(nameEl); };
    nameEl.replaceWith(input);
    input.focus();
    if (resuming) input.setSelectionRange(input.value.length, input.value.length);
    else input.select();

    activeRename = { rel, currentName, commit, value: input.value };

    let done = false;
    const finish = (save) => {
        if (done) return;
        done = true;
        activeRename = null;
        const value = input.value.trim();
        restore();
        if (save && value && value !== currentName) commit(value);
    };
    input.addEventListener("input", () => {
        if (activeRename) activeRename.value = input.value;
    });
    input.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Enter") finish(true);
        else if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => {
        // 重渲染拆掉了框而非用户点走。没有此测试，无关刷新会提交已输入的
        // 内容，把文件改成一个半完成的名字
        if (isRendering()) return;
        finish(true);
    });
    input.addEventListener("click", (e) => e.stopPropagation());
    input.addEventListener("dblclick", (e) => e.stopPropagation());
}

// ── 详情面板 ──────────────────────────────────────────────────────────────
// 打开前需要知道的工作流信息。缺失节点行是这里最值得的位置：加载后才发现
// 它跑不了、丢失画布上的东西，正是这个面板要消除的烦恼。

export function renderDetail(pane, state, H) {
    pane.textContent = "";
    const rels = [...state.selected];

    if (!rels.length) {
        pane.append(el("div", "sf-wb-empty", "Pick a workflow to see what is in it."));
        return;
    }

    if (rels.length > 1) {
        pane.append(el("div", "sf-wb-detname", `${rels.length} workflows selected`));
        pane.append(el("div", "sf-wb-detpath", "Drag them onto a folder to move them together."));
        const acts = el("div", "sf-wb-acts");
        const del = el("button", "sf-wb-tbtn sf-wb-danger", "Delete all");
        del.type = "button";
        del.addEventListener("click", () => H.onDeleteMany(rels));
        acts.append(del);
        pane.append(acts);
        return;
    }

    const entry = state.byRel.get(rels[0]);
    if (!entry) {
        // 选中但已不在索引——Explorer 里删了，或另一标签页改名
        pane.append(el("div", "sf-wb-empty",
            "That workflow is not there any more. It may have been renamed or deleted."));
        return;
    }

    const c = coverFor(entry, state.meta);
    if (c.kind === "image") {
        const img = el("img", "sf-wb-detcov");
        img.src = c.url;
        img.alt = "";
        // 与网格同款的保险：封面文件已被删时不留破图图标
        img.addEventListener("error", () => {
            const cv = el("canvas", "sf-wb-detcov");
            img.replaceWith(cv);
            requestAnimationFrame(() => drawMap(cv, entry.map));
        }, { once: true });
        pane.append(img);
    } else {
        const cv = el("canvas", "sf-wb-detcov");
        pane.append(cv);
        requestAnimationFrame(() => drawMap(cv, entry.map));
    }

    pane.append(el("div", "sf-wb-detname", entry.name));
    pane.append(el("div", "sf-wb-detpath", entry.folder ? "in " + entry.folder : "not in a folder"));

    const kv = (label, value, warn) => {
        const r = el("div", "sf-wb-kv" + (warn ? " sf-wb-warn" : ""));
        r.append(el("span", null, label), el("b", null, value));
        pane.append(r);
    };

    if (entry.error) {
        kv("Problem", entry.error, true);
    } else {
        kv("Changed", entry.modified ? new Date(entry.modified * 1000).toLocaleString() : "-");
        kv("Nodes", String(entry.node_count));
        if (entry._missing?.length) {
            kv("Missing nodes", String(entry._missing.length), true);
            const list = el("div", "sf-wb-modlist");
            for (const m of entry._missing.slice(0, 6)) list.append(el("div", "sf-wb-mod", m));
            if (entry._missing.length > 6) {
                list.append(el("div", "sf-wb-mod", `and ${entry._missing.length - 6} more`));
            }
            pane.append(list);
        }
    }

    // ── 用户自己的笔记 ──
    pane.append(el("div", "sf-wb-grouphead", "Your note"));
    const note = el("textarea", "sf-wb-note");
    note.placeholder = "What is this one for? Searchable.";
    note.value = state.meta?.notes?.[entry.rel] || "";
    note.addEventListener("keydown", (e) => { if (e.ctrlKey || e.metaKey || e.altKey) return; e.stopPropagation(); });   // Escape 不能关窗口(放行修饰键组合)
    installWheelZoomPassthrough(note); // 输入框滚轮透传(缩放画布/滚动文本, 对齐原生)
    let t = null;
    let sent = note.value;
    const flush = () => {
        clearTimeout(t);
        t = null;
        if (note.value !== sent) { sent = note.value; H.onNote(entry.rel, note.value); }
    };
    note.addEventListener("input", () => {
        clearTimeout(t);
        t = setTimeout(flush, 500);
    });
    // 点击别处立即 flush，不只等定时器。防抖为合并按键，但它留了半秒窗口
    // 其中最新文本只存在于这个框——窗口内改名会带走旧文本
    note.addEventListener("blur", flush);
    pane.append(note);

    // ── 动作 ──
    const acts = el("div", "sf-wb-acts");
    const btn = (label, fn, cls, title) => {
        const b = el("button", "sf-wb-tbtn" + (cls ? " " + cls : ""), label);
        b.type = "button";
        if (title) b.title = title;
        b.addEventListener("click", fn);
        acts.append(b);
        return b;
    };
    btn("Open", () => H.onOpen(entry), "sf-wb-primary");
    // 完整大小的收藏控件，卡片上的小星星不是唯一途径——列表视图无星星
    const fav = state.favourites.has(entry.rel);
    const favBtn = btn("Favourite", () => H.onStar(entry), null,
        fav ? "Remove from favourites" : "Add to favourites");
    const glyph = el("span", "sf-wb-btnstar" + (fav ? " on" : ""), fav ? "★" : "☆");
    favBtn.prepend(glyph);
    btn("Rename", () => H.onRename(entry));
    btn("Duplicate", () => H.onDuplicate(entry));
    const hasCover = hasHandCover(entry, state.meta);
    btn(hasCover ? "Replace cover" : "Set cover", () => H.onSetCover(entry), null,
        "Choose a picture for this card");
    if (hasCover) {
        btn("Remove cover", () => H.onClearCover(entry), null,
            "Go back to the drawn map, or this workflow's own last output");
    }
    btn("Reveal", () => H.onReveal(entry), null, "Open the folder it is in");
    btn("Delete", () => H.onDelete(entry), "sf-wb-danger", "There is no undo yet, so this asks first");
    pane.append(acts);

    // ── 它需要什么，最后 ──
    // 刻意在笔记与按钮下方：视频工作流可能需十几个文件，列表在上会把按钮
    // 推出面板底部
    const mods = [...(entry.models || []), ...(entry.loras || [])];
    if (mods.length) {
        const head = el("div", "sf-wb-headrow");
        head.append(el("div", "sf-wb-grouphead", `Needs these files (${mods.length})`));
        const copy = el("button", "sf-wb-copybtn", "Copy");
        copy.type = "button";
        copy.title = "Copy every filename, one per line";
        copy.addEventListener("click", () => copyList(mods, entry.name, copy));
        head.append(copy);
        pane.append(head);

        const list = el("div", "sf-wb-modlist");
        for (const m of mods) list.append(modChip(m));
        pane.append(list);
    }
}

/** 把文件名复制出来，可粘进下载列表或问人用的消息。 */
async function copyList(mods, workflowName, btn) {
    const text = `${workflowName}\n` + mods.map((m) => m).join("\n");
    const original = btn.textContent;
    const ok = await copyText(text);
    btn.textContent = ok ? "Copied" : "Could not copy";
    btn.classList.add("done");
    setTimeout(() => { btn.textContent = original; btn.classList.remove("done"); }, 1200);
}

/** 一眼可读的文件名：目录变暗、名字原样、扩展名用强调色。 */
function modChip(name) {
    const d = el("div", "sf-wb-mod");
    const cut = Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\"));
    const dir = cut >= 0 ? name.slice(0, cut) : "";
    const sep = cut >= 0 ? name[cut] : "";
    const file = cut >= 0 ? name.slice(cut + 1) : name;
    const dot = file.lastIndexOf(".");
    const base = dot > 0 ? file.slice(0, dot) : file;
    const ext = dot > 0 ? file.slice(dot) : "";
    if (dir) d.append(el("span", "sf-wb-moddir", dir));
    if (sep) d.append(el("span", "sf-wb-modsep", sep));
    d.append(el("span", "sf-wb-modname", base));
    if (ext) d.append(el("span", "sf-wb-modext", ext));
    d.title = name;
    return d;
}

// ── tidy 屏 ──────────────────────────────────────────────────────────────
// "Needs tidying" 曾只是普通过滤器：把受影响的工作流当普通卡片给你看，留
// 你猜每个有三个问题中的哪个。三种问题穿着同一张卡片不是审查屏。
// 这是审查屏：每个问题一节，每行携带对该问题真正适用的修复——遗留名改名、
// 一组副本 keep-one、缺失节点复制列表。这里没有任何东西自行行动：无撤销
// （见 confirmDanger 惯例），每个破坏性按钮都走与面板其余部分相同的确认。

function tidyActions(specs) {
    const wrap = el("div", "sf-wb-tdacts");
    for (const s of specs) {
        if (!s) continue;
        const b = el("button", "sf-wb-tdbtn" + (s.danger ? " danger" : "") + (s.primary ? " primary" : ""),
                     s.label);
        b.type = "button";
        if (s.title) b.title = s.title;
        b.addEventListener("click", (e) => { e.stopPropagation(); s.fn(); });
        wrap.append(b);
    }
    return wrap;
}

/** 一个工作流一行：图、名、所在，然后是它的修复。`renamable` 把行标记为
 *  beginRename 应编辑的那个（仅遗留名节），一行可同时出现在多个节。 */
function tidyRow(entry, state, H, extras, trailing, renamable) {
    const r = el("div", "sf-wb-tdrow");
    r.dataset.rel = entry.rel;
    if (renamable) r.dataset.rename = "1";
    if (state.selected.has(entry.rel)) r.classList.add("sel");
    if (state.kbdRel === entry.rel) r.classList.add("kbd");
    r.title = entry.rel;
    r.append(coverEl(entry, state, "sf-wb-rowcov"));

    const mid = el("div", "sf-wb-tdmid");
    mid.append(el("span", "sf-wb-rowname", entry.name));
    if (trailing) mid.append(el("span", "sf-wb-tdsub", trailing));
    r.append(mid);

    if (entry.folder) r.append(el("span", "sf-wb-tdfold", entry.folder));
    r.append(tidyActions(extras));
    r.addEventListener("click", (e) => H.onSelect(entry, e));
    r.addEventListener("dblclick", () => H.onOpen(entry));
    return r;
}

function tidySection(title, blurb, count) {
    const s = el("div", "sf-wb-tdsec");
    const head = el("div", "sf-wb-tdhead");
    head.append(el("span", "sf-wb-tdtitle", title));
    head.append(el("span", "sf-wb-tdcount", String(count)));
    s.append(head);
    if (blurb) s.append(el("div", "sf-wb-tdblurb", blurb));
    return s;
}

export function renderTidy(main, state, H) {
    // 见 renderGrid：清空会同步触发打开的改名框的 blur
    markRendering(() => { main.textContent = ""; });
    const { issues, byRel, query } = state;

    // 同一查询框仍会收窄屏幕。搜索收窄按 S.visible（真正的加权搜索），
    // 头部 "N of M" 从它的结果计数——只有一个匹配定义，本屏借用它
    const q = (query || "").trim();
    const vis = new Set((state.visible || []).map((e) => e.rel));
    const keep = (rel) => !q || vis.has(rel);
    const get = (rel) => byRel.get(rel);

    const wrap = el("div", "sf-wb-tidy");
    wrap.append(el("div", "sf-wb-tdintro",
        "Nothing on this screen is changed for you. Each row is a suggestion with "
        + "its fix beside it, and anything that deletes still asks first."));

    let shown = 0;

    // ── 1. 遗留名 ──
    const unsaved = (issues.unsaved_names || []).map((u) => get(u.rel))
        .filter((e) => e && keep(e.rel));
    if (unsaved.length) {
        shown += unsaved.length;
        const s = tidySection("Still called \u201cUnsaved Workflow\u201d",
            "Saved before they were given a name. Rename edits the name right here: "
            + "type over it and press Enter.", unsaved.length);
        for (const e of unsaved) {
            s.append(tidyRow(e, state, H, [
                { label: "Rename", primary: true, fn: () => H.onRename(e),
                  title: "Give it a name you will recognise" },
                { label: "Open", fn: () => H.onOpen(e) },
                { label: "Delete", danger: true, fn: () => H.onDelete(e) },
            ], null, true));
        }
        wrap.append(s);
    }

    // ── 2. 重复 ──
    // 一组只在还有两个成员时值得显示：搜索把二成员组滤到一，keep-one 对着
    // 空删、读起来像坏按钮
    const dupGroups = (issues.duplicates || [])
        .map((g) => g.map((d) => get(d.rel)).filter(Boolean))
        .filter((g) => g.length > 1 && g.some((e) => keep(e.rel)));
    if (dupGroups.length) {
        const files = dupGroups.reduce((n, g) => n + g.length, 0);
        shown += files;
        const s = tidySection("The same workflow saved more than once",
            "Same nodes and same models under different names. \u201cKeep this one\u201d "
            + "deletes the others in its set, and tells you which before it does.",
            `${dupGroups.length} set${dupGroups.length === 1 ? "" : "s"}`);
        for (const g of dupGroups) {
            const box = el("div", "sf-wb-tdgroup");
            for (const e of g) {
                const others = g.filter((x) => x.rel !== e.rel);
                const r = tidyRow(e, state, H, [
                    { label: "Keep this one", primary: true,
                      title: `Delete the other ${others.length} in this set:\n`
                             + others.map((x) => x.name).join("\n"),
                      // 名字进确认，不只数量——"删除 2 个工作流？"对用户从未
                      // 逐一手选的文件不够同意
                      fn: () => H.onDeleteMany(others.map((x) => x.rel), {
                          title: `Delete the other ${others.length} in this set?`,
                          message: `Keeping "${e.name}". These go:\n`
                                   + others.map((x) => x.rel).join("\n"),
                      }) },
                    { label: "Open", fn: () => H.onOpen(e) },
                    { label: "Delete", danger: true, fn: () => H.onDelete(e) },
                ], null);
                // 任成员匹配搜索时整组显示——半组无法评判。不匹配的行变暗说明
                if (q && !keep(e.rel)) {
                    r.classList.add("sf-wb-tddimmed");
                    r.title += "\nShown for context - it does not match your search, its set does.";
                }
                box.append(r);
            }
            s.append(box);
        }
        wrap.append(s);
    }

    // ── 3. 缺失节点 ──
    const missing = (issues.missing_nodes || [])
        .filter((m) => get(m.rel) && keep(m.rel));
    if (missing.length) {
        shown += missing.length;
        const s = tidySection("Needs nodes you do not have",
            "These will open with red boxes where the missing nodes should be. Copy "
            + "the list and search for it in ComfyUI Manager to find what installs them.",
            missing.length);
        for (const m of missing) {
            const e = get(m.rel);
            const names = m.missing || [];
            s.append(tidyRow(e, state, H, [
                { label: "Copy list", primary: true, title: names.join(", "),
                  fn: () => H.onCopyText(names.join("\n"), `Copied ${names.length} node names`) },
                { label: "Open", fn: () => H.onOpen(e) },
                { label: "Delete", danger: true, fn: () => H.onDelete(e) },
            ], names.join(", ")));
        }
        wrap.append(s);
    }

    if (!shown) {
        wrap.append(el("div", "sf-wb-empty", q
            ? `Nothing in here matches "${query}".`
            : "Nothing needs tidying. Your workflows folder is in good shape."));
    }

    main.append(wrap);
    // 与网格相同：重渲染前打开的改名框放回去
    restoreRename(main);
}

// ── 文件夹侧栏 ────────────────────────────────────────────────────────────

// 同一文件夹行两次点击这么近 = 改名。模块级：行在第一次点击后不存活。
const DBL_MS = 400;
let lastFoldClick = { path: null, at: 0 };

export function renderFolders(side, state, { onPick, onDropOn, onRenameFolder, onFolderMenu, onReorderFolder, onToggleFolder }) {
    // 清空会同步触发打开的文件夹改名框的 blur——标志必须盖住它
    markRendering(() => { side.textContent = ""; });
    const { entries, folders, collections, meta, favourites, sel, tidyRels } = state;
    const open = openSet(meta?.folderExpanded, sel);

    const is = (kind, value) => sel.kind === kind && (value === undefined || sel.value === value);

    /** 建行、挂点击，真实文件夹再挂 drop 目标。 */
    function addRow({ label, count, on, dot, indent = 0, title, muted, star, twisty }, pick, folderPath) {
        const b = el("button", "sf-wb-fold" + (on ? " on" : ""));
        b.type = "button";
        if (title) b.title = title;
        if (muted) b.style.color = "#6e6764";
        if (indent) {
            const sp = el("span", "sf-wb-nest");
            sp.style.width = indent * 11 + "px";
            b.append(sp);
        }
        if (twisty) {
            const c = el("span", "sf-wb-chev" + (twisty.open ? " sf-wb-chev-open" : ""), "▶");
            c.title = twisty.open ? "Hide what is inside" : "Show what is inside";
            c.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                lastFoldClick = { path: null, at: 0 };
                twisty.onToggle();
            });
            b.append(c);
        } else if (twisty === null) {
            b.append(el("span", "sf-wb-chevpad"));
        }
        if (dot) {
            const d = el("span", "sf-wb-dot");
            d.style.background = dot;
            b.append(d);
        }
        if (star) b.append(el("span", "sf-wb-favstar", "★"));
        b.append(el("span", "sf-wb-foldlbl", label));
        if (count != null) b.append(el("span", "sf-wb-cnt", String(count)));
        b.addEventListener("click", () => {
            // 双击改名在这里计数点击实现（dblclick 监听器在这类行上永不
            // 触发：第一次点击选择文件夹重渲染整列并拆掉本元素，第二次
            // 点击落在替代品上，浏览器不发 dblclick）
            if (folderPath && onRenameFolder) {
                const now = performance.now();
                if (lastFoldClick.path === folderPath && now - lastFoldClick.at < DBL_MS) {
                    lastFoldClick = { path: null, at: 0 };
                    onRenameFolder(folderPath, b);
                    return;
                }
                lastFoldClick = { path: folderPath, at: now };
            } else {
                lastFoldClick = { path: null, at: 0 };
            }
            onPick(pick);
        });

        if (folderPath !== undefined) {
            const clearMarks = () => b.classList.remove(
                "sf-wb-droptarget", "sf-wb-insert-above", "sf-wb-insert-below");

            // 标记此行是 Pixaroma 拖拽可合法落地的位置（守卫取消其它一切）
            b.dataset[DROP_TARGET_ATTR] = "1";

            b.addEventListener("dragover", (e) => {
                e.preventDefault();
                if (e.dataTransfer) e.dataTransfer.dropEffect = "move";
                clearMarks();
                if (isFolderDrag(e)) {
                    if (folderPath === "") return;   // (loose files) 不是真实文件夹
                    const r = b.getBoundingClientRect();
                    const above = (e.clientY - r.top) < r.height / 2;
                    b.classList.add(above ? "sf-wb-insert-above" : "sf-wb-insert-below");
                } else {
                    b.classList.add("sf-wb-droptarget");
                }
            });
            b.addEventListener("dragleave", (e) => {
                if (e.relatedTarget && b.contains(e.relatedTarget)) return;
                clearMarks();
            });
            b.addEventListener("drop", (e) => {
                e.preventDefault();
                const wasFolder = isFolderDrag(e);
                const r = b.getBoundingClientRect();
                const above = (e.clientY - r.top) < r.height / 2;
                clearMarks();
                if (wasFolder) {
                    const moved = e.dataTransfer.getData(FOLDER_MIME);
                    if (moved && moved !== folderPath && folderPath !== "") {
                        onReorderFolder?.(moved, folderPath, above);
                    }
                } else {
                    onDropOn?.(folderPath);
                }
            });

            // 文件夹本身可拖，能手工重排而不只靠右键菜单
            b.draggable = true;
            b.addEventListener("dragstart", (e) => {
                if (e.target.tagName === "INPUT") { e.preventDefault(); return; }  // 改名中
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData(FOLDER_MIME, folderPath);
                e.dataTransfer.setData("text/plain", folderPath);
                b.classList.add("sf-wb-dragging-me");
            });
            b.addEventListener("dragend", () => b.classList.remove("sf-wb-dragging-me"));
        }

        side.append(b);
        return b;
    }

    // ── 快捷方式 ──
    addRow({ label: "All workflows", count: entries.length, on: is("all") }, { kind: "all" });
    addRow({ label: "Favourites", star: true, count: favourites.size, on: is("fav") }, { kind: "fav" });
    addRow({ label: "Recent", count: Math.min(20, entries.length), on: is("recent") }, { kind: "recent" });

    const issueCount = tidyRels?.size || 0;
    if (issueCount) {
        addRow({
            label: "Needs tidying", count: issueCount, on: is("tidy"),
            title: "Leftover names, duplicates, and workflows needing things you do not have",
        }, { kind: "tidy" });
    }

    // ── 真实文件夹 ──
    side.append(el("div", "sf-wb-grouphead", "Folders"));

    // 子文件夹里的工作流也计入父级，否则只装子文件夹的目录读起来是空的
    const perFolder = new Map();
    for (const e of entries) {
        if (!e.folder) continue;
        const parts = e.folder.split("/");
        for (let i = 1; i <= parts.length; i++) {
            const key = parts.slice(0, i).join("/");
            perFolder.set(key, (perFolder.get(key) || 0) + 1);
        }
    }

    addRow({
        label: "(loose files)", count: entries.filter((e) => !e.folder).length,
        on: is("folder", ""), dot: "#5a5450", twisty: null,
        title: "Workflows sitting outside any folder",
    }, { kind: "folder", value: "" }, "");

    for (const f of folders) {
        // 关闭文件夹内的文件夹不绘制
        if (!ancestorsOf(f).every((a) => open.has(a))) continue;

        const kids = hasChildren(f, folders);
        const isOpen = open.has(f);
        const row = addRow({
            label: f.split("/").pop(),
            count: perFolder.get(f) || 0,
            on: is("folder", f),
            dot: folderColor(f, meta),
            indent: f.split("/").length - 1,
            twisty: kids ? { open: isOpen, onToggle: () => onToggleFolder?.(f, !isOpen) } : null,
            title: f + "\nDouble click to rename, right click for more",
        }, { kind: "folder", value: f }, f);

        row.addEventListener("dblclick", (e) => e.preventDefault());
        row.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            e.stopPropagation();
            onFolderMenu?.(f, e);
        });
    }

    addRow({ label: "+ New folder", muted: true }, { kind: "newfolder" });

    // ── 集合 ──
    const kinds = (collections || []).filter((c) => c.group === "kind");
    const models = (collections || []).filter((c) => c.group === "model");

    if (kinds.length) {
        side.append(el("div", "sf-wb-grouphead", "What it makes"));
        for (const c of kinds) {
            addRow({ label: c.label, count: c.count, on: is("collection", c.id) },
                   { kind: "collection", value: c.id });
        }
    }
    if (models.length) {
        side.append(el("div", "sf-wb-grouphead", "Model"));
        for (const c of models) {
            addRow({ label: c.label, count: c.count, on: is("collection", c.id) },
                   { kind: "collection", value: c.id });
        }
    }
}

/** 把文件夹行换成文本框。Enter 提交新名字（非路径）。 */
export function beginFolderRename(row, path, commit) {
    if (!row || row.querySelector("input")) return;
    const current = path.split("/").pop();
    const kept = [...row.childNodes];
    const input = el("input", "sf-wb-foldrename");
    input.value = current;
    row.textContent = "";
    row.append(input);
    input.focus();
    input.select();

    let done = false;
    const finish = (save) => {
        if (done) return;
        done = true;
        const value = input.value.trim();
        row.textContent = "";
        kept.forEach((n) => row.append(n));
        if (save && value && value !== current) commit(value);
    };
    input.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Enter") finish(true);
        else if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => {
        if (isRendering()) return;
        finish(true);
    });
    input.addEventListener("click", (e) => e.stopPropagation());
    input.addEventListener("dblclick", (e) => e.stopPropagation());
}
