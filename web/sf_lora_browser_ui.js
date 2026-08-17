// ==========================================================================
// SF LoRA 浏览器 - UI 层（窗口骨架 + CSS + 文件夹层级网格渲染）
// 与 LoRA Stack 信息面板同款风格：深色面板 + 全局强调色 --sf-acc。
// 无节点设计（应用面板，参照 sf_workflows*）。主扩展（sf_lora_browser.js）
// 持有状态与数据，调用本层渲染；本层不触碰服务端。
//
// 展示模型对齐 SF Load Image Browser 的浏览器（web/image_browser.js）：
// 面包屑 + 文件夹下钻——当前目录只显示「立即子文件夹」+「当前层文件」；
// 搜索激活时忽略层级、跨全部分层扁平匹配。
// ==========================================================================
import { thumbUrl } from "./sf_lora_stack_api.js";
import {
    splitName, filterLoras, folderContents, breadcrumbParts,
} from "./sf_lora_browser_lib.js";

let _cssInjected = false;

// 无缩略图时的占位图（内联 SVG：深色圆角底 + 层叠图标，与工具栏按钮图标
// 同语义；data URI 无网络请求、必成功渲染——替代浏览器默认的破损图）。
// 注意 SVG 里 # 与空格必须 URL 编码（引号内单引号属性）。
const THUMB_PLACEHOLDER =
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
    + "%3Crect width='64' height='64' rx='8' fill='%232b2a2e'/%3E"
    + "%3Cg fill='%23615d68'%3E%3Cpath d='M32 15 15 24l17 9 17-9-17-9z'/%3E"
    + "%3Cpath d='M15 30l17 9 17-9'/%3E%3Cpath d='M15 40l17 9 17-9'/%3E%3C/g%3E%3C/svg%3E";

export function injectBrowserCSS() {
    if (_cssInjected) return;
    _cssInjected = true;
    const s = document.createElement("style");
    s.id = "sf-lb-css";
    s.textContent = `
:root { --sf-lb-acc:var(--sf-acc, #f66744); }
.sf-lb-win { position:fixed; z-index:9960; background:#1b1a1a; border:1px solid #3d3936;
  border-radius:10px; box-shadow:0 20px 60px rgba(0,0,0,.6); flex-direction:column;
  color:#ddd; font:12px 'Segoe UI',sans-serif; overflow:hidden; display:none;
  min-width:480px; min-height:320px; }
.sf-lb-win * { box-sizing:border-box; }
.sf-lb-title { display:flex; align-items:center; gap:8px; padding:8px 12px; cursor:grab;
  background:#201f1e; border-bottom:1px solid #33302e; user-select:none; flex:0 0 auto; }
.sf-lb-title.sf-lb-dragging { cursor:grabbing; }
.sf-lb-name { font-weight:600; font-size:13px; color:#fff; display:flex; align-items:center; gap:8px; }
.sf-lb-logo { width:12px; height:12px; border-radius:3px; background:var(--sf-lb-acc); display:inline-block; }
.sf-lb-count { color:#9a938f; font-weight:400; font-size:11px; }
.sf-lb-sp { flex:1; }
.sf-lb-wbtn { background:none; border:0; color:#aaa; font-size:15px; cursor:pointer;
  padding:2px 8px; border-radius:4px; }
.sf-lb-wbtn:hover { background:rgba(255,255,255,.1); color:#fff; }
.sf-lb-bar { display:flex; align-items:center; gap:8px; padding:7px 10px; border-bottom:1px solid #33302e; flex:0 0 auto; }
.sf-lb-search { flex:1; display:flex; align-items:center; gap:6px; background:#141312;
  border:1px solid #3d3936; border-radius:6px; padding:5px 9px; min-width:140px; }
.sf-lb-search input { flex:1; background:transparent; border:0; outline:none; color:#e6e6e6;
  font:12px 'Segoe UI',sans-serif; }
.sf-lb-tbtn { background:rgba(255,255,255,.05); border:1px solid #4a4542; color:#cfcfcf; border-radius:5px;
  padding:5px 11px; font:12px 'Segoe UI',sans-serif; cursor:pointer; white-space:nowrap; flex:0 0 auto; }
.sf-lb-tbtn:hover:not(:disabled) { border-color:var(--sf-lb-acc); color:#fff; }
.sf-lb-tbtn:disabled { opacity:.45; cursor:default; }
.sf-lb-seg { display:flex; border:1px solid #4a4542; border-radius:5px; overflow:hidden; flex:0 0 auto; }
.sf-lb-segb { background:transparent; border:0; color:#a8a29e; padding:6px 9px; cursor:pointer;
  display:flex; align-items:center; justify-content:center; }
.sf-lb-segb:hover { color:#ddd; }
.sf-lb-segb.on { background:var(--sf-lb-acc); color:#fff; }
.sf-lb-segb + .sf-lb-segb { border-left:1px solid #4a4542; }
.sf-lb-segb .ic { width:15px; height:15px; background:currentColor;
  -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
  -webkit-mask-position:center; mask-position:center;
  -webkit-mask-size:contain; mask-size:contain; }
.sf-lb-segb.folder .ic { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3C/svg%3E"); }
/* 平面模式（全部层级）：层叠图标，与列表视图的三横线区分 */
.sf-lb-segb.flat .ic { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 2 2 7l10 5 10-5-10-5zM2 12l10 5 10-5M2 17l10 5 10-5'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 2 2 7l10 5 10-5-10-5zM2 12l10 5 10-5M2 17l10 5 10-5'/%3E%3C/svg%3E"); }
/* 视图切换：网格（九宫格）/ 列表（三横线） */
.sf-lb-segb.grid .ic { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 4h6v6H4zm10 0h6v6h-6zM4 14h6v6H4zm10 0h6v6h-6z'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 4h6v6H4zm10 0h6v6h-6zM4 14h6v6H4zm10 0h6v6h-6z'/%3E%3C/svg%3E"); }
.sf-lb-segb.list .ic { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 6h16M4 12h16M4 18h16' stroke='%23000' stroke-width='2'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 6h16M4 12h16M4 18h16' stroke='%23000' stroke-width='2'/%3E%3C/svg%3E"); }
/* 列表视图 */
.sf-lb-list { display:flex; flex-direction:column; gap:2px; }
.sf-lb-row { display:flex; align-items:center; gap:8px; padding:5px 8px; border-radius:6px;
  cursor:pointer; min-width:0; }
.sf-lb-row:hover { background:rgba(255,255,255,0.05); }
.sf-lb-row.sel { background:color-mix(in srgb, var(--sf-lb-acc) 16%, transparent); }
.sf-lb-thumb-sm { width:40px; height:40px; border-radius:5px; object-fit:cover; flex:none;
  background:radial-gradient(circle at 60% 35%, #3a3238, #1d1a1e 72%); display:block; }
.sf-lb-thumb-sm.noimg { background:none; }
.sf-lb-rowicon { flex:none; width:40px; height:40px; display:flex; align-items:center;
  justify-content:center; font-size:20px; background:rgba(255,255,255,0.03); border-radius:5px; }
.sf-lb-rowname { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  color:#e6e6e6; font-size:12px; }
.sf-lb-rowmeta { flex:none; max-width:40%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  color:#8a8581; font-size:10.5px; }
.sf-lb-loadmore { text-align:center; color:#8a8581; font-size:11px; padding:10px; }
/* 多 Stack 节点选择弹窗（双击添加时选择目标节点） */
.sf-lb-pick-mask { position:fixed; inset:0; z-index:10040; background:rgba(0,0,0,0.55);
  display:flex; align-items:center; justify-content:center; }
.sf-lb-pick { width:360px; max-width:90vw; background:#2b2b2b; border:1px solid var(--sf-lb-acc, #f66744);
  border-radius:10px; box-shadow:0 14px 44px rgba(0,0,0,0.6); color:#ddd;
  font:12px 'Segoe UI',sans-serif; overflow:hidden; }
.sf-lb-pick-t { padding:12px 14px; border-bottom:1px solid #1c1c1c; color:#fff;
  font-size:13px; font-weight:600; }
.sf-lb-pick-sub { padding:4px 14px 8px; font-size:11px; color:#8a8581; }
.sf-lb-pick-list { padding:6px; max-height:50vh; overflow-y:auto; }
.sf-lb-pick-row { display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:6px;
  cursor:pointer; }
.sf-lb-pick-row:hover { background:rgba(255,255,255,0.07); }
.sf-lb-pick-id { color:var(--sf-lb-acc, #f66744); font:11px monospace; flex:none; }
.sf-lb-pick-title { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  color:#e6e6e6; font-size:12px; }
.sf-lb-pick-meta { flex:none; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  color:#8a8581; font-size:11px; }
.sf-lb-pick-meta .n { color:var(--sf-lb-acc, #f66744); font-weight:600; }
.sf-lb-pick-cancel { text-align:center; padding:9px; border-top:1px solid #1c1c1c;
  color:#9a9a9a; cursor:pointer; font-size:12px; }
.sf-lb-pick-cancel:hover { color:#fff; }
/* 面包屑行（当前目录路径；搜索时仍显示当前层 context，列表转扁平匹配） */
.sf-lb-path { display:flex; align-items:center; flex-wrap:wrap; gap:2px;
  padding:5px 12px; border-bottom:1px solid #2c2a28; background:#1e1d1c; flex:0 0 auto;
  font-size:11.5px; min-height:28px; }
.sf-lb-crumb { color:#9a938f; cursor:pointer; padding:2px 6px; border-radius:4px;
  max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-lb-crumb:hover { color:#ddd; background:#2b2927; }
.sf-lb-crumb.cur { color:var(--sf-lb-acc); cursor:default; }
.sf-lb-crumb.cur:hover { background:transparent; }
.sf-lb-crumbsep { color:#555; cursor:default; padding:0 2px; user-select:none; }
.sf-lb-main { flex:1; min-height:0; overflow-y:auto; padding:10px; background:#1e1d1c; position:relative; }
.sf-lb-grid { display:grid;
  grid-template-columns:repeat(auto-fill, minmax(108px, 1fr)); gap:8px; align-content:start; }
.sf-lb-card { background:#232120; border:1px solid #34312f; border-radius:8px; padding:7px; cursor:pointer;
  display:flex; flex-direction:column; gap:5px; min-width:0; content-visibility:auto; }
.sf-lb-card:hover { border-color:#4c4744; }
.sf-lb-card.sel { border-color:var(--sf-lb-acc); box-shadow:0 0 0 1px var(--sf-lb-acc); }
.sf-lb-thumb { width:100%; aspect-ratio:1/1; border-radius:5px; background:radial-gradient(circle at 60% 35%, #3a3238, #1d1a1e 72%);
  object-fit:cover; display:block; }
.sf-lb-thumb.noimg { background:none; }
.sf-lb-cardname { font-size:11.5px; color:#e6e6e6; min-width:0;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
  word-break:break-all; line-height:1.35; }
.sf-lb-cardmeta { font-size:9.5px; color:#8a8581; min-width:0; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }
/* 文件夹卡片：居中大图标 + 名称（无缩略图） */
.sf-lb-card.folder { align-items:center; justify-content:center; gap:6px; padding:12px 7px; }
.sf-lb-card.folder .sf-lb-foldericon { font-size:30px; line-height:1; filter:grayscale(.2); }
.sf-lb-card.folder .sf-lb-cardname { text-align:center; -webkit-line-clamp:2; }
.sf-lb-card.folder:hover .sf-lb-foldericon { filter:none; }
.sf-lb-empty { padding:30px; text-align:center; color:#8a8581; }
.sf-lb-toast { position:absolute; left:50%; bottom:10px; transform:translateX(-50%); background:#2a2725;
  border:1px solid #4c4744; border-radius:6px; padding:7px 14px; font-size:11.5px; color:#eee;
  box-shadow:0 6px 18px rgba(0,0,0,.5); z-index:9; display:none; max-width:70%; }
.sf-lb-grip { position:absolute; right:0; bottom:0; width:18px; height:18px; cursor:nwse-resize; z-index:3; }
.sf-lb-grip::after { content:""; position:absolute; right:3px; bottom:3px; width:7px; height:7px;
  border-right:2px solid #7a7a7a; border-bottom:2px solid #7a7a7a; }
.sf-lb-grip:hover::after { border-color:var(--sf-lb-acc); }
/* 工具栏按钮（与 SF Workflows 按钮同排同型） */
.sf-lb-btn { width:36px; height:36px; border-radius:8px; background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.12); display:flex; align-items:center; justify-content:center;
  cursor:pointer; color:#ccc; }
.sf-lb-btn:hover { background:var(--sf-lb-acc); border-color:var(--sf-lb-acc); color:#fff; }
.sf-lb-btn.sf-lb-btn-open { background:var(--sf-lb-acc); border-color:var(--sf-lb-acc); color:#fff; }
.sf-lb-btn-icon { width:16px; height:16px; border-radius:3px; background:currentColor;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 2 2 7l10 5 10-5-10-5zM2 12l10 5 10-5M2 17l10 5 10-5'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 2 2 7l10 5 10-5-10-5zM2 12l10 5 10-5M2 17l10 5 10-5'/%3E%3C/svg%3E");
  -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat; -webkit-mask-position:center; mask-position:center;
  -webkit-mask-size:contain; mask-size:contain; }
/* 命令面板图标（Vue 命令系统） */
.sf-lb-cmd-icon { background:var(--sf-lb-acc); -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 2 2 7l10 5 10-5-10-5zM2 12l10 5 10-5M2 17l10 5 10-5'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 2 2 7l10 5 10-5-10-5zM2 12l10 5 10-5M2 17l10 5 10-5'/%3E%3C/svg%3E");
  -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat; -webkit-mask-position:center; mask-position:center;
  -webkit-mask-size:contain; mask-size:contain; }
`;
    document.head.appendChild(s);
}

export function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

// ── 窗口几何持久化 ──────────────────────────────────────────────────────────
// 设置键惯例（与 workflows 同款；window.sfnodesGetSetting/SetSetting 由主扩展注入，ui 层经此读写）
const RECT_KEY = "sfnodes.LoraBrowser.Rect";
const MIN_W = 480, MIN_H = 320;

function defaultRect() {
    const vw = window.innerWidth, vh = window.innerHeight;
    const top = Math.max(8, 56);           // 工具栏地板：别盖住顶栏
    const w = Math.max(MIN_W, Math.min(920, vw - 48));
    const h = Math.max(MIN_H, Math.min(640, vh - top - 32));
    return { x: Math.max(24, Math.min(120, vw - w - 24)), y: top, w, h };
}
function clampRect(r) {
    const d = defaultRect();
    const vw = window.innerWidth, vh = window.innerHeight;
    const top = Math.max(8, 56);
    const w = Math.round(Math.max(MIN_W, Math.min(r?.w ?? d.w, vw - 16)));
    const h = Math.round(Math.max(MIN_H, Math.min(r?.h ?? d.h, vh - top - 8)));
    return {
        x: Math.round(Math.max(8, Math.min(r?.x ?? d.x, vw - w - 8))),
        y: Math.round(Math.max(top, Math.min(r?.y ?? d.y, vh - h - 8))),
        w, h,
    };
}
function readRect() {
    try {
        const raw = window.sfnodesGetSetting?.(RECT_KEY, null);
        if (typeof raw === "string") return clampRect(JSON.parse(raw));
        if (raw && typeof raw === "object") return clampRect(raw);
    } catch { /* 落到默认 */ }
    return defaultRect();
}
let _rectSaveTimer = null;
function saveRect(rect) {
    clearTimeout(_rectSaveTimer);
    _rectSaveTimer = setTimeout(() => {
        try { window.sfnodesSetSetting?.(RECT_KEY, rect); } catch { /* 保存几何从不弄坏 UI */ }
    }, 350);
}

function startDrag(handle, e, onMove, onEnd) {
    e.preventDefault();
    try { handle.setPointerCapture(e.pointerId); } catch { /* 不可捕获 */ }
    let done = false;
    const move = (ev) => {
        if (!(ev.buttons & 1)) return up();
        onMove(ev);
    };
    const up = () => {
        if (done) return;
        done = true;
        try { handle.releasePointerCapture(e.pointerId); } catch { /* 已离开 */ }
        handle.removeEventListener("pointermove", move, true);
        handle.removeEventListener("pointerup", up, true);
        handle.removeEventListener("pointercancel", up, true);
        handle.removeEventListener("lostpointercapture", up, true);
        onEnd?.();
    };
    handle.addEventListener("pointermove", move, true);
    handle.addEventListener("pointerup", up, true);
    handle.addEventListener("pointercancel", up, true);
    handle.addEventListener("lostpointercapture", up, true);
}

// ── 窗口 ────────────────────────────────────────────────────────────────────
export function createLoraBrowserWindow({ onRender, onClose } = {}) {
    injectBrowserCSS();

    const win = el("div", "sf-lb-win");
    win.style.display = "none";

    const title = el("div", "sf-lb-title");
    const name = el("div", "sf-lb-name");
    const count = el("span", "sf-lb-count", "");
    name.append(el("span", "sf-lb-logo"), el("span", null, "LoRA"), count);
    const closeBtn = el("button", "sf-lb-wbtn", "✕");
    closeBtn.type = "button";
    closeBtn.title = "Close (Esc)";
    title.append(name, el("div", "sf-lb-sp"), closeBtn);

    const bar = el("div", "sf-lb-bar");
    const search = el("div", "sf-lb-search");
    const ic = el("span", null, "⌕");
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Search LoRAs…";
    input.addEventListener("keydown", (e) => { if (e.ctrlKey || e.metaKey || e.altKey) return; e.stopPropagation(); });
    search.append(ic, input);
    const seg = el("div", "sf-lb-seg");
    const folderBtn = el("button", "sf-lb-segb on folder");
    folderBtn.type = "button";
    folderBtn.dataset.mode = "folder";
    folderBtn.title = "Browse by folders (breadcrumbs + drill-down)";
    folderBtn.append(el("span", "ic"));
    const flatBtn = el("button", "sf-lb-segb flat");
    flatBtn.type = "button";
    flatBtn.dataset.mode = "flat";
    flatBtn.title = "All LoRAs across every folder (scroll to load more)";
    flatBtn.append(el("span", "ic"));
    seg.append(folderBtn, flatBtn);
    const viewSeg = el("div", "sf-lb-seg");
    const gridBtn = el("button", "sf-lb-segb on grid");
    gridBtn.type = "button";
    gridBtn.dataset.view = "grid";
    gridBtn.title = "Grid view";
    gridBtn.append(el("span", "ic"));
    const listBtn = el("button", "sf-lb-segb list");
    listBtn.type = "button";
    listBtn.dataset.view = "list";
    listBtn.title = "List view";
    listBtn.append(el("span", "ic"));
    viewSeg.append(gridBtn, listBtn);
    const refreshBtn = el("button", "sf-lb-tbtn", "↻");
    refreshBtn.type = "button";
    refreshBtn.title = "Refresh the list from disk";
    bar.append(search, seg, viewSeg, refreshBtn);

    const path = el("div", "sf-lb-path");
    const main = el("div", "sf-lb-main");
    const grip = el("div", "sf-lb-grip");
    win.append(title, bar, path, main, grip);
    document.body.appendChild(win);

    let rect = clampRect(readRect());
    const applyRect = () => {
        win.style.left = rect.x + "px";
        win.style.top = rect.y + "px";
        win.style.width = rect.w + "px";
        win.style.height = rect.h + "px";
    };
    applyRect();

    title.addEventListener("pointerdown", (e) => {
        if (e.target.closest(".sf-lb-wbtn")) return;
        const ox = e.clientX - win.offsetLeft;
        const oy = e.clientY - win.offsetTop;
        startDrag(title, e, (ev) => {
            rect.x = Math.max(0, Math.min(ev.clientX - ox, window.innerWidth - Math.min(rect.w, 200)));
            rect.y = Math.max(8, Math.min(ev.clientY - oy, window.innerHeight - 40));
            applyRect();
        }, () => { title.classList.remove("sf-lb-dragging"); saveRect(rect); });
        title.classList.add("sf-lb-dragging");
    });

    grip.addEventListener("pointerdown", (e) => {
        const left = win.offsetLeft, top = win.offsetTop;
        const ox = e.clientX - (left + win.offsetWidth);
        const oy = e.clientY - (top + win.offsetHeight);
        startDrag(grip, e, (ev) => {
            rect.w = Math.max(MIN_W, Math.min(ev.clientX - ox - left, window.innerWidth - left));
            rect.h = Math.max(MIN_H, Math.min(ev.clientY - oy - top, window.innerHeight - top));
            applyRect();
        }, () => saveRect(rect));
        e.stopPropagation();
    });

    window.addEventListener("resize", () => {
        if (win.style.display === "none") return;
        rect = clampRect(rect);
        applyRect();
    });

    // 面板内点击不得到达 canvas（浏览不应取消画布选中）
    win.addEventListener("pointerdown", (e) => e.stopPropagation());

    // ── toast ──
    let toastEl = null, toastTimer = null;
    function toast(message) {
        if (!message) {
            if (toastEl) toastEl.style.display = "none";
            clearTimeout(toastTimer);
            return;
        }
        if (!toastEl) { toastEl = el("div", "sf-lb-toast"); main.appendChild(toastEl); }
        toastEl.textContent = message;
        toastEl.style.display = "block";
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { if (toastEl) toastEl.style.display = "none"; }, 2600);
    }

    const api = {
        el: win, main, bar, path, count, searchInput: input, refreshBtn,
        segButtons: [folderBtn, flatBtn], viewButtons: [gridBtn, listBtn],
        isOpen: () => win.style.display !== "none",
        toast,
        setCount: (text) => { count.textContent = text; },
        focusSearch: () => input.focus({ preventScroll: true }),
        open() {
            rect = clampRect(rect);
            applyRect();
            win.style.display = "flex";
            onRender?.();
            setTimeout(() => input.focus({ preventScroll: true }), 20);
        },
        close() {
            win.style.display = "none";
            if (input.value) { input.value = ""; input.dispatchEvent(new Event("input", { bubbles: true })); }
            if (toastEl) toastEl.style.display = "none";
            onClose?.();
        },
        toggle() { api.isOpen() ? api.close() : api.open(); },
        destroy() { win.remove(); },
    };

    // Esc 先清搜索再关窗口（一个键同时丢查询与窗口是错误量的撤销）
    win.addEventListener("keydown", (e) => {
        if (e.key !== "Escape") return;
        if (input.value && document.activeElement === input) {
            input.value = "";
            input.dispatchEvent(new Event("input", { bubbles: true }));
            e.stopPropagation();
            return;
        }
        e.stopPropagation();
        api.close();
    });

    closeBtn.addEventListener("click", () => api.close());
    return api;
}

// ── 面包屑 ──────────────────────────────────────────────────────────────────
// "All LoRAs ▸ a ▸ b（当前）"。目录名来自用户文件系统：全部走 textContent /
// dataset 赋值（无 innerHTML 注入面，< 或 " 的目录名天然安全）。
export function renderCrumbs(elPath, folder, onCrumb) {
    if (!elPath) return;
    elPath.innerHTML = "";
    const parts = breadcrumbParts(folder);
    const mkCrumb = (label, target) => {
        const c = el("span", "sf-lb-crumb" + (target == null ? " cur" : ""));
        c.textContent = label;
        if (target != null) {
            c.dataset.folder = target;      // 纯数据属性，非 HTML 注入面
            c.addEventListener("click", () => onCrumb?.(target));
        }
        return c;
    };
    elPath.appendChild(mkCrumb("All LoRAs", parts.length ? "" : null));
    let accumulated = "";
    for (let i = 0; i < parts.length; i++) {
        accumulated += (i > 0 ? "/" : "") + parts[i];
        elPath.appendChild(el("span", "sf-lb-crumbsep", "›"));
        elPath.appendChild(mkCrumb(parts[i], i === parts.length - 1 ? null : accumulated));
    }
}

// ── 单击/双击（卡片与列表行共用）────────────────────────────────────────────
// 单击 = 打开信息面板（延迟 250ms 等双击判定），双击 = 用 SF LoRA Stack 加载到
// 工作流。浏览器对双击先派发两次 click 再 dblclick：第二次 click 覆盖第一次的
// timer，dblclick 再清一次——无残留。
function attachPickAdd(el2, name, onPick, onAdd) {
    let pickTimer = null;
    el2.addEventListener("click", () => {
        clearTimeout(pickTimer);
        pickTimer = setTimeout(() => onPick?.(name, el2), 250);
    });
    el2.addEventListener("dblclick", (e) => {
        e.preventDefault();
        clearTimeout(pickTimer);
        onAdd?.(name, el2);
    });
}

// 缩略图 error -> 内联 SVG 占位（卡片 108px 与列表行 40px 共用；有 src 才不会被
// 浏览器画成破损图；守卫防占位自身 error 循环）。
function wireThumb(th, bust) {
    th.loading = "lazy";
    th.alt = "";
    th.addEventListener("error", () => {
        if (th.src === THUMB_PLACEHOLDER) return;
        th.classList.add("noimg");
        th.src = THUMB_PLACEHOLDER;
    });
    th.src = thumbUrl(String(th.dataset.name || ""), bust);
}

// ── 网格卡片（视图 grid）────────────────────────────────────────────────────
function folderCard(folderName, onEnterFolder) {
    const c = el("div", "sf-lb-card folder");
    c.dataset.folderName = folderName;
    c.append(el("div", "sf-lb-foldericon", "📁"), el("div", "sf-lb-cardname", folderName));
    c.title = folderName;
    c.addEventListener("click", () => onEnterFolder?.(folderName));
    return c;
}

function fileCard(name, { selectedName = null, onPick, onAdd, thumbBust = 0 } = {}) {
    const { base, folder } = splitName(name);
    const c = el("div", "sf-lb-card" + (name === selectedName ? " sel" : ""));
    c.dataset.name = name;
    const th = document.createElement("img");
    th.className = "sf-lb-thumb";
    th.dataset.name = name;
    wireThumb(th, thumbBust);
    const nm = el("div", "sf-lb-cardname", base);
    nm.title = base;
    // 层级浏览下当前层文件同目录：副行显示扩展名（有信息量且不冗余）
    const ext = base.includes(".") ? base.slice(base.lastIndexOf(".") + 1) : "";
    const pt = el("div", "sf-lb-cardmeta", (folder ? folder + " / " : "") + (ext ? ext.toUpperCase() : "LORA"));
    pt.title = name;
    c.append(th, nm, pt);
    attachPickAdd(c, name, onPick, onAdd);
    return c;
}

// ── 列表行（视图 list）──────────────────────────────────────────────────────
function folderRow(folderName, onEnterFolder) {
    const r = el("div", "sf-lb-row folder");
    r.dataset.folderName = folderName;
    r.append(el("span", "sf-lb-rowicon", "📁"), el("div", "sf-lb-rowname", folderName));
    r.title = folderName;
    r.addEventListener("click", () => onEnterFolder?.(folderName));
    return r;
}

function fileRow(name, { selectedName = null, onPick, onAdd, thumbBust = 0 } = {}) {
    const { base, folder } = splitName(name);
    const r = el("div", "sf-lb-row" + (name === selectedName ? " sel" : ""));
    r.dataset.name = name;
    const th = document.createElement("img");
    th.className = "sf-lb-thumb-sm";
    th.dataset.name = name;
    wireThumb(th, thumbBust);
    const nm = el("div", "sf-lb-rowname", base);
    nm.title = name;
    const ext = base.includes(".") ? base.slice(base.lastIndexOf(".") + 1) : "";
    const meta = el("div", "sf-lb-rowmeta", (folder ? folder + " / " : "") + (ext ? ext.toUpperCase() : "LORA"));
    meta.title = name;
    r.append(th, nm, meta);
    attachPickAdd(r, name, onPick, onAdd);
    return r;
}

// 容器：网格（.sf-lb-grid）或列表（.sf-lb-list）
function container(view) {
    return el("div", view === "list" ? "sf-lb-list" : "sf-lb-grid");
}

// ── 层级渲染（文件夹模式）───────────────────────────────────────────────────
// 无查询：当前目录 = 立即子文件夹 + 当前层文件（文件夹下钻模型）。
// 有查询：跨全部层级扁平过滤（对齐 image_browser 的搜索语义，忽略目录）。
// `view`：grid（卡片）| list（行）。文件项点击 onPick/onAdd；文件夹点击 onEnterFolder。
export function renderFolder(main, { list = [], folder = "", query = "", selectedName = null,
    view = "grid", onPick, onAdd, onEnterFolder, thumbBust = 0 } = {}) {
    main.innerHTML = "";
    if (!list.length) {
        main.appendChild(el("div", "sf-lb-empty", "No LoRAs on this machine yet. Add some to the models/loras folder."));
        return;
    }
    const listView = view === "list";
    const q = String(query || "").trim();
    const box = container(view);
    const addFile = (name) => box.appendChild(listView
        ? fileRow(name, { selectedName, onPick, onAdd, thumbBust })
        : fileCard(name, { selectedName, onPick, onAdd, thumbBust }));
    if (q) {
        // 搜索：扁平匹配（镜像 image_browser：搜索时忽略目录层级）
        const hits = filterLoras(list, q);
        if (!hits.length) {
            main.appendChild(el("div", "sf-lb-empty", "No LoRAs match your search."));
            return;
        }
        for (const name of hits) addFile(name);
    } else {
        const { folders, files } = folderContents(list, folder);
        if (!folders.length && !files.length) {
            main.appendChild(el("div", "sf-lb-empty",
                folder ? "This folder is empty." : "No LoRAs here."));
            return;
        }
        for (const fd of folders) box.appendChild(listView ? folderRow(fd, onEnterFolder) : folderCard(fd, onEnterFolder));
        for (const name of files) addFile(name);
    }
    main.appendChild(box);
}

// ── 平面渲染（所有 LoRA 模式）───────────────────────────────────────────────
// 一次只渲染 `shown` 项（分批），剩余部分在滚动接近底部时经 attachFlatScroll
// 通知主扩展续载——列表上千条也不一次性建 DOM/拉图。
export function renderFlat(main, { names = [], shown = names.length, selectedName = null,
    view = "grid", onPick, onAdd, thumbBust = 0 } = {}) {
    main.innerHTML = "";
    if (!names.length) {
        main.appendChild(el("div", "sf-lb-empty", "No LoRAs on this machine yet. Add some to the models/loras folder."));
        return;
    }
    const listView = view === "list";
    const box = container(view);
    const slice = names.slice(0, Math.max(1, shown));
    for (const name of slice) box.appendChild(listView
        ? fileRow(name, { selectedName, onPick, onAdd, thumbBust })
        : fileCard(name, { selectedName, onPick, onAdd, thumbBust }));
    main.appendChild(box);
    if (slice.length < names.length) {
        main.appendChild(el("div", "sf-lb-loadmore", "Loading more… (" + slice.length + " / " + names.length + ")"));
    }
}
// ── 滚动动态加载（平面模式用）───────────────────────────────────────────────
// 幂等绑定：接近底部 300px 时回调 onNeedMore()。主扩展负责判断是否还有更多
// 并推进批次。
export function attachFlatScroll(scrollEl, onNeedMore) {
    if (!scrollEl || scrollEl._sfFlatScroll) return;
    scrollEl._sfFlatScroll = true;
    scrollEl.addEventListener("scroll", () => {
        const top = scrollEl.scrollTop || 0;
        const ch = scrollEl.clientHeight || 0;
        const sh = scrollEl.scrollHeight || 0;
        if (top + ch >= sh - 300) onNeedMore?.();
    }, { passive: true });
}