// ==========================================================================
// sf_prompt_tags_editor.js - SFPromptTags 全屏标签库编辑器
// ==========================================================================
//
// 从节点的 "Tags" 按钮打开，铺满整个视口（与 Pixaroma 编辑器同款）：
// 左侧分类侧栏（Text/List 双块 + 桶 + 拖拽排序 + ⋯ 菜单），右侧分类头部 +
// 顶部创建表单 + 标签卡片网格（即时编辑、Text/List 切换、Picks 模式行、
// Insert / 删除）。
//
// 编辑的是库的"工作副本"：所有变更经 commitLibrary（防抖持久化 + 实时通知
// 每个节点）；打开时 reloadLibrary 强制重读（跨标签页同步）；关闭时仅当
// isSameAsStored 判定确实有变化才写回，绝不覆盖另一个标签页的编辑。
//
// 没有撤销（applyChange 直接落工作副本），所以一切可能丢失的操作都先经过
// confirmDanger 提问，并尽量提供"先导出备份"。
//
// 数据 / 存储见 sf_prompt_tags_lib.js / sf_prompt_tags_store.js，
// 游标见 sf_prompt_tags_cursors.js，Ctrl+Z 守卫见 sf_prompt_tags_guard.js。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installGraphUndoGuard } from "./sf_prompt_tags_guard.js";
import {
    getLibrary, reloadLibrary, isSameAsStored, commitLibrary, flushLibrary, applyImport,
    fetchDefaultLibrary,
} from "./sf_prompt_tags_store.js";
import {
    isListTag, tagLines, catOf, sideOfCat, tagMode, catMode,
    reorderCategoryStep, reorderCategoryTo, canMoveCategory,
    exportLibraryJSON, parseImport, importCategories, subsetImport,
    TEXT_BUCKET, LIST_BUCKET, NAME_RE, MODES, MODE_LABEL, DEFAULT_MODE, hasPosition,
    uniqueTagName, normalizeLibrary,
} from "./sf_prompt_tags_lib.js";
import {
    listKey, catKey, cursorInfo, resetCursor, renameCursor, flushCursors,
} from "./sf_prompt_tags_cursors.js";
import { pinyinMatch } from "./sf_prompt_tags_pinyin.js";

const BRAND = "#f66744";
const PAL = ["#e0894b", "#5aa9e6", "#8e7bd6", "#5fbf8f", "#d76b98", "#c9a24b", "#6fb3b8"];
const MAX_IMPORT_BYTES = 8 * 1024 * 1024;
// 拖拽分类行时在 MIME 类型里携带它的侧，因为 dragover 阶段唯一可读的就是类型
// 列表（getData 被浏览器封到 drop 才放行）——Text 行可以当场拒绝 List 行。
const CAT_MIME = (side) => `application/x-sfnodes-prompt-tag-cat-${side}`;
// 侧栏宽度（用户拖过），记住到未注册设置；读写都夹紧，防手改值把列表弄到不可用
const SIDE_W_SETTING = "sfnodes.PromptTags.LibrarySidebar";
const SIDE_W_DEFAULT = 220, SIDE_W_MIN = 150, SIDE_W_MAX = 460;
const clampSideW = (n) => Math.max(SIDE_W_MIN, Math.min(SIDE_W_MAX, Math.round(Number(n) || 0) || SIDE_W_DEFAULT));

// 两个图标内联为 data URI（sfnodes 没有 pixaroma 的资产服务路由）
const ICON_DELETE = "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M11.381,21.396h41.237l-5.143,38.175c-.281,1.924-1.931,3.35-3.876,3.35h-23.2c-1.944,0-3.594-1.426-3.876-3.35l-5.143-38.175ZM50.148,6.863h-12.997v-2.935c0-1.176-.953-2.13-2.13-2.13h-6.043c-1.176,0-2.13.953-2.13,2.13v2.935h-12.997c-3.934.235-7.003,3.493-7.003,7.434v2.984h50.302v-2.984c0-3.941-3.07-7.199-7.003-7.434Z"/></svg>');
const ICON_HELP = "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M58.115,1.482H5.885C3.239,1.482,1.094,3.627,1.094,6.273v51.453c0,2.646,2.145,4.791,4.791,4.791h52.23c2.646,0,4.791-2.145,4.791-4.79V6.273c0-2.646-2.145-4.791-4.791-4.791ZM31.568,55.964c-2.992,0-5.417-2.425-5.417-5.417s2.425-5.417,5.417-5.417,5.417,2.425,5.417,5.417-2.425,5.417-5.417,5.417ZM45.58,25.271c-3.529,7.741-9.903,6.913-10.121,15.722h-8.529c.08-11.174,5.01-11.133,8.593-16.076,6.349-8.782-8.514-13.088-8.625-3.557h-9.752c.312-21.491,36.915-14.63,28.435,3.911Z"/></svg>');

let _overlay = null;
let _node = null;
let _opts = null;
let _data = null;       // 工作副本
let _curCat = "All";
let _search = "";
let _undoGuardOff = null;
let _catMenu = null;
let _accent = BRAND;
// 创建表单进行中的值（点侧栏分类 / 打字搜索都会重渲染表单，这些值跨重渲染存活）。
// Create 与关闭时清空。多行文本（2+ 行）初始即判为 List；用户手动切过后
// （kindTouched）不再跟随文本自动切换。
function newDraft(text) {
    const t = text || "";
    return { name: "", text: t, cat: null, kind: tagLines(t).length > 1 ? "list" : "text", kindTouched: false };
}
let _createDraft = newDraft();

function clone(d) {
    return {
        version: 1,
        categories: [...d.categories],
        listCats: [...(d.listCats || [])],
        catModes: { ...(d.catModes || {}) },
        tags: d.tags.map((t) => ({ ...t })),
    };
}
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function sanitizeName(n) { return String(n || "").replace(NAME_RE, ""); }
function colorOf(cat) {
    // 桶不是真实分类——中性灰
    if (!cat || cat === TEXT_BUCKET || cat === LIST_BUCKET) return "#7a7a7a";
    const i = _data.categories.indexOf(cat);
    return PAL[(i < 0 ? 0 : i) % PAL.length];
}
function tagsIn(cat) { return _data.tags.filter((t) => catOf(t) === cat); }
// 侧：对工作副本判断（不是持久化库）
function sideOf(cat) { return sideOfCat(cat, _data); }
function catsOnSide(side) { return _data.categories.filter((c) => sideOf(c) === side); }
// 桶只有在真有标签坐在里面时才显示
function bucketUsed(side) {
    return _data.tags.some((t) => !t.cat && (isListTag(t) ? "list" : "text") === side);
}
const bucketOf = (side) => (side === "list" ? LIST_BUCKET : TEXT_BUCKET);
// "Text" / "List"（及旧 "Uncategorized"）是桶名，永远不能成为分类
function isReservedName(v) {
    const k = String(v || "").trim().toLowerCase();
    return k === TEXT_BUCKET.toLowerCase() || k === LIST_BUCKET.toLowerCase() || k === "uncategorized";
}
function addCategory(name, side) {
    _data.categories.push(name);
    if (side === "list") _data.listCats.push(name);
}
function uniqueNameExcept(base, exceptTag) {
    return uniqueTagName(base, _data.tags, exceptTag);
}
function commit() { commitLibrary(_data); }

// ── 应用变更 ────────────────────────────────────────────────────────────
// 有意不做撤销：一切可能丢东西的操作先确认（confirmDanger），然后直接落工作副本。
// 别在这里重新引入撤销（历史上它曾是本编辑器最大的 bug 源）。
function applyChange(mutate) {
    mutate();
    commit();
    render();
}

function injectCSS() {
    if (document.getElementById("sf-ptge-css")) return;
    const s = document.createElement("style");
    s.id = "sf-ptge-css";
    s.textContent = `
    .sf-ptge { position:fixed; inset:0; z-index:10040; background:#181818; color:#e6e6e6;
      font:14px 'Segoe UI',system-ui,sans-serif; display:flex; flex-direction:column; }
    .sf-ptge * { scrollbar-color:#3d3d3d #181818; scrollbar-width:thin; }
    .sf-ptge ::-webkit-scrollbar { width:12px; height:12px; }
    .sf-ptge ::-webkit-scrollbar-track { background:#181818; }
    .sf-ptge ::-webkit-scrollbar-thumb { background:#3d3d3d; border-radius:6px; border:2px solid #181818; }
    .sf-ptge ::-webkit-scrollbar-thumb:hover { background:#505050; }
    .sf-ptge-bar { display:flex; align-items:center; gap:10px; background:#161616; border-bottom:1px solid #0e0e0e; padding:11px 16px; }
    .sf-ptge-bar .ttl { font-weight:500; font-size:15px; color:#fff; display:flex; align-items:center; gap:8px; }
    .sf-ptge-bar .ttl .cr { color:var(--acc); }
    .sf-ptge-srch { width:320px; max-width:36vw; display:flex; align-items:center; gap:8px; background:#1d1d1d; border:1px solid #3a3a3a; border-radius:6px; padding:6px 10px; margin-left:8px; }
    .sf-ptge-srch input { flex:1; background:transparent; border:0; outline:none; color:#e6e6e6; font:13px 'Segoe UI',sans-serif; }
    .sf-ptge-srch .i { color:#767676; }
    .sf-ptge-bar .priv { margin-left:6px; color:#767676; font-size:11.5px; }
    .sf-ptge-bar .help { margin-left:auto; width:30px; height:30px; display:flex; align-items:center; justify-content:center; color:#a6a6a6; cursor:pointer; border-radius:6px; }
    .sf-ptge-bar .help:hover { background:rgba(255,255,255,.08); color:#fff; }
    .sf-ptge-bar .help .sf-ptge-svg { width:17px; height:17px; }
    .sf-ptge-bar .x { color:#a6a6a6; cursor:pointer; font-size:20px; line-height:1; padding:3px 9px; border-radius:6px; }
    .sf-ptge-bar .x:hover { background:rgba(255,255,255,.08); color:#fff; }
    .sf-ptge-main { flex:1; display:flex; min-height:0; }
    .sf-ptge-side { width:220px; flex:none; background:#1b1b1b; border-right:1px solid #101010; padding:10px; overflow-y:auto; display:flex; flex-direction:column; gap:3px; }
    /* 拖侧栏与卡片之间的接缝来加宽分类列表；6px 的条压在边框上，边框保持 1px */
    .sf-ptge-grip { flex:none; width:6px; margin-left:-3px; margin-right:-3px; z-index:2;
      cursor:col-resize; background:transparent; transition:background .12s; }
    .sf-ptge-grip:hover, .sf-ptge-grip.on { background:var(--acc); }
    /* 拖接缝期间，其它任何东西不得抢指针或画选区 */
    .sf-ptge.resizing { cursor:col-resize; user-select:none; }
    .sf-ptge.resizing .sf-ptge-main * { pointer-events:none; }
    .sf-ptge.resizing .sf-ptge-grip { pointer-events:auto; background:var(--acc); }
    /* 分类行排序：强调线表示落在行上/下，被拖的行变暗 */
    .sf-ptge-cat.ins-above { box-shadow: inset 0 2px 0 0 var(--acc); }
    .sf-ptge-cat.ins-below { box-shadow: inset 0 -2px 0 0 var(--acc); }
    .sf-ptge-cat.dragging-me { opacity:.45; }
    .sf-ptge-side .lbl { font:600 10px 'Segoe UI',sans-serif; letter-spacing:.1em; text-transform:uppercase; color:#767676; padding:4px 8px 8px; }
    .sf-ptge-cat { display:flex; align-items:center; gap:9px; padding:9px 10px; border-radius:7px; cursor:pointer; color:#c9c9c9; font:13px 'Segoe UI',sans-serif; }
    .sf-ptge-cat:hover { background:rgba(255,255,255,.05); color:#fff; }
    .sf-ptge-cat.on { background:color-mix(in srgb, var(--acc) 18%, transparent); color:#fff; }
    .sf-ptge-cat .cd { width:11px; height:11px; border-radius:50%; flex:none; }
    .sf-ptge-cat .nm { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-ptge-cat .cnt { font-size:11px; color:#767676; }
    .sf-ptge-cat.on .cnt { color:rgba(255,255,255,.7); }
    .sf-ptge-cat .act { opacity:0; color:#767676; font-size:12px; padding:0 2px; }
    /* ⋯ 常驻（悬停前变暗）——hover-only 的按钮没人找得到 */
    .sf-ptge-cat .act.more { opacity:.6; font-size:14px; line-height:1; padding:1px 5px; border-radius:4px; }
    .sf-ptge-cat:hover .act { opacity:1; }
    .sf-ptge-cat .act:hover { color:var(--acc); }
    .sf-ptge-cat .act.more:hover { background:rgba(255,255,255,.1); color:#fff; }
    .sf-ptge-cat.on .act.more { opacity:.85; }
    /* 桶行不是分类：斜体 + 暗淡，一眼看出"另一类行" */
    .sf-ptge-cat.bucket .nm { font-style:italic; color:#9a9a9a; }
    .sf-ptge-cat.bucket:hover .nm { color:#fff; }
    .sf-ptge-cat.bucket.on .nm { color:#e0e0e0; }
    .sf-ptge-cat .catinput { flex:1; min-width:0; background:#151515; border:1px solid var(--acc); border-radius:4px; color:#e6e6e6; font:12.5px monospace; padding:4px 6px; outline:none; }
    .sf-ptge-newcat { margin-top:6px; padding-top:9px; border-top:1px solid #262626; }
    .sf-ptge-btn { background:rgba(255,255,255,.05); border:1px solid #4a4a4a; color:#a6a6a6; border-radius:6px; padding:7px 13px; font:12.5px 'Segoe UI',sans-serif; cursor:pointer; display:inline-flex; gap:6px; align-items:center; transition:.12s; }
    .sf-ptge-btn:hover { border-color:var(--acc); color:#fff; }
    .sf-ptge-btn.pri { color:#fff; background:var(--acc); border-color:var(--acc); }
    .sf-ptge-btn.pri:hover { filter:brightness(1.08); }
    .sf-ptge-newcat .sf-ptge-btn { width:100%; justify-content:center; }
    .sf-ptge-content { flex:1; display:flex; flex-direction:column; min-width:0; background:#212121; }
    .sf-ptge-chead { display:flex; align-items:center; gap:10px; padding:12px 16px; border-bottom:1px solid #171717; }
    /* min-width:0 + 省略号，防长分类名把 Picks 控件推出右缘 */
    .sf-ptge-chead .h { display:flex; align-items:center; gap:9px; font-size:15px; color:#fff; font-weight:500; min-width:0; overflow:hidden; }
    .sf-ptge-chead .h > span:not(.cd) { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-ptge-chead .h .cd { width:12px; height:12px; border-radius:50%; }
    .sf-ptge-chead .h .c { color:#767676; font-weight:400; font-size:12.5px; }
    /* 顶部创建表单：一处填名称 + 文本，不用去编辑器另一头找按钮 */
    .sf-ptge-create { display:flex; align-items:center; flex-wrap:wrap; row-gap:8px; gap:8px; padding:11px 16px; background:#1e1e1e; border-bottom:1px solid #171717; }
    .sf-ptge-create .ccat { max-width:220px; }
    .sf-ptge-create input, .sf-ptge-create textarea { background:#151515; border:1px solid #3a3a3a; border-radius:5px; color:#e6e6e6; font:12.5px monospace; padding:8px 9px; outline:none; height:36px; box-sizing:border-box; }
    .sf-ptge-create input:focus, .sf-ptge-create textarea:focus { border-color:var(--acc); }
    .sf-ptge-create .cnm { width:170px; flex:none; color:var(--acc); }
    .sf-ptge-create .ctx { flex:1; min-width:0; resize:none; line-height:1.5; white-space:pre-wrap; overflow-y:auto; }
    .sf-ptge-create .ccat { flex:none; height:36px; }
    .sf-ptge-create .ccat .car { font-size:9px; opacity:.85; margin-left:1px; }
    .sf-ptge-create .cbtn { flex:none; background:var(--acc); border:none; color:#fff; border-radius:5px; padding:9px 15px; font:500 12.5px 'Segoe UI',sans-serif; cursor:pointer; height:36px; }
    .sf-ptge-create .cbtn:hover { filter:brightness(1.08); }
    /* 卡片网格：紧凑卡片自动填满多列 */
    .sf-ptge-grid { flex:1; overflow-y:auto; padding:13px 15px; display:grid;
      grid-template-columns:repeat(auto-fill, minmax(255px, 1fr)); gap:11px; align-content:start; }
    .sf-ptge-card { background:#282828; border:1px solid #333; border-radius:9px; padding:10px; display:flex; flex-direction:column; gap:7px; min-width:0; }
    .sf-ptge-card .ctop { display:flex; align-items:center; gap:6px; }
    .sf-ptge-card .cnm { flex:1; min-width:0; background:#1d1d1d; border:1px solid #3a3a3a; border-radius:5px; color:var(--acc); font:13px monospace; padding:6px 8px; outline:none; }
    .sf-ptge-card .cnm:focus { border-color:var(--acc); }
    .sf-ptge-card .ctop .sf-ptge-pill { flex:none; max-width:52%; }
    .sf-ptge-card .ctx { background:#1d1d1d; border:1px solid #3a3a3a; border-radius:5px; color:#e0e0e0; font:11.5px/1.45 monospace; padding:7px 8px; outline:none; resize:vertical; min-height:66px; }
    .sf-ptge-card .ctx:focus { border-color:var(--acc); }
    .sf-ptge-card .cfoot { display:flex; gap:6px; }
    .sf-ptge-svg { display:block; width:15px; height:15px; background-color:currentColor;
      -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat; -webkit-mask-position:center; mask-position:center; -webkit-mask-size:contain; mask-size:contain; }
    .sf-ptge-empty { color:#767676; font-size:13px; padding:24px; text-align:center; }
    .sf-ptge-pill { display:inline-flex; align-items:center; gap:7px; background:#3a3a3a; border:1px solid #4a4a4a; border-radius:20px; padding:6px 11px; font:12px 'Segoe UI',sans-serif; color:#d6d6d6; cursor:pointer; white-space:nowrap; overflow:hidden; }
    .sf-ptge-pill:hover { border-color:var(--acc); color:#fff; }
    .sf-ptge-pill .cd { width:10px; height:10px; border-radius:50%; flex:none; }
    .sf-ptge-insert { flex:1; min-width:74px; height:30px; border-radius:5px; border:1px solid var(--acc); background:transparent;
      color:var(--acc); cursor:pointer; font:12px 'Segoe UI',sans-serif; display:flex; align-items:center; justify-content:center; gap:5px; }
    .sf-ptge-insert:hover { background:var(--acc); color:#fff; }
    .sf-ptge-insert .sf-ptge-svg { width:13px; height:13px; }
    .sf-ptge-insert.ok, .sf-ptge-insert.ok:hover { background:#3ec371; border-color:#3ec371; color:#fff; }
    .sf-ptge-ic { width:32px; height:30px; border-radius:5px; border:1px solid #4a4a4a; background:transparent; color:#a6a6a6; cursor:pointer; display:flex; align-items:center; justify-content:center; font-size:14px; }
    .sf-ptge-ic:hover { border-color:var(--acc); color:#fff; }
    .sf-ptge-ic.del:hover { background:#e2554a; border-color:#e2554a; color:#fff; }
    /* Text / List 切换：两个选项常显，选中的用强调色 */
    .sf-ptge-kindsw { flex:none; display:inline-flex; height:30px; border:1px solid #4a4a4a; border-radius:5px; overflow:hidden; }
    .sf-ptge-kindsw:hover { border-color:var(--acc); }
    .sf-ptge-kindsw button { background:transparent; border:0; color:#a6a6a6; padding:0 9px; cursor:pointer;
      font:11.5px 'Segoe UI',sans-serif; display:inline-flex; align-items:center; white-space:nowrap; }
    .sf-ptge-kindsw button:hover { background:rgba(255,255,255,.07); color:#fff; }
    .sf-ptge-kindsw button.on, .sf-ptge-kindsw button.on:hover { background:var(--acc); color:#fff; }
    .sf-ptge-card.islist { border-color:color-mix(in srgb, var(--acc) 42%, #333); }
    .sf-ptge-card .cfoot { flex-wrap:wrap; row-gap:6px; }
    .sf-ptge-create .sf-ptge-kindsw { height:36px; }
    /* Picks 模式行：位置有地方可显示 */
    .sf-ptge-moderow { display:flex; align-items:center; gap:7px; min-width:0; }
    .sf-ptge-moderow .cap { flex:none; color:#767676; font:600 9.5px 'Segoe UI',sans-serif; letter-spacing:.09em; text-transform:uppercase; }
    .sf-ptge-mode { flex:none; height:26px; padding:0 9px; border-radius:5px; border:1px solid #4a4a4a; background:transparent;
      color:#a6a6a6; cursor:pointer; font:11.5px 'Segoe UI',sans-serif; display:inline-flex; align-items:center; gap:6px; white-space:nowrap; }
    .sf-ptge-mode:hover { border-color:var(--acc); color:#fff; }
    .sf-ptge-mode.set { border-color:var(--acc); color:var(--acc); }
    .sf-ptge-mode .car { font-size:9px; opacity:.85; }
    .sf-ptge-moderow .pos { flex:1; min-width:0; text-align:right; color:#767676; font-size:11px;
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-ptge-moderow .rst { flex:none; width:24px; height:24px; border-radius:5px; border:1px solid #4a4a4a;
      background:transparent; color:#a6a6a6; cursor:pointer; font-size:12px; line-height:1; display:none;
      align-items:center; justify-content:center; }
    .sf-ptge-moderow.on .rst { display:flex; }
    .sf-ptge-moderow .rst:hover { border-color:var(--acc); color:#fff; }
    .sf-ptge-menu .mi.on { color:var(--acc); }
    .sf-ptge-chead .sf-ptge-moderow { margin-left:auto; flex:0 0 auto; }
    .sf-ptge-chead .sf-ptge-moderow .pos { flex:0 0 auto; }
    /* 导入预览：从文件带哪些分类 */
    .sf-ptge-pick { display:flex; flex-direction:column; gap:6px; max-height:42vh; overflow-y:auto; padding:2px 16px 8px; }
    .sf-ptge-pick .row { display:flex; align-items:center; gap:10px; background:#262626; border:1px solid #333;
      border-radius:8px; padding:9px 12px; cursor:pointer; }
    .sf-ptge-pick .row:hover { border-color:var(--acc); }
    .sf-ptge-pick .row input { accent-color:var(--acc); width:15px; height:15px; cursor:pointer; flex:none; }
    .sf-ptge-pick .row .cd { width:10px; height:10px; border-radius:50%; flex:none; }
    .sf-ptge-pick .row .nm { flex:1; min-width:0; color:#fff; font:13px 'Segoe UI',sans-serif; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-ptge-pick .row .cnt { color:#a6a6a6; font-size:11.5px; flex:none; }
    .sf-ptge-mfoot { display:flex; align-items:center; gap:9px; padding:2px 16px 16px; }
    .sf-ptge-mfoot .push { margin-left:auto; }
    .sf-ptge-mlink { background:none; border:0; color:var(--acc); font:12px 'Segoe UI',sans-serif; cursor:pointer; padding:2px 4px; }
    .sf-ptge-mlink:hover { text-decoration:underline; }
    .sf-ptge-menu .mrow { display:flex; align-items:center; gap:9px; }
    .sf-ptge-menu .mrow .nm { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-ptge-menu .mrow .cnt { color:#767676; font-size:11px; flex:none; }
    .sf-ptge-menu .mi.dim { opacity:.45; cursor:default; }
    .sf-ptge-menu .mi.dim:hover { background:none; color:#cfcfcf; }
    .sf-ptge-menu .msep { height:1px; background:#2a2a2a; margin:4px 2px; }
    .sf-ptge-menu .mhead { padding:4px 10px 5px; font:600 9.5px 'Segoe UI',sans-serif; letter-spacing:.09em; text-transform:uppercase; color:#767676; }
    .sf-ptge-menu .mnote { padding:0 10px 7px; font:11.5px/1.45 'Segoe UI',sans-serif; color:#8f8f8f; white-space:normal; }
    .sf-ptge-foot { display:flex; align-items:center; gap:9px; padding:10px 16px; border-top:1px solid #0e0e0e; background:#161616; }
    .sf-ptge-foot .push { margin-left:auto; }
    /* max-height + 滚动，防分类很多时菜单被推出屏幕 */
    .sf-ptge-menu { position:fixed; z-index:10050; background:#1d1d1d; border:1px solid #4a4a4a; border-radius:7px; padding:5px; box-shadow:0 12px 30px rgba(0,0,0,.6); min-width:170px; max-height:min(60vh,520px); overflow-y:auto; }
    .sf-ptge-menu .mi { display:flex; align-items:center; gap:9px; padding:7px 10px; border-radius:5px; cursor:pointer; font:12.5px 'Segoe UI',sans-serif; color:#cfcfcf; }
    .sf-ptge-menu .mi:hover { background:rgba(255,255,255,.06); color:#fff; }
    .sf-ptge-menu .mi .cd { width:10px; height:10px; border-radius:50%; }
    .sf-ptge-menu .mi.newc { border-top:1px solid #2a2a2a; margin-top:4px; padding-top:8px; color:var(--acc); }
    .sf-ptge-menu input { width:100%; background:#151515; border:1px solid #4a4a4a; border-radius:4px; color:#e6e6e6; font:12px monospace; padding:6px 8px; outline:none; margin-top:5px; }
    .sf-ptge-modal { position:absolute; inset:0; background:rgba(0,0,0,.6); display:flex; align-items:center; justify-content:center; z-index:10045; }
    .sf-ptge-mcard { background:#202020; border:1px solid #0e0e0e; border-radius:12px; width:460px; max-width:92vw; box-shadow:0 20px 60px rgba(0,0,0,.6); overflow:hidden; }
    .sf-ptge-mcard .mh { padding:14px 16px; border-bottom:1px solid #171717; font:500 15px 'Segoe UI',sans-serif; color:#fff; }
    .sf-ptge-mcard .mb { padding:14px 16px; color:#a6a6a6; font-size:13px; line-height:1.6; }
    .sf-ptge-mcard .mb b { color:#fff; font-weight:500; }
    .sf-ptge-mcard .conf { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:7px; padding:8px 11px; margin:9px 0;
      font:12px monospace; color:#e0894b; max-height:110px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; }
    .sf-ptge-opts { display:flex; flex-direction:column; gap:8px; padding:2px 16px 16px; }
    .sf-ptge-opt { display:flex; align-items:center; gap:11px; background:#262626; border:1px solid #333; border-radius:8px; padding:11px 13px; cursor:pointer; transition:.12s; }
    .sf-ptge-opt:hover, .sf-ptge-opt.rec { border-color:var(--acc); }
    .sf-ptge-opt .oic { width:30px; height:30px; border-radius:7px; background:color-mix(in srgb, var(--acc) 16%, transparent); color:var(--acc); display:flex; align-items:center; justify-content:center; font-size:15px; flex:none; }
    .sf-ptge-opt .t { font:500 13px 'Segoe UI',sans-serif; color:#fff; }
    .sf-ptge-opt .t small { display:block; color:#a6a6a6; font-weight:400; font-size:11.5px; margin-top:1px; }
    .sf-ptge-opt .rtag { margin-left:auto; font-size:10px; color:#3ec371; border:1px solid rgba(62,195,113,.4); border-radius:12px; padding:1px 8px; }
    .sf-ptge-help-card { width:560px; }
    .sf-ptge-help-card .mb { max-height:60vh; overflow-y:auto; }
    .sf-ptge-help-card .mb p { margin:0 0 11px; }
    .sf-ptge-help-card .mb p:last-child { margin-bottom:0; }
    .sf-ptge-help-foot { display:flex; justify-content:flex-end; padding:0 16px 16px; }
    /* 会销毁东西的菜单行绝不能看起来和普通行一样 */
    .sf-ptge-menu .mi.danger { color:#e2554a; }
    .sf-ptge-menu .mi.danger:hover { background:rgba(226,85,74,.15); color:#ff8d81; }
    .sf-ptge-menu .mi.danger .cnt { color:#d98079; }
    .sf-ptge-btn.danger { border-color:#e2554a; color:#ff8378; background:rgba(226,85,74,.12); }
    .sf-ptge-btn.danger:hover { background:#e2554a; border-color:#e2554a; color:#fff; }
  `;
    document.head.appendChild(s);
}

function hideCatMenu() { if (_catMenu) { _catMenu.remove(); _catMenu = null; } }

// 把挂在 body 的弹窗夹进视口，下方空间不足时翻到上方。菜单变高后需重新调用。
function placeMenu(menu, anchor) {
    const r = anchor.getBoundingClientRect();
    menu.style.left = Math.max(8, Math.min(r.left, window.innerWidth - menu.offsetWidth - 8)) + "px";
    const below = window.innerHeight - r.bottom;
    menu.style.top = (below < menu.offsetHeight + 8
        ? Math.max(8, r.top - menu.offsetHeight - 6)
        : r.bottom + 4) + "px";
}

// 单侧分类选择器（该侧分类 + 该侧桶 + 新建分类行），onPick(catValue)（"" = 桶）。
// 不重渲染——由调用方决定（创建表单的输入因此不被清掉）。
function openCategoryMenu(anchor, onPick, side) {
    hideCatMenu();
    const sd = side === "list" ? "list" : "text";
    const menu = document.createElement("div");
    menu.className = "sf-ptge-menu";
    for (const c of [...catsOnSide(sd), bucketOf(sd)]) {
        const mi = document.createElement("div");
        mi.className = "mi";
        mi.innerHTML = `<span class="cd" style="background:${colorOf(c)}"></span>${esc(c)}`;
        mi.addEventListener("click", () => { hideCatMenu(); onPick(c === bucketOf(sd) ? "" : c); });
        menu.appendChild(mi);
    }
    const nc = document.createElement("div");
    nc.className = "mi newc";
    nc.innerHTML = `<span>＋</span> New ${sd === "list" ? "list " : ""}category`;
    const inp = document.createElement("input");
    inp.placeholder = "name";
    inp.style.display = "none";
    nc.addEventListener("click", () => {
        inp.style.display = "block";
        inp.focus();
        // 菜单是按展开前的高度定位的，矮窗口上露出的输入框可能掉到屏幕外
        placeMenu(menu, anchor);
    });
    inp.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Enter") {
            const v = inp.value.trim();
            // 空输入框直接 Enter：提示而不是默默关掉（从桶的"全部移入分类"进来时
            // 这读起来像个坏按钮）
            if (!v) { toast("info", "Type a name for the new category."); inp.focus(); return; }
            // 桶名不是分类：输入它只是把标签归档到桶（绝不能塞进一个幻影分类行）
            const reserved = v && isReservedName(v);
            // 与既有分类大小写冲突时用既有的（canonical），绝不造一个侧栏对不上的
            // 错大小写分类
            const existing = (v && !reserved) ? _data.categories.find((c) => c.toLowerCase() === v.toLowerCase()) : null;
            if (v && !reserved && !existing) {
                addCategory(v, sd);
                commit();
                // 只刷侧栏。整页 render() 会重建创建表单、丢掉用户已输入的内容
                // （这正是本选择器不重渲染的原因）；但不刷的话新分类要在侧栏/计数/
                // 导出菜单里等到别的巧合重渲染才出现。
                const side = _overlay && _overlay.querySelector(".sf-ptge-side");
                if (side) renderSidebar(side);
            }
            hideCatMenu();
            // 既有名字保留它自己的侧，只有侧一致才可选它
            if (v) onPick(reserved || (existing && sideOf(existing) !== sd) ? "" : (existing || v));
        }
        if (e.key === "Escape") hideCatMenu();
    });
    menu.append(nc, inp);
    _overlay.appendChild(menu);
    placeMenu(menu, anchor);
    _catMenu = menu;
}
// 把现有标签移到别的分类：只提供它自己那一侧（分类只容纳一种 kind）。持久化 + 重渲染。
function openCatMenu(tag, anchor) {
    openCategoryMenu(anchor, (c) => { tag.cat = c; commit(); render(); }, isListTag(tag) ? "list" : "text");
}
document.addEventListener("mousedown", (e) => {
    if (_catMenu && !_catMenu.contains(e.target) && !e.target.closest(".sf-ptge-pill")) hideCatMenu();
}, true);

// ── render ─────────────────────────────────────────────────────────────
// Text / List 切换（卡片与创建表单共用）。paint(isList, count) 点亮活动侧，
// List 时显示其选项数。
function makeKindSwitch(onPick) {
    const sw = document.createElement("div");
    sw.className = "sf-ptge-kindsw";
    const bText = document.createElement("button");
    bText.type = "button";
    bText.textContent = "Text";
    bText.title = "Text：一段文本，@name 整体插入";
    const bList = document.createElement("button");
    bList.type = "button";
    bList.textContent = "List";
    bList.title = "List：每行一个选项，#name 每次 run 随机取一行";
    sw.append(bText, bList);
    bText.addEventListener("click", (e) => { e.stopPropagation(); onPick(false); });
    bList.addEventListener("click", (e) => { e.stopPropagation(); onPick(true); });
    return {
        el: sw,
        paint(isList, count) {
            bText.classList.toggle("on", !isList);
            bList.classList.toggle("on", !!isList);
            bList.textContent = isList && count != null ? `List · ${count}` : "List";
        },
    };
}

const MODE_HINT = {
    random: "每次任取一个",
    shuffle: "全部出现一次后才重复",
    order: "1、2、3 循环往复",
};
function openModeMenu(anchor, current, onPick) {
    hideCatMenu();
    const menu = document.createElement("div");
    menu.className = "sf-ptge-menu";
    menu.style.minWidth = "240px";
    for (const m of MODES) {
        const mi = document.createElement("div");
        mi.className = "mi mrow" + (m === current ? " on" : "");
        mi.innerHTML = `<span class="nm">${MODE_LABEL[m]}</span><span class="cnt">${MODE_HINT[m]}</span>`;
        mi.addEventListener("click", () => { hideCatMenu(); onPick(m); });
        menu.appendChild(mi);
    }
    _overlay.appendChild(menu);
    placeMenu(menu, anchor);
    _catMenu = menu;
}
// "Random ▾ · next 3 of 12 · ↺" 行（List 卡片与分类头部共用）。
// getMode/setMode 读写模式所在地；key/len 驱动位置文本。
function makeModeRow({ getMode, setMode, key, len, what }) {
    const row = document.createElement("div");
    row.className = "sf-ptge-moderow";
    const cap = document.createElement("span");
    cap.className = "cap";
    cap.textContent = "Picks";
    const btn = document.createElement("button");
    btn.className = "sf-ptge-mode";
    const pos = document.createElement("span");
    pos.className = "pos";
    const rst = document.createElement("button");
    rst.className = "rst";
    rst.textContent = "↺";
    const paint = () => {
        const m = getMode();
        // 强调色 = "非默认"：行为异常的列表在网格里一眼可见
        btn.classList.toggle("set", m !== DEFAULT_MODE);
        btn.innerHTML = `<span>${MODE_LABEL[m]}</span><span class="car">▾</span>`;
        btn.title = `这个${what}如何选取：${MODE_LABEL[m]} - ${MODE_HINT[m]}`;
        row.classList.toggle("on", hasPosition(m));
        // Random 没有位置可显示，就说 Random 做什么——这正是让你想点这个控件的那行字
        pos.textContent = cursorInfo(key(), len(), m) || MODE_HINT[m];
        rst.title = `重新开始这个${what}`;
    };
    btn.addEventListener("click", (e) => {
        e.stopPropagation();
        openModeMenu(btn, getMode(), (m) => { setMode(m); commit(); paint(); });
    });
    rst.addEventListener("click", (e) => {
        e.stopPropagation();
        resetCursor(key());
        paint();
        toast("info", `Started that ${what} over`);
    });
    row.append(cap, btn, pos, rst);
    paint();
    return { el: row, paint };
}

function makeCard(tag) {
    const card = document.createElement("div");
    card.className = "sf-ptge-card";
    const top = document.createElement("div");
    top.className = "ctop";
    const nm = document.createElement("input");
    nm.className = "cnm";
    nm.value = tag.name;
    nm.spellcheck = false;
    nm.addEventListener("input", () => {
        const cleaned = sanitizeName(nm.value);
        if (cleaned !== nm.value) {
            // 直接写回会把光标移到末尾：在名字中间输入非法字符，光标会瞬移、
            // 后续输入全部落错位。把光标放回原处（减去被剥掉的数量）。
            const at = nm.selectionStart;
            const dropped = nm.value.length - cleaned.length;
            nm.value = cleaned;
            const p = Math.max(0, (at == null ? cleaned.length : at) - dropped);
            try { nm.setSelectionRange(p, p); } catch { /* detached / unsupported */ }
        }
        // 非法名字绝不能进工作副本，而不只是不提交。曾经它无条件赋值、只拦 commit，
        // 于是名字字段空着/重复时 _data 里的标签就是非法的——之后任何一次别的
        // commit() 都会把空名标签整个丢掉。
        const dup = !!cleaned && _data.tags.some((o) => o !== tag && o.name.toLowerCase() === cleaned.toLowerCase());
        if (cleaned && !dup) {
            tag.name = cleaned;
            // 存储在这一键即改名，所以位置现在就得搬走。拖到 blur 会让一次 run
            // 查新名字查不到、开新序列，而 blur 再覆盖它。
            if (nameAtFocus.v && nameAtFocus.v !== tag.name) {
                try { renameCursor(listKey(nameAtFocus.v), listKey(tag.name)); } catch { /* ignore */ }
                nameAtFocus.v = tag.name;
            }
            commit();
        }
        paintKind(); // kind 按钮的 tooltip 引用标签名
    });
    // 记录位置当前登记在哪个名字下，改名时把它带走（存储写入有防抖，逐键搬零成本）
    const nameAtFocus = { v: tag.name };
    // 与 nameAtFocus 分离：input 处理器逐键移动它（它追踪序列位置当前在哪），
    // Escape 需要的是进入字段那一刻的名字，任何其它代码都不能碰它
    let nameOnEntry = tag.name;
    nm.addEventListener("focus", () => { nameAtFocus.v = tag.name; nameOnEntry = tag.name; });
    // Escape 必须放弃输入的内容。onKey 是 capture 阶段，不接管的话会落到通用的
    // `active.blur()`，而 blur 监听器会提交——按 Escape 结果把名字改成了
    // `thatname-2`（没人输入过的名字）。先把原名字放回去，让 blur 的等值检查
    // 变成无操作。
    nm._sfCancel = () => {
        const back = nameOnEntry;
        if (back && back !== tag.name && !_data.tags.some((o) => o !== tag && o.name.toLowerCase() === back.toLowerCase())) {
            try { renameCursor(listKey(tag.name), listKey(back)); } catch { /* ignore */ }
            tag.name = back;
            nm.value = back;
            commit();
            nm.blur();
            render();          // 卡片的 Insert / 删除 / 切换标签都引用名字
            return;
        }
        nm.value = tag.name;
        nm.blur();
    };
    nm.addEventListener("blur", () => {
        // 留空：标签保留原名，字段显示原名（uniqueNameExcept("") 会发明 "tag" 或
        // "tag-2"，因为清空重打然后点走，白白丢掉用户没想改的名字）
        if (!sanitizeName(nm.value)) { nm.value = tag.name; return; }
        const u = uniqueNameExcept(nm.value, tag);
        // 没变化（通常情况：点进去又点出来，或 input 已应用了有效改名）就不提交
        if (u === tag.name) { if (nm.value !== u) nm.value = u; return; }
        tag.name = u;
        nm.value = u;
        if (nameAtFocus.v && nameAtFocus.v !== tag.name) {
            try { renameCursor(listKey(nameAtFocus.v), listKey(tag.name)); } catch { /* ignore */ }
            nameAtFocus.v = tag.name;
        }
        commit();
    });
    nm.addEventListener("keydown", (e) => e.stopPropagation());
    const cc = catOf(tag);
    const pill = document.createElement("button");
    pill.className = "sf-ptge-pill";
    pill.title = "移动到另一个分类";
    pill.innerHTML = `<span class="cd" style="background:${colorOf(cc)}"></span><span>${esc(cc)}</span>`;
    pill.addEventListener("click", (e) => { e.stopPropagation(); openCatMenu(tag, pill); });
    top.append(nm, pill);
    const tx = document.createElement("textarea");
    tx.className = "ctx";
    tx.value = tag.text;
    tx.spellcheck = false;
    tx.rows = 3;
    tx.addEventListener("input", () => { tag.text = tx.value; commit(); paintKind(); });
    tx.addEventListener("keydown", (e) => e.stopPropagation());
    const foot = document.createElement("div");
    foot.className = "cfoot";
    // 在 paintKind 之前声明，让 tooltip 挂在正确的名词上（List 卡片的删除按钮
    // 曾经写 "tag"，而同一个点击的确认条却写 "Deleted #name"）
    const del = document.createElement("button");
    const ins = document.createElement("button");
    ins.className = "sf-ptge-insert";
    ins.innerHTML = `<span class="lbl">Insert</span>`;
    ins.addEventListener("click", () => {
        // List 卡片插入 #name（每次 run 掷一行）；片段插入 @name
        _opts?.onInsert?.(tag.name, isListTag(tag) ? "#" : "@");
        ins.classList.add("ok");
        const l = ins.querySelector(".lbl");
        if (l) l.textContent = "Inserted ✓";
        setTimeout(() => { ins.classList.remove("ok"); const ll = ins.querySelector(".lbl"); if (ll) ll.textContent = "Insert"; }, 850);
    });
    // Text <-> List。存储的 kind 只是外观 + 便利（提示词里的符号才是真正的裁决者），
    // 所以翻转它永远不会弄坏既有提示词：@name 怎么都给整块文本。
    const kindSw = makeKindSwitch((toList) => {
        if (isListTag(tag) === !!toList) return;
        const side = toList ? "list" : "text";
        const bucket = bucketOf(side);
        // 分类只容纳一种，翻转的标签不能留在原分类：去新侧的桶，用户可从 pill 归档
        const from = tag.cat && sideOf(tag.cat) !== side ? tag.cat : "";
        const flip = () => applyChange(() => {
            if (toList) tag.kind = "list"; else delete tag.kind;
            if (from) tag.cat = "";
        });
        // 从分类里翻出来等于丢掉归档，立即翻回去也不会恢复（标签落到另一个桶）。
        // 没有撤销，这是编辑器里唯一可能被一下未确认的点击弄丢东西的地方——所以问，
        // 但只在确实有东西可丢时才问。已经在桶里的直接翻。
        if (!from) { flip(); return; }
        confirmDanger({
            title: `Move ${esc(tag.name)} out of ${esc(from)}?`,
            lead: `A category holds one kind, so making this a <b>${toList ? "List" : "Text"}</b> tag takes it out of ` +
                `<b>${esc(from)}</b> and puts it in <b>${esc(bucket)}</b>, ready to file somewhere else. ` +
                `Switching back will not return it to ${esc(from)}.`,
            confirmLabel: `Make it a ${toList ? "List" : "Text"} tag`,
            onConfirm: () => {
                flip();
                toast("info", `${toList ? "#" : "@"}${tag.name} moved to ${bucket}`);
            },
        });
    });
    // 只有 List 才在选项间挑选，所以只有 List 需要模式行
    const modeRow = makeModeRow({
        getMode: () => tagMode(tag),
        setMode: (m) => { if (m === DEFAULT_MODE) delete tag.mode; else tag.mode = m; },
        key: () => listKey(tag.name),
        len: () => tagLines(tag.text).length,
        what: "list",
    });
    function paintKind() {
        const list = isListTag(tag);
        card.classList.toggle("islist", list);
        kindSw.paint(list, tagLines(tag.text).length);
        tx.placeholder = list ? "one option per line" : "what it expands to - the full prompt text";
        ins.title = list ? "插入 #" + tag.name + " 到提示词（每次 run 取一个选项）" : "插入 @" + tag.name + " 到提示词";
        del.title = `删除这个${list ? "列表" : "标签"}`;
        modeRow.el.style.display = list ? "flex" : "none";
        if (list) modeRow.paint();
    }
    paintKind();
    del.className = "sf-ptge-ic del";
    del.innerHTML = `<span class="sf-ptge-svg" style="-webkit-mask-image:url(${ICON_DELETE});mask-image:url(${ICON_DELETE})"></span>`;
    // 先问，并把标签自己的文本展示在问题里，好确认是不是你要删的那一个。背后没有撤销。
    del.addEventListener("click", () => {
        const list = isListTag(tag);
        const sym = list ? "#" : "@";
        const body = (tag.text || "").trim();
        confirmDanger({
            title: `Delete ${sym}${tag.name}?`,
            lead: `This deletes the ${list ? "list" : "tag"} <b>${esc(sym + tag.name)}</b>` +
                (body ? ` and what it holds:` : `, which is empty.`),
            listing: body ? (body.length > 400 ? body.slice(0, 400) + " …" : body) : "",
            confirmLabel: "Delete it",
            onConfirm: () => applyChange(() => {
                const i = _data.tags.indexOf(tag);
                if (i > -1) _data.tags.splice(i, 1);
                // 位置也一起丢，否则以后同名的新 List 会继承死牌堆的半副牌
                try { resetCursor(listKey(tag.name)); } catch { /* ignore */ }
            }),
        });
    });
    foot.append(ins, kindSw.el, del);
    card.append(top, tx, modeRow.el, foot);
    return card;
}

function renderSidebar(sideEl) {
    sideEl.innerHTML = "";
    // `menu`: null（All tags）| "cat"（真实分类）| "bucket"（Text/List 桶行）
    const mkCat = (label, color, count, key, menu) => {
        const bucket = menu === "bucket";
        const r = document.createElement("div");
        r.className = "sf-ptge-cat" + (_curCat === key ? " on" : "") + (bucket ? " bucket" : "");
        if (bucket) {
            r.title = `不是分类：这里是${key === LIST_BUCKET ? "列表" : "标签"}中无自有分类者的归属处。` +
                `一旦清空它自动消失。`;
        }
        if (menu === "cat") r.title = "拖拽调整它在列表中的上下顺序";
        r.innerHTML = (color ? `<span class="cd" style="background:${color}"></span>` : `<span style="width:11px"></span>`) +
            `<span class="nm">${esc(label)}</span>` +
            (menu ? `<span class="act more" title="${bucket ? "这一行是什么，以及你能对它做什么" : "移动、改名、导出或删除此分类"}">⋯</span>` : "") +
            `<span class="cnt">${count}</span>`;
        r.addEventListener("click", (e) => {
            if (e.target.classList.contains("more")) {
                e.stopPropagation();
                if (bucket) openBucketActions(key, e.target); else openCatActions(r, key, e.target);
                return;
            }
            if (_curCat !== key) {
                _curCat = key;
                // 侧栏选中分类 = 宣告你要做什么，创建表单必须重新跟随它。
                // kindTouched 曾跨 Create 残留：造完一个 List 标签后表单不再跟随侧栏，
                // 之后的每个标签都掉进桶里。
                _createDraft.kindTouched = false;
            }
            render();
        });
        // 右键行打开同一菜单——人们最先伸手的地方
        if (menu) {
            r.addEventListener("contextmenu", (e) => {
                // 不在改名输入框上（startRenameCat 在行内放了真 <input>，分类名是
                // 自由文本、人们会粘贴——吞掉浏览器菜单等于夺走唯一粘贴途径）
                if (e.target.closest("input, textarea")) return;
                e.preventDefault();
                e.stopPropagation();
                const anchor = r.querySelector(".more") || r;
                if (bucket) openBucketActions(key, anchor); else openCatActions(r, key, anchor);
            });
        }
        // ── 拖拽排序（仅真实分类）──
        // 桶无处存放，"All tags" 不是分类，二者都不能拖也不能接
        if (menu === "cat") {
            const mime = CAT_MIME(sideOf(key));
            const carries = (e) => !!e.dataTransfer && [...e.dataTransfer.types].includes(mime);
            const dropAbove = (e) => {
                const box = r.getBoundingClientRect();
                return (e.clientY - box.top) < box.height / 2;
            };
            const clearMarks = () => r.classList.remove("ins-above", "ins-below");
            r.draggable = true;
            r.addEventListener("dragstart", (e) => {
                // 改名中，或从 ⋯ 上抓起的，都不是排序手势
                if (e.target.closest("input, textarea, .act")) { e.preventDefault(); return; }
                if (r.querySelector("input, textarea")) { e.preventDefault(); return; }
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData(mime, key);
                e.dataTransfer.setData("text/plain", key);   // 部分浏览器没有 text/plain 会拒绝拖拽
                r.classList.add("dragging-me");
            });
            r.addEventListener("dragend", () => {
                r.classList.remove("dragging-me");
                // 拖到空白处松手不会触发任何行的 drop，清掉侧栏里残留的插入线
                for (const el of sideEl.querySelectorAll(".ins-above, .ins-below")) el.classList.remove("ins-above", "ins-below");
            });
            r.addEventListener("dragover", (e) => {
                // 不 preventDefault 正是浏览器显示"这里不能放"的方式。分类属于一侧，
                // 拖跨侧会把分类从它所有标签上清掉。
                if (!carries(e)) return;
                e.preventDefault();
                e.dataTransfer.dropEffect = "move";
                const above = dropAbove(e);
                r.classList.toggle("ins-above", above);
                r.classList.toggle("ins-below", !above);
            });
            r.addEventListener("dragleave", (e) => {
                // 行里有子 span（圆点/名字/⋯/计数），跨到它们上面会触发 dragleave
                // 而光标并未离开行，插入线会闪
                if (e.relatedTarget && r.contains(e.relatedTarget)) return;
                clearMarks();
            });
            r.addEventListener("drop", (e) => {
                if (!carries(e)) return;
                e.preventDefault();
                const above = dropAbove(e);
                clearMarks();
                const moved = e.dataTransfer.getData(mime);
                if (!moved || moved === key) return;
                const next = reorderCategoryTo(_data, moved, key, above);
                if (!next) return;   // 被拒绝或本来就在那里：不提交不重渲染
                applyChange(() => { _data.categories = next; });
            });
        }
        return r;
    };
    sideEl.appendChild(mkCat("All tags", "", _data.tags.length, "All", null));

    // 每侧一块。分类恰好属于其中一块，两块互不混杂；各带自己的 New category 按钮
    const block = (sd, heading) => {
        sideEl.appendChild(Object.assign(document.createElement("div"), { className: "lbl", textContent: heading }));
        if (bucketUsed(sd)) {
            const b = bucketOf(sd);
            sideEl.appendChild(mkCat(b, colorOf(b), tagsIn(b).length, b, "bucket"));
        }
        for (const c of catsOnSide(sd)) sideEl.appendChild(mkCat(c, colorOf(c), tagsIn(c).length, c, "cat"));
        const nc = document.createElement("div");
        nc.className = "sf-ptge-newcat";
        const btn = document.createElement("button");
        btn.className = "sf-ptge-btn";
        btn.innerHTML = `<span>＋</span> New category`;
        btn.title = sd === "list" ? "容纳列表的分类" : "容纳文本标签的分类";
        btn.addEventListener("click", () => {
            const inp = document.createElement("input");
            inp.placeholder = sd === "list" ? "list category name" : "category name";
            inp.style.cssText = "width:100%;margin-top:6px;background:#151515;border:1px solid var(--acc);border-radius:6px;color:#e6e6e6;font:12px monospace;padding:7px 9px;outline:none;";
            btn.style.display = "none";
            nc.appendChild(inp);
            inp.focus();
            // 放弃该字段绝不能触发全局 render()：那会拆掉用户正要操作的东西。
            // 只把按钮放回来，屏幕上其它东西不受影响。
            const cancel = () => { if (inp.isConnected) inp.remove(); btn.style.display = ""; };
            inp._sfCancel = cancel;   // 让 Escape 直接取消，不经过 blur 事件
            inp.addEventListener("keydown", (e) => {
                e.stopPropagation();
                if (e.key === "Enter") {
                    const v = inp.value.trim();
                    if (v && !isReservedName(v) && !_data.categories.some((c) => c.toLowerCase() === v.toLowerCase())) {
                        addCategory(v, sd);
                        _curCat = v;
                        commit();
                        // 落到刚建的分类 = 选中它，创建表单必须跟随它的侧
                        _createDraft.kindTouched = false;
                        render();   // 真实动作：新分类需要一行和一个选中态
                        return;
                    }
                    // 说出原因而不是默默关掉——重名/保留名曾经无声消失，读起来像坏按钮
                    if (v) {
                        toast("info", isReservedName(v)
                            ? `"${v}" is a built-in name, so it cannot be a category.`
                            : `You already have a category called "${v}".`);
                        inp.focus();
                        return;
                    }
                    cancel();
                    return;
                }
                if (e.key === "Escape") cancel();
            });
            inp.addEventListener("blur", () => setTimeout(cancel, 120));
        });
        nc.appendChild(btn);
        sideEl.appendChild(nc);
    };
    block("text", "Text categories");
    block("list", "List categories");
}

function startRenameCat(row, cat) {
    const nmSpan = row.querySelector(".nm");
    if (!nmSpan) return;   // 该行已在改名（标签已被换掉）
    const inp = document.createElement("input");
    inp.className = "catinput";
    inp.value = cat;
    // 可拖拽的祖先会劫持它自己文本框里的拖选：浏览器开始拖行、选中静默失败。
    // 字段存活期间关掉 draggable 是唯一不依赖浏览器把哪个元素报为 dragstart 的修法。
    // 每条退出路径（commit / cancel / 触发真实改名的 render()——render 会整行重建、
    // 自动恢复 draggable）都要恢复。
    const wasDraggable = row.draggable;
    row.draggable = false;
    const restoreDrag = () => { row.draggable = wasDraggable; };
    nmSpan.replaceWith(inp);
    inp.focus();
    inp.select();
    // 点进字段放光标/选字母不能冒泡到行的 click 处理器（会重渲染侧栏、销毁字段）
    inp.addEventListener("mousedown", (e) => e.stopPropagation());
    inp.addEventListener("click", (e) => e.stopPropagation());
    const commitRename = () => {
        const v = inp.value.trim();
        // 没变化：把标签放回原位而不是 render()。blur 上的整页 render 会毁掉你
        // mousedown 的目标，那次点击永远落不了地，侧栏/卡片按钮要点两下。
        // "没变化"是精确比较；占用名检查跳过正在改名的行。两者都不折叠大小写——
        // 重新大写（"styles" -> "Styles"）是真实改名，是人们在每行/每菜单/导出文件
        // 里看到的名字。
        if (!v || v === cat || isReservedName(v) ||
            _data.categories.some((c) => c !== cat && c.toLowerCase() === v.toLowerCase())) {
            if (inp.isConnected) inp.replaceWith(nmSpan);
            restoreDrag();     // 该行留在屏幕上，必须恢复可拖
            return;
        }
        const idx = _data.categories.indexOf(cat);
        if (idx > -1) _data.categories[idx] = v;
        const li = _data.listCats.indexOf(cat);   // 保持同一侧
        if (li > -1) _data.listCats[li] = v;
        if (_data.catModes && _data.catModes[cat]) {   // 以及它怎么选
            _data.catModes[v] = _data.catModes[cat];
            delete _data.catModes[cat];
        }
        // ……以及它进行到哪了。改名不是内容变化。
        try { renameCursor(catKey(cat), catKey(v)); } catch { /* ignore */ }
        for (const t of _data.tags) if (t.cat === cat) t.cat = v;
        if (_curCat === cat) _curCat = v;
        commit();
        render();   // 名字真变了，侧栏和头部必须跟随
    };
    const cancelRename = () => { if (inp.isConnected) inp.replaceWith(nmSpan); restoreDrag(); };
    // onKey 是 capture 阶段的 window 监听，字段自己的 keydown 永远看不见 Escape。
    // 暴露一个它可以直接调的取消句柄——没有它，onKey 的通用 `active.blur()` 会跑
    // 下面的 blur 监听器，把"放弃改名"变成"提交改名"，再无回头路。
    inp._sfCancel = cancelRename;
    inp.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") commitRename(); if (e.key === "Escape") cancelRename(); });
    inp.addEventListener("blur", commitRename);
}

// 在分类自己的块内上移/下移一步。非破坏（无可丢失、反向即复原），直接应用不提问
function moveCatStep(cat, dir) {
    const next = reorderCategoryStep(_data, cat, dir);
    if (!next) return;
    applyChange(() => { _data.categories = next; });
}

// 对分类能做的一切，集中在一个常驻屏幕的地方。两个删除刻意分行：
// "丢文件夹、留标签" 与 "连同标签一起删" 是完全不同的结果。
function openCatActions(row, cat, anchor) {
    hideCatMenu();
    const n = tagsIn(cat).length;
    const word = sideOf(cat) === "list" ? "list" : "tag";
    const many = (k) => `${k} ${word}${k === 1 ? "" : "s"}`;
    const menu = document.createElement("div");
    menu.className = "sf-ptge-menu";
    menu.style.minWidth = "250px";
    const add = (label, hint, cls, fn) => {
        const mi = document.createElement("div");
        mi.className = "mi mrow" + (cls ? " " + cls : "");
        mi.innerHTML = `<span class="nm">${esc(label)}</span>` + (hint ? `<span class="cnt">${esc(hint)}</span>` : "");
        if (fn) mi.addEventListener("click", () => { hideCatMenu(); fn(); });
        menu.appendChild(mi);
    };
    // 排序放最前：拖拽是更快的操作但界面上没人宣布它，这里是人们发现"顺序归你定"
    // 的地方。变暗状态与执行移动的是同一个函数——看起来可用的行永远不会变成
    // 点了没反应。
    const canUp = canMoveCategory(_data, cat, -1), canDn = canMoveCategory(_data, cat, 1);
    add("Move up", canUp ? "" : "already first", canUp ? "" : "dim", canUp ? () => moveCatStep(cat, -1) : null);
    add("Move down", canDn ? "" : "already last", canDn ? "" : "dim", canDn ? () => moveCatStep(cat, 1) : null);
    menu.appendChild(Object.assign(document.createElement("div"), { className: "msep" }));
    add("Rename", "", "", () => startRenameCat(row, cat));
    add("Export this category", n ? many(n) : "empty", "", () => exportScope(cat));
    menu.appendChild(Object.assign(document.createElement("div"), { className: "msep" }));
    if (n) {
        add("Delete category", `keeps the ${many(n)}`, "danger", () => confirmDeleteCat(cat));
        add(`Delete category and its ${n === 1 ? word : word + "s"}`, `${many(n)} deleted`, "danger", () => confirmDeleteCatWithTags(cat));
    } else {
        add("Delete category", "it is empty", "danger", () => confirmDeleteCat(cat));
    }
    _overlay.appendChild(menu);
    placeMenu(menu, anchor);
    _catMenu = menu;
}

// Text / List 桶行。不是分类：只在那一侧存在无分类标签时绘制，条件不成立就自行
// 消失。所以没有可改的名、没有可删的——菜单第一行就说明这一点。提供的是真正
// 让这行消失的两件事。
function openBucketActions(bucket, anchor) {
    hideCatMenu();
    const side = bucket === LIST_BUCKET ? "list" : "text";
    const n = tagsIn(bucket).length;
    const word = side === "list" ? "list" : "tag";
    const many = (k) => `${k} ${word}${k === 1 ? "" : "s"}`;
    const menu = document.createElement("div");
    menu.className = "sf-ptge-menu";
    menu.style.minWidth = "285px";
    menu.appendChild(Object.assign(document.createElement("div"), { className: "mhead", textContent: "This is not a category" }));
    menu.appendChild(Object.assign(document.createElement("div"), {
        className: "mnote",
        textContent: `It is where ${side === "list" ? "lists" : "tags"} with no category of their own are shown. ` +
            `Give them one and this row disappears on its own.`,
    }));
    menu.appendChild(Object.assign(document.createElement("div"), { className: "msep" }));
    const add = (label, hint, cls, fn) => {
        const mi = document.createElement("div");
        mi.className = "mi mrow" + (cls ? " " + cls : "");
        mi.innerHTML = `<span class="nm">${esc(label)}</span>` + (hint ? `<span class="cnt">${esc(hint)}</span>` : "");
        mi.addEventListener("click", () => { hideCatMenu(); fn(); });
        menu.appendChild(mi);
    };
    add(n === 1 ? "Put it in a category…" : "Put them all in a category…", many(n), "", () => {
        // openCategoryMenu 会先 hideCatMenu 再替换成选择器
        openCategoryMenu(anchor, (c) => {
            // "" 表示选择器落回桶本身，或输入的名字不可用。什么都不说会让按钮
            // 看起来是坏的。
            if (!c) { toast("info", `Pick a real category to move ${side === "list" ? "these lists" : "these tags"} into.`); return; }
            moveBucketTags(bucket, c);
        }, side);
    });
    const these = n === 1 ? `this ${word}` : `these ${word}s`;
    add(`Export ${these}`, many(n), "", () => exportScope(bucket));
    add(`Delete ${these}`, `${many(n)} deleted`, "danger", () => confirmDeleteBucket(bucket));
    _overlay.appendChild(menu);
    placeMenu(menu, anchor);
    _catMenu = menu;
}

// 把桶里每个标签归档进真实分类——让桶行消失的整洁方式。移动而非损失，不提问。
function moveBucketTags(bucket, cat) {
    const moving = tagsIn(bucket);
    if (!moving.length) return;
    const word = bucket === LIST_BUCKET ? "list" : "tag";
    applyChange(() => {
        for (const t of moving) t.cat = cat;
        // 桶被清空，它自己的 *wildcard 位置不再有意义。其它清空桶的路径都会丢它；
        // 这一条曾经把它留下来，让后来同尺寸的桶继承。
        try { resetCursor(catKey(bucket)); } catch { /* ignore */ }
        if (_data.catModes) delete _data.catModes[bucket];
        if (_curCat === bucket) {
            _curCat = cat;   // 跟着它们，旧行即将消失
            _createDraft.kindTouched = false;
        }
    });
}

function confirmDeleteBucket(bucket) {
    const shown = tagsIn(bucket);
    const n = shown.length;
    const word = bucket === LIST_BUCKET ? "list" : "tag";
    confirmDanger({
        title: `Delete the ${n} ${word}${n === 1 ? "" : "s"} with no category?`,
        lead: `<b>${esc(bucket)}</b> is not a category, so there is nothing there to delete on its own. ` +
            `This deletes the ${word}${n === 1 ? "" : "s"} sitting in it:`,
        listing: shown.slice(0, 40).map((t) => (isListTag(t) ? "#" : "@") + t.name).join(" · ") +
            (n > 40 ? ` … and ${n - 40} more` : ""),
        confirmLabel: `Delete ${n} ${word}${n === 1 ? "" : "s"}`,
        offerExport: true,
        exportCat: bucket,
        onConfirm: () => {
            // 确认时重解析，而不是用对话框构建时捕获的标签对象：库可能在对话框
            // 打开期间被改动，按身份删除会一个都删不到却报告成功
            const doomed = tagsIn(bucket);
            const k = doomed.length;
            if (!k) { toast("info", "Nothing left to delete there."); return; }
            applyChange(() => {
                const gone = new Set(doomed);
                _data.tags = _data.tags.filter((t) => !gone.has(t));
                for (const t of doomed) { try { resetCursor(listKey(t.name)); } catch { /* ignore */ } }
                // 桶是真实 *wildcard 目标，有自己的 Picks 模式与位置。其它删除都
                // 两者同弃；这一条曾把它们遗留，重建同尺寸桶会续死序列、继承旧模式
                try { resetCursor(catKey(bucket)); } catch { /* ignore */ }
                if (_data.catModes) delete _data.catModes[bucket];
                if (_curCat === bucket) _curCat = "All";
            });
        },
    });
}

// 分类"记录"本身（它在顺序里的位置、侧、怎么选、进行到哪）。两个删除共用，
// 永远同步。
function dropCategoryRecord(cat) {
    const idx = _data.categories.indexOf(cat);
    if (idx > -1) _data.categories.splice(idx, 1);
    const li = _data.listCats.indexOf(cat);
    if (li > -1) _data.listCats.splice(li, 1);
    if (_data.catModes) delete _data.catModes[cat];
    try { resetCursor(catKey(cat)); } catch { /* ignore */ }   // 不遗留它的位置
}

// 删分类，保留其标签（落入各自侧的桶）。和别的一样先确认：标签还在，但分类的
// 名字/侧/怎么选不在了，没有撤销，误点即终局。
function confirmDeleteCat(cat) {
    const n = tagsIn(cat).length;
    const word = sideOf(cat) === "list" ? "list" : "tag";
    confirmDanger({
        title: `Delete the category ${cat}?`,
        lead: n
            ? `The <b>${n} ${word}${n === 1 ? "" : "s"}</b> in it are kept - they move to ` +
                `<b>${esc(sideOf(cat) === "list" ? LIST_BUCKET : TEXT_BUCKET)}</b>, ready to file somewhere else. ` +
                `Only the category itself goes.`
            : `It is empty, so only the category itself goes.`,
        confirmLabel: "Delete the category",
        offerExport: true,   // 空分类也还有名字、侧和 Picks 模式
        exportCat: cat,
        onConfirm: () => deleteCat(cat),
    });
}
function deleteCat(cat) {
    applyChange(() => {
        dropCategoryRecord(cat);
        const landed = new Set();
        for (const t of _data.tags) {
            if (t.cat !== cat) continue;
            t.cat = "";                                        // -> 该标签自己的桶
            landed.add(isListTag(t) ? LIST_BUCKET : TEXT_BUCKET);
        }
        // 这些桶刚变了尺寸，它们的序列位置不再有意义
        for (const b of landed) { try { resetCursor(catKey(b)); } catch { /* ignore */ } }
        if (_curCat === cat) _curCat = "All";
    });
}

// 连分类带其中所有东西一起删。永远在 confirmDanger 后面。
function deleteCatWithTags(cat) {
    const doomed = tagsIn(cat);
    const n = doomed.length;
    const word = sideOf(cat) === "list" ? "list" : "tag";
    applyChange(() => {
        const gone = new Set(doomed);
        _data.tags = _data.tags.filter((t) => !gone.has(t));
        // 每个位置也一起走，否则以后同名标签继承半副牌（单标签删除同规则）
        for (const t of doomed) { try { resetCursor(listKey(t.name)); } catch { /* ignore */ } }
        dropCategoryRecord(cat);
        if (_curCat === cat) _curCat = "All";
    });
}
function confirmDeleteCatWithTags(cat) {
    const doomed = tagsIn(cat);
    const n = doomed.length;
    const word = sideOf(cat) === "list" ? "list" : "tag";
    confirmDanger({
        title: `Delete ${cat} and everything in it?`,
        lead: `This deletes the category <b>${esc(cat)}</b> and the <b>${n} ${word}${n === 1 ? "" : "s"}</b> filed under it:`,
        listing: doomed.slice(0, 40).map((t) => (isListTag(t) ? "#" : "@") + t.name).join(" · ") +
            (n > 40 ? ` … and ${n - 40} more` : ""),
        confirmLabel: `Delete ${n} ${word}${n === 1 ? "" : "s"}`,
        offerExport: true,
        exportCat: cat,
        onConfirm: () => deleteCatWithTags(cat),
    });
}

// 从零开始。藏在底部 ⋯ 后面防误触，对话框先给你导出再放行
function confirmDeleteEverything() {
    const n = _data.tags.length;
    const c = _data.categories.length;
    if (!n && !c) { toast("info", "Your library is already empty."); return; }
    confirmDanger({
        title: "Delete your whole tag library?",
        lead: `This removes <b>${n} tag${n === 1 ? "" : "s"}</b> and <b>${c} categor${c === 1 ? "y" : "ies"}</b>, ` +
            `leaving you with an empty library. Any @tag, #list or *category already typed into a Prompt node stops working.`,
        confirmLabel: "Delete everything",
        offerExport: true,
        exportCat: null,
        onConfirm: () => applyChange(() => {
            for (const t of _data.tags) { try { resetCursor(listKey(t.name)); } catch { /* ignore */ } }
            for (const cc of _data.categories) { try { resetCursor(catKey(cc)); } catch { /* ignore */ } }
            // 两个桶也是 *wildcard 目标，持有各自的位置，上面的循环（真实分类）会漏掉
            for (const b of [TEXT_BUCKET, LIST_BUCKET]) { try { resetCursor(catKey(b)); } catch { /* ignore */ } }
            _data.tags = [];
            _data.categories = [];
            _data.listCats = [];
            _data.catModes = {};
            _curCat = "All";
            // 搜索框也一起清，不只是标志位——顶栏不属于 render()，字段会继续显示
            // 一个已不再生效的过滤器
            _search = "";
            const s = _overlay && _overlay.querySelector(".sf-ptge-srch input");
            if (s) s.value = "";
        }),
    });
}

// 底部 ⋯——不是导出/导入的库级操作
function openLibraryMenu(anchor) {
    hideCatMenu();
    const menu = document.createElement("div");
    menu.className = "sf-ptge-menu";
    menu.style.minWidth = "230px";
    const n = _data.tags.length;
    const ri = document.createElement("div");
    ri.className = "mi mrow";
    ri.innerHTML = `<span class="nm">Restore default library…</span>`;
    ri.addEventListener("click", () => { hideCatMenu(); restoreDefaultLibrary(); });
    menu.appendChild(ri);
    const sep = document.createElement("div");
    sep.className = "msep";
    menu.appendChild(sep);
    const mi = document.createElement("div");
    mi.className = "mi mrow danger";
    mi.innerHTML = `<span class="nm">Delete everything…</span><span class="cnt">${n} tag${n === 1 ? "" : "s"}</span>`;
    mi.addEventListener("click", () => { hideCatMenu(); confirmDeleteEverything(); });
    menu.appendChild(mi);
    _overlay.appendChild(menu);
    // 底部按钮：下方没空间时向上开
    placeMenu(menu, anchor);
    _catMenu = menu;
}

// 恢复插件内置默认库（覆盖当前工作副本与存储）。与 Delete everything 同款
// 确认模式：先问、可先导出备份、无撤销；被替换的标签游标位置作废。
export function restoreDefaultLibrary() {
    // 编辑器未打开时静默返回（export 供冒烟直调，外部调用需自守）
    if (!_data) return Promise.resolve();
    const n = _data.tags.length;
    const c = _data.categories.length;
    return fetchDefaultLibrary().then((def) => {
        if (!def) { toast("error", "Could not load the built-in default library."); return; }
        // 与工作副本比较：两侧 normalize 后键序统一再比（clone 的键序与
        // normalize 产物不同，直接 stringify 会误判）。不用 isSameAsStored
        // （它比 store 缓存）——防抖 350ms 窗口内工作副本领先存储。
        if (JSON.stringify(normalizeLibrary(def)) === JSON.stringify(normalizeLibrary(_data))) {
            toast("info", "Your library is already the default."); return;
        }
        confirmDanger({
            title: "Restore the default library?",
            lead: `This replaces your <b>${n} tag${n === 1 ? "" : "s"}</b> and <b>${c} categor${c === 1 ? "y" : "ies"}</b> ` +
                `with the built-in default library (<b>${def.tags.length} tags</b> in ${def.categories.length} categories). ` +
                `Any @tag, #list or *category typed into a Prompt node that only exists in your library stops working.`,
            confirmLabel: "Restore defaults",
            offerExport: true,
            exportCat: null,
            onConfirm: () => {
                for (const t of _data.tags) { try { resetCursor(listKey(t.name)); } catch { /* ignore */ } }
                for (const cc of _data.categories) { try { resetCursor(catKey(cc)); } catch { /* ignore */ } }
                for (const b of [TEXT_BUCKET, LIST_BUCKET]) { try { resetCursor(catKey(b)); } catch { /* ignore */ } }
                _data = clone(def);
                _curCat = "All";
                _search = "";
                const s = _overlay && _overlay.querySelector(".sf-ptge-srch input");
                if (s) s.value = "";
                applyChange(() => {});
                toast("success", `Restored the default library (${def.tags.length} tags).`);
            },
        });
    });
}

// 顶部创建表单：名称 + 文本一处填完按 Create。新标签落在当前选中分类，或
// "All" 下该侧的 Text/List 桶。
function buildCreateForm() {
    // 侧栏显示哪侧决定你要造什么：打开 List 分类，表单就绪给列表。
    // "All tags" 没有侧，用户自己切过后（kindTouched）选择固定。
    const isRealCat = (c) => c !== "All" && c !== TEXT_BUCKET && c !== LIST_BUCKET;
    const sidebarSide = _curCat === "All" ? null : sideOf(_curCat);
    if (sidebarSide && !_createDraft.kindTouched) _createDraft.kind = sidebarSide;
    const sideNow = () => (_createDraft.kind === "list" ? "list" : "text");
    // 用户挑的分类赢过侧栏，且跨重渲染存活；装不下当前 kind 或已不存在时丢弃
    const picked = _createDraft.cat;
    const pickedUsable = picked != null &&
        (picked === "" || (_data.categories.some((c) => c === picked) && sideOf(picked) === sideNow()));
    let createCat = pickedUsable ? picked
        : (isRealCat(_curCat) && sideOf(_curCat) === sideNow() ? _curCat : "");
    if (!pickedUsable) _createDraft.cat = null;
    const form = document.createElement("div");
    form.className = "sf-ptge-create";
    const nm = document.createElement("input");
    nm.className = "cnm";
    nm.placeholder = "new tag name";
    nm.spellcheck = false;
    // <textarea> 而非 <input>：多行 "save selection as a tag" 保持换行
    const tx = document.createElement("textarea");
    tx.className = "ctx";
    tx.spellcheck = false;
    tx.rows = 1;
    // 待创建标签的 Text/List。住在 draft 上跨重渲染存活，且跟随文本（2+ 行 =
    // List）直到用户自己选——之后他们的选择固定。
    const kindSw = makeKindSwitch((toList) => {
        _createDraft.kind = toList ? "list" : "text";
        _createDraft.kindTouched = true;
        paintKind();
    });
    const paintKind = () => {
        const list = _createDraft.kind === "list";
        kindSw.paint(list, null);
        tx.placeholder = list ? "one option per line - press Enter for the next one" : "what it expands to - the full prompt text";
        // 列表需要空间打几行；文本标签保持单行
        tx.style.height = list ? "76px" : "36px";
        // 选中的分类只容纳一侧，kind 翻走时丢掉它
        if (createCat && sideOf(createCat) !== sideNow()) { createCat = ""; _createDraft.cat = null; }
        paintCat();
    };
    // 从进行中的 draft 播种，名称 + 文本跨重渲染存活（侧栏分类点击 / 搜索）
    nm.value = _createDraft.name;
    tx.value = _createDraft.text;
    nm.addEventListener("input", () => { _createDraft.name = nm.value; });
    tx.addEventListener("input", () => {
        _createDraft.text = tx.value;
        // 只在 "All tags" 下从文本猜测（其它地方侧已定，List 分类里打一行
        // 不应把标签扔回 Text）
        if (_createDraft.kindTouched || sidebarSide) return;
        const k = tagLines(tx.value).length > 1 ? "list" : "text";
        if (k !== _createDraft.kind) { _createDraft.kind = k; paintKind(); }
    });
    const catBtn = document.createElement("button");
    catBtn.className = "sf-ptge-pill ccat";
    catBtn.title = "新标签的分类——点击更改";
    const paintCat = () => {
        const label = createCat || bucketOf(sideNow());
        catBtn.innerHTML = `<span class="cd" style="background:${colorOf(label)}"></span><span>${esc(label)}</span><span class="car">▾</span>`;
    };
    paintKind();   // 只有此时 paintCat 已存在（paintKind 调用它），首次绘制才能跑
    catBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        openCategoryMenu(catBtn, (c) => { createCat = c; _createDraft.cat = c; paintCat(); }, sideNow());
    });
    const btn = document.createElement("button");
    btn.className = "cbtn";
    btn.textContent = "Create tag";
    btn.title = "把此标签加入库（Ctrl+Enter）";
    const doCreate = () => {
        const name = sanitizeName(nm.value);
        // 输入 "!!!" 剥成空：默默拒绝读起来像坏按钮
        if (!name) {
            toast("info", nm.value.trim()
                ? "A tag name can only use letters, numbers, - and _."
                : "Give the tag a name first.");
            nm.focus();
            return;
        }
        const uniq = uniqueNameExcept(name, null);
        const isList = _createDraft.kind === "list";
        const kindAtCreate = _createDraft.kind;
        const kindTouchedAtCreate = _createDraft.kindTouched;
        const rec = { name: uniq, cat: createCat, text: tx.value };
        if (isList) rec.kind = "list";   // 只在 List 时写（库 normalize 同规则）
        _data.tags.unshift(rec);
        _createDraft = newDraft();       // 标签已存 -> 下次渲染的表单是空的
        // ……但保留用户的 Text/List 选择。重置 kindTouched 会让表单重新从侧栏
        // 推导，造完一个单行 List 后下一个就悄悄变成 Text。
        if (kindTouchedAtCreate) { _createDraft.kind = kindAtCreate; _createDraft.kindTouched = true; }
        commit();
        render();
        const nf = _overlay && _overlay.querySelector(".sf-ptge-create .cnm");
        if (nf) nf.focus();
        toast("success", "Created tag " + (isList ? "#" : "@") + uniq);
    };
    btn.addEventListener("click", doCreate);
    nm.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") { e.preventDefault(); doCreate(); } });
    // List 模式下 Enter 必须开始下一个选项（打列表就是全部意义），所以只有
    // Ctrl/Cmd+Enter 创建。Text 模式 Enter 创建、Shift+Enter 换行。
    tx.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key !== "Enter") return;
        if (e.ctrlKey || e.metaKey) { e.preventDefault(); doCreate(); return; }
        if (_createDraft.kind === "list" || e.shiftKey) return;  // 放换行通过
        e.preventDefault();
        doCreate();
    });
    form.append(nm, tx, catBtn, kindSw.el, btn);
    return form;
}

function buildGrid() {
    const grid = document.createElement("div");
    grid.className = "sf-ptge-grid";
    const q = _search.toLowerCase();
    const rows = _data.tags.filter((t) =>
        (_curCat === "All" || catOf(t) === _curCat) &&
        (!q || pinyinMatch(t.name, q) || t.text.toLowerCase().includes(q)));
    if (!rows.length) {
        const e = document.createElement("div");
        e.className = "sf-ptge-empty";
        e.style.gridColumn = "1 / -1";
        e.textContent = _search ? "No tags match your search." : "No tags here yet - create one above.";
        grid.appendChild(e);
    } else for (const t of rows) grid.appendChild(makeCard(t));
    return grid;
}

function renderContent(content) {
    content.innerHTML = "";
    const head = document.createElement("div");
    head.className = "sf-ptge-chead";
    const h = document.createElement("div");
    h.className = "h";
    if (_curCat === "All") h.innerHTML = `<span>All tags</span><span class="c">· ${_data.tags.length}</span>`;
    else {
        const n = tagsIn(_curCat).length;
        const word = sideOf(_curCat) === "list" ? "list" : "tag";
        h.innerHTML = `<span class="cd" style="background:${colorOf(_curCat)}"></span><span>${esc(_curCat)}</span>` +
            `<span class="c">· ${n} ${word}${n === 1 ? "" : "s"}</span>`;
    }
    head.append(h);
    // *这个分类 怎么选它的一个标签。"All tags" 下不显示（没有 *All 可配置）
    if (_curCat !== "All") {
        const cat = _curCat;
        head.appendChild(makeModeRow({
            getMode: () => catMode(cat, _data),
            setMode: (m) => {
                _data.catModes = _data.catModes || {};
                if (m === DEFAULT_MODE) delete _data.catModes[cat]; else _data.catModes[cat] = m;
            },
            key: () => catKey(cat),
            len: () => tagsIn(cat).length,
            what: "category",
        }).el);
    }
    content.append(head, buildCreateForm(), buildGrid());
}

function render() {
    if (!_overlay) return;
    hideCatMenu();
    // 桶行只在有标签坐镇时绘制。归档/删除最后一个，选中就指向一个不再绘制的行：
    // 无高亮、头部读作 "Text · 0 tags" 却带着一个空桶的活 Picks 控件、创建表单
    // 仍被锁在该侧。
    if ((_curCat === TEXT_BUCKET || _curCat === LIST_BUCKET) &&
        !bucketUsed(_curCat === LIST_BUCKET ? "list" : "text")) {
        _curCat = "All";
    }
    renderSidebar(_overlay.querySelector(".sf-ptge-side"));
    renderContent(_overlay.querySelector(".sf-ptge-content"));
}

// ── import / export ────────────────────────────────────────────────────
// 把库（或它的一个分类）写入文件。`cat` null = 全部
function exportScope(cat) {
    try {
        const count = cat == null ? _data.tags.length : tagsIn(cat).length;
        const blob = new Blob([exportLibraryJSON(_data, cat)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = cat == null ? "prompt-tags.json" : `prompt-tags-${String(cat).replace(/[^\p{L}\p{N}_\-]+/gu, "-")}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        // 名词跟随侧：一行刚说 "3 lists"，紧接着说 "3 tags" 会让一个动作用两个词
        const word = cat == null ? "tag" : (sideOf(cat) === "list" ? "list" : "tag");
        const what = count ? `${count} ${word}${count === 1 ? "" : "s"}` : "an empty category";
        toast("info", cat == null ? `Exported ${what}.` : `Exported ${what} from ${cat}.`);
    } catch (err) {
        console.error("sfnodes.PromptTags export failed", err);
        toast("warn", "Could not write that file");
    }
}

// 全部 / 单个分类，各一下（复用深色菜单，Escape / 外部点击关闭）
function openExportMenu(anchor) {
    hideCatMenu();
    const menu = document.createElement("div");
    menu.className = "sf-ptge-menu";
    const add = (label, color, count, cat) => {
        const word = cat == null ? "tag" : (sideOf(cat) === "list" ? "list" : "tag");
        const mi = document.createElement("div");
        // 只有完全空的库上 "Everything" 才是真没什么可导出
        const nothing = count === 0 && cat == null;
        mi.className = "mi mrow" + (nothing ? " dim" : "");
        mi.innerHTML = (color ? `<span class="cd" style="background:${color}"></span>` : `<span style="width:10px"></span>`) +
            `<span class="nm">${esc(label)}</span><span class="cnt">${count ? `${count} ${word}${count === 1 ? "" : "s"}` : "empty"}</span>`;
        if (!nothing) mi.addEventListener("click", () => { hideCatMenu(); exportScope(cat); });
        menu.appendChild(mi);
    };
    add("Everything", "", _data.tags.length, null);
    // 与侧栏相同的两块，菜单读起来和库长得一样
    const block = (sd, heading) => {
        const names = [...(bucketUsed(sd) ? [bucketOf(sd)] : []), ...catsOnSide(sd)];
        if (!names.length) return;
        menu.appendChild(Object.assign(document.createElement("div"), { className: "msep" }));
        menu.appendChild(Object.assign(document.createElement("div"), { className: "mhead", textContent: heading }));
        for (const c of names) add(c, colorOf(c), tagsIn(c).length, c);
    };
    block("text", "Text categories");
    block("list", "List categories");
    _overlay.appendChild(menu);
    // 按钮在底部栏，下方空间不足时向上开
    placeMenu(menu, anchor);
    _catMenu = menu;
}

function pickImportFile() {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = ".json,application/json";
    inp.style.display = "none";
    inp.addEventListener("change", () => {
        const file = inp.files && inp.files[0];
        inp.remove();
        if (!file) return;
        // 标签库是文本，真正的库只有几 KB。直接把超大文件读进 JSON.parse 会在
        // 我们任何检查跑起来之前就把标签页内存耗尽。
        if (file.size > MAX_IMPORT_BYTES) {
            toast("warn", "That file is too big to be a tag library (over 8 MB).");
            return;
        }
        const reader = new FileReader();
        reader.onload = () => startImport(String(reader.result || ""));
        reader.onerror = () => toast("warn", "Could not read that file");
        reader.readAsText(file);
    });
    // 取消 OS 文件对话框不会触发 "change"，没有这个，隐藏输入框会伴随页面存活
    // 一个——每次被取消的导入一个
    inp.addEventListener("cancel", () => inp.remove());
    document.body.appendChild(inp);
    inp.click();
}

function startImport(text) {
    // 文件输入在 document.body 上、FileReader 是异步的，选完文件到读完成之间
    // 编辑器可能已被关闭（Escape / Done / 节点被删）。以下全都需要 _overlay 和 _data。
    if (!_overlay || !_data) return;
    flushLibrary(); // 先落盘任何未写入的编辑，再开始并入
    const parsed = parseImport(text, getLibrary());
    if (parsed.error) { toast("warn", parsed.error); return; }
    showImportPick(parsed);
}

// 导入第一步：按分类展示文件内容，只带想带进来的。总是显示（导入少见，先看内容
// 正是意义）；其后的冲突步只在所选标签真的冲突时出现。
function showImportPick(parsed) {
    if (!_overlay || !_data) return;
    const cats = importCategories(parsed);
    const total = parsed.data.tags.length;
    const modal = document.createElement("div");
    modal.className = "sf-ptge-modal";
    modal.innerHTML =
        `<div class="sf-ptge-mcard"><div class="mh">Import tags</div>` +
        `<div class="mb">This file has <b>${total} tag${total === 1 ? "" : "s"}</b> in ` +
        `<b>${cats.length} categor${cats.length === 1 ? "y" : "ies"}</b>. Tick what you want to bring in.</div>` +
        // 无法使用的名字在到达这里之前就被丢弃，上面的计数是丢之后的——直说，
        // 别让文件看起来比实际小
        (parsed.dropped
            ? `<div class="mb" style="padding-top:0"><div class="conf">${parsed.dropped} more ` +
                `tag${parsed.dropped === 1 ? "" : "s"} cannot be brought in: a tag name can only ` +
                `contain letters, numbers, Chinese characters, - and _.</div></div>`
            : "") +
        `<div class="sf-ptge-pick"></div>` +
        `<div class="sf-ptge-mfoot">` +
        `<button class="sf-ptge-mlink pk-all">All</button>` +
        `<button class="sf-ptge-mlink pk-none">None</button>` +
        `<button class="sf-ptge-btn push pk-cancel">Cancel</button>` +
        `<button class="sf-ptge-btn pri pk-go">Import</button>` +
        `</div></div>`;
    const pick = modal.querySelector(".sf-ptge-pick");
    for (const c of cats) {
        // <label> 行：点行内任意处原生切换勾选框（无 JS 切换，不会在点框本身时双触发）
        const row = document.createElement("label");
        row.className = "row";
        row.dataset.cat = c.name;
        row.innerHTML = `<input type="checkbox" checked><span class="cd" style="background:${colorOf(c.name)}"></span>` +
            `<span class="nm">${esc(c.name)}</span><span class="cnt">${c.count} tag${c.count === 1 ? "" : "s"}</span>`;
        pick.appendChild(row);
    }
    const boxes = () => [...pick.querySelectorAll(".row")];
    modal.querySelector(".pk-all").addEventListener("click", () => boxes().forEach((r) => { r.querySelector("input").checked = true; }));
    modal.querySelector(".pk-none").addEventListener("click", () => boxes().forEach((r) => { r.querySelector("input").checked = false; }));
    modal.querySelector(".pk-cancel").addEventListener("click", () => modal.remove());
    modal.querySelector(".pk-go").addEventListener("click", () => {
        const names = boxes().filter((r) => r.querySelector("input").checked).map((r) => r.dataset.cat);
        const sub = subsetImport(parsed, names, getLibrary());
        // 勾选一个空分类是真实选择：importCategories 会提供它们让备份得以恢复，
        // 仅按标签判定会拒绝它们且消息也不属实（"没选任何东西"而明明勾了）
        if (!sub.data.tags.length && !sub.data.categories.length) {
            toast("info", "Nothing selected to import.");
            return;
        }
        modal.remove();
        if (!sub.conflicts.length) { applyLibraryImport(sub, "both"); return; }
        showImportModal(sub);
    });
    modal.addEventListener("mousedown", (e) => { if (e.target === modal) modal.remove(); });
    _overlay.appendChild(modal);
}

function applyLibraryImport(parsed, mode) {
    if (!_overlay) return;
    const before = { categories: [..._data.categories] };
    const res = applyImport(parsed, mode);
    _data = clone(getLibrary());
    render();
    const bits = [];
    if (res.added) bits.push(`${res.added} tag${res.added === 1 ? "" : "s"} added`);
    if (res.replaced) bits.push(`${res.replaced} replaced`);
    // applyImport 独立于标签数合并分类/侧/Picks 模式，所以标签全被当作重复跳过时
    // 仍可能加了分类。报告 "Nothing was imported" 是假话。
    const hadCat = new Set(before.categories.map((c) => c.toLowerCase()));
    const catsAdded = _data.categories.filter((c) => !hadCat.has(c.toLowerCase())).length;
    if (catsAdded) bits.push(`${catsAdded} categor${catsAdded === 1 ? "y" : "ies"} added`);
    toast("info", bits.length ? "Imported: " + bits.join(", ") + "." : "Nothing was imported.");
}

function showImportModal(parsed) {
    if (!_overlay) return;
    const modal = document.createElement("div");
    modal.className = "sf-ptge-modal";
    const total = parsed.data.tags.length;
    // 与其它两个清单一致：列表用 #（不总是 @），被截断时说明而不是戛然而止
    const symOf = (n) => {
        const t = parsed.data.tags.find((x) => x.name === n);
        return t && t.kind === "list" ? "#" : "@";
    };
    const conf = parsed.conflicts.slice(0, 40).map((n) => symOf(n) + n).join(" · ") +
        (parsed.conflicts.length > 40 ? ` … and ${parsed.conflicts.length - 40} more` : "");
    modal.innerHTML =
        `<div class="sf-ptge-mcard"><div class="mh">Import tags</div>` +
        `<div class="mb">Importing <b>${total} tag${total === 1 ? "" : "s"}</b>. ` +
        (parsed.conflicts.length === 1
            ? `<b>1</b> has a name you already use:`
            : `<b>${parsed.conflicts.length}</b> have names you already use:`) +
        `<div class="conf">${esc(conf)}</div>How should ${parsed.conflicts.length === 1 ? "it" : "the clashes"} be handled?</div>` +
        `<div class="sf-ptge-opts">` +
        `<div class="sf-ptge-opt rec" data-mode="both"><span class="oic">＋</span><span class="t">Keep both<small>Renames the imported one (e.g. @${esc(parsed.conflicts[0])}-2) so nothing is lost</small></span><span class="rtag">recommended</span></div>` +
        `<div class="sf-ptge-opt" data-mode="replace"><span class="oic">⟳</span><span class="t">Replace mine<small>Overwrite my tag's text with the imported one</small></span></div>` +
        `<div class="sf-ptge-opt" data-mode="skip"><span class="oic">⊘</span><span class="t">Skip duplicates<small>Only add the tags I don't already have</small></span></div>` +
        `</div></div>`;
    modal.addEventListener("mousedown", (e) => { if (e.target === modal) modal.remove(); });
    modal.querySelectorAll(".sf-ptge-opt").forEach((o) => o.addEventListener("click", () => {
        const m = o.dataset.mode;
        if (m !== "replace") { modal.remove(); applyLibraryImport(parsed, m); return; }
        // "Replace mine" 一键覆盖用户写过的文本。没有撤销，它曾是唯一不提问的
        // 损失路径。问——并让选项弹窗保持打开，Cancel 落回选择而非虚无。
        const n = parsed.conflicts.length;
        confirmDanger({
            title: `Overwrite ${n === 1 ? "1 tag" : n + " tags"} of yours?`,
            lead: `The imported text replaces your own on <b>${n}</b> tag${n === 1 ? "" : "s"}. What you have there now goes away.`,
            listing: parsed.conflicts.slice(0, 40).join(" · ") + (n > 40 ? ` … and ${n - 40} more` : ""),
            confirmLabel: "Replace mine",
            offerExport: true,
            exportCat: null,
            onConfirm: () => { modal.remove(); applyLibraryImport(parsed, "replace"); },
        });
    }));
    _overlay.appendChild(modal);
}

// 真正的 "你确定吗？"，只用于一下点击会带走不止一件事的地方。没有撤销
// （见 applyChange），所以每条可能丢东西的路径都用它。
// `lead` 是 HTML（以便加粗计数）；传入前先转义任何用户值。
function confirmDanger({ title, lead, listing, confirmLabel, offerExport, exportCat, onConfirm }) {
    if (!_overlay) return;
    const modal = document.createElement("div");
    modal.className = "sf-ptge-modal";
    modal.innerHTML =
        `<div class="sf-ptge-mcard"><div class="mh">${esc(title)}</div>` +
        `<div class="mb">${lead}` +
        (listing ? `<div class="conf">${esc(listing)}</div>` : "") +
        // 自己的块：没有 listing 时它会直接贴着 lead 结尾、没有间距
        `<div style="margin-top:10px">This cannot be undone.</div></div>` +
        `<div class="sf-ptge-mfoot">` +
        (offerExport ? `<button class="sf-ptge-btn dg-exp" type="button">⭳ Export a backup first</button>` : "") +
        `<button class="sf-ptge-btn push dg-cancel" type="button">Cancel</button>` +
        `<button class="sf-ptge-btn danger dg-go" type="button">${esc(confirmLabel)}</button>` +
        `</div></div>`;
    // 导出不得关闭对话框：意义在于先存文件、然后仍然坐在决定面前
    modal.querySelector(".dg-exp")?.addEventListener("click", () => exportScope(exportCat == null ? null : exportCat));
    modal.querySelector(".dg-cancel").addEventListener("click", () => modal.remove());
    modal.querySelector(".dg-go").addEventListener("click", () => { modal.remove(); onConfirm(); });
    modal.addEventListener("mousedown", (e) => { if (e.target === modal) modal.remove(); });
    _overlay.appendChild(modal);
}

function toast(sev, msg) {
    const t = app?.extensionManager?.toast;
    if (t?.add) t.add({ severity: sev, summary: "SF Prompt Tags", detail: msg, life: 2600 });
    else console.warn("[sfnodes.PromptTags]", msg);
}

// 自包含的帮助面板，挂在 overlay 上（复用 modal 外观，X / 点击外部 / Escape 关闭）
function showLibraryHelp() {
    if (!_overlay) return;
    const modal = document.createElement("div");
    modal.className = "sf-ptge-modal";
    modal.innerHTML =
        `<div class="sf-ptge-mcard sf-ptge-help-card"><div class="mh">标签库如何工作</div>` +
        `<div class="mb">` +
        `<p><b>它是什么。</b> 你的个人、可复用的提示词片段。在 Prompt 节点里输入简短的 <b>@名称</b>，运行时它会变成完整文本，所以输入框保持简短。你的库保存在本机、只属于你，且随插件更新存活——它永远不会被存进工作流。</p>` +
        `<p><b>创建标签。</b> 在顶部填写名称与完整提示词文本，选一个分类，按 <b>Create tag</b>。新标签出现在最前面。</p>` +
        `<p><b>编辑标签。</b> 点击卡片的名称或文本直接修改——编辑自动保存。</p>` +
        `<p><b>Text 或 List。</b> 每张卡片底部都有包含两种选择的开关。<b>Text</b> 是一段文本，<b>@名称</b> 会整体插入。<b>List</b> 每行一个选项（猫、狗、鼠），<b>#名称</b> 每次 run 随机插入一个。随时可以翻转开关：它改变卡片的作用，绝不改变你已保存的提示词。顶部创建框设为 List 时，Enter 开始下一个选项、Ctrl+Enter 添加标签。</p>` +
        `<p><b>分类。</b> 在左侧侧栏创建。点击卡片的彩色 pill 把该标签移到另一个分类。分类行上的 <b>⋯</b>（右键该行同样）可以改名、只导出该分类、或删除它。在提示词里输入 <b>*分类</b> 每次 run 从中随机选一个标签。</p>` +
        `<p><b>给分类排你自己的顺序。</b> 上下拖拽分类行来移动它，或在其 <b>⋯</b> 菜单里用 <b>Move up</b> 与 <b>Move down</b>。你设置的顺序在处处一致：侧栏、导出菜单、卡片 pill、以及输入 <b>@</b>、<b>#</b> 或 <b>*</b> 时弹出的列表。Text 与 List 分类是两个独立分组，行只在自己组内移动。也可以拖拽侧栏与卡片之间的分隔条来加宽分类列表，下次打开保持不变（双击分隔条复位）。</p>` +
        `<p><b>斜体的 Text 与 List 行不是分类。</b> 它们是展示无自有分类标签的地方，所以行本身没有可改的名、没有可删的。它们的 <b>⋯</b> 可以把这些标签一次性归档进某个分类（行随即自行消失）、导出它们、或删除它们。</p>` +
        `<p><b>删除。</b> 任何移除操作都会先问，并精确展示将被删掉的内容，好让你确认是你要删的那个。删除分类有两个选择：保留其标签（它们移到 Text 或 List）或连同删除。<b>Export</b> 与 <b>Import</b> 旁边的 <b>⋯</b> 里有 <b>Delete everything</b> 用于重新开始。删除整组时，对话框还会先提供保存备份文件。没有撤销，所以一旦确认就是最终结果。</p>` +
        `<p><b>Picks：Shuffle、Random 或 In order。</b> List 卡片与任何可用 <b>*名称</b> 掷取的内容的头部，都有一个 <b>Picks</b> 控件决定如何选取。<b>Shuffle</b> 是默认：像洗牌发牌，每个选项出现一次后才重复。<b>Random</b> 每次任取一个，所以同一个可能连续出现两次。<b>In order</b> 按 1、2、3 循环。Shuffle 与 In order 会记住它们在 run 之间的位置（卡片会显示），<b>↺</b> 按钮让该列表重新开始。</p>` +
        `<p><b>使用标签。</b> 在提示词框里输入 <b>@</b>（列表用 <b>#</b>，分类用 <b>*</b>）会弹出可搜索列表，或按卡片上的 <b>Insert</b> 直接插入提示词。</p>` +
        `<p><b>分享。</b> <b>Export</b> 把你的标签保存到文件：全部或单个分类。<b>Import</b> 展示文件内容，让你勾选要导入的分类；若名字已存在，可选择保留两个、替换、或跳过。</p>` +
        `</div>` +
        `<div class="sf-ptge-help-foot"><button class="sf-ptge-btn pri hgot">知道了</button></div>` +
        `</div>`;
    modal.addEventListener("mousedown", (e) => { if (e.target === modal) modal.remove(); });
    modal.querySelector(".hgot").addEventListener("click", () => modal.remove());
    _overlay.appendChild(modal);
}

// ── open / close ───────────────────────────────────────────────────────
export function openLibraryEditor(node, opts) {
    closeLibraryEditor();
    injectCSS();
    _node = node;
    _opts = opts || {};
    _accent = _opts.accent || BRAND;
    _createDraft = newDraft((_opts.prefill || "").trim());
    // 从存储重读，绝不使用内存缓存：另一个标签页/窗口可能在本页加载后改过库，
    // 而关闭路径会把这份工作副本整体写回
    _data = clone(reloadLibrary());
    _curCat = "All";
    _search = "";

    const ov = document.createElement("div");
    ov.className = "sf-ptge";
    ov.style.setProperty("--acc", _accent);
    ov.innerHTML =
        `<div class="sf-ptge-bar">` +
        `<div class="ttl"><span class="cr">☲</span> Tag library</div>` +
        `<div class="sf-ptge-srch"><span class="i">🔍</span><input placeholder="search tags and text"></div>` +
        `<span class="priv">private to you · survives plugin updates</span>` +
        `<span class="help" title="标签库如何工作"><span class="sf-ptge-svg" style="-webkit-mask-image:url(${ICON_HELP});mask-image:url(${ICON_HELP})"></span></span>` +
        `<span class="x" title="关闭">✕</span></div>` +
        `<div class="sf-ptge-main"><div class="sf-ptge-side"></div>` +
        `<div class="sf-ptge-grip" title="拖拽调整分类列表宽度。双击复位。"></div>` +
        `<div class="sf-ptge-content"></div></div>` +
        `<div class="sf-ptge-foot"><button class="sf-ptge-btn imp-export" title="把你的标签保存到文件：全部或单个分类"><span>⭳</span> Export ▾</button>` +
        `<button class="sf-ptge-btn imp-import" title="从文件导入标签——你可以选择要导入的分类"><span>⭱</span> Import</button>` +
        `<button class="sf-ptge-btn imp-more" title="更多库操作">⋯</button>` +
        `<button class="sf-ptge-btn push imp-done">Done</button></div>`;
    document.body.appendChild(ov);
    _overlay = ov;

    const search = ov.querySelector(".sf-ptge-srch input");
    search.addEventListener("input", () => { _search = search.value; renderContent(ov.querySelector(".sf-ptge-content")); });
    search.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Escape" && _search) {
            _search = "";
            search.value = "";
            renderContent(ov.querySelector(".sf-ptge-content"));
            e.stopImmediatePropagation();
        }
    });
    ov.querySelector(".x").addEventListener("click", closeLibraryEditor);
    ov.querySelector(".help").addEventListener("click", showLibraryHelp);
    ov.querySelector(".imp-done").addEventListener("click", closeLibraryEditor);
    ov.querySelector(".imp-export").addEventListener("click", (e) => openExportMenu(e.currentTarget));
    ov.querySelector(".imp-import").addEventListener("click", pickImportFile);
    ov.querySelector(".imp-more").addEventListener("click", (e) => openLibraryMenu(e.currentTarget));
    installSidebarResize(ov);
    // 被拖的分类行同时带 text/plain（让所有浏览器都能开拖）——这也让编辑器里
    // 每个文本框都成了原生 drop 目标：把行放到标签卡片的文本框上会把分类名拼进
    // 片段，而卡片自己的 input 处理器当场提交、没有撤销。行上没有拖拽把手，拖过
    // 到卡片网格是常见的学习手势，网格上也没有插入线提醒。取消任何携带我们行
    // 类型、但没落在分类行上的 drop（capture 阶段，压过字段自己的默认）。
    // 从别处拖来的普通文本不带这两种类型，不受影响，拖进标签框仍可用。
    ov.addEventListener("drop", (e) => {
        if (!e.dataTransfer) return;
        const t = [...e.dataTransfer.types];
        if (!t.includes(CAT_MIME("text")) && !t.includes(CAT_MIME("list"))) return;
        if (e.target.closest && e.target.closest(".sf-ptge-cat")) return;   // 真正的排序目标
        e.preventDefault();
        e.stopPropagation();
    }, true);

    render();
    // 来自 "save selection as a tag"：文本已进创建表单，聚焦名称字段——
    // 用户只需命名并按 Create
    if ((_opts.prefill || "").trim()) {
        const nf = ov.querySelector(".sf-ptge-create .cnm");
        if (nf) { nf.focus(); }
    } else {
        search.focus();
    }

    _undoGuardOff = installGraphUndoGuard(() => !!_overlay && _overlay.isConnected);
    window.addEventListener("keydown", onKey, true);
}

// ── 侧栏宽度 ───────────────────────────────────────────────────────────
function readSidebarWidth() {
    try {
        const v = app.ui?.settings?.getSettingValue(SIDE_W_SETTING);
        return v == null ? SIDE_W_DEFAULT : clampSideW(v);
    } catch { return SIDE_W_DEFAULT; }
}
function writeSidebarWidth(px) {
    try {
        const s = app.ui?.settings, w = clampSideW(px);
        if (typeof s?.setSettingValueAsync === "function") s.setSettingValueAsync(SIDE_W_SETTING, w);
        else if (typeof s?.setSettingValue === "function") s.setSettingValue(SIDE_W_SETTING, w);
    } catch { /* ignore */ }
}
function installSidebarResize(overlay) {
    const side = overlay.querySelector(".sf-ptge-side");
    const grip = overlay.querySelector(".sf-ptge-grip");
    if (!side || !grip) return;
    side.style.width = readSidebarWidth() + "px";
    let pid = null;
    let startX = 0;
    let startW = 0;
    const move = (ev) => {
        // 两道防线都要：真鼠标可能丢 release（指针出视口，或别的抢了 capture），
        // 接缝会永远跟着光标。合成事件复现不了，这层防护不能"测试掉"。
        if (!(ev.buttons & 1)) { end(); return; }
        side.style.width = clampSideW(startW + (ev.clientX - startX)) + "px";
    };
    const end = () => {
        if (pid === null) return;                 // 幂等：上面的防线也会调用它
        try { grip.releasePointerCapture(pid); } catch { /* already released */ }
        pid = null;
        grip.classList.remove("on");
        overlay.classList.remove("resizing");
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", end);
        window.removeEventListener("pointercancel", end);
        writeSidebarWidth(parseFloat(side.style.width));
    };
    grip.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        pid = e.pointerId;
        startX = e.clientX;
        startW = side.getBoundingClientRect().width;
        try { grip.setPointerCapture(pid); } catch { /* window listeners still cover it */ }
        grip.classList.add("on");
        overlay.classList.add("resizing");
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
        e.preventDefault();
    });
    grip.addEventListener("lostpointercapture", end);
    // 拖过头的一条回头路
    grip.addEventListener("dblclick", () => {
        side.style.width = SIDE_W_DEFAULT + "px";
        writeSidebarWidth(SIDE_W_DEFAULT);
    });
}

function onKey(e) {
    if (e.key !== "Escape") return;
    // 关闭最上层 modal（DOM 顺序最后一个），而不是第一个：Replace-mine 确认
    // 叠在仍打开的导入选项弹窗上，删掉第一个匹配会静默删掉被盖住的那个——
    // Escape 看起来什么都没做，然后 Cancel 落在只剩选择弹窗的编辑器上。
    const _modals = _overlay ? _overlay.querySelectorAll(".sf-ptge-modal") : [];
    if (_modals.length) { _modals[_modals.length - 1].remove(); e.stopPropagation(); return; }
    if (_catMenu) { hideCatMenu(); e.stopPropagation(); return; }
    // 这是 capture 阶段的 window 监听，压过每个字段自己的 keydown（那是冒泡阶段）。
    // Escape 因此曾经从文本框里直接关闭整个编辑器——那从来不是输入框里 Escape
    // 的意思：它扔掉了打了一半的标签（含 "Save selection as a tag" 移交的文本），
    // 而新建分类/改名输入框自己的 Escape 处理永远轮不到。先放掉字段；第二次
    // Escape 才照常关闭编辑器。
    const active = document.activeElement;
    if (active && _overlay && _overlay.contains(active)) {
        if (active.classList.contains("catinput")) {   // 改名分类：取消它
            // 调字段自己的取消。blur() 会跑 blur 监听器并提交——与 Escape 的含义
            // 正好相反。
            if (typeof active._sfCancel === "function") active._sfCancel(); else active.blur();
            e.stopPropagation();
            return;
        }
        // 搜索框：第一次 Escape 清过滤器，下一次放弃焦点。下面的通用 INPUT 分支
        // 曾经吞掉它（capture 阶段，字段自己的处理器永远不跑），留下一个显示着
        // 已丢过滤器的框。
        if (active.closest(".sf-ptge-srch")) {
            if (active.value) {
                _search = "";
                active.value = "";
                renderContent(_overlay.querySelector(".sf-ptge-content"));
            } else {
                active.blur();
            }
            e.stopPropagation();
            return;
        }
        if (active.closest(".sf-ptge-newcat")) {     // 命名新分类：取消它
            if (typeof active._sfCancel === "function") active._sfCancel(); else active.blur();
            e.stopPropagation();
            return;
        }
        const form = _overlay.querySelector(".sf-ptge-create");
        if (form && form.contains(active) && (_createDraft.name || _createDraft.text)) {
            active.blur();
            e.stopPropagation();
            return;
        }
        // 编辑器里任何其它字段——卡片的名称或文本，人们真正花时间的地方，也是
        // Escape 本能反应是关掉浏览器自动填充列表的地方。放掉字段，不放编辑器。
        const t = active.tagName;
        if (t === "INPUT" || t === "TEXTAREA") {
            // blur 即提交的字段必须走它自己的取消句柄，绝不能用 blur()——
            // blur 就是 Escape 变成应用编辑的那条路
            if (typeof active._sfCancel === "function") active._sfCancel(); else active.blur();
            e.stopPropagation();
            return;
        }
    }
    e.stopPropagation();
    closeLibraryEditor();
}

export function closeLibraryEditor() {
    window.removeEventListener("keydown", onKey, true);
    hideCatMenu();
    // 双保险，不是活体恢复。卡片名称输入框已不再把非法名字写进工作副本（见
    // makeCard 的 blur/input 处理器），所以 _data.tags 里不该有空的/重名的，
    // 这层应该永远跑不到。留着：空名会被 normalize 静默丢弃，丢标签不可恢复。
    if (_data) {
        for (const t of _data.tags) {
            const u = uniqueNameExcept(t.name, t);
            if (u !== t.name) t.name = u;
        }
        // 只在确实与存储不同时写回。无条件提交意味着仅仅打开又关闭编辑器就把
        // 本标签页的快照盖上去，静默撤销另一个标签页的编辑。
        try { if (!isSameAsStored(_data)) commitLibrary(_data); } catch { /* ignore */ }
    }
    // flushLibrary 只在有待写的防抖写入时写，所以既丢不了最后一次编辑，也不会
    // 把本标签页快照盖到别人头上（无条件的 flush 会静默取消两行前的 isSameAsStored）
    try { flushLibrary(); } catch { /* ignore */ }
    try { flushCursors(); } catch { /* ignore */ }   // 立刻落盘任何 Start-over
    try { _undoGuardOff?.(); } catch { /* ignore */ }
    _undoGuardOff = null;
    if (_overlay) { try { _overlay.remove(); } catch { /* ignore */ } }
    _overlay = null;
    _node = null;
    _opts = null;
    _data = null;
    _createDraft = newDraft();
}
export function closeLibraryEditorFor(node) { if (_node === node) closeLibraryEditor(); }
