// ==========================================================================
// SF LoRA Stack - 信息面板（点击行的 i）。展示 LoRA 的 info + 直接从文件
// 读取的触发词，让用户勾选哪些词进入 triggers 输出，并提供可选的 Civitai
// 查询（searching / found / not found / offline 四态）。选择持久在行上。
// ==========================================================================
import { app } from "/scripts/app.js";
import { readState, patchLora, accentOf, BRAND } from "./sf_lora_stack_core.js";
import { renderMarkdown } from "./sf_markdown.js";
import { loadImageAsWorkflow } from "./sf_lora_info.js";
import { loraInfo, thumbUrl, civitaiLookup, invalidateInfo, deleteCivitai, saveCustomTriggers,
    saveCustomDescription, saveLoraPreview, deleteLoraPreview, saveCivitaiThumb, migrateLoraData } from "./sf_lora_stack_api.js";
import { getNodeRect } from "./sf_lora_stack_settings.js";
import { copyText } from "./sf_workflows_ui.js";
import { escapeHtml, installWheelZoomPassthrough } from "./sf_common.js";

let _panel = null;
let _cleanup = null;
// 面板归属键（Stack：节点对象；LoRA 浏览器：字符串 key）。closeInfoPanelFor
// 用它在节点删除时只关自己拥有的面板。
let _ownerKey = null;
let _followRaf = null;   // 让面板跟随其节点，见 startFollowing()
let _userMoved = false;  // 用户拖过它，停止跟随
// 用户手动调整过的面板大小（会话级记忆：关闭重开保持，刷新页面回默认）。
let _panelSize = null;
let _panelAccent = BRAND; // 当前面板 accent（关闭确认框同主题用）

// ── Description 编辑态（模块级：closeInfoPanel 需要读 dirty 判定确认框）──
// 关闭面板时重置（doCloseInfoPanel）——残留会让下一行面板带着旧行草稿
// 直接进编辑态（泄漏 bug）。
let _descEditing = false;
let _descDraft = "";
let _descBase = "";
let _descDirty = false;

const _PANEL_MIN_W = 280;
const _PANEL_MIN_H = 240;

function injectCSS() {
    if (document.getElementById("sf-ls-info-css")) return;
    const s = document.createElement("style");
    s.id = "sf-ls-info-css";
    s.textContent = `
    .sf-ls-info-p { position:fixed; z-index:10025; width:420px; max-width:94vw; background:#2b2b2b;
      border:1px solid var(--acc, var(--sf-acc, #f66744)); border-radius:10px; box-shadow:0 14px 44px rgba(0,0,0,0.6);
      overflow:hidden; font:12px 'Segoe UI',system-ui,sans-serif; color:#ddd;
      display:flex; flex-direction:column; max-height:92vh; }
    /* 中间滚动容器：面板被用户拉高/拉矮后，头部与 footer 固定，内容滚动。
       flex:0 1 auto——拉高增量只给 Description，触发词区保持内容高度；
       拉矮时按内容比例收缩、内部滚动。 */
    .sf-ls-info-body { flex:0 1 auto; min-height:0; overflow-y:auto; }
    /* 右下角拖拽调大小手柄 */
    .sf-ls-resize { position:absolute; right:0; bottom:0; width:16px; height:16px;
      cursor:se-resize; z-index:3; }
    .sf-ls-resize::after { content:""; position:absolute; right:3px; bottom:3px;
      width:7px; height:7px; border-right:2px solid #7a7a7a; border-bottom:2px solid #7a7a7a; }
    .sf-ls-resize:hover::after { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-info-top { display:flex; gap:11px; padding:12px; border-bottom:1px solid #1c1c1c; cursor:grab; }
    .sf-ls-info-th { width:64px; height:64px; border-radius:7px; flex:none; border:1px solid #000;
      background:radial-gradient(circle at 60% 35%,#4a3a5b,#221a2e 72%); background-size:cover; background-position:center;
      position:relative; overflow:hidden; cursor:pointer; }
    /* 悬停标签同时充当拖放目标反馈和保存中状态，图片上永远只有一层提示。 */
    .sf-ls-info-th::after { content:attr(data-hint); position:absolute; inset:0; display:flex;
      align-items:center; justify-content:center; text-align:center; padding:2px;
      background:rgba(0,0,0,0.58); color:#fff; font:9.5px 'Segoe UI',system-ui,sans-serif;
      opacity:0; transition:opacity .12s; pointer-events:none; }
    .sf-ls-info-th:hover::after, .sf-ls-info-th.drop::after, .sf-ls-info-th.busy::after { opacity:1; }
    .sf-ls-info-th.drop { border-color:var(--acc, var(--sf-acc, #f66744)); box-shadow:inset 0 0 0 2px var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-thx { position:absolute; top:2px; right:2px; width:15px; height:15px; z-index:2;
      border-radius:50%; background:rgba(0,0,0,0.7); color:#ddd;
      font:10px/15px 'Segoe UI',system-ui,sans-serif; text-align:center; opacity:0; transition:opacity .12s; }
    .sf-ls-info-th:hover .sf-ls-thx { opacity:1; }
    .sf-ls-thx:hover { background:#e0604a; color:#fff; }
    .sf-ls-info-h { min-width:0; flex:1; }
    .sf-ls-info-h h3 { margin:0 0 4px; font-size:13.5px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .sf-ls-info-meta { font:10px monospace; color:#7a7a7a; line-height:1.7; }
    .sf-ls-civlink { display:inline-block; margin-top:5px; font:10.5px 'Segoe UI'; color:#8fc0ff;
      cursor:pointer; }
    .sf-ls-civlink:hover { color:#b8d8ff; text-decoration:underline; }
    .sf-ls-info-x { margin-left:auto; color:#8a8a8a; cursor:pointer; align-self:flex-start; }
    .sf-ls-info-x:hover { color:#fff; }
    .sf-ls-info-sec { padding:11px 12px; }
    /* Description 区块是面板直接 flex 子项（在 bodyWrap 之外、footer 之前）：
       flex:1 1 auto 让它随面板拉高/拉矮弹性伸缩，查看态正文充满剩余空间。 */
    .sf-ls-desc { padding:11px 12px 12px; border-top:1px solid #1c1c1c;
      flex:1 1 auto; min-height:0; display:flex; flex-direction:column; }
    .sf-ls-desc h4 { margin:0 0 6px; font:600 9.5px 'Segoe UI'; text-transform:uppercase; letter-spacing:.7px;
      color:var(--acc, var(--sf-acc, #f66744)); display:flex; align-items:center; gap:7px; }
    .sf-ls-desc h4 .src { margin-left:auto; font:9px 'Segoe UI'; text-transform:none; letter-spacing:0;
      color:#8a8a8a; border:1px solid #444; border-radius:99px; padding:1px 7px; }
    .sf-ls-desc h4 .src.net { color:#8fc0ff; border-color:#3a5a80; }
    .sf-ls-desc h4 .qa { margin-left:8px; font:9.5px 'Segoe UI'; text-transform:none; letter-spacing:0;
      color:#9a9a9a; cursor:pointer; }
    .sf-ls-desc h4 .qa:hover { color:var(--acc, var(--sf-acc, #f66744)); }
    /* Save 按钮 dirty 高亮：草稿 ≠ 进入编辑时的基准值（改动未保存） */
    .sf-ls-desc h4 .qa.dirty { background:var(--acc, var(--sf-acc, #f66744)); color:#fff;
      border-radius:4px; padding:1px 7px; font-weight:600; }
    .sf-ls-desc-body { font-size:11px; color:#d0d0d0; line-height:1.6; white-space:normal;
      word-break:break-word; flex:1 1 auto; min-height:0; overflow-y:auto; padding-right:2px; }
    .sf-ls-desc-none { color:#777; font-size:11px; }
    .sf-ls-desc textarea { width:100%; box-sizing:border-box; background:#161616;
      border:1px solid rgba(255,255,255,0.14); border-radius:6px; color:#fff;
      font:11px 'Segoe UI'; padding:6px 8px; outline:none; resize:vertical; min-height:64px;
      flex:1 1 auto; }
    .sf-ls-desc textarea:focus { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-desc-actions { display:flex; gap:5px; margin-top:6px; }
    .sf-ls-desc-actions button { background:rgba(255,255,255,0.06);
      border:1px solid rgba(255,255,255,0.14); color:#ccc; border-radius:6px; padding:5px 11px;
      font:11px 'Segoe UI'; cursor:pointer; }
    .sf-ls-desc-actions button:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-ls-desc-actions .rm { color:#c9736a; }
    .sf-ls-desc-actions .rm:hover { border-color:#e0604a; color:#fff; }
    /* 编辑态：上传示例图 + 图库网格（点击插入 markdown 引用） */
    .sf-ls-desc-upload { display:flex; align-items:center; gap:7px; margin-top:7px; flex-wrap:wrap; }
    .sf-ls-desc-upload button { background:rgba(255,255,255,0.06);
      border:1px solid rgba(255,255,255,0.18); color:#ccc; border-radius:6px; padding:4px 10px;
      font:11px 'Segoe UI'; cursor:pointer; }
    .sf-ls-desc-upload button:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-ls-desc-upload .hint { font-size:10px; color:#7a7a7a; line-height:1.4; }
    .sf-ls-desc-grid { display:flex; flex-wrap:wrap; gap:6px; margin-top:7px;
      max-height:132px; overflow-y:auto; padding-right:2px; }
    .sf-ls-desc-grid .cell { position:relative; width:56px; height:56px; }
    .sf-ls-desc-grid img { width:56px; height:56px; object-fit:cover; border-radius:6px;
      border:1px solid #3a3a3e; cursor:pointer; display:block; }
    .sf-ls-desc-grid img:hover { border-color:var(--acc, var(--sf-acc, #f66744)); }
    /* 悬停显示的角标按钮：右上角删除、右下角载入工作流、左下角复制 prompt（SVG 统一样式） */
    .sf-ls-desc-grid .cell .x, .sf-ls-desc-grid .cell .load, .sf-ls-desc-grid .cell .prompt {
      position:absolute; width:16px; height:16px; padding:0; border:none;
      display:none; align-items:center; justify-content:center; cursor:pointer; }
    .sf-ls-desc-grid .cell:hover .x, .sf-ls-desc-grid .cell:hover .load, .sf-ls-desc-grid .cell:hover .prompt { display:flex; }
    .sf-ls-desc-grid .cell .x { top:0; right:0; background:rgba(224,96,74,0.92); border-radius:0 5px 0 5px; }
    .sf-ls-desc-grid .cell .load { bottom:0; right:0; background:rgba(79,124,255,0.92); border-radius:5px 0 5px 0; }
    .sf-ls-desc-grid .cell .prompt { bottom:0; left:0; background:rgba(46,160,90,0.92); border-radius:0 5px 0 5px; }
    .sf-ls-desc-grid .cell .x .ic, .sf-ls-desc-grid .cell .load .ic, .sf-ls-desc-grid .cell .prompt .ic {
      width:10px; height:10px; background-color:#fff;
      -webkit-mask-size:contain; mask-size:contain;
      -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
      -webkit-mask-position:center; mask-position:center; display:block; }
    .sf-ls-desc-grid .cell .x .ic { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 3h6l1 2h4v2H4V5h5l1-2zm-2 6h10l-1 9a1 1 0 01-1 1H8a1 1 0 01-1-1L6 9zM10 11v6M14 11v6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 3h6l1 2h4v2H4V5h5l1-2zm-2 6h10l-1 9a1 1 0 01-1 1H8a1 1 0 01-1-1L6 9zM10 11v6M14 11v6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E"); }
    .sf-ls-desc-grid .cell .load .ic { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7z' fill='black'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7z' fill='black'/%3E%3C/svg%3E"); }
    .sf-ls-desc-grid .cell .prompt .ic { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-2M9 5a2 2 0 002 2h6a2 2 0 002-2M9 5a2 2 0 012-2h4a2 2 0 012 2M9 12h6M9 16h6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-2M9 5a2 2 0 002 2h6a2 2 0 002-2M9 5a2 2 0 012-2h4a2 2 0 012 2M9 12h6M9 16h6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E"); }
    .sf-ls-desc-grid .none { font-size:10px; color:#777; }
    .sf-ls-info-sec h4 { margin:0 0 6px; font:600 9.5px 'Segoe UI'; text-transform:uppercase; letter-spacing:.7px;
      color:var(--acc, var(--sf-acc, #f66744)); display:flex; align-items:center; gap:7px; }
    .sf-ls-info-sec h4 .src { margin-left:auto; font:9px 'Segoe UI'; text-transform:none; letter-spacing:0;
      color:#8a8a8a; border:1px solid #444; border-radius:99px; padding:1px 7px; }
    .sf-ls-info-sec h4 .src.net { color:#8fc0ff; border-color:#3a5a80; }
    .sf-ls-info-sec h4 .qa { margin-left:8px; font:9.5px 'Segoe UI'; text-transform:none; letter-spacing:0;
      color:#9a9a9a; cursor:pointer; }
    .sf-ls-info-sec h4 .qa:hover { color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-info-note { font-size:10px; color:#7a7a7a; margin:0 0 8px; }
    .sf-ls-chips { display:flex; flex-wrap:wrap; gap:5px; max-height:36vh; overflow-y:auto; padding-right:2px; }
    .sf-ls-chips::-webkit-scrollbar { width:7px; }
    .sf-ls-chips::-webkit-scrollbar-thumb { background:#555; border-radius:3px; }
    .sf-ls-chip { font:10.5px 'Segoe UI'; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.14);
      color:#b8b8b8; border-radius:99px; padding:3px 9px; cursor:pointer; user-select:none; display:flex; align-items:center; gap:4px; max-width:100%; }
    .sf-ls-chip:hover { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-chip.sel { background:color-mix(in srgb, var(--acc, var(--sf-acc, #f66744)) 18%, transparent); border-color:var(--acc, var(--sf-acc, #f66744)); color:#f8a48c; }
    .sf-ls-chip.sel::before { content:"✓"; font-size:9px; flex:none; }
    .sf-ls-chip .ct { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-ls-chip-none { color:#777; font-size:11px; }
    .sf-ls-chip .cx { margin-left:1px; color:#f8a48c; cursor:pointer; opacity:.6; font-size:10px; flex:none; }
    .sf-ls-chip .cx:hover { opacity:1; }
    .sf-ls-srctoggle { margin-left:auto; display:flex; border:1px solid #444; border-radius:99px; overflow:hidden; }
    .sf-ls-srctoggle .sg { font:9px 'Segoe UI'; text-transform:none; letter-spacing:0; color:#9a9a9a; padding:2px 9px; cursor:pointer; }
    .sf-ls-srctoggle .sg:hover { color:#ddd; }
    .sf-ls-srctoggle .sg.on { background:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-ls-addtrig { display:flex; gap:5px; margin-top:8px; }
    .sf-ls-addtrig input { flex:1; min-width:0; box-sizing:border-box; background:#161616;
      border:1px solid rgba(255,255,255,0.14); border-radius:6px; color:#fff; font:11px 'Segoe UI';
      padding:5px 8px; outline:none; }
    .sf-ls-addtrig input:focus { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-ls-addtrig button { flex:0 0 auto; background:rgba(255,255,255,0.06);
      border:1px solid rgba(255,255,255,0.14); color:#ccc; border-radius:6px; padding:5px 11px;
      font:11px 'Segoe UI'; cursor:pointer; }
    .sf-ls-addtrig button:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-ls-strip { margin:0 12px 11px; border-radius:7px; padding:9px 10px; font-size:11px; line-height:1.5;
      display:flex; gap:9px; align-items:flex-start; }
    .sf-ls-strip .st-ic { flex:none; width:18px; height:18px; border-radius:50%; color:#fff; font-size:11px;
      display:flex; align-items:center; justify-content:center; }
    .sf-ls-strip.searching { background:rgba(90,160,230,0.12); } .sf-ls-strip.searching .st-ic { background:#5aa0e6; }
    .sf-ls-strip.found { background:rgba(62,195,113,0.12); } .sf-ls-strip.found .st-ic { background:#3ec371; }
    .sf-ls-strip.nofind { background:rgba(255,255,255,0.05); } .sf-ls-strip.nofind .st-ic { background:#6f6f6f; }
    .sf-ls-strip.offline { background:rgba(233,165,61,0.12); } .sf-ls-strip.offline .st-ic { background:#e9a53d; }
    .sf-ls-strip-acts { display:flex; gap:5px; margin-left:auto; flex:none; align-items:center; }
    .sf-ls-strip-acts button { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.18);
      color:#ccc; border-radius:6px; padding:4px 10px; font:11px 'Segoe UI',sans-serif; cursor:pointer; }
    .sf-ls-strip-acts button:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-ls-strip-acts button.pri { background:var(--acc, var(--sf-acc, #f66744)); border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; font-weight:600; }
    .sf-ls-strip-acts button.pri:hover { filter:brightness(1.1); }
    .sf-ls-spin { width:11px; height:11px; border:2px solid rgba(255,255,255,.3); border-top-color:#fff;
      border-radius:50%; animation:sf-ls-sp 1s linear infinite; }
    @keyframes sf-ls-sp { to { transform:rotate(360deg); } }
    .sf-ls-info-foot { display:flex; gap:6px; padding:10px 12px; border-top:1px solid #1c1c1c; background:#242424; }
    .sf-ls-info-foot .b { flex:1; text-align:center; font-size:11px; padding:7px; border-radius:5px; cursor:pointer; }
    .sf-ls-info-foot .b.pri { background:var(--acc, var(--sf-acc, #f66744)); color:#fff; font-weight:600; }
    .sf-ls-info-foot .b.gh { border:1px solid rgba(255,255,255,0.14); color:#b8b8b8; }
    .sf-ls-info-foot .b.gh:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-ls-info-foot .b.dis { opacity:.4; pointer-events:none; }
    .sf-ls-info-foot .b.del { flex:0 0 auto; min-width:38px; border:1px solid rgba(255,255,255,0.14); color:#c9736a; }
    .sf-ls-info-foot .b.del:hover { border-color:#e0604a; color:#fff; background:rgba(224,96,74,0.12); }

    /* 面板风确认框（替代原生 confirm，避免 UI 割裂） */
    .sf-ls-confirm-mask { position:fixed; inset:0; z-index:10040; background:rgba(0,0,0,0.55);
      display:flex; align-items:center; justify-content:center; }
    .sf-ls-confirm { width:300px; max-width:90vw; background:#2b2b2b; border:1px solid var(--acc, var(--sf-acc, #f66744));
      border-radius:10px; box-shadow:0 14px 44px rgba(0,0,0,0.6); color:#ddd;
      font:12px 'Segoe UI',system-ui,sans-serif; overflow:hidden; }
    .sf-ls-confirm-t { padding:12px 14px; border-bottom:1px solid #1c1c1c; color:#fff;
      font-size:13px; font-weight:600; }
    .sf-ls-confirm-b { padding:12px 14px; font-size:12px; line-height:1.6; color:#d0d0d0; }
    .sf-ls-confirm-f { display:flex; gap:8px; padding:10px 14px; border-top:1px solid #1c1c1c;
      background:#242424; justify-content:flex-end; }
    .sf-ls-confirm-f .b { padding:6px 14px; border-radius:6px; font-size:12px; cursor:pointer;
      user-select:none; }
    .sf-ls-confirm-f .b.pri { background:var(--acc, var(--sf-acc, #f66744)); color:#fff; font-weight:600; }
    .sf-ls-confirm-f .b.pri:hover { filter:brightness(1.1); }
    .sf-ls-confirm-f .b.gh { border:1px solid rgba(255,255,255,0.18); color:#ccc; }
    .sf-ls-confirm-f .b.gh:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
  `;
    document.head.appendChild(s);
}

// ── Description 内嵌示例图（与 sf_lora_info.js 编辑器同款机制）─────────────
// 图片上传到 `models/loras/<lora 目录>/sample/`（后端 /api/sfnodes/lora_samples/
// upload 复用），描述里以相对路径 `sample/<文件名>` 引用；查看态用
// resolveSampleUrl 把它解析回图片 URL（目录改名/移动后按当前 lora 路径解析，
// 无需修复 markdown 文本）。插入格式与 sf_lora_info.js 一致。

function buildSampleMarkdown(path) {
    const base = String(path || "").split("/").pop() || "image";
    const alt = base.replace(/\.[^.]+$/, "");
    const rel = `sample/${encodeURIComponent(base)}`;
    return `![${alt}](${rel})`;
}

function insertAtCursor(textarea, text) {
    const start = textarea.selectionStart ?? textarea.value.length;
    const end = textarea.selectionEnd ?? start;
    textarea.setRangeText(text, start, end, "end");
    textarea.focus();
    const pos = start + text.length;
    textarea.selectionStart = textarea.selectionEnd = pos;
}

// 描述里 `sample/xxx.png` 相对路径 -> 图片 URL（基于当前 lora 的目录）。
function resolveSampleUrl(rel, loraName) {
    let r = rel;
    try { r = decodeURIComponent(rel); } catch { /* 保留原样 */ }
    const idx = loraName.lastIndexOf("/");
    const dir = idx === -1 ? "" : loraName.slice(0, idx + 1);
    return `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(dir + r)}`;
}

// ── 右下角拖拽调大小手柄 ───────────────────────────────────────────────────
// renderBody 每次重建面板时调用（handle 是面板子元素，随 innerHTML 清除）。
// 拖拽调整 width/height 并记忆（_panelSize）；最小尺寸钳制；释放即结束
// （buttons 守卫兜底丢失的 pointerup）。
function attachResize(panel) {
    const h = el("div", "sf-ls-resize");
    h.title = "Drag to resize";
    panel.appendChild(h);
    h.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const r = panel.getBoundingClientRect();
        try { h.setPointerCapture(e.pointerId); } catch { /* 不可捕获 */ }
        let done = false;
        const move = (ev) => {
            if (!panel.isConnected) return up();
            if (!(ev.buttons & 1)) return up();
            const w = Math.max(_PANEL_MIN_W, Math.min(window.innerWidth - 16, ev.clientX - r.left));
            const hh = Math.max(_PANEL_MIN_H, Math.min(window.innerHeight - 16, ev.clientY - r.top));
            panel.style.width = w + "px";
            panel.style.height = hh + "px";
            _panelSize = { w, h: hh };
        };
        const up = () => {
            if (done) return;
            done = true;
            try { h.releasePointerCapture(e.pointerId); } catch { /* 已离开 */ }
            h.removeEventListener("pointermove", move, true);
            h.removeEventListener("pointerup", up, true);
            h.removeEventListener("pointercancel", up, true);
            h.removeEventListener("lostpointercapture", up, true);
        };
        h.addEventListener("pointermove", move, true);
        h.addEventListener("pointerup", up, true);
        h.addEventListener("pointercancel", up, true);
        h.addEventListener("lostpointercapture", up, true);
    });
}

// 面板风确认框：返回 Promise<boolean>。遮罩点击 / Esc = 取消。与信息面板
// 同主题（accent 边框 + 主按钮），替代割裂的原生 confirm。行菜单（载入预设
// 等）也复用它。
export function confirmDialog(opts) {
    // 确认框样式随信息面板的 injectCSS 注入——预设菜单等独立入口从未打开
    // 信息面板，不注入会得到无样式遮罩（透明、无定位，挂在 body 末尾不可
    // 见，"确认框没出现"）。injectCSS 幂等（#sf-ls-info-css 守卫）。
    injectCSS();
    return new Promise((resolve) => {
        const { title, message, okLabel = "Replace", cancelLabel = "Keep mine", accent = BRAND } = opts || {};
        const mask = el("div", "sf-ls-confirm-mask");
        const box = el("div", "sf-ls-confirm");
        box.style.setProperty("--acc", accent);
        const t = el("div", "sf-ls-confirm-t", title);
        const b = el("div", "sf-ls-confirm-b", message);
        const f = el("div", "sf-ls-confirm-f");
        const cancel = el("div", "b gh", cancelLabel);
        const ok = el("div", "b pri", okLabel);
        const onKey = (e) => {
            if (e.key === "Escape") { e.stopPropagation(); done(false); }
        };
        const done = (v) => {
            document.removeEventListener("keydown", onKey, true);
            mask.remove();
            resolve(v);
        };
        cancel.addEventListener("click", () => done(false));
        ok.addEventListener("click", () => done(true));
        f.append(cancel, ok);
        box.append(t, b, f);
        mask.appendChild(box);
        // 遮罩点击 = 取消；点击框内不算。
        mask.addEventListener("pointerdown", (e) => { if (e.target === mask) done(false); });
        document.addEventListener("keydown", onKey, true);
        document.body.appendChild(mask);
        ok.focus();
    });
}

// 关闭面板。有未保存的 Description 修改时先经同主题确认框（返回
// Promise<boolean>：true = 已关闭/无需确认；false = 用户取消保留草稿）。
// ✕/Esc 等调用方忽略返回值即可（内部异步确认后自行关闭）；openInfoPanel
// 切换行时 await 它，取消则不切换。节点删除路径走 doCloseInfoPanel
// （closeInfoPanelFor）——删除不能弹框阻塞，且面板随节点消失。
export function closeInfoPanel() {
    if (_panel && _descEditing && _descDirty) {
        return confirmDialog({
            title: "Discard description changes?",
            message: "You have unsaved changes to this description. Close and discard them?",
            okLabel: "Discard",
            cancelLabel: "Keep editing",
            accent: _panelAccent,
        }).then((ok) => {
            if (ok) doCloseInfoPanel();
            return ok;
        });
    }
    doCloseInfoPanel();
    return Promise.resolve(true);
}

function doCloseInfoPanel() {
    // 关闭即丢弃草稿：残留的编辑态/基准会让下一次打开带着上一行的旧草稿
    // 直接进编辑态（泄漏 bug），必须重置。
    _descEditing = false;
    _descDraft = "";
    _descBase = "";
    _descDirty = false;
    if (_cleanup) { try { _cleanup(); } catch { /* 忽略 */ } }
    _cleanup = null;
    stopFollowing();
    // 在关闭而非打开时重置：用户拖过的面板不能让下一个学着静坐。
    _userMoved = false;
    if (_panel) { try { _panel.remove(); } catch { /* 忽略 */ } }
    _panel = null;
    _ownerKey = null;
}

// 只在本节点拥有打开的面板时关闭（删除无关的 LoRA Stack 节点不能扯走
// 另一个节点开着的面板）。节点删除路径不弹未保存确认——删除不能被阻塞。
export function closeInfoPanelFor(node) { if (_ownerKey === node) doCloseInfoPanel(); }

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

// 在节点旁放置面板。必须在第一次 await 前调用，否则 fetch 期间面板画在
// 视口左上角：`.sf-ls-info-p` 是 position:fixed 无 left/top，直到这里运行
// 它都坐在 0,0。缓存命中时 await 在微任务里解决、无帧画出，这就是闪光只
// 在选新 LoRA 后出现的原因。与下拉同款（先放再等，从未闪）。
function place(panel, ctx) {
    // Stack：锚定节点矩形；浏览器类宿主：锚定 anchorRect() 的结果。两者都
    // 是屏幕坐标矩形（节点矩形经 getNodeRect 折算画布 transform）。
    const r = ctx?.node ? getNodeRect(ctx.node)
        : (typeof ctx?.anchorRect === "function" ? ctx.anchorRect() : null);
    const pad = 8, gap = 12;
    const pw = panel.offsetWidth, ph = panel.offsetHeight;
    let left = r ? r.right + gap : (window.innerWidth - pw) / 2;
    if (left + pw > window.innerWidth - pad) left = r ? Math.max(pad, r.left - gap - pw) : left;
    let top = r ? r.top : (window.innerHeight - ph) / 2;
    top = Math.max(pad, Math.min(top, window.innerHeight - ph - pad));
    panel.style.left = Math.max(pad, left) + "px";
    panel.style.top = top + "px";
}

// 面板长高后把它拉回屏内（Civitai 查询加了状态条和更多 chips）。刻意不是
// 重新放置：标题可拖动，完整重放会扯走用户自己放好的面板。只纠正真正越
// 界的边。
function clampIntoView(panel) {
    // 面板真正放置前空操作。renderBody() 在第一次 place() 前先跑一次，
    // 那时 style.left/top 还是空——没有这个守卫 parseFloat("") 读 0，
    // 把面板钉死在左上角。
    if (!panel.style.left || !panel.style.top) return;
    const pad = 8;
    const pw = panel.offsetWidth, ph = panel.offsetHeight;
    const left = parseFloat(panel.style.left) || 0;
    const top = parseFloat(panel.style.top) || 0;
    panel.style.left = Math.max(pad, Math.min(left, window.innerWidth - pw - pad)) + "px";
    panel.style.top = Math.max(pad, Math.min(top, window.innerHeight - ph - pad)) + "px";
}

export async function openInfoPanel(node, id, refresh) {
    // Stack 行 UI 入口（兼容旧签名）：构造宿主上下文后委托 openInfoPanelFor，
    // 行为与旧版逐字节一致——行读写仍是节点状态。
    if (!node) return;
    return openInfoPanelFor({
        key: node,
        node,
        getRow: () => readState(node).loras.find((e) => e.id === id) || null,
        patchRow: (patch) => patchLora(node, id, patch),
        accent: accentOf(node),
        prefs: () => {
            const st = readState(node);
            return { civitai: st.civitai !== false, thumbs: st.thumbs !== false };
        },
        refresh,
    }, id);
}

/**
 * 打开信息面板（宿主上下文入口，LoRA 浏览器等非节点宿主用）。ctx:
 *   key        归属键（closeInfoPanelFor 在节点删除时只关自己的面板）
 *   node       可选。Stack 用它锚定节点并让面板跟随画布
 *   anchorRect 可选函数。浏览器类宿主锚定被点击卡片/元素的矩形
 *   getRow     返回当前行对象 {id,name,triggers,custom}，可 null
 *   patchRow   以局部 patch 更新行（{triggers,custom}）
 *   accent     强调色
 *   prefs      可选函数，返回 {civitai, thumbs}（缺省全开）
 *   refresh    可选，行变更后回调
 */
export async function openInfoPanelFor(ctx, id) {
    if (!ctx || typeof ctx.getRow !== "function" || typeof ctx.patchRow !== "function") return;
    const refresh = ctx.refresh;
    const prefsOf = () => (typeof ctx.prefs === "function" ? ctx.prefs() : {});
    // 切换行/重开：上一面板有未保存 Description 修改时先确认——取消则
    // 保留草稿、不打开新面板（await 返回 false）。
    if (!(await closeInfoPanel())) return;
    injectCSS();
    const entry0 = ctx.getRow();
    if (!entry0) return;
    const name = entry0.name;
    const accent = ctx.accent;
    _panelAccent = accent;

    const panel = el("div", "sf-ls-info-p");
    panel.style.setProperty("--acc", accent);   // body 级面板不继承任何东西
    panel.style.borderColor = accent;
    // 恢复用户上次手动调整的大小（会话级记忆）。
    if (_panelSize) {
        panel.style.width = _panelSize.w + "px";
        panel.style.height = _panelSize.h + "px";
    }
    document.body.appendChild(panel);
    _panel = panel;
    _ownerKey = ctx.key;
    // _panel 赋值之后：循环的第一件事是检查它拥有该面板。
    if (ctx.node) startFollowing(panel, ctx.node);

    // 本面板会话的视图数据
    let info = { title: name || "LoRA", triggers: [], file_triggers: [], sidecar_triggers: [], source: "file", has_preview: false, custom_preview: false, preview_v: 0, description: "", custom_description: "", orphan_key: "" };
    // 孤儿迁移提示条按面板会话 dismiss（文件没动的话每次打开都值得再看一眼）。
    let _orphanDismissed = false;
    let civ = null; // { state:"searching"|"found"|"nofind"|"offline", info?, message? }
    // 展示哪组候选词："file" | "civitai"。null = 自动（有已存侧车/刚取回则
    // Civitai，否则文件自己的词）。用户选中的词与视图无关地持久在 row.triggers。
    let viewSource = null;
    // 本面板重写侧车（Civitai 查询/删除）或用户预览时 bump，让 thumb()
    // 越过浏览器一小时图片缓存。
    let _thumbBust = 0;
    // 标题下的一行问题注释（没存上的图）。下次成功清掉。
    let _msg = null;
    // 图片编码上传期间为 true，第二次拖放/点击不能对同一文件名发起竞争保存。
    let _busy = false;

    const selected = () => new Set((ctx.getRow()?.triggers || []).map((w) => w.toLowerCase()));

    // 图片问题是关于用户最后一次尝试的，任何其它有意动作清掉它。没有这个
    // 条会粘住：误拖 .txt 后警告坐在标题下，经历每次 chip 点击和每次 Civitai
    // 查询直到面板寿命结束。
    function clearMsg() { _msg = null; }

    function toggleWord(word) {
        clearMsg();
        const e = ctx.getRow();
        if (!e) return;
        const key = word.toLowerCase();
        const has = e.triggers.some((w) => w.toLowerCase() === key);
        const next = has ? e.triggers.filter((w) => w.toLowerCase() !== key) : [...e.triggers, word];
        ctx.patchRow({ triggers: next });
        refresh?.(false);
        renderBody();
    }
    function setWords(words) {
        clearMsg();
        ctx.patchRow({ triggers: words.slice() });
        refresh?.(false);
        renderBody();
    }

    // 有 Civitai 词可用时（已存侧车或刚取回的结果）。
    function civitaiAvailable() {
        return (info.sidecar_triggers?.length || 0) > 0 || civ?.state === "found";
    }
    function fileWords() { return info.file_triggers?.length ? info.file_triggers : (info.triggers || []); }
    function civitaiWords() {
        if (info.sidecar_triggers?.length) return info.sidecar_triggers;
        if (civ?.state === "found") return civ.info?.triggers || [];
        return [];
    }
    // 活动视图源：尊重用户切换，否则自动。
    function effectiveSource() {
        if (viewSource === "file" || viewSource === "civitai") return viewSource;
        return civitaiAvailable() ? "civitai" : "file";
    }
    // 当前视图展示的候选词。
    function sourceWords() {
        return effectiveSource() === "civitai" ? (civitaiWords().length ? civitaiWords() : fileWords()) : fileWords();
    }

    // 展示的 chips：源词 + 用户自定义词 + 任何已选词，去重。
    // `isCustom` 按 `custom` 的成员关系（而非推入顺序）判定——自定义词后来
    // 也成为源词（如 Civitai 查询后）仍带可移除 ✕。
    function chipList() {
        const src = sourceWords();
        const row = ctx.getRow();
        const custom = row?.custom || [];
        const customSet = new Set(custom.map((w) => w.toLowerCase()));
        const out = []; const seen = new Set();
        const push = (w) => {
            const k = w.toLowerCase();
            if (w && !seen.has(k)) { seen.add(k); out.push({ w, isCustom: customSet.has(k) }); }
        };
        for (const w of src) push(w);
        for (const w of custom) push(w);
        for (const w of (row?.triggers || [])) push(w);
        return out;
    }

    // 自定义词属于 LoRA 文件，不属于这一行：存在 ComfyUI user 目录的单一
    // 存储里，同一 LoRA 在任意行/节点/工作流都回来。行仍持一份副本（chipList
    // 和勾选的 `triggers` 读它），所以双写——行管当下，存储管存续。
    function persistCustom(words) {
        if (!name) return;
        saveCustomTriggers(name, words);   // fire and forget：行已有
    }

    function addCustom(word) {
        clearMsg();
        const w = (word || "").trim();
        if (!w) return;
        const e = ctx.getRow();
        if (!e) return;
        const key = w.toLowerCase();
        // 文件/Civitai 已提供该词就只选中它——别再把它塞进 `custom`
        // （那会是源词的隐藏重复）。
        const inSrc = sourceWords().some((x) => x.toLowerCase() === key);
        const custom = (inSrc || (e.custom || []).some((x) => x.toLowerCase() === key))
            ? (e.custom || []) : [...(e.custom || []), w];
        const trig = (e.triggers || []).some((x) => x.toLowerCase() === key) ? e.triggers : [...(e.triggers || []), w];
        ctx.patchRow({ custom, triggers: trig }); // 添加即选中，到达输出
        persistCustom(custom);
        refresh?.(false);
        renderBody();
        setTimeout(() => panel.querySelector(".sf-ls-addtrig input")?.focus(), 0);
    }

    function removeCustom(word) {
        clearMsg();
        const key = (word || "").toLowerCase();
        const e = ctx.getRow();
        if (!e) return;
        const custom = (e.custom || []).filter((x) => x.toLowerCase() !== key);
        ctx.patchRow({
            custom,
            triggers: (e.triggers || []).filter((x) => x.toLowerCase() !== key),
        });
        persistCustom(custom);
        refresh?.(false);
        renderBody();
    }

    // ── Description：Civitai/文件说明 + 用户自定义覆盖 ─────────────────────
    // 自定义描述与自定义触发词同存储（user 目录单一文件，按 LoRA 名键控）。
    // 展示优先级：custom > 当次查询（Civitai live）> 侧车/文件说明。
    // 编辑态草稿独立于 renderBody 生命周期——勾词等重渲染不丢已打文字。
    // 状态（_descEditing/_descDraft/_descBase/_descDirty）是模块级：关闭
    // 面板的确认判定在模块级 closeInfoPanel，闭包需共享同一份。
    const shownDesc = () => info.custom_description
        || (civ?.state === "found" && civ.info?.description)
        || info.description || "";
    const descSrc = () => {
        if (info.custom_description) return "custom";
        if ((civ?.state === "found" && civ.info?.description) || info.source === "sidecar") return "civitai";
        return info.description ? "file" : "";
    };

    function saveDesc(desc) {
        clearMsg();
        if (!name) { showMsg("Pick a LoRA first."); return; }   // 无名行没有可设对象
        saveCustomDescription(name, desc).then((res) => {
            if (!panel.isConnected) return;
            if (!res?.ok) { showMsg(res?.message || "Could not save that description."); return; }
            _msg = null;
            _descEditing = false;
            _descDraft = "";
            _descBase = "";
            _descDirty = false;
            info.custom_description = res.description || "";   // 本地即画，不等 loadInfo
            // 使任何在途 loadInfo 作废：它的响应是保存前的旧快照，落地会把
            // 刚保存的自定义描述覆盖回 Civitai/文件原文（"保存后仍显示来自
            // Civitai"就是这么来的）。
            _infoSeq++;
            renderBody();
        });
    }

    // 放弃编辑：恢复浏览态并重置 dirty 状态（基准/草稿都清，防泄漏）
    function cancelDescEdit() {
        _descEditing = false;
        _descDraft = "";
        _descBase = "";
        _descDirty = false;
    }

    function clearDesc() {
        saveDesc("");   // 空描述 = 清除覆盖，回到 Civitai/文件原文
    }

    // ── 示例图：上传到 sample/ 目录 + 图库网格点击插入 markdown ──────────
    async function uploadSample(file, onInsert) {
        if (!file || !name) return;
        const _t = file.type || "";
        if (_t && !/^image\//.test(_t) && !/^video\//.test(_t)) {
            showMsg("That is not a picture. Use a jpg, png, webp or mp4.");
            return;
        }
        const fd = new FormData();
        fd.append("image", file);
        fd.append("filename", name);
        try {
            const resp = await app.api.fetchApi("/api/sfnodes/lora_samples/upload", {
                method: "POST",
                body: fd,
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
            if (onInsert) onInsert(data.path);
        } catch (e) {
            showMsg("Upload failed: " + (e.message || e));
        }
    }

    async function refreshSampleGrid(grid, onInsert) {
        if (!name || !grid.isConnected) return;
        try {
            const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples?filename=${encodeURIComponent(name)}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            grid.innerHTML = "";
            const imgs = Array.isArray(data.images) ? data.images : [];
            if (!imgs.length) {
                grid.appendChild(el("span", "none",
                    "No sample images yet - upload one to insert it into the description."));
                return;
            }
            for (const p of imgs) {
                const cell = el("div", "cell");
                const isVideo = /\.(mp4|m4v|mov|webm|mkv)$/i.test(p);
                let thumb;
                if (isVideo) {
                    thumb = el("div");
                    thumb.style.cssText = "width:56px;height:56px;border-radius:6px;border:1px solid #3a3a3e;display:flex;align-items:center;justify-content:center;background:#1c1c1e;color:#777;font:9px 'Segoe UI';cursor:pointer;";
                    thumb.textContent = p.split("/").pop().slice(0, 12);
                    thumb.title = p.split("/").pop() + " (video)";
                    thumb.addEventListener("click", () => onInsert(p));
                } else {
                    thumb = document.createElement("img");
                    thumb.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(p)}&w=256`;
                    thumb.title = p.split("/").pop();
                    thumb.loading = "lazy";
                    thumb.addEventListener("click", () => onInsert(p));
                }
                // 删除：悬停显示右上角（SVG）
                const del = el("button", "x");
                del.title = "Delete this sample image from disk";
                const delIc = el("span", "ic");
                del.appendChild(delIc);
                del.addEventListener("click", async (ev) => {
                    ev.stopPropagation();
                    const fileName = p.split("/").pop();
                    const ok = await confirmDialog({
                        title: "Delete sample image?",
                        message: `Delete "${fileName}" from this LoRA's sample/ folder? This cannot be undone.`,
                        okLabel: "Delete",
                        cancelLabel: "Cancel",
                        accent,
                    });
                    if (!ok || !panel.isConnected) return;
                    try {
                        const r = await app.api.fetchApi(
                            `/api/sfnodes/lora_samples?path=${encodeURIComponent(p)}`,
                            { method: "DELETE" });
                        if (!r.ok) throw new Error(`HTTP ${r.status}`);
                        refreshSampleGrid(grid, onInsert);
                    } catch (err) {
                        showMsg("Could not delete that picture: " + (err.message || err));
                    }
                });
                // 载入为工作流：悬停显示右下角（SVG）
                const load = el("button", "load");
                load.title = "Load this picture as a workflow (needs embedded workflow data)";
                const loadIc = el("span", "ic");
                load.appendChild(loadIc);
                load.addEventListener("click", async (ev) => {
                    ev.stopPropagation();
                    await loadImageAsWorkflow(p, (msg) => showMsg(msg));
                });
                // 读取 prompt 并复制到剪贴板：悬停显示左下角（SVG）
                const promptBtn = el("button", "prompt");
                promptBtn.title = "Copy prompt from this image to clipboard";
                const promptIc = el("span", "ic");
                promptBtn.appendChild(promptIc);
                promptBtn.addEventListener("click", async (ev) => {
                    ev.stopPropagation();
                    promptBtn.style.opacity = "0.5";
                    promptBtn.style.pointerEvents = "none";
                    try {
                        const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples/prompt?path=${encodeURIComponent(p)}`);
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok) throw new Error(data.message || `HTTP ${resp.status}`);
                        if (!data.found || !data.text) {
                            showMsg(data.message || "No prompt found in this image.");
                            return;
                        }
                        const ok = await copyText(data.text);
                        if (ok) showMsg("Prompt copied to clipboard.");
                        else showMsg("Could not copy to clipboard.");
                    } catch (err) {
                        showMsg("Could not read prompt: " + (err.message || err));
                    } finally {
                        promptBtn.style.opacity = "";
                        promptBtn.style.pointerEvents = "";
                    }
                });
                cell.append(thumb, del, load, promptBtn);
                grid.appendChild(cell);
            }
        } catch (e) {
            grid.innerHTML = "";
            grid.appendChild(el("span", "none", "Could not load sample images."));
        }
    }

    // 面板打开时把这个 LoRA 的已存词带到行上。也反向迁移：行已在存储出现
    // 前（或在另一台机器做的工作流里）带着自定义词，则推入存储——任何人
    // 已有的词都不会因这次改动丢失。
    function hydrateCustom() {
        const stored = Array.isArray(info?.custom_triggers) ? info.custom_triggers : [];
        const e = ctx.getRow();
        if (!e) return;
        const rowWords = e.custom || [];
        const seen = new Set();
        const merged = [];
        for (const w of [...stored, ...rowWords]) {
            const k = String(w || "").trim().toLowerCase();
            if (!k || seen.has(k)) continue;
            seen.add(k);
            merged.push(w);
        }
        // 只有真变了才碰状态——空操作写会在每次打开面板时弄脏干净的工作流。
        const same = merged.length === rowWords.length
            && merged.every((w, i) => w === rowWords[i]);
        if (!same) { ctx.patchRow({ custom: merged }); refresh?.(false); }
        // 行带着存储缺的词时才写回。
        if (merged.length > stored.length) persistCustom(merged);
    }

    function thumb() {
        // 用户自己的图胜过一切，包括实时 Civitai 结果。"换成我的"必须在下次
        // ↻ Civitai 后仍存活，否则覆盖在第一次查询时悄悄自我撤销。
        // preview_v 是文件 mtime，另一个节点/早前会话设的图也能越过浏览器
        // 一小时图片缓存。
        if (info.custom_preview && name) {
            return thumbUrl(name, Math.max(_thumbBust, info.preview_v || 0));
        }
        if (civ?.state === "found" && civ.info?.thumbnail) return civ.info.thumbnail;
        // _thumbBust 在本面板改了侧车时设置（Civitai 查询/删除）：缩略图 URL
        // 永不变化且路由发 max-age=3600，没有 bust 浏览器会显示变化前的图
        // 一小时。
        if (info.has_preview && name) return thumbUrl(name, _thumbBust);
        return null;
    }

    // ── 用户自己的预览图 ───────────────────────────────────────────────────
    // 点框选文件、拖放一个、或粘贴一个。存在 ComfyUI user 目录（绝不进
    // models 目录，那里常只读/网络盘）且胜过 Civitai 图；Remove 让自动图
    // 回来，已有的东西永不销毁。

    const PREVIEW_MAX = 512;   // 最长边；框只显示 64px，4K 拖放否则要传几 MB

    /** 齿轮的 "Show preview thumbnails"。每次渲染重读（与 `civitaiOn` 同款
     *  习惯）让齿轮切换永不过期。关闭时图片框完全不建——Civitai 预览在点
     *  i 的瞬间出现，这正是录制时能关掉它的意义，留个空方框（或活拖放
     *  目标）就错过了。因此下面一切都要容忍框缺席。 */
    function thumbsOn() { return prefsOf().thumbs !== false; }

    /** 浏览器端降采样成小 jpeg，上传只有几十 KB，服务端无需图片库。服务端
     *  仍查大小和 magic bytes——这是为重量，不是为信任。 */
    function toPreviewDataUrl(blob) {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                URL.revokeObjectURL(url);
                try {
                    const w0 = img.naturalWidth, h0 = img.naturalHeight;
                    if (!w0 || !h0) { reject(new Error("That file is not a picture.")); return; }
                    const k = Math.min(1, PREVIEW_MAX / Math.max(w0, h0));
                    const w = Math.max(1, Math.round(w0 * k)), h = Math.max(1, Math.round(h0 * k));
                    const c = document.createElement("canvas");
                    c.width = w; c.height = h;
                    const ctx = c.getContext("2d");
                    // jpeg 无透明通道，带 alpha 的 png 会落在黑色上。先铺面板
                    // 自己的深色底。
                    ctx.fillStyle = "#1d1d1d";
                    ctx.fillRect(0, 0, w, h);
                    ctx.drawImage(img, 0, 0, w, h);
                    resolve(c.toDataURL("image/jpeg", 0.9));
                } catch (err) { reject(err); }
            };
            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error("That file is not a picture the browser can show."));
            };
            img.src = url;
        });
    }

    // 碰活元素而非重渲染：这里整段 renderBody() 会在每次保存时重建两次
    // 头部，毫无可见收益。提示必须跟着 class 走——busy 期间的重渲染会在
    // class 移掉后把 "Saving…" 留成悬停标签。
    function setBusy(v) {
        _busy = !!v;
        const t = panel.querySelector(".sf-ls-info-th");
        if (!t) return;
        t.classList.toggle("busy", _busy);
        t.dataset.hint = hintFor();
    }

    function hintFor() { return _busy ? "Saving…" : (thumb() ? "Change" : "+ Picture"); }

    /**
     * 加载这个 LoRA 的 info 并采纳——除非此后已有更新的加载开始，则这个
     * 答案已过时，必须丢弃。
     *
     * 本面板每次 info 加载都取票，因为可能同时多个在飞且乱序落地：点 ↻
     * Civitai 起一个，随后丢图又起一个，第二个先答时慢的 Civitai 答案会
     * 覆盖它——把 `custom_preview:false` 放回去，图和它的 ✕ 消失，不重开
     * 面板没有找回途径。`panel.isConnected` 单独挡不住：两个请求属于同一
     * 个活面板。
     *
     * `.stale`（来自 api 模块）是不同信号，两者都需要：它表示数据已知过时
     * （请求在途时有人 invalidate 了这个 LoRA），无论本面板是否起了更新的。
     */
    let _infoSeq = 0;
    let _hydrated = false;
    // 封面静默恢复已尝试（防重复请求；失败也停，下次打开面板再试）。
    let _thumbRestoreTried = false;

    async function attemptInfo(force) {
        const ticket = ++_infoSeq;
        const j = await loraInfo(name, force);
        if (!panel.isConnected) return "dead";
        if (ticket !== _infoSeq) return "superseded";   // 有更新的持有答案
        if (!j?.ok || !j.info) return "failed";
        if (j.stale) return "stale";                    // 已知过时，不画
        info = j.info;
        // 每面板恰一次，在最先成功的加载上。只绑第一次加载曾丢它：那次加载
        // 回来 stale/superseded 时，hydrateCustom——没有其它调用点——永不跑，
        // 用户存储的触发词到不了行。
        if (!_hydrated) { _hydrated = true; hydrateCustom(); }
        // 封面静默恢复：自动保存的封面以路径 hash 命名，文件移动/改名后
        // hash 失配、本地找不到，但侧车（跟随文件）里有同一张缩略图——
        // 静默重下载到新 hash 名下，不打扰用户。失败静默（下次打开再试）。
        if (info.restorable_thumb && !_thumbRestoreTried && name) {
            _thumbRestoreTried = true;
            saveCivitaiThumb(name).then((res) => {
                if (!panel.isConnected || !res?.ok) return;
                _thumbBust = res.v || Date.now();
                loadInfo({ force: true }).then((ok) => { if (ok) renderBody(); });
            });
        }
        return "ok";
    }

    /** 加载这个 LoRA 的 info 并采纳。stale 答案再问一次，仅一次。 */
    async function loadInfo({ force = false } = {}) {
        const r = await attemptInfo(force);
        // "stale" 表示请求在途时有人 invalidate 了这个 LoRA。丢弃是对的；
        // 停在这里不对，因为某些 invalidator（保存自定义词）后没有别的东西
        // 会刷新我们。所以强制再问一次，恰好一次——永不循环。
        if (r === "stale") return await attemptInfo(true);
        return r;
    }

    async function reloadInfo() {
        await loadInfo({ force: true });
        if (!panel.isConnected) return;
        renderBody();
    }

    async function applyPicture(blob) {
        if (!name || _busy) return;
        if (!blob || !/^image\//.test(blob.type || "")) {
            showMsg("That is not a picture. Use a jpg, png or webp.");
            return;
        }
        setBusy(true);
        try {
            const dataUrl = await toPreviewDataUrl(blob);
            const res = await saveLoraPreview(name, dataUrl);
            if (!panel.isConnected) return;
            if (!res?.ok) { showMsg(res?.message || "Could not save that picture."); return; }
            _msg = null;
            _thumbBust = res.v || Date.now();
            await reloadInfo();          // custom_preview / has_preview / preview_v
        } catch (err) {
            if (panel.isConnected) showMsg(String(err?.message || err));
        } finally {
            setBusy(false);
        }
    }

    async function removePicture() {
        if (!name || _busy) return;
        setBusy(true);
        try {
            const res = await deleteLoraPreview(name);
            if (!panel.isConnected) return;
            if (!res?.ok) { showMsg(res?.message || "Could not remove that picture."); return; }
            _msg = null;
            _thumbBust = Date.now();     // 回到自动图，越过 1h 缓存
            await reloadInfo();
        } finally {
            setBusy(false);
        }
    }

    function pickPicture() {
        if (_busy) return;
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept = "image/*";
        inp.style.display = "none";
        document.body.appendChild(inp);
        inp.addEventListener("change", () => {
            const f = inp.files && inp.files[0];
            inp.remove();
            if (f) applyPicture(f);
        });
        // 取消对话框不发 `change`，输入框会永远坐在 body 里。窗口重获焦点是
        // 两条路径共享的信号。
        window.addEventListener("focus", () => setTimeout(() => inp.remove(), 800), { once: true });
        inp.click();
    }

    /** 拖进来的 URL 而非文件——比如从 ComfyUI 自己的输出预览拖的图。
     *  仅同源：其它东西会让我们替用户抓任意站点，且跨源图污染 canvas，
     *  toDataURL 也读不回来。 */
    async function pictureFromUrl(url) {
        // 下面的 fetch 是无 busy 标记的 await，没有这个守卫 URL 拖放和文件
        // 拖放会同时被接受并以错误顺序落地。
        if (_busy) return;
        let u = null;
        try { u = new URL(url, window.location.href); } catch { u = null; }
        if (!u || u.origin !== window.location.origin) {
            showMsg("Drag the picture file itself from your computer, or copy the image and paste it here.");
            return;
        }
        try {
            const r = await fetch(u.href);
            if (!r.ok) throw new Error("Could not read that image.");
            await applyPicture(await r.blob());
        } catch (err) {
            if (panel.isConnected) showMsg(String(err?.message || err));
        }
    }

    /**
     * 拖放落在面板任意处，不只是 64px 小框。
     *
     * 实测：文件落在框旁 10px 就完全离开我们的代码，冒泡到 document，
     * ComfyUI 把它变成画布上的 Load Image 节点（节点数 2 -> 3）。一个邀请
     * 用户往 340px 面板里的 64px 目标拖文件的功能必须接住擦边球。
     *
     * 所以面板吞掉落在自己身上的每个 drop：是图且图开着就路由给图片框，
     * 否则直接取消。在持久面板元素上只接一次（renderBody 重建其子元素），
     * 先于框自己的处理器看到任何东西；框仍 stopPropagation，正中目标不会
     * 被处理两次。
     */
    function wirePanelDrop(p) {
        // 光标必须如实承诺 drop 会做什么。用不同条件门控两者曾显示 "copy"
        // 光标在无 LoRA 的行上，放开时却什么都不做——静默空操作比拒绝更糟。
        const takes = () => thumbsOn() && !!name;
        p.addEventListener("dragover", (ev) => {
            ev.preventDefault();               // "这里允许 drop"，即不是画布
            if (ev.dataTransfer) ev.dataTransfer.dropEffect = takes() ? "copy" : "none";
        });
        p.addEventListener("drop", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            if (!takes()) return;               // 吞掉，刻意：没有图位可给
            const f = ev.dataTransfer?.files && ev.dataTransfer.files[0];
            if (f) { applyPicture(f); return; }
            // 不是文件。说清图放哪——但释放落在 add-a-word 字段时不提示：
            // showMsg 重渲染，renderBody 新建那个输入框，打了一半的词会为
            // 一次误拖被扔掉。
            if (!ev.target.closest?.(".sf-ls-addtrig")) {
                showMsg("Drop the picture onto the small box at the top left.");
            }
        });
    }

    function wireThumb(th, hasOwn) {
        th.dataset.hint = hintFor();
        if (_busy) th.classList.add("busy");
        th.title = "Click to use your own picture for this LoRA. You can also drop an image "
            + "here, or copy one and press Ctrl+V.";
        th.addEventListener("click", (ev) => {
            if (ev.target.closest(".sf-ls-thx")) return;   // 移除徽章有自己的活
            pickPicture();
        });
        if (hasOwn) {
            const rm = el("span", "sf-ls-thx", "✕");
            rm.title = "Remove your picture (the automatic one comes back)";
            rm.addEventListener("click", (ev) => { ev.stopPropagation(); removePicture(); });
            th.appendChild(rm);
        }
        const stop = (ev) => { ev.preventDefault(); ev.stopPropagation(); };
        th.addEventListener("dragenter", (ev) => { stop(ev); th.classList.add("drop"); });
        th.addEventListener("dragover", (ev) => {
            stop(ev);
            if (ev.dataTransfer) ev.dataTransfer.dropEffect = "copy";
            th.classList.add("drop");
        });
        th.addEventListener("dragleave", (ev) => {
            // relatedTarget 仍在框内的 leave 是光标掠过覆盖层或 ✕，不是离开——
            // 没有这个高亮会闪烁。
            if (ev.relatedTarget && th.contains(ev.relatedTarget)) return;
            th.classList.remove("drop");
        });
        th.addEventListener("drop", (ev) => {
            stop(ev);
            th.classList.remove("drop");
            const dt = ev.dataTransfer;
            const f = dt?.files && dt.files[0];
            if (f) { applyPicture(f); return; }
            const raw = (dt?.getData("text/uri-list") || dt?.getData("text/plain") || "").trim();
            const first = raw.split(/[\r\n]+/).find((x) => x && !x.startsWith("#"));
            if (first) pictureFromUrl(first);
            else showMsg("Nothing to use there. Drop an image file.");
        });
    }

    function showMsg(m) { _msg = m || null; renderBody(); }

    function msgStrip() {
        const strip = el("div", "sf-ls-strip offline");
        strip.append(el("span", "st-ic", "!"), el("div", null, _msg));
        return strip;
    }

    function renderBody() {
        panel.innerHTML = "";
        const sel = selected();
        const civitaiOn = prefsOf().civitai !== false; // 重读，齿轮切换不陈旧

        // ── 头部 ───────────────────────────────────────────────────────────
        const top = el("div", "sf-ls-info-top");
        // 图片框——仅当齿轮开了图且行真有 LoRA。无名行（加行后关掉选择器）
        // 没有可设图的对象，框是死开关：一直保持指针光标和悬停变暗显示空
        // 标签，因为提示只由 wireThumb 写入。
        let th = null;
        if (thumbsOn() && name) {
            th = el("div", "sf-ls-info-th");
            const turl = thumb();
            // 剥引号/反斜杠，Civitai 图 URL 里的杂字符不能弄坏 CSS url() 值
            // （thumbUrl 已百分号编码；civ.info.thumbnail 是原始串）。
            if (turl) th.style.backgroundImage = `url("${String(turl).replace(/["\\]/g, "")}")`;
            // 选/拖/粘自己的图，再移除。每次渲染都接，因为 renderBody 每次
            // 建新头部。
            wireThumb(th, !!info.custom_preview);
        }
        const h = el("div", "sf-ls-info-h");
        const title = el("h3", null, (civ?.state === "found" && civ.info?.name) || info.title || "LoRA");
        const metaBits = [];
        if (info.base_model) metaBits.push(info.base_model);
        if (info.rank) metaBits.push("rank " + info.rank + (info.alpha ? " / α" + info.alpha : ""));
        if (info.num_images) metaBits.push(info.num_images + " imgs");
        if (info.date) metaBits.push(String(info.date).slice(0, 10));
        const meta = el("div", "sf-ls-info-meta");
        meta.innerHTML = (metaBits.length ? escapeHtml(metaBits.join(" · ")) : "&nbsp;") +
            "<br>" + escapeHtml(name || "");
        h.append(title, meta);
        // 知道 id 时链接到 Civitai 模型页。两个 id 取同一来源（实时查询，
        // 否则离线/缓存 info），model+version 对不能跨源混搭。
        const idSrc = (civ?.state === "found") ? civ.info : info;
        const mid = idSrc?.model_id;
        const vid = idSrc?.version_id;
        if (mid != null) {
            const link = el("span", "sf-ls-civlink", "View on Civitai ↗");
            link.addEventListener("click", () => {
                // 按账户主机偏好选网页域：red 用户看 civitai.red（成人模型在
                // com 网页可能受限）。idSrc 取自实时查询/离线 info，host 恒读
                // 面板 info（api_lora_info 附加）。
                const host = info.civitai_host === "red" ? "civitai.red" : "civitai.com";
                const u = "https://" + host + "/models/" + mid + (vid ? "?modelVersionId=" + vid : "");
                window.open(u, "_blank", "noopener");
            });
            h.appendChild(link);
        }
        const x = el("span", "sf-ls-info-x", "✕");
        x.addEventListener("click", closeInfoPanel);
        if (th) top.append(th, h, x); else top.append(h, x);
        panel.appendChild(top);

        // 中间滚动容器：头部与 footer 固定，内容随面板高度滚动（用户可
        // 拖拽右下角手柄调整面板大小）。消息/状态/迁移条、触发词都进
        // bodyWrap——漏掉一个会在面板拉矮时被 overflow:hidden 裁剪。
        // Description 是例外：它要随面板拉高弹性伸缩，放在 bodyWrap 之外
        // 作为面板的直接 flex 子项（见 panel.appendChild(dsec)）。
        const bodyWrap = el("div", "sf-ls-info-body");

        // ── 图片的问题（如果有）────────────────────────────────────────────
        if (_msg) bodyWrap.appendChild(msgStrip());

        // ── 可选 Civitai 状态条 ────────────────────────────────────────────
        if (civ) bodyWrap.appendChild(civStrip());

        // ── 文件已不存在（改名/移动后旧路径行）：数据在旧 key 下，无法
        // 迁移（迁移端点需要文件存在）——提示用户重新选择 LoRA 路径。──────
        if (info._file_missing && info.orphan_key && !_orphanDismissed && name) {
            const strip = el("div", "sf-ls-strip nofind");
            strip.append(el("span", "st-ic", "↻"));
            const body = el("div");
            body.textContent = "This LoRA file was moved or renamed. Saved data is under the old path ("
                + info.orphan_key + "). Pick this LoRA again from the list to read it.";
            const acts = el("div", "sf-ls-strip-acts");
            const dis = el("button", null, "Dismiss");
            dis.title = "Hide this notice for this panel session";
            dis.addEventListener("click", () => { _orphanDismissed = true; renderBody(); });
            acts.append(dis);
            strip.append(body, acts);
            bodyWrap.appendChild(strip);
        }

        // ── 孤儿数据迁移提示（文件移动/改名后旧键数据仍在）────────────────
        if (info.orphan_key && !info._file_missing && !_orphanDismissed && name) {
            const parts = [];
            if ((info.orphan_triggers?.length || 0) > 0) {
                parts.push(info.orphan_triggers.length + " trigger word" + (info.orphan_triggers.length > 1 ? "s" : ""));
            }
            if (info.orphan_description) parts.push("a description");
            if (info.orphan_preview) parts.push("a preview picture");
            const strip = el("div", "sf-ls-strip nofind");
            strip.append(el("span", "st-ic", "↻"));
            const body = el("div");
            body.textContent = "Found saved data under the old path ("
                + (parts.join(", ") || "saved data") + "). Migrate it to this file?";
            const acts = el("div", "sf-ls-strip-acts");
            const mig = el("button", "pri", "Migrate");
            mig.title = "Move the words, description and preview from the old path to this file";
            mig.addEventListener("click", () => runMigrate());
            const dis = el("button", null, "Dismiss");
            dis.title = "Leave the old data where it is";
            dis.addEventListener("click", () => { _orphanDismissed = true; renderBody(); });
            acts.append(mig, dis);
            strip.append(body, acts);
            bodyWrap.appendChild(strip);
        }

        // ── 触发词 ─────────────────────────────────────────────────────────
        const sec = el("div", "sf-ls-info-sec");
        const head = el("h4");
        head.appendChild(el("span", null, "Trigger words"));
        const all = el("span", "qa", "all");
        all.title = "Select every word";
        all.addEventListener("click", () => setWords(chipList().map((c) => c.w)));
        const none = el("span", "qa", "none");
        none.title = "Clear selection";
        none.addEventListener("click", () => setWords([]));
        // 复制全部触发词到剪贴板（SVG，与样例图按钮统一）
        const copyAll = el("span", "qa");
        copyAll.title = "Copy all trigger words to clipboard";
        copyAll.style.cssText = "display:inline-flex;align-items:center;gap:3px;";
        const copyIc = el("span");
        copyIc.style.cssText = "width:10px;height:10px;background-color:currentColor;display:block;-webkit-mask-size:contain;mask-size:contain;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;";
        const _clipSvg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-2M9 5a2 2 0 002 2h6a2 2 0 002-2M9 5a2 2 0 012-2h4a2 2 0 012 2M9 12h6M9 16h6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E";
        copyIc.style.webkitMaskImage = `url("${_clipSvg}")`;
        copyIc.style.maskImage = `url("${_clipSvg}")`;
        copyAll.appendChild(copyIc);
        copyAll.appendChild(document.createTextNode("copy"));
        copyAll.addEventListener("click", async () => {
            const words = chipList().map((c) => c.w).filter((w) => w);
            if (!words.length) { showMsg("No trigger words to copy."); return; }
            const ok = await copyText(words.join(", "));
            showMsg(ok ? "Trigger words copied to clipboard." : "Could not copy to clipboard.");
        });
        head.append(all, none, copyAll);
        // 源：仅当文件有自己的词且 Civitai 词存在时显示 File / Civitai 切换
        // （以 file_triggers 而非 fileWords() 为门槛——后者回退到合并的侧车
        // 列表，纯侧车 LoRA 会显示一个装着 Civitai 词的假 "File" 标签）。
        // 否则显示普通徽章。
        if (civitaiAvailable() && (info.file_triggers?.length || 0) > 0) {
            const es = effectiveSource();
            const seg = el("div", "sf-ls-srctoggle");
            const fBtn = el("span", "sg" + (es === "file" ? " on" : ""), "File");
            fBtn.title = "Show the LoRA's own words (from the file)";
            fBtn.addEventListener("click", () => { viewSource = "file"; renderBody(); });
            const cBtn = el("span", "sg" + (es === "civitai" ? " on" : ""), "Civitai");
            cBtn.title = "Show the saved Civitai words";
            cBtn.addEventListener("click", () => { viewSource = "civitai"; renderBody(); });
            seg.append(fBtn, cBtn);
            head.appendChild(seg);
        } else {
            const srcBadge = el("span", "src" + (info.source === "civitai" ? " net" : ""),
                civ?.state === "found" ? "from Civitai" : info.source === "sidecar" ? "from Civitai (saved)" : "from file");
            head.appendChild(srcBadge);
        }
        sec.appendChild(head);
        sec.appendChild(el("p", "sf-ls-info-note",
            "Tap the ones you want. Only these, and only if the LoRA is on, reach the triggers output."));

        const chips = el("div", "sf-ls-chips");
        const list = chipList();
        if (!list.length) {
            chips.appendChild(el("span", "sf-ls-chip-none",
                "No trigger words in this file - add your own below" + (civitaiOn ? ", or try Civitai." : ".")));
        } else {
            for (const { w, isCustom } of list) {
                const c = el("span", "sf-ls-chip" + (sel.has(w.toLowerCase()) ? " sel" : ""));
                c.title = w;                              // 悬停看全文（chips 单行截断）
                c.appendChild(el("span", "ct", w));
                c.addEventListener("click", () => toggleWord(w));
                if (isCustom) {
                    const x = el("span", "cx", "✕");
                    x.title = "Remove this custom word";
                    x.addEventListener("click", (ev) => { ev.stopPropagation(); removeCustom(w); });
                    c.appendChild(x);
                }
                chips.appendChild(c);
            }
        }
        sec.appendChild(chips);

        // ── 添加自己的触发词（持久在这个 LoRA 上）──────────────────────────
        const addRow = el("div", "sf-ls-addtrig");
        const inp = el("input");
        inp.type = "text";
        inp.placeholder = "add your own trigger word…";
        inp.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Enter") { ev.preventDefault(); addCustom(inp.value); }
        });
        const addBtn = el("button", null, "Add");
        addBtn.addEventListener("click", () => addCustom(inp.value));
        addRow.append(inp, addBtn);
        sec.appendChild(addRow);

        bodyWrap.appendChild(sec);

        // ── Description（Civitai 说明 + 自定义覆盖）───────────────────────
        const dsec = el("div", "sf-ls-desc");
        const dhead = el("h4");
        dhead.appendChild(el("span", null, "Description"));
        const dsrc = descSrc();
        if (dsrc) {
            dhead.appendChild(el("span", "src" + (dsrc === "civitai" ? " net" : ""),
                dsrc === "custom" ? "custom" : dsrc === "civitai" ? "from Civitai" : "from file"));
        }
        if (_descEditing) {
            const save = el("span", "qa" + (_descDirty ? " dirty" : ""), "Save");
            save.title = "Save my description";
            save.addEventListener("click", () => saveDesc(_descDraft));
            const cancel = el("span", "qa", "Cancel");
            cancel.title = "Discard changes";
            cancel.addEventListener("click", () => { cancelDescEdit(); renderBody(); });
            dhead.append(save, cancel);
        } else {
            const edit = el("span", "qa", "✏️");
            edit.title = "Write your own description (overrides Civitai / file)";
            edit.addEventListener("click", () => {
                _descBase = shownDesc();
                _descDraft = _descBase;
                _descDirty = false;
                _descEditing = true;
                renderBody();
                setTimeout(() => panel.querySelector(".sf-ls-desc textarea")?.focus(), 0);
            });
            dhead.appendChild(edit);
        }
        dsec.appendChild(dhead);
        if (_descEditing) {
            const ta = document.createElement("textarea");
            ta.value = _descDraft;
            ta.rows = 6;
            ta.placeholder = "write your own description…\nMarkdown supported - upload a sample image and it is inserted as ![alt](sample/xxx.png)";
            installWheelZoomPassthrough(ta); // 输入框滚轮透传(缩放画布/滚动文本, 对齐原生)
            ta.addEventListener("keydown", (ev) => {
                if (ev.ctrlKey || ev.metaKey || ev.altKey) return; // 放行修饰键组合(保存/复制等)
                ev.stopPropagation();
                if (ev.key === "Escape") {
                    ev.preventDefault();
                    if (_descDirty) {
                        // 有未保存修改：误按保护——确认后才丢弃
                        confirmDialog({
                            title: "Discard description changes?",
                            message: "You have unsaved changes to this description. Discard them?",
                            okLabel: "Discard",
                            cancelLabel: "Keep editing",
                            accent,
                        }).then((ok) => {
                            if (!ok || !panel.isConnected) return;
                            cancelDescEdit();
                            renderBody();
                        });
                    } else {
                        cancelDescEdit();
                        renderBody();
                    }
                }
            });
            ta.addEventListener("input", () => {
                _descDraft = ta.value;
                const dirty = _descDraft !== _descBase;
                if (dirty === _descDirty) return;
                _descDirty = dirty;
                // 碰活元素更新 Save 按钮高亮（renderBody 重建时按 _descDirty 重画）
                const save = panel.querySelector(".sf-ls-desc h4 .qa");
                if (save) save.classList.toggle("dirty", dirty);
            });
            dsec.appendChild(ta);

            // 上传示例图（存 <lora>/sample/）+ 图库网格（点击插入 markdown）
            const insertInto = (path) => {
                const cur = panel.querySelector(".sf-ls-desc textarea");
                if (cur) {
                    insertAtCursor(cur, buildSampleMarkdown(path));
                    _descDraft = cur.value;   // 同步草稿（插入后 input 事件也会触发）
                }
            };
            const upRow = el("div", "sf-ls-desc-upload");
            const upBtn = el("button", null, "Upload sample image");
            upBtn.title = "Upload a picture next to this LoRA (sample/ folder) and insert it at the cursor";
            upBtn.addEventListener("click", () => {
                if (!name) { showMsg("Pick a LoRA first."); return; }
                const inp = document.createElement("input");
                inp.type = "file";
                inp.accept = "image/*,video/*";
                inp.style.display = "none";
                document.body.appendChild(inp);
                inp.addEventListener("change", () => {
                    const f = inp.files && inp.files[0];
                    inp.remove();
                    if (f) uploadSample(f, insertInto).then(() => refreshSampleGrid(grid, insertInto));
                });
                // 取消对话框不发 change——窗口重获焦点时清掉孤儿 input。
                window.addEventListener("focus", () => setTimeout(() => inp.remove(), 800), { once: true });
                inp.click();
            });
            upRow.append(upBtn, el("span", "hint",
                "Stored in this LoRA's sample/ folder; referenced as sample/xxx.png (follows the LoRA)."));
            dsec.appendChild(upRow);

            const grid = el("div", "sf-ls-desc-grid");
            dsec.appendChild(grid);
            // 进入编辑态自动加载 sample 图库。必须推迟到 dsec 挂上 panel 之后
            // （refreshSampleGrid 开头有 grid.isConnected 守卫，同步调用时 dsec
            // 尚未被 renderBody 尾部 appendChild，会静默跳过——图库永不加载）。
            queueMicrotask(() => refreshSampleGrid(grid, insertInto));

            if (info.custom_description) {
                const act = el("div", "sf-ls-desc-actions");
                const rm = el("button", "rm", "✕ Remove my description");
                rm.title = "Back to the Civitai / file text";
                rm.addEventListener("click", () => clearDesc());
                act.appendChild(rm);
                dsec.appendChild(act);
            }
        } else {
            const shown = shownDesc();
            if (shown) {
                // 查看态渲染 Markdown（sf_markdown.js：先转义后白名单结构化，
                // 无原始 HTML 通过）；sample/ 相对路径解析为图片 URL。
                // 编辑态仍编辑源码（textarea），保存原文。
                const db = el("div", "sf-ls-desc-body");
                db.innerHTML = renderMarkdown(shown, { resolveRelative: (rel) => resolveSampleUrl(rel, name) });
                db.title = shown;
                dsec.appendChild(db);
            } else {
                dsec.appendChild(el("div", "sf-ls-desc-none",
                    "No description in this file - write your own, or try the Civitai lookup."));
            }
        }
        panel.appendChild(bodyWrap);
        // Description 在 bodyWrap 之外、footer 之前：面板直接 flex 子项，
        // 拉高面板时它弹性占满剩余高度（查看态正文/编辑态 textarea 同步
        // 长高），拉矮时与 bodyWrap 按比例收缩、各自滚动。
        panel.appendChild(dsec);

        // ── footer ──────────────────────────────────────────────────────────
        const foot = el("div", "sf-ls-info-foot");
        // 复制已勾选的触发词（row.triggers = 最终进 triggers 输出的词），
        // 逗号+空格拼接。复用 copyText（navigator.clipboard 在 LAN 明文 http
        // 下缺席时回退 execCommand）。成功闪烁按钮文本，不触发 renderBody。
        const copyBtn = el("div", "b gh", "Copy");
        copyBtn.title = "Copy the selected trigger words to the clipboard";
        copyBtn.addEventListener("click", async () => {
            clearMsg();
            const row = ctx.getRow();
            const words = (row?.triggers || []).filter((w) => w);
            if (!words.length) {
                showMsg("Nothing selected - tap the words you want first.");
                return;
            }
            const ok = await copyText(words.join(", "));
            if (!panel.isConnected) return;
            if (!ok) { showMsg("Could not copy to clipboard."); return; }
            copyBtn.textContent = "Copied";
            setTimeout(() => { if (copyBtn.isConnected) copyBtn.textContent = "Copy"; }, 1500);
        });
        foot.appendChild(copyBtn);
        const done = el("div", "b pri", "Done");
        done.addEventListener("click", closeInfoPanel);
        foot.appendChild(done);
        if (civitaiOn && name) {
            const searching = civ?.state === "searching";
            const cbtn = el("div", "b gh" + (searching ? " dis" : ""), searching ? "Looking up…" : "↻ Civitai");
            if (!searching) cbtn.addEventListener("click", runCivitai);
            foot.appendChild(cbtn);
        }
        // 删除已存 Civitai info（仅当侧车存在）——回到文件自己的词。
        if ((info.sidecar_triggers?.length || 0) > 0) {
            const del = el("div", "b del", "🗑");
            del.title = "Delete the saved Civitai info (back to the file's own words)";
            del.addEventListener("click", runDeleteCivitai);
            foot.appendChild(del);
        }
        panel.appendChild(foot);
        attachResize(panel);
        // 每次重渲染都可能改变面板高度（勾词、加自定义词、切 File/Civitai
        // 视图、查询落地）。在这里统一钳制覆盖所有调用点，footer 永不会被
        // 推出屏幕底部。
        clampIntoView(panel);
    }

    function civStrip() {
        const strip = el("div", "sf-ls-strip " +
            (civ.state === "searching" ? "searching" : civ.state === "found" ? "found"
                : civ.state === "offline" ? "offline" : "nofind"));
        const ic = el("span", "st-ic");
        if (civ.state === "searching") ic.appendChild(el("span", "sf-ls-spin"));
        else ic.textContent = civ.state === "found" ? "✓" : civ.state === "offline" ? "!" : "?";
        const body = el("div");
        if (civ.state === "searching") body.textContent = "Looking up on Civitai… matching this file's fingerprint.";
        else if (civ.state === "found") body.innerHTML = "Found on Civitai. Saved next to the file, so it's instant and offline next time."
            + (civ.note ? " " + escapeHtml(civ.note) : "");
        else if (civ.state === "nofind") body.innerHTML = "Not on Civitai. This exact file isn't in their database (it may be private, renamed, or custom-trained). The words read from the file are still shown.";
        else body.textContent = civ.message || "Couldn't reach Civitai. No connection, or it's busy. Use the file's own words, or try again.";
        strip.append(ic, body);
        return strip;
    }

    async function runCivitai() {
        clearMsg();
        civ = { state: "searching" };
        renderBody();
        const res = await civitaiLookup(name);
        if (!panel.isConnected) return;
        if (res.ok && res.found) {
            civ = { state: "found", info: res.info || {} };
            viewSource = "civitai";               // 找到 -> 视图切到它的词
            // 封面保存结果附在状态条上：成功静默（本地图经 loadInfo 刷新后
            // 自动显示）；被跳过（已有自定义预览）稍后用面板风确认框询问；
            // 失败则提示。
            if (res.thumb_v) _thumbBust = res.thumb_v;
            else if (res.thumb_skipped) civ.note = "Your own preview picture was kept.";
            else if (res.thumb_error) civ.note = "Couldn't save the preview: " + res.thumb_error;
            invalidateInfo(name);
            // 刷新离线 info 让源徽章/缓存 id 反映新侧车，再重绘。走 loadInfo，
            // 慢答案不能覆盖用户在查询进行中设的图。
            loadInfo({ force: true }).then((ok) => { if (ok) renderBody(); });
        } else if (res.reason === "notfound") {
            civ = { state: "nofind" };
        } else {
            civ = { state: "offline", message: res.message };
        }
        renderBody();   // renderBody 重新钳制：状态条让面板变高

        // 已有用户自定义预览时查询不覆盖保存（thumb_skipped）——用面板风
        // 确认框询问是否替换。确认后走独立保存端点（读侧车同一张图下载，
        // 无需重新查询）；取消保留本地图，信息照常更新。
        if (!panel.isConnected || !(res?.ok && res.found && res.thumb_skipped)) return;
        const replace = await confirmDialog({
            title: "Replace your preview picture?",
            message: "This LoRA already has a preview picture you set.\n"
                + "Replace it with the one found on Civitai?",
            okLabel: "Replace",
            cancelLabel: "Keep mine",
            accent,
        });
        if (!panel.isConnected) return;
        if (!replace) { civ.note = "Your own preview picture was kept."; renderBody(); return; }
        const sv = await saveCivitaiThumb(name);
        if (!panel.isConnected) return;
        if (!sv?.ok) {
            civ.note = "Couldn't save the preview: " + ((sv && sv.message) || "unknown error");
        } else {
            _thumbBust = sv.v || Date.now();
            civ.note = "Preview replaced with the Civitai picture.";
            loadInfo({ force: true }).then((ok) => { if (ok) renderBody(); });
        }
        renderBody();
    }

    async function runDeleteCivitai() {
        clearMsg();
        await deleteCivitai(name);
        if (!panel.isConnected) return;
        _thumbBust = Date.now();              // 侧车（因此预览）变了
        invalidateInfo(name);                 // 丢缓存的（侧车味的）info
        civ = null;
        viewSource = "file";                  // 没有可切换的了——显示文件词
        await loadInfo({ force: true });
        if (!panel.isConnected) return;
        renderBody();
    }

    // 迁移旧路径键下的自定义数据到当前文件名（词/描述/预览图）。成功后
    // info 刷新（custom_triggers/custom_description 出现、orphan 字段消失）。
    async function runMigrate() {
        clearMsg();
        const res = await migrateLoraData(name, info.orphan_key);
        if (!panel.isConnected) return;
        if (!res?.ok) { showMsg((res && res.message) || "Nothing to migrate."); return; }
        _msg = null;
        _orphanDismissed = true;
        await loadInfo({ force: true });
        if (!panel.isConnected) return;
        renderBody();
    }

    // 初始画布：先画缓存，再读真实离线信息。放两次：一次现在（fetch 在飞
    // 时面板绝不显示在 0,0），一次内容定稿后（对照真实高度摆正）。
    renderBody();
    place(panel, ctx);
    // 在 await 前接线。面板在下面整个加载期间都在屏上可交互，加载完再接
    // drop 拦截会留下一个窗口：掉在它上面的文件逃到 ComfyUI 变成画布上的
    // Load Image 节点——正是这个东西要阻止的。place() 同理。
    dragBy(panel);
    wirePanelDrop(panel);

    await loadInfo();
    if (!panel.isConnected) return;
    renderBody();
    place(panel, ctx);

    // 点击面板外部 = 离开意图：查看态点击工作流其他位置关闭面板；编辑态
    // （有未保存修改）不关——误关会丢草稿，与 Esc/✕ 的确认保护同对象。
    // 拖动（位移 > 6px）不视为点击；确认框挂在 body、不在面板内，需豁免。
    // onPaste 照常（Ctrl+V 在画布上是粘贴图片节点，面板开着时改为设预览图）。
    // 确认框（.sf-ls-confirm-mask）挂在 body 上，onKey/onPaste 需豁免。
    const onKey = (e) => {
        if (e.target.closest?.(".sf-ls-confirm-mask")) return;
        if (e.key === "Escape") { e.stopPropagation(); closeInfoPanel(); }
    };
    // Ctrl+V 从剪贴板设置 LoRA 的图。CAPTURE 且 stopPropagation：ComfyUI 把
    // 图片粘到 CANVAS 上成为节点——面板开着时那绝不是本意，两者同时发生
    // 比任一更糟。只在真取到图时介入：普通文本粘贴原样放行。
    const onPaste = (e) => {
        if (!panel.isConnected || !name) return;
        // 确认框开着时不动图（用户在问答话框，不是在图框上）。
        if (e.target.closest?.(".sf-ls-confirm-mask")) return;
        // 图关着时没有框，粘贴会设一张看不见也删不掉的图。放行粘贴。
        if (!thumbsOn()) return;
        // 绝不抢指向文本框的粘贴——"add your own trigger word" 字段就在
        // 那里，往它里粘词必须照常工作。
        const t = e.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
        for (const it of (e.clipboardData?.items || [])) {
            if (it.kind !== "file" || !/^image\//.test(it.type || "")) continue;
            const f = it.getAsFile();
            if (!f) continue;
            e.preventDefault();
            e.stopPropagation();
            applyPicture(f);
            return;
        }
    };
    // 外部点击判定：pointerdown 记坐标，click 里比位移区分"点击"与"拖动"
    // （LiteGraph 拖动节点/画布后 mouseup 也会在同一 canvas 上触发 click，
    // 浏览器不检查位移，不判则拖动即误关）。
    let _downX = 0, _downY = 0;
    const onPointerDown = (e) => { _downX = e.clientX; _downY = e.clientY; };
    const onDocClick = (e) => {
        if (!panel.isConnected || _panel !== panel) return;
        if (e.target.closest?.(".sf-ls-confirm-mask")) return;   // 确认框豁免
        if (panel.contains(e.target)) return;                    // 面板内不关
        if (Math.hypot((e.clientX ?? 0) - _downX, (e.clientY ?? 0) - _downY) > 6) return; // 拖动
        if (_descDirty) return;                                  // 编辑态保留草稿
        doCloseInfoPanel();
    };
    setTimeout(() => {
        if (_panel !== panel) return; // 同一 tick 被关/被替换 - 不挂孤儿监听器
        document.addEventListener("keydown", onKey, true);
        document.addEventListener("paste", onPaste, true);
        document.addEventListener("pointerdown", onPointerDown, true);
        document.addEventListener("click", onDocClick, true);
    }, 0);
    _cleanup = () => {
        document.removeEventListener("keydown", onKey, true);
        document.removeEventListener("paste", onPaste, true);
        document.removeEventListener("pointerdown", onPointerDown, true);
        document.removeEventListener("click", onDocClick, true);
    };
}

/**
 * 画布移动时让信息面板跟随其节点——与设置面板相同的循环，因为两者从同一
 * 节点打开、锚定同法，一个跟随一个搁浅会读作"后注意到的那个有 bug"。
 *
 * rAF 循环而非事件：LiteGraph 对变换变化不发任何事件。每帧比三个数字，
 * 空闲成本为零，只在面板打开时运行。用户拖走面板即停。
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
        place(panel, node);
    };
    _followRaf = requestAnimationFrame(tick);
}

function stopFollowing() {
    if (_followRaf != null) cancelAnimationFrame(_followRaf);
    _followRaf = null;
}

function dragBy(panel) {
    // 在持久面板上委托：renderBody 每次重渲染都重建头部，直接给头部元素
    // 接线会在第一次重渲染后死掉。
    panel.addEventListener("pointerdown", (e) => {
        if (!e.target.closest?.(".sf-ls-info-top")) return;
        // 拖动柄里的每个可点东西都必须先在这里退出，不只是 ✕。一旦对面板
        // 设了 setPointerCapture，Chromium 会把该指针的 mouseup——因此 click
        // ——重定向到捕获元素，子元素永远看不到 click，释放捕获也补不回来。
        // 那曾静默杀死 "View on Civitai ↗"：链接在头部，被捕获后不再开页。
        if (e.target.closest(".sf-ls-info-x, .sf-ls-civlink, .sf-ls-info-th")) return;
        const r = panel.getBoundingClientRect();
        const ox = e.clientX - r.left, oy = e.clientY - r.top;

        // 防拖拽粘住光标的两道防线：pointerup 真会丢（窗口外/第二显示器/
        // 被上游吞掉）。
        const handle = e.currentTarget;
        try { handle.setPointerCapture(e.pointerId); } catch { /* 不可捕获 */ }

        const move = (ev) => {
            if (!panel.isConnected) return up();
            if (!(ev.buttons & 1)) return up();
            // 从这里起面板在用户放的地方，停止跟随节点。
            _userMoved = true;
            panel.style.left = Math.max(0, Math.min(window.innerWidth - panel.offsetWidth, ev.clientX - ox)) + "px";
            panel.style.top = Math.max(0, Math.min(window.innerHeight - panel.offsetHeight, ev.clientY - oy)) + "px";
        };
        let done = false;
        const up = () => {
            if (done) return;
            done = true;
            try { handle.releasePointerCapture(e.pointerId); } catch { /* 已离开 */ }
            handle.removeEventListener("pointermove", move, true);
            handle.removeEventListener("pointerup", up, true);
            handle.removeEventListener("pointercancel", up, true);
            handle.removeEventListener("lostpointercapture", up, true);
        };
        handle.addEventListener("pointermove", move, true);
        handle.addEventListener("pointerup", up, true);
        handle.addEventListener("pointercancel", up, true);
        handle.addEventListener("lostpointercapture", up, true);
    });
}
