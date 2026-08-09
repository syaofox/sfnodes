// ==========================================================================
// sf_prompt_tags.js - SFPromptTags 前端扩展
// ==========================================================================
//
// 复刻 Pixaroma Prompt（完整版）：
//   - 多行提示词输入框 + 彩色高亮 backdrop（@tag 橙 / *category 绿 / #list 紫，
//     未知名称红色波浪线）+ @/*/# 自动补全弹窗
//   - 标签库存于 ComfyUI 未注册设置 "sfnodes.PromptTags.Library"（机器私有、
//     跨工作流共享、分享工作流不泄露标签；存储层见 sf_prompt_tags_store.js）
//   - 队列时经 app.graphToPrompt hook 展开 token 并注入隐藏 PromptState 输入
//     （Sliders / Seed 模式）；*wildcard / #list 经 Picks 游标选择
//     （shuffle / random / in order，见 sf_prompt_tags_cursors.js），且仅当
//     queue 真正被接受才推进（queuePrompt patcher + commitPicks）
//   - 可选 text_in 输入：与输入框内容按顺序/分隔符拼接（与后端 run() 逻辑镜像）
//   - "Tags" 按钮打开全屏标签库编辑器（sf_prompt_tags_editor.js）；输入框
//     右键菜单提供 Copy / Save selection as tag
//
// 与原件（js/prompt/index.js）的差异（已确认范围）：无强调色/gear 设置、
// 无拖图回读 lastRun；其余功能对齐。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import {
    scanTokens,
    expandAll,
    tagLines,
    isListTag,
    catOf,
    hasTags,
    hasWilds,
    hasLists,
    tagMode,
    catMode,
} from "./sf_prompt_tags_lib.js";
import {
    getLibrary,
    getTags,
    getCategories,
    findTag,
    subscribe,
} from "./sf_prompt_tags_store.js";
import {
    listKey,
    catKey,
    nextIndex,
    beginPickBuild,
    commitPicks,
} from "./sf_prompt_tags_cursors.js";
import {
    openLibraryEditor,
    closeLibraryEditorFor,
} from "./sf_prompt_tags_editor.js";
import { pinyinMatch } from "./sf_prompt_tags_pinyin.js";

const STATE_KEY = "promptState";
const DEFAULT_STATE = { text: "", order: "mine", sep: ", ", showExpanded: true };

const DEFAULT_W = 470;
const DEFAULT_H = 210;
const MIN_W = 440;
const MIN_H = 172;
const WIDGET_MIN_H = 148;
const TAWRAP_MIN = 44;
const EXPAND_MIN = 30;

const lc = (s) => String(s == null ? "" : s).toLowerCase();

// ── state（node.properties，随工作流保存）───────────────────────────────
function readState(node) {
    const s = (node.properties && node.properties[STATE_KEY]) || {};
    return {
        text: typeof s.text === "string" ? s.text : DEFAULT_STATE.text,
        order: s.order === "wired" ? "wired" : "mine",
        sep: typeof s.sep === "string" ? s.sep : DEFAULT_STATE.sep,
        showExpanded: s.showExpanded !== false,
    };
}
function writeState(node, patch) {
    node.properties = node.properties || {};
    const cur = readState(node);
    node.properties[STATE_KEY] = { ...cur, ...patch };
}

// ── *wildcard / #list 解析 ────────────────────────────────────────────────
// 分类查找（大小写不敏感），返回 {canonical, pool} 或 null（未知/空分类 -> token 保留字面量）
function wildCat(name) {
    const canonical = getCategories().find((c) => c && lc(c) === lc(name));
    if (!canonical) return null;
    const pool = getTags().filter((t) => lc(catOf(t)) === lc(canonical) && typeof t.text === "string");
    return pool.length ? { canonical, pool } : null;
}
// #list 解析：任意标签（含普通文本标签）按行拆分；无可用行 -> null
function listOf(name) {
    const t = findTag(name);
    if (!t || typeof t.text !== "string") return null;
    const lines = tagLines(t.text);
    return lines.length ? { tag: t, lines } : null;
}
// RUN 解析：经 Picks 游标（shuffle / random / in order）选择；抽中 List 标签时
// 再用该列表自己的模式取一行（分类装列表时组合：先选列表，再选一行）。
// nextOcc 是同一个 build 内该 key 第几次使用（#fruit #fruit 各拿一张新牌）。
function pickWild(name, nextOcc) {
    const w = wildCat(name);
    if (!w) return null;
    const ck = catKey(w.canonical);
    const i = nextIndex(ck, w.pool.length, catMode(w.canonical, getLibrary()), nextOcc ? nextOcc(ck) : 0);
    const t = w.pool[i < 0 ? 0 : i];
    if (isListTag(t)) {
        const lines = tagLines(t.text);
        if (lines.length) {
            const lk = listKey(t.name);
            const j = nextIndex(lk, lines.length, tagMode(t), nextOcc ? nextOcc(lk) : 0);
            return lines[j < 0 ? 0 : j];
        }
    }
    return t.text;
}
function pickList(name, nextOcc) {
    const l = listOf(name);
    if (!l) return null;
    const lk = listKey(l.tag.name);
    const i = nextIndex(lk, l.lines.length, tagMode(l.tag), nextOcc ? nextOcc(lk) : 0);
    return l.lines[i < 0 ? 0 : i];
}
// PREVIEW 解析：稳定占位符（真实选择发生在 run 时，预览不能随按键闪烁），
// 但指名模式（random / shuffled / next），一眼知道这个槽会怎么动
const MODE_WORD = { random: "random", shuffle: "shuffled", order: "next" };
function previewWild(name) {
    const w = wildCat(name);
    return w ? `[${MODE_WORD[catMode(w.canonical, getLibrary())]}: ${w.canonical}]` : null;
}
function previewList(name) {
    const l = listOf(name);
    return l ? `[${MODE_WORD[tagMode(l.tag)]} line: ${l.tag.name}]` : null;
}
const PREVIEW_RESOLVERS = { resolveWild: previewWild, resolveList: previewList };
// RUN 解析器按节点展开构建：带 per-use 计数器，使一个盒子里重复的 #list 发新牌；
// 每个节点从 0 起算正是两个 Prompt 节点第一次使用保持同步的原因。
function makeRunResolvers() {
    const used = new Map();
    const nextOcc = (k) => { const n = used.get(k) || 0; used.set(k, n + 1); return n; };
    return {
        resolveWild: (name) => pickWild(name, nextOcc),
        resolveList: (name) => pickList(name, nextOcc),
    };
}
// expandAll 是纯函数（tags 默认空表），调用点统一注入当前库；预览/队列都走这里，
// 保证 @tag 展开、*wildcard / #list 解析两处永不脱节
function expandWith(text, resolvers) {
    return expandAll(text, { ...resolvers, tags: getTags() });
}

// ── CSS ──────────────────────────────────────────────────────────────────
let _cssInjected = false;
function injectCSS() {
    if (_cssInjected) return;
    _cssInjected = true;
    const style = document.createElement("style");
    style.textContent = `
.sf-ptg-root { --acc:var(--sf-acc, #f66744); position:relative; display:flex; flex-direction:column; gap:6px; padding:6px;
  width:100%; height:100%; box-sizing:border-box; color:#e0e0e0; font:12px 'Segoe UI',sans-serif; }
.sf-ptg-portrow { position:absolute; top:-26px; left:0; right:0; margin:0; z-index:3; pointer-events:none;
  display:none; align-items:center; justify-content:center; gap:8px; user-select:none; overflow:hidden; }
.sf-ptg-portrow.on { display:flex; }
.sf-ptg-portrow .cl { font-size:10.5px; color:var(--acc); display:inline-flex; align-items:center; gap:5px; }
.sf-ptg-portrow .cl .wd { width:8px; height:8px; border-radius:50%; background:var(--acc); }
.sf-ptg-seg { pointer-events:auto; display:inline-flex; border:1px solid var(--acc); border-radius:6px; overflow:hidden; background:#1d1d1d; }
.sf-ptg-seg button { background:transparent; border:0; color:var(--acc); padding:4px 9px; font:500 11px 'Segoe UI',sans-serif; cursor:pointer; }
.sf-ptg-seg button:hover { color:#fff; background:rgba(255,255,255,.06); }
.sf-ptg-seg button.on { background:var(--acc); color:#fff; }
.sf-ptg-dd { pointer-events:auto; position:relative; display:inline-flex; }
.sf-ptg-dd-btn { display:inline-flex; align-items:center; gap:6px; background:#1d1d1d; border:1px solid var(--acc);
  border-radius:5px; color:var(--acc); font:11px 'Segoe UI',sans-serif; padding:3px 8px; cursor:pointer; white-space:nowrap; }
.sf-ptg-dd-btn:hover { color:#fff; }
.sf-ptg-dd-btn .car { font-size:9px; opacity:.85; }
.sf-ptg-dd-pop { position:fixed; z-index:10032; background:#1d1d1d; border:1px solid #4a4a4a; border-radius:6px;
  overflow:hidden; box-shadow:0 10px 26px rgba(0,0,0,.55); min-width:120px; }
.sf-ptg-dd-item { padding:6px 11px; cursor:pointer; color:#cfcfcf; font:12px 'Segoe UI',sans-serif; }
.sf-ptg-dd-item:hover, .sf-ptg-dd-item.sel { background:#3a2a24; color:#fff; }
/* 深色背景在 WRAPPER 上，textarea 保持透明以露出 backdrop 的高亮 */
.sf-ptg-tawrap { position:relative; flex:2 1 0; min-height:${TAWRAP_MIN}px; display:flex;
  background:#1d1d1d; border:1px solid #333; border-radius:4px; }
.sf-ptg-tawrap:focus-within { border-color:var(--acc); }
.sf-ptg-backdrop { position:absolute; inset:0; padding:6px 8px; border:0;
  font:12px/1.5 monospace; color:#e0e0e0; white-space:pre-wrap; word-wrap:break-word; overflow:hidden; scrollbar-gutter:stable; pointer-events:none; box-sizing:border-box; }
.sf-ptg-ta { flex:1 1 auto; width:100%; height:100%; box-sizing:border-box; background:transparent; color:transparent;
  border:0; border-radius:4px; padding:6px 8px; font:12px/1.5 monospace; resize:none; outline:none; scrollbar-gutter:stable; caret-color:var(--acc); }
.sf-ptg-ta::placeholder { color:#6a6a6a; }
/* 色相 = token 种类（@tag 橙 / *cat 绿 / #list 紫），明度 = 同种第几个（s1/s2 区分相邻同类）；
   未知名称（含未输完的）白色 + 红色波浪线，像拼写检查一样可读 */
.sf-ptg-chip { color:var(--acc); }
.sf-ptg-chip.s1 { color:color-mix(in srgb, var(--acc) 55%, #ffd27a); }
.sf-ptg-chip.s2 { color:color-mix(in srgb, var(--acc) 25%, #ffe3a2); }
.sf-ptg-wild { color:#4fc98a; }
.sf-ptg-wild.s1 { color:#86d977; }
.sf-ptg-wild.s2 { color:#b6e58d; }
.sf-ptg-list { color:#b98cff; }
.sf-ptg-list.s1 { color:#d79bf0; }
.sf-ptg-list.s2 { color:#efaadf; }
.sf-ptg-chip.bad, .sf-ptg-wild.bad, .sf-ptg-list.bad {
  color:#f0f0f0; text-decoration:underline wavy #ff2d55; text-underline-offset:2px; }
.sf-ptg-expand { flex:1 1 0; background:#2d2d2d; border:1px solid #3a3a3a; border-radius:4px; padding:6px 8px;
  font:11px/1.5 monospace; white-space:pre-wrap; min-height:${EXPAND_MIN}px; overflow-y:auto; color:#d8d8d8; }
.sf-ptg-expand .lbl { color:#6d6d6d; }
.sf-ptg-expand .note { color:#8a8a8a; font-style:italic; }
.sf-ptg-bar { display:flex; align-items:center; flex:0 0 auto; gap:4px; flex-wrap:wrap; row-gap:4px; padding:0 2px; user-select:none; }
.sf-ptg-btn { box-sizing:border-box; user-select:none; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.15);
  border-radius:4px; color:rgba(255,255,255,.85); cursor:pointer; font:11px 'Segoe UI',sans-serif; padding:4px 9px;
  transition:background .1s,color .1s,border-color .1s; display:inline-flex; align-items:center; gap:5px; }
.sf-ptg-btn:hover { background:var(--acc); border-color:var(--acc); color:#fff; }
.sf-ptg-btn[disabled] { color:rgba(255,255,255,.3); cursor:default; background:rgba(255,255,255,.02); border-color:rgba(255,255,255,.08); }
.sf-ptg-btn[disabled]:hover { background:rgba(255,255,255,.02); border-color:rgba(255,255,255,.08); color:rgba(255,255,255,.3); }
.sf-ptg-btn.is-flashing, .sf-ptg-btn.is-flashing:hover { background:#3ec371; border-color:#3ec371; color:#fff; }
.sf-ptg-sw { box-sizing:border-box; display:inline-flex; align-items:center; gap:5px; flex:0 0 auto; user-select:none; white-space:nowrap;
  background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.15); border-radius:4px; color:rgba(255,255,255,.7);
  cursor:pointer; font:11px 'Segoe UI',sans-serif; padding:4px 9px; transition:background .1s,color .1s,border-color .1s; }
.sf-ptg-sw:hover { border-color:var(--acc); color:rgba(255,255,255,.92); }
.sf-ptg-sw-dot { width:8px; height:8px; border-radius:50%; border:1.5px solid rgba(255,255,255,.55); background:transparent; box-sizing:border-box; }
.sf-ptg-sw.on { background:var(--acc); border-color:var(--acc); color:#fff; }
.sf-ptg-sw.on .sf-ptg-sw-dot { background:#fff; border-color:#fff; }
/* @ 自动补全弹窗（挂在 body 上，节点不会裁剪它） */
.sf-ptg-ac { position:fixed; z-index:10030; background:#1d1d1d; border:1px solid #4a4a4a; border-radius:7px;
  overflow-y:auto; max-height:230px; min-width:260px; box-shadow:0 12px 30px rgba(0,0,0,.6);
  font:12px 'Segoe UI',sans-serif; display:none; }
.sf-ptg-ac-h { padding:5px 11px 3px; font:600 9.5px 'Segoe UI',sans-serif; letter-spacing:.1em; text-transform:uppercase; color:#767676;
  display:flex; align-items:center; gap:6px; border-top:1px solid #262626; }
.sf-ptg-ac-h:first-child { border-top:none; }
.sf-ptg-ac-h .cd { width:8px; height:8px; border-radius:50%; }
.sf-ptg-ac-i { padding:6px 11px; cursor:pointer; }
.sf-ptg-ac-i.sel, .sf-ptg-ac-i:hover { background:#3a2a24; }
.sf-ptg-ac-n { font:12px monospace; color:var(--acc, #f66744); }
.sf-ptg-ac-i.wild .sf-ptg-ac-n, .sf-ptg-ac-i.list .sf-ptg-ac-n { color:#b98cff; }
.sf-ptg-ac-d { font-size:10.5px; color:#767676; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:320px; }
.sf-ptg-ac-empty { padding:9px 11px; color:#767676; font-size:11.5px; }
`;
    document.head.appendChild(style);
}

// ── 小工具 ────────────────────────────────────────────────────────────────
function toast(severity, msg) {
    const t = app?.extensionManager?.toast;
    if (t?.add) t.add({ severity, summary: "SF Prompt Tags", detail: msg, life: 2200 });
    else console.warn("[sfnodes.PromptTags]", msg);
}
function flashBtnText(btn, label) {
    if (btn._sfFlashTimer) clearTimeout(btn._sfFlashTimer);
    else btn._sfFlashOrig = btn.textContent;
    btn.textContent = label;
    btn.classList.add("is-flashing");
    btn._sfFlashTimer = setTimeout(() => {
        btn.textContent = btn._sfFlashOrig;
        btn.classList.remove("is-flashing");
        btn._sfFlashTimer = null;
    }, 700);
}
function escapeHTML(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function catColor(name) {
    // 两个桶不是真实分类——中性灰
    if (!name || name === "Text" || name === "List") return "#7a7a7a";
    const i = getCategories().indexOf(name);
    if (i < 0) return "#7a7a7a";
    const PAL = ["#e0894b", "#5aa9e6", "#8e7bd6", "#5fbf8f", "#d76b98", "#c9a24b", "#6fb3b8"];
    return PAL[i % PAL.length];
}

// 深色自定义下拉（绝不使用原生白底 select）。返回 { el, set(value) }
let _ddPop = null;
let _ddOutside = null;
function closeDD() {
    if (_ddPop) { _ddPop.remove(); _ddPop = null; }
    if (_ddOutside) {
        document.removeEventListener("mousedown", _ddOutside, true);
        document.removeEventListener("pointerdown", _ddOutside, true);
        document.removeEventListener("wheel", _ddOutside, true);
        document.removeEventListener("keydown", _ddEsc, true);
        _ddOutside = null;
    }
}
function _ddEsc(e) { if (e.key === "Escape") closeDD(); }
function makeDropdown(value, options, onChange) {
    const wrap = document.createElement("div");
    wrap.className = "sf-ptg-dd";
    const btn = document.createElement("div");
    btn.className = "sf-ptg-dd-btn";
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    const car = document.createElement("span");
    car.className = "car";
    car.textContent = "▾";
    btn.append(lbl, car);
    wrap.appendChild(btn);
    let cur = value;
    const labelOf = (v) => { const o = options.find((o) => o.value === v); return o ? o.label : v; };
    const set = (v) => { cur = v; lbl.textContent = labelOf(v); };
    set(value);
    btn.addEventListener("mousedown", (e) => e.stopPropagation());
    btn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (_ddPop) { closeDD(); return; }
        const pop = document.createElement("div");
        pop.className = "sf-ptg-dd-pop";
        for (const o of options) {
            const it = document.createElement("div");
            it.className = "sf-ptg-dd-item" + (o.value === cur ? " sel" : "");
            it.textContent = o.label;
            it.addEventListener("mousedown", (ev) => { ev.preventDefault(); ev.stopPropagation(); set(o.value); onChange(o.value); closeDD(); });
            pop.appendChild(it);
        }
        document.body.appendChild(pop);
        _ddPop = pop;
        const r = btn.getBoundingClientRect();
        pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - pop.offsetWidth - 8)) + "px";
        const below = window.innerHeight - r.bottom;
        const top = (below < pop.offsetHeight + 8 && r.top > below) ? r.top - pop.offsetHeight - 4 : r.bottom + 4;
        pop.style.top = Math.max(8, Math.min(top, window.innerHeight - pop.offsetHeight - 8)) + "px";
        _ddOutside = (ev) => { if (!pop.contains(ev.target) && !btn.contains(ev.target)) closeDD(); };
        setTimeout(() => {
            document.addEventListener("mousedown", _ddOutside, true);
            document.addEventListener("pointerdown", _ddOutside, true);
            document.addEventListener("wheel", _ddOutside, true);
            document.addEventListener("keydown", _ddEsc, true);
        }, 0);
    });
    return { el: wrap, set };
}
const SEP_OPTIONS = [
    { value: ", ", label: ", comma" },
    { value: " ", label: "space" },
    { value: "\n", label: "new line" },
    { value: "\n\n", label: "blank line" },
    { value: " | ", label: "| pipe" },
    { value: ". ", label: ". period" },
    { value: " BREAK ", label: "BREAK" },
];

// ── @ 自动补全（body 级单例弹窗）────────────────────────────────────────
// token 名支持 Unicode 字母/数字/-/_（含中文）
const TAG_TOKEN_RE = /@([\p{L}\p{N}_\-]*)$/u;
const WILD_TOKEN_RE = /\*([\p{L}\p{N}_\-]*)$/u;
const LIST_TOKEN_RE = /#([\p{L}\p{N}_\-]*)$/u;
let _acEl = null;
let _ac = null; // { node, ta, start, items, sel }

function acPopup() {
    if (_acEl) return _acEl;
    _acEl = document.createElement("div");
    _acEl.className = "sf-ptg-ac";
    document.body.appendChild(_acEl);
    return _acEl;
}
function closeAC() {
    if (_acEl) _acEl.style.display = "none";
    _ac = null;
}
let _acSelInstalled = false;
function installACSelWatch() {
    if (_acSelInstalled) return;
    _acSelInstalled = true;
    document.addEventListener("selectionchange", () => {
        if (!_ac) return;
        const ta = _ac.ta;
        if (!ta || document.activeElement !== ta) { closeAC(); return; }
        maybeAC(_ac.node, ta);
    });
}
function maybeAC(node, ta) {
    const pos = ta.selectionStart;
    const upto = ta.value.slice(0, pos);
    // 边界与 lib scanTokens 一致：前字符为拉丁/数字等阻断类时不弹（email 等），
    // 中文等其它文字之前照常弹（"画@" 输入中文标签）
    const mt = TAG_TOKEN_RE.exec(upto);
    if (mt) {
        const start = pos - mt[0].length;
        const prev = prevCodePointOf(ta.value, start);
        if (prev && /[\p{Script=Latin}\p{Script=Greek}\p{Script=Cyrillic}\p{N}\p{M}_@]/u.test(prev)) { closeAC(); return; }
        openAC(node, ta, start, mt[1].toLowerCase(), "tag");
        return;
    }
    const mw = WILD_TOKEN_RE.exec(upto);
    if (mw) {
        const start = pos - mw[0].length;
        const prev = prevCodePointOf(ta.value, start);
        if (prev && /[\p{Script=Latin}\p{Script=Greek}\p{Script=Cyrillic}\p{N}\p{M}_*]/u.test(prev)) { closeAC(); return; }
        openAC(node, ta, start, mw[1].toLowerCase(), "wild");
        return;
    }
    const ml = LIST_TOKEN_RE.exec(upto);
    if (ml) {
        const start = pos - ml[0].length;
        const prev = prevCodePointOf(ta.value, start);
        if (prev && /[\p{Script=Latin}\p{Script=Greek}\p{Script=Cyrillic}\p{N}\p{M}_#]/u.test(prev)) { closeAC(); return; }
        openAC(node, ta, start, ml[1].toLowerCase(), "list");
        return;
    }
    closeAC();
}
function prevCodePointOf(text, at) {
    if (!(at > 0)) return "";
    const c = text[at - 1];
    if (at >= 2 && c >= "\uDC00" && c <= "\uDFFF") {
        const hi = text[at - 2];
        if (hi >= "\uD800" && hi <= "\uDBFF") return hi + c;
    }
    return c;
}
function openAC(node, ta, start, q, mode) {
    installACSelWatch();
    const el = acPopup();
    el.innerHTML = "";
    const flat = [];
    const sym = mode === "wild" ? "*" : mode === "list" ? "#" : "@";

    if (mode === "list") {
        const lists = getTags()
            .filter((t) => isListTag(t) && pinyinMatch(t.name, q))
            .map((t) => ({ name: t.name, cat: catOf(t), lines: tagLines(t.text) }))
            .filter((t) => t.lines.length > 0);
        if (!lists.length) {
            const e = document.createElement("div");
            e.className = "sf-ptg-ac-empty";
            e.textContent = q ? `No list matches "#${q}".` : "No lists yet. Open Tags, then switch a tag to List and put one option per line.";
            el.appendChild(e);
        } else {
            const byCat = new Map();
            for (const t of lists) { if (!byCat.has(t.cat)) byCat.set(t.cat, []); byCat.get(t.cat).push(t); }
            for (const c of getCategories("list").filter((c) => byCat.has(c))) {
                const h = document.createElement("div");
                h.className = "sf-ptg-ac-h";
                h.innerHTML = `<span class="cd" style="background:${catColor(c)}"></span>${escapeHTML(c)}`;
                el.appendChild(h);
                for (const t of byCat.get(c)) {
                    const idx = flat.length;
                    flat.push({ name: t.name });
                    const d = document.createElement("div");
                    d.className = "sf-ptg-ac-i list" + (idx === 0 ? " sel" : "");
                    d.dataset.i = String(idx);
                    d.innerHTML = `<div class="sf-ptg-ac-n">#${escapeHTML(t.name)}</div>` +
                        `<div class="sf-ptg-ac-d">${t.lines.length} option${t.lines.length === 1 ? "" : "s"} · ${escapeHTML(t.lines.slice(0, 3).join(" · "))}</div>`;
                    d.addEventListener("mousedown", (e) => { e.preventDefault(); pickAC(flat[idx]); });
                    el.appendChild(d);
                }
            }
        }
    } else if (mode === "wild") {
        // 仅列出能作为单个 *token 输入的分类（名称须为纯 [\p{L}\p{N}_-]，
        // 中文分类名可作为 *中文分类）且有标签的分类
        const cats = getCategories()
            .filter((c) => c && pinyinMatch(c, q) && /^[\p{L}\p{N}_\-]+$/u.test(c))
            .map((c) => ({ name: c, count: (wildCat(c)?.pool.length) || 0 }))
            .filter((c) => c.count > 0);
        if (!cats.length) {
            const e = document.createElement("div");
            e.className = "sf-ptg-ac-empty";
            e.textContent = q ? `No category matches "*${q}".` : "No categories with tags yet. Open Tags to add one.";
            el.appendChild(e);
        } else {
            const h = document.createElement("div");
            h.className = "sf-ptg-ac-h";
            h.innerHTML = `<span class="cd" style="background:#b98cff"></span>random from category`;
            el.appendChild(h);
            for (const c of cats) {
                const idx = flat.length;
                flat.push({ name: c.name });
                const d = document.createElement("div");
                d.className = "sf-ptg-ac-i wild" + (idx === 0 ? " sel" : "");
                d.dataset.i = String(idx);
                d.innerHTML = `<div class="sf-ptg-ac-n">*${escapeHTML(c.name)}</div><div class="sf-ptg-ac-d">${c.count} tag${c.count === 1 ? "" : "s"} · random each run</div>`;
                d.addEventListener("mousedown", (e) => { e.preventDefault(); pickAC(flat[idx]); });
                el.appendChild(d);
            }
        }
    } else {
        // @tags：仅文本标签（List 标签归 #），按分类分组
        const tags = getTags().filter((t) => !isListTag(t) && pinyinMatch(t.name, q));
        const byCat = new Map();
        for (const t of tags) {
            const c = catOf(t);
            if (!byCat.has(c)) byCat.set(c, []);
            byCat.get(c).push(t);
        }
        if (!tags.length) {
            const e = document.createElement("div");
            e.className = "sf-ptg-ac-empty";
            e.textContent = q ? `No text tag matches "@${q}". Type # for your lists, or open Tags to add one.` : "No tags yet. Open Tags to add one.";
            el.appendChild(e);
        } else {
            // 组头顺序 = 该侧分类顺序 + 桶（Text），与编辑器侧栏一致
            const order = getCategories("text").filter((c) => byCat.has(c));
            for (const c of order) {
                const h = document.createElement("div");
                h.className = "sf-ptg-ac-h";
                h.innerHTML = `<span class="cd" style="background:${catColor(c)}"></span>${escapeHTML(c)}`;
                el.appendChild(h);
                for (const t of byCat.get(c)) {
                    const idx = flat.length;
                    flat.push(t);
                    const d = document.createElement("div");
                    d.className = "sf-ptg-ac-i" + (idx === 0 ? " sel" : "");
                    d.dataset.i = String(idx);
                    d.innerHTML = `<div class="sf-ptg-ac-n">@${escapeHTML(t.name)}</div><div class="sf-ptg-ac-d">${escapeHTML(t.text)}</div>`;
                    d.addEventListener("mousedown", (e) => { e.preventDefault(); pickAC(flat[idx]); });
                    el.appendChild(d);
                }
            }
        }
    }

    _ac = { node, ta, start, items: flat, sel: 0, sym };
    const r = ta.getBoundingClientRect();
    el.style.display = "block";
    el.style.minWidth = Math.max(260, Math.min(360, r.width)) + "px";
    const below = window.innerHeight - r.bottom;
    const need = Math.min(el.offsetHeight || 230, 230);
    el.style.left = Math.max(8, Math.min(r.left, window.innerWidth - el.offsetWidth - 8)) + "px";
    if (below < need && r.top > below) { el.style.top = ""; el.style.bottom = (window.innerHeight - r.top + 4) + "px"; }
    else { el.style.bottom = ""; el.style.top = (r.bottom + 4) + "px"; }
}
function updateACSel() {
    if (!_acEl) return;
    _acEl.querySelectorAll(".sf-ptg-ac-i").forEach((c) => c.classList.toggle("sel", +c.dataset.i === _ac.sel));
    const sel = _acEl.querySelector(".sf-ptg-ac-i.sel");
    if (sel) sel.scrollIntoView({ block: "nearest" });
}
// 前导空格：仅当前字符为拉丁/数字/下划线/组合标记（或刚插入的符号）时才加空格，
// 避免 @a@b 挤在一起；中文前后不插空格（"画@水彩" 是中文习惯）
function tagSep(before) {
    return (before && /[a-zA-Z0-9_\p{Script=Latin}\p{Script=Greek}\p{Script=Cyrillic}\p{M}@*#]$/u.test(before)) ? " " : "";
}
// 尾随空格：让继续输入成为独立单词而非延长标签名（同样仅拉丁/数字语境）
function tagTrail(after) {
    return (!after || /^[a-zA-Z0-9_\p{Script=Latin}\p{Script=Greek}\p{Script=Cyrillic}\p{M}@*#]/u.test(after)) ? " " : "";
}
function pickAC(item) {
    if (!_ac) return;
    const { node, ta, start, sym } = _ac;
    const v = ta.value;
    const before = v.slice(0, start);
    const after = v.slice(ta.selectionStart);
    const ins = tagSep(before) + sym + item.name + tagTrail(after);
    ta.value = before + ins + after;
    const p = (before + ins).length;
    ta.selectionStart = ta.selectionEnd = p;
    closeAC();
    ta.focus();
    writeState(node, { text: ta.value });
    refreshBody(node);
}
document.addEventListener("mousedown", (e) => {
    if (_acEl && _acEl.style.display === "block" && !_acEl.contains(e.target)) {
        if (!_ac || e.target !== _ac.ta) closeAC();
    }
}, true);
document.addEventListener("wheel", (e) => {
    if (_acEl && _acEl.style.display === "block" && !_acEl.contains(e.target)) closeAC();
}, true);

// ── DOM ──────────────────────────────────────────────────────────────────
function buildRoot(node) {
    const root = document.createElement("div");
    root.className = "sf-ptg-root";

    const portrow = document.createElement("div");
    portrow.className = "sf-ptg-portrow";
    const cl = document.createElement("span");
    cl.className = "cl";
    cl.innerHTML = `<span class="wd"></span>join`;
    const seg = document.createElement("div");
    seg.className = "sf-ptg-seg";
    const bMine = document.createElement("button");
    bMine.type = "button";
    bMine.textContent = "My prompt first";
    bMine.dataset.order = "mine";
    const bWired = document.createElement("button");
    bWired.type = "button";
    bWired.textContent = "Wired first";
    bWired.dataset.order = "wired";
    seg.append(bMine, bWired);
    const sepDD = makeDropdown(readState(node).sep, SEP_OPTIONS, (v) => { writeState(node, { sep: v }); renderExpand(node); });
    sepDD.el.title = "两个提示词之间的分隔符";
    portrow.append(cl, seg, sepDD.el);

    const tawrap = document.createElement("div");
    tawrap.className = "sf-ptg-tawrap";
    const backdrop = document.createElement("div");
    backdrop.className = "sf-ptg-backdrop";
    const ta = document.createElement("textarea");
    ta.className = "sf-ptg-ta";
    ta.placeholder = "your prompt - @ a tag, * a random tag, # a random line";
    ta.title = "输入你的提示词。@name 插入标签，*category 每次 run 随机选一个标签，#name 从列表随机选一行。Ctrl+Enter 运行工作流。";
    ta.spellcheck = false;
    tawrap.append(backdrop, ta);

    const expand = document.createElement("div");
    expand.className = "sf-ptg-expand";

    const bar = document.createElement("div");
    bar.className = "sf-ptg-bar";
    const mkBtn = (label, title) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "sf-ptg-btn";
        b.textContent = label;
        b.title = title;
        return b;
    };
    const copyBtn = mkBtn("Copy all", "复制整个提示词到剪贴板");
    const clearBtn = mkBtn("Clear", "立即清空输入框");
    const tagsBtn = mkBtn("Tags", "打开标签库");
    tagsBtn.innerHTML = "<span>☲</span>Tags";
    const expandSw = document.createElement("button");
    expandSw.type = "button";
    expandSw.className = "sf-ptg-sw";
    expandSw.title = "预览每个 @tag 展开后的提示词";
    expandSw.innerHTML = '<span class="sf-ptg-sw-dot"></span>Show expanded';
    bar.append(copyBtn, clearBtn, tagsBtn, expandSw);

    root.append(portrow, tawrap, expand, bar);
    root._els = { portrow, seg, bMine, bWired, sepDD, tawrap, backdrop, ta, expand, copyBtn, clearBtn, tagsBtn, expandSw };
    return root;
}

// ── render ───────────────────────────────────────────────────────────────
function isWired(node) {
    for (const inp of (node.inputs || [])) if (inp && inp.name === "text_in" && inp.link != null) return true;
    return false;
}
// 显示名为 "text"（与后端 kwarg text_in 对应）；幂等，避免弄脏已保存工作流
function relabelInputSlot(node) {
    for (const inp of (node.inputs || [])) if (inp && inp.name === "text_in" && inp.label !== "text") inp.label = "text";
}

// 尽力读取接线输入的上游文本，让预览显示真实拼接结果；浏览器无法获知时返回 null
function resolveWiredText(node) {
    const inp = (node.inputs || []).find((i) => i && i.name === "text_in");
    if (!inp || inp.link == null) return null;
    const g = node.graph || app.graph;
    let link = g.links?.[inp.link];
    if (!link && typeof g.links?.get === "function") link = g.links.get(inp.link);
    if (!link) return null;
    const src = g.getNodeById ? g.getNodeById(link.origin_id) : (g._nodes || []).find((n) => n.id === link.origin_id);
    return src ? readNodeText(src, 0) : null;
}
function readNodeText(src, depth) {
    if (!src || depth > 4) return null;
    const cls = src.comfyClass || src.type;
    if (cls === "SFPromptTags") {
        const t = src.properties?.promptState?.text;
        return typeof t === "string" ? expandWith(t, PREVIEW_RESOLVERS).out : null;
    }
    const readW = (names) => {
        for (const name of names) {
            const w = (src.widgets || []).find((w) => w && w.name === name && typeof w.value === "string");
            if (w) return w.value;
        }
        return null;
    };
    const byName = readW(["text", "string", "value", "prompt", "wildcard_text", "t"]);
    if (byName != null) return byName;
    const strs = (src.widgets || []).filter((w) => w && typeof w.value === "string");
    if (strs.length === 1) return strs[0].value;
    return null;
}

// 光标在 textarea 里、但看到的文字在 backdrop 上，二者列宽必须一致，否则换行错位、
// 光标逐行漂移。不信任 CSS（主题可能改滚动条宽度），直接测量两列并修正 backdrop
// 右内边距。仅改 backdrop 样式，不触碰 node.size / properties。
function syncBackdropBox(els) {
    const ta = els?.ta, bd = els?.backdrop;
    if (!ta || !bd || !ta.isConnected) return;
    bd.style.paddingRight = "";
    bd.style.width = "";
    const cs = getComputedStyle(ta), bs = getComputedStyle(bd);
    const taText = ta.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const padR = parseFloat(bs.paddingRight);
    const bdText = bd.clientWidth - parseFloat(bs.paddingLeft) - padR;
    if (!(taText > 0) || !(bdText > 0)) return;
    const gap = bdText - taText;
    if (Math.abs(gap) < 0.5) return;
    if (padR + gap >= 0) bd.style.paddingRight = (padR + gap) + "px";
    else bd.style.width = Math.max(0, bd.offsetWidth - gap) + "px";
}

const KIND_CLASS = { tag: "sf-ptg-chip", wild: "sf-ptg-wild", list: "sf-ptg-list" };
const TOKEN_SHADES = 3;
function tokenClass(kind, nth, known) {
    const base = KIND_CLASS[kind] || KIND_CLASS.tag;
    if (!known) return base + " bad";
    const s = nth % TOKEN_SHADES;
    return s ? `${base} s${s}` : base;
}
const newTokenCounts = () => ({ tag: 0, wild: 0, list: 0 });

function renderBackdrop(node) {
    const els = node._sfPromptTagsRoot?._els;
    if (!els) return;
    const text = els.ta.value;
    const toks = scanTokens(text);
    const seen = newTokenCounts();
    let html = "";
    let i = 0;
    for (const h of toks) {
        html += escapeHTML(text.slice(i, h.start));
        const known = h.kind === "tag" ? !!findTag(h.name)
            : h.kind === "wild" ? !!wildCat(h.name)
                : !!listOf(h.name);
        const nth = seen[h.kind]++;
        html += `<span class="${tokenClass(h.kind, nth, known)}">${escapeHTML(h.raw)}</span>`;
        i = h.end;
    }
    html += escapeHTML(text.slice(i));
    // 尾随换行会让 <div> 比 textarea 矮一行，光标会偏离；补一个空格使两列同高
    if (text.endsWith("\n")) html += " ";
    els.backdrop.innerHTML = html;
}
function renderExpand(node) {
    const els = node._sfPromptTagsRoot?._els;
    if (!els) return;
    const st = readState(node);
    const wired = isWired(node);
    const v = els.ta.value;
    if (!st.showExpanded || (!hasTags(v) && !hasWilds(v) && !hasLists(v) && !wired)) { els.expand.style.display = "none"; return; }
    els.expand.style.display = "block";
    // 一次 run 之后显示真实用词（而非 [random: x] 占位符）；文本被编辑则自动退回占位符
    const ran = node._sfPromptTagsLastRun;
    const useRan = ran && ran.src === v;
    const res = useRan ? null : expandWith(v, PREVIEW_RESOLVERS);
    const mine = useRan ? ran.out : res.out;
    const mineHTML = paintExpanded(mine, useRan ? ran.spans : res.spans);
    if (!wired) {
        els.expand.innerHTML = `<span class="mine">${mineHTML}</span>`;
        return;
    }
    const other = resolveWiredText(node);
    if (other != null) {
        els.expand.innerHTML = `<span class="mine">${joinHTML(mineHTML, mine, other, st.order, st.sep)}</span>`;
    } else {
        const where = st.order === "wired" ? "before" : "after";
        els.expand.innerHTML = `<span class="mine">${mineHTML}</span> <span class="note">(+ wired text goes ${where}, shown here once it can be read)</span>`;
    }
}
// 展开文本按 token 着色，与输入框高亮一一对应（未知 token 已保留字面量，不着色）
function paintExpanded(text, spans) {
    if (!spans || !spans.length) return escapeHTML(text);
    const seen = newTokenCounts();
    let html = "";
    let i = 0;
    for (const s of spans) {
        const nth = (s && seen[s.kind] !== undefined) ? seen[s.kind]++ : 0;
        if (!s || s.start < i || s.end > text.length || s.end < s.start) continue;
        html += escapeHTML(text.slice(i, s.start));
        const body = escapeHTML(text.slice(s.start, s.end));
        html += s.known ? `<span class="${tokenClass(s.kind, nth, true)}">${body}</span>` : body;
        i = s.end;
    }
    return html + escapeHTML(text.slice(i));
}
// 预览拼接：与 nodes/text/prompt_tags.py run() 完全镜像（任一侧空白则丢弃分隔符）
function joinHTML(mineHTML, mineRaw, other, order, sep) {
    if (!String(other).trim()) return mineHTML;
    if (!String(mineRaw).trim()) return escapeHTML(other);
    const o = escapeHTML(other), s = escapeHTML(sep);
    return order === "wired" ? (o + s + mineHTML) : (mineHTML + s + o);
}
function refreshBody(node) {
    renderBackdrop(node);
    renderExpand(node);
    updateClearEnabled(node);
}
function updateClearEnabled(node) {
    const els = node._sfPromptTagsRoot?._els;
    if (!els) return;
    els.clearBtn.disabled = !(els.ta.value && els.ta.value.length > 0);
}
function applyOrderUI(node) {
    const els = node._sfPromptTagsRoot?._els;
    if (!els) return;
    const st = readState(node);
    els.bMine.classList.toggle("on", st.order !== "wired");
    els.bWired.classList.toggle("on", st.order === "wired");
    els.sepDD.set(st.sep);
}
function applyExpandSwitch(node) {
    const els = node._sfPromptTagsRoot?._els;
    if (!els) return;
    els.expandSw.classList.toggle("on", readState(node).showExpanded);
}
function refreshWireLock(node) {
    const els = node._sfPromptTagsRoot?._els;
    if (!els) return;
    els.portrow.classList.toggle("on", isWired(node));
    renderExpand(node);
}
// 上游接线节点被编辑时本节点无事件（无跨节点变更钩子），轮询让预览跟随
function startWiredPoll(node) {
    stopWiredPoll(node);
    node._sfLastWired = undefined;
    node._sfPromptPoll = setInterval(() => {
        if (!node._sfPromptTagsRoot) return;
        if (!isWired(node) || !readState(node).showExpanded) { node._sfLastWired = undefined; return; }
        const cur = resolveWiredText(node);
        if (cur !== node._sfLastWired) { node._sfLastWired = cur; renderExpand(node); }
    }, 400);
}
function stopWiredPoll(node) {
    if (node._sfPromptPoll) { clearInterval(node._sfPromptPoll); node._sfPromptPoll = null; }
}

// ── events ────────────────────────────────────────────────────────────────
function wireEvents(node, root) {
    const els = root._els;

    els.ta.addEventListener("input", () => {
        writeState(node, { text: els.ta.value });
        refreshBody(node);
        maybeAC(node, els.ta);
    });
    els.ta.addEventListener("scroll", () => { els.backdrop.scrollTop = els.ta.scrollTop; els.backdrop.scrollLeft = els.ta.scrollLeft; });
    // 主题样式可能在节点构建后落地并改变滚动条宽度，聚焦时重测列宽
    els.ta.addEventListener("focus", () => syncBackdropBox(els));
    els.ta.addEventListener("blur", () => closeAC());
    els.ta.addEventListener("keydown", (e) => {
        if (_ac && _acEl && _acEl.style.display === "block") {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { closeAC(); return; }
            if (e.key === "ArrowDown" && _ac.items.length) { e.preventDefault(); _ac.sel = Math.min(_ac.sel + 1, _ac.items.length - 1); updateACSel(); return; }
            if (e.key === "ArrowUp" && _ac.items.length) { e.preventDefault(); _ac.sel = Math.max(_ac.sel - 1, 0); updateACSel(); return; }
            if ((e.key === "Enter" || e.key === "Tab") && _ac.items.length) { e.preventDefault(); e.stopPropagation(); pickAC(_ac.items[_ac.sel]); return; }
            if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeAC(); return; }
        }
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") return;
        e.stopPropagation();
    });
    els.ta.addEventListener("mousedown", (e) => e.stopPropagation());

    els.copyBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const txt = els.ta.value || "";
        if (!txt) { toast("info", "Nothing to copy"); return; }
        try {
            if (!navigator.clipboard?.writeText) throw new Error("no clipboard");
            await navigator.clipboard.writeText(txt);
            flashBtnText(els.copyBtn, "Copied");
        } catch { toast("warn", "Could not copy to clipboard"); }
    });
    els.clearBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (els.clearBtn.disabled) return;
        els.ta.value = "";
        writeState(node, { text: "" });
        refreshBody(node);
    });
    els.tagsBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        openLibraryFor(node);
    });
    // 右键输入框：Copy / Save as tag。有选区作用于选区，否则作用于整段。
    // 空框时放行浏览器自己的菜单（粘贴仍可用）。
    els.ta.addEventListener("contextmenu", (e) => {
        const ta = els.ta;
        const hasSel = ta.selectionEnd > ta.selectionStart;
        const text = hasSel ? ta.value.slice(ta.selectionStart, ta.selectionEnd) : ta.value;
        if (!text.trim()) return;
        e.preventDefault();
        e.stopPropagation();
        openTextMenu(node, e.clientX, e.clientY, text, hasSel);
    });
    els.expandSw.addEventListener("click", (e) => {
        e.stopPropagation();
        writeState(node, { showExpanded: !readState(node).showExpanded });
        applyExpandSwitch(node);
        renderExpand(node);
    });
    els.bMine.addEventListener("click", (e) => { e.stopPropagation(); writeState(node, { order: "mine" }); applyOrderUI(node); renderExpand(node); });
    els.bWired.addEventListener("click", (e) => { e.stopPropagation(); writeState(node, { order: "wired" }); applyOrderUI(node); renderExpand(node); });

    for (const b of [els.copyBtn, els.clearBtn, els.tagsBtn, els.expandSw, els.bMine, els.bWired]) {
        b.addEventListener("pointerdown", (ev) => ev.stopPropagation());
        b.addEventListener("mousedown", (ev) => ev.stopPropagation());
    }
}

// 从编辑器 Insert 按钮插入 @name / #name 到输入框光标处
function insertToken(node, name, sym) {
    const els = node._sfPromptTagsRoot?._els;
    if (!els) return;
    const s = sym === "#" ? "#" : "@";
    const ta = els.ta;
    const p = ta.selectionStart;
    const before = ta.value.slice(0, p);
    const after = ta.value.slice(p);
    const ins = tagSep(before) + s + name + tagTrail(after);
    ta.value = before + ins + after;
    ta.selectionStart = ta.selectionEnd = p + ins.length;
    writeState(node, { text: ta.value });
    refreshBody(node);
}

// 打开全屏标签库编辑器。`prefill` 来自右键 "Save as tag"（选中文字进创建表单）
function openLibraryFor(node, prefill) {
    openLibraryEditor(node, {
        accent: "#f66744",
        prefill: prefill || "",
        onInsert: (name, sym) => {
            // 每次重新解析（Vue 重建 DOM widget 后旧引用会失效），绝不捕获一次
            const els = node._sfPromptTagsRoot?._els;
            if (!els) return;
            insertToken(node, name, sym);
            toast("info", "Inserted " + sym + name + " into the prompt");
        },
    });
}

// ── 输入框右键菜单（Copy / Save as tag）──────────────────────────────────
let _txtMenu = null;
let _txtMenuOutside = null;
let _txtMenuKey = null;
function closeTextMenu() {
    if (_txtMenuOutside) {
        document.removeEventListener("pointerdown", _txtMenuOutside, true);
        document.removeEventListener("wheel", _txtMenuOutside, true);
        _txtMenuOutside = null;
    }
    if (_txtMenuKey) { document.removeEventListener("keydown", _txtMenuKey, true); _txtMenuKey = null; }
    if (_txtMenu) { _txtMenu.remove(); _txtMenu = null; }
}
function openTextMenu(node, x, y, text, hasSel) {
    closeTextMenu();
    const menu = document.createElement("div");
    menu.className = "sf-ptg-dd-pop";          // 复用深色弹窗外观
    menu.style.minWidth = "184px";
    const mkItem = (label, fn) => {
        const it = document.createElement("div");
        it.className = "sf-ptg-dd-item";
        it.textContent = label;
        it.addEventListener("mousedown", (ev) => { ev.preventDefault(); ev.stopPropagation(); closeTextMenu(); fn(); });
        menu.appendChild(it);
    };
    mkItem(hasSel ? "Copy selection" : "Copy all", async () => {
        try {
            if (!navigator.clipboard?.writeText) throw new Error("no clipboard");
            await navigator.clipboard.writeText(text);
            toast("info", "Copied to clipboard");
        } catch { toast("warn", "Could not copy to clipboard"); }
    });
    mkItem(hasSel ? "Save selection as tag" : "Save all as tag", () => openLibraryFor(node, text));
    document.body.appendChild(menu);
    _txtMenu = menu;
    menu.style.left = Math.max(8, Math.min(x, window.innerWidth - menu.offsetWidth - 8)) + "px";
    menu.style.top = Math.max(8, Math.min(y, window.innerHeight - menu.offsetHeight - 8)) + "px";
    _txtMenuOutside = (e) => { if (_txtMenu && !_txtMenu.contains(e.target)) closeTextMenu(); };
    _txtMenuKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); closeTextMenu(); } };
    setTimeout(() => {
        document.addEventListener("pointerdown", _txtMenuOutside, true);
        document.addEventListener("wheel", _txtMenuOutside, true);
        document.addEventListener("keydown", _txtMenuKey, true);
    }, 0);
}

// ── setup ────────────────────────────────────────────────────────────────
function setupNode(node) {
    injectCSS();
    const root = buildRoot(node);
    node._sfPromptTagsRoot = root;

    const st = readState(node);
    root._els.ta.value = st.text;

    node.addDOMWidget("sf_prompt_tags_ui", "sf_prompt_tags", root, {
        getValue: () => null,
        setValue: () => {},
        getMinHeight: () => WIDGET_MIN_H,
        margin: 4,
        serialize: false,
    });

    if (typeof ResizeObserver === "function") {
        node._sfPromptBoxRO = new ResizeObserver(() => syncBackdropBox(root._els));
        node._sfPromptBoxRO.observe(root._els.ta);
    }

    wireEvents(node, root);

    // 库变更（弹窗编辑）时重高亮 / 重预览
    node._sfPromptUnsub = subscribe(() => { refreshBody(node); });
    startWiredPoll(node);

    if (node.size[0] < MIN_W) node.size[0] = DEFAULT_W;
    if (node.size[1] < MIN_H) node.size[1] = DEFAULT_H;

    queueMicrotask(() => {
        relabelInputSlot(node);
        applyOrderUI(node);
        applyExpandSwitch(node);
        refreshWireLock(node);
        syncBackdropBox(root._els);
        refreshBody(node);
    });
    node.setDirtyCanvas(true, true);
}


// ── extension ────────────────────────────────────────────────────────────
app.registerExtension({
    name: "sfnodes.PromptTags",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SFPromptTags") return;

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = origConfigure?.apply(this, arguments);
            queueMicrotask(() => {
                const root = this._sfPromptTagsRoot;
                if (root && root._els) {
                    const st = readState(this);
                    if (root._els.ta.value !== st.text) root._els.ta.value = st.text;
                    relabelInputSlot(this);
                    applyOrderUI(this);
                    applyExpandSwitch(this);
                    refreshWireLock(this);
                    refreshBody(this);
                }
            });
            return r;
        };

        const origOCC = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = origOCC?.apply(this, arguments);
            queueMicrotask(() => refreshWireLock(this));
            return r;
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            closeAC();
            closeDD();
            closeTextMenu();
            closeLibraryEditorFor(this);
            this._sfPromptBoxRO?.disconnect();
            this._sfPromptBoxRO = null;
            this._sfPromptUnsub?.();
            this._sfPromptUnsub = null;
            stopWiredPoll(this);
            this._sfPromptTagsRoot = null;
            if (origRemoved) return origRemoved.apply(this, arguments);
        };
    },

    nodeCreated(node) {
        if (node.comfyClass !== "SFPromptTags" && node.type !== "SFPromptTags") return;
        setupNode(node);
    },
});

// ── graphToPrompt：展开 token 并注入 PromptState（Sliders / Seed 模式）──────
function buildPromptNodeIndex() {
    const index = new Map();
    const visit = (graph, prefix) => {
        if (!graph) return;
        const nodes = graph._nodes || graph.nodes || [];
        for (const n of nodes) {
            if (!n) continue;
            const fullId = prefix + String(n.id);
            if (n.comfyClass === "SFPromptTags" || n.type === "SFPromptTags") index.set(fullId, n);
            const inner = n.subgraph || n.graph || n._graph;
            if (inner && inner !== graph) visit(inner, fullId + ":");
        }
    };
    visit(app.graph, "");
    return index;
}
function findPromptNode(index, promptId) {
    const sId = String(promptId);
    if (index.has(sId)) return index.get(sId);
    // 子图复合 id（"5:3"）尾部回退：仅当恰好一个节点携带该尾部且非顶层节点时接受，
    // 否则注入可能把 A 节点的提示词换成 B 的
    const tail = sId.includes(":") ? sId.slice(sId.lastIndexOf(":") + 1) : null;
    if (!tail) return null;
    let hit = null;
    let seen = 0;
    for (const [key, node] of index) {
        const cut = key.lastIndexOf(":");
        if (cut < 0) continue;
        if (key.slice(cut + 1) === tail) { hit = node; seen++; if (seen > 1) return null; }
    }
    return seen === 1 ? hit : null;
}

// 防双包装：模块被二次求值（混用带戳/不带戳的导入、热重载）时，随机槽会每个 run 掷两次
if (!app._sfPromptTagsPatched) {
    app._sfPromptTagsPatched = true;
    const _origGraphToPrompt = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...args) {
        const result = await _origGraphToPrompt(...args);
        try {
            const prompt = result?.output;
            if (prompt && typeof prompt === "object") {
                // 把 build id 盖到 prompt 对象上：commitPicks 随后只消耗恰好被
                // 入队的那次 build 的选择——Export / 分享 / 校验失败都不会推进
                // shuffle 牌堆或 in-order 序列
                beginPickBuild(prompt);
                let index = null;
                for (const key of Object.keys(prompt)) {
                    try {
                        const entry = prompt[key];
                        if (!entry || entry.class_type !== "SFPromptTags") continue;
                        if (!index) index = buildPromptNodeIndex();
                        const node = findPromptNode(index, key);
                        if (!node) {
                            // 无法匹配时不动 entry（Python 回退到默认空文本），并提示
                            console.warn("sfnodes.PromptTags: could not match prompt node", key, "- leaving its PromptState alone");
                            toast("warn", "A SF Prompt Tags node could not be matched, so its typed prompt was not sent.");
                            continue;
                        }
                        const st = readState(node);
                        // 每次 run 经 Picks 游标掷 *wildcard / #list；@tags 确定性展开。
                        // 不同随机结果改变字符串 -> 缓存键变化 -> 自动重跑（无 nonce 即可）
                        const ranRes = expandWith(st.text, makeRunResolvers());
                        // 记住本次 build 实际产生的词，让展开预览显示真实结果（仅运行时，
                        // 写 properties 会弄脏工作流）
                        node._sfPromptTagsLastRun = { src: st.text, out: ranRes.out, spans: ranRes.spans };
                        queueMicrotask(() => { try { renderExpand(node); } catch { /* 节点可能已删除 */ } });
                        entry.inputs = entry.inputs || {};
                        entry.inputs.PromptState = JSON.stringify({ text: ranRes.out, order: st.order, sep: st.sep });
                    } catch (nodeErr) {
                        console.error("sfnodes.PromptTags: graphToPrompt node failed", key, nodeErr);
                    }
                }
            }
        } catch (err) {
            console.error("sfnodes.PromptTags: graphToPrompt hook failed", err);
        }
        return result;
    };
}

// queuePrompt patcher：只有队列被真正接受才推进游标（commitPicks）。
// graphToPrompt 也会因 Export / 分享 / 保存按钮运行，那些场合绝不消耗选择；
// 一次被拒绝的 run 同样不消耗（_origQueuePrompt 会抛错 -> 选择保留）。
if (!app._sfPromptTagsQueuePatched && api && typeof api.queuePrompt === "function") {
    app._sfPromptTagsQueuePatched = true;
    const _origQueuePrompt = api.queuePrompt.bind(api);
    api.queuePrompt = async function (...args) {
        const res = await _origQueuePrompt(...args);   // 拒绝的队列抛错 -> 选择保留
        try {
            // 交出被 POST 的确切 prompt 对象，消耗的恰好是那次 build 的选择。
            // 在参数里找而不是假设固定位置，签名变动也不会失效。
            let queued = null;
            for (const a of args) {
                if (a && typeof a === "object" && a.output && typeof a.output === "object") { queued = a.output; break; }
            }
            commitPicks(queued);
        } catch (err) { console.error("sfnodes.PromptTags: commitPicks failed", err); }
        return res;
    };
}
