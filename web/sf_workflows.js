// ==========================================================================
// sf_workflows.js - SF Workflows 主扩展
// ==========================================================================
//
// 复刻 Pixaroma Workflows：浮动面板，用于查找与整理工作流（工具栏按钮 +
// Alt+W + canvas 右键菜单打开）。刻意没有节点——节点会被存进工作流文件，
// 分享工作流会把一个多余节点带给每个打开的人。这属于应用，正如帮助。
//
// 分层：
//   - api 层（本文件内）：唯一触碰服务端与 ComfyUI 工作流 store 的代码，
//     涉及可能让人丢工作的调用都集中在此、可通读
//   - 纯函数（sf_workflows_lib.js）：名称清洗/文件夹顺序/搜索评分
//   - DOM（sf_workflows_ui.js）：窗口/菜单/封面/网格/文件夹/CSS
//
// 与原件差异（已确认范围）：阶段 1 无详情面板与 tidy 分组屏（tidy 选中时
// 以普通网格展示）；收藏走 pinia（Vue 新版），旧版前端隐藏收藏入口；
// sidecar/缓存文件名与设置键加 sf_ 前缀；无版本行/帮助浏览器。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import {
    cleanName, nameProblem, orderedFolders, siblingsOf, searchEntries,
} from "./sf_workflows_lib.js";
import {
    createWorkflowWindow, renderGrid, renderFolders, renderDetail, renderTidy,
    openContextMenu, closeContextMenu,
    setMenuFocusHome, setRenameLostNotifier, dropRename, beginRename, beginFolderRename,
    installOutputCoverCapture, hasHandCover, markRendering, copyText, pixApiUrl, el,
    CARD_MIME, injectWorkflowCSS,
} from "./sf_workflows_ui.js";

const CMD_ID = "sfnodes.OpenWorkflowBrowser";
const VIEW_SETTING = "sfnodes.Workflows.View";
const SORT_SETTING = "sfnodes.Workflows.Sort";
const DENSITY_SETTING = "sfnodes.Workflows.Density";

// 全局设置读写桥（ui 模块经 window 调用，主扩展在此注入实现）
window.sfnodesGetSetting = (key, dflt) => {
    try { return app.ui?.settings?.getSettingValue(key) ?? dflt; } catch { return dflt; }
};
window.sfnodesSetSetting = (key, val) => {
    try {
        const s = app.ui?.settings;
        if (typeof s?.setSettingValueAsync === "function") s.setSettingValueAsync(key, val);
        else if (typeof s?.setSettingValue === "function") s.setSettingValue(key, val);
    } catch { /* ignore */ }
};

// ── api 层：唯一触碰服务端 / 工作流 store 的代码 ─────────────────────────
// BASE 刻意裸着：它被拼接，托管 ComfyUI 会以 QUERY STRING 追加鉴权令牌——
// 包裹前缀会把令牌放进 URL 中间。整个 URL 在 fetch 处包裹。
const BASE = "/api/sfnodes/workflows";

const store = () => app.extensionManager?.workflow;

async function getJSON(url) {
    // no-store：列表必须匹配磁盘，启发式缓存会静默显示已改名/删除的工作流
    const r = await fetch(pixApiUrl(url), { cache: "no-store" });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
}

async function postJSON(url, body) {
    const r = await fetch(pixApiUrl(url), {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
}

const fetchIndex = () => getJSON(`${BASE}/index`);
const fetchMeta = () => getJSON(`${BASE}/meta`);
const saveMeta = (patch) => postJSON(`${BASE}/meta`, patch);
const folderAction = (body) => postJSON(`${BASE}/folder`, body);
const reveal = (path) => postJSON(`${BASE}/reveal`, { path });
const setCover = (rel, dataUrl) => postJSON(`${BASE}/cover`, { rel, dataUrl });
const clearCover = (rel) => saveMeta({ covers: { [rel]: null } });

// ── ComfyUI 工作流 store ──────────────────────────────────────────────────
// store 以 "workflows/<相对路径>" 为键。
const toStorePath = (rel) => (rel.startsWith("workflows/") ? rel : `workflows/${rel}`);
const fromStorePath = (p) => (p || "").replace(/^workflows\//, "");

function activePath() {
    return fromStorePath(store()?.activeWorkflow?.path || "");
}

function openPaths() {
    return (store()?.openWorkflows || []).map((w) => fromStorePath(w.path));
}

function isModified(rel) {
    const w = store()?.getWorkflowByPath?.(toStorePath(rel));
    return !!w?.isModified;
}

/**
 * store 的对象，带一次 syncWorkflows 重试。面板自己的列表来自我们的路由
 * （读文件夹），所以文件一存在就显示；ComfyUI 的 store 只认识它已同步的
 * 文件——从 Explorer 拖入的（或 ComfyUI 运行期间任何东西写入的）不在其中。
 * 一次同步补齐差距；文件真没了，重试也错过，消息终于为真。
 */
async function storeWorkflow(rel) {
    const s = store();
    if (!s?.getWorkflowByPath) throw new Error("This ComfyUI build has no workflow store.");
    let wf = s.getWorkflowByPath(toStorePath(rel));
    if (!wf && typeof s.syncWorkflows === "function") {
        try {
            await s.syncWorkflows();
            wf = s.getWorkflowByPath(toStorePath(rel));
        } catch { /* 下面的 throw 说明要紧的 */ }
    }
    return wf;
}

/**
 * 打开工作流。两条规则绝不放松：
 *   1. 绝不向 load() 传 { force: true }——它会从磁盘重取，静默丢掉未保存编辑
 *   2. 绝不主动调用 save()/saveAs()——只在明确的用户动作里
 */
async function openWorkflow(rel) {
    const wf = await storeWorkflow(rel);
    if (!wf) throw new Error("That workflow is no longer there.");

    // 无 { force: true }：对已打开的工作流这是 no-op，未保存编辑存活
    await wf.load();
    await app.loadGraphData(wf.activeState, true, true, wf);
    return wf;
}

/** 这个路径是否已有工作流？改名/移动/另存为之前问。 */
async function exists(rel) {
    try {
        const r = await fetch(pixApiUrl(`/api/userdata/${encodeURIComponent(toStorePath(rel))}`),
                              { method: "HEAD", cache: "no-store" });
        return r.ok;
    } catch {
        return false;      // 服务器不可达：让真实调用报告问题
    }
}

/** 改名或移动——移动只是名字里换了文件夹。 */
async function renameOrMove(rel, newRel) {
    const leaf = () => newRel.split("/").pop();
    // 只改大小写做不到：大小写不敏感磁盘上目标解析为同一文件，ComfyUI
    // 自己的移动以 409 拒绝——这是 core 限制，如实报告
    const caseOnly = rel !== newRel && rel.toLowerCase() === newRel.toLowerCase();
    if (caseOnly) {
        throw new Error("Only the capitalisation changed, and ComfyUI cannot rename a "
            + "workflow to the same name in a different case. Rename it to something "
            + "else first, then back.");
    }
    if (rel !== newRel && await exists(newRel)) {
        throw new Error(`There is already a workflow called "${leaf()}" there.`);
    }
    const s = store();
    const wf = await storeWorkflow(rel);
    if (!wf) throw new Error("That workflow is no longer there.");
    // 走 store 而非绕过它移文件：让打开的标签指向正确文件、modified 标志完好
    try {
        if (typeof wf.rename === "function") await wf.rename(toStorePath(newRel));
        else if (typeof s.renameWorkflow === "function") await s.renameWorkflow(wf, toStorePath(newRel));
        else throw new Error("This ComfyUI build cannot rename workflows.");
    } catch (err) {
        const msg = String(err?.message || err);
        if (/\b409\b|conflict|exists/i.test(msg)) {
            throw new Error(`There is already a workflow called "${leaf()}" there.`);
        }
        throw err;
    }
    await s.syncWorkflows?.();
}

async function remove(rel) {
    const s = store();
    const wf = await storeWorkflow(rel);
    if (!wf) throw new Error("That workflow is no longer there.");
    if (typeof wf.delete === "function") await wf.delete();
    else if (typeof s.deleteWorkflow === "function") await s.deleteWorkflow(wf);
    else throw new Error("This ComfyUI build cannot delete workflows.");
    await s.syncWorkflows?.();
}

/**
 * 把当前打开的工作流存进一个文件夹。仅用户动作。
 * ⚠ 从未保存的工作流不能直接 saveAs（临时文件 saveAs 只会存它自己）——
 * 与 core 的 saveWorkflowAs 相同：先改名到目标再保存。
 */
async function saveCurrentAs(newRel, { overwrite = false } = {}) {
    if (!overwrite && await exists(newRel)) {
        throw new Error(`There is already a workflow called "${newRel.split("/").pop()}" there.`);
    }
    const s = store();
    const wf = s?.activeWorkflow;
    if (!wf) throw new Error("Nothing is open to save.");
    const path = toStorePath(newRel);

    if (wf.isTemporary) {
        if (typeof s.renameWorkflow === "function") await s.renameWorkflow(wf, path);
        else if (typeof wf.rename === "function") await wf.rename(path);
        else throw new Error("This ComfyUI build cannot save an unsaved workflow into a folder.");
        wf.changeTracker?.prepareForSave?.();
        if (typeof s.saveWorkflow === "function") await s.saveWorkflow(wf);
        else if (typeof wf.save === "function") await wf.save();
        else throw new Error("This ComfyUI build cannot save workflows.");
    } else {
        if (typeof wf.saveAs !== "function") throw new Error("This ComfyUI build cannot save-as.");
        await wf.saveAs(path);
    }
    await s.syncWorkflows?.();
}

/** 复制一份到旁边。用 ComfyUI 自己的 userdata 端点，副本字节一致。 */
async function duplicate(rel, newRel) {
    const enc = (p) => encodeURIComponent(toStorePath(p));
    const r = await fetch(pixApiUrl(`/api/userdata/${enc(rel)}`), { cache: "no-store" });
    if (!r.ok) throw new Error("Could not read that workflow.");
    const body = await r.text();
    const w = await fetch(pixApiUrl(`/api/userdata/${enc(newRel)}?overwrite=false`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
    });
    if (!w.ok) throw new Error(w.status === 409 ? "A workflow with that name already exists." : "Could not save the copy.");
    await store()?.syncWorkflows?.();
}

// ── 收藏（pinia workflowBookmark，Vue 新版）──────────────────────────────
// ComfyUI 启动时不读收藏文件，其书签 store 直到有人调用 loadBookmarks() 才
// 有内容——通常其 Workflows 侧栏被打开。直接切换会向空内存列表追加并保存，
// 覆盖文件、清掉磁盘上每个收藏。所以先加载再读/切。
async function ensureFavouritesLoaded() {
    const bm = bookmarkStore();
    if (typeof bm?.loadBookmarks !== "function") return false;
    try {
        await bm.loadBookmarks();
        return true;
    } catch {
        return false;
    }
}

function favourites() {
    return new Set((store()?.bookmarkedWorkflows || []).map((w) => fromStorePath(w.path)));
}

function bookmarkStore() {
    try {
        const pinia = document.querySelector("#vue-app")?.__vue_app__?.config?.globalProperties?.$pinia;
        return pinia?._s?.get("workflowBookmark") || null;
    } catch {
        return null;
    }
}

async function toggleFavourite(rel) {
    const bm = bookmarkStore();
    if (typeof bm?.toggleBookmarked !== "function") {
        throw new Error("This ComfyUI build keeps favourites somewhere this panel cannot reach. "
            + "Use the star in ComfyUI's own Workflows sidebar.");
    }
    // 必须先来：对未读文件的 store 切换会保存一个从零建的列表并抹掉每个
    // 现有收藏
    await ensureFavouritesLoaded();
    await bm.toggleBookmarked(toStorePath(rel));
    return true;
}

// ── 状态 ──────────────────────────────────────────────────────────────────

const DENSITY = {
    s: { k: 1, label: "Small - the most workflows on screen at once" },
    m: { k: 1.15, label: "Medium - the default" },
    l: { k: 1.32, label: "Large - biggest text and biggest pictures" },
};

function applyDensity(which) {
    const k = DENSITY[which]?.k ?? DENSITY.m.k;
    try { document.documentElement.style.setProperty("--sfwb-k", String(k)); } catch { /* nothing */ }
}

const SORT_LABELS = { recent: "Recent", name: "Name", nodes: "Nodes" };

const S = {
    win: null,
    btn: null,
    loading: false,
    entries: [],
    rawFolders: [],
    folders: [],
    sortBtn: null,
    collections: [],
    issues: {},
    tidyRels: new Set(),
    meta: { notes: {}, covers: {}, folderColors: {}, folderExpanded: [] },
    favourites: new Set(),
    openPaths: [],
    byRel: new Map(),
    sel: { kind: "all" },
    selected: new Set(),
    kbdRel: null,
    query: "",
    view: "grid",
    sort: "recent",
    density: "m",
    visible: [],
};

// ── 数据 ──────────────────────────────────────────────────────────────────

let loadSeq = 0;

async function loadData() {
    // 每次加载带票号。两次加载容易重叠（打开面板一次，任何动作经 guard
    // 再触发一次）——最后 RESOLVE 的曾获胜而非最后 START 的
    const ticket = ++loadSeq;
    S.loading = true;
    try {
        const [idx, meta] = await Promise.all([fetchIndex(), fetchMeta()]);
        if (ticket !== loadSeq) return;
        S.entries = idx.entries || [];
        S.rawFolders = idx.folders || [];
        S.collections = idx.collections || [];
        S.issues = idx.issues || {};
        S.meta = meta.meta || { notes: {}, covers: {}, folderColors: {} };
        // 服务器按字母列出；用户选择顺序在 sidecar，此处应用一次
        S.folders = orderedFolders(S.rawFolders, S.meta.folderOrder);

        // 哪些节点真的缺失在此处算，不在服务器：Python 的节点列表只含
        // Python 节点，对 Note/MarkdownNote/Primitive 会误报。浏览器注册表
        // 两种都有，是"这工作流本机能否打开"的唯一诚实答案
        const registry = window.LiteGraph?.registered_node_types || null;
        const missingNodes = [];
        S.byRel = new Map();
        for (const e of S.entries) {
            e._note = S.meta.notes?.[e.rel] || "";
            e._missing = registry
                ? (e.class_types || []).filter((t) => !(t in registry))
                : [];
            if (e._missing.length) missingNodes.push({ rel: e.rel, name: e.name, missing: e._missing });
            S.byRel.set(e.rel, e);
        }
        S.issues.missing_nodes = missingNodes;
        S.tidyRels = collectTidyRels(S.issues);

        // 选中要挺过重载，"挺过"包括指向仍存在的东西。消失的文件夹选中
        // 会把每个工作流都过滤掉——回退 All workflows
        if (S.sel.kind === "folder" && S.sel.value !== "" && !S.folders.includes(S.sel.value)) {
            S.sel = { kind: "all" };
        } else if (S.sel.kind === "collection" && !S.collections.some((c) => c.id === S.sel.value)) {
            S.sel = { kind: "all" };
        } else if (S.sel.kind === "tidy" && !S.tidyRels.size) {
            S.sel = { kind: "all" };
        }
    } catch (err) {
        if (ticket !== loadSeq) return;
        S.entries = [];
        S.win?.toast("Could not read the workflows folder: " + err.message);
    } finally {
        if (ticket === loadSeq) S.loading = false;
    }
    refreshLive();
}

/** 每个需要注意的工作流，作为一个路径集。徽标与视图必须来自同一集合。 */
function collectTidyRels(issues) {
    const rels = new Set();
    for (const u of issues.unsaved_names || []) rels.add(u.rel);
    for (const g of issues.duplicates || []) for (const d of g) rels.add(d.rel);
    for (const m of issues.missing_nodes || []) rels.add(m.rel);
    return rels;
}

/** 不随磁盘变化而变的部分：现在开着哪些、哪些收藏。每次渲染重读，绝不
 *  跨工作流切换缓存（面板跨切换存活）。 */
function refreshLive() {
    try {
        S.favourites = favourites();
        S.openPaths = openPaths();
    } catch {
        S.favourites = new Set();
        S.openPaths = [];
    }
}

// ── 中间列显示什么 ────────────────────────────────────────────────────────

function computeVisible() {
    let list = S.entries;
    const sel = S.sel;

    if (sel.kind === "fav") {
        list = list.filter((e) => S.favourites.has(e.rel));
    } else if (sel.kind === "recent") {
        list = [...list].sort((a, b) => (b.modified || 0) - (a.modified || 0)).slice(0, 20);
    } else if (sel.kind === "folder") {
        // 文件夹显示其中内容，含子文件夹
        list = list.filter((e) => sel.value === ""
            ? !e.folder
            : e.folder === sel.value || e.folder.startsWith(sel.value + "/"));
    } else if (sel.kind === "collection") {
        const c = S.collections.find((x) => x.id === sel.value);
        const set = new Set(c?.items || []);
        list = list.filter((e) => set.has(e.rel));
    } else if (sel.kind === "tidy") {
        list = list.filter((e) => S.tidyRels.has(e.rel));
    }

    list = searchEntries(list, S.query);

    // 搜索已按匹配度排序，再按日期排会丢掉它
    if (!S.query && S.sel.kind !== "recent") {
        const by = {
            recent: (a, b) => (b.modified || 0) - (a.modified || 0),
            name: (a, b) => a.name.localeCompare(b.name),
            nodes: (a, b) => (b.node_count || 0) - (a.node_count || 0),
        }[S.sort];
        if (by) list = [...list].sort(by);
    }
    S.visible = list;
}

// ── 渲染 ──────────────────────────────────────────────────────────────────

function render() {
    if (!S.win?.isOpen()) return;
    refreshLive();
    computeVisible();

    // 包一层让打开的改名框能区分"面板重绘"与"用户点走"
    markRendering(() => {
        renderFolders(S.win.side, S, {
            onPick: onPickFolder,
            onDropOn: onDropOnFolder,
            onRenameFolder: startFolderRename,
            onFolderMenu: showFolderMenu,
            onReorderFolder: reorderFolderByDrop,
            onToggleFolder: setFolderExpanded,
        });
        // tidy 屏走专用审查界面，其余走网格
        if (S.sel.kind === "tidy") renderTidy(S.win.main, S, HANDLERS);
        else renderGrid(S.win.main, S, HANDLERS);
        if (S.win.isDetailVisible()) renderDetail(S.win.detail, S, HANDLERS);
    });

    refreshSortButton();

    const total = S.entries.length;
    S.win.setCount(S.visible.length === total
        ? `${total} workflows`
        : `${S.visible.length} of ${total}`);
}

/** 搜索结果按相关度排序、Recent 按日期排序，两个视图下排序控件真不做事，
 *  禁用并说明而非让它看似活着。 */
function sortDisabledReason() {
    if (S.query) return "Search results are ordered by how well they match, so sorting is off.";
    if (S.sel.kind === "recent") return "Recent is already ordered by when you last changed a workflow.";
    return "";
}

function refreshSortButton() {
    const b = S.sortBtn;
    if (!b) return;
    const why = sortDisabledReason();
    b.disabled = !!why;
    b.title = why || "Change the order";
}

// ── 面板自己的小对话框 ────────────────────────────────────────────────────

let openAsk = null;

function closeAsk() {
    const cancel = openAsk;
    openAsk = null;
    if (cancel) { try { cancel(); } catch { /* 已消失 */ } }
}

function ask({ title, message, value, okLabel = "OK", danger }) {
    return new Promise((resolve) => {
        const back = el("div");
        back.tabIndex = -1;
        back.style.cssText = "position:absolute;inset:0;background:rgba(0,0,0,.55);z-index:8;display:flex;align-items:center;justify-content:center;";
        const box = el("div");
        const listy = (message || "").includes("\n");
        box.style.cssText = "background:#1d1c1b;border:1px solid #3d3936;border-radius:8px;"
            + `padding:14px 16px;width:min(${listy ? 460 : 330}px,90%);`
            + "box-shadow:0 12px 30px rgba(0,0,0,.6);";
        box.append(el("div", "sf-wb-cardname", title));
        if (message) {
            const m = el("div", "sf-wb-cardmeta", message);
            m.style.whiteSpace = "pre-wrap";
            if (listy) { m.style.maxHeight = "38vh"; m.style.overflowY = "auto"; }
            box.append(m);
        }

        // 对话框在时停止每个键：遮罩挡鼠标但键盘事件仍冒泡到面板处理器
        back.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Escape") { e.preventDefault(); done(null); }
            else if (e.key === "Enter" && !input) {
                e.preventDefault();
                // Enter 仅在确认按钮可见聚焦时确认；其它任何地方都是安全取消
                done(document.activeElement === ok ? true : null);
            } else if (e.key === "Tab") {
                e.preventDefault();
                const stops = [input, ok, no].filter(Boolean);
                const at = stops.indexOf(document.activeElement);
                const next = (at + (e.shiftKey ? -1 : 1) + stops.length) % stops.length;
                stops[at < 0 ? 0 : next].focus();
            }
        });

        let input = null;
        if (value !== undefined) {
            input = el("input", "sf-wb-rename");
            input.style.minHeight = "0";
            input.value = value;
            input.addEventListener("keydown", (e) => {
                e.stopPropagation();
                if (e.key === "Enter") done(input.value.trim());
                if (e.key === "Escape") done(null);
            });
            box.append(input);
        }

        const acts = el("div", "sf-wb-seg");
        const ok = el("button", "sf-wb-tbtn " + (danger ? "sf-wb-danger" : "sf-wb-primary"), okLabel);
        const no = el("button", "sf-wb-tbtn", "Cancel");
        ok.type = no.type = "button";
        ok.title = danger ? "This cannot be undone" : "Enter also does this";
        no.title = "Escape also does this";
        if (input) input.title = "Enter to confirm, Escape to cancel";
        acts.append(ok, no);
        box.append(acts);
        back.append(box);
        S.win.el.querySelector(".sf-wb-body").append(back);
        setTimeout(() => (input || ok).focus(), 20);
        setTimeout(() => { if (!back.contains(document.activeElement)) back.focus(); }, 40);

        let settled = false;
        function done(v) {
            if (settled) return;
            settled = true;
            if (openAsk === cancel) openAsk = null;
            back.remove();
            resolve(v);
            S.win?.focusSearch?.();
        }
        const cancel = () => done(null);
        openAsk = cancel;
        ok.addEventListener("click", () => done(input ? input.value.trim() : true));
        no.addEventListener("click", () => done(null));
        back.addEventListener("mousedown", (e) => { if (e.target === back) done(null); });
    });
}

const confirmAsk = (title, message, okLabel = "Delete") =>
    ask({ title, message, okLabel, danger: true });

// ── 动作 ──────────────────────────────────────────────────────────────────

/** 把工作流的笔记与手选封面移到新路径。先写新键后清旧键（清封面会删图
 *  除非别的键已指向它）。 */
async function carryMeta(oldRel, newRel) {
    if (oldRel === newRel) return;
    const note = S.meta?.notes?.[oldRel];
    const cover = S.meta?.covers?.[oldRel];
    if (!note && !cover) return;
    const patch = {};
    if (note) patch.notes = { [newRel]: note, [oldRel]: null };
    if (cover) patch.covers = { [newRel]: cover, [oldRel]: null };
    try { await saveMeta(patch); } catch { /* 改名本身已成功 */ }
}

/** 从选中里丢一个路径，或指到文件去的地方。 */
function forgetRel(rel, replacement) {
    if (S.selected.delete(rel) && replacement) S.selected.add(replacement);
    if (S.kbdRel === rel) S.kbdRel = replacement || null;
}

const dirOf = (rel) => (rel.includes("/") ? rel.slice(0, rel.lastIndexOf("/")) : "");
const joinRel = (folder, file) => (folder ? `${folder}/${file}` : file);

async function guard(fn, okMessage) {
    let failure = null;
    try {
        await fn();
    } catch (err) {
        failure = err;
    }
    // 成败都重载：半失败的批量已改变磁盘，留着旧列表会招人重跑
    try {
        await loadData();
        render();
    } catch { /* toast 是更有用的消息 */ }
    if (failure) S.win?.toast(failure.message || String(failure));
    else if (okMessage) S.win?.toast(okMessage);
}

const HANDLERS = {
    onSelect(entry, e) {
        if (e.shiftKey || e.ctrlKey || e.metaKey) {
            S.selected.has(entry.rel) ? S.selected.delete(entry.rel) : S.selected.add(entry.rel);
        } else {
            S.selected = new Set([entry.rel]);
        }
        S.kbdRel = entry.rel;
        render();
    },

    onOpen(entry) {
        guard(async () => {
            await openWorkflow(entry.rel);
            S.win.toast(`Opened ${entry.name}`);
        });
    },

    onStar(entry) {
        guard(() => toggleFavourite(entry.rel));
    },

    onRename(entry) {
        // beginRename 就地编辑卡片，需要卡片在屏上。被搜索过滤掉或从详情
        // 面板来就不在——回退到对话框
        const onScreen = S.win.main.querySelector(`[data-rel="${CSS.escape(entry.rel)}"]`);
        if (!onScreen) {
            ask({ title: "Rename", message: entry.rel, value: entry.name, okLabel: "Rename" })
                .then((v) => { if (v) commitRename(entry, v); });
            return;
        }
        beginRename(S.win.main, entry.rel, entry.name, (newName) => {
            const clean = cleanName(newName);
            const bad = nameProblem(clean);
            if (bad) { S.win.toast(bad); return; }
            const target = joinRel(dirOf(entry.rel), clean + ".json");
            if (target === entry.rel) return;
            guard(async () => {
                await renameOrMove(entry.rel, target);
                await carryMeta(entry.rel, target);
                forgetRel(entry.rel, target);
            }, "Renamed");
        });
    },

    onDuplicate(entry) {
        guard(async () => {
            const target = joinRel(dirOf(entry.rel), entry.name + " copy.json");
            await duplicate(entry.rel, target);
            // 副本也拿笔记与封面。它是副本，应长得像被复制的东西。
            // 刻意不用 carryMeta：那会移动它们、清掉原件的
            const note = S.meta?.notes?.[entry.rel];
            const cover = S.meta?.covers?.[entry.rel];
            const patch = {};
            if (note) patch.notes = { [target]: note };
            if (cover) patch.covers = { [target]: cover };
            if (Object.keys(patch).length) {
                try { await saveMeta(patch); } catch { /* 副本本身成功了 */ }
            }
        }, "Copied");
    },

    async onDelete(entry) {
        // 删除正在编辑的工作流同时损失未保存工作
        const dirty = isModified(entry.rel);
        const yes = await confirmAsk(
            `Delete "${entry.name}"?`,
            dirty
                ? "This one is OPEN with unsaved changes. Deleting it loses those changes too, and there is no undo."
                : "There is no undo yet, so this really does remove the file.");
        if (!yes) return;
        guard(async () => {
            await remove(entry.rel);
            forgetRel(entry.rel);
        }, "Deleted");
    },

    async onDeleteMany(rels, wording) {
        const dirty = rels.filter((r) => isModified(r));
        const warn = dirty.length
            ? `${dirty.length} of them are open with unsaved changes, which go too. There is no undo.`
            : "There is no undo yet, so this really does remove the files.";
        const yes = await confirmAsk(
            wording?.title || `Delete ${rels.length} workflows?`,
            wording?.message ? `${wording.message}\n\n${warn}` : warn);
        if (!yes) return;
        // 越过失败继续并在结尾报告：首个失败即停会留下其余静默未删
        guard(async () => {
            const failed = [];
            for (const rel of rels) {
                try { await remove(rel); forgetRel(rel); }
                catch { failed.push(rel.split("/").pop()); }
            }
            if (failed.length) throw new Error(`Could not delete ${failed.length}: ${failed.join(", ")}`);
        }, `Deleted ${rels.length}`);
    },

    onReveal(entry) {
        guard(() => reveal(entry.rel), "Opened the folder - look in your taskbar");
    },

    onNote(rel, text) {
        // 先更新内存副本再走往返。任何快照 S.meta 的东西——文件夹改名搬一
        // 整子树、单文件 carryMeta——必须看到用户最后输入的样子；只成功时
        // 更新会在窗口里让改名搬走旧文本
        S.meta.notes = S.meta.notes || {};
        if (text) S.meta.notes[rel] = text; else delete S.meta.notes[rel];
        const e = S.byRel.get(rel);
        if (e) e._note = text || "";
        saveMeta({ notes: { [rel]: text || null } })
            .then((res) => {
                // 检查 RESULT 而非"没抛"：路由在 sidecar 写失败时以 {ok:false}
                // 答 200，用户眼看打出来的笔记可能悄悄没到磁盘
                if (!res || res.ok === false) throw new Error("not saved");
            })
            .catch(() => S.win.toast("Could not save that note."));
    },

    onCopyText(text, okMessage) {
        copyText(text).then((ok) => {
            S.win.toast(ok ? okMessage : "Could not reach the clipboard.");
        });
    },

    onSetCover(entry) {
        const picker = el("input");
        picker.type = "file";
        // 服务器实际接受的格式，而非 "image/*"：全给再在最后一步拒 SVG
        // 是更差的体验
        picker.accept = "image/jpeg,image/png,image/gif,image/bmp,image/webp,image/avif,image/heic";
        picker.addEventListener("change", async () => {
            const file = picker.files?.[0];
            if (!file) return;
            // 发送前缩小：卡片 132px 宽，整张照片会以相机尺寸存与重发
            const dataUrl = await shrinkToDataURL(file, 360).catch(() => null);
            if (!dataUrl) { S.win.toast("That file is not a picture."); return; }
            guard(async () => {
                const res = await setCover(entry.rel, dataUrl);
                if (!res?.ok) throw new Error(res?.message || "Could not save that cover.");
            }, "Cover set");
        });
        picker.click();
    },

    onClearCover(entry) {
        guard(() => clearCover(entry.rel), "Cover removed");
    },

    onContext(entry, e) {
        // 右键选中之外只作用于该卡；选中之内保持选中，菜单仍作用于全部
        if (!S.selected.has(entry.rel)) S.selected = new Set([entry.rel]);
        S.kbdRel = entry.rel;
        render();
        showCardMenu(entry, e.clientX, e.clientY);
    },

    onDragStart(entry, e) {
        // 拖未选卡片拖的是那张卡，不是旧选中
        if (!S.selected.has(entry.rel)) S.selected = new Set([entry.rel]);
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData(CARD_MIME, entry.rel);
        e.dataTransfer.setData("text/plain", entry.rel);
    },
};

/** 封面存小 JSON sidecar，12MP png 会撑爆它并拖慢每次打开。先缩小——
 *  132px 卡片只需要这些。 */
function shrinkToDataURL(file, maxW) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
            URL.revokeObjectURL(url);
            const scale = Math.min(1, maxW / (img.naturalWidth || maxW));
            const c = document.createElement("canvas");
            c.width = Math.max(1, Math.round((img.naturalWidth || maxW) * scale));
            c.height = Math.max(1, Math.round((img.naturalHeight || maxW) * scale));
            c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
            resolve(c.toDataURL("image/jpeg", 0.82));
        };
        img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("That file is not a picture.")); };
        img.src = url;
    });
}

// ── 卡片菜单 ──────────────────────────────────────────────────────────────

function showCardMenu(entry, x, y) {
    const many = [...S.selected];
    const multi = many.length > 1 && S.selected.has(entry.rel);
    const fav = S.favourites.has(entry.rel);

    if (multi) {
        openContextMenu(x, y, [
            { label: `${many.length} workflows selected`, disabled: true },
            null,
            { label: "Move to folder…", fn: () => promptMoveTo(many) },
            null,
            { label: `Delete ${many.length}…`, danger: true, fn: () => HANDLERS.onDeleteMany(many) },
        ]);
        return;
    }

    openContextMenu(x, y, [
        { label: "Open", fn: () => HANDLERS.onOpen(entry) },
        { label: fav ? "Remove from favourites" : "Add to favourites", fn: () => HANDLERS.onStar(entry) },
        null,
        { label: "Rename", fn: () => HANDLERS.onRename(entry) },
        { label: "Duplicate", fn: () => HANDLERS.onDuplicate(entry) },
        { label: "Move to folder…", fn: () => promptMoveTo([entry.rel]) },
        { label: hasHandCover(entry, S.meta) ? "Replace cover…" : "Set cover…",
          fn: () => HANDLERS.onSetCover(entry) },
        { label: "Remove cover", fn: () => HANDLERS.onClearCover(entry),
          disabled: !hasHandCover(entry, S.meta) },
        null,
        { label: "Reveal in explorer", fn: () => guard(() => reveal(entry.rel), "Opened the folder - look in your taskbar") },
        null,
        { label: "Delete…", danger: true, fn: () => HANDLERS.onDelete(entry) },
    ]);
}

function promptMoveTo(rels) {
    const folders = ["", ...S.folders];
    const r = S.win.el.getBoundingClientRect();
    openContextMenu(r.left + 60, r.top + 90, [
        { label: "Move to which folder?", disabled: true },
        null,
        ...folders.map((f) => ({
            label: f === "" ? "(no folder)" : f,
            fn: () => moveWorkflowsTo(rels, f),
        })),
    ]);
}

function moveWorkflowsTo(rels, folderPath) {
    guard(async () => {
        let moved = 0;
        const failed = [];
        for (const rel of rels) {
            const file = rel.slice(rel.lastIndexOf("/") + 1);
            const target = joinRel(folderPath, file);
            if (target === rel) continue;
            try {
                await renameOrMove(rel, target);
                await carryMeta(rel, target);
                forgetRel(rel, target);
                moved++;
            } catch (err) {
                failed.push(`${file} (${err.message || "failed"})`);
            }
        }
        if (failed.length) {
            throw new Error(moved
                ? `Moved ${moved}, but could not move ${failed.length}: ${failed.join("; ")}`
                : `Could not move: ${failed.join("; ")}`);
        }
        if (!moved) throw new Error("Already in that folder.");
    }, `Moved to ${folderPath || "the workflows folder"}`);
}

// ── 文件夹动作 ────────────────────────────────────────────────────────────

const parentOf = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");

function commitRename(entry, newName) {
    const clean = cleanName(newName);
    const bad = nameProblem(clean);
    if (bad) { S.win.toast(bad); return; }
    const target = joinRel(dirOf(entry.rel), clean + ".json");
    // 实际没变。同路径改名不无害：请求服务器把文件移到自身，carryMeta
    // 会把笔记与封面塌成一个 null
    if (target === entry.rel) return;
    guard(async () => {
        await renameOrMove(entry.rel, target);
        await carryMeta(entry.rel, target);
        forgetRel(entry.rel, target);
    }, "Renamed");
}

function startFolderRename(path, row) {
    beginFolderRename(row, path, (newName) => {
        const clean = cleanName(newName);
        const bad = nameProblem(clean);
        if (bad) { S.win.toast(bad); return; }
        const target = parentOf(path) ? `${parentOf(path)}/${clean}` : clean;
        guard(async () => {
            // 后代也重定向：只匹配精确路径会让改名后的父级每个子级仍记在
            // 旧前缀下
            const reparent = (p) => (p === path || p.startsWith(path + "/")
                ? target + p.slice(path.length) : p);

            // 三步，顺序让任何失败都赔不掉一张图：
            // 1) 把笔记/封面复制到新键（旧键不动）；失败 -> 无变化中止
            // 2) 磁盘改名；失败 -> 尽力移除副本
            // 3) 清旧键并带走顺序与颜色
            const newNotes = {};
            const newCovers = {};
            const oldNotes = {};
            const oldCovers = {};
            for (const [k, v] of Object.entries(S.meta.notes || {})) {
                const moved = reparent(k);
                if (moved !== k) { newNotes[moved] = v; oldNotes[k] = null; }
            }
            for (const [k, v] of Object.entries(S.meta.covers || {})) {
                const moved = reparent(k);
                if (moved !== k) { newCovers[moved] = v; oldCovers[k] = null; }
            }

            const preAdd = {};
            if (Object.keys(newNotes).length) preAdd.notes = newNotes;
            if (Object.keys(newCovers).length) preAdd.covers = newCovers;
            if (Object.keys(preAdd).length) {
                const resA = await saveMeta(preAdd);
                if (!resA || resA.ok === false) {
                    throw new Error("Could not save the folder's records, so nothing was renamed.");
                }
            }

            const undoCopies = async () => {
                if (!Object.keys(preAdd).length) return;
                const undo = {};
                if (preAdd.notes) undo.notes = Object.fromEntries(Object.keys(newNotes).map((k) => [k, null]));
                if (preAdd.covers) undo.covers = Object.fromEntries(Object.keys(newCovers).map((k) => [k, null]));
                try { await saveMeta(undo); } catch { /* 自愈遍会清理封面 */ }
            };

            let res;
            try {
                res = await folderAction({ action: "rename", path, newPath: target });
            } catch (err) {
                await undoCopies();
                throw err;
            }
            if (!res.ok) {
                await undoCopies();
                throw new Error(res.message || "Could not rename that folder.");
            }

            // 视图跟随磁盘（现已确定变化）：最终写入之前
            if (S.sel.kind === "folder" && typeof S.sel.value === "string") {
                const moved = reparent(S.sel.value);
                if (moved !== S.sel.value) S.sel = { kind: "folder", value: moved };
            }
            S.selected = new Set([...S.selected].map(reparent));
            if (S.kbdRel) S.kbdRel = reparent(S.kbdRel);

            const patch = {};
            const order = (S.meta.folderOrder || []).map(reparent);
            if (order.length) patch.folderOrder = order;
            const expanded = (S.meta.folderExpanded || []).map(reparent);
            if (expanded.length) patch.folderExpanded = expanded;
            const colours = {};
            for (const [k, v] of Object.entries(S.meta.folderColors || {})) {
                const moved = reparent(k);
                if (moved !== k) { colours[k] = null; colours[moved] = v; }
            }
            if (Object.keys(colours).length) patch.folderColors = colours;
            if (Object.keys(oldNotes).length) patch.notes = oldNotes;
            if (Object.keys(oldCovers).length) patch.covers = oldCovers;

            if (Object.keys(patch).length) {
                const res2 = await saveMeta(patch);
                if (!res2 || res2.ok === false) {
                    throw new Error("The folder was renamed and its notes and covers are safe, "
                        + "but its colour and place in the list could not be saved.");
                }
            }
        }, "Folder renamed");
    });
}

/** 写一个同级组的新顺序。 */
function commitSiblingOrder(sibs, reordered) {
    const others = (S.meta.folderOrder || []).filter((p) => !sibs.includes(p));
    const folderOrder = [...others, ...reordered];
    guard(async () => {
        const res = await saveMeta({ folderOrder });
        if (!res?.meta?.folderOrder || !res.meta.folderOrder.length) {
            throw new Error("Folder order could not be saved. Restart ComfyUI - this part needs the newer server files.");
        }
        S.meta.folderOrder = folderOrder;
    });
}

// 展开/收起写入链式而非并行：每个都发整个列表（列表分区整体替换），
// 两个并行会乱序落地
let expandWrites = Promise.resolve();

function setFolderExpanded(path, open) {
    const current = new Set(S.meta.folderExpanded || []);
    if (open === current.has(path)) return expandWrites;

    const before = [...current];
    if (open) current.add(path); else current.delete(path);
    const next = [...current];
    S.meta.folderExpanded = next;

    // 关闭正在看的分支会自我打架：选中文件夹的祖先在渲染时被强制展开。
    // 关闭它意味着"改看这个文件夹"，选中升到被关的那个
    if (!open && S.sel.kind === "folder" && typeof S.sel.value === "string"
        && S.sel.value.startsWith(path + "/")) {
        S.sel = { kind: "folder", value: path };
        S.selected = new Set();
        S.kbdRel = null;
    }
    render();

    expandWrites = expandWrites.then(async () => {
        try {
            const res = await saveMeta({ folderExpanded: next });
            if (!res || res.ok === false) {
                throw new Error("Your folder choice could not be saved. Something else may have the "
                    + "workflows folder open, or it is read-only.");
            }
            if (!Array.isArray(res?.meta?.folderExpanded)) {
                throw new Error("Restart ComfyUI - remembering open folders needs the newer server files.");
            }
        } catch (err) {
            S.meta.folderExpanded = before;
            render();
            S.win?.toast(err?.message || "Could not remember that folder.");
        }
    });
    return expandWrites;
}

/** 把文件夹在自己的同级里移一步。 */
function moveFolder(path, delta) {
    const sibs = siblingsOf(path, S.folders, S.meta.folderOrder);
    const at = sibs.indexOf(path);
    const to = at + delta;
    if (at < 0 || to < 0 || to >= sibs.length) return;
    const reordered = sibs.slice();
    reordered.splice(to, 0, reordered.splice(at, 1)[0]);
    commitSiblingOrder(sibs, reordered);
}

/** 把一文件夹丢到另一之上/之下。只重排，绝不移动——拖进另一文件夹会
 *  重写其下每个路径，是看起来不像的破坏性操作。 */
function reorderFolderByDrop(moved, target, above) {
    const parent = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");
    if (parent(moved) !== parent(target)) {
        S.win.toast("Folders can be re-ordered within the same level, not moved into each other.");
        return;
    }
    const sibs = siblingsOf(moved, S.folders, S.meta.folderOrder);
    const from = sibs.indexOf(moved);
    if (from < 0) return;
    const without = sibs.filter((p) => p !== moved);
    const at = without.indexOf(target);
    if (at < 0) return;
    const insert = above ? at : at + 1;
    without.splice(insert, 0, moved);
    if (without.join("|") === sibs.join("|")) return;
    commitSiblingOrder(sibs, without);
}

function showFolderMenu(path, ev) {
    const sibs = siblingsOf(path, S.folders, S.meta.folderOrder);
    const at = sibs.indexOf(path);
    const rowEl = ev.currentTarget;
    openContextMenu(ev.clientX, ev.clientY, [
        { label: "New folder inside", fn: () => createFolder(path) },
        { label: "Rename", fn: () => startFolderRename(path, rowEl) },
        { label: "Move up", fn: () => moveFolder(path, -1), disabled: at <= 0 },
        { label: "Move down", fn: () => moveFolder(path, 1), disabled: at < 0 || at >= sibs.length - 1 },
        null,
        { label: "Reveal in explorer", fn: () => guard(() => reveal(path), "Opened the folder - look in your taskbar") },
        null,
        {
            label: "Delete folder",
            danger: true,
            fn: () => guard(async () => {
                const res = await folderAction({ action: "delete", path });
                if (!res.ok) throw new Error(res.message || "Could not delete that folder.");
                if (S.sel.kind === "folder" && S.sel.value === path) S.sel = { kind: "all" };
                const kept = (S.meta.folderExpanded || [])
                    .filter((p) => p !== path && !p.startsWith(path + "/"));
                if (kept.length !== (S.meta.folderExpanded || []).length) {
                    S.meta.folderExpanded = kept;
                    try { await saveMeta({ folderExpanded: kept }); } catch { /* 仅外观 */ }
                }
            }, "Folder deleted"),
        },
    ]);
}

function createFolder(parent) {
    ask({
        title: parent ? "New folder inside" : "New folder",
        message: parent ? `It is created inside ${parent}.` : "It is created inside the workflows folder.",
        value: "",
        okLabel: "Create",
    }).then((nameRaw) => {
        if (!nameRaw) return;
        const clean = cleanName(nameRaw);
        const bad = nameProblem(clean);
        if (bad) { S.win.toast(bad); return; }
        const path = parent ? `${parent}/${clean}` : clean;
        guard(async () => {
            const res = await folderAction({ action: "create", path });
            if (!res.ok) throw new Error(res.message || "Could not create that folder.");
            if (parent) await setFolderExpanded(parent, true);
        }, "Folder created");
    });
}

function onPickFolder(pick) {
    if (pick.kind === "newfolder") {
        createFolder("");
        return;
    }
    S.sel = pick;
    S.selected = new Set();
    S.kbdRel = null;
    render();
}

function onDropOnFolder(folderPath) {
    const rels = [...S.selected];
    if (rels.length) moveWorkflowsTo(rels, folderPath);
}

// ── 窗口内工具栏行 ────────────────────────────────────────────────────────

function buildBar(bar) {
    bar.textContent = "";

    const search = el("div", "sf-wb-search");
    const input = el("input");
    input.type = "text";
    input.placeholder = "Search names, models, prompts...";
    input.title = "Searches inside the files too: a model or LoRA filename, a phrase from a prompt, or your own note";
    input.value = S.query || "";
    try { input.selectionStart = input.selectionEnd = input.value.length; } catch { /* 不可选 */ }
    input.addEventListener("input", () => {
        S.query = input.value;
        S.kbdRel = null;
        render();
    });
    search.append(input);
    bar.append(search);

    const seg = el("div", "sf-wb-seg");
    for (const [id, label, tip] of [
        ["grid", "Grid", "Picture cards, for browsing by eye"],
        ["list", "List", "A dense list, easier once you have hundreds"],
    ]) {
        const b = el("button", S.view === id ? "on" : "", label);
        b.type = "button";
        b.title = tip;
        b.addEventListener("click", () => {
            S.view = id;
            window.sfnodesSetSetting(VIEW_SETTING, id);
            buildBar(bar);
            render();
        });
        seg.append(b);
    }
    bar.append(seg);

    const sort = el("button", "sf-wb-tbtn", "Sort: " + SORT_LABELS[S.sort]);
    sort.type = "button";
    const why = sortDisabledReason();
    if (why) { sort.disabled = true; sort.title = why; } else { sort.title = "Change the order"; }
    sort.addEventListener("click", () => {
        if (sort.disabled) return;
        const order = Object.keys(SORT_LABELS);
        S.sort = order[(order.indexOf(S.sort) + 1) % order.length];
        window.sfnodesSetSetting(SORT_SETTING, S.sort);
        buildBar(bar);
        render();
    });
    bar.append(sort);
    S.sortBtn = sort;

    const openFolder = el("button", "sf-wb-tbtn", "Open folder");
    openFolder.type = "button";
    openFolder.title = "Open this folder on your computer. It opens behind the browser, so look in your taskbar.";
    openFolder.addEventListener("click", () => {
        const path = S.sel.kind === "folder" ? S.sel.value : "";
        guard(() => reveal(path), "Opened the folder - look in your taskbar");
    });
    bar.append(openFolder);

    const saveHere = el("button", "sf-wb-tbtn sf-wb-primary", "Save open workflow here");
    saveHere.type = "button";
    saveHere.title = "Save whatever is on the canvas into the selected folder";
    saveHere.addEventListener("click", onSaveHere);
    bar.append(saveHere);
}

function onSaveHere() {
    const folder = S.sel.kind === "folder" ? S.sel.value : "";
    const current = activePath();
    const suggested = current ? current.slice(current.lastIndexOf("/") + 1).replace(/\.json$/i, "") : "My workflow";
    ask({
        title: "Save the open workflow",
        message: folder ? `Into ${folder}` : "Into the workflows folder",
        value: suggested,
        okLabel: "Save",
    }).then((nameRaw) => {
        if (!nameRaw) return;
        const clean = cleanName(nameRaw);
        const bad = nameProblem(clean);
        if (bad) { S.win.toast(bad); return; }
        guard(() => saveCurrentAs(joinRel(folder, clean + ".json")), "Saved");
    });
}

// ── 键盘 ──────────────────────────────────────────────────────────────────

function gridColumns() {
    const grid = S.win?.main?.querySelector(".sf-wb-grid");
    if (!grid) return 1;
    const cols = getComputedStyle(grid).gridTemplateColumns;
    const n = cols ? cols.trim().split(/\s+/).filter(Boolean).length : 0;
    return Math.max(1, n);
}

function onPanelKeys(e) {
    // 改名框与搜索框 stopPropagation，输入不受影响。搜索框刻意放箭头通过
    let list = S.visible;
    if (S.sel.kind === "tidy") {
        // tidy 屏把同一批工作流分组显示，视觉顺序不是 S.visible 的顺序，
        // 一个工作流可出现两次。按渲染行读顺序，首次出现即一站
        const seenRel = new Set();
        list = [];
        for (const row of S.win.main.querySelectorAll(".sf-wb-tdrow[data-rel]")) {
            const rel = row.dataset.rel;
            if (seenRel.has(rel)) continue;
            seenRel.add(rel);
            const entry = S.byRel.get(rel);
            if (entry) list.push(entry);
        }
    }
    if (!list.length) return;
    const idx = S.kbdRel ? list.findIndex((x) => x.rel === S.kbdRel) : -1;

    const ARROWS = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: "up", ArrowDown: "down" };
    if (e.key in ARROWS) {
        // 左右属于有文本时的光标；上下总是导航
        const el0 = e.target;
        const horizontal = e.key === "ArrowLeft" || e.key === "ArrowRight";
        if (horizontal && el0 && el0.tagName === "INPUT" && (el0.value || "").length) {
            const at = el0.selectionStart ?? 0;
            const atEdge = e.key === "ArrowLeft" ? at === 0 : at >= el0.value.length;
            if (!atEdge || el0.selectionStart !== el0.selectionEnd) return;
        }
        e.preventDefault();
        const cols = (S.view === "list" || S.sel.kind === "tidy") ? 1 : gridColumns();
        const raw = ARROWS[e.key];
        const step = raw === "up" ? -cols : raw === "down" ? cols : raw;
        let next = idx < 0 ? (step > 0 ? 0 : list.length - 1) : idx + step;
        if (next < 0) next = raw === "up" ? Math.max(0, idx % cols) : 0;
        if (next > list.length - 1) next = list.length - 1;
        S.kbdRel = list[next].rel;
        S.selected = new Set([S.kbdRel]);
        render();
        S.win.main.querySelector(".kbd")?.scrollIntoView({ block: "nearest" });
        return;
    }
    if (e.key === "Enter") {
        e.preventDefault();
        const target = idx >= 0 ? list[idx] : list[0];
        if (target) HANDLERS.onOpen(target);
        return;
    }
    if (e.key === "F2") {
        e.preventDefault();
        const target = idx >= 0 ? list[idx] : null;
        if (target) HANDLERS.onRename(target);
    }
}

function buildFooter(foot) {
    foot.textContent = "";
    const hint = (keys, what) => {
        const w = el("span");
        w.append(el("b", null, keys), document.createTextNode(" " + what));
        foot.append(w);
    };
    hint("type", "search");
    hint("← → ↑ ↓", "move");
    hint("Enter", "open");
    hint("F2", "rename");
    hint("double click", "open");
    hint("drag", "onto a folder to move");
    hint("Esc", "close");
}

// ── 打开/关闭 ─────────────────────────────────────────────────────────────

function ensureWindow() {
    if (S.win) return S.win;
    S.win = createWorkflowWindow({
        onRender: (opts) => {
            if (opts?.resizeOnly) return;      // 缩放不重取文件夹
            if (opts?.repaintOnly) {            // 拖角时每次面板可见性变化
                // 文件夹改名中跳过：此渲染来自 pointermove 而非用户完成的
                // 动作，重建列会拆走带输入名的改名框
                if (S.win.el.querySelector("input.sf-wb-foldrename")) return;
                render();
                return;
            }
            buildBar(S.win.bar);
            buildFooter(S.win.foot);
            loadData().then(render);
        },
        onClose: () => {
            closeContextMenu();
            dropRename(false);
            closeAsk();
            syncButton();
        },
    });
    // 面板级键盘（提示说箭头移动选中，所以无论焦点在哪都必须工作）
    S.win.el.addEventListener("keydown", onPanelKeys);
    setMenuFocusHome(() => S.win?.focusSearch());
    setRenameLostNotifier((name) => {
        S.win?.toast(`Stopped renaming "${name}" - it is no longer on screen.`);
    });
    return S.win;
}

function toggle() {
    const win = ensureWindow();
    win.toggle();
    syncButton();
}

function syncButton() {
    if (!S.btn) return;
    S.btn.classList.toggle("sf-wb-btn-open", !!S.win?.isOpen());
}

// ── 工具栏按钮 ────────────────────────────────────────────────────────────

function mountToolbarButton() {
    if (document.querySelector(".sf-wb-btn")) return;
    injectWorkflowCSS();
    const settingsGroupEl = app.menu?.settingsGroup?.element;
    if (!settingsGroupEl) {
        // 冷启动菜单未就绪。重试几次后静默放弃
        if (mountToolbarButton._tries == null) mountToolbarButton._tries = 0;
        if (++mountToolbarButton._tries > 20) {
            console.warn("[sfnodes.Workflows] toolbar mount: app.menu.settingsGroup never appeared");
            return;
        }
        setTimeout(mountToolbarButton, 250);
        return;
    }

    const group = document.createElement("div");
    group.className = "comfyui-button-group sf-wb-group-btn";
    const btn = document.createElement("button");
    btn.className = "comfyui-button sf-wb-btn";
    btn.title = "SF Workflows: find, organise and open your workflows (Alt+Shift+W)";
    btn.append(el("span", "sf-wb-btn-icon"));
    btn.addEventListener("click", toggle);
    group.append(btn);
    settingsGroupEl.before(group);
    S.btn = btn;
    syncButton();
}

// ── 注册 ──────────────────────────────────────────────────────────────────
// 防重复注册（与 SFPromptTags 防双包装同款）：模块被求值两次（混用带戳/不带
// 戳的 import、前端热重载以不同 URL 重新加载）会让 keybinding 以同 id 注册
// 两次，ComfyUI 报 "Keybinding on Alt + w already exists"。

if (!app._sfWorkflowsRegistered) {
    app._sfWorkflowsRegistered = true;
    app.registerExtension({
    name: "sfnodes.WorkflowBrowser",
    commands: [{
        id: CMD_ID,
        label: "SF Workflows",
        icon: "sf-wb-cmd-icon",
        function: toggle,
    }],
    // Alt+Shift+W：原版 pixaroma 的 Workflows 面板占用 Alt+W，并存时同 combo
    // 注册会让后加载者抛 "Keybinding already exists"（ComfyUI 前端按 combo
    // 全局去重）。加 Shift 避开
    keybindings: [{ combo: { key: "w", alt: true, shift: true }, commandId: CMD_ID }],

    getCanvasMenuItems() {
        return [{ content: "🎞 SF Workflows", callback: toggle }];
    },

    async setup() {
        try {
            S.view = window.sfnodesGetSetting(VIEW_SETTING, "grid");
            const savedSort = window.sfnodesGetSetting(SORT_SETTING, null);
            S.sort = SORT_LABELS[savedSort] ? savedSort : "recent";
            const savedDensity = window.sfnodesGetSetting(DENSITY_SETTING, null);
            S.density = DENSITY[savedDensity] ? savedDensity : "m";
        } catch { /* 未注册设置，首次运行缺席 */ }
        applyDensity(S.density);
        mountToolbarButton();
        installOutputCoverCapture({
            getActiveRel: () => activePath(),
            saveMeta,
        });
    },
});
}
