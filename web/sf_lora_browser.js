// ==========================================================================
// sf_lora_browser.js - SF LoRA 浏览器（主扩展）
// ==========================================================================
//
// 应用面板（无节点设计，同 sf_workflows*）：工具栏按钮（SF Workflows 按钮
// 旁）+ Alt+Shift+L + canvas 右键菜单打开。按文件夹层级浏览 models/loras
// 下全部 LoRA（展示对齐 SF Load Image Browser 的浏览器：面包屑 + 文件夹
// 下钻 + 当前层文件；搜索时跨层级扁平匹配），点击文件卡片打开 LoRA Stack
// 同款信息编辑面板（触发词/Civitai/描述/封面——复用 sf_lora_stack_info.js
// 的宿主适配入口 openInfoPanelFor，见该文件 ctx 文档）。
//
// 分层：
//   - 本文件：状态 + 服务端数据 + 信息面板宿主 ctx + 扩展注册
//   - sf_lora_browser_ui.js：窗口/面包屑/网格/CSS（不触碰服务端）
//   - sf_lora_browser_lib.js：纯函数（分层/过滤/排序，可 Node 单测）
//
// 后端零新增：列表/信息/缩略图/自定义词/描述/封面全部复用 SFLoraStack 的
// /api/sfnodes/lora_* 路由与 sf_lora_stack_api.js 封装。
//
// ==========================================================================
import { app } from "/scripts/app.js";
import { listLoras, thumbUrl } from "./sf_lora_stack_api.js";
import { accentOf, addLora, readState } from "./sf_lora_stack_core.js";
import { openInfoPanelFor, closeInfoPanel } from "./sf_lora_stack_info.js";
import {
    createLoraBrowserWindow, renderFolder, renderFlat, renderCrumbs,
    attachFlatScroll, injectBrowserCSS, el,
} from "./sf_lora_browser_ui.js";
import { filterLoras } from "./sf_lora_browser_lib.js";

const CMD_ID = "sfnodes.OpenLoraBrowser";
// 信息面板归属键：与 Stack（以节点对象为 key）互不冲突，closeInfoPanelFor
// 只会按 key 关闭自己的面板。
const BROWSER_KEY = "sfnodes.lora-browser";
// 浏览位置持久化设置键（记住上次所在文件夹：下次打开回到原处）
const FOLDER_KEY = "sfnodes.LoraBrowser.Folder";
// 展示模式持久化设置键（folder 层级浏览 / flat 平面列表）
const MODE_KEY = "sfnodes.LoraBrowser.Mode";
// 视图持久化设置键（grid 卡片网格 / list 列表行）
const VIEW_KEY = "sfnodes.LoraBrowser.View";
// 平面模式分批渲染步长：一次最多建这么多卡片，滚动接近底部再续一批
const FLAT_STEP = 60;

// ── 设置读写桥（与 sf_workflows.js 同款注入；幂等,不依赖其加载顺序）────────
if (typeof window.sfnodesGetSetting !== "function") {
    window.sfnodesGetSetting = (key, dflt) => {
        try { return app.ui?.settings?.getSettingValue(key) ?? dflt; } catch { return dflt; }
    };
}
if (typeof window.sfnodesSetSetting !== "function") {
    window.sfnodesSetSetting = (key, val) => {
        try {
            const s = app.ui?.settings;
            if (typeof s?.setSettingValueAsync === "function") s.setSettingValueAsync(key, val);
            else if (typeof s?.setSettingValue === "function") s.setSettingValue(key, val);
        } catch { /* 忽略 */ }
    };
}

// ── 状态 ────────────────────────────────────────────────────────────────────
const S = {
    win: null,      // 窗口 api（sf_lora_browser_ui）
    btn: null,      // 工具栏按钮
    list: [],       // 全量 LoRA 文件名（含子文件夹前缀）
    folder: "",     // 当前浏览目录（"" = 根）
    query: "",      // 搜索词（激活时列表转扁平匹配）
    sel: null,      // 当前选中（信息面板已打开）的 LoRA 名
    loading: false, // 列表加载中（首帧显示加载态）
    rows: new Map(),// 信息面板行宿主：name -> {id,name,triggers,custom}（会话内存）
    mode: "folder", // 展示模式：folder（层级浏览）| flat（平面列表+滚动加载）
    view: "grid",    // 视图：grid（卡片网格）| list（列表行）
    flat: { page: 0 }, // 平面模式批次游标
};

function readFolder() {
    try {
        const v = window.sfnodesGetSetting?.(FOLDER_KEY, "");
        return typeof v === "string" ? v : "";
    } catch { return ""; }
}
function saveFolder(f) {
    try { window.sfnodesSetSetting?.(FOLDER_KEY, f || ""); } catch { /* 忽略 */ }
}
function readMode() {
    try {
        return window.sfnodesGetSetting?.(MODE_KEY, "") === "flat" ? "flat" : "folder";
    } catch { return "folder"; }
}
function saveMode(m) {
    try { window.sfnodesSetSetting?.(MODE_KEY, m === "flat" ? "flat" : "folder"); } catch { /* 忽略 */ }
}
function readView() {
    try {
        return window.sfnodesGetSetting?.(VIEW_KEY, "") === "list" ? "list" : "grid";
    } catch { return "grid"; }
}
function saveView(v) {
    try { window.sfnodesSetSetting?.(VIEW_KEY, v === "list" ? "list" : "grid"); } catch { /* 忽略 */ }
}
// 目录有效性：目录被删除/改名后回根（有列表时才校得准；列表空时原样保留，
// 数据到达后再校一次——loadData 里）。
function validFolder(folder, list) {
    const f = String(folder || "").replace(/\/+$/, "");
    if (!f) return "";
    return (list || []).some((p) => String(p || "").startsWith(f + "/")) ? f : "";
}

// 信息面板行宿主：浏览器行只承载触发词勾选/自定义词的会话副本——真源始终
// 在服务器统一存储（saveCustomTriggers/saveCustomDescription 按 LoRA 名写回
// user/sfnodes/lora_triggers.json），行副本纯粹为了让面板有东西可读可写。
function getRowFor(name) {
    if (!S.rows.has(name)) S.rows.set(name, { id: name, name, triggers: [], custom: [] });
    return S.rows.get(name);
}
function patchRowFor(name, patch) { Object.assign(getRowFor(name), patch); }

// ── 信息面板（复用 LoRA Stack 编辑能力）──────────────────────────────────
function updateSelection(name) {
    // 仅更新选中高亮，避免整列表重建导致滚动回顶（平面模式尤为明显）
    try {
        const prev = S.win.main.querySelector(".sf-lb-card.sel, .sf-lb-row.sel");
        if (prev) prev.classList.remove("sel");
        // data-name 可能含特殊字符，用 CSS.escape
        const esc = (typeof CSS !== "undefined" && CSS.escape) ? CSS.escape(name) : String(name).replace(/["\\]/g, "\\$&");
        const next = S.win.main.querySelector(`[data-name="${esc}"]`);
        if (next) next.classList.add("sel");
        else throw new Error("not found");
        return true;
    } catch {
        return false;
    }
}

async function openInfoFor(name, card) {
    if (!name) return;
    await openInfoPanelFor({
        key: BROWSER_KEY,
        getRow: () => getRowFor(name),
        patchRow: (patch) => patchRowFor(name, patch),
        accent: accentOf(null),                 // 全局强调色（accentOf 忽略参数）
        anchorRect: () => (card?.isConnected ? card.getBoundingClientRect() : null),
        prefs: () => ({ civitai: true, thumbs: true }),
        refresh: () => {},
    }, name);
    if (S.sel !== name) {
        S.sel = name;
        if (!updateSelection(name)) render({ preserveScroll: true });
    }
}

// ── 双击：用 SF LoRA Stack 加载该 LoRA 并添加到当前工作流 ────────────────
// 在画布可视中心新建一个 SFLoraStack 节点并预置这一行（addLora 走 core 的
// 状态机写 node.properties.loraStackState；node._sfLsRefresh 由 stack 扩展的
// setupNode 挂上——renderNode + fitToContent 重渲染，是复用的公共入口）。
// 返回新建的节点；失败抛错由调用方 toast。
// 画布上所有 SFLoraStack 节点（type 或 comfyClass 匹配）。
function stackNodes() {
    return (app.graph?._nodes || []).filter((n) => n && (n.comfyClass === "SFLoraStack" || n.type === "SFLoraStack"));
}

// 向一个 Stack 节点插入 LoRA 行并重渲染（公共入口：新建/单节点插入/选择器共用）。
function addLoraRow(node, name) {
    const res = addLora(node, name);  // core 状态机写 node.properties.loraStackState
    if (!res?.ok) throw new Error(res?.reason === "max" ? "Stack is full." : "Could not add the LoRA.");
    node._sfLsRefresh?.(true);        // 重渲染 + 高度贴合
    try { app.canvas?.selectNode?.(node); } catch { /* Vue 无此 API，忽略 */ }
    return node;
}

// 无 Stack 节点时新建一个（画布可视中心 + 随机偏移防重叠）并插入该 LoRA。
async function createStackNode(name) {
    let node = null;
    // 1) 官方命令（Vue 新版前端自己的节点创建流程：类型注册/初始化/widget
    //    store 同步都走官方路径——实测裸 createNode + graph.add 在 Vue 下
    //    只弹成功 toast 却不渲染节点）。命令参数 {type}，创建后从 graph 里
    //    按新增找回来（不依赖命令返回值形状）。
    try {
        const cmd = app.extensionManager?.command;
        if (cmd && typeof cmd.execute === "function" && app?.graph) {
            const before = new Set(app.graph._nodes || []);
            await cmd.execute("Comfy.AddNode", { type: "SFLoraStack" });
            node = (app.graph._nodes || []).find((n) => !before.has(n)) || null;
        }
    } catch { node = null; }
    // 2) 兜底：LiteGraph.createNode + graph.add（Classic 前端/命令缺失）
    if (!node) {
        const LG = window.LiteGraph;
        if (!LG || !app?.graph) throw new Error("No canvas available.");
        node = LG.createNode("SFLoraStack");
        if (!node) throw new Error("SFLoraStack node type is not registered.");
        app.graph.add(node);   // 触发扩展 nodeCreated（setupNode 同步挂 _sfLsRefresh）
    }
    const [x, y] = canvasCenterPos();
    // 轻微随机偏移：连续双击两个 LoRA 不会严丝合缝叠在同一位置
    node.pos = [x + Math.round((Math.random() - 0.5) * 60), y + Math.round((Math.random() - 0.5) * 40)];
    addLoraRow(node, name);
    return node;
}

// 多个 Stack 节点时弹出选择器：列出每个节点（id + 已有 LoRA 数 + 首行名），
// 点行选择 / 点遮罩或 Esc 或 Cancel 取消。返回 Promise<node|null>。
function pickStackNode(nodes) {
    closeInfoPanel();   // 选择器盖住画布前收掉可能开着的信息面板
    return new Promise((resolve) => {
        const mask = el("div", "sf-lb-pick-mask");
        const box = el("div", "sf-lb-pick");
        box.append(el("div", "sf-lb-pick-t", "Choose a SF LoRA Stack"),
                   el("div", "sf-lb-pick-sub", nodes.length + " SF LoRA Stack nodes on the canvas."));
        const list = el("div", "sf-lb-pick-list");
        for (const n of nodes) {
            const st = readState(n);
            const count = st?.loras?.length || 0;
            const first = st?.loras?.[0]?.name ? " · " + st.loras[0].name : "";
            const meta = el("span", "sf-lb-pick-meta");
            meta.append(el("span", "n", count + " LoRA" + (count !== 1 ? "s" : "")), document.createTextNode(first));
            const row = el("div", "sf-lb-pick-row");
            row.dataset.nodeId = String(n.id);
            // title 优先显示用户改过的节点标题，缺省回退类名
            row.append(el("span", "sf-lb-pick-id", "#" + n.id),
                       el("span", "sf-lb-pick-title", n.title || "SFLoraStack"),
                       meta);
            row.title = "SF LoRA Stack #" + n.id;
            row.addEventListener("click", () => done(n));
            list.appendChild(row);
        }
        box.append(list);
        const cancel = el("div", "sf-lb-pick-cancel", "Cancel");
        cancel.addEventListener("click", () => done(null));
        box.append(cancel);
        mask.appendChild(box);
        document.body.appendChild(mask);
        function done(node) {
            document.removeEventListener("keydown", onKey, true);
            if (mask.isConnected) mask.remove();
            resolve(node);
        }
        const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); done(null); } };
        document.addEventListener("keydown", onKey, true);
        mask.addEventListener("pointerdown", (e) => { if (e.target === mask) done(null); });
    });
}

// 画布可视中心（图坐标）。ds 缺失（极端环境）回退到左上角附近。
function canvasCenterPos() {
    const ds = app.canvas?.ds;
    if (!ds || !ds.scale) return [80, 80];
    const cx = (window.innerWidth / 2 - (ds.offset?.[0] ?? 0)) / ds.scale;
    const cy = (window.innerHeight / 2 - (ds.offset?.[1] ?? 0)) / ds.scale;
    return [Math.max(0, Math.round(cx - 168)), Math.max(0, Math.round(cy - 60))];
}

async function addToWorkflow(name) {
    if (!name) return;
    try {
        const existing = stackNodes();
        if (!existing.length) {
            // 无 Stack 节点 → 新建并插入
            await createStackNode(name);
            S.win?.toast("Added \"" + name + "\" to a new SF LoRA Stack node.");
        } else if (existing.length === 1) {
            // 恰一个 → 直接插入
            addLoraRow(existing[0], name);
            S.win?.toast("Added \"" + name + "\" to the SF LoRA Stack node.");
        } else {
            // 多个 → 弹出选择器
            const picked = await pickStackNode(existing);
            if (!picked) { S.win?.toast("Cancelled."); return; }
            addLoraRow(picked, name);
            S.win?.toast("Added \"" + name + "\" to SF LoRA Stack #" + picked.id + ".");
        }
    } catch (e) {
        S.win?.toast("Could not add LoRA: " + String(e?.message || e));
    }
}

// ── 数据与渲染 ──────────────────────────────────────────────────────────────
async function loadData(force = false) {
    S.loading = true;
    try {
        S.list = await listLoras(force);       // api 模块内部已降级处理失败
        // 列表到达后校正浏览位置：目录可能已被删除/改名
        S.folder = validFolder(S.folder, S.list);
        saveFolder(S.folder);
    } catch { /* 保留旧列表 */ }
    S.loading = false;
    render();
}

let searchTimer = null;
function onSearchInput() {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
        S.query = S.win?.searchInput?.value || "";
        render();
    }, 180);
}

function render(opts = {}) {
    if (!S.win) return;
    const preserveScroll = !!opts.preserveScroll;
    const savedScroll = preserveScroll ? (S.win.main?.scrollTop ?? 0) : null;
    // 展示模式/视图切换控件高亮
    for (const b of S.win.segButtons || []) b.classList.toggle("on", b.dataset.mode === S.mode);
    for (const b of S.win.viewButtons || []) b.classList.toggle("on", b.dataset.view === S.view);

    if (S.mode === "flat") {
        // 平面模式：全部 LoRA 一次性列出（搜索时 = 扁平命中），分批渲染，
        // 滚动接近底部经 attachFlatScroll 续载。
        S.win.path.style.display = "none";
        const q = S.query.trim();
        const all = q ? filterLoras(S.list, q) : S.list.slice();
        const shownCount = Math.min(all.length, (S.flat.page + 1) * FLAT_STEP);
        renderFlat(S.win.main, {
            names: all,
            shown: shownCount,
            selectedName: S.sel,
            view: S.view,
            onPick: (name, card) => openInfoFor(name, card),
            onAdd: (name) => addToWorkflow(name),
        });
        // 视口还没被当前批填满（高窗口/小步长）-> 自动续批直到填满或到底
        // （有限步，列表为空即停）。
        if (shownCount < all.length
            && (S.win.main.scrollHeight || 0) <= (S.win.main.clientHeight || 0) + 8) {
            S.flat.page++;
            render({ preserveScroll: true });
            return;
        }
        S.win.setCount(all.length
            ? (shownCount < all.length ? shownCount + " / " + all.length : String(all.length))
            : "");
    } else {
        // 文件夹模式：面包屑 + 层级下钻
        S.win.path.style.display = "";
        renderCrumbs(S.win.path, S.folder, (folder) => {
            S.folder = folder || "";
            saveFolder(S.folder);
            render();
        });
        renderFolder(S.win.main, {
            list: S.list,
            folder: S.folder,
            query: S.query,
            selectedName: S.sel,
            view: S.view,
            onPick: (name, card) => openInfoFor(name, card),
            onAdd: (name) => addToWorkflow(name),
            onEnterFolder: (name) => {
                S.folder = S.folder ? S.folder + "/" + name : name;
                saveFolder(S.folder);
                render();
            },
        });
        // 标题计数：无搜索 = 总数；有搜索 = 命中数 / 总数
        const q = S.query.trim();
        const total = S.list.length;
        const shown = q ? filterLoras(S.list, q).length : total;
        S.win.setCount(total ? (q ? shown + " / " + total : String(total)) : "");
    }
    if (S.loading && !S.list.length) {
        // 首帧仍在拉列表：占位提示
        const empty = S.win.main.querySelector(".sf-lb-empty");
        if (empty) empty.textContent = "Loading LoRAs…";
    }
    if (preserveScroll && savedScroll != null) {
        const restore = () => { try { S.win.main.scrollTop = savedScroll; } catch {} };
        restore();
        // content-visibility:auto 会使 scrollHeight 延迟计算，下一帧再试一次
        try { requestAnimationFrame(restore); } catch { setTimeout(restore, 16); }
    }
}

// 切换视图（grid 卡片 / list 列表行）：记忆
function switchView(view) {
    const v = view === "list" ? "list" : "grid";
    if (S.view === v) return;
    S.view = v;
    saveView(v);
    render();
}

// 切换展示模式（folder 层级 / flat 平面）：重置平面批次游标并记忆
function switchMode(mode) {
    const m = mode === "flat" ? "flat" : "folder";
    if (S.mode === m) return;
    S.mode = m;
    S.flat.page = 0;
    saveMode(m);
    render();
}

// 任一入口（本浏览器/Stack）改了某个 LoRA 的数据 -> 刷新它的封面
// （缩略图路由 max-age=3600，加时间戳 bust 越过浏览器缓存）。
function refreshCardThumb(name) {
    if (!name || !S.win?.isOpen()) return;
    const cards = S.win.main.querySelectorAll?.(".sf-lb-card");
    for (const c of cards || []) {
        if (c.dataset.name !== name) continue;
        const th = c.querySelector(".sf-lb-thumb");
        if (th && th.src && th.isConnected) th.src = thumbUrl(name, Date.now());
        return;
    }
}
document.addEventListener("sfnodes.lora-data-changed", (e) => refreshCardThumb(e?.detail?.name));

// ── 窗口 ────────────────────────────────────────────────────────────────────
function ensureWindow() {
    if (S.win) return S.win;
    injectBrowserCSS();
    S.win = createLoraBrowserWindow({
        onRender: () => render(),
        onClose: () => syncButton(),
    });
    S.win.searchInput.addEventListener("input", onSearchInput);
    S.win.refreshBtn.addEventListener("click", () => loadData(true));
    for (const b of S.win.segButtons || []) {
        b.addEventListener("click", () => switchMode(b.dataset.mode));
    }
    for (const b of S.win.viewButtons || []) {
        b.addEventListener("click", () => switchView(b.dataset.view));
    }
    // 平面模式滚动续载：接近底部推进一批（render 会判断是否还有更多）
    attachFlatScroll(S.win.main, () => {
        if (S.mode !== "flat") return;
        const q = S.query.trim();
        const all = q ? filterLoras(S.list, q) : S.list;
        if ((S.flat.page + 1) * FLAT_STEP >= all.length) return;   // 已全部渲染
        S.flat.page++;
        render({ preserveScroll: true });
    });
    return S.win;
}

function syncButton() {
    if (!S.btn) return;
    S.btn.classList.toggle("sf-lb-btn-open", !!S.win?.isOpen());
}

function toggle() {
    const win = ensureWindow();
    if (win.isOpen()) {
        closeInfoPanel();   // 开着信息面板时关窗口一并收掉
        win.close();
    } else {
        // 恢复上次浏览位置与展示模式（打开的瞬间列表可能还没到——位置先记
        // 下，loadData 到达后再做目录有效性校正）。
        S.folder = readFolder();
        S.mode = readMode();
        S.view = readView();
        S.flat.page = 0;
        win.open();
        loadData(true);     // 打开即强制刷新（文件夹/改名可能已变化）
    }
    syncButton();
}

// ── 工具栏按钮（插在 SF Workflows 按钮旁）────────────────────────────────
function mountToolbarButton() {
    if (document.querySelector(".sf-lb-btn")) return;
    injectBrowserCSS();
    const settingsGroupEl = app.menu?.settingsGroup?.element;
    if (!settingsGroupEl) {
        // 冷启动菜单未就绪。重试几次后静默放弃
        if (mountToolbarButton._tries == null) mountToolbarButton._tries = 0;
        if (++mountToolbarButton._tries > 20) {
            console.warn("[sfnodes.LoraBrowser] toolbar mount: app.menu.settingsGroup never appeared");
            return;
        }
        setTimeout(mountToolbarButton, 250);
        return;
    }

    const group = document.createElement("div");
    group.className = "comfyui-button-group sf-lb-group-btn";
    const btn = document.createElement("button");
    btn.className = "comfyui-button sf-lb-btn";
    btn.title = "SF LoRA Browser: browse and edit all your LoRAs (Alt+Shift+L)";
    btn.append(el("span", "sf-lb-btn-icon"));
    btn.addEventListener("click", toggle);
    group.append(btn);

    // 紧贴 SF Workflows 按钮：已挂载则插其 group 之后，否则兜底插 settings
    // 组前（workflows 后挂载时两者仍相邻）。
    const wfBtn = document.querySelector(".sf-wb-btn");
    if (wfBtn) {
        (wfBtn.closest(".comfyui-button-group") || wfBtn.parentElement)?.after(group);
    } else {
        settingsGroupEl.before(group);
    }
    S.btn = btn;
    syncButton();
}

// ── 注册（防重复：模块可能被求值两次）────────────────────────────────────
if (!app._sfLoraBrowserRegistered) {
    app._sfLoraBrowserRegistered = true;
    app.registerExtension({
        name: "sfnodes.LoraBrowser",

        commands: [{
            id: CMD_ID,
            label: "SF LoRA Browser",
            icon: "sf-lb-cmd-icon",
            function: toggle,
        }],
        // Alt+Shift+L：包内仅有 workflows 的 Alt+Shift+W，组合罕见、冲突面
        // 小。若第三方包已占用会在注册时报错——换一个修饰键组合即可。
        keybindings: [{ combo: { key: "l", alt: true, shift: true }, commandId: CMD_ID }],

        getCanvasMenuItems() {
            return [{ content: "📚 SF LoRA Browser", callback: toggle }];
        },

        async setup() {
            mountToolbarButton();
        },
    });
}
