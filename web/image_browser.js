import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import {
    escapeHtml,
    injectCSSOnce,
    copyText,
    sfToast,
    el,
    parseAnnotatedImageValue,
    buildSourceURL,
} from "./sf_common.js";
import { attachPopupDismiss, clampToViewport } from "./sf_popup.js";
import { loadWorkflowFromImageUrl } from "./sf_lora_shared_info.js";

const PAGE_SIZE = 50;
const SORT_KEY = "sfnodes_image_browser_sort";
const LOCATION_KEY = "sfnodes_image_browser_location";

function injectModalStyles() {
    injectCSSOnce("sf-imgbrowser-css", `
        .sf-imgbrowser-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7); z-index: 99999;
            display: flex; align-items: center; justify-content: center;
        }
        .sf-imgbrowser-modal {
            background: var(--sf-panel-bg); border-radius: 8px;
            width: 90%; height: 90%; max-width: 1400px;
            display: flex; flex-direction: column;
            box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        }
        .sf-imgbrowser-header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 12px 16px; border-bottom: 1px solid var(--sf-border-soft);
        }
        .sf-imgbrowser-header h3 {
            margin: 0; color: var(--sf-text); font-size: 16px;
        }
        .sf-imgbrowser-close {
            background: none; border: none; color: var(--sf-text-dim); font-size: 24px;
            cursor: pointer; padding: 0 4px;
        }
        .sf-imgbrowser-close:hover { color: var(--sf-text-strong); }
        .sf-imgbrowser-search {
            padding: 8px 16px; border-bottom: 1px solid var(--sf-border-soft);
        }
        .sf-imgbrowser-search input {
            width: 100%; padding: 6px 10px; border-radius: 4px;
            border: 1px solid var(--sf-border-soft); background: var(--sf-input-bg); color: var(--sf-text);
            box-sizing: border-box; outline: none;
        }
        .sf-imgbrowser-search input:focus { border-color: #89B; }
        .sf-imgbrowser-pathbar {
            display: flex; align-items: center; flex-wrap: wrap; gap: 4px;
            padding: 6px 16px; border-bottom: 1px solid var(--sf-border-soft);
            background: var(--sf-panel-bg-2); font-size: 13px; min-height: 32px;
        }
        .sf-imgbrowser-pathbar span {
            color: var(--sf-text-dim); cursor: pointer; padding: 2px 6px;
            border-radius: 3px; white-space: nowrap;
        }
        .sf-imgbrowser-pathbar span:hover { color: var(--sf-text); background: var(--sf-surface-hover); }
        .sf-imgbrowser-pathbar .sep { color: var(--sf-text-faint); cursor: default; padding: 0 2px; }
        .sf-imgbrowser-pathbar span:hover.sep { background: transparent; }
        .sf-imgbrowser-pathbar .current { color: #89B; cursor: default; }
        .sf-imgbrowser-pathbar .current:hover { background: transparent; }
        .sf-imgbrowser-crumbs {
            display: flex; align-items: center; flex-wrap: wrap; gap: 4px;
            flex: 1; min-width: 0;
        }
        .sf-imgbrowser-locate {
            margin-left: auto; flex: none;
            background: none; border: 1px solid var(--sf-border-soft); color: var(--sf-text-dim);
            padding: 2px 10px; border-radius: 3px; cursor: pointer;
            font-size: 12px; transition: 0.15s;
        }
        .sf-imgbrowser-locate:hover { border-color: #89B; color: var(--sf-text); }
        .sf-imgbrowser-sortbar {
            display: flex; align-items: center; gap: 4px;
            padding: 4px 16px; border-bottom: 1px solid var(--sf-border-soft);
            background: var(--sf-panel-bg-2); font-size: 12px;
        }
        .sf-imgbrowser-sortbar .label { color: var(--sf-text-faint); margin-right: 4px; }
        .sf-imgbrowser-sortbtn {
            background: none; border: 1px solid var(--sf-border-soft); color: var(--sf-text-dim);
            padding: 2px 10px; border-radius: 3px; cursor: pointer;
            font-size: 12px; transition: 0.15s;
        }
        .sf-imgbrowser-sortbtn:hover { border-color: #89B; color: var(--sf-text); }
        .sf-imgbrowser-sortbtn.active { border-color: #89B; color: #89B; background: rgba(136,153,187,0.1); }
        .sf-imgbrowser-sortbtn .arrow { margin-left: 4px; }
        .sf-imgbrowser-grid {
            flex: 1; overflow-y: auto; padding: 12px;
            display: flex; flex-wrap: wrap; gap: 10px;
            align-content: flex-start; min-height: 0;
        }
        .sf-imgbrowser-item {
            width: 140px; flex: 0 0 140px;
            border-radius: 6px; overflow: hidden; cursor: pointer;
            background: var(--sf-input-bg); border: 2px solid transparent;
            transition: border-color 0.2s;
        }
        .sf-imgbrowser-item:hover { border-color: #89B; }
        .sf-imgbrowser-item img {
            display: block; width: 100%; height: 140px;
            object-fit: cover; background: var(--sf-panel-bg-2);
        }
        .sf-imgbrowser-item-label {
            padding: 4px 6px; font-size: 11px; color: var(--sf-text);
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            background: color-mix(in srgb, var(--sf-panel-bg) 85%, transparent);
        }
        .sf-imgbrowser-item.selected { border-color: #6af; }
        .sf-imgbrowser-item.selected .sf-imgbrowser-item-label { color: #6af; }
        .sf-imgbrowser-item.folder {
            border-color: var(--sf-border-soft); background: var(--sf-panel-bg-2);
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            min-height: 162px;
        }
        .sf-imgbrowser-item.folder:hover { border-color: #89B; background: var(--sf-surface-hover); }
        .sf-imgbrowser-folder-icon {
            font-size: 40px; color: var(--sf-text-faint); line-height: 1;
        }
        .sf-imgbrowser-item.folder .sf-imgbrowser-item-label {
            width: 100%; text-align: center; color: var(--sf-text-dim);
            background: transparent;
        }
        .sf-imgbrowser-spinner {
            width: 100%; text-align: center; padding: 40px 0;
            color: var(--sf-text-dim); font-size: 14px;
        }
        .sf-imgbrowser-spinner::after {
            content: ""; display: inline-block; width: 24px; height: 24px;
            margin-left: 8px; vertical-align: middle;
            border: 2px solid var(--sf-border-soft); border-top-color: #89B;
            border-radius: 50%; animation: sf-spin 0.8s linear infinite;
        }
        @keyframes sf-spin { to { transform: rotate(360deg); } }
        .sf-imgbrowser-loadmore {
            width: 100%; text-align: center; padding: 16px;
            color: var(--sf-text-dim); font-size: 13px;
        }
        .sf-imgbrowser-error {
            width: 100%; text-align: center; padding: 30px;
            color: #f77; font-size: 14px;
        }
        .sf-imgbrowser-img-error {
            display: flex; align-items: center; justify-content: center;
            width: 100%; height: 140px; background: var(--sf-input-bg);
            color: var(--sf-text-faint); font-size: 12px;
        }
        .sf-imgbrowser-item { position: relative; }
        .sf-imgbrowser-del {
            position: absolute; top: 4px; right: 4px; z-index: 2;
            width: 24px; height: 24px; border-radius: 4px;
            background: rgba(200,0,0,0.8); border: none; cursor: pointer;
            display: none; align-items: center; justify-content: center;
            font-size: 14px; color: #fff; line-height: 1;
        }
        .sf-imgbrowser-item:hover .sf-imgbrowser-del { display: flex; }
        .sf-imgbrowser-del:hover { background: rgba(255,0,0,0.95); }
        .sf-imgbrowser-type-toggle {
            display: flex; gap: 0; border: 1px solid var(--sf-border-soft); border-radius: 4px; overflow: hidden;
        }
        .sf-imgbrowser-typebtn {
            background: var(--sf-panel-bg-2); border: none; color: var(--sf-text-dim); padding: 4px 14px;
            cursor: pointer; font-size: 13px; transition: 0.15s;
        }
        .sf-imgbrowser-typebtn:hover { color: var(--sf-text); background: var(--sf-surface-hover); }
        .sf-imgbrowser-typebtn.active { background: #89B; color: #fff; }
        .sf-imgbrowser-ctxmenu {
            position: fixed; z-index: 100000;
            background: var(--sf-panel-bg); border: 1px solid var(--sf-border-soft); border-radius: 6px;
            padding: 4px; min-width: 150px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
        }
        .sf-imgbrowser-ctxitem {
            padding: 6px 12px; font-size: 13px; color: var(--sf-text);
            border-radius: 4px; cursor: pointer; white-space: nowrap;
        }
        .sf-imgbrowser-ctxitem:hover { background: rgba(136,153,187,0.2); color: #fff; }
    `);
}

function getThumbUrl(item, type) {
    const params = new URLSearchParams({ path: item.path, type: type || "input" });
    return api.apiURL(`/api/sfnodes/images/thumb?${params}`);
}

function getImageFolderFromValue(value) {
    if (!value) return { type: "input", folder: "" };
    const isOutput = value.endsWith(" [output]");
    const path = isOutput ? value.slice(0, -9) : value;
    const parts = path.split("/");
    const folder = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
    return { type: isOutput ? "output" : "input", folder };
}

// 目录有效性：folder 为空恒有效（根目录），否则需存在至少一个文件位于该目录下
function folderExists(items, folder) {
    if (!folder) return true;
    const prefix = folder + "/";
    return items.some(it => it.path.startsWith(prefix));
}

// 打开图片浏览弹窗。默认宿主为 LoadImage 系节点（写 image widget）；
// 传入 opts.onPick(value, item, type) 时切换为"选择器"模式——选中项交给
// 宿主回调处理（widget 写入路径整体跳过，imageWidget 置 null，
// 选中高亮/定位按钮退化为无值状态），SFImageCropExpand 的 Browse 按钮复用。
export function showImageBrowser(node, opts = {}) {
    const pickHandler = typeof opts.onPick === "function" ? opts.onPick : null;
    injectModalStyles();

    let allItems = [];
    let currentType = "input";
    let currentFolder = "";
    let sortBy = "name";
    let sortAsc = true;
    let page = 0;
    let isLoadingMore = false;
    let hasMore = true;
    let currentFolderItems = [];

    const overlay = document.createElement("div");
    overlay.className = "sf-imgbrowser-overlay";

    overlay.innerHTML = `
        <div class="sf-imgbrowser-modal">
            <div class="sf-imgbrowser-header">
                <h3>Select Image</h3>
                <div class="sf-imgbrowser-type-toggle">
                    <button class="sf-imgbrowser-typebtn active" data-type="input">Input</button>
                    <button class="sf-imgbrowser-typebtn" data-type="output">Output</button>
                </div>
                <button class="sf-imgbrowser-close">&times;</button>
            </div>
            <div class="sf-imgbrowser-search">
                <input type="text" placeholder="Filter images..." autofocus>
            </div>
            <div class="sf-imgbrowser-pathbar">
                <div class="sf-imgbrowser-crumbs"></div>
                <button class="sf-imgbrowser-locate" title="跳转到当前选中文件所在目录">定位当前</button>
            </div>
            <div class="sf-imgbrowser-sortbar"></div>
            <div class="sf-imgbrowser-grid"></div>
        </div>
    `;

    document.body.appendChild(overlay);

    const grid = overlay.querySelector(".sf-imgbrowser-grid");
    const searchInput = overlay.querySelector(".sf-imgbrowser-search input");
    const pathbar = overlay.querySelector(".sf-imgbrowser-pathbar");
    const sortbar = overlay.querySelector(".sf-imgbrowser-sortbar");
    const closeBtn = overlay.querySelector(".sf-imgbrowser-close");

    // 选择器模式不碰 image widget（宿主自持状态）；默认模式维持原行为
    const imageWidget = pickHandler ? null : node.widgets.find(w => w.name === "image");
    const currentValue = imageWidget ? imageWidget.value : (opts.selectedValue || "");
    const typeToggle = overlay.querySelector(".sf-imgbrowser-type-toggle");

    // 拉取当前类型全量列表 → 校验目录有效性（失效回退根目录）→ 渲染
    function loadListAndRender() {
        grid.innerHTML = '<div class="sf-imgbrowser-spinner">Loading images</div>';
        api.fetchApi(`/api/sfnodes/images/list?type=${currentType}`)
            .then(r => { if (!r.ok) throw new Error("Failed to fetch images"); return r.json(); })
            .then(data => {
                allItems = data;
                if (!folderExists(allItems, currentFolder)) {
                    currentFolder = "";
                }
                saveLocationPref();
                loadCurrentFolder();
            })
            .catch(() => {
                grid.innerHTML = '<div class="sf-imgbrowser-error">Failed to load images</div>';
            });
    }

    function switchType(newType, folder) {
        if (newType === currentType) return;
        currentType = newType;
        typeToggle.querySelectorAll(".sf-imgbrowser-typebtn").forEach(btn => {
            btn.classList.toggle("active", btn.dataset.type === newType);
        });
        currentFolder = folder || "";
        page = 0;
        hasMore = true;
        isLoadingMore = false;
        loadListAndRender();
    }

    typeToggle.querySelectorAll(".sf-imgbrowser-typebtn").forEach(btn => {
        btn.addEventListener("click", () => switchType(btn.dataset.type));
    });

    // 定位到当前选中文件所在目录（显式触发，不做自动跟随——系统写入
    // 如蒙版编辑保存的 clipspace 会静默改写 image 值，自动跟随会割裂浏览上下文）
    // 选择器模式无 widget 值可定位，按钮隐藏
    const locateBtn = overlay.querySelector(".sf-imgbrowser-locate");
    if (pickHandler) locateBtn.style.display = "none";
    locateBtn.addEventListener("click", () => {
        const v = imageWidget ? imageWidget.value : "";
        const { type: t, folder: f } = getImageFolderFromValue(v);
        if (t === currentType) {
            // 同类型：allItems 已加载，本地校验目录有效性
            currentFolder = folderExists(allItems, f) ? f : "";
            page = 0;
            hasMore = true;
            isLoadingMore = false;
            grid.innerHTML = "";
            saveLocationPref();
            loadCurrentFolder();
        } else {
            // 跨类型：allItems 属旧类型不可校验，交给列表回调回退
            switchType(t, f);
        }
    });

    function saveSortPref() {
        try {
            localStorage.setItem(SORT_KEY, JSON.stringify({ sortBy, sortAsc }));
        } catch (e) { /* ignore */ }
    }

    function loadSortPref() {
        try {
            const raw = localStorage.getItem(SORT_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            return null;
        }
    }

    function saveLocationPref() {
        try {
            localStorage.setItem(LOCATION_KEY, JSON.stringify({ type: currentType, folder: currentFolder }));
        } catch (e) { /* ignore */ }
    }

    function loadLocationPref() {
        try {
            const raw = localStorage.getItem(LOCATION_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            return null;
        }
    }

    function close() {
        saveSortPref();
        closeContextMenu();
        overlay.remove();
    }

    closeBtn.addEventListener("click", close);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) close();
    });

    function getFolderContents(items, folder) {
        const prefix = folder ? folder + "/" : "";
        const folders = new Set();
        const files = [];

        for (const item of items) {
            if (!item.path.startsWith(prefix)) continue;
            const remaining = item.path.slice(prefix.length);
            if (remaining.includes("/")) {
                folders.add(remaining.split("/")[0]);
            } else {
                files.push(item);
            }
        }

        return { folders: [...folders].sort(), files };
    }

    function renderBreadcrumbs() {
        const crumbsEl = pathbar.querySelector(".sf-imgbrowser-crumbs");
        if (!currentFolder) {
            crumbsEl.innerHTML = '<span class="current">All Images</span>';
            return;
        }
        const parts = currentFolder.split("/");
        let html = '<span data-folder="">All Images</span>';
        let accumulated = "";
        for (let i = 0; i < parts.length; i++) {
            accumulated += (i > 0 ? "/" : "") + parts[i];
            const isLast = i === parts.length - 1;
            html += '<span class="sep">&rsaquo;</span>';
            // 目录名来自用户可写文件系统：文本与 data-folder 属性都必须转义
            // （含 < 或 " 的目录名会注入 HTML / 破坏属性）。
            if (isLast) {
                html += `<span class="current">${escapeHtml(parts[i])}</span>`;
            } else {
                html += `<span data-folder="${escapeHtml(accumulated)}">${escapeHtml(parts[i])}</span>`;
            }
        }
        crumbsEl.innerHTML = html;

        crumbsEl.querySelectorAll("[data-folder]").forEach(span => {
            span.addEventListener("click", () => {
                currentFolder = span.dataset.folder;
                page = 0;
                hasMore = true;
                isLoadingMore = false;
                grid.innerHTML = "";
                loadCurrentFolder();
            });
        });
    }

    function renderSortbar() {
        const active = (key) => sortBy === key ? "active" : "";
        const arrow = (key) => sortBy === key ? (sortAsc ? "\u25B2" : "\u25BC") : "";
        sortbar.innerHTML = `
            <span class="label">Sort</span>
            <button class="sf-imgbrowser-sortbtn ${active("name")}" data-sort="name">
                Name <span class="arrow">${arrow("name")}</span>
            </button>
            <button class="sf-imgbrowser-sortbtn ${active("mtime")}" data-sort="mtime">
                Date <span class="arrow">${arrow("mtime")}</span>
            </button>
        `;
        sortbar.querySelectorAll(".sf-imgbrowser-sortbtn").forEach(btn => {
            btn.addEventListener("click", () => {
                const key = btn.dataset.sort;
                if (sortBy === key) {
                    sortAsc = !sortAsc;
                } else {
                    sortBy = key;
                    sortAsc = key === "name";
                }
                loadCurrentFolder();
                saveSortPref();
            });
        });
    }

    function applySort(items) {
        const sorted = [...items];
        sorted.sort((a, b) => {
            if (a._isFolder && !b._isFolder) return -1;
            if (!a._isFolder && b._isFolder) return 1;
            let cmp;
            if (sortBy === "mtime" && !a._isFolder) {
                cmp = (a.mtime || 0) - (b.mtime || 0);
            } else {
                cmp = (a.name || a.path || "").localeCompare(b.name || b.path || "");
            }
            return sortAsc ? cmp : -cmp;
        });
        return sorted;
    }

    function renderFolderItem(folderName) {
        const div = document.createElement("div");
        div.className = "sf-imgbrowser-item folder";
        div.innerHTML = `
            <div class="sf-imgbrowser-folder-icon">&#128193;</div>
            <div class="sf-imgbrowser-item-label">${escapeHtml(folderName)}</div>
        `;
        div.addEventListener("click", () => {
            currentFolder = currentFolder ? currentFolder + "/" + folderName : folderName;
            page = 0;
            hasMore = true;
            isLoadingMore = false;
            grid.innerHTML = "";
            loadCurrentFolder();
        });
        return div;
    }

    function deleteImage(e, item) {
        e.stopPropagation();
        if (!confirm(`Delete "${item.path}"?`)) return;

        const params = new URLSearchParams({ path: item.path, type: "input" });
        api.fetchApi(`/api/sfnodes/images/delete?${params}`, { method: "DELETE" })
            .then(r => {
                if (!r.ok) throw new Error("Delete failed");
                allItems = allItems.filter(i => i.path !== item.path);
                currentFolderItems = currentFolderItems.filter(i => i.path !== item.path);
                loadCurrentFolder();
            })
            .catch(() => alert("Failed to delete image"));
    }

    // ── 右键菜单（复制提示词 / 载入工作流）─────────────────────────────
    // 单例菜单挂 body（z-index 高于弹窗 overlay），三关闭走 sf_popup 公共件。
    let ctxMenu = null;
    let ctxDetach = null;

    function closeContextMenu() {
        if (ctxDetach) { ctxDetach(); ctxDetach = null; }
        if (ctxMenu) { ctxMenu.remove(); ctxMenu = null; }
    }

    function ctxEntry(label, onClick) {
        const item = el("div", "sf-imgbrowser-ctxitem", label);
        item.addEventListener("click", () => {
            closeContextMenu();
            onClick();
        });
        return item;
    }

    function openContextMenu(e, item) {
        e.preventDefault();
        e.stopPropagation();
        closeContextMenu();

        // output 项拼 ComfyUI 注解后缀：prompt_reader/extract 与 /view 都按
        // annotated filepath 解析（get_annotated_filepath 同款语义）
        const annotated = currentType === "output" ? `${item.path} [output]` : item.path;
        const menu = el("div", "sf-imgbrowser-ctxmenu");
        menu.appendChild(ctxEntry("复制提示词", () => copyImagePrompt(annotated)));
        menu.appendChild(ctxEntry("载入工作流（新标签）", () => loadImageWorkflow(annotated)));

        document.body.appendChild(menu);
        menu.style.left = `${e.clientX}px`;
        menu.style.top = `${e.clientY}px`;
        clampToViewport(menu);
        ctxMenu = menu;
        ctxDetach = attachPopupDismiss(menu, { onClose: closeContextMenu });
    }

    function toast(severity, detail) {
        sfToast({ summary: "SF Image Browser", severity, detail, fallbackTag: "SF Image Browser" });
    }

    async function copyImagePrompt(annotated) {
        try {
            const r = await api.fetchApi(`/api/sfnodes/prompt_reader/extract?filename=${encodeURIComponent(annotated)}`);
            const data = await r.json();
            if (data.found && data.text) {
                const ok = await copyText(data.text);
                toast(ok ? "success" : "error", ok ? "正向提示词已复制到剪贴板" : "复制到剪贴板失败");
            } else {
                toast("warn", data.message || "未在图片元数据中找到提示词");
            }
        } catch (err) {
            toast("error", "读取提示词失败：" + (err.message || err));
        }
    }

    async function loadImageWorkflow(annotated) {
        const part = parseAnnotatedImageValue(annotated);
        const url = buildSourceURL(part);
        if (!url) return;
        close();
        try {
            await loadWorkflowFromImageUrl(url, (msg) => toast("error", msg));
        } catch (err) {
            toast("error", "载入工作流失败：" + (err.message || err));
        }
    }

    function renderImageItem(item) {
        const div = document.createElement("div");
        div.className = "sf-imgbrowser-item";
        if (item.path === currentValue || item.path + " [output]" === currentValue) {
            div.classList.add("selected");
        }

        if (currentType === "input") {
            const del = document.createElement("button");
            del.className = "sf-imgbrowser-del";
            del.textContent = "\u2716";
            del.addEventListener("click", (e) => deleteImage(e, item));
            div.appendChild(del);
        }

        const imgUrl = getThumbUrl(item, currentType);
        const img = document.createElement("img");
        img.src = imgUrl;
        img.alt = item.path;
        img.onerror = function () {
            this.style.display = "none";
            const fallback = document.createElement("div");
            fallback.className = "sf-imgbrowser-img-error";
            fallback.textContent = "\u2716";
            div.insertBefore(fallback, this.nextSibling);
        };

        const label = document.createElement("div");
        label.className = "sf-imgbrowser-item-label";
        label.textContent = item.path;

        div.appendChild(img);
        div.appendChild(label);

        div.addEventListener("click", () => {
            const value = currentType === "output" ? item.path + " [output]" : item.path;
            if (pickHandler) {
                pickHandler(value, item, currentType);
            } else if (imageWidget) {
                imageWidget.value = value;
                if (imageWidget.callback) {
                    imageWidget.callback(value);
                }
                node.setDirtyCanvas(true, true);
            }
            saveLocationPref();
            close();
        });

        div.addEventListener("contextmenu", (e) => openContextMenu(e, item));

        return div;
    }

    function renderPage() {
        if (!hasMore || isLoadingMore) return;
        isLoadingMore = true;

        const start = page * PAGE_SIZE;
        const end = start + PAGE_SIZE;
        const batch = currentFolderItems.slice(start, end);

        if (batch.length === 0) {
            isLoadingMore = false;
            hasMore = false;
            return;
        }

        const fragment = document.createDocumentFragment();
        for (const entry of batch) {
            if (entry._isFolder) {
                fragment.appendChild(renderFolderItem(entry.name));
            } else {
                fragment.appendChild(renderImageItem(entry));
            }
        }

        const loadmoreEl = grid.querySelector(".sf-imgbrowser-loadmore");
        if (loadmoreEl) {
            grid.insertBefore(fragment, loadmoreEl);
        } else {
            grid.appendChild(fragment);
        }

        const loadmore = grid.querySelector(".sf-imgbrowser-loadmore");
        if (loadmore) loadmore.remove();

        page++;

        if (page * PAGE_SIZE < currentFolderItems.length) {
            const loadEl = document.createElement("div");
            loadEl.className = "sf-imgbrowser-loadmore";
            loadEl.textContent = `Loading more... (${Math.min(page * PAGE_SIZE, currentFolderItems.length)} / ${currentFolderItems.length})`;
            grid.appendChild(loadEl);
        } else {
            hasMore = false;
        }

        isLoadingMore = false;

        if (hasMore && grid.scrollHeight <= grid.clientHeight) {
            requestAnimationFrame(() => renderPage());
        }
    }

    function loadCurrentFolder() {
        saveLocationPref();
        const q = searchInput.value.toLowerCase().trim();

        if (q) {
            currentFolderItems = allItems.filter(item =>
                item.path.toLowerCase().includes(q)
            ).map(item => ({ ...item, _isFolder: false }));
        } else {
            const { folders, files } = getFolderContents(allItems, currentFolder);
            currentFolderItems = [
                ...folders.map(name => ({ _isFolder: true, name })),
                ...files,
            ];
        }

        renderBreadcrumbs();
        renderSortbar();

        currentFolderItems = applySort(currentFolderItems);

        page = 0;
        hasMore = true;
        isLoadingMore = false;
        grid.innerHTML = "";

        if (currentFolderItems.length === 0) {
            grid.innerHTML = '<div class="sf-imgbrowser-error">No images found</div>';
            return;
        }

        renderPage();
    }

    grid.addEventListener("scroll", () => {
        if (!hasMore || isLoadingMore) return;
        if (grid.scrollTop + grid.clientHeight >= grid.scrollHeight - 200) {
            renderPage();
        }
    });

    searchInput.addEventListener("input", () => {
        loadCurrentFolder();
    });

    grid.innerHTML = '<div class="sf-imgbrowser-spinner">Loading images</div>';

    const pref = loadSortPref();
    if (pref) {
        sortBy = pref.sortBy || "name";
        sortAsc = pref.sortAsc !== undefined ? pref.sortAsc : true;
    }

    // 初始位置：优先恢复上次浏览位置（localStorage 记忆），无记忆 → input 根目录。
    // 不复用 widget 值推导——蒙版编辑等系统写入会静默改写 image 值（如
    // clipspace），跟随会割裂浏览上下文；widget 值只影响选中高亮。
    const loc = loadLocationPref();
    if (loc) {
        currentType = loc.type === "output" ? "output" : "input";
        currentFolder = typeof loc.folder === "string" ? loc.folder : "";
    }

    typeToggle.querySelectorAll(".sf-imgbrowser-typebtn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.type === currentType);
    });

    loadListAndRender();
}

app.registerExtension({
    name: "sfnodes.image_browser",
    nodeCreated(node) {
        if (node.comfyClass !== "SFLoadImageBrowser") return;

        node.addWidget("button", "Browse Images", null, () => {
            showImageBrowser(node);
        });

        const imageWidget = node.widgets.find(w => w.name === "image");
        if (imageWidget) {
            const origCB = imageWidget.callback;
            imageWidget.callback = function (value) {
                if (origCB) origCB.call(this, value);
                node.setDirtyCanvas(true, true);
            };
        }
    },
});

// 官方原生 LoadImage/LoadImageMask 的 Browse 按钮：选中值写回原生 `image`
// widget 并触发其 callback——核心 image_upload 借此刷新预览。原生 LoadImage
// 的 VALIDATE_INPUTS 带 image 参数，使后端 combo 列表校验被跳过（execution.py
// validate_prompt 的 `x not in validate_function_inputs` 守卫），故 output 图片
// 的 "xxx [output]" 注解值也可直接提交。
//
// 注意：新前端（1.53+）的“缺失媒体”校验只认 combo 成员资格——原生 image
// combo 仅列 input 根目录（LoadImage 用 os.listdir），子目录值会被误判
// missing（红框 + "A required media input has no file selected."，见 §119）。
// 因此值必须补进 options.values（核心上传流程 addToComboValues 同款不变量）。
export function ensureNativeImageOption(node) {
    const w = node?.widgets?.find(w => w.name === "image");
    const value = w?.value;
    if (typeof value !== "string" || !value.trim()) return false;
    const values = w.options?.values;
    if (!Array.isArray(values) || values.includes(value)) return false;
    values.push(value);
    return true;
}

// 递归列表（§120）：把 sfnodes 的 input 全量子目录图片列表合并进原生
// image combo 的 options.values，让原生下拉直接可选子目录图——仅前端
// 列表扩张，不触碰后端 INPUT_TYPES/VALIDATE_INPUTS。幂等（按值去重 +
// 排序），fetch 失败静默保持根目录列表。
export function mergeNativeImageOptions(node, paths) {
    const w = node?.widgets?.find(w => w.name === "image");
    const values = w?.options?.values;
    if (!Array.isArray(values) || !Array.isArray(paths) || paths.length === 0) return false;
    const seen = new Set(values);
    let changed = false;
    for (const p of paths) {
        if (typeof p !== "string" || !p || seen.has(p)) continue;
        seen.add(p);
        values.push(p);
        changed = true;
    }
    if (changed) values.sort();
    return changed;
}

export function applyNativeLoadImagePick(node, value) {
    const w = node?.widgets?.find(w => w.name === "image");
    if (!w || value == null) return false;
    w.value = value;
    ensureNativeImageOption(node);
    if (typeof w.callback === "function") w.callback(value);
    node.setDirtyCanvas?.(true, true);
    return true;
}

// 挂到官方原生加载节点的选择器模式扩展（精确 comfyClass 匹配，不误伤
// SFLoadImageBrowser——后者已自挂 Browse 按钮）。受设置开关门控（默认开）。
const NATIVE_LOAD_IMAGE_TYPES = ["LoadImage", "LoadImageMask"];

export const NATIVE_BROWSE_SETTING = "sfnodes.LoadImage.BrowseButton.Enabled";
const NATIVE_BROWSE_BUTTON_NAME = "Browse Images";

function isNativeBrowseEnabled() {
    try { return app.ui?.settings?.getSettingValue?.(NATIVE_BROWSE_SETTING) ?? true; }
    catch { return true; }
}

// 纯函数（好测）：按钮 widget 在 widgets 中的下标，无则 -1。
export function findNativeBrowseButton(widgets) {
    return (widgets || []).findIndex(w => w?.type === "button" && w.name === NATIVE_BROWSE_BUTTON_NAME);
}

function addNativeBrowseButton(node) {
    if (findNativeBrowseButton(node.widgets) !== -1) return;
    node.addWidget("button", NATIVE_BROWSE_BUTTON_NAME, null, () => {
        const imageWidget = node.widgets?.find(w => w.name === "image");
        showImageBrowser(node, {
            selectedValue: imageWidget?.value || "",
            onPick: (value) => applyNativeLoadImagePick(node, value),
        });
    });
}

// 递归图片列表：懒拉取一次（复用 Image Browser 的列表路由），每个原生节点
// 的 nodeCreated/loadedGraphNode 都会挂 then——先于 fetch 完成创建的节点在
// 完成时统一补齐，之后创建的节点微任务内即时合并。
const NATIVE_IMAGE_LIST_URL = "/api/sfnodes/images/list?type=input";
let _nativeImagePathsPromise = null;

function fetchNativeImagePaths() {
    if (!_nativeImagePathsPromise) {
        _nativeImagePathsPromise = api.fetchApi(NATIVE_IMAGE_LIST_URL, { cache: "no-store" })
            .then(r => (r.ok ? r.json() : null))
            .then(data => Array.isArray(data)
                ? data.map(it => it?.path).filter(p => typeof p === "string" && p)
                : null)
            .catch(() => null);
    }
    return _nativeImagePathsPromise;
}

function ensureNativeImageList(node) {
    fetchNativeImagePaths().then(paths => {
        if (!paths || !mergeNativeImageOptions(node, paths)) return;
        node.setDirtyCanvas?.(true, true);
    });
}

// 开关变更时对现存节点即时增删按钮（默认开；关闭后新节点也不再挂）。
function refreshNativeBrowseButtons() {
    const on = isNativeBrowseEnabled();
    const nodes = app.graph?._nodes || app.graph?.nodes || [];
    for (const n of nodes) {
        if (!NATIVE_LOAD_IMAGE_TYPES.includes(n?.comfyClass)) continue;
        const idx = findNativeBrowseButton(n.widgets);
        if (on && idx === -1) addNativeBrowseButton(n);
        else if (!on && idx !== -1) n.widgets.splice(idx, 1);
        n.setDirtyCanvas?.(true, true);
    }
    app.graph?.setDirtyCanvas?.(true, true);
}

let _nativeBrowseSettingRegistered = false;
function registerNativeBrowseSettingOnce() {
    if (_nativeBrowseSettingRegistered) return;
    _nativeBrowseSettingRegistered = true;
    try {
        app.ui.settings.addSetting({
            id: NATIVE_BROWSE_SETTING,
            name: "SF: show Browse Images button on native LoadImage / LoadImageMask",
            type: "boolean",
            defaultValue: true,
            // 官方 info 开关先例：onChange 时 store 尚未更新，延后一 tick 读值
            onChange: () => { setTimeout(refreshNativeBrowseButtons, 0); },
        });
    } catch { /* 设置系统不可用则退化为默认值（开） */ }
}

app.registerExtension({
    name: "sfnodes.native_load_image_browse",
    init() {
        registerNativeBrowseSettingOnce();
    },
    nodeCreated(node) {
        if (!NATIVE_LOAD_IMAGE_TYPES.includes(node?.comfyClass)) return;
        if (isNativeBrowseEnabled()) addNativeBrowseButton(node);
        ensureNativeImageList(node);
    },
    // 工作流加载后补回 options（configure 恢复的 widget 值先于此钩子，
    // loadedGraphNode 先于核心缺失媒体校验管线）——不受按钮设置门控：
    // 粘贴/上传产生的子目录值同样受益。
    loadedGraphNode(node) {
        if (!NATIVE_LOAD_IMAGE_TYPES.includes(node?.comfyClass)) return;
        ensureNativeImageList(node);
        if (ensureNativeImageOption(node)) node.setDirtyCanvas?.(true, true);
    },
});

// 在 window 捕获阶段抢先接管 SFLoadImageBrowser 上的图片拖拽，
// 避免被第三方扩展（如 Fill-Nodes 的 LoadImageDropFix）在 document 捕获阶段
// 劫持为"新建 LoadImage 节点"（其 isLoadImageNode 硬编码只识别 LoadImage/LoadImageMask）
function registerDropFix() {
    const getDropTarget = (event) => {
        const canvas = app.canvas;
        const graph = canvas?.graph;
        if (!canvas || !graph) return null;

        let node = null;
        const domNode = event.target?.closest?.("[data-node-id]");
        if (domNode) {
            node = graph.getNodeById?.(domNode.dataset.nodeId);
        }
        if (!node) {
            canvas.adjustMouseEvent?.(event);
            node = graph.getNodeOnPos?.(event.canvasX, event.canvasY) ?? null;
        }
        return node?.comfyClass === "SFLoadImageBrowser" ? node : null;
    };

    const isDraggingFiles = (event) => {
        return (
            Array.from(event.dataTransfer?.items ?? []).some(item => item.kind === "file") ||
            Array.from(event.dataTransfer?.types ?? []).includes("Files")
        );
    };

    const isImageDrop = (event) => {
        return Array.from(event.dataTransfer?.files ?? [])
            .some(file => file.type.startsWith("image/"));
    };

    window.addEventListener("dragover", (event) => {
        if (!getDropTarget(event) || !isDraggingFiles(event)) return;
        event.preventDefault();
    }, true);

    window.addEventListener("drop", async (event) => {
        const node = getDropTarget(event);
        if (!node || !isImageDrop(event)) return;

        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        try {
            await node.onDragDrop?.(event);
        } catch (err) {
            console.error("SFLoadImageBrowser drop failed:", err);
        } finally {
            // 原生 drop 处理器（负责清除 app.dragOverNode 高亮）被 stopPropagation 阻断，需自行清除
            app.dragOverNode = null;
            app.canvas?.setDirty?.(false, true);
        }
    }, true);
}

app.registerExtension({
    name: "sfnodes.image_browser_drop",
    setup() {
        registerDropFix();
    },
});
