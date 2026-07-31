import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PAGE_SIZE = 50;
const SORT_KEY = "sfnodes_image_browser_sort";

let modalStyleInjected = false;

function injectModalStyles() {
    if (modalStyleInjected) return;
    modalStyleInjected = true;
    const style = document.createElement("style");
    style.textContent = `
        .sf-imgbrowser-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7); z-index: 99999;
            display: flex; align-items: center; justify-content: center;
        }
        .sf-imgbrowser-modal {
            background: #2a2a2a; border-radius: 8px;
            width: 90%; height: 90%; max-width: 1400px;
            display: flex; flex-direction: column;
            box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        }
        .sf-imgbrowser-header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 12px 16px; border-bottom: 1px solid #444;
        }
        .sf-imgbrowser-header h3 {
            margin: 0; color: #ddd; font-size: 16px;
        }
        .sf-imgbrowser-close {
            background: none; border: none; color: #aaa; font-size: 24px;
            cursor: pointer; padding: 0 4px;
        }
        .sf-imgbrowser-close:hover { color: #fff; }
        .sf-imgbrowser-search {
            padding: 8px 16px; border-bottom: 1px solid #444;
        }
        .sf-imgbrowser-search input {
            width: 100%; padding: 6px 10px; border-radius: 4px;
            border: 1px solid #555; background: #1a1a1a; color: #ddd;
            box-sizing: border-box; outline: none;
        }
        .sf-imgbrowser-search input:focus { border-color: #89B; }
        .sf-imgbrowser-pathbar {
            display: flex; align-items: center; flex-wrap: wrap; gap: 4px;
            padding: 6px 16px; border-bottom: 1px solid #333;
            background: #222; font-size: 13px; min-height: 32px;
        }
        .sf-imgbrowser-pathbar span {
            color: #888; cursor: pointer; padding: 2px 6px;
            border-radius: 3px; white-space: nowrap;
        }
        .sf-imgbrowser-pathbar span:hover { color: #ddd; background: #333; }
        .sf-imgbrowser-pathbar .sep { color: #555; cursor: default; padding: 0 2px; }
        .sf-imgbrowser-pathbar span:hover.sep { background: transparent; }
        .sf-imgbrowser-pathbar .current { color: #89B; cursor: default; }
        .sf-imgbrowser-pathbar .current:hover { background: transparent; }
        .sf-imgbrowser-sortbar {
            display: flex; align-items: center; gap: 4px;
            padding: 4px 16px; border-bottom: 1px solid #333;
            background: #1e1e1e; font-size: 12px;
        }
        .sf-imgbrowser-sortbar .label { color: #666; margin-right: 4px; }
        .sf-imgbrowser-sortbtn {
            background: none; border: 1px solid #444; color: #888;
            padding: 2px 10px; border-radius: 3px; cursor: pointer;
            font-size: 12px; transition: 0.15s;
        }
        .sf-imgbrowser-sortbtn:hover { border-color: #89B; color: #ddd; }
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
            background: #1a1a1a; border: 2px solid transparent;
            transition: border-color 0.2s;
        }
        .sf-imgbrowser-item:hover { border-color: #89B; }
        .sf-imgbrowser-item img {
            display: block; width: 100%; height: 140px;
            object-fit: cover; background: #222;
        }
        .sf-imgbrowser-item-label {
            padding: 4px 6px; font-size: 11px; color: #aaa;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            background: rgba(0,0,0,0.6);
        }
        .sf-imgbrowser-item.selected { border-color: #6af; }
        .sf-imgbrowser-item.selected .sf-imgbrowser-item-label { color: #6af; }
        .sf-imgbrowser-item.folder {
            border-color: #555; background: #222;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            min-height: 162px;
        }
        .sf-imgbrowser-item.folder:hover { border-color: #89B; background: #2a2a2a; }
        .sf-imgbrowser-folder-icon {
            font-size: 40px; color: #666; line-height: 1;
        }
        .sf-imgbrowser-item.folder .sf-imgbrowser-item-label {
            width: 100%; text-align: center; color: #999;
            background: transparent;
        }
        .sf-imgbrowser-spinner {
            width: 100%; text-align: center; padding: 40px 0;
            color: #888; font-size: 14px;
        }
        .sf-imgbrowser-spinner::after {
            content: ""; display: inline-block; width: 24px; height: 24px;
            margin-left: 8px; vertical-align: middle;
            border: 2px solid #555; border-top-color: #89B;
            border-radius: 50%; animation: sf-spin 0.8s linear infinite;
        }
        @keyframes sf-spin { to { transform: rotate(360deg); } }
        .sf-imgbrowser-loadmore {
            width: 100%; text-align: center; padding: 16px;
            color: #888; font-size: 13px;
        }
        .sf-imgbrowser-error {
            width: 100%; text-align: center; padding: 30px;
            color: #f77; font-size: 14px;
        }
        .sf-imgbrowser-img-error {
            display: flex; align-items: center; justify-content: center;
            width: 100%; height: 140px; background: #1a1a1a;
            color: #555; font-size: 12px;
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
            display: flex; gap: 0; border: 1px solid #555; border-radius: 4px; overflow: hidden;
        }
        .sf-imgbrowser-typebtn {
            background: #2a2a2a; border: none; color: #888; padding: 4px 14px;
            cursor: pointer; font-size: 13px; transition: 0.15s;
        }
        .sf-imgbrowser-typebtn:hover { color: #ddd; background: #333; }
        .sf-imgbrowser-typebtn.active { background: #89B; color: #fff; }
    `;
    document.head.appendChild(style);
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

function showImageBrowser(node) {
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
            <div class="sf-imgbrowser-pathbar"></div>
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

    const imageWidget = node.widgets.find(w => w.name === "image");
    const currentValue = imageWidget ? imageWidget.value : "";
    const typeToggle = overlay.querySelector(".sf-imgbrowser-type-toggle");

    function switchType(newType) {
        if (newType === currentType) return;
        currentType = newType;
        typeToggle.querySelectorAll(".sf-imgbrowser-typebtn").forEach(btn => {
            btn.classList.toggle("active", btn.dataset.type === newType);
        });
        currentFolder = "";
        page = 0;
        hasMore = true;
        isLoadingMore = false;
        grid.innerHTML = '<div class="sf-imgbrowser-spinner">Loading images</div>';
        api.fetchApi(`/api/sfnodes/images/list?type=${currentType}`)
            .then(r => { if (!r.ok) throw new Error("Failed to fetch images"); return r.json(); })
            .then(data => {
                allItems = data;
                loadCurrentFolder();
            })
            .catch(() => {
                grid.innerHTML = '<div class="sf-imgbrowser-error">Failed to load images</div>';
            });
    }

    typeToggle.querySelectorAll(".sf-imgbrowser-typebtn").forEach(btn => {
        btn.addEventListener("click", () => switchType(btn.dataset.type));
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

    function close() {
        saveSortPref();
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
        if (!currentFolder) {
            pathbar.innerHTML = '<span class="current">All Images</span>';
            return;
        }
        const parts = currentFolder.split("/");
        let html = '<span data-folder="">All Images</span>';
        let accumulated = "";
        for (let i = 0; i < parts.length; i++) {
            accumulated += (i > 0 ? "/" : "") + parts[i];
            const isLast = i === parts.length - 1;
            html += '<span class="sep">&rsaquo;</span>';
            if (isLast) {
                html += `<span class="current">${parts[i]}</span>`;
            } else {
                html += `<span data-folder="${accumulated}">${parts[i]}</span>`;
            }
        }
        pathbar.innerHTML = html;

        pathbar.querySelectorAll("[data-folder]").forEach(el => {
            el.addEventListener("click", () => {
                currentFolder = el.dataset.folder;
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
            <div class="sf-imgbrowser-item-label">${folderName}</div>
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
            if (imageWidget) {
                const value = currentType === "output" ? item.path + " [output]" : item.path;
                imageWidget.value = value;
                if (imageWidget.callback) {
                    imageWidget.callback(value);
                }
                node.setDirtyCanvas(true, true);
            }
            close();
        });

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
            const el = document.createElement("div");
            el.className = "sf-imgbrowser-loadmore";
            el.textContent = `Loading more... (${Math.min(page * PAGE_SIZE, currentFolderItems.length)} / ${currentFolderItems.length})`;
            grid.appendChild(el);
        } else {
            hasMore = false;
        }

        isLoadingMore = false;

        if (hasMore && grid.scrollHeight <= grid.clientHeight) {
            requestAnimationFrame(() => renderPage());
        }
    }

    function loadCurrentFolder() {
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

    // Navigate to the folder of the currently selected image
    if (currentValue) {
        const { type: imgType, folder: imgFolder } = getImageFolderFromValue(currentValue);
        currentType = imgType;
        if (imgFolder) currentFolder = imgFolder;
    }

    typeToggle.querySelectorAll(".sf-imgbrowser-typebtn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.type === currentType);
    });

    api.fetchApi(`/api/sfnodes/images/list?type=${currentType}`)
        .then(r => { if (!r.ok) throw new Error("Failed to fetch images"); return r.json(); })
        .then(data => {
            allItems = data;
            loadCurrentFolder();
        })
        .catch(() => {
            grid.innerHTML = '<div class="sf-imgbrowser-error">Failed to load images</div>';
        });
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
