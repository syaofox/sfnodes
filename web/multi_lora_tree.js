// ==========================================================================
// SF Combo Dropdown - Folder Tree View
// ==========================================================================
//
// Description:
// JavaScript extension that provides folder tree view for combo dropdown menus
// across all nodes (native LoRA/checkpoint pickers, third-party nodes, SF
// nodes): values containing "/" or "\" are grouped into collapsible folders.
// Menus without path-like values (e.g. sampler/scheduler names), menus with
// native combo group headers ("--xxx--", e.g. SFCanvasSizePreset), and LIST
// mode leave the menu untouched.
//
// Features:
// - Displays path values in a collapsible folder tree structure
// - Supports nested subfolders (e.g., "sdxl/beauty.safetensors")
// - Toggle between List mode and Tree mode via settings
// - Collapsible folders with expand/collapse functionality
// - Filter search shows full path when typing
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { $el } from "/scripts/ui.js";
import { escapeHtml } from "./sf_common.js";

const DISPLAY_MODE = {
    LIST: 0,
    TREE: 1
};

// 设置 id：sfnodes.Combo.* 自成分组，不再挂 LoraLoader 分组下。
// 旧 id（sfnodes.LoraLoader.DisplayMode）已废弃，不做迁移——默认值即 Tree，
// 曾手动设为 List 的用户升级后重设一次即可。
const DISPLAY_SETTING_ID = "sfnodes.Combo.DisplayMode";

app.registerExtension({
    name: "sfnodes.LoraLoader.TreeView",

    init() {
        const displayOptions = {
            "List (flat)": DISPLAY_MODE.LIST,
            "Tree (subfolders)": DISPLAY_MODE.TREE
        };

        app.ui.settings.addSetting({
            id: DISPLAY_SETTING_ID,
            name: "SF: combo dropdown display mode (flat list / folder tree)",
            defaultValue: DISPLAY_MODE.TREE,
            type: "combo",
            options: () => {
                return Object.entries(displayOptions).map(([text, value]) => ({
                    value,
                    text,
                    selected: app.ui.settings.getSettingValue(DISPLAY_SETTING_ID) == value
                }));
            },
            onChange: () => {
                app.graph.setDirtyCanvas(true);
            }
        });

        $el("style", {
            textContent: `
                .sfnodes-combo-folder {
                    opacity: 0.85;
                    font-weight: 500;
                    cursor: pointer;
                    user-select: none;
                }
                .sfnodes-combo-folder:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                }
                .sfnodes-combo-folder-arrow {
                    display: inline-block;
                    width: 15px;
                    text-align: center;
                }
                .sfnodes-combo-prefix {
                    display: none;
                    opacity: 0.6;
                    font-size: 0.9em;
                }
                .sfnodes-combo-folder-contents {
                    display: block;
                }

                /* When filter input has text, show flat list with paths */
                .litecontextmenu:has(input:not(:placeholder-shown)) .sfnodes-combo-folder-contents {
                    display: block !important;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .sfnodes-combo-folder {
                    display: none !important;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .sfnodes-combo-prefix {
                    display: inline;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .litemenu-entry {
                    padding-left: 2px !important;
                }
            `,
            parent: document.body,
        });
    },

    setup() {
        const mutationObserver = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const added of mutation.addedNodes) {
                    if (added.classList?.contains("litecontextmenu")) {
                        requestAnimationFrame(() => {
                            if (!added.querySelector(".comfy-context-menu-filter")) return;
                            updateMenu(added);
                        });
                        return;
                    }
                }
            }
        });

        mutationObserver.observe(document.body, { childList: true, subtree: false });

        const updateMenu = (menu) => {
            const displayMode = app.ui.settings.getSettingValue(DISPLAY_SETTING_ID);
            if (displayMode !== DISPLAY_MODE.TREE) return;

            // 非路径型菜单（选项值都不含子目录分隔符，如 sampler/scheduler 名）
            // 无分组意义，直接放行，避免无意义 DOM 重排。
            // 带 ComfyUI 原生 combo 分组头（"--xxx--"）的菜单自带分组，同样跳过
            // ——否则某个非路径值恰好含 "/"（如 SFCanvasSizePreset 的 model 标签
            // "Krea 2 (Turbo/RAW)"）会把整个下拉误当路径树折叠。
            const entries = menu.querySelectorAll(".litemenu-entry");
            let hasPath = false;
            let hasGroupHeader = false;
            for (const entry of entries) {
                const v = entry.getAttribute("data-value") ?? entry.textContent ?? "";
                if (/^--.*--$/.test(v.trim())) {
                    hasGroupHeader = true;
                } else if (v.includes("/") || v.includes("\\")) {
                    hasPath = true;
                }
                if (hasPath && hasGroupHeader) break;
            }
            if (!hasPath || hasGroupHeader) return;

            const position = menu.getBoundingClientRect();
            const maxHeight = window.innerHeight - position.top - 20;
            menu.style.maxHeight = `${maxHeight}px`;

            createTree(menu);
        };

        const createTree = (menu) => {
            const items = menu.querySelectorAll(".litemenu-entry");
            if (!items.length) return;

            const folderMap = new Map();
            const rootItems = [];
            const splitBy = /\/|\\/;
            const itemsSymbol = Symbol("items");

            for (const item of items) {
                const raw = item.getAttribute("data-value") || item.textContent || "";
                // ComfyUI 原生 combo 分组头（"--xxx--" 只显示不可选）：原位保留，
                // 不参与树分组（分组头文本若含 "/" 会被误拆成文件夹）
                if (/^--.*--$/.test(raw.trim())) continue;
                const path = raw.split(splitBy);

                item.textContent = path[path.length - 1];
                if (path.length > 1) {
                    const prefix = $el("span.sfnodes-combo-prefix", {
                        textContent: path.slice(0, -1).join("/") + "/",
                    });
                    item.prepend(prefix);
                }

                if (path.length === 1) {
                    rootItems.push(item);
                    continue;
                }

                item.remove();

                let currentLevel = folderMap;
                for (let i = 0; i < path.length - 1; i++) {
                    const folder = path[i];
                    if (!currentLevel.has(folder)) {
                        currentLevel.set(folder, new Map());
                    }
                    currentLevel = currentLevel.get(folder);
                }

                if (!currentLevel.has(itemsSymbol)) {
                    currentLevel.set(itemsSymbol, []);
                }
                currentLevel.get(itemsSymbol).push(item);
            }

            const createFolderElement = (name) => {
                return $el("div.litemenu-entry.sfnodes-combo-folder", {
                    // 文件夹名来自用户可写文件系统：必须转义，含 < 或 " 的名字
                    // 会注入 HTML。
                    innerHTML: `<span class="sfnodes-combo-folder-arrow">&#9658;</span> ${escapeHtml(name)}`,
                    style: { paddingLeft: "5px" },
                });
            };

            const insertFolderStructure = (parentElement, map, level = 0) => {
                for (const [folderName, content] of map.entries()) {
                    if (folderName === itemsSymbol) continue;

                    const folderElement = createFolderElement(folderName);
                    folderElement.style.paddingLeft = `${level * 10 + 5}px`;
                    parentElement.appendChild(folderElement);

                    const childContainer = $el("div.sfnodes-combo-folder-contents", {
                        style: { display: "none" },
                    });

                    const items = content.get(itemsSymbol) || [];
                    for (const item of items) {
                        item.style.paddingLeft = `${(level + 1) * 10 + 14}px`;
                        childContainer.appendChild(item);
                    }

                    insertFolderStructure(childContainer, content, level + 1);
                    parentElement.appendChild(childContainer);

                    folderElement.addEventListener("click", (e) => {
                        e.stopPropagation();
                        const arrow = folderElement.querySelector(".sfnodes-combo-folder-arrow");
                        const contents = folderElement.nextElementSibling;
                        if (contents.style.display === "none") {
                            contents.style.display = "block";
                            arrow.innerHTML = "&#9660;";
                        } else {
                            contents.style.display = "none";
                            arrow.innerHTML = "&#9658;";
                        }
                    });
                }
            };

            insertFolderStructure(items[0]?.parentElement || menu, folderMap);

            let left = app.canvas.last_mouse[0] - 10;
            let top = app.canvas.last_mouse[1] - 10;
            const body_rect = document.body.getBoundingClientRect();
            const root_rect = menu.getBoundingClientRect();

            if (body_rect.width && left > body_rect.width - root_rect.width - 10) {
                left = body_rect.width - root_rect.width - 10;
            }
            if (body_rect.height && top > body_rect.height - root_rect.height - 10) {
                top = body_rect.height - root_rect.height - 10;
            }

            menu.style.left = `${left}px`;
            menu.style.top = `${top}px`;
        };
    }
});
