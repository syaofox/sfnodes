// ==========================================================================
// Multi LoRA Loader - Folder Tree View for Dropdown
// ==========================================================================
//
// Description:
// JavaScript extension that provides folder tree view for LoRA dropdown menus
// in MultiLoraLoader and MultiLoraLoaderModelOnly nodes.
//
// Features:
// - Displays LoRA files in a collapsible folder tree structure
// - Supports nested subfolders (e.g., "sdxl/beauty.safetensors")
// - Toggle between List mode and Tree mode via settings
// - Collapsible folders with expand/collapse functionality
// - Filter search shows full path when typing
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { $el } from "/scripts/ui.js";

const DISPLAY_MODE = {
    LIST: 0,
    TREE: 1
};

app.registerExtension({
    name: "sfnodes.MultiLoraLoader.TreeView",

    init() {
        const displayOptions = {
            "List (flat)": DISPLAY_MODE.LIST,
            "Tree (subfolders)": DISPLAY_MODE.TREE
        };

        app.ui.settings.addSetting({
            id: "sfnodes.MultiLoraLoader.DisplayMode",
            name: "SF LoRA Loader: Multi-Lora display mode",
            defaultValue: DISPLAY_MODE.TREE,
            type: "combo",
            options: () => {
                return Object.entries(displayOptions).map(([text, value]) => ({
                    value,
                    text,
                    selected: app.ui.settings.getSettingValue("sfnodes.MultiLoraLoader.DisplayMode") == value
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
            const node = app.canvas.current_node;
            
            if (!node) return;
            
            const isMultiLora = node.comfyClass === "SFMultiLoraLoader" || 
                               node.comfyClass === "SFMultiLoraLoaderModelOnly";
            if (!isMultiLora) return;

            for (const mutation of mutations) {
                for (const added of mutation.addedNodes) {
                    if (added.classList?.contains("litecontextmenu")) {
                        const overWidget = app.canvas.getWidgetAtCursor();
                        
                        if (overWidget?.name?.match(/^lora_\d+_name$/)) {
                            requestAnimationFrame(() => {
                                if (!added.querySelector(".comfy-context-menu-filter")) return;
                                updateMenu(added, overWidget);
                            });
                        }
                        return;
                    }
                }
            }
        });

        mutationObserver.observe(document.body, { childList: true, subtree: false });

        const updateMenu = (menu, widget) => {
            const displayMode = app.ui.settings.getSettingValue("sfnodes.MultiLoraLoader.DisplayMode");
            
            const position = menu.getBoundingClientRect();
            const maxHeight = window.innerHeight - position.top - 20;
            menu.style.maxHeight = `${maxHeight}px`;

            if (displayMode === DISPLAY_MODE.TREE) {
                createTree(menu);
            }
        };

        const createTree = (menu) => {
            const items = menu.querySelectorAll(".litemenu-entry");
            if (!items.length) return;

            const folderMap = new Map();
            const rootItems = [];
            const splitBy = /\/|\\/;
            const itemsSymbol = Symbol("items");

            for (const item of items) {
                const path = item.getAttribute("data-value").split(splitBy);

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
                    innerHTML: `<span class="sfnodes-combo-folder-arrow">&#9658;</span> ${name}`,
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
