// SF Text Dropdown 前端扩展

import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

let cachedOptions = [];

async function loadConfigFromAPI() {
    try {
        const resp = await fetch("/api/sfnodes/text_dropdown/load");
        if (resp.ok) {
            const data = await resp.json();
            cachedOptions = data.options || [];
        }
    } catch (e) {
        console.error("[SFTextDropdown] Failed to load config:", e);
    }
    return cachedOptions;
}

app.registerExtension({
    name: "sfnodes.text_dropdown",

    nodeCreated(node) {
        if (node?.comfyClass !== "SFTextDropdown") return;

        const optionsWidget = node.widgets?.find((w) => w.name === "options_json");
        if (!optionsWidget) return;

        const origWidgets = {};
        for (const w of node.widgets) {
            origWidgets[w.name] = w;
        }

        for (const name in origWidgets) {
            const w = origWidgets[name];
            w.hidden = true;
            w.computeSize = () => [0, 0];
            w.draw = () => {};
        }

        node.widgets = node.widgets.filter(w =>
            w.name === "options_json" || w.name === "category" || w.name === "selected_text"
        );

        function parseOptions() {
            try {
                const data = JSON.parse(optionsWidget.value || "[]");
                if (!Array.isArray(data)) return cachedOptions;
                if (data.length === 0) return cachedOptions;
                return data.filter(x => x && typeof x === "object" && x.alias && x.content)
                    .map(x => ({
                        category: String(x.category || "default").trim() || "default",
                        alias: String(x.alias).trim(),
                        content: String(x.content)
                    }));
            } catch (e) {
                return cachedOptions;
            }
        }

        function getCategories(options) {
            const cats = new Set(options.map(o => o.category));
            return cats.size > 0 ? Array.from(cats).sort() : ["default"];
        }

        function getFilteredOptions(options, cat) {
            return options.filter(o => o.category === cat);
        }

        let catWidget, aliasWidget, contentWidget;
        let currentOptions = [];

        function updateContent(opt) {
            if (!contentWidget) return;
            const content = opt ? opt.content : "";
            contentWidget.value = content;
            origWidgets["selected_text"].value = content;
            node.setDirtyCanvas(true, true);
        }

        async function refreshWidgets() {
            const options = await loadConfigFromAPI();
            currentOptions = options;
            
            const categories = getCategories(options);
            const currentCat = catWidget ? catWidget.value : (categories[0] || "default");
            const validCat = categories.includes(currentCat) ? currentCat : (categories[0] || "default");
            
            if (catWidget) {
                catWidget.options.values = categories;
                catWidget.value = validCat;
                origWidgets["category"].value = validCat;
            }
            
            const filtered = getFilteredOptions(options, validCat);
            const aliases = filtered.map(o => o.alias);
            
            if (aliasWidget) {
                aliasWidget.options.values = aliases;
                if (aliases.length > 0 && aliases.includes(aliasWidget.value)) {
                    const opt = filtered.find(o => o.alias === aliasWidget.value);
                    updateContent(opt || filtered[0]);
                } else if (aliases.length > 0) {
                    aliasWidget.value = aliases[0];
                    updateContent(filtered[0]);
                } else {
                    aliasWidget.value = "";
                    updateContent(null);
                }
            }
            
            node.setDirtyCanvas(true, true);
        }

        async function init() {
            await loadConfigFromAPI();
            currentOptions = cachedOptions;
            
            const options = currentOptions;
            const categories = getCategories(options);

            const savedCat = origWidgets["category"]?.value || categories[0] || "default";
            const savedContent = origWidgets["selected_text"]?.value || "";

            const validCat = categories.includes(savedCat) ? savedCat : (categories[0] || "default");
            const filtered = getFilteredOptions(options, validCat);

            let defaultAlias = filtered[0]?.alias || "";
            let defaultContent = filtered[0]?.content || "";

            if (savedContent) {
                const savedOpt = options.find(o => o.content === savedContent);
                if (savedOpt && savedOpt.category === validCat) {
                    defaultAlias = savedOpt.alias;
                    defaultContent = savedContent;
                }
            }

            catWidget = node.addWidget(
                "combo",
                "category_select",
                validCat,
                async (value) => {
                    const opts = await loadConfigFromAPI();
                    currentOptions = opts;
                    const filtered = getFilteredOptions(opts, value);
                    const aliases = filtered.map(o => o.alias);
                    aliasWidget.options.values = aliases;
                    if (aliases.length > 0) {
                        aliasWidget.value = aliases[0];
                        updateContent(filtered[0]);
                    } else {
                        aliasWidget.value = "";
                        updateContent(null);
                    }
                    origWidgets["category"].value = value;
                    return value;
                },
                { values: categories }
            );
            const aliases0 = filtered.map(o => o.alias);
            aliasWidget = node.addWidget(
                "combo",
                "selected_alias",
                defaultAlias,
                async (value) => {
                    const opts = await loadConfigFromAPI();
                    currentOptions = opts;
                    const filtered = getFilteredOptions(opts, catWidget.value);
                    const opt = filtered.find(o => o.alias === value);
                    updateContent(opt || null);
                    return value;
                },
                { values: aliases0 }
            );

            contentWidget = ComfyWidgets["STRING"](
                node,
                "content_display",
                ["STRING", { multiline: true }],
                app
            ).widget;
            contentWidget.serialize = false;
            contentWidget.inputEl.readOnly = true;

            origWidgets["category"].value = validCat;
            origWidgets["selected_text"].value = defaultContent;
            updateContent(filtered[0] || null);

            setTimeout(() => {
                refreshWidgets();
            }, 100);
        }

        node._refreshTextDropdown = refreshWidgets;
        init();
    },
});

app.registerExtension({
    name: "sfnodes.text_dropdown.settings_listener",
    setup() {
        setInterval(async () => {
            for (const node of app.graph.nodes) {
                if (node?.comfyClass === "SFTextDropdown" && node._refreshTextDropdown) {
                    node._refreshTextDropdown();
                }
            }
        }, 2000);
    },
});
