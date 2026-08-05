import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const DISABLED = "禁用";

const WIDGET_CATEGORY = {
    celebrity_preset: "Celebrity",
    outfit_preset: "Outfit",
    pose_preset: "Pose",
    couple_preset: "Couple Pose",
    environment_preset: "Environment",
    lighting_preset: "Lighting",
    style_preset: "Style",
    camera_angle_preset: "Camera Angle",
    camera_distance_preset: "Camera Distance",
    camera_lens_preset: "Camera Lens",
};

// 分类 tab 顺序（与后端 _CATEGORY_KEYS 一致）
const CATEGORY_TABS = [
    ["Celebrity", "名人"],
    ["Outfit", "服装"],
    ["Pose", "单人动作"],
    ["Couple Pose", "双人动作"],
    ["Environment", "环境"],
    ["Lighting", "灯光"],
    ["Style", "风格"],
    ["Camera Angle", "镜头角度"],
    ["Camera Distance", "镜头距离"],
    ["Camera Lens", "镜头"],
];

let presetData = null; // { 分类: { 选项名: {description, group} } }

// 把当前选中预设的英文 description 写入 widget.tooltip，
// ComfyUI（新旧前端）均优先显示 widget.tooltip，随选中值变化自动更新
function syncWidgetTooltip(w) {
    const category = WIDGET_CATEGORY[w?.name];
    if (!category) return;
    const desc = presetData?.[category]?.[w.value]?.description;
    w.tooltip = desc || null;
}

// ---------------- 分组选择器弹窗 ----------------
let pickerEl = null;
let pickerCategory = "Celebrity";
let pickerGroup = null; // null = 全部
let pickerSearch = "";
let lastPickerCategory = null;
let lastPickerGroup = null;
let pickerStyleInjected = false;

const PICKER_CSS = `
.sf-preset-picker-overlay{position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,0.45);display:flex;align-items:center;justify-content:center;}
.sf-preset-picker{background:#232323;border:1px solid #555;border-radius:8px;box-shadow:0 8px 30px rgba(0,0,0,0.6);width:min(520px,92vw);max-height:80vh;display:flex;flex-direction:column;font-size:12px;color:#ddd;}
.sf-preset-picker-head{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;border-bottom:1px solid #3a3a3a;font-weight:600;font-size:13px;}
.sf-preset-picker-close{background:none;border:none;color:#aaa;font-size:16px;cursor:pointer;padding:0 4px;}
.sf-preset-picker-close:hover{color:#fff;}
.sf-preset-picker-tabs{display:flex;flex-wrap:wrap;gap:4px;padding:8px 12px;border-bottom:1px solid #3a3a3a;}
.sf-preset-picker-tab{padding:3px 10px;border-radius:4px;cursor:pointer;background:#2c2c2c;border:1px solid transparent;color:#bbb;}
.sf-preset-picker-tab:hover{background:#3a3a3a;color:#fff;}
.sf-preset-picker-tab.active{background:#3a5f8a;border-color:#4a7ab0;color:#fff;}
.sf-preset-picker-groups{display:flex;flex-wrap:wrap;gap:4px;padding:0 12px 8px;}
.sf-preset-picker-group-chip{padding:2px 10px;border-radius:10px;cursor:pointer;background:#2c2c2c;color:#bbb;font-size:11px;border:1px solid transparent;}
.sf-preset-picker-group-chip:hover{background:#3a3a3a;color:#fff;}
.sf-preset-picker-group-chip.active{background:#3a5f8a;border-color:#4a7ab0;color:#fff;}
.sf-preset-picker-search{margin:8px 12px;padding:5px 8px;background:#2c2c2c;border:1px solid #555;border-radius:4px;color:#ddd;font-size:12px;box-sizing:border-box;width:calc(100% - 24px);}
.sf-preset-picker-list{overflow-y:auto;padding:4px 8px 12px;min-height:200px;}
.sf-preset-picker-group{position:sticky;top:0;background:#232323;padding:5px 8px 3px;font-size:11px;color:#9a9a9a;font-weight:600;border-bottom:1px solid #3a3a3a;margin-top:4px;}
.sf-preset-picker-item{padding:5px 8px;border-radius:4px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sf-preset-picker-item:hover{background:#3a3a3a;}
.sf-preset-picker-item.active{background:#3a5f8a;}
.sf-preset-picker-empty{padding:20px;text-align:center;color:#888;}
.sf-preset-picker-preview{padding:6px 12px;border-top:1px solid #3a3a3a;color:#aaa;font-size:11px;line-height:1.5;min-height:16px;max-height:48px;overflow:hidden;}
`;

function injectPickerStyle() {
    if (pickerStyleInjected) return;
    pickerStyleInjected = true;
    const style = document.createElement("style");
    style.textContent = PICKER_CSS;
    document.head.appendChild(style);
}

function widgetForCategory(category) {
    const name = Object.keys(WIDGET_CATEGORY).find((k) => WIDGET_CATEGORY[k] === category);
    if (!name) return null;
    return app.graph?._nodes
        ?.flatMap((n) => n.widgets ?? [])
        .find((w) => w.name === name);
}

function closePicker() {
    if (pickerEl) {
        pickerEl.remove();
        pickerEl = null;
    }
    lastPickerCategory = pickerCategory;
    lastPickerGroup = pickerGroup;
    pickerGroup = null;
    pickerSearch = "";
}

function renderPickerList() {
    if (!pickerEl) return;
    const listEl = pickerEl.querySelector(".sf-preset-picker-list");
    const widget = widgetForCategory(pickerCategory);
    const categoryData = presetData?.[pickerCategory] ?? {};
    const entries = Object.entries(categoryData);
    const filtered = entries.filter(([v, meta]) =>
        (!pickerGroup || meta?.group === pickerGroup) &&
        (!pickerSearch || v.toLowerCase().includes(pickerSearch))
    );

    listEl.replaceChildren();
    if (filtered.length === 0) {
        const empty = document.createElement("div");
        empty.className = "sf-preset-picker-empty";
        empty.textContent = "无匹配选项";
        listEl.appendChild(empty);
        return;
    }

    const previewEl = pickerEl.querySelector(".sf-preset-picker-preview");
    const addItem = (value, meta) => {
        const item = document.createElement("div");
        item.className = "sf-preset-picker-item" + (widget?.value === value ? " active" : "");
        item.textContent = value;
        item.addEventListener("mouseenter", () => {
            if (previewEl && meta?.description) {
                previewEl.textContent = meta.description;
            }
        });
        item.addEventListener("mouseleave", () => {
            if (previewEl) previewEl.textContent = "";
        });
        item.addEventListener("click", () => {
            if (!widget) return;
            widget.value = value;
            widget.callback?.();
            app.graph?.setDirtyCanvas?.(true, true);
            closePicker();
        });
        listEl.appendChild(item);
    };
    const addGroupTitle = (group) => {
        const title = document.createElement("div");
        title.className = "sf-preset-picker-group";
        title.style.cssText = "display:flex;align-items:center;justify-content:space-between;";
        const titleText = document.createElement("span");
        titleText.textContent = group;
        const dice = document.createElement("button");
        dice.type = "button";
        dice.textContent = "🎲 随机";
        dice.style.cssText = [
            "background:none", "border:none", "color:#9a9a9a", "cursor:pointer",
            "font-size:11px", "padding:0 4px", "border-radius:3px",
        ].join(";");
        dice.addEventListener("mouseenter", () => { dice.style.color = "#fff"; });
        dice.addEventListener("mouseleave", () => { dice.style.color = "#9a9a9a"; });
        dice.addEventListener("click", () => {
            if (!widget) return;
            widget.value = "随机·" + group;
            widget.callback?.();
            app.graph?.setDirtyCanvas?.(true, true);
            closePicker();
        });
        title.appendChild(titleText);
        title.appendChild(dice);
        listEl.appendChild(title);
    };

    // 按 group 稳定分组（组顺序 = 首次出现顺序，组内保持数据顺序），同组标题只渲染一次
    const grouped = new Map();
    for (const [value, meta] of filtered) {
        const g = meta?.group ?? "";
        if (!grouped.has(g)) grouped.set(g, []);
        grouped.get(g).push([value, meta]);
    }
    for (const [group, values] of grouped) {
        if (group) addGroupTitle(group);
        for (const [value, meta] of values) addItem(value, meta);
    }
}

function renderGroupBar() {
    if (!pickerEl) return;
    const groupBar = pickerEl.querySelector(".sf-preset-picker-groups");
    if (!groupBar) return;
    const categoryData = presetData?.[pickerCategory] ?? {};
    const groups = [...new Set(Object.values(categoryData).map((m) => m?.group).filter(Boolean))];
    groupBar.replaceChildren();
    const addChip = (label, value) => {
        const chip = document.createElement("div");
        chip.className = "sf-preset-picker-group-chip" + (pickerGroup === value ? " active" : "");
        chip.textContent = label;
        chip.addEventListener("click", () => {
            pickerGroup = value;
            renderGroupBar();
            renderPickerList();
        });
        groupBar.appendChild(chip);
    };
    // 全随机：当前分类整体随机（与"随机" combo 值一致）
    const randomChip = document.createElement("div");
    randomChip.className = "sf-preset-picker-group-chip";
    randomChip.textContent = "🎲 全随机";
    randomChip.addEventListener("click", () => {
        const widget = widgetForCategory(pickerCategory);
        if (!widget) return;
        widget.value = "随机";
        widget.callback?.();
        app.graph?.setDirtyCanvas?.(true, true);
        closePicker();
    });
    groupBar.appendChild(randomChip);
    addChip("全部", null);
    for (const g of groups) addChip(g, g);
}

function openGroupPicker() {
    injectPickerStyle();
    if (pickerEl) closePicker();
    if (lastPickerCategory) {
        pickerCategory = lastPickerCategory;
        pickerGroup = lastPickerGroup;
    }

    pickerEl = document.createElement("div");
    pickerEl.className = "sf-preset-picker-overlay";

    const panel = document.createElement("div");
    panel.className = "sf-preset-picker";

    const head = document.createElement("div");
    head.className = "sf-preset-picker-head";
    const title = document.createElement("span");
    title.textContent = "预设分组选择";
    const closeBtn = document.createElement("button");
    closeBtn.className = "sf-preset-picker-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        closePicker();
    });
    head.appendChild(title);
    head.appendChild(closeBtn);
    panel.appendChild(head);

    const tabs = document.createElement("div");
    tabs.className = "sf-preset-picker-tabs";
    for (const [category, label] of CATEGORY_TABS) {
        const tab = document.createElement("div");
        tab.className = "sf-preset-picker-tab" + (category === pickerCategory ? " active" : "");
        tab.textContent = label;
        tab.addEventListener("click", () => {
            pickerCategory = category;
            pickerGroup = null;
            tabs.querySelectorAll(".sf-preset-picker-tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            const searchEl = pickerEl.querySelector(".sf-preset-picker-search");
            searchEl.value = "";
            pickerSearch = "";
            renderGroupBar();
            renderPickerList();
        });
        tabs.appendChild(tab);
    }
    panel.appendChild(tabs);

    const groupBar = document.createElement("div");
    groupBar.className = "sf-preset-picker-groups";
    panel.appendChild(groupBar);

    const search = document.createElement("input");
    search.className = "sf-preset-picker-search";
    search.type = "text";
    search.placeholder = "搜索...";
    search.addEventListener("input", () => {
        pickerSearch = search.value.trim().toLowerCase();
        renderPickerList();
    });
    panel.appendChild(search);

    const list = document.createElement("div");
    list.className = "sf-preset-picker-list";
    panel.appendChild(list);

    const preview = document.createElement("div");
    preview.className = "sf-preset-picker-preview";
    panel.appendChild(preview);

    pickerEl.appendChild(panel);
    document.body.appendChild(pickerEl);

    pickerEl.addEventListener("click", (e) => {
        if (e.target === pickerEl) closePicker();
    });

    renderGroupBar();
    renderPickerList();
    search.focus();
}

function onKeyDownCapture(e) {
    if (pickerEl && e.key === "Escape") closePicker();
}

app.registerExtension({
    name: "sfnodes.prompt_preset",

    async setup() {
        try {
            const resp = await api.fetchApi("/api/sfnodes/prompt_presets");
            if (resp.ok) {
                presetData = await resp.json();
                // 工作流恢复后节点已存在的情况：加载完成后统一同步一次
                for (const node of app.graph?._nodes ?? []) {
                    if (node?.type !== "SFPromptPreset" || !node.widgets) continue;
                    for (const w of node.widgets) {
                        if (WIDGET_CATEGORY[w.name]) {
                            syncWidgetTooltip(w);
                        }
                    }
                }
            }
        } catch (err) {
            console.log("[SFPromptPreset] 预设数据加载失败，说明与分组选择器不可用:", err);
        }

        document.addEventListener("keydown", onKeyDownCapture, true);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SFPromptPreset") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const widgets = this.widgets ?? [];

            // 单人/双人动作互斥：选择其一自动将另一个置为"禁用"
            const poseWidget = widgets.find((w) => w.name === "pose_preset");
            const coupleWidget = widgets.find((w) => w.name === "couple_preset");
            const linkMutualExclusion = (w1, w2) => {
                const originalCallback = w1.callback;
                w1.callback = function (...args) {
                    if (this.value !== DISABLED) {
                        w2.value = DISABLED;
                    }
                    if (typeof originalCallback === "function") {
                        return originalCallback.apply(this, args);
                    }
                };
            };
            if (poseWidget && coupleWidget) {
                linkMutualExclusion(poseWidget, coupleWidget);
                linkMutualExclusion(coupleWidget, poseWidget);
            }

            // 预设 widget：选中值变化时把 description 同步到 tooltip
            for (const w of widgets) {
                if (!WIDGET_CATEGORY[w.name]) continue;
                const originalCallback = w.callback;
                w.callback = function (...args) {
                    syncWidgetTooltip(this);
                    if (typeof originalCallback === "function") {
                        return originalCallback.apply(this, args);
                    }
                };
                if (presetData) {
                    syncWidgetTooltip(w);
                }
            }

            // 分组选择器按钮（LiteGraph 原生 button widget，canvas 模式绘制可点击）
            if (!this.widgets.some((w) => w.type === "button")) {
                this.addWidget("button", "\u2630 预设", null, () => {
                    openGroupPicker();
                });
            }
        };
    },
});
