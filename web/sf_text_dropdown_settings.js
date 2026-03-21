// SF Text Dropdown Settings - 设置页面管理分类和选项
// - 分类管理：添加/编辑/删除分类
// - 选项管理：添加/编辑/删除选项（按分类）

import { app } from "/scripts/app.js";

let config = {
    categories: ["default"],
    options: []
};

let settingsEl = null;
let selectedCategory = "default";

function normalizeItem(x) {
    if (x && typeof x === "object" && "alias" in x && "content" in x) {
        return {
            category: String(x.category || "default").trim() || "default",
            alias: String(x.alias).trim(),
            content: String(x.content)
        };
    }
    return null;
}

async function loadConfig() {
    try {
        const resp = await fetch("/api/sfnodes/text_dropdown/load");
        if (resp.ok) {
            config = await resp.json();
        }
    } catch (e) {
        console.error("Failed to load config:", e);
    }
}

async function saveConfig() {
    try {
        await fetch("/api/sfnodes/text_dropdown/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ config })
        });
    } catch (e) {
        console.error("[SFTextDropdown Settings] Failed to save config:", e);
    }
}

function getCategoryOptions() {
    return config.categories.map(c => ({
        value: c,
        text: c,
        selected: c === selectedCategory
    }));
}

function getFilteredOptions() {
    return config.options.filter(o => o.category === selectedCategory);
}

function renderCategoryList() {
    const container = document.createElement("div");
    container.className = "sf-text-dropdown-categories";

    const title = document.createElement("div");
    title.className = "sf-text-dropdown-section-title";
    title.textContent = "分类列表";
    container.appendChild(title);

    const list = document.createElement("div");
    list.className = "sf-text-dropdown-category-list";

    config.categories.forEach(cat => {
        const item = document.createElement("div");
        item.className = "sf-text-dropdown-category-item";

        const name = document.createElement("span");
        name.textContent = cat;
        if (cat === selectedCategory) {
            name.className = "selected";
        }
        item.appendChild(name);

        const actions = document.createElement("div");
        actions.className = "sf-text-dropdown-actions";

        const editBtn = document.createElement("button");
        editBtn.textContent = "编辑";
        editBtn.className = "sf-text-dropdown-btn";
        editBtn.onclick = () => editCategory(cat);
        actions.appendChild(editBtn);

        if (cat !== "default") {
            const delBtn = document.createElement("button");
            delBtn.textContent = "删除";
            delBtn.className = "sf-text-dropdown-btn sf-text-dropdown-btn-danger";
            delBtn.onclick = () => deleteCategory(cat);
            actions.appendChild(delBtn);
        }

        item.appendChild(actions);
        item.onclick = () => selectCategory(cat);
        list.appendChild(item);
    });

    container.appendChild(list);

    const addRow = document.createElement("div");
    addRow.className = "sf-text-dropdown-add-row";

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "新分类名称";
    input.className = "sf-text-dropdown-input";
    input.id = "sf-text-dropdown-new-category";

    const addBtn = document.createElement("button");
    addBtn.textContent = "添加分类";
    addBtn.className = "sf-text-dropdown-btn sf-text-dropdown-btn-primary";
    addBtn.onclick = addCategory;

    addRow.appendChild(input);
    addRow.appendChild(addBtn);
    container.appendChild(addRow);

    return container;
}

function renderOptionsList() {
    const container = document.createElement("div");
    container.className = "sf-text-dropdown-options";

    const title = document.createElement("div");
    title.className = "sf-text-dropdown-section-title";
    title.textContent = `选项列表 - ${selectedCategory}`;
    container.appendChild(title);

    const list = document.createElement("div");
    list.className = "sf-text-dropdown-option-list";

    const filtered = getFilteredOptions();
    if (filtered.length === 0) {
        const empty = document.createElement("div");
        empty.className = "sf-text-dropdown-empty";
        empty.textContent = "该分类暂无选项";
        list.appendChild(empty);
    } else {
        filtered.forEach((opt, idx) => {
            const item = document.createElement("div");
            item.className = "sf-text-dropdown-option-item";

            const name = document.createElement("span");
            name.textContent = opt.alias;
            item.appendChild(name);

            const actions = document.createElement("div");
            actions.className = "sf-text-dropdown-actions";

            const editBtn = document.createElement("button");
            editBtn.textContent = "编辑";
            editBtn.className = "sf-text-dropdown-btn";
            editBtn.onclick = () => editOption(opt);
            actions.appendChild(editBtn);

            const delBtn = document.createElement("button");
            delBtn.textContent = "删除";
            delBtn.className = "sf-text-dropdown-btn sf-text-dropdown-btn-danger";
            delBtn.onclick = () => deleteOption(opt);
            actions.appendChild(delBtn);

            item.appendChild(actions);
            list.appendChild(item);
        });
    }

    container.appendChild(list);

    const form = document.createElement("div");
    form.className = "sf-text-dropdown-option-form";

    const aliasLabel = document.createElement("label");
    aliasLabel.textContent = "别名:";
    form.appendChild(aliasLabel);

    const aliasInput = document.createElement("input");
    aliasInput.type = "text";
    aliasInput.placeholder = "选项别名";
    aliasInput.className = "sf-text-dropdown-input";
    aliasInput.id = "sf-text-dropdown-option-alias";
    form.appendChild(aliasInput);

    const contentLabel = document.createElement("label");
    contentLabel.textContent = "内容:";
    form.appendChild(contentLabel);

    const contentInput = document.createElement("textarea");
    contentInput.placeholder = "选项内容（支持多行）";
    contentInput.className = "sf-text-dropdown-textarea";
    contentInput.id = "sf-text-dropdown-option-content";
    form.appendChild(contentInput);

    const btnRow = document.createElement("div");
    btnRow.className = "sf-text-dropdown-btn-row";

    const addBtn = document.createElement("button");
    addBtn.textContent = "添加选项";
    addBtn.className = "sf-text-dropdown-btn sf-text-dropdown-btn-primary";
    addBtn.onclick = addOption;

    const clearBtn = document.createElement("button");
    clearBtn.textContent = "清空";
    clearBtn.className = "sf-text-dropdown-btn";
    clearBtn.onclick = clearForm;

    btnRow.appendChild(addBtn);
    btnRow.appendChild(clearBtn);
    form.appendChild(btnRow);

    container.appendChild(form);

    return container;
}

function selectCategory(cat) {
    selectedCategory = cat;
    updateUI();
}

function editCategory(oldCat) {
    const newCatInput = prompt("编辑分类名称:", oldCat);
    if (!newCatInput || newCatInput.trim() === oldCat) return;
    const newCat = newCatInput.trim();
    if (config.categories.includes(newCat)) {
        alert("分类名称已存在");
        return;
    }
    const idx = config.categories.indexOf(oldCat);
    config.categories[idx] = newCat;
    config.options.forEach(opt => {
        if (opt.category === oldCat) {
            opt.category = newCat;
        }
    });
    if (selectedCategory === oldCat) {
        selectedCategory = newCat;
    }
    saveConfig();
    updateUI();
}

function deleteCategory(cat) {
    if (!confirm(`确定删除分类 "${cat}" 及其所有选项？`)) return;
    config.categories = config.categories.filter(c => c !== cat);
    config.options = config.options.filter(o => o.category !== cat);
    if (selectedCategory === cat) {
        selectedCategory = config.categories[0] || "default";
    }
    saveConfig();
    updateUI();
}

function addCategory() {
    const input = document.getElementById("sf-text-dropdown-new-category");
    if (!input) return;
    const name = input.value.trim();
    if (!name) return;
    if (config.categories.includes(name)) {
        alert("分类名称已存在");
        return;
    }
    config.categories.push(name);
    config.categories.sort();
    saveConfig();
    input.value = "";
    updateUI();
}

function editOption(opt) {
    const alias = prompt("编辑别名:", opt.alias);
    if (alias === null) return;
    const content = prompt("编辑内容:", opt.content);
    if (content === null) return;

    const normalizedAlias = alias.trim();
    const existing = config.options.find(o =>
        o.category === opt.category && o.alias === normalizedAlias && o !== opt
    );
    if (existing) {
        alert("别名已存在");
        return;
    }

    opt.alias = normalizedAlias;
    opt.content = content;
    saveConfig();
    updateUI();
}

function deleteOption(opt) {
    if (!confirm(`确定删除选项 "${opt.alias}"？`)) return;
    config.options = config.options.filter(o => o !== opt);
    saveConfig();
    updateUI();
}

function addOption() {
    const aliasInput = document.getElementById("sf-text-dropdown-option-alias");
    const contentInput = document.getElementById("sf-text-dropdown-option-content");
    if (!aliasInput || !contentInput) return;

    const alias = aliasInput.value.trim();
    const content = contentInput.value;

    if (!alias) {
        alert("请输入别名");
        return;
    }

    const existing = config.options.find(o =>
        o.category === selectedCategory && o.alias === alias
    );
    if (existing) {
        alert("该分类下别名已存在");
        return;
    }

    config.options.push({
        category: selectedCategory,
        alias,
        content
    });
    saveConfig();
    clearForm();
    updateUI();
}

function clearForm() {
    const aliasInput = document.getElementById("sf-text-dropdown-option-alias");
    const contentInput = document.getElementById("sf-text-dropdown-option-content");
    if (aliasInput) aliasInput.value = "";
    if (contentInput) contentInput.value = "";
}

function updateUI() {
    if (!settingsEl) return;

    settingsEl.innerHTML = "";

    const container = document.createElement("div");
    container.className = "sf-text-dropdown-settings";

    const catList = renderCategoryList();
    const optList = renderOptionsList();

    container.appendChild(catList);
    container.appendChild(optList);
    settingsEl.appendChild(container);
}

const style = document.createElement("style");
style.textContent = `
.sf-text-dropdown-settings {
    padding: 10px;
    max-height: 500px;
    overflow-y: auto;
}
.sf-text-dropdown-section-title {
    font-weight: bold;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #333;
}
.sf-text-dropdown-category-list,
.sf-text-dropdown-option-list {
    margin: 8px 0;
    max-height: 150px;
    overflow-y: auto;
    border: 1px solid #444;
    border-radius: 4px;
}
.sf-text-dropdown-category-item,
.sf-text-dropdown-option-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    cursor: pointer;
    border-bottom: 1px solid #333;
}
.sf-text-dropdown-category-item:last-child,
.sf-text-dropdown-option-item:last-child {
    border-bottom: none;
}
.sf-text-dropdown-category-item:hover,
.sf-text-dropdown-option-item:hover {
    background: rgba(255,255,255,0.05);
}
.sf-text-dropdown-category-item span.selected {
    color: #4af;
    font-weight: bold;
}
.sf-text-dropdown-actions {
    display: flex;
    gap: 6px;
}
.sf-text-dropdown-btn {
    padding: 3px 10px;
    border: 1px solid #555;
    border-radius: 3px;
    background: #333;
    color: #ccc;
    cursor: pointer;
    font-size: 12px;
}
.sf-text-dropdown-btn:hover {
    background: #444;
}
.sf-text-dropdown-btn-primary {
    background: #2a4;
    border-color: #2a4;
    color: #fff;
}
.sf-text-dropdown-btn-primary:hover {
    background: #3b5;
}
.sf-text-dropdown-btn-danger {
    color: #f66;
}
.sf-text-dropdown-btn-danger:hover {
    background: rgba(255,68,68,0.2);
}
.sf-text-dropdown-add-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
}
.sf-text-dropdown-input {
    flex: 1;
    padding: 5px 8px;
    border: 1px solid #444;
    border-radius: 3px;
    background: #222;
    color: #ccc;
}
.sf-text-dropdown-textarea {
    width: 100%;
    min-height: 60px;
    padding: 5px 8px;
    border: 1px solid #444;
    border-radius: 3px;
    background: #222;
    color: #ccc;
    resize: vertical;
    margin: 4px 0;
}
.sf-text-dropdown-option-form {
    margin-top: 10px;
    padding: 10px;
    border: 1px solid #444;
    border-radius: 4px;
}
.sf-text-dropdown-btn-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
}
.sf-text-dropdown-empty {
    padding: 20px;
    text-align: center;
    color: #888;
}
`;
document.head.appendChild(style);

app.registerExtension({
    name: "sfnodes.text_dropdown.settings",

    async setup() {
        await loadConfig();

        app.ui.settings.addSetting({
            id: "sfnodes.text_dropdown.config",
            name: "📁 SF Text Dropdown Settings",
            type: (name, setter, value) => {
                const tr = document.createElement("tr");
                tr.style.width = "100%";

                const td = document.createElement("td");
                td.style.width = "100%";
                td.style.padding = "10px";

                settingsEl = td;

                const title = document.createElement("div");
                title.style.fontWeight = "bold";
                title.style.marginBottom = "10px";
                title.textContent = name;
                td.appendChild(title);

                updateUI();

                tr.appendChild(td);
                return tr;
            },
            defaultValue: null,
        });
    },
});
