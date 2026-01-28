// SF Text Dropdown 前端扩展（别名 + 内容）
// - 下拉框显示别名，输出为对应项目文本内容（可多行）
// - 单行「别名」、多行「项目内容」+ 添加/删除

import { app } from "/scripts/app.js";

function normalizeOptions(data) {
    if (!Array.isArray(data)) return [];
    return data
        .map((x) => {
            if (x && typeof x === "object" && "alias" in x && "content" in x) {
                return { alias: String(x.alias).trim(), content: String(x.content) };
            }
            if (typeof x === "string") {
                const s = x.trim();
                const first = (s.split(/\r?\n/)[0] || "").trim() || "未命名";
                return { alias: first.slice(0, 64), content: s };
            }
            return null;
        })
        .filter((o) => o && o.alias);
}

app.registerExtension({
    name: "sfnodes.text_dropdown",

    nodeCreated(node) {
        if (node?.comfyClass !== "SFTextDropdown") return;

        if (!node.widgets) node.widgets = [];

        let selectedWidget = node.widgets.find((w) => w.name === "selected_text");
        let optionsWidget = node.widgets.find((w) => w.name === "options_json");

        if (!optionsWidget) {
            optionsWidget = { name: "options_json", type: "text", value: "[]" };
            node.widgets.push(optionsWidget);
        }
        optionsWidget.hidden = true;
        optionsWidget.computeSize = () => [0, 0];
        optionsWidget.draw = () => {};

        function parseOptions() {
            try {
                const v = optionsWidget.value || "[]";
                const data = JSON.parse(v);
                return normalizeOptions(data);
            } catch (e) {
                return [];
            }
        }

        let options = parseOptions();

        if (!selectedWidget) {
            selectedWidget = {
                name: "selected_text",
                type: "text",
                value: options[0] ? options[0].content : "",
            };
            node.widgets.unshift(selectedWidget);
        }
        selectedWidget.hidden = true;
        selectedWidget.computeSize = () => [0, 0];
        selectedWidget.draw = () => {};

        let comboWidget = null;

        function ensureComboWidget() {
            if (comboWidget) return;
            const aliases = options.map((o) => o.alias);
            const sel = options.find((o) => o.content === selectedWidget.value);
            const currentAlias = sel ? sel.alias : aliases[0] || "";

            comboWidget = node.addWidget(
                "combo",
                "selected_alias",
                currentAlias,
                function (value) {
                    const it = options.find((o) => o.alias === value);
                    selectedWidget.value = it ? it.content : "";
                    if (inputWidget) inputWidget.value = it ? it.content : "";
                    if (aliasWidget) aliasWidget.value = "";
                    return value;
                },
                { values: aliases }
            );
            comboWidget.serialize = false;
        }

        function syncOptionsWidget() {
            optionsWidget.value = JSON.stringify(options);
            node.setDirtyCanvas(true, true);
        }

        function saveOptionsToServer() {
            fetch("/api/sfnodes/text_dropdown/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ options }),
            }).catch(() => {});
        }

        function refreshSelectedWidget() {
            ensureComboWidget();
            if (!comboWidget) return;

            const aliases = options.map((o) => o.alias);
            if (!comboWidget.options) comboWidget.options = {};
            comboWidget.options.values = aliases;

            let curAlias = comboWidget.value;
            if (!aliases.includes(curAlias)) curAlias = aliases[0] || "";
            comboWidget.value = curAlias;

            const it = options.find((o) => o.alias === curAlias);
            selectedWidget.value = it ? it.content : "";
            if (inputWidget) inputWidget.value = it ? it.content : "";
            if (aliasWidget) aliasWidget.value = "";

            node.setDirtyCanvas(true, true);
        }

        let aliasWidget = node.widgets.find((w) => w.name === "alias");
        if (!aliasWidget) {
            aliasWidget = node.addWidget("text", "alias", "", (v) => v, {
                multiline: false,
            });
        }

        let inputWidget = node.widgets.find((w) => w.name === "new_item");
        if (!inputWidget) {
            inputWidget = node.addWidget("text", "new_item", "", (v) => v, {
                multiline: true,
            });
        }

        ensureComboWidget();
        const comboIdx = node.widgets.indexOf(comboWidget);
        const firstVisible = 2;
        if (comboIdx > firstVisible && comboIdx > 0) {
            node.widgets.splice(comboIdx, 1);
            node.widgets.splice(firstVisible, 0, comboWidget);
        }

        const addButton = node.addWidget("button", "添加", null, function () {
            const alias = (aliasWidget.value || "").trim();
            const content = (inputWidget.value || "").trim();
            const curAlias = comboWidget.value;

            if (alias) {
                if (!content) return;
                if (options.some((o) => o.alias === alias)) return;
                options.push({ alias, content });
                aliasWidget.value = "";
                inputWidget.value = "";
            } else if (curAlias && content) {
                const it = options.find((o) => o.alias === curAlias);
                if (it) it.content = content;
            } else return;

            syncOptionsWidget();
            refreshSelectedWidget();
            saveOptionsToServer();
            node.setDirtyCanvas(true, true);
        });
        addButton.serialize = false;

        const deleteButton = node.addWidget("button", "删除选中", null, function () {
            const curAlias = comboWidget.value;
            if (!curAlias) return;
            const idx = options.findIndex((o) => o.alias === curAlias);
            if (idx === -1) return;
            options.splice(idx, 1);
            syncOptionsWidget();
            refreshSelectedWidget();
            saveOptionsToServer();
        });
        deleteButton.serialize = false;

        syncOptionsWidget();
        refreshSelectedWidget();
    },
});
