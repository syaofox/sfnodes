// SF Text Dropdown 前端扩展
// 功能：
// - 从 options_json 中读取全局文本列表
// - 使用下拉框展示选项，并输出选中的文本到 selected_text
// - 提供文本输入框 + 添加按钮，将新文本加入列表
// - 提供删除按钮，从列表中删除当前选中项
// - 将完整列表和当前选中文本写回 options_json / selected_text，由后端 Python 节点持久化到 JSON 文件

import { app } from "/scripts/app.js";

app.registerExtension({
    name: "sfnodes.text_dropdown",

    /**
     * 在节点实例创建后增强 SFTextDropdown 的 UI
     */
    nodeCreated(node) {
        // 仅处理我们的节点
        if (node?.comfyClass !== "SFTextDropdown") {
            return;
        }

        if (!node.widgets) {
            node.widgets = [];
        }

        // 找到 Python 定义的两个基础 widget
        let selectedWidget = node.widgets.find((w) => w.name === "selected_text");
        let optionsWidget = node.widgets.find((w) => w.name === "options_json");

        // 确保存在承载 JSON 的隐藏 widget
        if (!optionsWidget) {
            optionsWidget = {
                name: "options_json",
                type: "text",
                value: "[]",
            };
            node.widgets.push(optionsWidget);
        }

        // 隐藏 options_json，不在 UI 中占空间，但仍参与序列化
        optionsWidget.hidden = true;
        optionsWidget.computeSize = () => [0, 0];
        optionsWidget.draw = () => {};

        function parseOptions() {
            try {
                const v = optionsWidget.value || "[]";
                const data = JSON.parse(v);
                if (Array.isArray(data)) {
                    return data.map((x) => String(x));
                }
            } catch (e) {
                // ignore
            }
            return [];
        }

        let options = parseOptions();

        // 原始 selected_text 作为隐藏存储字段，实际显示用一个新的 combo widget
        if (!selectedWidget) {
            selectedWidget = {
                name: "selected_text",
                type: "text",
                value: options[0] || "",
            };
            node.widgets.unshift(selectedWidget);
        }
        // 隐藏原始 widget（仍用于序列化）
        selectedWidget.hidden = true;
        selectedWidget.computeSize = () => [0, 0];
        selectedWidget.draw = () => {};

        let comboWidget = null;

        function ensureComboWidget() {
            if (comboWidget) {
                return;
            }
            comboWidget = node.addWidget(
                "combo",
                "selected_text",
                selectedWidget.value || options[0] || "",
                function (value) {
                    // 同步到隐藏的 selected_text，供后端使用
                    selectedWidget.value = value;
                    return value;
                },
                {
                    values: options,
                }
            );
        }

        function syncOptionsWidget() {
            optionsWidget.value = JSON.stringify(options);
            node.setDirtyCanvas(true, true);
        }

        function refreshSelectedWidget() {
            ensureComboWidget();
            if (!comboWidget) return;

            // 更新可选值列表
            if (!comboWidget.options) {
                comboWidget.options = {};
            }
            comboWidget.options.values = options;

            // 当前值修正
            let current = comboWidget.value;
            if (!current && options.length > 0) {
                current = options[0];
            }
            if (options.length === 0) {
                current = "";
            } else if (!options.includes(current)) {
                current = options[0];
            }
            comboWidget.value = current;
            selectedWidget.value = current;

            node.setDirtyCanvas(true, true);
        }

        // 确保下拉框已创建
        ensureComboWidget();

        // new_item 由 Python INPUT_TYPES 定义（multiline=True），前端会渲染为多行框
        let inputWidget = node.widgets.find((w) => w.name === "new_item");
        if (!inputWidget) {
            inputWidget = node.addWidget("text", "new_item", "", (v) => v, {
                multiline: true,
            });
        }

        // 添加按钮：将多行输入整块作为一条项目保存
        const addButton = node.addWidget(
            "button",
            "添加",
            null,
            function () {
                const text = (inputWidget.value || "").trim();
                if (!text) {
                    return;
                }
                if (!options.includes(text)) {
                    options.push(text);
                    syncOptionsWidget();
                    refreshSelectedWidget();
                }
                inputWidget.value = "";
                node.setDirtyCanvas(true, true);
            }
        );
        addButton.serialize = false;

        // 删除按钮
        const deleteButton = node.addWidget(
            "button",
            "删除选中",
            null,
            function () {
                const current = selectedWidget.value;
                if (!current) {
                    return;
                }
                const idx = options.indexOf(current);
                if (idx === -1) {
                    return;
                }
                options.splice(idx, 1);
                syncOptionsWidget();
                refreshSelectedWidget();
            }
        );
        deleteButton.serialize = false;

        // 初始同步一次，确保 options_json 与 UI 一致
        syncOptionsWidget();
        refreshSelectedWidget();
    },
});

