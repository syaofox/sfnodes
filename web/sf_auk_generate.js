// SFAuKGenerateEdit 官方提示词预设（前端）
//
// 节点顶部（instruction 之前）挂两个下拉 + 一个按钮：
//   预设分类 → 提示词模板（随分类重建选项）→ 选中模板即整段写入 instruction（保留 {占位符}）
//   「填入 instruction」按钮用于重复套用当前模板（下拉值未变时 callback 不触发）
//
// 三个控件 serialize:false：不进 widgets_values，旧工作流位置零迁移；
// 选择状态存 node.properties（随工作流保存），加载时恢复。
// 模板数据/纯函数在 sf_auk_presets_lib.js（无 app 依赖，可 .mjs 直测）。

import { app } from "/scripts/app.js";

import {
    DEFAULT_GROUP,
    DEFAULT_TEMPLATE,
    groupNames,
    itemsOf,
    templateText,
} from "./sf_auk_presets_lib.js";

const NODE_CLASS = "SFAuKGenerateEdit";
const PROP_GROUP = "sfAukPresetGroup";
const PROP_TEMPLATE = "sfAukPresetTemplate";

function setComboValues(widget, values) {
    if (!widget.options) widget.options = {};
    widget.options.values = values;
}

app.registerExtension({
    name: "sfnodes.auk_generate",

    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        const instruction = node.widgets?.find((widget) => widget.name === "instruction");
        if (!instruction) return;

        const applyTemplate = () => {
            const text = templateText(groupWidget.value, templateWidget.value);
            if (!text) return;
            instruction.value = text;
            node.setDirtyCanvas?.(true, true);
        };

        const syncTemplates = (keepValue) => {
            const labels = itemsOf(groupWidget.value).map((item) => item.label);
            setComboValues(templateWidget, labels);
            if (!keepValue || !labels.includes(templateWidget.value)) {
                templateWidget.value = labels[0] ?? "";
            }
        };

        const remember = () => {
            node.properties[PROP_GROUP] = groupWidget.value;
            node.properties[PROP_TEMPLATE] = templateWidget.value;
        };

        const groupWidget = node.addWidget("combo", "预设分类", DEFAULT_GROUP, () => {
            syncTemplates(false);
            remember();
            node.setDirtyCanvas?.(true, true);
        }, { values: groupNames(), serialize: false });

        const templateWidget = node.addWidget("combo", "提示词模板", DEFAULT_TEMPLATE, () => {
            applyTemplate();
            remember();
        }, { values: itemsOf(DEFAULT_GROUP).map((item) => item.label), serialize: false });

        const applyWidget = node.addWidget("button", "填入 instruction", null, () => {
            applyTemplate();
        }, { serialize: false });

        // 置顶到 instruction 之前（serialize:false 不占 widgets_values 位，旧工作流零迁移）
        const added = [groupWidget, templateWidget, applyWidget];
        const rest = node.widgets.filter((widget) => !added.includes(widget));
        node.widgets.length = 0;
        node.widgets.push(...added, ...rest);

        const restore = () => {
            const names = groupNames();
            const savedGroup = node.properties?.[PROP_GROUP];
            groupWidget.value = names.includes(savedGroup) ? savedGroup : DEFAULT_GROUP;
            syncTemplates(false);
            const labels = itemsOf(groupWidget.value).map((item) => item.label);
            const savedTemplate = node.properties?.[PROP_TEMPLATE];
            templateWidget.value = labels.includes(savedTemplate) ? savedTemplate : (labels[0] ?? "");
        };

        const origConfigure = node.configure;
        node.configure = function () {
            const result = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            restore();
            return result;
        };
        const origAfterConfigure = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function (...args) {
            if (origAfterConfigure) origAfterConfigure.apply(this, args);
            restore();
        };
    },
});
