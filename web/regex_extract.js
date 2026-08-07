// SF Text Regex Extract 前端扩展
// 功能：preset 选中内置预设时自动把正则填入 pattern（并同步 group），
// 手动编辑 pattern 时按值反查预设名，匹配则 combo 切回对应预设，否则置「自定义」。

import { app } from "/scripts/app.js";

const NODE_TYPE = "SFTextRegexExtract";
const PRESET_CUSTOM = "自定义";

// 与后端 regex_extract.py 的 _REGEX_PRESETS 保持一致
const PRESETS = [
    { name: "提取数字", regex: "-?\\d+(?:\\.\\d+)?", group: 0 },
    { name: "提取整数", regex: "\\d+", group: 0 },
    { name: "提取中文", regex: "[\\u4e00-\\u9fff]+", group: 0 },
    { name: "提取英文单词", regex: "[A-Za-z]+", group: 0 },
    { name: "提取邮箱", regex: "[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+", group: 0 },
    { name: "提取网址", regex: "https?://[^\\s\"'<>]+", group: 0 },
    { name: "提取手机号", regex: "1[3-9]\\d{9}", group: 0 },
    { name: "提取日期", regex: "\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}", group: 0 },
    { name: "提取时间", regex: "\\d{1,2}:\\d{2}(?::\\d{2})?", group: 0 },
    { name: "提取圆括号内容", regex: "\\(([^)]*)\\)", group: 1 },
    { name: "提取方括号内容", regex: "\\[([^\\]]*)\\]", group: 1 },
    { name: "提取文件扩展名", regex: "\\.([A-Za-z0-9]+)", group: 1 },
];
const PRESET_BY_REGEX = new Map(PRESETS.map((p) => [p.regex, p]));

app.registerExtension({
    name: "sfnodes.TextRegexExtract",

    nodeCreated(node) {
        if (node?.comfyClass !== NODE_TYPE) return;

        const presetWidget = node.widgets?.find((w) => w.name === "preset");
        const patternWidget = node.widgets?.find((w) => w.name === "pattern");
        const groupWidget = node.widgets?.find((w) => w.name === "group");
        if (!presetWidget || !patternWidget) return;

        const applyPreset = (name) => {
            const p = PRESETS.find((x) => x.name === name);
            if (!p) return;
            patternWidget.value = p.regex;
            if (groupWidget) groupWidget.value = p.group;
            node.setDirtyCanvas(true, true);
        };

        // 按 pattern 值反查预设名，同步 combo 显示；pattern 为空时按 combo 预设自动填入
        const syncPattern = () => {
            const value = (patternWidget.value || "").trim();
            if (value) {
                const p = PRESET_BY_REGEX.get(value);
                const target = p ? p.name : PRESET_CUSTOM;
                if (presetWidget.value !== target) {
                    presetWidget.value = target;
                    node.setDirtyCanvas(true, true);
                }
            } else {
                const p = PRESETS.find((x) => x.name === presetWidget.value);
                if (p) applyPreset(p.name);
            }
        };

        const origPresetCb = presetWidget.callback;
        presetWidget.callback = function (...args) {
            if (this.value !== PRESET_CUSTOM) applyPreset(this.value);
            if (typeof origPresetCb === "function") return origPresetCb.apply(this, args);
        };

        const origPatternCb = patternWidget.callback;
        patternWidget.callback = function (...args) {
            syncPattern();
            if (typeof origPatternCb === "function") return origPatternCb.apply(this, args);
        };

        // 工作流恢复 widget 值不触发 callback，按保存的 pattern 值同步一次 combo 显示
        const origOnAfter = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function (...args) {
            syncPattern();
            if (typeof origOnAfter === "function") return origOnAfter.apply(this, args);
        };

        // 诊断接口（真实环境排障用）
        node._sfRegexDiagnose = () => ({
            preset: presetWidget.value,
            pattern: patternWidget.value,
            group: groupWidget?.value,
            presetRegex: PRESET_BY_REGEX.get(String(presetWidget.value))?.regex ?? null,
            patternMatchesPreset: PRESET_BY_REGEX.has((patternWidget.value || "").trim()),
        });

        syncPattern();
    },
});
