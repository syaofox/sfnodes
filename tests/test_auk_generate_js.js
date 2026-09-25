// SFAuKGenerateEdit 提示词预设前端冒烟测试（Node 直接运行：node tests/test_auk_generate_js.js）
// 用 mock app/node 真实加载 web/sf_auk_generate.js，验证：
//   - 扩展注册（sfnodes.auk_generate）
//   - nodeCreated：两个下拉 + 按钮挂载、置顶到 instruction 之前、全部 serialize:false
//   - 分类切换：模板选项重建（instruction 不动）
//   - 选模板 / 点按钮：模板文本整段写入 instruction
//   - 选择存 node.properties；configure 恢复（且不占 widgets_values 位——旧工作流零迁移）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock app ──
globalThis.app = {
    registerExtension(ext) { this._ext = ext; },
};
globalThis.window = { app: globalThis.app };

// ── FakeNode（模拟真实 configure 的位置恢复：serialize:false 不占位）──
function makeNode() {
    return {
        comfyClass: "SFAuKGenerateEdit",
        properties: {},
        widgets: [
            { name: "instruction", value: "", options: {}, serialize: undefined },
            { name: "generation_seconds", value: 0, options: {}, serialize: undefined },
            { name: "use_prompt_enhancer", value: true, options: {}, serialize: undefined },
            { name: "seed", value: 42, options: {}, serialize: undefined },
            { name: "control_after_generate", value: "fixed", options: {}, serialize: undefined },
            { name: "nfe_steps", value: 32, options: {}, serialize: undefined },
            { name: "cfg_strength", value: 2, options: {}, serialize: undefined },
            { name: "sway_sampling_coef", value: -1, options: {}, serialize: undefined },
        ],
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options: options || {} };
            if (widget.options.serialize !== undefined) widget.serialize = widget.options.serialize;
            this.widgets.push(widget);
            return widget;
        },
        setDirtyCanvas() {},
        configure(info) {
            Object.assign(this.properties, info.properties || {});
            let index = 0;
            for (const widget of this.widgets) {
                if (widget.serialize === false) continue;
                if (info.widgets_values && index < info.widgets_values.length) widget.value = info.widgets_values[index];
                index++;
            }
        },
    };
}

function copyModules(dir) {
    for (const name of ["sf_auk_presets_lib.js", "sf_auk_generate.js"]) {
        const code = fs
            .readFileSync(path.join(__dirname, "..", "web", name), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
            .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
        fs.writeFileSync(path.join(dir, name.replace(/\.js$/, ".mjs")), code);
    }
}

(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_auk_gen_"));
    copyModules(tmpDir);
    await import(path.join(tmpDir, "sf_auk_generate.mjs"));

    const ext = app._ext;
    check("扩展已注册", !!ext && ext.name === "sfnodes.auk_generate");

    // ── nodeCreated：挂载与置顶 ──
    const node = makeNode();
    ext.nodeCreated(node);
    const names = node.widgets.map((w) => w.name);
    check("三个控件已挂载", names.includes("预设分类") && names.includes("提示词模板") && names.includes("填入 instruction"));
    check("控件置顶到 instruction 之前", names.slice(0, 4).join("|") === "预设分类|提示词模板|填入 instruction|instruction");
    check("控件 serialize:false", node.widgets.slice(0, 3).every((w) => w.serialize === false));
    check("原控件数量不变", names.filter((n) => !["预设分类", "提示词模板", "填入 instruction"].includes(n)).length === 8);

    const groupW = node.widgets.find((w) => w.name === "预设分类");
    const templateW = node.widgets.find((w) => w.name === "提示词模板");
    const applyW = node.widgets.find((w) => w.name === "填入 instruction");
    const instructionW = node.widgets.find((w) => w.name === "instruction");

    check("分类默认第一组", groupW.value === "1. 参考音色 TTS");
    check("模板默认第一项", templateW.value === "EN & CN");
    check("初始不写入 instruction", instructionW.value === "");

    // ── 分类切换：模板选项重建、instruction 不动 ──
    groupW.value = "13. 语音增强（降噪、去混响、修复）";
    groupW.callback(groupW.value);
    check("分类切换重建模板选项", templateW.options.values.includes("Denoise · CN") && templateW.options.values.length === 9);
    check("分类切换模板值回退首项", templateW.value === "Denoise · EN");
    check("分类切换不写 instruction", instructionW.value === "");

    // ── 选模板：整段写入 ──
    templateW.value = "Denoise · CN";
    templateW.callback(templateW.value);
    check("选模板写入 instruction", instructionW.value === "请只去除背景噪声，保留其他内容，输出等长结果。");
    check("选择已存 properties", node.properties.sfAukPresetGroup === "13. 语音增强（降噪、去混响、修复）"
          && node.properties.sfAukPresetTemplate === "Denoise · CN");

    // ── 按钮：重复套用（下拉值未变时 callback 不触发）──
    instructionW.value = "被手改过的内容";
    applyW.callback();
    check("按钮重复填入当前模板", instructionW.value === "请只去除背景噪声，保留其他内容，输出等长结果。");

    // ── 加载恢复：properties 生效 + 位置值归原控件（零迁移）──
    const restored = makeNode();
    restored.properties = { sfAukPresetGroup: "3. 语音内容编辑（替换、增添、删除）", sfAukPresetTemplate: "Replace · CN" };
    ext.nodeCreated(restored);
    restored.configure({
        properties: restored.properties,
        widgets_values: ["旧指令", 3.5, false, 7, "fixed", 4, 0, -1],
    });
    const rGroup = restored.widgets.find((w) => w.name === "预设分类");
    const rTemplate = restored.widgets.find((w) => w.name === "提示词模板");
    const rInstruction = restored.widgets.find((w) => w.name === "instruction");
    const rSeed = restored.widgets.find((w) => w.name === "seed");
    check("加载恢复分类/模板", rGroup.value === "3. 语音内容编辑（替换、增添、删除）" && rTemplate.value === "Replace · CN");
    check("位置值仍归原控件（零迁移）", rInstruction.value === "旧指令" && rSeed.value === 7);
    check("加载不覆盖 instruction", rInstruction.value === "旧指令");

    // ── properties 缺失/失效回退默认 ──
    const fresh = makeNode();
    ext.nodeCreated(fresh);
    fresh.properties = { sfAukPresetGroup: "不存在的分类", sfAukPresetTemplate: "不存在" };
    fresh.configure({ properties: fresh.properties, widgets_values: [] });
    check("失效选择回退默认", fresh.widgets.find((w) => w.name === "预设分类").value === "1. 参考音色 TTS"
          && fresh.widgets.find((w) => w.name === "提示词模板").value === "EN & CN");

    if (failures.length) {
        console.log(`\n${failures.length} 项失败：`);
        for (const name of failures) console.log("  -", name);
        process.exit(1);
    }
    console.log("\nOK");
})();
