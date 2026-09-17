// SFCanvasSizePreset 前端联动冒烟测试（Node 直接运行：node tests/test_canvas_size_js.js）
// 用 mock DOM/app/api/fetch 真实加载 web/canvas_size.js，验证：
//   - 扩展注册（sfnodes.CanvasSizePreset）
//   - nodeCreated：model callback 包装、管理按钮挂载、初始静态选项
//   - fetch 数据就绪后按当前 model 重建 resolution 选项（自定义组 --Custom-- 置顶 + 官方档位头）
//   - 切换 model：当前值保持（在列表内）/ 回退首个非分组头（不在列表内）
//   - 自定义库合并（与官方重复项跳过、非法项过滤）
//   - onAfterGraphConfigured 恢复场景重建
//   - fetch 失败降级（不炸、保持静态选项）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM ──
function makeEl() {
    const style = { setProperty() {}, getPropertyValue() { return ""; } };
    return {
        style, dataset: {}, children: [],
        className: "", textContent: "", innerHTML: "", value: "",
        classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        replaceChildren() { this.children = []; },
        remove() {}, querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        addEventListener() {}, removeEventListener() {},
        getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100, bottom: 100 }; },
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {},
    getElementById() { return null; },
    documentElement: { style: { setProperty() {} } },
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
    LiteGraph: { vueNodesMode: false },
};

// ── app / api / fetch mock ──
globalThis.app = {
    graph: { _nodes: [], setDirtyCanvas() {} },
    canvas: { setDirty() {}, ds: { scale: 1 } },
    registerExtension(ext) { this._ext = ext; },
};
globalThis.window.app = globalThis.app;
globalThis.api = {};

let officialResult = null; // 测试可控：null = 失败降级
let customResult = null;
globalThis.fetch = async (url) => {
    const u = String(url || "");
    if (u.includes("canvas_size_custom")) {
        if (!customResult) throw new Error("fetch failed (mock custom)");
        return { ok: true, json: async () => customResult };
    }
    if (!officialResult) throw new Error("fetch failed (mock official)");
    return { ok: true, json: async () => officialResult };
};

// ── mock 预设数据（与 Python 常量形状一致的最小集，含分组头）──
const PAYLOAD = {
    models: [
        "-- Image --", "Z-Image (Turbo)", "Qwen-Image (2512)", "Flux.1 (dev/schnell)",
        "Krea 2 (Turbo/RAW)", "Flux.2 Klein 9B", "SDXL / SD 3.5",
        "-- Video --", "Wan2.2 T2V", "Wan2.2 I2V", "Wan2.2 TI2V-5B",
        "HunyuanVideo 1.5", "LTX-2.5",
        "-- Custom --", "Custom Resolution",
    ],
    custom_model: "Custom Resolution",
    values: {
        "Z-Image (Turbo)": [
            "--1MP--", "1024x1024 (1:1)", "1152x896 (9:7)", "1280x720 (16:9)", "1344x576 (21:9)",
            "--1.6MP--", "1280x1280 (1:1)", "1536x864 (16:9)",
        ],
        "Qwen-Image (2512)": [
            "--Official--", "1328x1328 (1:1)", "1664x928 (16:9)", "928x1664 (9:16)",
        ],
        "Wan2.2 T2V": [
            "--480p--", "832x480 (26:15)", "480x832 (15:26)",
            "--720p--", "1280x720 (16:9)", "720x1280 (9:16)",
        ],
        "HunyuanVideo 1.5": [
            "--720p--", "1280x720 (16:9)", "720x1280 (9:16)",
            "--480p--", "848x480 (53:30)", "480x848 (30:53)",
        ],
        "LTX-2.5": [
            "--0.9MP--", "1280x736 (40:23)", "736x1280 (23:40)",
            "--1K--", "1024x1024 (1:1)", "1376x768 (43:24)",
        ],
        "Flux.2 Klein 9B": ["--1K--", "1024x1024 (1:1)", "--2K--", "2048x1152 (16:9)"],
    },
};

// 自定义库：Wide 生效；Bad(1) 非法名被过滤（重复跳过去重由 lib 测试覆盖）
const CUSTOM_PAYLOAD = { presets: [
    { name: "Wide", w: 1600, h: 900 },
    { name: "Bad(1)", w: 10, h: 10 },
]};
// 伪模型选项 = --Custom-- 头 + 自定义项；真实模型只列官方档位（不混自定义）
const CUSTOM_VALUES = ["--Custom--", "1600x900 (Wide)"];

// ── FakeNode ──
function makeNode(modelValue, resValue) {
    return {
        comfyClass: "SFCanvasSizePreset",
        widgets: [
            { name: "model", value: modelValue, callback: null, options: { values: PAYLOAD.models } },
            {
                name: "resolution", value: resValue,
                options: { values: ["--1MP--", "1024x1024 (1:1)", "1280x720 (16:9)"] },
                updateOptions() {},
            },
        ],
        _domWidgets: [],
        addDOMWidget(name, type, element) {
            this._domWidgets.push({ name, type, element });
            return { name };
        },
        setDirtyCanvas() {},
        onAfterGraphConfigured: null,
    };
}

const head = (arr) => arr.find((v) => !(v.startsWith("--") && v.endsWith("--")));

function copyModules(dir) {
    for (const n of ["sf_common.js", "sf_popup.js", "sf_canvas_size_lib.js", "canvas_size.js"]) {
        const code = fs
            .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
            .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
            .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
        fs.writeFileSync(path.join(dir, n.replace(/\.js$/, ".mjs")), code);
    }
}

(async () => {
    // ── 先测 fetch 失败降级（独立 tmp 目录 = 独立模块实例，无 promise 缓存）──
    {
        const tmpFail = fs.mkdtempSync(path.join(os.tmpdir(), "sf_csz_fail_"));
        copyModules(tmpFail);
        officialResult = null;
        customResult = null;
        await import(path.join(tmpFail, "canvas_size.mjs"));
        const extFail = app._ext;
        const node3 = makeNode("Z-Image (Turbo)", "1024x1024 (1:1)");
        const staticValues = [...node3.widgets[1].options.values];
        extFail.nodeCreated(node3);
        node3.widgets[0].value = "Wan2.2 T2V";
        node3.widgets[0].callback("Wan2.2 T2V");
        await new Promise((r) => setTimeout(r, 10));
        check("fetch 失败不炸且值不变", node3.widgets[1].value === "1024x1024 (1:1)");
        check("fetch 失败保持静态选项", JSON.stringify(node3.widgets[1].options.values) === JSON.stringify(staticValues));
    }

    // ── 主流程（独立 tmp 目录，全新模块实例 + fetch 成功）──
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_csz_"));
    copyModules(tmpDir);
    officialResult = PAYLOAD;
    customResult = CUSTOM_PAYLOAD;
    await import(path.join(tmpDir, "canvas_size.mjs"));

    const ext = app._ext;
    check("扩展已注册", !!ext && ext.name === "sfnodes.CanvasSizePreset");

    // ── nodeCreated 基础 ──
    const node = makeNode("Z-Image (Turbo)", "1024x1024 (1:1)");
    const modelWidget = node.widgets[0];
    const resWidget = node.widgets[1];
    ext.nodeCreated(node);
    check("model callback 已包装", typeof modelWidget.callback === "function");
    check("onAfterGraphConfigured 已挂", typeof node.onAfterGraphConfigured === "function");
    check("管理按钮已挂", node._domWidgets.some((w) => w.name === "sfCsCustomManage"));
    await new Promise((r) => setTimeout(r, 10)); // 让 fetch promise 落地
    check("fetch 后重建为 Z-Image 表（不含自定义）",
          JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["Z-Image (Turbo)"]));
    check("当前值保持", resWidget.value === "1024x1024 (1:1)");

    // ── 切换 model：值在新列表 → 保持（缓存就绪，同步重建）──
    modelWidget.value = "Flux.2 Klein 9B";
    modelWidget.callback("Flux.2 Klein 9B");
    check("切到 Klein 重建表（同步）",
          JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["Flux.2 Klein 9B"]));
    check("值在列表中保持", resWidget.value === "1024x1024 (1:1)");

    // ── 切换 model：值不在新列表 → 回退首个非分组头 ──
    modelWidget.value = "Wan2.2 T2V";
    modelWidget.callback("Wan2.2 T2V");
    check("切到 Wan2.2 T2V 重建表",
          JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["Wan2.2 T2V"]));
    check("值回退到 832x480 (26:15)", resWidget.value === head(PAYLOAD.values["Wan2.2 T2V"]));

    // ── 切到 LTX-2.5（跨分组，且 1024x1024 在表中）──
    modelWidget.value = "LTX-2.5";
    modelWidget.callback("LTX-2.5");
    check("切到 LTX-2.5 重建表",
          JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["LTX-2.5"]));
    check("值回退到 1280x736 (40:23)", resWidget.value === head(PAYLOAD.values["LTX-2.5"]));

    // ── 切到 Custom Resolution：只列自定义库（官方档位不混入）──
    modelWidget.value = "Custom Resolution";
    modelWidget.callback("Custom Resolution");
    check("切到 Custom 只列自定义库",
          JSON.stringify(resWidget.options.values) === JSON.stringify(["--Custom--", "1600x900 (Wide)"]));
    check("值回退到自定义项", resWidget.value === "1600x900 (Wide)");

    // ── onAfterGraphConfigured 恢复场景（直接赋链接不触发 callback）──
    const node2 = makeNode("Z-Image (Turbo)", "1024x1024 (1:1)");
    ext.nodeCreated(node2);
    await new Promise((r) => setTimeout(r, 10));
    node2.widgets[0].value = "Wan2.2 T2V"; // 模拟 widget 值恢复（不触发 callback）
    node2.widgets[1].value = "1280x720 (16:9)";
    node2.onAfterGraphConfigured();
    await new Promise((r) => setTimeout(r, 10)); // 等 Promise.all(官方+自定义) 落地
    check("恢复后按 model 重建",
          JSON.stringify(node2.widgets[1].options.values) === JSON.stringify(PAYLOAD.values["Wan2.2 T2V"]));
    check("恢复后值保持", node2.widgets[1].value === "1280x720 (16:9)");

    // ── 恢复保留"库外"自定义值（删库/离线旧值仍可执行）──
    const node4 = makeNode("Z-Image (Turbo)", "999x999 (Gone)");
    ext.nodeCreated(node4);
    await new Promise((r) => setTimeout(r, 10));
    node4.onAfterGraphConfigured();
    await new Promise((r) => setTimeout(r, 10));
    check("恢复保留库外自定义值", node4.widgets[1].value === "999x999 (Gone)");
    check("恢复仍重建官方选项", JSON.stringify(node4.widgets[1].options.values)
        === JSON.stringify(PAYLOAD.values["Z-Image (Turbo)"]));

    console.log();
    if (failures.length) { console.log("FAILED:", failures.length, "项"); process.exit(1); }
    console.log("ALL PASS");
})();
