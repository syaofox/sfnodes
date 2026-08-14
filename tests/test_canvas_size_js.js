// SFCanvasSizePreset 前端联动冒烟测试（Node 直接运行：node tests/test_canvas_size_js.js）
// 用 mock DOM/app/api/fetch 真实加载 web/canvas_size.js，验证：
//   - 扩展注册（sfnodes.CanvasSizePreset）
//   - nodeCreated：model callback 包装、初始静态选项
//   - fetch 数据就绪后按当前 model 重建 resolution 选项（含 --档位-- 分组头）
//   - 切换 model：当前值保持（在列表内）/ 回退首个非分组头（不在列表内）
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
        remove() {}, querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        addEventListener() {}, removeEventListener() {},
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
    canvas: { setDirty() {} },
    registerExtension(ext) { this._ext = ext; },
};
globalThis.window.app = globalThis.app;
globalThis.api = {};

let fetchResult = null; // 测试可控：null = 失败降级
globalThis.fetch = async () => {
    if (!fetchResult) throw new Error("fetch failed (mock)");
    return { ok: true, json: async () => fetchResult };
};

// ── mock 预设数据（与 Python 常量形状一致的最小集，含分组头）──
const PAYLOAD = {
    models: [
        "-- Image --", "Z-Image (Turbo)", "Qwen-Image (2512)", "Flux.1 (dev/schnell)",
        "Krea 2 (Turbo/RAW)", "Flux.2 Klein 9B", "SDXL / SD 3.5",
        "-- Video --", "Wan2.2 T2V", "Wan2.2 I2V", "Wan2.2 TI2V-5B",
        "HunyuanVideo 1.5", "LTX-2.5",
    ],
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
            "--540p--", "960x540 (16:9)", "540x960 (9:16)",
        ],
        "LTX-2.5": [
            "--0.9MP--", "1216x704 (19:11)", "704x1216 (11:19)",
            "--1K--", "1024x1024 (1:1)", "1024x576 (16:9)",
        ],
        "Flux.2 Klein 9B": ["--1K--", "1024x1024 (1:1)", "--2K--", "2048x1152 (16:9)"],
    },
};

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
        setDirtyCanvas() {},
        onAfterGraphConfigured: null,
    };
}

const head = (arr) => arr.find((v) => !(v.startsWith("--") && v.endsWith("--")));

(async () => {
    // ── 先测 fetch 失败降级（独立 tmp 目录 = 独立模块实例，无 promise 缓存）──
    {
        const tmpFail = fs.mkdtempSync(path.join(os.tmpdir(), "sf_csz_fail_"));
        for (const n of ["sf_common.js", "canvas_size.js"]) {
            const code = fs
                .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
                .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
                .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
                .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
            fs.writeFileSync(path.join(tmpFail, n.replace(/\.js$/, ".mjs")), code);
        }
        fetchResult = null;
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
    for (const n of ["sf_common.js", "canvas_size.js"]) {
        const code = fs
            .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
            .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
            .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
        fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
    }
    fetchResult = PAYLOAD;
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
    await new Promise((r) => setTimeout(r, 10)); // 让 fetch promise 落地
    check("fetch 后重建为 Z-Image 表", JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["Z-Image (Turbo)"]));
    check("当前值保持", resWidget.value === "1024x1024 (1:1)");

    // ── 切换 model：值在新列表 → 保持 ──
    modelWidget.value = "Flux.2 Klein 9B";
    modelWidget.callback("Flux.2 Klein 9B");
    // 切换瞬间（fetch 微任务未落地）：选项保持旧表，绝不被默认模型表污染
    check("切换瞬间选项保持旧表", JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["Z-Image (Turbo)"]));
    await new Promise((r) => setTimeout(r, 10));
    check("切到 Klein 重建表", JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["Flux.2 Klein 9B"]));
    check("值在列表中保持", resWidget.value === "1024x1024 (1:1)");

    // ── 切换 model：值不在新列表 → 回退首个非分组头 ──
    modelWidget.value = "Wan2.2 T2V";
    modelWidget.callback("Wan2.2 T2V");
    check("二次切换瞬间保持 Klein 表", JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["Flux.2 Klein 9B"]));
    await new Promise((r) => setTimeout(r, 10));
    check("切到 Wan2.2 T2V 重建表", JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["Wan2.2 T2V"]));
    check("值回退到 832x480 (26:15)", resWidget.value === head(PAYLOAD.values["Wan2.2 T2V"]));

    // ── 切到新模型（LTX-2.5，跨分组）──
    modelWidget.value = "LTX-2.5";
    modelWidget.callback("LTX-2.5");
    await new Promise((r) => setTimeout(r, 10));
    check("切到 LTX-2.5 重建表", JSON.stringify(resWidget.options.values) === JSON.stringify(PAYLOAD.values["LTX-2.5"]));
    check("值回退到 1216x704 (19:11)", resWidget.value === head(PAYLOAD.values["LTX-2.5"]));

    // ── onAfterGraphConfigured 恢复场景（直接赋链接不触发 callback）──
    const node2 = makeNode("Wan2.2 T2V", "1280x720 (16:9)");
    ext.nodeCreated(node2);
    await new Promise((r) => setTimeout(r, 10));
    node2.widgets[0].value = "Wan2.2 T2V"; // 模拟 widget 值恢复（不触发 callback）
    node2.onAfterGraphConfigured();
    await new Promise((r) => setTimeout(r, 10));
    check("恢复后按 model 重建", JSON.stringify(node2.widgets[1].options.values) === JSON.stringify(PAYLOAD.values["Wan2.2 T2V"]));
    check("恢复后值保持", node2.widgets[1].value === "1280x720 (16:9)");

    console.log();
    if (failures.length) { console.log("FAILED:", failures.length, "项"); process.exit(1); }
    console.log("ALL PASS");
})();
