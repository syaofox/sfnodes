import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}
const tick = () => new Promise((r) => setTimeout(r, 0));

const WEB = path.resolve(__dirname, "..", "web");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sf_k2p_"));

// ── stub app ──
globalThis.__SF_TEST_NODES__ = [];
fs.writeFileSync(path.join(tmp, "stub_app.mjs"),
    "export const app = { graph: { _nodes: globalThis.__SF_TEST_NODES__ } };\n");
// ── stub sf_popup（真实无依赖模块，但保持纯净：空实现即可，本测试不打开 popup）──
fs.writeFileSync(path.join(tmp, "sf_popup.mjs"),
    "export function attachPopupDismiss(){ return () => {}; }\n" +
    "export function clampToViewport(){ return null; }\n");

// ── 转换并写模块 ──
let src = fs.readFileSync(path.join(WEB, "sf_krea2_presets.js"), "utf-8");
src = src.replace('import { app } from "/scripts/app.js";', 'import { app } from "./stub_app.mjs";');
src = src.replace('from "./sf_popup.js"', 'from "./sf_popup.mjs"');
fs.writeFileSync(path.join(tmp, "sf_krea2_presets.mjs"), src);

// ── mock document / CustomEvent / fetch ──
const dispatched = [];
globalThis.CustomEvent = class { constructor(type, opts) { this.type = type; this.detail = opts?.detail; } };
globalThis.document = {
    addEventListener() {},
    dispatchEvent(ev) { dispatched.push(ev.type); },
    createElement: () => ({ style: {}, appendChild() {}, setProperty() {} }),
    getElementById: () => null,
    body: { appendChild() {} },
    head: { appendChild() {} },
};
const requests = [];
globalThis.fetch = async (url, opts = {}) => {
    const method = opts.method || "GET";
    requests.push({ url, method, body: opts.body ? JSON.parse(opts.body) : undefined });
    let data;
    if (method === "GET") data = { presets: { default: "D", a: "A", u: "U" }, builtin: { default: "D", a: "A" }, user: { u: "U" }, deleted: [] };
    else if (method === "POST" && url.endsWith("/reset")) data = { reset: true };
    else data = { ok: true };
    return { ok: true, json: async () => data };
};

(async () => {
    const mod = await import(path.join(tmp, "sf_krea2_presets.mjs"));

    // fetchPresets
    const data = await mod.fetchPresets("interrogator");
    check("fetchPresets 解析 presets", data.presets.default === "D");

    // savePreset POST body
    await mod.savePreset("interrogator", "新预设", "文本");
    const saveReq = requests.find((r) => r.method === "POST" && r.url === "/api/sfnodes/interrogator_presets");
    check("savePreset POST 正确 body", saveReq && saveReq.body.name === "新预设" && saveReq.body.text === "文本");

    // deletePreset DELETE 带 name query
    await mod.deletePreset("krea2", "foo bar");
    const delReq = requests.find((r) => r.method === "DELETE");
    check("deletePreset DELETE name 编码", delReq && delReq.url.includes("name=foo%20bar"));

    // resetAll POST /reset {all:true}
    await mod.resetAllPresets("interrogator");
    const resetReq = requests.find((r) => r.method === "POST" && r.url.endsWith("/reset"));
    check("resetAllPresets POST all", resetReq && resetReq.body.all === true);

    // setPresetOptions：重建 options（ComfyUI combo 用 {values:[...]}）并保留当前值
    const fakeNode = {
        widgets: [{ name: "preset", value: "a", options: {} }],
        setDirtyCanvas: () => {},
    };
    mod.setPresetOptions(fakeNode, { default: "D", a: "A", u: "U" });
    check("setPresetOptions 重建 options.values", JSON.stringify(fakeNode.widgets[0].options.values) === JSON.stringify(["default", "a", "u"]));
    check("setPresetOptions 保留当前值", fakeNode.widgets[0].value === "a");

    // nodesOfClass / reloadNodes（重建但不广播——事件监听回调用它防无限循环）
    globalThis.__SF_TEST_NODES__.push(
        { comfyClass: "SFImageInterrogator", widgets: [{ name: "preset" }], setDirtyCanvas: () => {} },
        { comfyClass: "Other", widgets: [] },
    );
    dispatched.length = 0;
    const d2 = await mod.reloadNodes("interrogator", "SFImageInterrogator");
    check("reloadNodes 返回数据", d2 && d2.presets.a === "A");
    check("reloadNodes 不广播（防循环）", dispatched.length === 0);

    // refreshAllNodes（重建 + 广播）
    dispatched.length = 0;
    const r = await mod.refreshAllNodes("interrogator", "SFImageInterrogator");
    check("refreshAllNodes 返回数据", r && r.presets.a === "A");
    const target = globalThis.__SF_TEST_NODES__[0];
    check("refreshAllNodes 重建目标节点 options", JSON.stringify(target.widgets[0].options.values) === JSON.stringify(["default", "a", "u"]));
    check("refreshAllNodes 广播事件", dispatched.includes("sfnodes.interrogator-presets-changed"));

    // presetsChangedEvent 命名
    check("presetsChangedEvent 命名", mod.presetsChangedEvent("krea2") === "sfnodes.krea2-presets-changed");

    if (failures.length) { console.log("\nFAIL:", failures.join(", ")); process.exit(1); }
    console.log("\nALL PASS");
})();
