// SF Model Info - showLoraInfoDialog shim 冒烟测试（Node 直接运行）
// 旧 dialog 已收敛为浮动面板（sf_lora_stack_info.openInfoPanelFor），本测试验证：
//   - showLoraInfoDialog 仍导出且为函数（向后兼容）
//   - 调用时委托 openInfoPanelFor，锚点取事件坐标，无事件时回退 null
//   - None/空名不打开
//   - getLoraMetadata / loraMetadataCache 仍可用
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

function makeEl(tag) {
    const el = {
        tagName: (tag || "div").toUpperCase(),
        style: {}, dataset: {}, children: [],
        className: "", _textContent: "", _innerHTML: "", value: "",
        addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
        appendChild(c) { this.children.push(c); return c; },
        contains() { return false; }, closest() { return null; },
        querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 }; },
        showModal() { this.open = true; }, close() { this.open = false; },
    };
    return el;
}
globalThis.document = {
    createElement(tag) { return makeEl(tag); },
    createTextNode(t) { return { textContent: String(t) }; },
    body: { children: [], appendChild(c) { this.children.push(c); return c; } },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {}, dispatchEvent() {}, getElementById() { return null; },
};
globalThis.window = { addEventListener() {}, removeEventListener() {}, innerWidth: 1280, innerHeight: 720 };
globalThis.navigator = { clipboard: { writeText: async () => {} } };
globalThis.CustomEvent = class { constructor(type, opts) { this.type = type; this.detail = opts?.detail; } };
globalThis.app = { graph: { setDirtyCanvas() {} }, canvas: { ds: { scale: 1 } }, api: { fetchApi: async () => ({ ok: false }) }, ui: { settings: { getSettingValue() { return null; } } } };
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}) });
globalThis.LGraphCanvas = function () {};
globalThis.LGraphCanvas.prototype.adjustMouseEvent = function () {};
globalThis.LiteGraph = { WIDGET_TEXT_COLOR: "#fff" };

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_li_shim_"));
// 桩：sf_lora_stack_info 捕获 openInfoPanelFor 调用
let captured = null;
fs.writeFileSync(path.join(tmpDir, "sf_lora_stack_info.mjs"),
    "export async function openInfoPanelFor(ctx, id) { globalThis.__captured = { ctx, id }; }\n" +
    "export function closeInfoPanel(){}; export function closeInfoPanelFor(){};");
fs.writeFileSync(path.join(tmpDir, "sf_lora_stack_settings.mjs"),
    "export function getNodeRect(){return {left:0,top:0,right:100,bottom:100,width:100,height:100}};");
fs.writeFileSync(path.join(tmpDir, "sf_lora_stack_core.mjs"),
    "export const BRAND='#f66744'; export function readState(){return {loras:[]}}; export function patchLora(){}; export function accentOf(){return '#f66744'};");
fs.writeFileSync(path.join(tmpDir, "sf_lora_shared_info.mjs"),
    "export function loadImageAsWorkflow(){};\n");
fs.writeFileSync(path.join(tmpDir, "sf_markdown.mjs"),
    "export function renderMarkdown(s){return String(s||'');}\n");
fs.writeFileSync(path.join(tmpDir, "sf_common.mjs"),
    "export async function copyText(t){return true;}\n" +
    "export function escapeHtml(s){return String(s);}\n" +
    "export function installWheelZoomPassthrough(){return ()=>{};}\n");
fs.writeFileSync(path.join(tmpDir, "sf_lora_stack_api.mjs"),
    "export const loraInfo = async ()=>({ok:false});\n" +
    "export const civitaiLookup = async ()=>({ok:false});\n" +
    "export const deleteCivitai = async ()=>({ok:true});\n" +
    "export const saveCivitaiThumb = async ()=>({ok:true});\n" +
    "export const getCivitaiAccount = async ()=>({ok:true});\n" +
    "export const setCivitaiAccount = async ()=>({ok:true});\n" +
    "export const migrateLoraData = async ()=>({ok:true});\n" +
    "export const saveCustomTriggers = async ()=>({ok:true});\n" +
    "export const saveCustomDescription = async ()=>({ok:true});\n" +
    "export const thumbUrl = (n)=>'/thumb';\n" +
    "export const invalidateInfo = ()=>{};\n");

const code = fs.readFileSync(path.join(__dirname, "..", "web", "sf_lora_info.js"), "utf8")
    .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
    .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
fs.writeFileSync(path.join(tmpDir, "sf_lora_info.mjs"), code);

(async () => {
    const mod = await import(path.join(tmpDir, "sf_lora_info.mjs"));
    check("showLoraInfoDialog 已导出", typeof mod.showLoraInfoDialog === "function");
    check("getLoraMetadata 已导出", typeof mod.getLoraMetadata === "function");
    check("loraMetadataCache 已导出", mod.loraMetadataCache instanceof Map);

    // None 不打开
    globalThis.__captured = null;
    mod.showLoraInfoDialog(null, "None", {});
    check("None 不打开面板", globalThis.__captured === null);
    mod.showLoraInfoDialog(null, "", {});
    check("空名不打开", globalThis.__captured === null);

    // 带事件坐标：anchorRect 返回事件附近矩形
    globalThis.__captured = null;
    mod.showLoraInfoDialog({ clientX: 123, clientY: 456 }, "test/lora_a.safetensors", { trigger_words: "a, b" });
    check("带事件时打开面板", globalThis.__captured && globalThis.__captured.id === "test/lora_a.safetensors");
    const rect = globalThis.__captured?.ctx?.anchorRect?.();
    check("anchorRect 取事件坐标", rect && rect.left === 123 && rect.top === 456);

    // 无事件时 anchorRect 为 null（让面板 place() 居中）
    globalThis.__captured = null;
    mod.showLoraInfoDialog(null, "test/lora_b.safetensors", {});
    check("无事件时仍打开", globalThis.__captured && globalThis.__captured.id === "test/lora_b.safetensors");
    check("无事件时 anchorRect 为 null", globalThis.__captured.ctx.anchorRect === null);

    // 触发词透传到 shim 行（首次）
    globalThis.__captured = null;
    mod.showLoraInfoDialog(null, "test/lora_c.safetensors", { trigger_words: "x, y, z" });
    const row = globalThis.__captured.ctx.getRow();
    check("shim 行已同步 trigger_words", Array.isArray(row.custom) && row.custom.length === 3);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("shim crashed:", e);
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
    process.exit(1);
});
