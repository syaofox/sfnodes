// SF Power Lora Loader lora 项显示方式（sfnodes.PowerLoraLoader.DisplayName）
// 冒烟测试（Node 直接运行：node tests/test_power_lora_display_smoke.js）
// mock app 后真实加载 web/power_lora_loader.js，验证：
//   - displayLoraName 五模式转换（full/filename/basename/folder/parent_basename）+ 边界
//   - 扩展 init 注册设置（id/选项/默认值）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock app ──
const addedSettings = [];
let settings = {};
globalThis.app = {
    ui: { settings: { addSetting: (s) => addedSettings.push(s) } },
    graph: { setDirtyCanvas() {} },
    canvas: { ds: { scale: 1 }, editor_alpha: 1 },
    registerExtension(ext) { this._ext = ext; },
};
Object.defineProperty(globalThis.app.ui.settings, "getSettingValue", {
    value: (id) => settings[id] ?? undefined,
    writable: true,
});

// ── 加载模块（改 import + 追加导出纯函数）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pld_"));
let code = fs
    .readFileSync(path.join(__dirname, "..", "web", "power_lora_loader.js"), "utf8")
    .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
    // sf_lora_info 的 import 也被替换为相对模块——只需提供同名 stub
    .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
code += "\n\nexport { displayLoraName, DISPLAY_MODES, DISPLAY_MODE_SETTING };\n";
fs.writeFileSync(path.join(tmpDir, "power_lora_loader.mjs"), code);
fs.writeFileSync(path.join(tmpDir, "sf_lora_info.mjs"),
    "export const loraMetadataCache = new Map();\n" +
    "export const getLoraMetadata = async () => null;\n" +
    "export const showLoraInfoDialog = () => {};\n" +
    "export const ensureEventHook = () => {};\n" +
    "export const getLastCanvasEvent = () => null;\n");
// sf_common 复制真实文件（loraDisplayName/loraRowLabel 单一真源在此直测；
// 顶层 installGraphLoadingGuard 因 mock app 无 loadGraphData 而早退）
fs.writeFileSync(path.join(tmpDir, "sf_common.mjs"),
    fs.readFileSync(path.join(__dirname, "..", "web", "sf_common.js"), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = {};"));

(async () => {
    const mod = await import(path.join(tmpDir, "power_lora_loader.mjs"));
    const { displayLoraName, DISPLAY_MODES, DISPLAY_MODE_SETTING } = mod;
    const commonMod = await import(path.join(tmpDir, "sf_common.mjs"));
    const { loraRowLabel, loraDisplayName, getLoraDisplayMode } = commonMod;
    const { app } = globalThis;

    // ── displayLoraName 转换 ──
    const L = "sdxl/style/beauty.safetensors";
    check("full：完整相对路径原样", displayLoraName(L, DISPLAY_MODES.FULL) === L);
    check("filename：文件名含扩展名", displayLoraName(L, DISPLAY_MODES.FILENAME) === "beauty.safetensors");
    check("basename：去扩展名", displayLoraName(L, DISPLAY_MODES.BASENAME) === "beauty");
    check("folder：最近一层文件夹名", displayLoraName(L, DISPLAY_MODES.FOLDER) === "style");
    check("folder 多层取最近层", displayLoraName("a/b/c/x.safetensors", DISPLAY_MODES.FOLDER) === "c");
    check("folder 根目录文件降级为文件名", displayLoraName("model.safetensors", DISPLAY_MODES.FOLDER) === "model.safetensors");
    check("parent_basename：上级目录+去扩展名文件名", displayLoraName(L, DISPLAY_MODES.PARENT_BASENAME) === "style/beauty");
    check("parent_basename 多层取最近层目录", displayLoraName("a/b/c/x.safetensors", DISPLAY_MODES.PARENT_BASENAME) === "c/x");
    check("parent_basename 根目录文件降级 basename", displayLoraName("model.safetensors", DISPLAY_MODES.PARENT_BASENAME) === "model");
    check("parent_basename 反斜杠路径", displayLoraName("sub\\dir\\x.safetensors", DISPLAY_MODES.PARENT_BASENAME) === "dir/x");
    check("parent_basename 版本化名剥 .0 保留", displayLoraName("sdxl/MoXin_v1.0.safetensors", DISPLAY_MODES.PARENT_BASENAME) === "sdxl/MoXin_v1.0");
    check("parent_basename 无扩展名原样", displayLoraName("a/b/noext", DISPLAY_MODES.PARENT_BASENAME) === "b/noext");
    check("parent_basename 点开头文件", displayLoraName("a/.hidden", DISPLAY_MODES.PARENT_BASENAME) === "a/.hidden");
    check("None 原样", displayLoraName("None", DISPLAY_MODES.FILENAME) === "None");
    check("空值兜底", displayLoraName("", DISPLAY_MODES.FOLDER) === "None");
    check("basename 无扩展名原样", displayLoraName("noext", DISPLAY_MODES.BASENAME) === "noext");
    check("basename 点开头文件原样", displayLoraName(".hidden", DISPLAY_MODES.BASENAME) === ".hidden");
    check("反斜杠路径（Windows 风格）", displayLoraName("sub\\dir\\x.safetensors", DISPLAY_MODES.FOLDER) === "dir");
    check("filename 反斜杠取 basename", displayLoraName("sub\\dir\\x.safetensors", DISPLAY_MODES.FILENAME) === "x.safetensors");
    check("未知模式回退完整路径", displayLoraName(L, "bogus") === L);

    // ── 设置注册 ──
    check("扩展已注册", !!app._ext && app._ext.name === "sfnodes.SFPowerLoraLoader");
    const ext = app._ext;
    ext.init();
    const st = addedSettings.find((s) => s.id === DISPLAY_MODE_SETTING);
    check("设置已注册", !!st && st.type === "combo");
    check("设置默认值 = full", st.defaultValue === DISPLAY_MODES.FULL);
    check("五选项齐全", Array.isArray(st.options()) && st.options().length === 5);
    check("选项值集合", ["full", "filename", "basename", "folder", "parent_basename"]
        .every((v) => st.options().some((o) => o.value === v)));

    // ── getDisplayMode 读取当前设置 ──
    settings[DISPLAY_MODE_SETTING] = "filename";
    check("设置读取 filename 生效（函数层）", displayLoraName("a/b.safetensors", "filename") === "b.safetensors");
    settings[DISPLAY_MODE_SETTING] = "folder";
    check("设置读取 folder 生效（函数层）", displayLoraName("a/b.safetensors", "folder") === "a");
    check("设置默认（未设时）回退 full", displayLoraName("a/b.safetensors", "full") === "a/b.safetensors");

    // ── loraRowLabel（SFLoraStack/SFLoraPlot 行名：全局模式 + full 回退）──
    const set = (m) => {
        if (m == null) delete settings[DISPLAY_MODE_SETTING];
        else settings[DISPLAY_MODE_SETTING] = m;
    };
    set("full");
    check("row full+hideExt=true 剥模型扩展名", loraRowLabel("sdxl/style/beauty.safetensors", true) === "beauty");
    check("row full+hideExt=false 保留扩展名", loraRowLabel("sdxl/style/beauty.safetensors", false) === "beauty.safetensors");
    set("filename");
    check("row filename 含扩展名（hideExt 让位）", loraRowLabel("sdxl/style/beauty.safetensors", true) === "beauty.safetensors");
    set("basename");
    check("row basename 去扩展名", loraRowLabel("sdxl/style/beauty.safetensors", false) === "beauty");
    check("row basename 版本化名保留 .0", loraRowLabel("sdxl/MoXin_v1.0.safetensors", true) === "MoXin_v1.0");
    check("row basename 无模型扩展名也剥（lastIndexOf 语义）", loraRowLabel("sdxl/xyz.v1.0", true) === "xyz.v1");
    set("full");
    check("row full 无模型扩展名保留（白名单语义）", loraRowLabel("sdxl/xyz.v1.0", true) === "xyz.v1.0");
    set("folder");
    check("row folder 最近文件夹名", loraRowLabel("sdxl/style/beauty.safetensors", true) === "style");
    check("row folder 根目录文件降级文件名", loraRowLabel("beauty.safetensors", true) === "beauty.safetensors");
    set("parent_basename");
    check("row parent_basename 上级目录+去扩展名", loraRowLabel("sdxl/style/beauty.safetensors", true) === "style/beauty");
    check("row parent_basename 根目录文件降级 basename", loraRowLabel("beauty.safetensors", true) === "beauty");
    check("row parent_basename 版本化名剥 .0 保留", loraRowLabel("sdxl/MoXin_v1.0.safetensors", true) === "sdxl/MoXin_v1.0");
    check("row parent_basename 反斜杠路径", loraRowLabel("sub\\dir\\x.safetensors", true) === "dir/x");
    set("full");
    check("row 反斜杠路径", loraRowLabel("sub\\dir\\x.safetensors", true) === "x");
    set(null);
    check("row 设置未设回退 full+hideExt", loraRowLabel("a/b.safetensors", true) === "b");
    check("getLoraDisplayMode 未设回退 full", getLoraDisplayMode() === "full");
    set("folder");
    check("getLoraDisplayMode 读取设置", getLoraDisplayMode() === "folder");

    // ── 共享实现一致性：Power 行与 Stack/Plot 行同源 ──
    check("loraDisplayName 与 Power 行同实现", loraDisplayName("a/b.safetensors", "filename") === "b.safetensors");

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(1);
});
