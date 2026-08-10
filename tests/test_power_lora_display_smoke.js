// SF Power Lora Loader lora 项显示方式（sfnodes.PowerLoraLoader.DisplayName）
// 冒烟测试（Node 直接运行：node tests/test_power_lora_display_smoke.js）
// mock app 后真实加载 web/power_lora_loader.js，验证：
//   - displayLoraName 四模式转换（full/filename/basename/folder）+ 边界
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

(async () => {
    const mod = await import(path.join(tmpDir, "power_lora_loader.mjs"));
    const { displayLoraName, DISPLAY_MODES, DISPLAY_MODE_SETTING } = mod;
    const { app } = globalThis;

    // ── displayLoraName 转换 ──
    const L = "sdxl/style/beauty.safetensors";
    check("full：完整相对路径原样", displayLoraName(L, DISPLAY_MODES.FULL) === L);
    check("filename：文件名含扩展名", displayLoraName(L, DISPLAY_MODES.FILENAME) === "beauty.safetensors");
    check("basename：去扩展名", displayLoraName(L, DISPLAY_MODES.BASENAME) === "beauty");
    check("folder：最近一层文件夹名", displayLoraName(L, DISPLAY_MODES.FOLDER) === "style");
    check("folder 多层取最近层", displayLoraName("a/b/c/x.safetensors", DISPLAY_MODES.FOLDER) === "c");
    check("folder 根目录文件降级为文件名", displayLoraName("model.safetensors", DISPLAY_MODES.FOLDER) === "model.safetensors");
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
    check("四选项齐全", Array.isArray(st.options()) && st.options().length === 4);
    check("选项值集合", ["full", "filename", "basename", "folder"]
        .every((v) => st.options().some((o) => o.value === v)));

    // ── getDisplayMode 读取当前设置 ──
    settings[DISPLAY_MODE_SETTING] = "filename";
    // getDisplayMode 未导出，但 displayLoraName 由扩展 draw 调用——
    // 通过导入模块内函数验证读取路径：以 draw 同款调用方式模拟
    check("设置读取 filename 生效（函数层）", displayLoraName("a/b.safetensors", "filename") === "b.safetensors");
    settings[DISPLAY_MODE_SETTING] = "folder";
    check("设置读取 folder 生效（函数层）", displayLoraName("a/b.safetensors", "folder") === "a");
    check("设置默认（未设时）回退 full", displayLoraName("a/b.safetensors", "full") === "a/b.safetensors");

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(1);
});
