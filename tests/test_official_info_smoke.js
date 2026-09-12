// 官方节点 info 图标接线冒烟测试（Node 直接运行：node tests/test_official_info_smoke.js）
// 验证：
//   1. lora_loader.js / lora_loader_model_only.js / sf_load_diffusion_model.js
//      的 NODE_TYPES 同时包含 SF 类型与官方类型（LoraLoader / LoraLoaderModelOnly / UNETLoader）
//   2. sf_lora_info.setupLoaderInfoWidget 在官方同款 combo 名（lora_name / unet_name）
//      上可装配 _info widget，且 configure 重放后不重复（幂等）
//   3. 官方开关 sfnodes.OfficialInfo.Enabled：默认开（未注册/异常回退 true）；
//      enabledOf 门控装配与 configure；注册函数幂等；开关切换即时增删官方节点 widget
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

globalThis.document = {
    createElement: () => ({ style: {}, dataset: {}, children: [], className: "",
        addEventListener() {}, removeEventListener() {}, appendChild(c) { return c; },
        querySelector() { return null; }, querySelectorAll() { return []; },
        getBoundingClientRect: () => ({ left: 0, top: 0, right: 10, bottom: 10, width: 10, height: 10 }),
    }),
    createTextNode: (t) => ({ textContent: String(t) }),
    body: { appendChild(c) { return c; }, contains: () => false },
    head: { appendChild() {} },
    addEventListener() {}, removeEventListener() {}, getElementById: () => null,
};
globalThis.window = { addEventListener() {}, removeEventListener() {}, innerWidth: 1280, innerHeight: 720 };
globalThis.navigator = { clipboard: { writeText: async () => {} } };
// 设置 mock：值可配，供开关默认/显式断言；addSetting 计数供幂等断言
const settingsStore = { values: {}, addCalls: [] };
globalThis.app = { graph: { _nodes: [], setDirtyCanvas() {} }, canvas: { ds: { scale: 1 } },
    api: { fetchApi: async () => ({ ok: false }) },
    ui: { settings: {
        getSettingValue: (id) => (id in settingsStore.values ? settingsStore.values[id] : null),
        addSetting: (def) => { settingsStore.addCalls.push(def); },
    } } };
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}) });
globalThis.LGraphCanvas = function () {};
globalThis.LGraphCanvas.prototype.adjustMouseEvent = function () {};
globalThis.LiteGraph = { WIDGET_TEXT_COLOR: "#fff" };

// ── 1. 静态接线 ──
const webDir = path.join(__dirname, "..", "web");
function src(n) { return fs.readFileSync(path.join(webDir, n), "utf8"); }
check("lora_loader 覆盖官方 LoraLoader",
    src("lora_loader.js").includes('"LoraLoader"') && src("lora_loader.js").includes('"SFLoraLoader"'));
check("lora_loader_model_only 覆盖官方 LoraLoaderModelOnly",
    src("lora_loader_model_only.js").includes('"LoraLoaderModelOnly"'));
check("sf_load_diffusion_model 覆盖官方 UNETLoader",
    src("sf_load_diffusion_model.js").includes('"UNETLoader"')
    && src("sf_load_diffusion_model.js").includes('"SFLoadDiffusionModel"'));
check("扩展名保持 sfnodes.* 前缀",
    (src("lora_loader.js").match(/registerExtension\(\{\s*name:\s*"([^"]+)"/) || [])[1]?.startsWith("sfnodes.")
    && (src("lora_loader_model_only.js").match(/registerExtension\(\{\s*name:\s*"([^"]+)"/) || [])[1]?.startsWith("sfnodes.")
    && (src("sf_load_diffusion_model.js").match(/registerExtension\(\{\s*name:\s*"([^"]+)"/) || [])[1]?.startsWith("sfnodes."));

// ── 2. 行为：setupLoaderInfoWidget 在官方同款 combo 上装配 ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_off_info_"));
for (const n of ["sf_lora_stack_core.js", "sf_lora_stack_api.js", "sf_lora_stack_settings.js",
    "sf_common.js", "sf_markdown.js", "sf_lora_shared_info.js",
    "sf_lora_stack_dropdown.js", "sf_lora_stack_render.js", "sf_lora_stack_interaction.js",
    "sf_workflows_ui.js", "sf_workflows_lib.js"]) {
    if (!fs.existsSync(path.join(webDir, n))) continue;
    const code = fs.readFileSync(path.join(webDir, n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}
// sf_lora_stack_info 体积大但 setupLoaderInfoWidget 经它转调 openInfoPanelFor；
// 此处用桩替代（只验证装配，不打开面板）
fs.writeFileSync(path.join(tmpDir, "sf_lora_stack_info.mjs"),
    "export async function openInfoPanelFor(){}\n" +
    "export function closeInfoPanel(){}; export function closeInfoPanelFor(){};");
const infoCode = src("sf_lora_info.js")
    .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
    .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
fs.writeFileSync(path.join(tmpDir, "sf_lora_info.mjs"), infoCode);

function fakeNode(comboName, value, comfyClass = "official") {
    const combo = { name: comboName, value, callback: null };
    return { id: 7, comfyClass, widgets: [combo], configure(info) {},
        setDirtyCanvas() {} };
}

(async () => {
    const mod = await import(path.join(tmpDir, "sf_lora_info.mjs"));
    check("setupLoaderInfoWidget 已导出", typeof mod.setupLoaderInfoWidget === "function");

    // 官方 LoraLoader 同款：lora_name
    const n1 = fakeNode("lora_name", "test/a.safetensors");
    mod.setupLoaderInfoWidget(n1, "lora_name", { prefetch: null });
    check("lora_name 装配出 _info", n1.widgets.some((w) => w.name === "_info"));
    n1.configure({});
    check("configure 后 _info 唯一（幂等）",
        n1.widgets.filter((w) => w.name === "_info").length === 1);

    // 官方 UNETLoader 同款：unet_name
    const n2 = fakeNode("unet_name", "checkpoints/b.safetensors");
    mod.setupLoaderInfoWidget(n2, "unet_name", { prefetch: null });
    check("unet_name 装配出 _info", n2.widgets.some((w) => w.name === "_info"));

    // ── 3. 官方开关 ──
    check("开关默认开（未注册回退 true）", mod.isOfficialInfoEnabled() === true);
    settingsStore.values[mod.OFFICIAL_INFO_SETTING] = false;
    check("显式 false 即关", mod.isOfficialInfoEnabled() === false);
    settingsStore.values[mod.OFFICIAL_INFO_SETTING] = true;

    // enabledOf 门控：关时不装配，configure 亦不补
    const n3 = fakeNode("lora_name", "test/c.safetensors");
    mod.setupLoaderInfoWidget(n3, "lora_name", { prefetch: null, enabledOf: () => false });
    check("enabledOf=false 不装配", !n3.widgets.some((w) => w.name === "_info"));
    n3.configure({});
    check("enabledOf=false 时 configure 不补挂", !n3.widgets.some((w) => w.name === "_info"));

    // 重复 setup 仍幂等（开关即时重挂路径）
    const n4 = fakeNode("lora_name", "test/d.safetensors");
    mod.setupLoaderInfoWidget(n4, "lora_name", { prefetch: null });
    mod.setupLoaderInfoWidget(n4, "lora_name", { prefetch: null });
    n4.configure({});
    check("重复 setup 后 _info 唯一",
        n4.widgets.filter((w) => w.name === "_info").length === 1);

    // 注册幂等（三文件各调一次只注册一项）
    const before = settingsStore.addCalls.length;
    mod.registerOfficialInfoSettingOnce();
    mod.registerOfficialInfoSettingOnce();
    const added = settingsStore.addCalls.slice(before);
    check("设置只注册一次", added.length === 1);
    check("设置默认 true", added[0] && added[0].id === mod.OFFICIAL_INFO_SETTING
        && added[0].type === "boolean" && added[0].defaultValue === true);

    // 即时刷新：挂载 spec 后切开关，官方节点 widget 增删，非官方不动
    mod.registerOfficialInfoSpec({ classes: ["LoraLoader"], comboName: "lora_name", opts: { prefetch: null } });
    const off = fakeNode("lora_name", "test/e.safetensors", "LoraLoader");
    const sf = fakeNode("lora_name", "test/f.safetensors", "SFLoraLoader");
    mod.setupLoaderInfoWidget(off, "lora_name", { prefetch: null });
    mod.setupLoaderInfoWidget(sf, "lora_name", { prefetch: null });
    globalThis.app.graph._nodes = [off, sf];
    const onChange = settingsStore.addCalls[settingsStore.addCalls.length - 1].onChange;
    check("onChange 已导出", typeof onChange === "function");
    settingsStore.values[mod.OFFICIAL_INFO_SETTING] = false;
    // onChange 经 setTimeout 推迟（Accent 时序先例），此处直调需等一 tick
    await new Promise((r) => { onChange(); setTimeout(r, 10); });
    check("关后官方节点 _info 被摘除", !off.widgets.some((w) => w.name === "_info"));
    check("关后 SF 节点 _info 保留", sf.widgets.some((w) => w.name === "_info"));
    settingsStore.values[mod.OFFICIAL_INFO_SETTING] = true;
    await new Promise((r) => { onChange(); setTimeout(r, 10); });
    check("开后官方节点 _info 恢复", off.widgets.some((w) => w.name === "_info"));

    console.log(failures.length ? `\nFAILED: ${failures.length}` : "\nALL PASS");
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => { console.error(e); try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {} process.exit(1); });
