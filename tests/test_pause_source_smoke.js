// SFPauseImage 外载图模块冒烟测试（Node 直接运行：node tests/test_pause_source_smoke.js）
// 用 mock DOM/app/api/fetch 真实加载 sf_pause_source.js，验证：
//   - extension 注册 + onNodeCreated 构建 Load/Browse/Clear 行并插到预览前
//   - 拖放监听得已挂；Ctrl+V 粘贴处理器以 disconnectInput:false 安装
//   - loadSource：CropAPI.uploadSrc → POST /api/sfnodes/pause/load → frame 回填 + 预览
//   - Clear 复位 frame/预览/Clear 禁用
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
    return {
        className: "", textContent: "", title: "", disabled: false, value: "",
        style: {}, dataset: {}, children: [], _listeners: {},
        append(...kids) { this.children.push(...kids); },
        before(x) { this._before = this._before || []; this._before.push(x); },
        removeAttribute() {},
        addEventListener(type, fn) { this._listeners[type] = fn; },
        click() {},
    };
}
globalThis.document = {
    createElement() { return makeEl(); },
    body: { appendChild() {} },
};

// ── mock app / api / 外部依赖 ──
const pasteInstalls = [];
const browseCalls = [];
let uploadCalls = 0;
const fetchCalls = [];
globalThis.app = { registerExtension(ext) { this._ext = ext; } };
globalThis.window = { app: globalThis.app };
globalThis.api = { apiURL: (r) => r };
globalThis.fetch = async (url, opts) => {
    fetchCalls.push({ url, opts });
    return { ok: true, status: 200, json: async () => ({ status: "success", frame: { filename: "sf_pause_1.png", subfolder: "", type: "temp" } }) };
};
globalThis.__installPasteHandler = (opts) => { pasteInstalls.push(opts); };
globalThis.__parseAnnotatedImageValue = (v) => ({ filename: v });
globalThis.__buildSourceURL = (p) => `/view?${p.filename}`;
globalThis.__showImageBrowser = (node, opts) => { browseCalls.push(opts); };
globalThis.__CropAPI = {
    uploadSrc: async () => { uploadCalls++; return { path: "sfnodes_crop/crop_src_x.png" }; },
};

// ── 加载模块（重依赖全桩掉）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pause_src_"));
{
    const code = fs.readFileSync(path.join(__dirname, "..", "web", "sf_pause_source.js"), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/import \{ installPasteHandler, parseAnnotatedImageValue, buildSourceURL \} from "\.\/sf_common\.js";/,
            "const installPasteHandler = globalThis.__installPasteHandler; const parseAnnotatedImageValue = globalThis.__parseAnnotatedImageValue; const buildSourceURL = globalThis.__buildSourceURL;")
        .replace(/import \{ showImageBrowser \} from "\.\/image_browser\.js";/,
            "const showImageBrowser = globalThis.__showImageBrowser;")
        .replace(/import \{ CropAPI \} from "\.\/sf_crop_core\.js";/,
            "const CropAPI = globalThis.__CropAPI;");
    fs.writeFileSync(path.join(tmpDir, "sf_pause_source.mjs"), code);
}

(async () => {
    const mod = await import(path.join(tmpDir, "sf_pause_source.mjs"));

    const showFrames = [];
    const flashes = [];
    const gate = {
        state: { getState: (n) => n.properties.pauseImageState },
        body: {
            showFrame: (n, f) => { showFrames.push(f); n._sfPauseImageHasSnapshot = true; },
            renderPause: () => {},
        },
        flash: (n, m) => { flashes.push(m); },
        props: { hasSnapProp: "_sfPauseImageHasSnapshot" },
    };
    const src = mod.attachSourceControls({
        gate, classy: "SFPauseImage", propPrefix: "_sfPauseImage",
        cssPrefix: "sf-pi-", emptyText: "empty", logTag: "t",
    });

    const ext = app._ext;
    check("extension 已注册", !!ext && ext.name === "sfnodes.SFPauseImageSource");
    const proto = {};
    ext.beforeRegisterNodeDef({ name: "SFPauseImage", prototype: proto }, { name: "SFPauseImage" });
    check("原型钩子已安装", typeof proto.onNodeCreated === "function" &&
        typeof proto.onConfigure === "function" && typeof proto.getExtraMenuOptions === "function");

    const els = {
        preview: makeEl(), img: makeEl(), empty: makeEl(), dims: makeEl(), status: makeEl(),
        segPause: makeEl(), segPass: makeEl(),
        btnContinue: makeEl(), btnRegen: makeEl(), btnFlip: makeEl(),
        btnCopy: makeEl(), btnSaveDisk: makeEl(), btnSaveOut: makeEl(), btnOpen: makeEl(),
    };
    const node = {
        id: 3, comfyClass: "SFPauseImage", properties: { pauseImageState: { gate: "pause", frame: null, flip: false } },
        size: [300, 300], inputs: [{ name: "image", link: null }], _sfPauseImageEls: els,
    };
    proto.onNodeCreated.call(node);
    check("Load/Browse/Clear 按钮已建", !!els.btnLoad && !!els.btnBrowse && !!els.btnClear);
    check("按钮行插到预览前", Array.isArray(els.preview._before) && els.preview._before.length === 1);
    check("Clear 初始禁用", els.btnClear.disabled === true);
    check("拖放 dragover 监听已挂", typeof els.preview._listeners.dragover === "function");
    check("拖放 drop 监听已挂", typeof els.preview._listeners.drop === "function");
    check("粘贴处理器以 disconnectInput:false 安装",
        pasteInstalls.length === 1 && pasteInstalls[0].disconnectInput === false &&
        pasteInstalls[0].comfyClass === "SFPauseImage");

    // ── loadSource：上传 → 物化 → 回填 frame/预览 ──
    await src.loadSource(node, "data:image/png;base64,AAA");
    check("CropAPI.uploadSrc 被调用", uploadCalls === 1);
    check("调用了 /api/sfnodes/pause/load",
        fetchCalls.some((c) => c.url === "/api/sfnodes/pause/load" && c.opts.method === "POST"));
    check("state.frame 已置 temp 记录",
        node.properties.pauseImageState.frame && node.properties.pauseImageState.frame.filename === "sf_pause_1.png");
    check("showFrame 已调用", showFrames.length === 1);
    check("hasSnap 置真", node._sfPauseImageHasSnapshot === true);
    check("Clear 启用", els.btnClear.disabled === false);

    // ── Clear：复位 ──
    src.clearSource(node);
    check("Clear 后 frame=null", node.properties.pauseImageState.frame === null);
    check("Clear 后 hasSnap false", node._sfPauseImageHasSnapshot === false);
    check("Clear 后按钮禁用", els.btnClear.disabled === true);

    // ── Browse：弹窗选择器回调 → loadSource ──
    els.btnBrowse._listeners.click({ stopPropagation() {} });
    check("Browse 打开图片浏览器", browseCalls.length === 1 && typeof browseCalls[0].onPick === "function");

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
