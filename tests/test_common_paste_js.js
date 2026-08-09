// sf_common.js installPasteHandler 多类回归测试（Node 直接运行：node tests/test_common_paste_js.js）
// 覆盖：SFImageCrop 与 SFInpaintCrop 各自安装监听器（去重键 comfyClass:hook），
// 同键重复安装幂等；粘贴事件按选中节点类分发到各自的 onPasteImage。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mocks ──
const pasteListeners = [];
globalThis.window = {
    addEventListener(type, fn, capture) { if (type === "paste") pasteListeners.push(fn); },
    removeEventListener() {},
};

const graphNodes = [];
globalThis.app = {
    graph: { _nodes: graphNodes, links: {}, remove() {} },
    canvas: {
        selected_nodes: {},
        current_node: null,
        node_over: null,
        ds: { scale: 1 },
    },
    loadGraphData: () => {},
};
globalThis.window.app = globalThis.app;
globalThis.api = { apiURL: (r) => r };

globalThis.FileReader = class {
    readAsDataURL() { this.result = "data:image/png;base64,AAA"; this.onload?.({ target: this }); }
};

// ── 加载模块 ──
(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_common_"));
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", "sf_common.js"), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;");
    fs.writeFileSync(path.join(tmpDir, "sf_common.mjs"), code);
    const mod = await import(path.join(tmpDir, "sf_common.mjs"));

    const cropPastes = [];
    const inpaintPastes = [];
    const crop = { comfyClass: "SFImageCrop", inputs: [{ name: "image", link: null }], _sfCropPaste: () => {} };
    const inpaint = { comfyClass: "SFInpaintCrop", inputs: [{ name: "image", link: null }], _sfInpaintPaste: () => {} };
    graphNodes.push(crop, inpaint);

    mod.installPasteHandler({
        comfyClass: "SFImageCrop", hook: "_sfCropPaste",
        onPasteImage: (n, d) => cropPastes.push(n.comfyClass),
    });
    mod.installPasteHandler({
        comfyClass: "SFInpaintCrop", hook: "_sfInpaintPaste",
        onPasteImage: (n, d) => inpaintPastes.push(n.comfyClass),
    });
    check("两个类各注册一个监听器", pasteListeners.length === 2);

    // 同键重复安装幂等
    mod.installPasteHandler({
        comfyClass: "SFImageCrop", hook: "_sfCropPaste",
        onPasteImage: () => {},
    });
    check("同键重复安装不新增监听器", pasteListeners.length === 2);

    const firePaste = (selected) => {
        app.canvas.selected_nodes = selected ? { [selected.id || "x"]: selected } : {};
        const e = {
            target: {},
            clipboardData: { items: [{ type: "image/png", getAsFile: () => ({}) }] },
            preventDefault() {}, stopImmediatePropagation() {},
        };
        for (const fn of pasteListeners) fn(e);
    };

    firePaste(crop);
    firePaste(inpaint);
    await new Promise((r) => setTimeout(r, 60));   // 等 sweep setTimeout
    check("粘贴到 crop 节点只触发 crop 处理", cropPastes.length === 1 && cropPastes[0] === "SFImageCrop");
    check("粘贴到 inpaint 节点只触发 inpaint 处理", inpaintPastes.length === 1 && inpaintPastes[0] === "SFInpaintCrop");

    firePaste(null);
    check("无选中节点不触发任何处理", cropPastes.length === 1 && inpaintPastes.length === 1);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
