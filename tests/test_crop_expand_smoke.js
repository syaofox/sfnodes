// sf_crop_expand.js 主扩展冒烟测试（Node 直接运行）
// 覆盖拖拽释放粘鼠标修复：
//   1. onMouseMove 在主键已松（buttons:0）但 _sfExpandDrag 残留时立即落定，
//      绝不再改框（primaryButtonReleased 守卫）；
//   2. installNodeReleaseGuard 走 window capture 注册释放监听（复用 sf_common）；
//   3. onMouseUp / onMouseDown(非左键) 语义。
// 加载方式：主扩展 import 改写为同目录桩（app/CropAPI/sf_popup/image_browser），
// sf_common 剥 app/api import 作真实实现（与 test_common_paste_js.js 同法），
// 纯库 sf_crop_expand_lib.js 拷真实实现。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

// ── 全局桩 ──
const winListeners = [];
globalThis.window = {
  addEventListener(type, fn, capture) { winListeners.push({ type, fn, capture: !!capture }); },
  removeEventListener(type, fn, capture) {
    const i = winListeners.findIndex(
      (l) => l.type === type && l.fn === fn && l.capture === !!capture);
    if (i >= 0) winListeners.splice(i, 1);
  },
};
globalThis.document = {
  getElementById: () => null,
  createElement: () => ({ style: {}, set textContent(_) {}, appendChild() {} }),
  head: { appendChild() {} },
  body: {},
  addEventListener() {},
  removeEventListener() {},
};
globalThis.app = {
  graph: { _nodes: [], setDirtyCanvas() {} },
  canvas: {},
  graphToPrompt: async () => ({ output: {} }),
  registerExtension(ext) { globalThis.__ceExt = ext; },
};
globalThis.api = { apiURL: (r) => r };

const fireWin = (type) => {
  for (const l of winListeners.filter((l) => l.type === type)) l.fn({ type });
};

(async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_crop_expand_smoke_"));
  const webDir = path.join(__dirname, "..", "web");

  // 桩模块
  fs.writeFileSync(path.join(tmpDir, "stub_app.js"),
    "export const app = globalThis.app;\n");
  fs.writeFileSync(path.join(tmpDir, "stub_core.js"),
    "export const CropAPI = { uploadSrc: async () => ({}) };\n");
  fs.writeFileSync(path.join(tmpDir, "stub_popup.js"),
    "export function attachPopupDismiss() {}\n");
  fs.writeFileSync(path.join(tmpDir, "stub_browser.js"),
    "export function showImageBrowser() {}\n");
  // 真实 sf_common：剥 import，走 globalThis
  let common = fs.readFileSync(path.join(webDir, "sf_common.js"), "utf8")
    .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
    .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;");
  fs.writeFileSync(path.join(tmpDir, "sf_common.js"), common);
  // 纯库真实实现
  fs.copyFileSync(path.join(webDir, "sf_crop_expand_lib.js"), path.join(tmpDir, "sf_crop_expand_lib.js"));

  // 主扩展：改写 import 指向桩
  let code = fs.readFileSync(path.join(webDir, "sf_crop_expand.js"), "utf8");
  code = code
    .replace('from "/scripts/app.js"', 'from "./stub_app.js"')
    .replace('from "./sf_crop_core.js"', 'from "./stub_core.js"')
    .replace('from "./sf_popup.js"', 'from "./stub_popup.js"')
    .replace('from "./image_browser.js"', 'from "./stub_browser.js"');
  check("import 改写无残留", !code.includes("/scripts/app.js") && !code.includes("sf_crop_core.js"));
  fs.writeFileSync(path.join(tmpDir, "sf_crop_expand.js"), code);
  await import(path.join(tmpDir, "sf_crop_expand.js"));

  const ext = globalThis.__ceExt;
  check("扩展注册名 sfnodes.CropExpand", ext && ext.name === "sfnodes.CropExpand");

  const nodeType = { prototype: {} };
  await ext.beforeRegisterNodeDef(nodeType, { name: "SFImageCropExpand" });
  const node = Object.create(nodeType.prototype);
  node.properties = {
    sfCropExpandState: JSON.stringify({
      src_path: "", src_w: 512, src_h: 512,
      crop_x: 0, crop_y: 0, crop_w: 512, crop_h: 512,
      fill_color: "#000000", aspect_ratio: "free", custom_w: 1, custom_h: 1,
    }),
  };
  node.size = [320, 300];
  node.comfyClass = "SFImageCropExpand";
  node.type = "SFImageCropExpand";
  node.flags = {};
  nodeType.prototype.onNodeCreated.call(node);

  check("按钮 13 项（竖列 11 + 底行 2）", node._sfExpandButtons && node._sfExpandButtons.length === 13);
  check("释放兜底 hook 已装", !!node._sfExpandReleaseGuard);
  const rel = winListeners.filter((l) => l.type === "mouseup");
  check("window capture mouseup 已注册", rel.length === 1 && rel[0].capture === true);
  check("blur 兜底已注册", winListeners.some((l) => l.type === "blur" && l.capture === true));

  const state = () => JSON.parse(node.properties.sfCropExpandState);

  // 起拖：点击裁剪框左上角手柄（显示坐标系 512×512 区域 offset≈[50,42]）
  const started = node.onMouseDown({ button: 0, buttons: 1 }, [50, 42]);
  check("左键命中手柄起拖", started === true && !!node._sfExpandDrag);

  // 按住拖动：改框
  const before = state();
  node.onMouseMove({ buttons: 1 }, [70, 62]);
  const after = state();
  check("按住拖动改变裁剪框", after.crop_w !== before.crop_w || after.crop_h !== before.crop_h);

  // 释放丢失：buttons:0 的移动必须立即落定，不再改框
  const held = state();
  node.onMouseMove({ buttons: 0 }, [90, 82]);
  check("buttons:0 清空拖拽状态", node._sfExpandDrag == null);
  const finalized = state();
  check("buttons:0 不再改框（仅取整落定）",
    Math.abs(finalized.crop_w - held.crop_w) <= 1 && Math.abs(finalized.crop_h - held.crop_h) <= 1);
  // 落定后再移动（即使 buttons:1）也不会自发改框
  node.onMouseMove({ buttons: 1 }, [120, 110]);
  const after2 = state();
  check("落定后再移动不改框", after2.crop_x === finalized.crop_x && after2.crop_w === finalized.crop_w);

  // 非左键不起拖
  const rightDown = node.onMouseDown({ button: 2, buttons: 2 }, [50, 42]);
  check("右键不起拖", rightDown === false && node._sfExpandDrag == null);

  // onMouseUp 正常落定（点框内部 → move 手柄，前次拖拽后仍在框内）
  node.onMouseDown({ button: 0, buttons: 1 }, [150, 140]);
  check("再次起拖", !!node._sfExpandDrag);
  node.onMouseUp({}, [], { canvas: { style: {} } });
  check("onMouseUp 落定", node._sfExpandDrag == null);

  // window capture 释放监听也能落定残留拖拽
  node.onMouseDown({ button: 0, buttons: 1 }, [150, 140]);
  check("释放监听用例已起拖", !!node._sfExpandDrag);
  fireWin("pointerup");
  check("window 释放监听落定", node._sfExpandDrag == null);

  // onRemoved 解绑
  nodeType.prototype.onRemoved.call(node);
  check("onRemoved 解绑释放监听", node._sfExpandReleaseGuard === null
    && winListeners.filter((l) => l.type === "mouseup").length === 0);

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})().catch((e) => { console.error("SMOKE ERROR:", e); process.exit(1); });
