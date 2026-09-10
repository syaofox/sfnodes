// sf_brush_mask.js 主扩展冒烟测试（Node 直接运行：node tests/test_brush_mask_smoke.js）
// 覆盖 §45.8 回归：底信息行背景必须画在底行按钮（Load/Browse）之前——
// 后画盖先画，半透明底栏若盖住按钮会导致按钮发虚 + 边框错层残影。
// 手段：FakeCtx 记录全部绘制 op（含调用时 fillStyle），FakeNode 跑真实
// onDrawForeground 一帧，断言底栏 fill 先于底行按钮 fill。
// 模块加载：主扩展的 import 改写为同目录桩（app/CropAPI/sf_common/
// image_browser），纯库拷真实 sf_brush_mask_lib.js。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

// ── 全局桩（模块顶层即用）──
globalThis.LiteGraph = { NODE_TEXT_COLOR: "#ffffff" };
globalThis.window = { addEventListener() {} };
globalThis.document = { addEventListener() {}, removeEventListener() {}, body: {} };
globalThis.Image = class {};

// ── FakeCtx：记录 op + 调用时 fillStyle ──
function makeCtx(ops) {
  const state = {};
  return new Proxy({ measureText: () => ({ width: 0 }) }, {
    get(t, p) {
      if (p in t) return t[p];
      return (...a) => { ops.push({ op: p, args: a, fill: state.fillStyle }); };
    },
    set(t, p, v) { state[p] = v; return true; },
  });
}

(async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_brush_mask_smoke_"));
  const webDir = path.join(__dirname, "..", "web");

  // 桩模块
  fs.writeFileSync(path.join(tmpDir, "stub_app.js"),
    `export const app = { graph: null, canvas: null, graphToPrompt: async function () { return {}; }, registerExtension(ext) { globalThis.__bmExt = ext; } };\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_core.js"),
    `export const CropAPI = { uploadSrc: async () => ({}) };\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_common.js"),
    `export const sfToast = () => {}; export const buildSourceURL = () => null; export const getSfAccent = () => null; export const installPasteHandler = () => {}; export const parseAnnotatedImageValue = () => null;\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_browser.js"),
    `export function showImageBrowser() {}\n`);
  // 纯库用真实实现
  fs.copyFileSync(path.join(webDir, "sf_brush_mask_lib.js"), path.join(tmpDir, "sf_brush_mask_lib.js"));

  // 主扩展：改写 import 指向桩
  let code = fs.readFileSync(path.join(webDir, "sf_brush_mask.js"), "utf8");
  code = code
    .replace('from "/scripts/app.js"', 'from "./stub_app.js"')
    .replace('from "./sf_crop_core.js"', 'from "./stub_core.js"')
    .replaceAll('from "./sf_common.js"', 'from "./stub_common.js"')
    .replace('from "./image_browser.js"', 'from "./stub_browser.js"');
  check("import 改写无残留", !code.includes("/scripts/app.js") && !code.includes("sf_crop_core.js"));
  fs.writeFileSync(path.join(tmpDir, "sf_brush_mask.js"), code);
  await import(path.join(tmpDir, "sf_brush_mask.js"));

  const ext = globalThis.__bmExt;
  check("扩展注册名 sfnodes.BrushMask", ext && ext.name === "sfnodes.BrushMask");

  // FakeNodeType + FakeNode 跑 onNodeCreated
  const nodeType = { prototype: {} };
  await ext.beforeRegisterNodeDef(nodeType, { name: "SFImageBrushMask" });
  const node = Object.create(nodeType.prototype);
  node.properties = {
    sfBrushMaskState: JSON.stringify({
      src_path: "", src_w: 880, src_h: 1184, brush_size: 88, strokes: [],
      brush_opacity: 0.5, brush_color: "255,255,255", eraser_color: "255,50,50", brush_mode: "brush",
    }),
  };
  node.size = [420, 320];
  node.flags = {};
  nodeType.prototype.onNodeCreated.call(node);
  check("控件 12 项（竖列 10 + 底行 2）", node._sfBrushCtrls && node._sfBrushCtrls.length === 12);

  // 跑一帧真实绘制
  const ops = [];
  node.onDrawForeground(makeCtx(ops));
  check("有绘制输出", ops.length > 0);

  // 底栏背景：roundRect(高 29 = 21+8) + fill（路径填充，非 fillRect）；
  // 底行按钮 fill：rgba(60,60,60,0.7) + y≈289
  const barRR = ops.map((o, i) => ({ o, i }))
    .filter(({ o }) => o.op === "roundRect" && o.args[3] === 29);
  check("底栏背景 roundRect 被绘制", barRR.length === 1);
  // 底栏 fill = 该 roundRect 之后紧跟的 fill（竖列底条共用同色，不可用色定位）
  let barIdx = -1;
  if (barRR.length === 1) {
    const f = ops.map((o, i) => ({ o, i }))
      .find(({ o, i }) => i > barRR[0].i && o.op === "fill");
    if (f && f.o.fill === "rgba(40,40,40,0.9)") barIdx = f.i;
  }
  check("底栏背景 fill 被绘制", barIdx >= 0);
  const bottomBtns = ops.filter((o) => o.op === "fillRect" && o.fill === "rgba(60,60,60,0.7)" && o.args[1] >= 285 && o.args[1] <= 295);
  check("底行两按钮被绘制", bottomBtns.length === 2);
  const btnIdx = ops.map((o, i) => ({ o, i }))
    .filter(({ o }) => o.op === "fillRect" && o.fill === "rgba(60,60,60,0.7)" && o.args[1] >= 285 && o.args[1] <= 295)
    .map(({ i }) => i);
  check("底行按钮画在底栏背景之后（§45.8）", btnIdx.length === 2 && btnIdx.every((i) => i > barIdx));

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})().catch((e) => { console.error("SMOKE ERROR:", e); process.exit(1); });
