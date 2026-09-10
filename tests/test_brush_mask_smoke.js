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
globalThis.__bmCanvas = { node_over: null }; // 画布桩（import 时被 stub_app 引用，可变）
globalThis.document = { addEventListener() {}, removeEventListener() {}, body: {} };
// 离屏画布工厂（遮罩合成用）：每画布独立 op 流，调用与属性赋值全记录
const createdCanvases = [];
function makeFullCtx(ops) {
  const st = {};
  return new Proxy({}, {
    get(t, p) {
      return (...a) => { ops.push({ op: p, args: a, fill: st.fillStyle, gco: st.globalCompositeOperation }); };
    },
    set(t, p, v) { st[p] = v; ops.push({ op: "set:" + p, value: v }); return true; },
  });
}
globalThis.document.createElement = (tag) => {
  if (tag !== "canvas") return { style: {}, appendChild() {}, querySelector() { return null; } };
  const ops = [];
  const c = { width: 0, height: 0, ops, getContext: () => makeFullCtx(ops) };
  createdCanvases.push(c);
  return c;
};
 
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
    `export const app = { graph: null, canvas: globalThis.__bmCanvas || null, graphToPrompt: async function () { return {}; }, registerExtension(ext) { globalThis.__bmExt = ext; } };\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_core.js"),
    `export const CropAPI = { uploadSrc: async () => ({}) };\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_common.js"),
    `export const sfToast = () => {}; export const buildSourceURL = () => "http://fake/view.png"; export const getSfAccent = () => null; export const installPasteHandler = () => {}; export const parseAnnotatedImageValue = () => null; export const sfApiUrl = (p) => p;\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_browser.js"),
    `export function showImageBrowser() {}\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_api.js"),
    `export const api = { fetchApi: async () => ({ ok: true, json: async () => ({}) }) };\n`);
  // 纯库用真实实现
  fs.copyFileSync(path.join(webDir, "sf_brush_mask_lib.js"), path.join(tmpDir, "sf_brush_mask_lib.js"));

  // 主扩展：改写 import 指向桩
  let code = fs.readFileSync(path.join(webDir, "sf_brush_mask.js"), "utf8");
  code = code
    .replace('from "/scripts/app.js"', 'from "./stub_app.js"')
    .replace('from "/scripts/api.js"', 'from "./stub_api.js"')
    .replace('from "./sf_crop_core.js"', 'from "./stub_core.js"')
    .replaceAll('from "./sf_common.js"', 'from "./stub_common.js"')
    .replace('from "./image_browser.js"', 'from "./stub_browser.js"');
  check("import 改写无残留", !code.includes("/scripts/app.js") && !code.includes("/scripts/api.js") && !code.includes("sf_crop_core.js"));
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
  check("控件 11 项（竖列 9 + 底行 2）", node._sfBrushCtrls && node._sfBrushCtrls.length === 11);

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

  const opts = [];
  node.getExtraMenuOptions({}, opts);
  check("菜单有 SAM 蒙版项", opts.some((o) => o.content.includes("SAM 蒙版")));
  check("菜单有卸载 SAM 项", opts.some((o) => o.content.includes("卸载 SAM")));
  check("菜单仅两项（无覆盖层清除项）", opts.length === 2);

  // fill 笔触绘制：整体填充闭合多边形（SAM 并入，§45.9）
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "", src_w: 100, src_h: 100, brush_size: 80, strokes: [
      { mode: "fill", size: 0, points: [[10, 10], [50, 10], [50, 50], [10, 50]] },
    ],
    brush_opacity: 0.5, brush_color: "255,255,255", eraser_color: "255,50,50", brush_mode: "brush",
    sam_prompt: "person", sam_threshold: 0.5,
  });
  const ops2 = [];
  node.onDrawForeground(makeCtx(ops2));
  // fill 画在离屏遮罩画布上（不透明白），不透明度在贴回时统一施加
  const fillCvs = createdCanvases[createdCanvases.length - 1];
  const fills = fillCvs ? fillCvs.ops.filter((o) => o.op === "fill") : [];
  check("fill 笔触触发填充", fills.length >= 1);
  check("fill 用画笔色不透明", fills.some((o) => o.fill === "rgba(255,255,255,1)"));

  // 真擦除预览（§45.9）：离屏按画序合成，erase 走 destination-out；
  // 主画布无红色、无 destination-out，只有一次贴回 blit
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "", src_w: 100, src_h: 100, brush_size: 20, strokes: [
      { mode: "brush", size: 20, points: [[10, 10], [30, 30]] },
      { mode: "erase", size: 20, points: [[15, 15], [25, 25]] },
    ],
    brush_opacity: 0.5, brush_color: "255,255,255", brush_mode: "brush",
    sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  });
  createdCanvases.length = 0;
  node._sfMaskCvs = null; // 清缓存，断言按尺寸重建
  const ops3 = [];
  node.onDrawForeground(makeCtx(ops3));
  check("离屏遮罩画布按源图尺寸创建", createdCanvases.length === 1
    && createdCanvases[0].width === 100 && createdCanvases[0].height === 100);
  const mops = createdCanvases[0].ops;
  const gcos = mops.filter((o) => o.op === "set:globalCompositeOperation").map((o) => o.value);
  check("erase 走 destination-out", gcos.includes("destination-out"));
  check("打洞后恢复 source-over", gcos[gcos.length - 1] === "source-over");
  check("主画布无红色叠加", !ops3.some((o) => o.fill && String(o.fill).startsWith("rgba(255,50,50")));
  check("合成一次贴回", ops3.filter((o) => o.op === "drawImage").length === 1);

  // 纯 brush 时不出现 destination-out
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "", src_w: 100, src_h: 100, brush_size: 20, strokes: [
      { mode: "brush", size: 20, points: [[10, 10], [30, 30]] },
    ],
    brush_opacity: 0.5, brush_color: "255,255,255", brush_mode: "brush",
    sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  });
  createdCanvases.length = 0;
  node._sfMaskCvs = null;
  node.onDrawForeground(makeCtx([]));
  check("纯 brush 无打洞", !createdCanvases[0].ops.some((o) => o.value === "destination-out"));

  // 笔刷光环：悬停图片区记录位置并画环（半径 = size/2×scale）；
  // 非悬停（node_over 他人）不画
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "", src_w: 100, src_h: 100, brush_size: 80, strokes: [],
    brush_opacity: 0.5, brush_color: "255,255,255", brush_mode: "brush",
    sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  });
  node._sfBrushCursor = null;
  node.onMouseMove({}, [100, 100]);
  check("悬停图片区记录光标", JSON.stringify(node._sfBrushCursor) === JSON.stringify([100, 100]));
  node.onMouseMove({}, [5, 5]);
  check("图片区外清空光标", node._sfBrushCursor === null);
  node.onMouseMove({}, [100, 100]);
  globalThis.__bmCanvas.node_over = node;
  const ops4 = [];
  node.onDrawForeground(makeCtx(ops4));
  // scale = min(290/100, 274/100) = 2.74，环半径 = 40×2.74 = 109.6
  const arcs = ops4.filter((o) => o.op === "arc");
  check("悬停时画光环", arcs.some((o) => Math.abs(o.args[2] - 109.6) < 1e-9));
  globalThis.__bmCanvas.node_over = {};
  const ops5 = [];
  node.onDrawForeground(makeCtx(ops5));
  check("非悬停不画光环", !ops5.some((o) => o.op === "arc"));

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})().catch((e) => { console.error("SMOKE ERROR:", e); process.exit(1); });
