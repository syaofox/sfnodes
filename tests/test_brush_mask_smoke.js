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
globalThis.__bmKeys = {}; // window 监听捕获（type -> [fn]）
globalThis.window = {
  addEventListener(type, fn) { (globalThis.__bmKeys[type] ??= []).push(fn); },
  removeEventListener() {},
};
globalThis.__bmCanvas = { node_over: null }; // 画布桩（import 时被 stub_app 引用，可变）
globalThis.__bmGraph = null; // 图桩（快捷键用例按需注入）
globalThis.document = { addEventListener() {}, removeEventListener() {}, body: {} };
globalThis.__bmEditorOpen = false; // 为 true 时 querySelector(".sf-px-overlay") 命中（编辑器互斥用例）
globalThis.document.querySelector = (sel) => (
  sel === ".sf-px-overlay" && globalThis.__bmEditorOpen ? {} : null);
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
      return (...a) => { ops.push({ op: p, args: a, fill: state.fillStyle, lw: state.lineWidth }); };
    },
    set(t, p, v) { state[p] = v; return true; },
  });
}

(async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_brush_mask_smoke_"));
  const webDir = path.join(__dirname, "..", "web");

  // 桩模块
  fs.writeFileSync(path.join(tmpDir, "stub_app.js"),
    `export const app = { get graph() { return globalThis.__bmGraph || null; }, canvas: globalThis.__bmCanvas || null, ui: { settings: { getSettingValue(id) { return globalThis.__bmSettingVals[id]; }, addSetting(def) { globalThis.__bmSettingDefs[def.id] = def; } } }, graphToPrompt: async function () { return {}; }, registerExtension(ext) { globalThis.__bmExt = ext; } };\n`);
  globalThis.__bmSettingVals = {};
  globalThis.__bmSettingDefs = {};
  fs.writeFileSync(path.join(tmpDir, "stub_core.js"),
    `export const CropAPI = { uploadSrc: async () => ({}) };\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_common.js"),
    `export const sfToast = () => {}; export const buildSourceURL = () => "http://fake/view.png"; export const getSfAccent = () => null; export const installPasteHandler = () => {}; export const parseAnnotatedImageValue = () => null; export const sfApiUrl = (p) => p; export const primaryButtonReleased = (e) => !!(e && typeof e.buttons === "number" && (e.buttons & 1) === 0); export const installNodeReleaseGuard = () => {}; export const removeNodeReleaseGuard = () => {}; export const installResizeCornerCursor = () => {}; export const pickColorInput = () => {}; export const rgbStringToHex = () => "#ffffff"; export const hexToRgbString = () => "255,255,255"; export const applyAdaptiveCanvasOnly = () => {}; export const injectCSSOnce = () => {}; export function sfFrameWidth() { const v = Number(globalThis.__bmSettingVals["sfnodes.Canvas.FrameWidth"]); return Number.isFinite(v) && v > 0 ? v : 1; } export function sfFrameThin() { return Math.max(0.5, sfFrameWidth() * 0.5); } export function sfCursorWidth() { const v = Number(globalThis.__bmSettingVals["sfnodes.Canvas.CursorWidth"]); return Number.isFinite(v) && v > 0 ? v : 1; } export function sfPolyVertexSize() { const v = Number(globalThis.__bmSettingVals["sfnodes.Canvas.PolyVertexSize"]); return Number.isFinite(v) && v >= 1 ? Math.min(20, v) : 2; } export function registerSfLineWidthSettings() { globalThis.__bmSettingDefs["sfnodes.Canvas.FrameWidth"] = { id: "sfnodes.Canvas.FrameWidth", defaultValue: 1.0, type: "slider", attrs: { min: 0.5, max: 3, step: 0.25 } }; globalThis.__bmSettingDefs["sfnodes.Canvas.CursorWidth"] = { id: "sfnodes.Canvas.CursorWidth", defaultValue: 1.0, type: "slider", attrs: { min: 0.5, max: 3, step: 0.25 } }; }\n`);
  // 共享模块用真实实现（源图链路 / 画笔工具），仅改写其 import 指向桩
  for (const [srcFile, dstFile, rules] of [
    ["sf_crop_source.js", "sf_crop_source.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
      ['from "./sf_crop_core.js"', 'from "./stub_core.js"'],
      ['from "./sf_common.js"', 'from "./stub_common.js"'],
      ['from "./image_browser.js"', 'from "./stub_browser.js"'],
    ]],
    ["sf_brush_tools.js", "sf_brush_tools.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
    ]],
    ["sf_pause_kit.js", "sf_pause_kit.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
      ['from "/scripts/api.js"', 'from "./stub_api.js"'],
      ['from "./sf_common.js"', 'from "./stub_common.js"'],
    ]],
    ["sf_pause_text_lib.js", "sf_pause_text_lib.js", []],
    ["sf_brush_ai.js", "sf_brush_ai.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
      ['from "/scripts/api.js"', 'from "./stub_api.js"'],
      ['from "./sf_common.js"', 'from "./stub_common.js"'],
      ['from "./sf_crop_core.js"', 'from "./stub_core.js"'],
    ]],
    ["sf_brush_poly.js", "sf_brush_poly.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
      ['from "./sf_common.js"', 'from "./stub_common.js"'],
    ]],
  ]) {
    let mod = fs.readFileSync(path.join(webDir, srcFile), "utf8");
    for (const [from, to] of rules) mod = mod.replaceAll(from, to);
    fs.writeFileSync(path.join(tmpDir, dstFile), mod);
  }
  fs.writeFileSync(path.join(tmpDir, "stub_browser.js"),
    `export function showImageBrowser() {}\n`);
  fs.writeFileSync(path.join(tmpDir, "stub_api.js"),
    `export const api = { fetchApi: async (url, opts) => { (globalThis.__apiCalls ??= []).push({ url, body: opts && opts.body }); return { ok: true, json: async () => (globalThis.__aiResponse || {}) }; } };\n`);
  // 纯库用真实实现
  fs.copyFileSync(path.join(webDir, "sf_brush_mask_lib.js"), path.join(tmpDir, "sf_brush_mask_lib.js"));
  fs.copyFileSync(path.join(webDir, "sf_canvas_align_lib.js"), path.join(tmpDir, "sf_canvas_align_lib.js"));

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
  check("控件 13 项（竖列 11 + 底行 2）", node._sfBrushCtrls && node._sfBrushCtrls.length === 13);

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
  const menuText = opts.map((o) => o.content).join("|");
  check("菜单 8 项", opts.length === 8);
  check("菜单有 SAM 文本项", menuText.includes("SAM 蒙版"));
  check("菜单有点选/框选", menuText.includes("点选") && menuText.includes("框选"));
  check("菜单有人物部位/YOLO/导入", menuText.includes("人物部位") && menuText.includes("YOLO") && menuText.includes("导入遮罩"));
  check("菜单有反选/卸载 AI", menuText.includes("反选") && menuText.includes("卸载 AI"));

  // 反选 toggle（菜单回调 → 状态位 + lean 注入）
  opts.find((o) => o.content.includes("反选")).callback();
  check("反选开启（状态位）", JSON.parse(node.properties.sfBrushMaskState).invert === true);
  // 面板反选按钮：再点一次关闭 + ON 状态色绘制
  const invBtn = node._sfBrushCtrls.find((b) => b.id === "invert");
  check("反选按钮存在（竖列）", !!invBtn && invBtn.isInvert === true);
  node.onMouseDown({ button: 0, buttons: 1 }, [invBtn.x + 15, invBtn.y + 9]);
  check("点按钮关闭反选", JSON.parse(node.properties.sfBrushMaskState).invert === false);
  node.onMouseDown({ button: 0, buttons: 1 }, [invBtn.x + 15, invBtn.y + 9]);
  const invOps = [];
  node.onDrawForeground(makeCtx(invOps));
  check("反选按钮 ON 用状态色", invOps.some((o) => o.op === "fillRect" && o.fill === "rgba(196,124,34,0.95)"));
  check("菜单项均带 callback", opts.every((o) => typeof o.callback === "function"));

  // SAM 点选模式：进入 → 正/负点 → Enter 执行（POST 载荷含坐标）
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "sfnodes_crop/x.png", src_w: 100, src_h: 100, brush_size: 80,
    strokes: [], brush_opacity: 0.5, brush_color: "255,255,255", brush_mode: "brush",
    sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  });
  globalThis.__apiCalls = [];
  opts.find((o) => o.content.includes("点选")).callback();
  check("进入点选模式", !!node._sfAiSam && node._sfAiSam.kind === "point");
  const modePanelClick = node.onMouseDown({ button: 0, buttons: 1 }, [25, 25]); // Brush 模式按钮
  check("点选模式仍可点控件列", modePanelClick === true && node._sfAiSam.pos.length === 0);
  node.onMouseDown({ button: 0, buttons: 1 }, [150, 100]);
  node.onMouseDown({ button: 0, buttons: 1, shiftKey: true }, [170, 110]);
  check("点选记录正/负点（Shift 负点）", node._sfAiSam.pos.length === 1 && node._sfAiSam.neg.length === 1);
  // 覆盖层绘制：点/提示必须画出来且不抛错（曾缺 imageToLocal import → 整帧中断）
  const samOps = [];
  let samDrawErr = null;
  try { node.onDrawForeground(makeCtx(samOps)); } catch (e) { samDrawErr = e; }
  check("点选覆盖层绘制不抛错", samDrawErr === null);
  check("点选画在图上（arc ×2）", samOps.filter((o) => o.op === "arc").length >= 2);
  check("点选提示条文本", samOps.some((o) => o.op === "fillText" && String(o.args[0]).includes("Shift")))
  for (const fn of (globalThis.__bmKeys.keydown || [])) {
    fn({ key: "Enter", timeStamp: Date.now() + Math.random(), ctrlKey: false, metaKey: false, altKey: false, preventDefault() {}, stopPropagation() {} });
  }
  check("Enter 退出模式", node._sfAiSam == null);
  await new Promise((r) => setTimeout(r, 20));
  const postCall = (globalThis.__apiCalls || []).find((c) => c.url.includes("/brush_mask/sam") && c.body);
  const postBody = postCall ? JSON.parse(postCall.body) : null;
  check("点选 POST 含正/负点", !!postBody && JSON.parse(postBody.positive_coords).length === 1
    && JSON.parse(postBody.negative_coords).length === 1);

  // YOLO 请求载荷：imgsz / classes 透传（直接调共享模块 runYolo）
  const aiMod = await import(path.join(tmpDir, "sf_brush_ai.js"));
  const ycfg = {
    toastTag: "SF Brush Mask", logTag: "[SF Brush Mask]",
    getState: (n) => JSON.parse(n.properties.sfBrushMaskState),
    patchState: () => {},
    addStrokes: () => {},
  };
  globalThis.__apiCalls = [];
  await aiMod.runYolo(ycfg, node, "bbox", "a.pt", 0.3, "rect", 960, [1, 2]);
  const yoloCall = (globalThis.__apiCalls || []).find((c) => c.url.includes("/brush_mask/yolo") && c.body);
  const yoloBody = yoloCall ? JSON.parse(yoloCall.body) : null;
  check("YOLO POST 含 imgsz/classes", !!yoloBody && yoloBody.imgsz === 960
    && JSON.stringify(yoloBody.classes) === "[1,2]" && yoloBody.kind === "bbox");

  // 模式即运算：Brush 模式识别结果保持 fill；Eraser 模式改写为 fill_erase
  const aiResp = { strokes: [{ mode: "fill", size: 0, points: [[0, 0], [2, 0], [2, 2]] }], coverage: 0.5 };
  const captureCfg = (captured) => ({
    toastTag: "SF Brush Mask", logTag: "[SF Brush Mask]",
    getState: (n) => JSON.parse(n.properties.sfBrushMaskState),
    patchState: () => {},
    addStrokes: (_n, inc) => { captured.push(...inc); },
  });
  globalThis.__aiResponse = aiResp;
  let aiCaptured = [];
  await aiMod.runYolo(captureCfg(aiCaptured), node, "bbox", "a.pt", 0.3, "rect", 640, []);
  check("Brush 模式识别结果保持 fill", aiCaptured.length === 1 && aiCaptured[0].mode === "fill");
  node.properties.sfBrushMaskState = JSON.stringify({
    ...JSON.parse(node.properties.sfBrushMaskState), brush_mode: "erase",
  });
  aiCaptured = [];
  await aiMod.runYolo(captureCfg(aiCaptured), node, "bbox", "a.pt", 0.3, "rect", 640, []);
  check("Eraser 模式识别结果转 fill_erase", aiCaptured.length === 1 && aiCaptured[0].mode === "fill_erase"
    && aiCaptured[0].points.length === 3);
  globalThis.__aiResponse = null;

  // 反选预览：白底打洞（paintInvertMask 在离屏画布 destination-out）
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "", src_w: 100, src_h: 100, brush_size: 80, invert: true,
    strokes: [{ mode: "brush", size: 20, points: [[10, 10]] }],
    brush_opacity: 0.5, brush_color: "255,255,255", brush_mode: "brush",
    sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  });
  const beforeInv = createdCanvases.length;
  node.onDrawForeground(makeCtx([]));
  const newCvs = createdCanvases.slice(beforeInv);
  check("反选新建离屏画布并打洞", newCvs.length >= 1
    && newCvs.some((c) => c.ops.some((o) => o.op === "set:globalCompositeOperation" && o.value === "destination-out")));

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
  const fillCvs = node._sfMaskCvs; // 离屏遮罩画布（避免被后续新建画布干扰索引）
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
  check("光环默认线宽 1（sfnodes.Canvas.CursorWidth）", arcs.every((o) => o.lw === 1));
  globalThis.__bmSettingVals["sfnodes.Canvas.CursorWidth"] = 2.5;
  const opsCursor = [];
  node.onDrawForeground(makeCtx(opsCursor));
  check("光环线宽随设置（2.5）", opsCursor.filter((o) => o.op === "arc").every((o) => o.lw === 2.5));
  delete globalThis.__bmSettingVals["sfnodes.Canvas.CursorWidth"];
  globalThis.__bmCanvas.node_over = {};
  const ops5 = [];
  node.onDrawForeground(makeCtx(ops5));
  check("非悬停不画光环", !ops5.some((o) => o.op === "arc"));

  // ── 多边形套索（§101）：多次点选闭合填充（Brush=fill / Erase=fill_erase）──
  const polyState = () => JSON.parse(node.properties.sfBrushMaskState);
  const polyBtn = node._sfBrushCtrls.find((b) => b.id === "poly");
  check("Poly 按钮存在（竖列状态位）", !!polyBtn && polyBtn.isPoly === true);
  const clickPoly = () => node.onMouseDown({ button: 0, buttons: 1 }, [polyBtn.x + 15, polyBtn.y + 9]);
  let __tsPoly = 900000;
  const firePolyKey = (key) => {
    for (const fn of globalThis.__bmKeys.keydown || []) {
      fn({ key, timeStamp: __tsPoly++, ctrlKey: false, metaKey: false, altKey: false, preventDefault() {}, stopPropagation() {} });
    }
  };
  clickPoly();
  check("Poly 开启（状态位）", polyState().brush_poly === true);
  // ON 状态色：按钮随开关变色（绿 POLY_ON_COLOR，区别模式强调色/INVERT 琥珀）
  const polyOnOps = [];
  node.onDrawForeground(makeCtx(polyOnOps));
  check("Poly ON 用状态色（绿）", polyOnOps.some((o) => o.op === "fillRect" && o.fill === "rgba(46,160,67,0.95)"));
  // 控件命中优先：Poly 模式下仍可切模式按钮
  node.onMouseDown({ button: 0, buttons: 1 }, [25, 17]);
  check("Poly 模式可切回 Brush", polyState().brush_mode === "brush" && polyState().brush_poly === true);
  // 显示区三次落点（scale=2.74, offset=(58,10)：img(10,10)→(85,37) 等）
  node.onMouseDown({ button: 0, buttons: 1 }, [85, 37]);
  node.onMouseDown({ button: 0, buttons: 1 }, [222, 37]);
  node.onMouseDown({ button: 0, buttons: 1 }, [222, 174]);
  check("三点会话收集顶点", !!node._sfBrushPoly && node._sfBrushPoly.points.length === 3);
  firePolyKey("Backspace");
  check("Backspace 删末点", !!node._sfBrushPoly && node._sfBrushPoly.points.length === 2);
  firePolyKey("Enter");
  check("不足 3 点 Enter 不闭合", !!node._sfBrushPoly && node._sfBrushPoly.points.length === 2
    && polyState().strokes.length === 0);
  node.onMouseDown({ button: 0, buttons: 1 }, [222, 174]); // 补回第三点
  node.onMouseDown({ button: 0, buttons: 1 }, [85, 37]);   // 点首点闭合
  const fillState = polyState();
  check("点首点闭合写入 fill 笔触", fillState.strokes.length === 1 && fillState.strokes[0].mode === "fill"
    && fillState.strokes[0].points.length === 3 && fillState.strokes[0].size === 0);
  check("闭合后清会话（开关保留）", node._sfBrushPoly == null && fillState.brush_poly === true);
  // Erase 模式 + 双击闭合 → fill_erase 打洞
  const eraseBtn = node._sfBrushCtrls.find((b) => b.id === "erase");
  node.onMouseDown({ button: 0, buttons: 1 }, [eraseBtn.x + 15, eraseBtn.y + 9]);
  node.onMouseDown({ button: 0, buttons: 1 }, [85, 37]);
  node.onMouseDown({ button: 0, buttons: 1 }, [222, 37]);
  node.onMouseDown({ button: 0, buttons: 1 }, [222, 174]);
  node.onDblClick();
  const eraseState = polyState();
  check("双击闭合 Erase 模式写入 fill_erase", eraseState.strokes.length === 2
    && eraseState.strokes[1].mode === "fill_erase");
  // 覆盖层：提示条 + 首顶点圆点（不抛错）
  node.onMouseDown({ button: 0, buttons: 1 }, [85, 37]);
  node.onMouseDown({ button: 0, buttons: 1 }, [222, 37]);
  const polyOps = [];
  let polyDrawErr = null;
  try { node.onDrawForeground(makeCtx(polyOps)); } catch (e) { polyDrawErr = e; }
  check("套索覆盖层绘制不抛错", polyDrawErr === null);
  check("套索覆盖层画提示条", polyOps.some((o) => o.op === "fillText" && String(o.args[0]).includes("Poly:")));
  check("套索覆盖层画顶点/首点圆", polyOps.some((o) => o.op === "arc"));
  // 顶点标记尺寸走设置 sfnodes.Canvas.PolyVertexSize（默认 2=2×2 方块，原内联 4）
  check("套索顶点默认小尺寸（2×2 方块）", polyOps.some((o) => o.op === "fillRect" && o.args[2] === 2 && o.args[3] === 2));
  globalThis.__bmSettingVals["sfnodes.Canvas.PolyVertexSize"] = 5;
  const polyOpsBig = [];
  node.onDrawForeground(makeCtx(polyOpsBig));
  check("套索顶点尺寸随设置（5×5）", polyOpsBig.some((o) => o.op === "fillRect" && o.args[2] === 5 && o.args[3] === 5));
  delete globalThis.__bmSettingVals["sfnodes.Canvas.PolyVertexSize"];
  // Esc：有会话先取消会话，再 Esc 关闭工具并解绑键盘
  firePolyKey("Escape");
  check("Esc 取消会话（开关保留）", node._sfBrushPoly == null && polyState().brush_poly === true);
  firePolyKey("Escape");
  check("再 Esc 关闭 Poly", polyState().brush_poly === false);
  // 按钮关闭：重开后关闭丢弃未闭合会话
  clickPoly();
  node.onMouseDown({ button: 0, buttons: 1 }, [85, 37]);
  clickPoly();
  check("按钮关闭丢弃会话", node._sfBrushPoly == null && polyState().brush_poly === false);

  // 笔刷 shortcut [ ]：选中本类节点才生效
  const sizeOf = () => JSON.parse(node.properties.sfBrushMaskState).brush_size;
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "", src_w: 100, src_h: 100, brush_size: 80, strokes: [],
    brush_opacity: 0.5, brush_color: "255,255,255", brush_mode: "brush",
    sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  });
  const decoy = { comfyClass: "Other", type: "Other", properties: {}, size: [100, 100], pos: [0, 0], flags: {} };
  node.comfyClass = "SFImageBrushMask";
  node.type = "SFImageBrushMask";
  globalThis.__bmGraph = { _nodes: [node, decoy], setDirtyCanvas() {} };
  globalThis.__bmCanvas.selected_nodes = { 12: node, 13: decoy };
  let __ts = 1000;
  const fireKey = (key, extra = {}) => {
    for (const fn of globalThis.__bmKeys.keydown || []) {
      fn({ key, timeStamp: __ts++, ctrlKey: false, metaKey: false, altKey: false, preventDefault() {}, stopPropagation() {}, ...extra });
    }
  };
  fireKey("]");
  check("] 放大", sizeOf() === 82);
  fireKey("[");
  check("[ 缩小", sizeOf() === 80);
  fireKey("a");
  check("无关键不动", sizeOf() === 80);
  fireKey("]", { target: { tagName: "INPUT" } });
  check("输入框内不动", sizeOf() === 80);
  fireKey("]", { ctrlKey: true });
  check("修饰键不动", sizeOf() === 80);
  globalThis.__bmEditorOpen = true;
  fireKey("]");
  check("编辑器打开时不动", sizeOf() === 80);
  globalThis.__bmEditorOpen = false;
  globalThis.__bmCanvas.selected_nodes = {};
  fireKey("]");
  check("未选中不动", sizeOf() === 80);

  // ── 模式快捷键（B 笔刷 / E 擦除；大小写不敏感）──
  const modeOf = () => JSON.parse(node.properties.sfBrushMaskState).brush_mode;
  globalThis.__bmCanvas.selected_nodes = { 12: node };
  fireKey("e");
  check("E 切 Eraser", modeOf() === "erase");
  fireKey("B");
  check("B 切 Brush（大写）", modeOf() === "brush");
  fireKey("e", { target: { tagName: "INPUT" } });
  check("输入框内不改模式", modeOf() === "brush");
  fireKey("e", { ctrlKey: true });
  check("修饰键不改模式", modeOf() === "brush");
  globalThis.__bmEditorOpen = true;
  fireKey("e");
  check("编辑器打开时不改模式", modeOf() === "brush");
  globalThis.__bmEditorOpen = false;
  globalThis.__bmCanvas.selected_nodes = {};
  fireKey("e");
  check("未选中不改模式", modeOf() === "brush");

  // 设置页步长：init 注册 + S+ 按钮读取自定义步长
  globalThis.__bmExt.init();
  check("注册 SizeStep 设置项", globalThis.__bmSettingDefs["sfnodes.BrushMask.SizeStep"]?.defaultValue === 2);
  check("注册 OpacityStep 设置项", globalThis.__bmSettingDefs["sfnodes.BrushMask.OpacityStep"]?.defaultValue === 5);
  const sizePlusBtn = node._sfBrushCtrls.find((b) => b.id === "sizePlus");
  const clickSizePlus = () => node.onMouseDown({}, [sizePlusBtn.x + 15, sizePlusBtn.y + 9]);
  clickSizePlus(); // S+ 按钮中心（按控件几何解析，勿写死行号）
  check("默认步长 +2", sizeOf() === 82);
  globalThis.__bmSettingVals["sfnodes.BrushMask.SizeStep"] = 5;
  clickSizePlus();
  check("自定义步长 +5", sizeOf() === 87);
  globalThis.__bmSettingVals["sfnodes.BrushMask.SizeStep"] = 999;
  clickSizePlus();
  check("非法设置回退默认", sizeOf() === 89);
  // 快捷键必须同样走设置（三路统一入口，§45.12 复修）
  globalThis.__bmSettingVals["sfnodes.BrushMask.SizeStep"] = 7;
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "", src_w: 100, src_h: 100, brush_size: 80, strokes: [],
    brush_opacity: 0.5, brush_color: "255,255,255", brush_mode: "brush",
    sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  });
  globalThis.__bmCanvas.selected_nodes = { 12: node };
  fireKey("]");
  check("快捷键走自定义步长", sizeOf() === 87);

  // 官方通道 onKeyDown（画布 processKey 分发）+ 同物理按键去重
  node.properties.sfBrushMaskState = JSON.stringify({
    src_path: "", src_w: 100, src_h: 100, brush_size: 80, strokes: [],
    brush_opacity: 0.5, brush_color: "255,255,255", brush_mode: "brush",
    sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  });
  globalThis.__bmSettingVals["sfnodes.BrushMask.SizeStep"] = 2;
  globalThis.__bmCanvas.selected_nodes = { 12: node };
  const ev1 = { key: "]", timeStamp: 5000, ctrlKey: false, metaKey: false, altKey: false, preventDefault() {}, stopPropagation() {} };
  check("onKeyDown 放大", node.onKeyDown(ev1) === false && sizeOf() === 82);
  node.onKeyDown(ev1);
  check("同戳去重（官方通道内）", sizeOf() === 82);
  for (const fn of globalThis.__bmKeys.keydown || []) fn(ev1);
  check("同戳去重（跨通道）", sizeOf() === 82);
  const ev2 = { key: "[", timeStamp: 5001, ctrlKey: false, metaKey: false, altKey: false, preventDefault() {}, stopPropagation() {} };
  node.onKeyDown(ev2);
  check("onKeyDown 缩小", sizeOf() === 80);
  check("非快捷键放行", node.onKeyDown({ key: "a", timeStamp: 5002 }) === undefined);

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})().catch((e) => { console.error("SMOKE ERROR:", e); process.exit(1); });
