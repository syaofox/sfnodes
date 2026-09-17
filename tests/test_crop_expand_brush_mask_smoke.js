// sf_crop_expand_brush_mask.js 主扩展冒烟测试（Node 直接运行：
// node tests/test_crop_expand_brush_mask_smoke.js）
// 覆盖（两节点合体的关键路径）：
//   1. 双列控件几何（11 + 10 + 2）与三模式单选切换
//   2. Crop 模式：手柄命中起拖 + 冻结快照改框 + buttons:0 释放兜底 + 落定后
//      不再改框 + 右键不起拖
//   3. Brush 模式：源图区内落笔/扩展区（源图外）不起笔/手柄不误触框
//   4. 笔触预览离屏合成（真擦除 destination-out，复用 paintStrokeMask）
//   5. lean 注入：graphToPrompt 注入隐藏输入且不含预览字段
//   6. [ ] 快捷键（真实 sf_brush_tools）+ computeSize 最小值 + onRemoved 解绑
// 加载方式：桩 app/api/CropAPI/browser/popup；真实共享模块（sf_common 剥 import、
// 纯库与 sf_crop_source/sf_crop_expand_ratios/sf_brush_tools/sf_pause_kit）。
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
    const i = winListeners.findIndex((l) => l.type === type && l.fn === fn && l.capture === !!capture);
    if (i >= 0) winListeners.splice(i, 1);
  },
};
globalThis.LiteGraph = { NODE_TEXT_COLOR: "#ffffff" };
globalThis.__settingVals = {};
globalThis.__settingDefs = {};
globalThis.__graph = { _nodes: [], setDirtyCanvas() {} };
globalThis.__canvas = { style: {}, node_over: null, selected_nodes: null, pointer: null, graph_mouse: null };
globalThis.__dirtyCount = 0;
globalThis.__graph.setDirtyCanvas = () => { globalThis.__dirtyCount++; };

const createdCanvases = [];
function makeFullCtx(ops) {
  const st = {};
  return new Proxy({}, {
    get(t, p) {
      if (p === "measureText") return () => ({ width: 0 });
      return (...a) => { ops.push({ op: p, args: a, fill: st.fillStyle, stroke: st.strokeStyle, gco: st.globalCompositeOperation }); };
    },
    set(t, p, v) { st[p] = v; ops.push({ op: "set:" + p, value: v }); return true; },
  });
}
globalThis.document = {
  getElementById: () => null,
  querySelector: () => null,
  addEventListener() {},
  removeEventListener() {},
  head: { appendChild() {} },
  body: { appendChild() {} },
  createElement(tag) {
    if (tag !== "canvas") return { style: {}, appendChild() {}, querySelector() { return null; }, click() {} };
    const ops = [];
    const c = { width: 0, height: 0, ops, getContext: () => makeFullCtx(ops) };
    createdCanvases.push(c);
    return c;
  },
};
globalThis.app = {
  get graph() { return globalThis.__graph; },
  get canvas() { return globalThis.__canvas; },
  ui: { settings: {
    getSettingValue(id) { return globalThis.__settingVals[id]; },
    addSetting(def) { globalThis.__settingDefs[def.id] = def; },
  } },
  graphToPrompt: async () => globalThis.__promptResult || { output: {} },
  registerExtension(ext) { globalThis.__cebmExt = ext; },
};
globalThis.api = { addEventListener() {}, fetchApi: async () => ({ ok: true, json: async () => ({}) }) };

const fireWin = (type, ev) => {
  for (const l of winListeners.filter((l) => l.type === type)) l.fn(ev || { type });
};

const CLASS = "SFImageCropExpandBrushMask";
const STATE_PROP = "sfCropExpandBrushMaskState";
const makeState = (patch = {}) => JSON.stringify({
  src_path: "", src_w: 512, src_h: 512,
  crop_x: 0, crop_y: 0, crop_w: 512, crop_h: 512,
  fill_color: "#000000", aspect_ratio: "free", custom_w: 1, custom_h: 1,
  brush_size: 80, strokes: [], brush_opacity: 0.5,
  brush_color: "255,255,255", brush_mode: "crop",
  sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  ...patch,
});

(async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_cebm_smoke_"));
  const webDir = path.join(__dirname, "..", "web");

  // 桩模块
  fs.writeFileSync(path.join(tmpDir, "stub_app.js"), "export const app = globalThis.app;\n");
  fs.writeFileSync(path.join(tmpDir, "stub_api.js"),
    "export const api = { addEventListener() {}, fetchApi: async () => ({ ok: true, json: async () => ({}) }) };\n");
  fs.writeFileSync(path.join(tmpDir, "stub_core.js"),
    "export const CropAPI = { uploadSrc: async () => ({}) };\n");
  fs.writeFileSync(path.join(tmpDir, "stub_browser.js"), "export function showImageBrowser() {}\n");
  fs.writeFileSync(path.join(tmpDir, "stub_popup.js"), "export function attachPopupDismiss() {}\n");
  // 真实 sf_common：剥 import，走 globalThis
  let common = fs.readFileSync(path.join(webDir, "sf_common.js"), "utf8")
    .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
    .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;");
  fs.writeFileSync(path.join(tmpDir, "sf_common.js"), common);

  // 真实纯库 + 共享模块（改写 import 指向桩/真实）
  for (const [srcFile, rules] of [
    ["sf_crop_expand_lib.js", []],
    ["sf_brush_mask_lib.js", []],
    ["sf_crop_expand_brush_mask_lib.js", []],
    ["sf_canvas_align_lib.js", []],
    ["sf_pause_text_lib.js", []],
    ["sf_crop_source.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
      ['from "./sf_crop_core.js"', 'from "./stub_core.js"'],
      ['from "./image_browser.js"', 'from "./stub_browser.js"'],
    ]],
    ["sf_crop_expand_ratios.js", [
      ['from "./sf_popup.js"', 'from "./stub_popup.js"'],
    ]],
    ["sf_brush_tools.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
    ]],
    ["sf_pause_kit.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
      ['from "/scripts/api.js"', 'from "./stub_api.js"'],
    ]],
    ["sf_brush_sam.js", [
      ['from "/scripts/api.js"', 'from "./stub_api.js"'],
    ]],
  ]) {
    let mod = fs.readFileSync(path.join(webDir, srcFile), "utf8");
    for (const [from, to] of rules) mod = mod.replaceAll(from, to);
    fs.writeFileSync(path.join(tmpDir, srcFile), mod);
  }

  // 主扩展：仅改写绝对 app import
  let code = fs.readFileSync(path.join(webDir, "sf_crop_expand_brush_mask.js"), "utf8");
  code = code.replace('from "/scripts/app.js"', 'from "./stub_app.js"');
  check("import 改写无残留", !code.includes("/scripts/app.js"));
  fs.writeFileSync(path.join(tmpDir, "sf_crop_expand_brush_mask.js"), code);
  await import(path.join(tmpDir, "sf_crop_expand_brush_mask.js"));

  const ext = globalThis.__cebmExt;
  check("扩展注册名 sfnodes.CropExpandBrushMask", ext && ext.name === "sfnodes.CropExpandBrushMask");

  // FakeNode 跑 onNodeCreated
  const nodeType = { prototype: {} };
  await ext.beforeRegisterNodeDef(nodeType, { name: CLASS });
  const node = Object.create(nodeType.prototype);
  node.id = 1;
  node.properties = { [STATE_PROP]: makeState() };
  node.size = [360, 300];
  node.comfyClass = CLASS;
  node.type = CLASS;
  node.flags = {};
  node.pos = [0, 0];
  globalThis.__graph._nodes.push(node);
  nodeType.prototype.onNodeCreated.call(node);

  check("控件 23 项（列1 11 + 列2 10 + 底行 2）", node._sfCEBCtrls && node._sfCEBCtrls.length === 23);
  check("释放兜底 hook 已装", !!node._sfCEBReleaseGuard);
  check("computeSize 钳最小值", JSON.stringify(nodeType.prototype.computeSize.call(node)) === JSON.stringify([360, 300]));

  // 右键菜单（sf_brush_sam 共享安装器）
  const menuOpts = [];
  node.getExtraMenuOptions({}, menuOpts);
  check("菜单有 SAM 蒙版项", menuOpts.some((o) => o.content.includes("SAM 蒙版")));
  check("菜单有卸载 SAM 项", menuOpts.some((o) => o.content.includes("卸载 SAM")));
  check("菜单仅两项", menuOpts.length === 2);
  const st0 = JSON.parse(node.properties[STATE_PROP]);
  check("SAM 记忆字段默认齐全", st0.sam_prompt === "" && st0.sam_threshold === 0.5 && st0.sam_refine === 2);

  const gc = () => ({ canvas: { style: {} }, setDirty() {} });
  const state = () => JSON.parse(node.properties[STATE_PROP]);

  // ── Crop 模式：手柄起拖（显示区 offsetX=90, offsetY=42, scale=190/512）──
  const cropBtn = node._sfCEBCtrls.find((b) => b.id === "crop");
  check("列2 Crop 按钮几何", cropBtn.x === 50 && cropBtn.y === 16);
  check("默认 Crop 模式", state().brush_mode === "crop");

  const started = node.onMouseDown({ button: 0, buttons: 1 }, [90, 42]); // NW 手柄
  check("左键命中手柄起拖", started === true && !!node._sfCEBDrag);
  const before = state();
  node.onMouseMove({ buttons: 1 }, [110, 62], gc());
  const after = state();
  check("按住拖动改变裁剪框", after.crop_x > before.crop_x && after.crop_w < before.crop_w);
  // 释放丢失：buttons:0 立即落定，不再改框
  const held = state();
  node.onMouseMove({ buttons: 0 }, [130, 82], gc());
  check("buttons:0 清空拖拽状态", node._sfCEBDrag == null);
  const finalized = state();
  check("buttons:0 仅取整落定", Math.abs(finalized.crop_x - held.crop_x) <= 1);
  node.onMouseMove({ buttons: 1 }, [160, 110], gc());
  check("落定后再移动不改框", state().crop_x === finalized.crop_x);
  // 右键不起拖
  check("右键不起拖", node.onMouseDown({ button: 2, buttons: 2 }, [90, 42]) === false);

  // ── 三模式切换（列2 按钮：Brush = TOOL_COL[1] → y=38, x=50..80）──
  node.onMouseDown({ button: 0, buttons: 1 }, [65, 47]);
  check("点击 Brush 切模式", state().brush_mode === "brush");
  node.onMouseDown({ button: 0, buttons: 1 }, [65, 69]); // Erase（TOOL_COL[2] y=60）
  check("点击 Erase 切模式", state().brush_mode === "erase");
  node.onMouseDown({ button: 0, buttons: 1 }, [65, 25]); // Crop（TOOL_COL[0] y=16）
  check("点击 Crop 切回", state().brush_mode === "crop");

  // ── Brush 模式：源图区内落笔 / 扩展区不起笔 ──
  node.onMouseDown({ button: 0, buttons: 1 }, [65, 47]); // Brush
  const s1 = node.onMouseDown({ button: 0, buttons: 1 }, [150, 100]);
  check("源图区内起笔", s1 === true && node._sfCEBDrawing === true);
  node.onMouseMove({ buttons: 1 }, [170, 120], { canvas: {} });
  check("拖动追加笔触点", node._sfCEBCur.length >= 2);
  node.onMouseUp({}, [], gc());
  const st2 = state();
  check("落定写入笔触（brush）", st2.strokes.length === 1 && st2.strokes[0].mode === "brush" && st2.strokes[0].points.length >= 2);
  check("笔触坐标钳制在源图内", st2.strokes[0].points.every(([x, y]) => x >= 0 && y >= 0 && x <= 511 && y <= 511));

  // 扩展区（源图外）不起笔：裁剪框外扩后显示区含扩展区
  node.properties[STATE_PROP] = makeState({ crop_x: -100, crop_y: -100, crop_w: 712, crop_h: 712, brush_mode: "brush" });
  // scale = 190/712；img(-50,-50) → local(90+50*0.2669, 42+50*0.2669) ≈ (103.3, 55.3)
  const s3 = node.onMouseDown({ button: 0, buttons: 1 }, [103, 55]);
  check("扩展区不起笔", s3 === false && !node._sfCEBDrawing);
  // 源图内（img 10,10 → local 90+110*0.2669=119.4, 42+110*0.2669=71.4）可起笔
  const s4 = node.onMouseDown({ button: 0, buttons: 1 }, [119, 71]);
  check("扩展框内源图区仍可起笔", s4 === true && node._sfCEBDrawing === true);
  node.onMouseUp({}, [], gc());

  // ── 笔触预览：离屏合成真擦除 ──
  node.properties[STATE_PROP] = makeState({
    brush_mode: "brush", brush_size: 20, brush_color: "255,0,0",
    strokes: [
      { mode: "brush", size: 20, points: [[10, 10], [30, 30]] },
      { mode: "erase", size: 20, points: [[15, 15], [25, 25]] },
    ],
  });
  createdCanvases.length = 0;
  node.onDrawForeground(makeFullCtx([]));
  const maskCvs = createdCanvases[createdCanvases.length - 1];
  check("预览离屏画布尺寸 = 源图", maskCvs && maskCvs.width === 512 && maskCvs.height === 512);
  const gcos = maskCvs ? maskCvs.ops.filter((o) => o.op === "set:globalCompositeOperation").map((o) => o.value) : [];
  check("预览含 destination-out 擦除", gcos.includes("destination-out") && gcos[gcos.length - 1] === "source-over");
  check("预览用笔刷颜色绘制", maskCvs && maskCvs.ops.some((o) => o.op === "stroke" && String(o.stroke).includes("255,0,0")));

  // ── lean 注入（graphToPrompt 补丁）──
  globalThis.__promptResult = { output: { "1": { class_type: CLASS, inputs: {} } } };
  node.properties[STATE_PROP] = makeState({
    src_path: "sfnodes_crop/cebm_src.png",
    crop_x: -10, crop_y: 0, crop_w: 600, crop_h: 512,
    fill_color: "#112233", brush_size: 30,
    strokes: [{ mode: "brush", size: 30, points: [[1, 2]] }],
    brush_opacity: 0.9, brush_color: "1,2,3", brush_mode: "erase", aspect_ratio: "16:9",
  });
  const prompted = await globalThis.app.graphToPrompt();
  const payloadRaw = prompted.output["1"].inputs.SFCropExpandBrushMaskJson;
  const payload = JSON.parse(payloadRaw);
  check("注入隐藏输入", !!payloadRaw && payload.crop_x === -10 && payload.fill_color === "#112233");
  check("注入含笔触", JSON.stringify(payload.strokes) === JSON.stringify([{ mode: "brush", size: 30, points: [[1, 2]] }]));
  check("注入排除预览字段", payload.brush_color === undefined && payload.brush_opacity === undefined
    && payload.brush_mode === undefined && payload.aspect_ratio === undefined);

  // ── [ ] 快捷键（真实 sf_brush_tools）──
  globalThis.__canvas.selected_nodes = { 1: node };
  node.properties[STATE_PROP] = makeState({ brush_size: 80 });
  const fireKey = (key, extra = {}) => {
    for (const l of winListeners.filter((l) => l.type === "keydown")) {
      l.fn({ key, timeStamp: Date.now() + Math.random(), ctrlKey: false, metaKey: false, altKey: false, preventDefault() {}, stopPropagation() {}, ...extra });
    }
  };
  fireKey("]");
  check("] 放大笔刷", state().brush_size === 82);
  fireKey("[");
  check("[ 缩小笔刷", state().brush_size === 80);
  fireKey("]", { target: { tagName: "INPUT" } });
  check("输入框内不响应", state().brush_size === 80);
  check("设置项已注册（init）", (ext.init(), globalThis.__settingDefs["sfnodes.BrushMask.SizeStep"]?.defaultValue === 2));

  // ── window capture 释放兜底 + onRemoved 解绑 ──
  node.properties[STATE_PROP] = makeState({ brush_mode: "brush" });
  node.onMouseDown({ button: 0, buttons: 1 }, [150, 100]);
  check("释放监听用例已起笔", node._sfCEBDrawing === true);
  fireWin("pointerup");
  check("window 释放监听落定", node._sfCEBDrawing === false && state().strokes.length === 1);
  nodeType.prototype.onRemoved.call(node);
  check("onRemoved 解绑释放监听", node._sfCEBReleaseGuard === null
    && winListeners.filter((l) => l.type === "mouseup").length === 0);

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})().catch((e) => { console.error("SMOKE ERROR:", e); process.exit(1); });
