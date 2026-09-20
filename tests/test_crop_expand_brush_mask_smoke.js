// sf_crop_expand_brush_mask.js 主扩展冒烟测试（Node 直接运行：
// node tests/test_crop_expand_brush_mask_smoke.js）
// 覆盖（两节点合体的关键路径）：
//   1. 双列控件几何（11 + 11 + 2）与三模式单选切换
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

// Image 桩：src 赋值即视为已解码（applyOrientation 上传后替换预览 <img>；
// onload 在 onload 已挂载的前提下同步触发）
globalThis.Image = class {
  constructor() { this.onload = null; this.complete = false; this.naturalWidth = 0; this.naturalHeight = 0; }
  set src(v) { this._src = v; this.complete = true; if (this.onload) this.onload(); }
  get src() { return this._src; }
};

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
    const c = {
      width: 0, height: 0, ops,
      getContext: () => makeFullCtx(ops),
      toDataURL(type) { ops.push({ op: "toDataURL", args: [type] }); return "data:image/png;base64,stub"; },
    };
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
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const CLASS = "SFImageCropExpandBrushMask";
const STATE_PROP = "sfCropExpandBrushMaskState";
const makeState = (patch = {}) => JSON.stringify({
  src_path: "", src_w: 512, src_h: 512,
  crop_x: 0, crop_y: 0, crop_w: 512, crop_h: 512,
  fill_color: "#000000", aspect_ratio: "free", custom_w: 1, custom_h: 1,
  brush_size: 80, strokes: [], brush_opacity: 0.5,
  brush_color: "255,255,255", brush_mode: "crop",
  sam_prompt: "", sam_threshold: 0.5, sam_refine: 2,
  invert: false, person_parts: [], person_confidence: 0.4, person_refine: false,
  yolo_kind: "bbox", yolo_model: "", yolo_conf: 0.25, yolo_box_shape: "rect",
  yolo_imgsz: 640, yolo_classes: [],
  ...patch,
});

(async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_cebm_smoke_"));
  const webDir = path.join(__dirname, "..", "web");

  // 桩模块
  fs.writeFileSync(path.join(tmpDir, "stub_app.js"), "export const app = globalThis.app;\n");
  fs.writeFileSync(path.join(tmpDir, "stub_api.js"),
    "export const api = { addEventListener() {}, fetchApi: async (url, opts) => { (globalThis.__apiCalls ??= []).push({ url, body: opts && opts.body }); return { ok: true, json: async () => (globalThis.__aiResponse || {}) }; } };\n");
  fs.writeFileSync(path.join(tmpDir, "stub_core.js"),
    "export const CropAPI = { uploadSrc: async (id, dataURL) => { (globalThis.__uploadCalls ??= []).push({ id, dataURL }); return { path: 'sfnodes_crop/crop_src_' + id + '.png' }; } };\n");
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
    ["sf_brush_ai.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
      ['from "/scripts/api.js"', 'from "./stub_api.js"'],
      ['from "./sf_crop_core.js"', 'from "./stub_core.js"'],
    ]],
    ["sf_brush_poly.js", [
      ['from "/scripts/app.js"', 'from "./stub_app.js"'],
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

  check("控件 29 项（列1 11 + 列2 12 + 列3 4 + 底行 2）", node._sfCEBCtrls && node._sfCEBCtrls.length === 29);
  check("释放兜底 hook 已装", !!node._sfCEBReleaseGuard);
  check("computeSize 钳最小值（三列 + 列2 12 项 → 400×340）", JSON.stringify(nodeType.prototype.computeSize.call(node)) === JSON.stringify([400, 340]));

  // ── 列头/分组/重命名（§109·§112）──
  const hdrOps = [];
  node.onDrawForeground(makeFullCtx(hdrOps));
  const drawnTexts = hdrOps.filter((o) => o.op === "fillText").map((o) => String(o.args[0]));
  check("列头 RATIO/TOOLS/ORIENT 绘制", drawnTexts.includes("RATIO") && drawnTexts.includes("TOOLS") && drawnTexts.includes("ORIENT"));
  check("歧义按钮重命名 Fill/Pen", drawnTexts.includes("Fill") && drawnTexts.includes("Pen"));
  check("首行下移列头 10px（crop y=26）", node._sfCEBCtrls.find((b) => b.id === "crop").y === 26);
  check("列3 按钮几何（ORIENT_COL_X=90 起 4 项）",
    ["flipH", "flipV", "rotL", "rotR"].every((id, i) => {
      const b = node._sfCEBCtrls.find((c) => c.id === id);
      return b && b.x === 90 && b.y === 26 + i * 22 && b.isOrient === true;
    }));

  // 右键菜单（sf_brush_ai 共享安装器）
  const menuOpts = [];
  node.getExtraMenuOptions({}, menuOpts);
  const menuText = menuOpts.map((o) => o.content).join("|");
  check("菜单 8 项", menuOpts.length === 8);
  check("菜单有 SAM 文本/点选/框选", menuText.includes("SAM 蒙版") && menuText.includes("点选") && menuText.includes("框选"));
  check("菜单有人物部位/YOLO/导入", menuText.includes("人物部位") && menuText.includes("YOLO") && menuText.includes("导入遮罩"));
  check("菜单有反选/卸载 AI", menuText.includes("反选") && menuText.includes("卸载 AI"));
  const st0 = JSON.parse(node.properties[STATE_PROP]);
  check("菜单参数记忆字段默认齐全", st0.sam_prompt === "" && st0.sam_threshold === 0.5 && st0.sam_refine === 2
    && st0.invert === false && st0.yolo_kind === "bbox");

  // 反选 toggle（菜单回调 → 状态位；合体节点语义 = 扩展区 ∪ (1 - 笔触)）
  menuOpts.find((o) => o.content.includes("反选")).callback();
  check("反选开启（状态位）", JSON.parse(node.properties[STATE_PROP]).invert === true);
  menuOpts.find((o) => o.content.includes("反选")).callback();
  check("反选关闭（再点一次）", JSON.parse(node.properties[STATE_PROP]).invert === false);
  // 面板反选按钮（列2）：点击 toggle + ON 状态色绘制
  const invBtn2 = node._sfCEBCtrls.find((b) => b.id === "invert");
  check("反选按钮存在（列2）", !!invBtn2 && invBtn2.isInvert === true);
  node.onMouseDown({ button: 0, buttons: 1 }, [invBtn2.x + 15, invBtn2.y + 9]);
  check("点按钮开启反选", JSON.parse(node.properties[STATE_PROP]).invert === true);
  const invOps2 = [];
  node.onDrawForeground(makeFullCtx(invOps2));
  check("反选按钮 ON 用状态色", invOps2.some((o) => o.op === "fillRect" && o.fill === "rgba(196,124,34,0.95)"));
  node.onMouseDown({ button: 0, buttons: 1 }, [invBtn2.x + 15, invBtn2.y + 9]);
  check("再点按钮关闭反选", JSON.parse(node.properties[STATE_PROP]).invert === false);


  const gc = () => ({ canvas: { style: {} }, setDirty() {} });
  const state = () => JSON.parse(node.properties[STATE_PROP]);

  // 控件坐标一律按几何解析（勿写死行号/坐标——§101 教训）
  const ctrl = (id) => node._sfCEBCtrls.find((b) => b.id === id);
  const clickCtrl = (id) => {
    const b = ctrl(id);
    return node.onMouseDown({ button: 0, buttons: 1 }, [b.x + 15, b.y + 9]);
  };

  // 悬停按钮 → 底行改显中文说明（§109）
  {
    const opaBtn = ctrl("opaMinus");
    node.onMouseMove({ buttons: 0 }, [opaBtn.x + 15, opaBtn.y + 9], gc());
    check("悬停记录控件", node._sfCEBHover === opaBtn);
    globalThis.__canvas.node_over = node;
    const hintOps = [];
    node.onDrawForeground(makeFullCtx(hintOps));
    check("悬停显示按钮中文说明", hintOps.some((o) => o.op === "fillText" && String(o.args[0]).includes("透明度")));
    globalThis.__canvas.node_over = null;
  }

  // ── Crop 模式：手柄起拖（显示区 offsetX=130, offsetY=62, scale=190/512）──
  const cropBtn = ctrl("crop");
  check("列2 Crop 按钮几何", cropBtn.x === 50 && cropBtn.y === 26);
  check("默认 Crop 模式", state().brush_mode === "crop");

  const started = node.onMouseDown({ button: 0, buttons: 1 }, [130, 62]); // NW 手柄
  check("左键命中手柄起拖", started === true && !!node._sfCEBDrag);
  const before = state();
  node.onMouseMove({ buttons: 1 }, [150, 82], gc());
  const after = state();
  check("按住拖动改变裁剪框", after.crop_x > before.crop_x && after.crop_w < before.crop_w);
  // 释放丢失：buttons:0 立即落定，不再改框
  const held = state();
  node.onMouseMove({ buttons: 0 }, [170, 102], gc());
  check("buttons:0 清空拖拽状态", node._sfCEBDrag == null);
  const finalized = state();
  check("buttons:0 仅取整落定", Math.abs(finalized.crop_x - held.crop_x) <= 1);
  node.onMouseMove({ buttons: 1 }, [200, 130], gc());
  check("落定后再移动不改框", state().crop_x === finalized.crop_x);
  // 右键不起拖
  check("右键不起拖", node.onMouseDown({ button: 2, buttons: 2 }, [130, 62]) === false);

  // ── 三模式切换（列2 按钮；坐标按控件几何解析）──
  clickCtrl("brush");
  check("点击 Brush 切模式", state().brush_mode === "brush");
  clickCtrl("erase");
  check("点击 Erase 切模式", state().brush_mode === "erase");
  clickCtrl("crop");
  check("点击 Crop 切回", state().brush_mode === "crop");

  // ── Brush 模式：源图区内落笔 / 扩展区不起笔 ──
  clickCtrl("brush");
  const s1 = node.onMouseDown({ button: 0, buttons: 1 }, [150, 110]);
  check("源图区内起笔", s1 === true && node._sfCEBDrawing === true);
  node.onMouseMove({ buttons: 1 }, [170, 120], { canvas: {} });
  check("拖动追加笔触点", node._sfCEBCur.length >= 2);
  // 落笔拖动时光环跟随（曾只在悬停分支更新 → 圆环停在起笔前位置）
  check("落笔拖动时光环跟随鼠标", JSON.stringify(node._sfCEBCursor) === JSON.stringify([170, 120]));
  node.onMouseUp({}, [], gc());
  const st2 = state();
  check("落定写入笔触（brush）", st2.strokes.length === 1 && st2.strokes[0].mode === "brush" && st2.strokes[0].points.length >= 2);
  check("笔触坐标钳制在源图内", st2.strokes[0].points.every(([x, y]) => x >= 0 && y >= 0 && x <= 511 && y <= 511));

  // 扩展区（源图外）不起笔：裁剪框外扩后显示区含扩展区
  node.properties[STATE_PROP] = makeState({ crop_x: -100, crop_y: -100, crop_w: 712, crop_h: 712, brush_mode: "brush" });
  // scale = 190/712；img(-50,-50) → local(130+50*0.2669, 62+50*0.2669) ≈ (143.3, 75.3)
  const s3 = node.onMouseDown({ button: 0, buttons: 1 }, [143, 75]);
  check("扩展区不起笔", s3 === false && !node._sfCEBDrawing);
  // 源图内（img 10,10 → local 130+110*0.2669=159.4, 62+110*0.2669=91.4）可起笔
  const s4 = node.onMouseDown({ button: 0, buttons: 1 }, [159, 91]);
  check("扩展框内源图区仍可起笔", s4 === true && node._sfCEBDrawing === true);
  node.onMouseUp({}, [], gc());

  // ── 多边形套索（§101）：Crop 下点击 Poly 自动切 Brush；落点限源图内 ──
  node.properties[STATE_PROP] = makeState({ brush_mode: "crop" });
  const polyBtn = node._sfCEBCtrls.find((b) => b.id === "poly");
  check("列2 Poly 按钮存在（状态位）", !!polyBtn && polyBtn.isPoly === true);
  node.onMouseDown({ button: 0, buttons: 1 }, [polyBtn.x + 15, polyBtn.y + 9]);
  check("Crop 下开启 Poly 自动切 Brush", state().brush_mode === "brush" && state().brush_poly === true);
  const polyOnOps = [];
  node.onDrawForeground(makeFullCtx(polyOnOps));
  check("Poly ON 用状态色（绿）", polyOnOps.some((o) => o.op === "fillRect" && o.fill === "rgba(46,160,67,0.95)"));
  // 三次落点（scale=190/512=0.3711, offset=(130,62)）：img(50,50)→(149,81) 等
  node.onMouseDown({ button: 0, buttons: 1 }, [149, 81]);
  node.onMouseDown({ button: 0, buttons: 1 }, [186, 81]);
  node.onMouseDown({ button: 0, buttons: 1 }, [186, 118]);
  check("套索三点会话", !!node._sfBrushPoly && node._sfBrushPoly.points.length === 3);
  const polyOpsC = [];
  node.onDrawForeground(makeFullCtx(polyOpsC));
  check("套索顶点默认小尺寸（2×2 方块）", polyOpsC.some((o) => o.op === "fillRect" && o.args[2] === 2 && o.args[3] === 2));
  node.onDblClick();
  check("双击闭合写入 fill 笔触", state().strokes.length === 1 && state().strokes[0].mode === "fill"
    && state().strokes[0].points.length === 3);
  // 扩展区不落点（裁剪框外扩后显示区含扩展区）
  node.properties[STATE_PROP] = makeState({ crop_x: -100, crop_y: -100, crop_w: 712, crop_h: 712, brush_mode: "brush", brush_poly: true });
  const extHit = node.onMouseDown({ button: 0, buttons: 1 }, [143, 75]);
  check("套索在扩展区不落点", extHit === false && node._sfBrushPoly == null);
  const srcHit = node.onMouseDown({ button: 0, buttons: 1 }, [159, 91]); // img(10,10)
  check("套索在源图内落点", srcHit === true && !!node._sfBrushPoly && node._sfBrushPoly.points.length === 1);
  // 按钮关闭：丢弃未闭合会话
  node.onMouseDown({ button: 0, buttons: 1 }, [polyBtn.x + 15, polyBtn.y + 9]);
  check("关闭 Poly 丢弃会话", node._sfBrushPoly == null && state().brush_poly === false);

  // ── 翻转/旋转（列3 ORIENT，§112）：canvas 重绘 + 上传新源图 + 状态联动重映射 ──
  node.properties[STATE_PROP] = makeState({
    src_path: "sfnodes_crop/crop_src_cebm_src.png",
    src_w: 512, src_h: 300,
    crop_x: 10, crop_y: 20, crop_w: 100, crop_h: 200,
    strokes: [{ mode: "brush", size: 20, points: [[0, 0], [511, 299]] }],
    aspect_ratio: "16:9", brush_mode: "brush",
  });
  globalThis.__uploadCalls = [];
  node._sfCEBImg = { complete: true, naturalWidth: 512, naturalHeight: 300 };
  createdCanvases.length = 0;
  clickCtrl("rotR");
  await sleep(20);
  const stR = state();
  const orientCvs = createdCanvases[createdCanvases.length - 1];
  check("rotR canvas 宽高交换（300×512）", orientCvs && orientCvs.width === 300 && orientCvs.height === 512);
  check("rotR 画布 rotate(π/2) + drawImage 全尺寸", !!orientCvs
    && orientCvs.ops.some((o) => o.op === "rotate" && Math.abs(o.args[0] - Math.PI / 2) < 1e-9)
    && orientCvs.ops.some((o) => o.op === "drawImage" && o.args[1] === -256 && o.args[2] === -150
      && o.args[3] === 512 && o.args[4] === 300));
  check("rotR 上传新源图（cebm_ 前缀）", (globalThis.__uploadCalls || []).length === 1
    && String(globalThis.__uploadCalls[0].id).startsWith("cebm_")
    && globalThis.__uploadCalls[0].dataURL === "data:image/png;base64,stub");
  check("rotR 状态尺寸交换", stR.src_w === 300 && stR.src_h === 512);
  check("rotR 裁剪框重映射 (H-y-h, x)", stR.crop_x === 80 && stR.crop_y === 10 && stR.crop_w === 200 && stR.crop_h === 100);
  check("rotR 笔触重映射 (H-1-y, x)", JSON.stringify(stR.strokes[0].points) === JSON.stringify([[299, 0], [0, 511]]));
  check("rotR 比例复位 free", stR.aspect_ratio === "free");
  check("rotR 更新 src_path + 替换预览 img", stR.src_path === "sfnodes_crop/crop_src_" + globalThis.__uploadCalls[0].id + ".png"
    && node._sfCEBImg && node._sfCEBImg.src === "data:image/png;base64,stub");

  // 翻转：尺寸不变、框/笔触镜像（在 rotR 后的 300×512 状态上继续；
  // 预览 img 由 Image 桩替换后无尺寸，这里按新状态补齐）
  globalThis.__uploadCalls = [];
  node._sfCEBImg = { complete: true, naturalWidth: 300, naturalHeight: 512 };
  clickCtrl("flipH");
  await sleep(20);
  const stF = state();
  check("flipH 尺寸不变 + 框镜像 W-x-w", stF.src_w === 300 && stF.src_h === 512
    && stF.crop_x === 20 && stF.crop_y === 10 && stF.crop_w === 200 && stF.crop_h === 100);
  check("flipH 笔触镜像 W-1-x", JSON.stringify(stF.strokes[0].points) === JSON.stringify([[0, 0], [299, 511]]));
  // 无源图：只提示不上传
  node.properties[STATE_PROP] = makeState({ src_path: "" });
  globalThis.__uploadCalls = [];
  clickCtrl("flipV");
  await sleep(20);
  check("无源图点击不触发上传", (globalThis.__uploadCalls || []).length === 0);

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

  // ── 模式快捷键（C 裁剪 / B 笔刷 / E 擦除；大小写不敏感）──
  fireKey("e");
  check("E 切 Erase", state().brush_mode === "erase");
  fireKey("B");
  check("B 切 Brush（大写）", state().brush_mode === "brush");
  fireKey("c");
  check("C 切 Crop", state().brush_mode === "crop");
  check("设置项已注册（init）", (ext.init(), globalThis.__settingDefs["sfnodes.BrushMask.SizeStep"]?.defaultValue === 2));
  {
    const frameDef = globalThis.__settingDefs["sfnodes.Canvas.FrameWidth"];
    const cursorDef = globalThis.__settingDefs["sfnodes.Canvas.CursorWidth"];
    check("画布线条设置已注册（默认 1.0 / slider 0.5–3 step 0.25）",
      frameDef?.defaultValue === 1.0 && cursorDef?.defaultValue === 1.0
      && frameDef.type === "slider" && frameDef.attrs?.min === 0.5 && frameDef.attrs?.max === 3
      && frameDef.attrs?.step === 0.25
      && cursorDef.type === "slider" && cursorDef.attrs?.step === 0.25);
    const polyDef = globalThis.__settingDefs["sfnodes.Canvas.PolyVertexSize"];
    check("套索顶点尺寸设置已注册（默认 2 / slider 1–8 step 0.5）",
      polyDef?.defaultValue === 2 && polyDef.type === "slider"
      && polyDef.attrs?.min === 1 && polyDef.attrs?.max === 8 && polyDef.attrs?.step === 0.5);
  }

  // ── window capture 释放兜底 + onRemoved 解绑 ──
  node.properties[STATE_PROP] = makeState({ brush_mode: "brush" });
  node.onMouseDown({ button: 0, buttons: 1 }, [150, 110]);
  check("释放监听用例已起笔", node._sfCEBDrawing === true);
  fireWin("pointerup");
  check("window 释放监听落定", node._sfCEBDrawing === false && state().strokes.length === 1);
  nodeType.prototype.onRemoved.call(node);
  check("onRemoved 解绑释放监听", node._sfCEBReleaseGuard === null
    && winListeners.filter((l) => l.type === "mouseup").length === 0);


  // SAM 框选模式：进入 → 拖框 → 松开执行（POST 载荷含 bbox）
  node.properties[STATE_PROP] = makeState({ src_path: "sfnodes_crop/x.png", brush_mode: "brush" });
  globalThis.__apiCalls = [];
  menuOpts.find((o) => o.content.includes("框选")).callback();
  check("进入框选模式", !!node._sfAiSam && node._sfAiSam.kind === "box");
  node.onMouseDown({ button: 0, buttons: 1 }, [140, 70]);
  node.onMouseMove({ buttons: 1 }, [180, 110], { canvas: { style: {} }, setDirty() {} });
  check("框选记录橡皮筋", !!node._sfAiSam.box && node._sfAiSam.box.x2 > node._sfAiSam.box.x1);
  // 覆盖层绘制：虚线框 + 提示条，不抛错（曾缺 imageToLocal import → 整帧中断）
  const boxOps = [];
  let boxDrawErr = null;
  try { node.onDrawForeground(makeFullCtx(boxOps)); } catch (e) { boxDrawErr = e; }
  check("框选覆盖层绘制不抛错", boxDrawErr === null);
  check("框选画虚线框", boxOps.some((o) => o.op === "strokeRect"));
  check("框选提示条文本", boxOps.some((o) => o.op === "fillText" && String(o.args[0]).includes("框选")));
  node.onMouseUp({}, [], { canvas: { style: {} }, setDirty() {} });
  check("松开退出模式", node._sfAiSam == null);
  await new Promise((r) => setTimeout(r, 20));
  const boxCall = (globalThis.__apiCalls || []).find((c) => c.url.includes("/brush_mask/sam") && c.body);
  const boxBody = boxCall ? JSON.parse(boxCall.body) : null;
  check("框选 POST 含 bbox", !!boxBody && Array.isArray(boxBody.bbox) && boxBody.bbox.length === 4);

  // 模式即运算：Crop/Brush 模式识别结果保持 fill；Eraser 模式改写 fill_erase
  const aiMod2 = await import(path.join(tmpDir, "sf_brush_ai.js"));
  const captureCfg2 = (captured) => ({
    toastTag: "SF Crop Expand Brush Mask", logTag: "[SF Crop Expand Brush Mask]",
    getState: (n) => JSON.parse(n.properties[STATE_PROP]),
    patchState: () => {},
    addStrokes: (_n, inc) => { captured.push(...inc); },
  });
  globalThis.__aiResponse = { strokes: [{ mode: "fill", size: 0, points: [[1, 1], [2, 1], [2, 2]] }], coverage: 0.25 };
  node.properties[STATE_PROP] = makeState({ src_path: "sfnodes_crop/x.png", brush_mode: "crop" });
  let aiCap2 = [];
  await aiMod2.runYolo(captureCfg2(aiCap2), node, "bbox", "a.pt", 0.3, "rect", 640, []);
  check("Crop 模式识别结果保持 fill", aiCap2.length === 1 && aiCap2[0].mode === "fill");
  node.properties[STATE_PROP] = makeState({ src_path: "sfnodes_crop/x.png", brush_mode: "erase" });
  aiCap2 = [];
  await aiMod2.runYolo(captureCfg2(aiCap2), node, "bbox", "a.pt", 0.3, "rect", 640, []);
  check("Eraser 模式识别结果转 fill_erase", aiCap2.length === 1 && aiCap2[0].mode === "fill_erase");
  globalThis.__aiResponse = null;

  // 反选预览：源图区白底打洞（离屏 destination-out）
  node.properties[STATE_PROP] = makeState({
    src_path: "", src_w: 100, src_h: 100, brush_size: 20, invert: true,
    strokes: [{ mode: "brush", size: 20, points: [[10, 10]] }],
  });
  const beforeInv = createdCanvases.length;
  node.onDrawForeground(makeFullCtx([]));
  const newCvs = createdCanvases.slice(beforeInv);
  check("反选新建离屏画布并打洞", newCvs.some((c) => c.ops.some((o) => o.op === "set:globalCompositeOperation" && o.value === "destination-out")));

  // ── 非 Crop 模式隐藏裁剪框（压暗/框线/九宫格/手柄），Crop 模式显示 ──
  const drawModeOps = (mode) => {
    node.properties[STATE_PROP] = makeState({ src_path: "", src_w: 512, src_h: 512, brush_mode: mode });
    const ops = [];
    node.onDrawForeground(makeFullCtx(ops));
    return ops;
  };
  // 手柄为 10×10 的白色 fillRect（Pen 按钮同为白色但尺寸 30×18，故按尺寸区分）
  const handleCount = (ops) => ops.filter((o) => o.op === "fillRect" && o.fill === "rgba(255,255,255,0.9)"
    && o.args[2] === 10 && o.args[3] === 10).length;
  const hasGrid = (ops) => ops.some((o) => o.op === "set:strokeStyle" && o.value === "rgba(255,255,255,0.4)");
  const cropOps2 = drawModeOps("crop");
  const brushOps = drawModeOps("brush");
  const eraseOps = drawModeOps("erase");
  check("Crop 模式显示裁剪框手柄", handleCount(cropOps2) >= 8);
  check("Crop 模式显示九宫格", hasGrid(cropOps2) === true);
  check("Brush 模式隐藏裁剪框", handleCount(brushOps) === 0 && hasGrid(brushOps) === false);
  check("Erase 模式隐藏裁剪框", handleCount(eraseOps) === 0 && hasGrid(eraseOps) === false);

  // ── 线宽随设置（sfnodes.Canvas.*，§97）：边框/虚线 = FrameWidth，
  //    九宫格/手柄/网格 = max(0.5, W/2)，光环 = CursorWidth ──
  {
    const lws = (ops) => ops.filter((o) => o.op === "set:lineWidth").map((o) => o.value);
    check("默认细线 0.5（九宫格/手柄/网格）", lws(cropOps2).includes(0.5));
    const saved = { ...globalThis.__settingVals };
    globalThis.__settingVals["sfnodes.Canvas.FrameWidth"] = 2.5;
    globalThis.__settingVals["sfnodes.Canvas.CursorWidth"] = 2;
    const wCrop = drawModeOps("crop");
    check("裁剪框线宽随 FrameWidth（边框 2.5 / 细线 1.25）", lws(wCrop).includes(2.5) && lws(wCrop).includes(1.25));
    node._sfCEBCursor = [170, 120];
    globalThis.__canvas.node_over = node;
    const wBrush = drawModeOps("brush");
    check("光环线宽随 CursorWidth（2）", lws(wBrush).includes(2));
    // 源图边界虚线也用 FrameWidth（全模式保留），故不能按线宽判框；
    // 改按框线唯一色辨别（源图虚线是蓝 rgba(100,150,255,0.6)）
    const hasBoxBorder = (ops) => ops.some((o) => o.op === "strokeRect" && o.stroke === "rgba(255,255,255,0.9)");
    check("非 Crop 模式无裁剪框边框（白色）", hasBoxBorder(wBrush) === false && hasBoxBorder(wCrop) === true);
    globalThis.__canvas.node_over = null;
    globalThis.__settingVals = saved;
  }
  node.properties[STATE_PROP] = makeState({ src_path: "", src_w: 512, src_h: 512, brush_mode: "crop" });

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})().catch((e) => { console.error("SMOKE ERROR:", e); process.exit(1); });
