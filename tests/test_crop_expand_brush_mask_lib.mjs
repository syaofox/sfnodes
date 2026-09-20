// SF Image Crop Expand Brush Mask lib 纯函数测试（Node 直接运行：
// node tests/test_crop_expand_brush_mask_lib.mjs）
// 覆盖：组合布局（LAYOUT/TOOL_COL/MIN/ensureMinSize/双列显示坐标系 extraLeft）
// / 基库新增对（imageToLocal 互逆、ensureMinSize 显式下限）
// / 共享绘制（sf_brush_mask_lib 的 colorTextStyle/drawStrokePath/paintStrokeMask
//   + sf_crop_expand_lib 的 drawPlaceholder/drawCropBox，FakeCtx 断言 op 流）。
// 加载方式：三个纯库（crop_expand / brush_mask / 组合）拷同目录后 import 组合库。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_cebm_lib_test_"));
for (const f of ["sf_crop_expand_lib.js", "sf_brush_mask_lib.js", "sf_crop_expand_brush_mask_lib.js"]) {
  fs.copyFileSync(path.join(here, "..", "web", f), path.join(tmpDir, f));
}

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}
const approx = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

// op 记录 ctx（属性赋值也记录，兼容 Proxy set 返回函数的老写法）
function makeCtx(ops) {
  const state = {};
  return new Proxy({}, {
    get(t, p) {
      if (p === "measureText") return () => ({ width: 0 });
      return (...a) => { ops.push({ op: p, args: a, fill: state.fillStyle, gco: state.globalCompositeOperation }); };
    },
    set(t, p, v) { state[p] = v; ops.push({ op: "set:" + p, value: v }); return true; },
  });
}

function makeCanvas(w = 10, h = 10) {
  const ops = [];
  return { width: w, height: h, ops, getContext: () => makeCtx(ops) };
}

(async () => {
  const L = await import(pathToFileURL(path.join(tmpDir, "sf_crop_expand_brush_mask_lib.js")).href);
  const Crop = await import(pathToFileURL(path.join(tmpDir, "sf_crop_expand_lib.js")).href);
  const Brush = await import(pathToFileURL(path.join(tmpDir, "sf_brush_mask_lib.js")).href);

  // ── 组合布局 ──
  check("TOOL_COL 12 项且 Crop 置顶", L.TOOL_COL.length === 12 && L.TOOL_COL[0] === "crop"
    && JSON.stringify(L.TOOL_COL.slice(1)) === JSON.stringify(Brush.TOOL_COL));
  check("ORIENT_COL 4 项（flipH/flipV/rotL/rotR）", JSON.stringify(L.ORIENT_COL) === JSON.stringify(["flipH", "flipV", "rotL", "rotR"]));
  check("列1/列2/列3 同宽 34", L.LAYOUT.ratioColW === 34 && L.LAYOUT.toolColW === 34 && L.LAYOUT.orientColW === 34);
  check("TOOL_COL_X = shiftLeft + 列1宽 + 间距", L.TOOL_COL_X === L.LAYOUT.shiftLeft + L.LAYOUT.ratioColW + L.LAYOUT.ratioColGap);
  check("ORIENT_COL_X = 列2 右缘 + 间距", L.ORIENT_COL_X === L.TOOL_COL_X + L.LAYOUT.toolColW + L.LAYOUT.toolColGap);
  check("EXTRA_LEFT = 列2 + 列3（含间距）= 80", L.EXTRA_LEFT === 80);
  check("MIN 400×340（列3 只加宽不加高）", L.MIN_NODE_WIDTH === 400 && L.MIN_NODE_HEIGHT === 340);
  check("ensureMinSize 抬升", JSON.stringify(L.ensureMinSize(10, 10)) === JSON.stringify([400, 340]));
  check("ensureMinSize 放行大尺寸", JSON.stringify(L.ensureMinSize(800, 600)) === JSON.stringify([800, 600]));

  // ── 列头/分组排布（§109）──
  check("列头/组间常量", L.HEADER_H === 10 && L.GROUP_EXTRA === 3 && L.FIRST_ROW_Y === 26);
  check("分组项数覆盖三列按钮", L.COL1_GROUPS.reduce((a, b) => a + b, 0) === 8 + 3
    && L.COL2_GROUPS.reduce((a, b) => a + b, 0) === L.TOOL_COL.length
    && L.COL3_GROUPS.reduce((a, b) => a + b, 0) === L.ORIENT_COL.length);
  check("列1 行位（组间 +3）", JSON.stringify(L.columnYs(L.COL1_GROUPS)) ===
    JSON.stringify([26, 48, 70, 92, 114, 136, 158, 180, 205, 227, 249]));
  check("列2 行位（组间 +3，末项 Pen 277）", JSON.stringify(L.columnYs(L.COL2_GROUPS)) ===
    JSON.stringify([26, 48, 70, 92, 117, 139, 161, 186, 208, 230, 252, 277]));
  check("列3 行位（单组四项）", JSON.stringify(L.columnYs(L.COL3_GROUPS)) ===
    JSON.stringify([26, 48, 70, 92]));

  // ── 三列显示坐标系（比基库多让 EXTRA_LEFT = 列2+列3）──
  const st = { cropX: 0, cropY: 0, cropW: 512, cropH: 512, srcW: 512, srcH: 512 };
  const m = L.computeDisplayMetrics(st, 360, 300, null);
  // areaW = 360-80-10-34-6-80 = 150；areaH = 300-20-26 = 254 → scale = 150/512
  check("组合 scale", approx(m.scale, 150 / 512));
  check("组合 offsetX 让出三列", approx(m.offsetX, 10 + 34 + 6 + 80));
  check("组合 offsetY 居中", approx(m.offsetY, 10 + (254 - 150) / 2));
  // 与基库显式 extraLeft 等价
  const mb = Crop.computeDisplayMetrics(st, 360, 300, null, L.EXTRA_LEFT);
  check("组合 metrics ≡ 基库 extraLeft", m.scale === mb.scale && m.offsetX === mb.offsetX && m.offsetY === mb.offsetY);
  // 冻结快照原样透传
  const frozen = { displayMinX: -1, displayMinY: -2, scale: 0.5, offsetX: 7, offsetY: 8, scaledDisplayWidth: 9, scaledDisplayHeight: 10 };
  const mf = L.computeDisplayMetrics(st, 999, 999, frozen);
  check("组合冻结快照透传", mf.scale === 0.5 && mf.offsetX === 7 && mf.displayMinX === -1);
  // 基库默认行为不变（extraLeft 省略 = 0）
  const m0 = Crop.computeDisplayMetrics(st, 360, 300, null);
  check("基库默认 extraLeft=0", approx(m0.offsetX, 10 + 34 + 6) && approx(m0.scale, 230 / 512));

  // ── 基库新增：imageToLocal 互逆 / ensureMinSize 显式下限 ──
  const p = Crop.imageToLocal(30, 40, m);
  const back = Crop.localToImage(p.x, p.y, m);
  check("imageToLocal 互逆", approx(back.x, 30) && approx(back.y, 40));
  check("基库 ensureMinSize 显式下限",
    JSON.stringify(Crop.ensureMinSize(1, 1, 500, 600)) === JSON.stringify([500, 600]));

  // ── orientState（源图整体翻转/旋转，§112）──
  // W=100,H=60；框 (10,20,30,40)；笔触两点 (0,0)/(99,59)
  const ost = {
    src_w: 100, src_h: 60,
    crop_x: 10, crop_y: 20, crop_w: 30, crop_h: 40,
    strokes: [{ mode: "brush", size: 8, points: [[0, 0], [99, 59]] }],
    aspect_ratio: "16:9",
  };
  const ostJSON = JSON.stringify(ost);
  const flat = (s) => JSON.stringify(s.strokes);
  {
    const r = L.orientState(ost, "flipH");
    check("flipH 尺寸不变 + 框镜像 W-x-w", r.src_w === 100 && r.src_h === 60
      && r.crop_x === 60 && r.crop_y === 20 && r.crop_w === 30 && r.crop_h === 40);
    check("flipH 笔触镜像 W-1-x", flat(r) === JSON.stringify([{ mode: "brush", size: 8, points: [[99, 0], [0, 59]] }]));
    check("flipH 保留比例预设", r.aspect_ratio === "16:9");
  }
  {
    const r = L.orientState(ost, "flipV");
    check("flipV 尺寸不变 + 框镜像 H-y-h", r.src_w === 100 && r.src_h === 60
      && r.crop_x === 10 && r.crop_y === 0 && r.crop_w === 30 && r.crop_h === 40);
    check("flipV 笔触镜像 H-1-y", flat(r) === JSON.stringify([{ mode: "brush", size: 8, points: [[0, 59], [99, 0]] }]));
  }
  {
    const r = L.orientState(ost, "rotL");
    check("rotL 尺寸交换 + 框 (y, W-x-w)", r.src_w === 60 && r.src_h === 100
      && r.crop_x === 20 && r.crop_y === 60 && r.crop_w === 40 && r.crop_h === 30);
    check("rotL 笔触 (y, W-1-x)", flat(r) === JSON.stringify([{ mode: "brush", size: 8, points: [[0, 99], [59, 0]] }]));
    check("rotL 比例复位 free", r.aspect_ratio === "free");
  }
  {
    const r = L.orientState(ost, "rotR");
    check("rotR 尺寸交换 + 框 (H-y-h, x)", r.src_w === 60 && r.src_h === 100
      && r.crop_x === 0 && r.crop_y === 10 && r.crop_w === 40 && r.crop_h === 30);
    check("rotR 笔触 (H-1-y, x)", flat(r) === JSON.stringify([{ mode: "brush", size: 8, points: [[59, 0], [0, 99]] }]));
    check("rotR 比例复位 free", r.aspect_ratio === "free");
  }
  // 逆变换恒等（旋转 ±90 互逆、镜像自逆）：尺寸/框/笔触全部还原
  for (const [op, inv] of [["flipH", "flipH"], ["flipV", "flipV"], ["rotL", "rotR"], ["rotR", "rotL"]]) {
    const a = L.orientState(ost, op);
    const b = L.orientState({ ...ost, ...a }, inv);
    check(`${op} + ${inv} 恒等还原`, b.src_w === 100 && b.src_h === 60
      && b.crop_x === 10 && b.crop_y === 20 && b.crop_w === 30 && b.crop_h === 40
      && flat(b) === flat(ost));
  }
  check("orientState 不改入参", JSON.stringify(ost) === ostJSON);
  check("未知 op 返回 null", L.orientState(ost, "rot180") === null);
  check("非法尺寸返回 null", L.orientState({ ...ost, src_w: 0 }, "flipH") === null
    && L.orientState({ ...ost, src_h: -1 }, "rotR") === null);
  check("无 strokes 安全（空数组）", flat(L.orientState({ ...ost, strokes: undefined }, "flipH")) === "[]");

  // ── colorTextStyle ──
  check("取色文字 白底黑字", Brush.colorTextStyle("#ffffff") === "rgba(0,0,0,0.9)");
  check("取色文字 黑底白字", Brush.colorTextStyle("#000000") === "rgba(255,255,255,0.9)");
  check("取色文字 rgb 串", Brush.colorTextStyle("255,255,255") === "rgba(0,0,0,0.9)");

  // ── drawStrokePath ──
  {
    const ops = [];
    Brush.drawStrokePath(makeCtx(ops), [[5, 5]], { scale: 1, offsetX: 0, offsetY: 0 }, 8, "rgba(1,2,3,1)", false);
    check("单点印章画圆盘", ops.some((o) => o.op === "arc" && approx(o.args[2], 4)) && ops.some((o) => o.op === "fill"));
  }
  {
    const ops = [];
    Brush.drawStrokePath(makeCtx(ops), [[0, 0], [10, 10]], { scale: 2, offsetX: 3, offsetY: 4 }, 5, "rgba(1,2,3,1)", false);
    const moved = ops.filter((o) => o.op === "moveTo");
    check("多点描边含坐标变换", moved.length === 1 && approx(moved[0].args[0], 3) && approx(moved[0].args[1], 4)
      && ops.some((o) => o.op === "stroke"));
  }
  {
    const ops = [];
    const pts = [[0, 0], [5, 0], [5, 5]];
    Brush.drawStrokePath(makeCtx(ops), pts, { scale: 1, offsetX: 0, offsetY: 0 }, 0, "rgba(1,2,3,1)", true);
    check("fill 笔触闭合填充", ops.some((o) => o.op === "closePath") && ops.some((o) => o.op === "fill"));
  }

  // ── paintStrokeMask（真擦除画序 + 进行中笔触）──
  {
    const cvs = makeCanvas(20, 20);
    Brush.paintStrokeMask(cvs, [
      { mode: "brush", size: 6, points: [[2, 2]] },
      { mode: "erase", size: 6, points: [[8, 8]] },
    ], { defaultSize: 6, paintStyle: "rgba(255,255,255,1)", current: { mode: "brush", size: 6, points: [[15, 15]] } });
    const clearIdx = cvs.ops.findIndex((o) => o.op === "clearRect");
    const firstArcIdx = cvs.ops.findIndex((o) => o.op === "arc");
    check("合成先清屏（笔触前）", clearIdx >= 0 && clearIdx < firstArcIdx);
    const gcos = cvs.ops.filter((o) => o.op === "set:globalCompositeOperation").map((o) => o.value);
    check("erase 用 destination-out 且随后复位", gcos.includes("destination-out") && gcos[gcos.length - 1] === "source-over");
    check("擦除后仍有进行中笔触绘制", cvs.ops.filter((o) => o.op === "arc").length >= 3);
  }
  {
    const cvs = makeCanvas(20, 20);
    Brush.paintStrokeMask(cvs, [{ mode: "brush", size: 6, points: [[2, 2]] }], {});
    check("无 erase 无打洞", !cvs.ops.some((o) => o.value === "destination-out"));
  }
  {
    // fill_erase（Eraser 模式 AI 结果）：destination-out + 多边形整体填充
    const cvs = makeCanvas(20, 20);
    Brush.paintStrokeMask(cvs, [
      { mode: "fill", size: 0, points: [[2, 2], [10, 2], [10, 10], [2, 10]] },
      { mode: "fill_erase", size: 0, points: [[4, 4], [8, 4], [8, 8], [4, 8]] },
    ], { paintStyle: "rgba(255,255,255,1)" });
    const gcos = cvs.ops.filter((o) => o.op === "set:globalCompositeOperation").map((o) => o.value);
    const dstIdx = cvs.ops.findIndex((o) => o.op === "set:globalCompositeOperation" && o.value === "destination-out");
    const closeIdx = cvs.ops.findIndex((o, i) => i > dstIdx && o.op === "closePath");
    check("fill_erase 走 destination-out 多边形", dstIdx >= 0 && closeIdx > dstIdx
      && gcos[gcos.length - 1] === "source-over");
    check("fill_erase 无 paintStyle 填充（打洞为纯擦除）",
      !cvs.ops.some((o, i) => o.op === "fill" && i > dstIdx && o.fill === "rgba(255,255,255,1)"));
  }

  // ── paintInvertMask（反选预览：白底 + 打洞，仅离屏）──
  {
    const src = makeCanvas(20, 20);
    const out = makeCanvas(20, 20);
    Brush.paintInvertMask(src, out);
    const gcos = out.ops.filter((o) => o.op === "set:globalCompositeOperation").map((o) => o.value);
    const fillIdx = out.ops.findIndex((o) => o.op === "fillRect");
    const holeIdx = out.ops.findIndex((o) => o.op === "drawImage");
    check("反选白底先画", fillIdx >= 0 && holeIdx > fillIdx);
    check("反选 destination-out 打洞并复位", gcos.includes("destination-out") && gcos[gcos.length - 1] === "source-over");
    check("反选返回目标画布", Brush.paintInvertMask(src, out) === out);
  }

  // ── drawPlaceholder / drawCropBox ──
  {
    const ops = [];
    Crop.drawPlaceholder(makeCtx(ops), 0, 0, 64, 64, 1);
    check("占位图背景 + 网格", ops.some((o) => o.op === "fillRect") && ops.filter((o) => o.op === "stroke").length >= 4);
  }
  {
    const ops = [];
    Crop.drawCropBox(makeCtx(ops), { x: 1, y: 1, w: 2, h: 2 }, 4, 4, m);
    const strokes = ops.filter((o) => o.op === "strokeRect");
    check("裁剪框边框 + 8 手柄", strokes.length >= 9);
    const handleFills = ops.filter((o) => o.op === "fillRect");
    check("裁剪框控制点填充", handleFills.length >= 8);
    check("框外压暗", ops.some((o) => o.op === "fillRect" && String(o.fill).includes("rgba(0,0,0,0.5)")));
  }
  {
    // 线宽参数（§97）：边框 = lineW，九宫格/手柄描边 = max(0.5, lineW/2)
    const lws = (ops) => ops.filter((o) => o.op === "set:lineWidth").map((o) => o.value);
    const withW = [];
    Crop.drawCropBox(makeCtx(withW), { x: 1, y: 1, w: 2, h: 2 }, 4, 4, m, 2.5);
    check("drawCropBox 线宽随参数（2.5 / 1.25）", lws(withW).includes(2.5) && lws(withW).includes(1.25));
    const def = [];
    Crop.drawCropBox(makeCtx(def), { x: 1, y: 1, w: 2, h: 2 }, 4, 4, m);
    check("drawCropBox 默认线宽（1 / 0.5）", lws(def).includes(1) && lws(def).includes(0.5));
  }

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})().catch((e) => { console.error("LIB TEST ERROR:", e); process.exit(1); });
