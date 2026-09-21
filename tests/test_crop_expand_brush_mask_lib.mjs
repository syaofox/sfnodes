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
for (const f of ["sf_crop_expand_lib.js", "sf_brush_mask_lib.js", "sf_crop_expand_brush_mask_lib.js",
  "sf_dynamic_slots.js", "sf_canvas_size_lib.js"]) {
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
  check("TOOL_COL 13 项：Crop 置顶 + Ext 插在 Invert 后", L.TOOL_COL.length === 13 && L.TOOL_COL[0] === "crop"
    && L.TOOL_COL[6] === "invert" && L.TOOL_COL[7] === "includeExt"
    && JSON.stringify(L.TOOL_COL.filter((id) => id !== "crop" && id !== "includeExt")) === JSON.stringify(Brush.TOOL_COL));
  check("ORIENT_COL 4 项（flipH/flipV/rotL/rotR）", JSON.stringify(L.ORIENT_COL) === JSON.stringify(["flipH", "flipV", "rotL", "rotR"]));
  check("列1/列2/列3 同宽 34", L.LAYOUT.ratioColW === 34 && L.LAYOUT.toolColW === 34 && L.LAYOUT.orientColW === 34);
  check("TOOL_COL_X = shiftLeft + 列1宽 + 间距", L.TOOL_COL_X === L.LAYOUT.shiftLeft + L.LAYOUT.ratioColW + L.LAYOUT.ratioColGap);
  check("ORIENT_COL_X = 列2 右缘 + 间距", L.ORIENT_COL_X === L.TOOL_COL_X + L.LAYOUT.toolColW + L.LAYOUT.toolColGap);
  check("EXTRA_LEFT = 列2 + 列3（含间距）= 80", L.EXTRA_LEFT === 80);
  check("MIN 400×360（列2 13 项含 Ext + 列3 只加宽）", L.MIN_NODE_WIDTH === 400 && L.MIN_NODE_HEIGHT === 360);
  check("ensureMinSize 抬升", JSON.stringify(L.ensureMinSize(10, 10)) === JSON.stringify([400, 360]));
  check("ensureMinSize 放行大尺寸", JSON.stringify(L.ensureMinSize(800, 600)) === JSON.stringify([800, 600]));

  // ── 列头/分组排布（§109）──
  check("列头/组间常量", L.HEADER_H === 10 && L.GROUP_EXTRA === 3 && L.FIRST_ROW_Y === 26);
  check("分组项数覆盖三列按钮", L.COL1_GROUPS.reduce((a, b) => a + b, 0) === 8 + 3
    && L.COL2_GROUPS.reduce((a, b) => a + b, 0) === L.TOOL_COL.length
    && L.COL3_GROUPS.reduce((a, b) => a + b, 0) === L.ORIENT_COL.length);
  check("列1 行位（组间 +3）", JSON.stringify(L.columnYs(L.COL1_GROUPS)) ===
    JSON.stringify([26, 48, 70, 92, 114, 136, 158, 180, 205, 227, 249]));
  check("列2 行位（组间 +3，末项 Pen 299）", JSON.stringify(L.columnYs(L.COL2_GROUPS)) ===
    JSON.stringify([26, 48, 70, 92, 117, 139, 161, 183, 208, 230, 252, 274, 299]));
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

  // ── 接线宽高比 wiredAspect / effectiveRatio（§118）──
  // 每项输入独立上游节点（各自恰好一个数值 widget）；link 断开为 null。
  const wiredNode = (wVal, hVal) => {
    const links = {};
    const nodes = {};
    if (wVal !== undefined) {
      links[11] = { origin_id: 101, origin_slot: 0 };
      nodes[101] = { widgets: [{ value: wVal }] };
    }
    if (hVal !== undefined) {
      links[12] = { origin_id: 102, origin_slot: 0 };
      nodes[102] = { widgets: [{ value: hVal }] };
    }
    return {
      inputs: [
        { name: "aspect_w", link: wVal !== undefined ? 11 : null },
        { name: "aspect_h", link: hVal !== undefined ? 12 : null },
      ],
      graph: { links, getNodeById: (id) => nodes[id] || null },
    };
  };
  const freeSt = { aspect_ratio: "free", custom_w: 1, custom_h: 1 };
  {
    const wa = L.wiredAspect(wiredNode(1920, 1080));
    check("wiredAspect 双接可读", wa.wired === true && wa.partial === false
      && approx(wa.ratio, 16 / 9) && wa.w === 1920 && wa.h === 1080);
    check("wiredAspect 半接（仅 w）", (() => {
      const r = L.wiredAspect(wiredNode(16, undefined));
      return r.wired === false && r.partial === true && r.ratio === null;
    })());
    check("wiredAspect 半接（仅 h）", (() => {
      const r = L.wiredAspect(wiredNode(undefined, 9));
      return r.wired === false && r.partial === true;
    })());
    check("wiredAspect 全未接", (() => {
      const r = L.wiredAspect(wiredNode(undefined, undefined));
      return r.wired === false && r.partial === false;
    })());
    // 值不可读：上游多数值 widget（seed/steps 类）→ ratio null（编辑期不约束）
    const unreadable = wiredNode(16, 9);
    unreadable.graph.getNodeById = () => ({ widgets: [{ value: 1 }, { value: 2 }] });
    check("wiredAspect 不可读 → ratio null", (() => {
      const r = L.wiredAspect(unreadable);
      return r.wired === true && r.ratio === null && r.w === null && r.h === null;
    })());
    check("wiredAspect 非法值（0/负）→ ratio null", (() => {
      const r = L.wiredAspect(wiredNode(0, 9));
      const r2 = L.wiredAspect(wiredNode(-16, 9));
      return r.wired === true && r.ratio === null && r2.ratio === null;
    })());
    // readWiredInt 截断镜像（2.7 -> 2，与后端 int() 同口径）
    check("wiredAspect 值截断", L.wiredAspect(wiredNode(16.7, 9.2)).w === 16);
  }
  // 分辨率预设上游（§118 实测修复）：combo 无数值 widget，按上游输出槽名
  // width/height + resolution 值静态解析；槽名不符/畸形值不套用
  const sizeNode = (resValue, slotNames = ["width", "height", "resolution", "aspect_ratio"]) => {
    const linkIds = { w: 31, h: 32 };
    return {
      inputs: [
        { name: "aspect_w", link: linkIds.w },
        { name: "aspect_h", link: linkIds.h },
      ],
      graph: {
        links: {
          [linkIds.w]: { origin_id: 201, origin_slot: 0 },
          [linkIds.h]: { origin_id: 201, origin_slot: 1 },
        },
        getNodeById: () => ({
          id: 201, type: "SFCanvasSizePreset",
          widgets: [{ name: "model", value: "Z-Image (Turbo)" }, { name: "resolution", value: resValue }],
          outputs: slotNames.map((name) => ({ name })),
        }),
      },
    };
  };
  {
    const wa = L.wiredAspect(sizeNode("1024x1024 (1:1)"));
    check("分辨率预设上游：1024x1024 → ratio 1", wa.wired === true && approx(wa.ratio, 1)
      && wa.w === 1024 && wa.h === 1024);
    const wa2 = L.wiredAspect(sizeNode("1024x768 (4:3)"));
    check("分辨率预设上游：4:3", approx(wa2.ratio, 4 / 3) && wa2.w === 1024 && wa2.h === 768);
    check("分辨率预设上游：裸 WxH", approx(L.wiredAspect(sizeNode("704x1408")).ratio, 704 / 1408));
    check("分辨率预设上游：畸形值 → ratio null", L.wiredAspect(sizeNode("bogus")).ratio === null);
    // 槽名不是 width/height（如 LATENT/batch_size）→ 不可读（宁可不套也不猜）
    check("分辨率预设上游：槽名不符 → ratio null", L.wiredAspect(
      sizeNode("960x544 (16:9)", ["LATENT", "batch_size", "foo"])).ratio === null);
    // 槽位序号不固定（[LATENT, width, height] 形态：width=1/height=2）按槽名识别
    const shifted = sizeNode("960x544 (16:9)", ["LATENT", "width", "height"]);
    shifted.graph.links[31].origin_slot = 1;
    shifted.graph.links[32].origin_slot = 2;
    check("分辨率预设上游：槽位序号不参与（按槽名）", approx(L.wiredAspect(shifted).ratio, 960 / 544));
    // 反序接线（height→aspect_w、width→aspect_h）：接线值优先 → 取反比（§118.3.7）
    const sizeRev = sizeNode("1024x768 (4:3)");
    sizeRev.graph.links[31].origin_slot = 1;
    sizeRev.graph.links[32].origin_slot = 0;
    check("分辨率预设上游：反序接线取反比（3:4）", (() => {
      const r = L.wiredAspect(sizeRev);
      return approx(r.ratio, 768 / 1024) && r.w === 768 && r.h === 1024;
    })());
    // 只有一项走分辨率预设时仍按半接/双接判定
    const partial = sizeNode("1024x1024 (1:1)");
    partial.inputs[1].link = null;
    check("分辨率预设上游：半接不生效", L.wiredAspect(partial).partial === true);
  }
  // 图片链上游（LoadImage → GetImageSize，§118 实测修复）：GetImageSize 无
  // widget，沿其 IMAGE 输入回溯 LoadImage 预览尺寸（imgs[0].naturalWidth/Height）
  const imgChainNode = (w, h, { withPreview = true, midHop = false, midPreview = null } = {}) => {
    const g = { links: {}, getNodeById: (id) => nodes[id] || null };
    const nodes = {};
    const loadImage = { id: 401, type: "LoadImage", inputs: [], widgets: [], imgs: [] };
    if (withPreview) loadImage.imgs = [{ naturalWidth: w, naturalHeight: h }];
    const getSize = {
      id: 402, type: "GetImageSize", widgets: [], graph: g,
      inputs: [{ name: "image", type: "IMAGE", link: 41 }],
      outputs: [{ name: "width" }, { name: "height" }, { name: "batch_size" }],
    };
    nodes[401] = loadImage; nodes[402] = getSize;
    if (midHop) {
      const scale = {
        id: 403, type: "ImageScale", widgets: [], graph: g,
        inputs: [{ name: "image", type: "IMAGE", link: 42 }], outputs: [{ name: "IMAGE", type: "IMAGE" }],
        imgs: midPreview ? [{ naturalWidth: midPreview[0], naturalHeight: midPreview[1] }] : [],
      };
      nodes[403] = scale;
      getSize.inputs[0].link = 43;
      g.links[43] = { origin_id: 403, origin_slot: 0 };
      g.links[42] = { origin_id: 401, origin_slot: 0 };
    } else {
      g.links[41] = { origin_id: 401, origin_slot: 0 };
    }
    loadImage.graph = g;
    return {
      inputs: [{ name: "aspect_w", link: 31 }, { name: "aspect_h", link: 32 }],
      graph: {
        links: {
          31: { origin_id: 402, origin_slot: 0 },
          32: { origin_id: 402, origin_slot: 1 },
        },
        getNodeById: (id) => nodes[id] || null,
      },
    };
  };
  {
    const wa = L.wiredAspect(imgChainNode(800, 600));
    check("图片链上游：LoadImage → GetImageSize（4:3）", approx(wa.ratio, 4 / 3) && wa.w === 800 && wa.h === 600);
    check("图片链上游：预览未就绪 → ratio null", L.wiredAspect(imgChainNode(800, 600, { withPreview: false })).ratio === null);
    check("图片链上游：中间输出图节点无预览 → 不猜（null）",
      L.wiredAspect(imgChainNode(1920, 1080, { midHop: true })).ratio === null);
    check("图片链上游：中间节点预览为准（非源图）", approx(
      L.wiredAspect(imgChainNode(1920, 1080, { midHop: true, midPreview: [1024, 1024] })).ratio, 1));
    // 上游槽名决定分量：batch_size 不是 width/height → 不可读（不再按端口猜）
    const oddSlot = imgChainNode(800, 600);
    oddSlot.graph.links[32].origin_slot = 2;   // batch_size → aspect_h
    check("图片链上游：槽名不符（batch_size）→ ratio null", L.wiredAspect(oddSlot).ratio === null);
    // 反序接线（height→aspect_w、width→aspect_h）：接线值优先 → 取反比
    const reversed = imgChainNode(800, 600);
    reversed.graph.links[31].origin_slot = 1;  // → aspect_w
    reversed.graph.links[32].origin_slot = 0;  // → aspect_h
    const rw = L.wiredAspect(reversed);
    check("图片链上游：反序接线取反比（600:800）", approx(rw.ratio, 600 / 800) && rw.w === 600 && rw.h === 800);
  }
  {
    const panelSt = { aspect_ratio: "16:9", custom_w: 1, custom_h: 1 };
    check("effectiveRatio 接线可读优先于面板", approx(L.effectiveRatio(panelSt, L.wiredAspect(wiredNode(4, 3))), 4 / 3));
    check("effectiveRatio 接线不可读 → 不约束（null）", (() => {
      const n = wiredNode(4, 3);
      n.graph.getNodeById = () => ({ widgets: [{ value: 1 }, { value: 2 }] });
      return L.effectiveRatio(panelSt, L.wiredAspect(n)) === null;
    })());
    check("effectiveRatio 半接回退面板", approx(L.effectiveRatio(panelSt, L.wiredAspect(wiredNode(4, undefined))), 16 / 9));
    check("effectiveRatio 未接回退面板", approx(L.effectiveRatio(panelSt, L.wiredAspect(wiredNode(undefined, undefined))), 16 / 9));
    check("effectiveRatio 未接 Free → null", L.effectiveRatio(freeSt, L.wiredAspect(wiredNode(undefined, undefined))) === null);
    check("effectiveRatio 未接自定义", approx(L.effectiveRatio(
      { aspect_ratio: "custom", custom_w: 21, custom_h: 9 }, L.wiredAspect(wiredNode(undefined, undefined))), 21 / 9));
  }

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
