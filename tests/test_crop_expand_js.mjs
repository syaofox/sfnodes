// SF Image Crop Expand lib 纯函数测试（Node 直接运行：node tests/test_crop_expand_js.mjs）
// 覆盖：ratioFromAspect / computeDisplayMetrics（动态 + 冻结快照）/ localToImage
// / getHandleAtPoint / updateCropByDrag（八向 + 比例约束 + 最小尺寸）/
// applyRatioToRect / roundRect / isExtended。
// 用例与原版 ycImageCrop 交互语义对齐（拖拽冻结快照防飘移是核心行为）。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_crop_expand_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_crop_expand_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

const approx = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

(async () => {
  const L = await import(tmpUrl);

  // ── ratioFromAspect ──
  check("free 无约束", L.ratioFromAspect("free") === null);
  check("1:1", L.ratioFromAspect("1:1") === 1);
  check("16:9", approx(L.ratioFromAspect("16:9"), 16 / 9));
  check("9:16", approx(L.ratioFromAspect("9:16"), 9 / 16));
  check("custom 16:9", approx(L.ratioFromAspect("custom", 16, 9), 16 / 9));
  check("custom 非法退化 null", L.ratioFromAspect("custom", 0, 9) === null && L.ratioFromAspect("custom", undefined, undefined) === null);
  check("未知 key null", L.ratioFromAspect("nope") === null);

  // ── computeDisplayMetrics（动态计算）──
  // 状态：源图 512×512，裁剪框 (-256,-256,1024,1024) → 显示区 -256..768 = 1024²
  const st = { cropX: -256, cropY: -256, cropW: 1024, cropH: 1024, srcW: 512, srcH: 512 };
  const m = L.computeDisplayMetrics(st, 500, 400, null);
  // 区面积：x 500-80-10=410，y 400-10-10-58=322 → scale = min(410/1024, 322/1024)
  check("动态 scale", approx(m.scale, 322 / 1024));
  check("动态 displayMin", m.displayMinX === -256 && m.displayMinY === -256);
  check("动态 scaled 尺寸", approx(m.scaledDisplayWidth, 322) && approx(m.scaledDisplayHeight, 322));
  // 显示区居中：offsetX = 10 + (410 - 322)/2
  check("动态 offsetX 居中", approx(m.offsetX, 10 + (410 - 322) / 2));
  check("动态 offsetY 居中", approx(m.offsetY, 10 + 58 + 0));

  // 框在图内不越界：displayMin = 0
  const m2 = L.computeDisplayMetrics({ cropX: 0, cropY: 0, cropW: 512, cropH: 512, srcW: 512, srcH: 512 }, 500, 400, null);
  check("框内 displayMin 为 0", m2.displayMinX === 0 && m2.displayMinY === 0);

  // 冻结快照原样返回
  const frozen = { displayMinX: -1, displayMinY: -2, scale: 0.5, offsetX: 7, offsetY: 8, scaledDisplayWidth: 9, scaledDisplayHeight: 10 };
  const m3 = L.computeDisplayMetrics(st, 500, 400, frozen);
  check("冻结快照透传", m3.scale === 0.5 && m3.offsetX === 7 && m3.displayMinY === -2 && m3.scaledDisplayHeight === 10);

  // ── localToImage ──
  // 图片坐标 → 屏幕坐标 → 反算应还原
  const p = L.localToImage(m.offsetX + (100 - m.displayMinX) * m.scale, m.offsetY + (200 - m.displayMinY) * m.scale, m);
  check("localToImage 还原", approx(p.x, 100) && approx(p.y, 200));

  // ── getHandleAtPoint ──
  const rect = { x: 0, y: 0, w: 100, h: 100 };
  const s1 = 1;
  check("命中 nw", L.getHandleAtPoint(0, 0, rect, s1) === "nw");
  check("命中 se", L.getHandleAtPoint(100, 100, rect, s1) === "se");
  check("命中 ne", L.getHandleAtPoint(100, 0, rect, s1) === "ne");
  check("命中 sw", L.getHandleAtPoint(0, 100, rect, s1) === "sw");
  check("命中 n 边", L.getHandleAtPoint(50, 0, rect, s1) === "n");
  check("命中 s 边", L.getHandleAtPoint(50, 100, rect, s1) === "s");
  check("命中 w 边", L.getHandleAtPoint(0, 50, rect, s1) === "w");
  check("命中 e 边", L.getHandleAtPoint(100, 50, rect, s1) === "e");
  check("框内 move", L.getHandleAtPoint(50, 50, rect, s1) === "move");
  check("框外 null", L.getHandleAtPoint(150, 50, rect, s1) === null);
  // 缩放后命中半径随 1/scale 放大（屏幕 10px 恒定）
  check("scale 0.5 命中半径放大", L.getHandleAtPoint(14, 50, rect, 0.5) === "w");
  check("scale 0.5 远处不命中", L.getHandleAtPoint(30, 50, rect, 0.5) === "move");

  // ── updateCropByDrag ──
  const drag = { startImgX: 0, startImgY: 0, startRect: { x: 0, y: 0, w: 100, h: 100 } };
  check("move 平移", JSON.stringify(L.updateCropByDrag(drag, "move", 10, 20, null)) === JSON.stringify({ x: 10, y: 20, w: 100, h: 100 }));
  check("se 扩张", JSON.stringify(L.updateCropByDrag(drag, "se", 30, 40, null)) === JSON.stringify({ x: 0, y: 0, w: 130, h: 140 }));
  check("nw 反向", JSON.stringify(L.updateCropByDrag(drag, "nw", -10, -20, null)) === JSON.stringify({ x: -10, y: -20, w: 110, h: 120 }));
  check("n 调高", JSON.stringify(L.updateCropByDrag(drag, "n", 0, -15, null)) === JSON.stringify({ x: 0, y: -15, w: 100, h: 115 }));
  check("e 调宽", JSON.stringify(L.updateCropByDrag(drag, "e", 25, 0, null)) === JSON.stringify({ x: 0, y: 0, w: 125, h: 100 }));
  // 最小尺寸钳制
  check("最小尺寸钳制", L.updateCropByDrag(drag, "se", -95, -95, null).w === 10);
  // 比例约束：角点以宽度为准 h = w / ratio
  const r16 = 16 / 9;
  const drag2 = { startImgX: 0, startImgY: 0, startRect: { x: 0, y: 0, w: 160, h: 90 } };
  const se16 = L.updateCropByDrag(drag2, "se", 90, 90, r16);
  check("角点比例 h=w/ratio", se16.w === 250 && se16.h === Math.round(250 / r16));
  // e 边：h = w / ratio
  const e16 = L.updateCropByDrag(drag2, "e", 180, 0, r16);
  check("e 边比例", e16.w === 340 && e16.h === Math.round(340 / r16));
  // s 边：w = h * ratio
  const s16 = L.updateCropByDrag(drag2, "s", 0, 45, r16);
  check("s 边比例", s16.h === 135 && s16.w === Math.round(135 * r16));
  // move 不受比例约束
  const mv = L.updateCropByDrag(drag2, "move", 5, 5, r16);
  check("move 不应用比例", mv.x === 5 && mv.y === 5 && mv.w === 160 && mv.h === 90);

  // ── roundRect ──
  check("roundRect 取整", JSON.stringify(L.roundRect({ x: 1.4, y: -2.6, w: 100.5, h: 3.5 })) === JSON.stringify({ x: 1, y: -3, w: 101, h: 4 }));

  // ── applyRatioToRect ──
  const ar = L.applyRatioToRect({ x: 10, y: 10, w: 160, h: 90 }, r16);
  check("applyRatio 保持 x/w", ar.x === 10 && ar.w === 160);
  check("applyRatio h=w/ratio", ar.h === Math.round(160 / r16));
  const ar2 = L.applyRatioToRect({ x: 10, y: 100, w: 160, h: 90 }, r16);
  check("applyRatio 中心点不变", approx(ar2.y + ar2.h / 2, 100 + 90 / 2, 0.5 + 1e-9));
  check("applyRatio null 不变", JSON.stringify(L.applyRatioToRect({ x: 1, y: 2, w: 3, h: 4 }, null)) === JSON.stringify({ x: 1, y: 2, w: 3, h: 4 }));

  // ── isExtended ──
  check("框内不 extended", L.isExtended({ x: 0, y: 0, w: 512, h: 512 }, 512, 512) === false);
  check("负坐标 extended", L.isExtended({ x: -1, y: 0, w: 512, h: 512 }, 512, 512) === true);
  check("右侧越界 extended", L.isExtended({ x: 0, y: 0, w: 513, h: 512 }, 512, 512) === true);
  check("底部越界 extended", L.isExtended({ x: 0, y: 0, w: 512, h: 600 }, 512, 512) === true);

  // ── 常量 ──
  check("RATIO_PRESETS_ROW2 与原版一致", JSON.stringify(L.RATIO_PRESETS_ROW2) === JSON.stringify(["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"]));
  check("LAYOUT 与原版一致", L.LAYOUT.shiftLeft === 10 && L.LAYOUT.shiftRight === 80 && L.LAYOUT.panelHeight === 58);
  check("ASPECT_RATIOS 含 12 项", L.ASPECT_RATIOS.length === 12);
  // 最小节点尺寸：覆盖按钮行（row1 至 x≈350）+ shiftRight 80 + 画布区 + 信息文本
  //（节点恰在最小高度且画布填满时，信息文本基线 y 最大 = H+5，故下限 ≥365）
  check("MIN_NODE尺寸覆盖按钮行", L.MIN_NODE_WIDTH >= 440 && L.MIN_NODE_HEIGHT >= L.LAYOUT.shiftLeft * 2 + L.LAYOUT.panelHeight + 100);
  check("MIN_NODE高度≥信息文本最坏位置", L.MIN_NODE_HEIGHT >= 365);

  // ── ensureMinSize ──
  const ems = L.ensureMinSize(0, 0);
  check("ensureMinSize 兜底", ems[0] === L.MIN_NODE_WIDTH && ems[1] === L.MIN_NODE_HEIGHT);
  const ems2 = L.ensureMinSize(200, 100);
  check("ensureMinSize 抬升", ems2[0] === L.MIN_NODE_WIDTH && ems2[1] === L.MIN_NODE_HEIGHT);
  const ems3 = L.ensureMinSize(600, 400);
  check("ensureMinSize 原样放行", ems3[0] === 600 && ems3[1] === 400);
  const ems4 = L.ensureMinSize(undefined, undefined);
  check("ensureMinSize undefined 兜底", ems4[0] === L.MIN_NODE_WIDTH && ems4[1] === L.MIN_NODE_HEIGHT);

  // ── 结果 ──
  console.log();
  if (failures.length) {
    console.log(`${failures.length} FAILED: ${failures}`);
    process.exit(1);
  }
  console.log("ALL PASS");
})();
