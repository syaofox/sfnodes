// SF Image Brush Mask lib 纯函数测试（Node 直接运行：node tests/test_brush_mask_lib.mjs）
// 覆盖：ensureMinSize / hitResizeCornerSE / computeDisplayMetrics /
// localToImage-imageToLocal 往返 / clampToImage / parseStroke 三格式兼容
// （与 sf_utils/brush_mask.py 双端镜像）/ parseBrushData / buildBrushData。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_brush_mask_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_brush_mask_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

const approx = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

(async () => {
  const L = await import(tmpUrl);

  // ── ensureMinSize ──
  check("低于下限抬升", JSON.stringify(L.ensureMinSize(10, 10)) === JSON.stringify([L.MIN_NODE_WIDTH, L.MIN_NODE_HEIGHT]));
  check("高于下限放行", JSON.stringify(L.ensureMinSize(800, 600)) === JSON.stringify([800, 600]));

  // ── hitResizeCornerSE ──
  check("右下角命中", L.hitResizeCornerSE(410, 310, 420, 320) === true);
  check("区外未命中", L.hitResizeCornerSE(100, 100, 420, 320) === false);

  // ── computeDisplayMetrics ──
  // 节点 420×320，源图 512×512：areaW=420-80-10=330，areaH=320-10-58-22-10=220
  const m = L.computeDisplayMetrics({ srcW: 512, srcH: 512 }, 420, 320);
  check("scale 取 min", approx(m.scale, 220 / 512));
  check("居中 offsetX", approx(m.offsetX, 10 + (330 - 512 * m.scale) / 2));
  check("居中 offsetY", approx(m.offsetY, 10 + 58 + (220 - 512 * m.scale) / 2));

  // ── 坐标往返 ──
  const p = L.localToImage(m.offsetX + 100 * m.scale, m.offsetY + 200 * m.scale, m);
  check("localToImage 还原", approx(p.x, 100) && approx(p.y, 200));
  const q = L.imageToLocal(100, 200, m);
  check("imageToLocal 还原", approx(q.x, m.offsetX + 100 * m.scale) && approx(q.y, m.offsetY + 200 * m.scale));

  // ── clampToImage ──
  check("越界钳制", JSON.stringify(L.clampToImage(-5, 999, 512, 512)) === JSON.stringify({ x: 0, y: 511 }));
  check("界内不动", JSON.stringify(L.clampToImage(10, 20, 512, 512)) === JSON.stringify({ x: 10, y: 20 }));

  // ── parseStroke 三格式（与 Python _parse_one_stroke 同语义）──
  let s = L.parseStroke("brush:20:1.0:10,10;20,20");
  check("新无色格式", s.mode === "brush" && s.size === 20 && s.points.length === 2 && s.points[0].x === 10);
  s = L.parseStroke("erase:30:0.5:255,0,0:5,5;6,6");
  check("新带色格式", s.mode === "erase" && s.size === 30 && s.points.length === 2);
  s = L.parseStroke("brush:7,7;8,8");
  check("旧 mode:points", s.mode === "brush" && s.points.length === 2);
  s = L.parseStroke("9,9;10,10");
  check("裸点列", s.mode === "brush" && s.points.length === 2);

  // ── parseBrushData / buildBrushData ──
  check("空串", L.parseBrushData("").length === 0);
  check("多笔", L.parseBrushData("brush:10:1.0:1,1|erase:10:1.0:2,2").length === 2);
  const wire = L.buildBrushData([{ mode: "brush", size: 20, points: [{ x: 10, y: 10 }, { x: 20, y: 20 }] }]);
  check("build 形状", wire === "brush:20:1:10,10;20,20");
  check("往返一致", L.parseBrushData(wire)[0].points.length === 2);

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})();
