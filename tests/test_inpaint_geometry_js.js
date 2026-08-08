// SF Inpaint Crop geometry 前端测试（Node 直接运行：node tests/test_inpaint_geometry_js.js）
// 覆盖：
// - computeRegion 三种模式（keep/force/free）+ 边界夹紧 + bank rounding（与 Python 一致）
// - maskBBoxFromImageData / growBBox
// - seamAlphaFromAlpha：遮罩内部 1、向外羽化 smoothstep、k<=0 边界
// 运行方式：复制为 .mjs 直跑（无 DOM 依赖的纯函数）。
const fs = require("fs");
const path = require("path");
const os = require("os");

const src = fs.readFileSync(path.join(__dirname, "..", "web", "sf_inpaint_geometry.js"), "utf8");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sf_inpaint_geo_"));
const mjs = path.join(tmp, "sf_inpaint_geometry.mjs");
fs.writeFileSync(mjs, src);

const { computeRegion, maskBBoxFromImageData, growBBox, seamAlphaFromAlpha } = require(mjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── keep：bbox 20x10，target 1024 -> 长边 1024，短边 512（镜像 Python）──
let r = computeRegion([10, 10, 30, 20], 100, 100, {
    size_mode: "keep", target: 1024, multiple: 8,
    context_px: 0, blend: 0, context_pct: 0, min_size: 256,
});
check("keep 长边到 target", r.out_w === 1024 && r.out_h === 512);
check("keep 源区域包含 bbox", r.rx <= 10 && r.ry <= 10 && r.rx + r.rw >= 30 && r.ry + r.rh >= 20);

// ── keep：target 太小 -> min_size 抬升（512x256，镜像 Python）──
r = computeRegion([10, 10, 30, 20], 200, 200, {
    size_mode: "keep", target: 64, multiple: 8,
    context_px: 0, blend: 0, context_pct: 0, min_size: 256, max_size: 2048,
});
check("keep min/max 夹紧", r.out_w === 512 && r.out_h === 256);

// ── force：恒 target x target 方形，源宽高比 1:1 ──
r = computeRegion([10, 10, 110, 60], 200, 200, {
    size_mode: "force", target: 512, target_w: 512, target_h: 512, multiple: 8,
    context_px: 0, blend: 0, context_pct: 0,
});
check("force 输出方形", r.out_w === 512 && r.out_h === 512);
check("force 源宽高比", r.rw === r.rh);

// ── free：bank rounding（20->16, 10->8），与 Python round() 一致 ──
r = computeRegion([10, 10, 30, 20], 100, 100, {
    size_mode: "free", multiple: 8, context_px: 0, blend: 0, context_pct: 0, max_size: 2048,
});
check("free 对齐倍数(bank)", r.out_w === 16 && r.out_h === 8);

// ── free：精确 .5 边界走 bank（1056/64 = 16.5 -> 16 -> 1024，不是 1088）──
r = computeRegion([10, 10, 1066, 20], 1200, 200, {
    size_mode: "free", multiple: 64, context_px: 0, blend: 0, context_pct: 0, max_size: 2048,
});
check("free 精确 .5 走 bank", r.out_w === 1024);

// ── bbox None -> 整图 ──
r = computeRegion(null, 100, 50, {
    size_mode: "free", multiple: 8, context_px: 0, blend: 0, context_pct: 0, max_size: 2048,
});
check("bbox None 整图", r.rx === 0 && r.ry === 0 && r.rw === 100 && r.rh === 50);

// ── 边缘遮罩 clamp ──
r = computeRegion([0, 0, 20, 20], 100, 100, {
    size_mode: "keep", target: 2048, multiple: 8,
    context_px: 0, blend: 0, context_pct: 0, max_size: 2048,
});
check("边缘遮罩 clamp", r.rx >= 0 && r.ry >= 0 && r.rx + r.rw <= 100 && r.ry + r.rh <= 100);

// ── context_pct ──
r = computeRegion([10, 10, 30, 20], 200, 200, {
    size_mode: "free", multiple: 8, context_px: 0, blend: 0, context_pct: 10, max_size: 2048,
});
check("context_pct 生效", r.rw >= 20 && r.rw < 25 && r.rh >= 10 && r.rh < 15);

// ── maskBBoxFromImageData ──
const w = 10, h = 10;
const data = new Uint8ClampedArray(w * h * 4);
check("bbox 空", maskBBoxFromImageData(data, w, h) === null);
for (let y = 2; y < 6; y++) for (let x = 3; x < 8; x++) data[(y * w + x) * 4 + 3] = 255;
check("bbox 非空", JSON.stringify(maskBBoxFromImageData(data, w, h)) === JSON.stringify([3, 2, 8, 6]));

// ── growBBox ──
check("growBBox 空", growBBox(null, 2, 100, 100) === null);
check("growBBox 外扩+夹紧", JSON.stringify(growBBox([0, 0, 10, 10], 2, 100, 100)) === JSON.stringify([0, 0, 12, 12]));
check("growBBox 夹紧", JSON.stringify(growBBox([98, 98, 100, 100], 4, 100, 100)) === JSON.stringify([94, 94, 100, 100]));

// ── seamAlphaFromAlpha ──
const sw = 32, sh = 32;
const sd = new Uint8ClampedArray(sw * sh * 4);
for (let y = 8; y < 24; y++) for (let x = 8; x < 24; x++) sd[(y * sw + x) * 4 + 3] = 255;
const a = seamAlphaFromAlpha(sd, sw, sh, 4);
check("seam 内部为 1", a[16 * sw + 16] === 1);
check("seam 遮罩边缘为 1", a[8 * sw + 16] === 1);
const outside = a[8 * sw + 2];  // 遮罩外 6px（> blend 4）-> 0
check("seam 远处为 0", outside === 0);
const mid = a[8 * sw + 6];      // 遮罩外 2px -> smoothstep 过渡
check("seam 过渡带", mid > 0 && mid < 1);
check("seam k<=0", seamAlphaFromAlpha(sd, sw, sh, 0)[16 * sw + 16] === 1);

console.log();
if (failures.length) { console.log(failures.length + " FAILURES: " + failures.join(", ")); process.exit(1); }
console.log("ALL PASS");
