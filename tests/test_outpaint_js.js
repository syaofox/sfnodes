// SF Image Outpaint 前端核心数学测试（Node 直接运行：node tests/test_outpaint_js.js）
// 覆盖（sf_outpaint_core.js 纯函数，与 Python tests/test_outpaint.py 交叉对齐）：
//   - parseRatio：合法/非法/inf/nan/多冒号
//   - anchorAxis / remapAnchor：轴判断与跨轴保留
//   - padsForRatio：三方向 + anchor + round-half-up 边界（与 Python 同用例）
//   - padsForState：sides 夹紧 MAX_PAD
//   - finalSize：snap-once、limit 缩放、clamp、MP 数学（与 Python 同用例）
//   - readState / writeState round-trip
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_outpaint_"));
    const code = fs.readFileSync(path.join(__dirname, "..", "web", "sf_outpaint_core.js"), "utf8");
    fs.writeFileSync(path.join(tmpDir, "sf_outpaint_core.mjs"), code);
    const C = await import(path.join(tmpDir, "sf_outpaint_core.mjs"));

    // ── parseRatio ──
    check("parseRatio 正常", JSON.stringify(C.parseRatio("3:2")) === JSON.stringify([3, 2]));
    check("parseRatio 空格", JSON.stringify(C.parseRatio(" 4 : 3 ")) === JSON.stringify([4, 3]));
    check("parseRatio inf 拒绝", C.parseRatio("inf:2") === null);
    check("parseRatio nan 拒绝", C.parseRatio("nan:1") === null);
    check("parseRatio 零宽拒绝", C.parseRatio("3:0") === null);
    check("parseRatio 零高拒绝", C.parseRatio("0:2") === null);
    check("parseRatio 多冒号拒绝", C.parseRatio("3:2:5") === null);
    check("parseRatio 无冒号拒绝", C.parseRatio("32") === null);
    check("parseRatio 非法字符拒绝", C.parseRatio("16:9abc") === null);

    // ── anchorAxis ──
    check("anchorAxis 更宽 h", C.anchorAxis("21:9", 100, 50) === "h");
    check("anchorAxis 更高 v", C.anchorAxis("4:3", 100, 50) === "v");
    check("anchorAxis 已匹配 null", C.anchorAxis("2:1", 100, 50) === null);
    check("anchorAxis 源未知 null", C.anchorAxis("2:1", 0, 0) === null);

    // ── remapAnchor ──
    check("remapAnchor h->v", C.remapAnchor("left", "v") === "top");
    check("remapAnchor v->h", C.remapAnchor("top", "h") === "left");
    check("remapAnchor 同轴保留", C.remapAnchor("right", "h") === "right");
    check("remapAnchor 跨轴跨轴保留", C.remapAnchor("bottom", "v") === "bottom");
    check("remapAnchor 未知回退", C.remapAnchor("nope", "h") === "centre");

    // ── padsForRatio（与 Python 同用例）──
    const pfr = C.padsForRatio;
    check("ratio 更高 top", JSON.stringify(pfr(100, 50, "4:3", "top")) === JSON.stringify({ top: 25, bottom: 0, left: 0, right: 0 }));
    check("ratio 更高 bottom", JSON.stringify(pfr(100, 50, "4:3", "bottom")) === JSON.stringify({ top: 0, bottom: 25, left: 0, right: 0 }));
    check("ratio 更高 middle 平分", JSON.stringify(pfr(100, 50, "4:3", "middle")) === JSON.stringify({ top: 12, bottom: 13, left: 0, right: 0 }));
    check("ratio 更宽 left", JSON.stringify(pfr(100, 50, "21:9", "left")) === JSON.stringify({ top: 0, bottom: 0, left: 17, right: 0 }));
    check("ratio 更宽 right", JSON.stringify(pfr(100, 50, "21:9", "right")) === JSON.stringify({ top: 0, bottom: 0, left: 0, right: 17 }));
    check("ratio 更宽 centre 平分", JSON.stringify(pfr(100, 50, "21:9", "centre")) === JSON.stringify({ top: 0, bottom: 0, left: 8, right: 9 }));
    check("ratio 已匹配 -> 零", JSON.stringify(pfr(100, 100, "1:1", "centre")) === JSON.stringify({ top: 0, bottom: 0, left: 0, right: 0 }));
    // round-half-up 边界：999*1.5 = 1498.5 -> 1499，add = 500（Python 同断言）
    check("ratio round-half-up 边界", pfr(999, 999, "3:2", "right").right === 500);
    check("ratio 非法比例 -> 零", JSON.stringify(pfr(100, 50, "oops", "centre")) === JSON.stringify({ top: 0, bottom: 0, left: 0, right: 0 }));

    // ── padsForState ──
    check("padsForState ratio 走推导", C.padsForState({ mode: "ratio", ratio: "4:3", anchor: "top" }, 100, 50).top === 25);
    check("padsForState sides 夹紧 MAX_PAD", C.padsForState({ mode: "sides", left: 99999, top: -5 }, 100, 50).left === C.MAX_PAD);
    check("padsForState sides 负值归零", C.padsForState({ mode: "sides", top: -5 }, 100, 50).top === 0);
    check("padsForState 无状态默认 sides 零", JSON.stringify(C.padsForState(null, 100, 50)) === JSON.stringify({ top: 0, bottom: 0, left: 0, right: 0 }));

    // ── finalSize（与 Python 同用例）──
    // pad 128x64 -> max_mp 0.05 -> 324x162
    const f1 = C.finalSize(32, 64, { top: 0, bottom: 0, left: 0, right: 96 }, 0.05, 0);
    check("finalSize limit 缩放", f1.w === 324 && f1.h === 162);
    // snap 16：52x80 -> 48x80（limit 关时 pad 过程吸附）
    const f2 = C.finalSize(32, 64, { top: 16, bottom: 0, left: 20, right: 0 }, 0, 16);
    check("finalSize snap 吸附", f2.w === 48 && f2.h === 80);
    // limit 开时 pad 不吸附、max_mp 吸附：pad 46x72 + limit 1MP（factor 封顶 8）
    // -> 368x576 -> snap 64 -> 320x576
    const f3 = C.finalSize(32, 64, { top: 5, bottom: 3, left: 13, right: 1 }, 1, 64);
    check("finalSize limit 开时 snap 只在 max_mp", f3.w === 320 && f3.h === 576);
    // 无 pad 无 limit -> 原尺寸
    const f4 = C.finalSize(1024, 1024, { top: 0, bottom: 0, left: 0, right: 0 }, 0, 0);
    check("finalSize 直通", f4.w === 1024 && f4.h === 1024);
    // 极端 pad clamp 到 16384
    const f5 = C.finalSize(16000, 16000, { top: 10000, bottom: 0, left: 0, right: 0 }, 0, 0);
    check("finalSize clamp 上限", f5.w <= 16384 && f5.h <= 16384);
    // limit 非法值（超上限/负数）-> 当作关
    const f6 = C.finalSize(100, 50, { top: 0, bottom: 0, left: 0, right: 0 }, 999, 0);
    check("finalSize 非法 limit 当作关", f6.w === 100 && f6.h === 50);

    // ── readState / writeState ──
    const node = { properties: {} };
    const s1 = C.readState(node);
    check("readState 默认", s1.mode === "ratio" && s1.color === "#808080" && s1.limit === 0);
    C.writeState(node, { mode: "sides", left: 5 });
    const s2 = C.readState(node);
    check("writeState 写入", s2.mode === "sides" && s2.left === 5 && s2.ratio === "3:2");
    C.writeState(node, { color: "#000000" });
    const s3 = C.readState(node);
    check("writeState 合并保留", s3.mode === "sides" && s3.left === 5 && s3.color === "#000000");
    const broken = { properties: { outpaintState: "not json" } };
    check("readState 坏 JSON 回退默认", C.readState(broken).mode === "ratio");

    console.log();
    if (failures.length) { console.log(failures.length + " FAILURES:", failures); process.exit(1); }
    console.log("ALL PASS");
})().catch((e) => { console.error(e); process.exit(1); });
