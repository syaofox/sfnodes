// SF Universal Slider lib 纯函数测试（Node 直接运行：node tests/test_universal_slider_lib.mjs）
// 覆盖：pct/clamp/snap/fmtVal/castVal/calcValue（原版 1:1）/
// normalizeSliderSettings（min/max 对调、step 非法回退、int 档取整）/
// outputSlotForType（int→INT/int，float→FLOAT/float）。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_universal_slider_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_universal_slider_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

const approx = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

(async () => {
  const L = await import(pathToFileURL(tmpMjs).href);

  // ── pct/clamp/snap ──
  check("pct 中点 50", L.pct(0.5, 0, 1) === 50);
  check("pct 零区间回 0", L.pct(5, 1, 1) === 0);
  check("clamp 上钳", L.clamp(9, 0, 1) === 1);
  check("clamp 下钳", L.clamp(-2, 0, 1) === 0);
  check("snap 步进", approx(L.snap(0.36, 0, 0.1), 0.4));
  check("snap 零步长直通", L.snap(0.36, 0, 0) === 0.36);

  // ── fmtVal/castVal ──
  check("fmtVal int", L.fmtVal(1.5, true) === "2");
  check("fmtVal float 两位", L.fmtVal(0.5, false) === "0.50");
  check("castVal int 真 int", L.castVal(1.5, true) === 2 && Number.isInteger(L.castVal(1.5, true)));
  check("castVal float 真 float", L.castVal(0.5, false) === 0.5);

  // ── calcValue（snap→钳制→取整）──
  check("calcValue float 步进吸附", approx(L.calcValue(0.36, 0, 1, 0.1, false), 0.4));
  check("calcValue int 取整", L.calcValue(1.5, 0, 10, 1, true) === 2);
  check("calcValue 越界钳制", L.calcValue(99, 0, 1, 0.01, false) === 1);

  // ── normalizeSliderSettings ──
  {
    const n = L.normalizeSliderSettings({ type: "float", min: "0", max: "1", step: "0.01", label: "value" });
    check("归一 float 原样", n.type === "float" && n.min === 0 && n.max === 1 && n.step === 0.01 && n.label === "value");
  }
  {
    const n = L.normalizeSliderSettings({ type: "float", min: "5", max: "1", step: "0.1", label: "x" });
    check("min>max 对调", n.min === 1 && n.max === 5);
  }
  {
    const n = L.normalizeSliderSettings({ type: "float", min: "a", max: "b", step: "-1", label: "" });
    check("非法回退 float", n.min === 0 && n.max === 1 && n.step === 0.01 && n.label === "value");
  }
  {
    const n = L.normalizeSliderSettings({ type: "int", min: "0.2", max: "9.8", step: "0.5", label: "n" });
    check("int 档取整+步长≥1", n.min === 0 && n.max === 10 && n.step === 1);
  }
  {
    const n = L.normalizeSliderSettings({ type: "int", min: "0", max: "10", step: "x", label: "n" });
    check("int 非法步长回 1", n.step === 1);
  }

  // ── outputSlotForType ──
  check("int 档槽 INT/int", JSON.stringify(L.outputSlotForType("int")) === JSON.stringify({ type: "INT", name: "int" }));
  check("float 档槽 FLOAT/float", JSON.stringify(L.outputSlotForType("float")) === JSON.stringify({ type: "FLOAT", name: "float" }));
  check("未知档回 float", L.outputSlotForType("x").type === "FLOAT");

  if (failures.length) {
    console.log(`\n${failures.length} FAILED`);
    process.exit(1);
  }
  console.log("\nALL PASSED");
})();
