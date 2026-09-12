// SF Boolean Switch lib 纯函数测试（Node 直接运行：node tests/test_boolean_switch_lib.mjs）
// 覆盖：TOGGLE 常量 / normalizeLabel / ellipsisText / trackX / toggleHit
// （命中判定与原版 toggleStartX=_w-72-10-20 等价：trackX(W)-14 === W-102）。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_boolean_switch_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_boolean_switch_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
  const L = await import(pathToFileURL(tmpMjs).href);

  // ── 常量 ──
  check("TOGGLE 几何与原版一致", L.TOGGLE.tw === 72 && L.TOGGLE.th === 28
    && L.TOGGLE.m === 10 && L.TOGGLE.xOff === 6 && L.TOGGLE.clickPad === 14);
  check("默认标签 value", L.DEFAULT_LABEL === "value");

  // ── normalizeLabel ──
  check("去首尾空", L.normalizeLabel("  开关  ") === "开关");
  check("空回落默认", L.normalizeLabel("   ") === "value");
  check("空串回落默认", L.normalizeLabel("") === "value");
  check("自定义回落", L.normalizeLabel("", "开关") === "开关");

  // ── ellipsisText ──
  const measure = (t) => ({ width: t.length * 10 });
  check("不超宽原样", L.ellipsisText("ab", 50, measure) === "ab");
  // "abcdef"62→ …本身占宽：abcd…=50>45 不行，abc…=40≤45 停
  check("超宽截断+…", L.ellipsisText("abcdef", 45, measure) === "abc…");
  check("单字超宽保底", L.ellipsisText("ab", 5, measure) === "a…");

  // ── trackX / toggleHit（与原版魔法数字等价）──
  check("trackX(W)=W-88", L.trackX(300) === 212);
  check("命中阈值=W-102（原版 toggleStartX 等价）", L.trackX(300) - L.TOGGLE.clickPad === 198);
  check("开关区命中", L.toggleHit(250, 300) === true);
  check("容差区命中（阈值右）", L.toggleHit(199, 300) === true);
  check("标签区未命中", L.toggleHit(100, 300) === false);
  check("阈值本身未命中（> 非 >=）", L.toggleHit(198, 300) === false);

  if (failures.length) {
    console.log(`\n${failures.length} FAILED`);
    process.exit(1);
  }
  console.log("\nALL PASSED");
})();
