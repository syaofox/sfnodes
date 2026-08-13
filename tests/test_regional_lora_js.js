// SFRegionalLoRA 前端 lib 冒烟测试（Node 直接运行：node tests/test_regional_lora_js.js）
// 无 DOM 依赖的公共库复制为 .mjs 后直跑，覆盖：
//   - defaultRegion/defaultRegions/defaultRegionsJson（与 Python default_regions_json 契约一致）
//   - readRegions/writeRegions（隐藏 SFRegionsJson widget 真源读写）
//   - normalizeRect/applyResize（8 向 resize 数学）
//   - hitTestRegions（8 手柄 + move 命中）
//   - shortName（文件名净化）
//   - ensureLoraList（单数组实例 push 填充，无引用竞态）
import { copyFileSync, mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join, dirname } from "path";
import { pathToFileURL } from "url";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const tmp = mkdtempSync(join(tmpdir(), "sf-regional-lora-"));
copyFileSync(join(root, "web", "sf_regional_lora_lib.js"), join(tmp, "lib.mjs"));
const lib = await import(pathToFileURL(join(tmp, "lib.mjs")).href);

let failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

// ── default regions ───────────────────────────────────────────────────────
const defs = lib.defaultRegions(2);
check("default: 2 regions", defs.length === 2);
check("default: equal columns", defs[0].x === 0 && defs[1].x === 0.5 && defs[0].w === 0.5);
check("default: lora None / strength 1.0 / enable", defs.every((r) => r.lora === "None"
  && r.strength === 1.0 && r.enable === true && r.h === 1.0));
const json = lib.defaultRegionsJson(2);
check("default: json parses back", lib.readRegions({ widgets: [{ name: "SFRegionsJson", value: json }] }).length === 2);

// ── read/write regions（隐藏 widget 真源）────────────────────────────────
const fakeNode = { widgets: [{ name: "SFRegionsJson", value: "[]", inputEl: { value: "" } }] };
lib.writeRegions(fakeNode, [{ lora: "a.safetensors", strength: 1.2, enable: true, x: 0, y: 0, w: 0.5, h: 1 }]);
check("write: widget value updated", fakeNode.widgets[0].value.includes("a.safetensors"));
check("write: inputEl synced", fakeNode.widgets[0].inputEl.value === fakeNode.widgets[0].value);
check("read: roundtrip", lib.readRegions(fakeNode)[0].strength === 1.2);
check("read: garbage json -> []", lib.readRegions({ widgets: [{ name: "SFRegionsJson", value: "{oops" }] }).length === 0);
check("read: missing widget -> []", lib.readRegions({ widgets: [] }).length === 0);

// ── normalizeRect / applyResize ───────────────────────────────────────────
const nr = lib.normalizeRect({ x: -0.2, y: 0.1, w: 0.5, h: 0.5 });
check("normalize: negative x clamped (w kept)", nr.x === 0 && nr.w === 0.5);
const inv = lib.normalizeRect({ x: 0.6, y: 0.6, w: -0.4, h: 0.3 });
check("normalize: inverted drag flipped", Math.abs(inv.x - 0.2) < 1e-9 && Math.abs(inv.w - 0.4) < 1e-9);
const res = lib.applyResize("resize-br", { x: 0, y: 0, w: 0.5, h: 0.5 }, 0.1, 0.2);
check("resize: br grows", Math.abs(res.w - 0.6) < 1e-9 && Math.abs(res.h - 0.7) < 1e-9);
const resTl = lib.applyResize("resize-tl", { x: 0.5, y: 0.5, w: 0.5, h: 0.5 }, 0.2, 0.1);
check("resize: tl shrinks", Math.abs(resTl.x - 0.7) < 1e-9 && Math.abs(resTl.y - 0.6) < 1e-9);
const resT = lib.applyResize("resize-t", { x: 0, y: 0.5, w: 1, h: 0.5 }, 0, 0.3);
check("resize: t shrinks from top", Math.abs(resT.y - 0.8) < 1e-9 && Math.abs(resT.h - 0.2) < 1e-9);

// ── hitTestRegions ────────────────────────────────────────────────────────
// region2 的 y 偏移到 0.6，避免与 region0 的右下角重叠（倒序命中歧义）
const regs = [
  { x: 0, y: 0, w: 0.5, h: 0.5 },
  { x: 0.5, y: 0.6, w: 0.5, h: 0.5 },
];
check("hit: move inside region0", lib.hitTestRegions(regs, 0.25, 0.25, 200, 200, 12).mode === "move");
check("hit: inside region2", lib.hitTestRegions(regs, 0.75, 0.75, 200, 200, 12).i === 1);
check("hit: br handle of region0", lib.hitTestRegions(regs, 0.5, 0.5, 200, 200, 12).mode === "resize-br");
check("hit: tr handle", lib.hitTestRegions(regs, 0.5, 0.0, 200, 200, 12).mode === "resize-tr");
check("hit: b edge", lib.hitTestRegions(regs, 0.25, 0.5, 200, 200, 12).mode === "resize-b");
check("hit: l edge", lib.hitTestRegions(regs, 0.0, 0.25, 200, 200, 12).mode === "resize-l");
check("hit: miss -> null", lib.hitTestRegions(regs, 0.2, 0.9, 200, 200, 12) === null);

// ── shortName ─────────────────────────────────────────────────────────────
check("short: basename + ext strip", lib.shortName("sub/dir/char_1.safetensors") === "char_1");
check("short: None -> empty", lib.shortName("None") === "" && lib.shortName("") === "");
check("short: long truncated", lib.shortName("averyverylongcharactername.safetensors").length <= 14);

// ── ensureLoraList：单数组实例、push 填充（无引用竞态）────────────────────
const listA = lib.getLoraList();
check("lora: starts with None only", listA.length === 1 && listA[0] === "None");
const p1 = lib.ensureLoraList(async () => ["lora1.safetensors", "lora2.safetensors"]);
const p2 = lib.ensureLoraList(async () => ["lora3.safetensors"]);
await Promise.all([p1, p2]);
const listB = lib.getLoraList();
check("lora: single instance (no re-assign)", listA === listB);
check("lora: first loader fills", listB.includes("lora1.safetensors") && listB.includes("lora2.safetensors"));
check("lora: second loader ignored (dedup guard)", !listB.includes("lora3.safetensors"));
check("lora: no duplicates", new Set(listB).size === listB.length);
const listC = lib.getLoraList();
listC.push("later.safetensors");
check("lora: in-place mutation visible via getLoraList", lib.getLoraList().includes("later.safetensors"));

// ── safeRebuildRows：工作流加载值恢复时序的值守卫重建 ────────────────────
// 场景：onNodeCreated 时 widget 是默认 JSON（__rc_lastJson 被记录），configure
// 恢复真实值（有 LoRA）后，多时机调用应重建一次；之后值未变则全部跳过。
const restoreNode = {
  widgets: [{ name: "SFRegionsJson", value: lib.defaultRegionsJson(2) }],
};
let rebuildCount = 0;
const bump = () => { rebuildCount++; };
check("safeRebuild: first call rebuilds", lib.safeRebuildRows(restoreNode, bump) === true
  && rebuildCount === 1);
check("safeRebuild: unchanged value skips", lib.safeRebuildRows(restoreNode, bump) === false
  && rebuildCount === 1);
check("safeRebuild: repeated timings all skip", lib.safeRebuildRows(restoreNode, bump) === false);
// configure 恢复真实值（含 LoRA）
restoreNode.widgets[0].value = JSON.stringify([{ lora: "a.safetensors", strength: 1.0, enable: true, x: 0, y: 0, w: 0.5, h: 1 }]);
check("safeRebuild: value restored -> rebuilds once", lib.safeRebuildRows(restoreNode, bump) === true
  && rebuildCount === 2);
check("safeRebuild: stable again", lib.safeRebuildRows(restoreNode, bump) === false && rebuildCount === 2);
// 值变回（用户编辑）后再次重建
restoreNode.widgets[0].value = "[]";
check("safeRebuild: user edit rebuilds", lib.safeRebuildRows(restoreNode, bump) === true && rebuildCount === 3);
// 缺 widget（异常）不重建不抛
delete restoreNode.widgets[0].name;
check("safeRebuild: missing widget -> false", lib.safeRebuildRows(restoreNode, bump) === false);

// ── bindRegionValue：行控件 value 活绑定 regions JSON（根治显示时序）──────
const bindNode = { widgets: [{ name: "SFRegionsJson", value: JSON.stringify([
  { lora: "a.safetensors", strength: 1.2, enable: true, x: 0, y: 0, w: 0.5, h: 1 },
  { lora: "None", strength: 0.8, enable: false, x: 0.5, y: 0, w: 0.5, h: 1 }]) }] };
const wLora = lib.bindRegionValue(bindNode, { value: "None" }, 0, "lora", "None");
const wStr = lib.bindRegionValue(bindNode, { value: 1.0 }, 1, "strength", 1.0);
const wEn = lib.bindRegionValue(bindNode, { value: true }, 1, "enable", true);
check("bind: combo reads json", wLora.value === "a.safetensors");
check("bind: number reads json as number", wStr.value === 0.8 && typeof wStr.value === "number");
check("bind: toggle reads json bool", wEn.value === false);
// JSON 更新（模拟 configure 值恢复 / 用户编辑）→ getter 立即反映，无需重建
bindNode.widgets[0].value = JSON.stringify([{ lora: "b.safetensors", strength: 2.0, enable: true, x: 0, y: 0, w: 0.5, h: 1 }]);
check("bind: json restore reflected immediately", wLora.value === "b.safetensors");
check("bind: region removed -> fallback to initial", wStr.value === 1.0);
// 外部赋 "None"（模拟恢复路径的陈旧写回）→ getter 仍从 JSON 读，显示不坏
wLora.value = "None";
check("bind: stale external write ignored", wLora.value === "b.safetensors");
// 用户选择（setter 阶段值）→ callback 写 JSON → getter 一致
wLora.value = "c.safetensors";
bindNode.widgets[0].value = JSON.stringify([{ lora: "c.safetensors", strength: 2.0, enable: true, x: 0, y: 0, w: 0.5, h: 1 }]);
check("bind: user pick reflected", wLora.value === "c.safetensors");
// idx 越界 → 回退 initial
const wOut = lib.bindRegionValue(bindNode, { value: "None" }, 9, "lora", "None");
check("bind: out-of-range falls back", wOut.value === "None");
// readRegions 缓存：同值不重复 parse
let parseCount = 0;
const cacheNode = { widgets: [{ name: "SFRegionsJson", value: "[]" }] };
const origParse = JSON.parse;
JSON.parse = (s) => { parseCount++; return origParse(s); };
lib.readRegions(cacheNode); lib.readRegions(cacheNode); lib.readRegions(cacheNode);
JSON.parse = origParse;
check("cache: same value parsed once", parseCount === 1);

console.log();
if (failures.length) {
  console.log(failures.length + " FAILURES:", failures);
  process.exit(1);
}
console.log("ALL PASS");
