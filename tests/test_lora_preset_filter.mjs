import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { pathToFileURL } from "node:url";

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "filter-"));
const src = fs.readFileSync("web/sf_lora_preset_filter.js", "utf8");
const mjsPath = path.join(tmp, "filter.mjs");
fs.writeFileSync(mjsPath, src);
const mod = await import(pathToFileURL(mjsPath).href);
const { filterPresets, highlight } = mod;

const presets = {
    "anime": { loras: [{ lora: "charA.safetensors" }, { lora: "style.safetensors" }], positive: "hello" },
    "realistic": { loras: [{ lora: "real.safetensors" }] },
    "empty": { loras: [] },
};

check("filter empty q returns all", Object.keys(filterPresets(presets, "")).length === 3);
check("filter by name", Object.keys(filterPresets(presets, "anime")).length === 1 && filterPresets(presets, "anime")["anime"]);
check("filter case insensitive", Object.keys(filterPresets(presets, "ANIME")).length === 1);
check("filter by lora", Object.keys(filterPresets(presets, "charA")).length === 1);
check("filter by lora partial", Object.keys(filterPresets(presets, "char")).length === 1);
check("filter no match", Object.keys(filterPresets(presets, "zzz")).length === 0);
check("filter trim", Object.keys(filterPresets(presets, "  anime  ")).length === 1);
check("highlight empty", highlight("hello", "") === "hello");
check("highlight found", highlight("anime", "nim").includes("<mark>"));
check("highlight case", highlight("Anime", "anime").includes("<mark>Anime</mark>") || highlight("Anime", "anime").includes("<mark>anime</mark>"));
check("highlight not found", highlight("hello", "zzz") === "hello");
check("highlight escape", highlight("<hi>", "hi").includes("&lt;"));

console.log("");
if (failures.length) { console.log(failures.length + " FAILURES", failures); process.exit(1); }
console.log("ALL PASS");
