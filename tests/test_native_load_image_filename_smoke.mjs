// 原生 LoadImage filename 设置注册冒烟测试（node tests/test_native_load_image_filename_smoke.mjs）
// 覆盖 web/native_load_image_filename.js：扩展注册名、设置 id/type/默认值、
// init 幂等注册、onChange 重启提示 toast。
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const WEB = path.resolve(__dirname, "..", "web");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sf_loadimg_fn_"));

fs.writeFileSync(path.join(tmp, "stub_app.mjs"),
    "export const app = { registerExtension(cfg){ globalThis.__SF_CAPTURED__ = cfg; }, ui: { settings: { addSetting(s){ (globalThis.__SF_SETTINGS__ ||= []).push(s); } } } };\n");
fs.writeFileSync(path.join(tmp, "stub_common.mjs"),
    "export function sfToast(opts) { (globalThis.__SF_TOASTS__ ||= []).push(opts); }\n");

let src = fs.readFileSync(path.join(WEB, "native_load_image_filename.js"), "utf-8");
src = src.replace('import { app } from "/scripts/app.js";', 'import { app } from "./stub_app.mjs";');
src = src.replace('import { sfToast } from "./sf_common.js";', 'import { sfToast } from "./stub_common.mjs";');
fs.writeFileSync(path.join(tmp, "native_load_image_filename.mjs"), src);

const mod = await import(`file://${path.join(tmp, "native_load_image_filename.mjs")}`);
const captured = globalThis.__SF_CAPTURED__;

check("扩展已注册", captured && captured.name === "sfnodes.native_load_image_filename");
check("设置 id 常量", mod.LOAD_IMAGE_FILENAME_SETTING === "sfnodes.LoadImage.FilenameOutput.Enabled");

// init 注册设置
captured.init();
const settings = globalThis.__SF_SETTINGS__ || [];
check("init 注册设置一次", settings.length === 1);
const s = settings[0] || {};
check("设置 id 与段", s.id === "sfnodes.LoadImage.FilenameOutput.Enabled");
check("设置 boolean", s.type === "boolean");
check("设置默认开", s.defaultValue === true);
check("设置名含重启提示", typeof s.name === "string" && s.name.includes("restart"));

// 幂等
captured.init();
check("重复 init 幂等", (globalThis.__SF_SETTINGS__ || []).length === 1);

// onChange 重启提示
s.onChange?.(false, true);
const toasts = globalThis.__SF_TOASTS__ || [];
check("onChange 弹重启提示", toasts.length === 1 && String(toasts[0]?.detail || "").includes("重启"));

if (failures.length) {
    console.log("FAILED:", failures.length, failures);
    process.exit(1);
}
console.log("ALL PASS");
