// SF AuK 官方提示词模板数据测试（Node 直接运行：node tests/test_auk_presets_lib.mjs）
// 覆盖：组/条目结构完整性（非空、组名与组内 label 唯一）、占位符花括号配对、
// 默认值、groupNames/itemsOf/templateText 查询（含未知输入回退）、若干精确文本抽查。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_auk_presets_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_auk_presets_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const L = await import(pathToFileURL(tmpMjs).href);
const groups = L.AUK_PRESET_GROUPS;

check("组数 17（16 类 + 附）", groups.length === 17);
check("组名唯一且非空", new Set(groups.map((g) => g.name)).size === groups.length
      && groups.every((g) => typeof g.name === "string" && g.name.trim().length > 0));
check("每组条目非空且字段完整", groups.every((g) => Array.isArray(g.items) && g.items.length > 0
      && g.items.every((it) => typeof it.label === "string" && it.label.trim().length > 0
          && typeof it.text === "string" && it.text.trim().length > 0)));
check("组内 label 唯一", groups.every((g) => new Set(g.items.map((it) => it.label)).size === g.items.length));
check("占位符花括号配对", groups.every((g) => g.items.every((it) =>
      (it.text.match(/\{/g) || []).length === (it.text.match(/\}/g) || []).length)));

check("默认组/模板有效", L.DEFAULT_GROUP === groups[0].name && L.DEFAULT_TEMPLATE === groups[0].items[0].label);

check("groupNames 与数据一致", JSON.stringify(L.groupNames()) === JSON.stringify(groups.map((g) => g.name)));

// ── scope 过滤（generate = 全部；process = 长音频处理可用项）──
const processGroups = L.groupNames(L.SCOPE_PROCESS);
check("process 组数 11", processGroups.length === 11);
check("process 排除 TTS 与长模式不适用组", ["1. 参考音色 TTS", "2. 声音描述 TTS", "3. 语音内容编辑（替换、增添、删除）",
      "4. 歌词编辑", "14. 多人语音分离", "16. 按说话内容提取目标说话人"]
      .every((name) => !processGroups.includes(name)));
check("process 含增强/语速/分离/附", ["5. 音高调整", "6. 语速调整", "13. 语音增强（降噪、去混响、修复）",
      "15. 音乐人声分离", "附：官方音质改善演示"].every((name) => processGroups.includes(name)));
check("process 组 11 仅 Remove", JSON.stringify(L.itemsOf("11. 非语言声音编辑", L.SCOPE_PROCESS).map((i) => i.label))
      === JSON.stringify(["Remove · EN", "Remove · CN"]));
check("process 查 TTS 组为空", L.itemsOf("1. 参考音色 TTS", L.SCOPE_PROCESS).length === 0);
check("process 组 5 全量", L.itemsOf("5. 音高调整", L.SCOPE_PROCESS).length === 5);
check("默认 scope = generate", L.itemsOf("11. 非语言声音编辑").length === 5
      && L.itemsOf("1. 参考音色 TTS").length === 2);
check("itemsOf 命中", L.itemsOf("5. 音高调整").length === 5);
check("itemsOf 未知回退空数组", Array.isArray(L.itemsOf("不存在")) && L.itemsOf("不存在").length === 0);
check("templateText 命中", L.templateText("1. 参考音色 TTS", "EN & CN") === `Say the following with the same voice: "{text}"`);
check("templateText 未知回退空串", L.templateText("1. 参考音色 TTS", "不存在") === ""
      && L.templateText("不存在", "EN & CN") === "");

check("参考音色演示原句", L.templateText("1. 参考音色 TTS", "官方演示原句").includes("Ladies and gentlemen"));
check("内容编辑 10 个模板 + 演示", L.itemsOf("3. 语音内容编辑（替换、增添、删除）").length === 11
      && L.templateText("3. 语音内容编辑（替换、增添、删除）", "Replace · CN") === "把‘{原文}’改成‘{新文}’");
check("语音增强 8 模板 + 演示", L.itemsOf("13. 语音增强（降噪、去混响、修复）").length === 9
      && L.templateText("13. 语音增强（降噪、去混响、修复）", "Denoise · CN") === "请只去除背景噪声，保留其他内容，输出等长结果。");
check("附组只有演示原句", L.itemsOf("附：官方音质改善演示").length === 1
      && L.templateText("附：官方音质改善演示", "官方演示原句") === "Improve the audio quality and make it clearer");

const total = groups.reduce((sum, g) => sum + g.items.length, 0);
check("条目总数 > 60", total > 60);

if (failures.length) {
    console.log(`\n${failures.length} 项失败：`);
    for (const name of failures) console.log("  -", name);
    process.exit(1);
}
console.log("\nOK");
