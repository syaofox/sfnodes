// SF LoRA 浏览器 lib 纯函数测试（Node 直接运行：node tests/test_lora_browser_lib.mjs）
// 覆盖：splitName 路径拆分 / filterLoras 搜索过滤 / folderContents 文件夹层级
// （立即子目录 + 当前层文件）/ breadcrumbParts 面包屑分段 / groupLoras 平铺分组。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_lora_browser_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_lora_browser_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

const LIST = [
  "a.safetensors",
  "characters/xiangling.safetensors",
  "characters/hutao.safetensors",
  "style/watercolor.safetensors",
  "style/lineart/ink.safetensors",
  "dummy.txt",
];

(async () => {
  const L = await import(tmpUrl);

  // ── splitName ──
  check("splitName 根文件", JSON.stringify(L.splitName("a.safetensors")) === JSON.stringify({ folder: "", base: "a.safetensors" }));
  check("splitName 子目录", L.splitName("characters/xiangling.safetensors").folder === "characters");
  check("splitName 多级子目录", L.splitName("style/lineart/ink.safetensors").folder === "style/lineart");
  check("splitName 反斜杠归一", L.splitName("style\\watercolor.safetensors").folder === "style");
  check("splitName 空串", L.splitName("").base === "");
  check("splitName 未定义", L.splitName(undefined).base === "");

  // ── filterLoras ──
  check("filter 空查询原样", L.filterLoras(LIST, "").length === LIST.length);
  check("filter 大小写不敏感", L.filterLoras(LIST, "XIANG").length === 1);
  check("filter 文件名主体命中（不带扩展名）", L.filterLoras(LIST, "hutao").length === 1);
  check("filter 全名命中", L.filterLoras(LIST, "style/watercolor").length === 1);
  check("filter 自定义词命中文件主体去扩展名", L.filterLoras(LIST, "dummy").length === 1);
  check("filter 无匹配", L.filterLoras(LIST, "zzz").length === 0);
  check("filter 空白查询", L.filterLoras(LIST, "   ").length === LIST.length);
  check("filter 空列表", L.filterLoras([], "a").length === 0);
  check("filter 非数组容忍", L.filterLoras(undefined, "a").length === 0);

  // ── folderContents（文件夹层级下钻模型）──
  const root = L.folderContents(LIST, "");
  check("fc 根层文件夹（立即子目录去重排序）", root.folders.join(",") === "characters,style");
  check("fc 根层文件（只有直接文件）", root.files.join(",") === "a.safetensors,dummy.txt");
  const chars = L.folderContents(LIST, "characters");
  check("fc 进入 characters", JSON.stringify({ f: chars.folders, files: chars.files }) === JSON.stringify({ f: [], files: ["characters/hutao.safetensors", "characters/xiangling.safetensors"] }));
  const style = L.folderContents(LIST, "style");
  check("fc style 立即子目录", style.folders.join(",") === "lineart");
  check("fc style 当前层文件", style.files.join(",") === "style/watercolor.safetensors");
  check("fc 子目录内层", L.folderContents(LIST, "style/lineart").files.join(",") === "style/lineart/ink.safetensors");
  check("fc 尾部斜杠归一", L.folderContents(LIST, "style/").folders.join(",") === "lineart");
  check("fc 不存在目录为空", JSON.stringify(L.folderContents(LIST, "nope")) === JSON.stringify({ folders: [], files: [] }));
  check("fc 空列表", L.folderContents([], "").files.length === 0);
  check("fc 未定义容忍", L.folderContents(undefined, "").folders.length === 0);

  // ── breadcrumbParts ──
  check("bc 根", L.breadcrumbParts("").length === 0);
  check("bc 单级", JSON.stringify(L.breadcrumbParts("characters")) === JSON.stringify(["characters"]));
  check("bc 多级", JSON.stringify(L.breadcrumbParts("style/lineart")) === JSON.stringify(["style", "lineart"]));
  check("bc 尾部斜杠归一", JSON.stringify(L.breadcrumbParts("style/")) === JSON.stringify(["style"]));
  check("bc 未定义", L.breadcrumbParts(undefined).length === 0);

  // ── groupLoras（平铺归档式展示，保留兼容）──
  const groups = L.groupLoras(LIST);
  check("group 根在前", groups[0].folder === "");
  check("group 组数", groups.length === 4);
  check("group 组名排序", groups.map((g) => g.folder).join(",") === ",characters,style,style/lineart");
  check("group 空列表", L.groupLoras([]).length === 0);

  // ── sortWithinGroup ──
  const sorted = L.sortWithinGroup(["b.safetensors", "A.safetensors", "c.safetensors"]);
  check("sort 字典序（大小写不敏感）", sorted.join(",") === "A.safetensors,b.safetensors,c.safetensors");
  check("sort 不修改原数组", (() => { const a = ["b", "a"]; L.sortWithinGroup(a); return a.join(",") === "b,a"; })());

  // 汇总
  if (failures.length) {
    console.log("FAILURES:", failures.join(", "));
    process.exit(1);
  }
  console.log("ALL PASS");
})();
