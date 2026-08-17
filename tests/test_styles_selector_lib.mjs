// SFStylesSelector lib 纯函数测试（Node 直接运行：node tests/test_styles_selector_lib.mjs）
// 覆盖：parseState/serializeState 容错与归一、resolveLabel 语言化、
// thumbnailOf 单值化、filterAndSort 搜索过滤/选中置顶/选中永不隐藏/raw 携带
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_styles_selector_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_styles_selector_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

const STYLES = [
  { name: "Fooocus Sharp", name_cn: "锐化", thumbnail: "https://x/a.jpg" },
  { name: "Fooocus Masterpiece", name_cn: "杰作", thumbnail: ["http://y/b.jpg", "/api/sfnodes/styles/image?path=s.jpg"] },
  { name: "Fooocus Enhance", name_cn: "增强", thumbnail: "/api/sfnodes/styles/image?path=z.jpg" },
  { name: "NoThumb", prompt: "p" },
];

(async () => {
  const L = await import(tmpUrl);

  // ── 状态解析/序列化 ──
  check("parseState 正常 JSON", JSON.stringify(L.parseState('["A","B"]')) === '["A","B"]');
  check("parseState 坏 JSON 容错", L.parseState("{bad").length === 0);
  check("parseState 空串容错", L.parseState("").length === 0);
  check("parseState 数组输入直接归一", JSON.stringify(L.parseState(["A", "", "B"])) === '["A","B"]');
  check("parseState 数字转字符串", JSON.stringify(L.parseState('[1, 2]')) === '["1","2"]');
  check("parseState 非数组 JSON 容错", L.parseState('"abc"').length === 0);
  check("serializeState 去重", L.serializeState(["A", "B", "A"]) === '["A","B"]');
  check("serializeState 过滤空", L.serializeState(["", "A"]) === '["A"]');

  // ── 语言化标签 ──
  check("中文环境优先 name_cn", L.resolveLabel("Fooocus Sharp", "锐化", true) === "锐化");
  check("中文环境无 name_cn 用原名", L.resolveLabel("Fooocus Sharp", "", true) === "Fooocus Sharp");
  check("英文环境用原名", L.resolveLabel("Fooocus Sharp", "锐化", false) === "Fooocus Sharp");

  // ── 缩略图单值化 ──
  check("thumbnailOf 字符串原样", L.thumbnailOf(STYLES[0]) === "https://x/a.jpg");
  check("thumbnailOf 数组取首项", L.thumbnailOf(STYLES[1]) === "http://y/b.jpg");
  check("thumbnailOf 缺省空串", L.thumbnailOf(STYLES[3]) === "");
  check("thumbnailOf 空对象", L.thumbnailOf({}) === "");

  // ── 远程 URL 判定 ──
  check("isRemoteThumb http", L.isRemoteThumb("http://x/y.jpg") === true);
  check("isRemoteThumb https", L.isRemoteThumb("https://x/y.jpg") === true);
  check("isRemoteThumb 本地路由", L.isRemoteThumb("/api/sfnodes/styles/image?path=a.jpg") === false);
  check("isRemoteThumb 非字符串", L.isRemoteThumb(undefined) === false);

  // ── filterAndSort：无查询按原序，选中置顶 ──
  let items = L.filterAndSort(STYLES, "", '["Fooocus Masterpiece"]', false);
  check("选中置顶", items[0].name === "Fooocus Masterpiece");
  check("置顶后其余保持原序", items.map((i) => i.name).slice(1).join(",") === "Fooocus Sharp,Fooocus Enhance,NoThumb");
  check("选中标记", items[0].selected === true && items[1].selected === false);
  check("raw 携带原条目", items[1].raw === STYLES[0] && items[1].raw.name_cn === "锐化");

  // ── filterAndSort：搜索过滤（匹配 name 或 label；选中项永不隐藏）──
  items = L.filterAndSort(STYLES, "sharp", "[]", false);
  check("搜索 name 命中", items.filter((i) => !i.hidden).map((i) => i.name).join(",") === "Fooocus Sharp");
  items = L.filterAndSort(STYLES, "杰作", "[]", true);
  check("搜索中文 label 命中（中文环境）", items.filter((i) => !i.hidden).map((i) => i.name).join(",") === "Fooocus Masterpiece");
  items = L.filterAndSort(STYLES, "杰作", "[]", false);
  check("英文环境搜中文不命中", items.filter((i) => !i.hidden).length === 0);
  items = L.filterAndSort(STYLES, "nomatch", '["NoThumb"]', false);
  check("选中项搜索时永不隐藏", items.find((i) => i.name === "NoThumb").hidden === false);
  items = L.filterAndSort(STYLES, "SHARP", "[]", false);
  check("搜索大小写不敏感", items.filter((i) => !i.hidden).length === 1);
  items = L.filterAndSort(STYLES, "   ", "[]", false);
  check("空白查询不隐藏", items.every((i) => !i.hidden));

  // ── 常量 ──
  check("STATE_WIDGET 契约", L.STATE_WIDGET === "SFStylesState");
  check("DOM_WIDGET 契约", L.DOM_WIDGET === "sf_styles_panel");
  check("STYLES_API 契约", L.STYLES_API === "/api/sfnodes/styles");

  console.log();
  if (failures.length) {
    console.log(`FAILED: ${failures.length}: ${failures.join(", ")}`);
    process.exit(1);
  }
  console.log("ALL PASS");
})();
