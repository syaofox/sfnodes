// SFIDClothingSelector lib 纯函数测试（Node 直接运行：node tests/test_id_clothing_lib.mjs）
// 覆盖：parseState/serializeSingle/firstSelected 单选收敛、resolveLabel 语言化、
// thumbnailOf 单值化、entryOf/displayPrompt 草稿优先、filterAndSort 单选置顶/搜索
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_id_clothing_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_id_clothing_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

const STYLES = [
  { name: "西装", name_cn: "西装", thumbnail: "/api/sfnodes/styles/image?path=samples/suit.jpg", prompt: "suit prompt" },
  { name: "马甲", name_cn: "马甲", thumbnail: "samples/vest.jpg", prompt: "vest" },
  { name: "Sharp", name_cn: "锐化", thumbnail: "https://x/a.jpg" },
  { name: "NoThumb", prompt: "p" },
];

(async () => {
  const L = await import(tmpUrl);

  // ── 常量契约 ──
  check("STATE_WIDGET 契约", L.STATE_WIDGET === "SFIDClothingState");
  check("PROMPT_WIDGET 契约", L.PROMPT_WIDGET === "SFIDClothingPrompt");
  check("DOM_WIDGET 契约", L.DOM_WIDGET === "sf_id_clothing_panel");
  check("STYLES_API 复用 styles", L.STYLES_API === "/api/sfnodes/styles");
  check("LIB_PREFIX 契约", L.LIB_PREFIX === "id_");

  // ── 状态解析/单选收敛 ──
  check("parseState 正常 JSON", JSON.stringify(L.parseState('["A","B"]')) === '["A","B"]');
  check("parseState 坏 JSON 容错", L.parseState("{bad").length === 0);
  check("parseState 空串容错", L.parseState("").length === 0);
  check("parseState 数组输入归一", JSON.stringify(L.parseState(["A", "", "B"])) === '["A","B"]');
  check("serializeSingle 有名", L.serializeSingle("A") === '["A"]');
  check("serializeSingle 空名", L.serializeSingle("") === "[]");
  check("firstSelected 取首个", L.firstSelected('["A","B"]') === "A");
  check("firstSelected 空", L.firstSelected("[]") === "");
  check("firstSelected 坏输入", L.firstSelected("{bad") === "");

  // ── 语言化/缩略图 ──
  check("中文环境优先 name_cn", L.resolveLabel("Sharp", "锐化", true) === "锐化");
  check("英文环境用原名", L.resolveLabel("Sharp", "锐化", false) === "Sharp");
  check("thumbnailOf 字符串原样", L.thumbnailOf(STYLES[2]) === "https://x/a.jpg");
  check("thumbnailOf 缺省空串", L.thumbnailOf(STYLES[3]) === "");
  check("isRemoteThumb http", L.isRemoteThumb("http://x/y.jpg") === true);
  check("isRemoteThumb 本地路由", L.isRemoteThumb("/api/sfnodes/styles/image?path=a.jpg") === false);

  // ── 条目与显示提示词 ──
  check("entryOf 命中", L.entryOf(STYLES, "西装") === STYLES[0]);
  check("entryOf 未命中 null", L.entryOf(STYLES, "不存在") === null);
  check("displayPrompt 模板原词", L.displayPrompt(STYLES, "西装", "") === "suit prompt");
  check("displayPrompt 草稿优先", L.displayPrompt(STYLES, "西装", "hand") === "hand");
  check("displayPrompt 无选择空串", L.displayPrompt(STYLES, "", "") === "");

  // ── filterAndSort：单选置顶 + 搜索 ──
  let items = L.filterAndSort(STYLES, "", '["马甲"]', false);
  check("选中置顶", items[0].name === "马甲");
  check("选中标记唯一", items.filter((i) => i.selected).length === 1);
  check("raw 携带原条目", items[0].raw === STYLES[1]);
  items = L.filterAndSort(STYLES, "", "马甲", false);
  check("单名入参同样收敛", items[0].name === "马甲");
  items = L.filterAndSort(STYLES, "suit", "[]", false);
  check("搜索 prompt 命中（服装英文描述在 prompt 里）", items.filter((i) => !i.hidden).map((i) => i.name).join(",") === "西装");
  items = L.filterAndSort(STYLES, "西装", "[]", false);
  check("搜索 name 命中", items.filter((i) => !i.hidden).map((i) => i.name).join(",") === "西装");
  items = L.filterAndSort(STYLES, "nomatch", '["NoThumb"]', false);
  check("选中项搜索时永不隐藏", items.find((i) => i.name === "NoThumb").hidden === false);
  items = L.filterAndSort(STYLES, "   ", "[]", false);
  check("空白查询不隐藏", items.every((i) => !i.hidden));

  console.log();
  if (failures.length) {
    console.log(`FAILED: ${failures.length}: ${failures.join(", ")}`);
    process.exit(1);
  }
  console.log("ALL PASS");
})();
