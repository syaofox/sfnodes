// SFCharacterSelect lib 纯函数测试（Node 直接运行：node tests/test_character_lib.mjs）
// 覆盖：parseState/serializeSingle/firstSelected 单选收敛、resolveLabel 语言化、
// shotUrl 分镜取值、entryOf/displayPrompt 草稿优先、filterAndSort 单选置顶/搜索
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_character_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_character_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

const ROLES = [
  { name: "主角A", name_cn: "主角A", prompt: "a girl with black hair", face: "/api/sfnodes/characters/image?path=samples_id_chara/x/face.jpg", half: "samples_id_chara/x/half.jpg", full: "https://h/full.jpg" },
  { name: "Hero B", prompt: "a boy", face: "f.jpg" },
  { name: "NoShot", prompt: "p" },
];

(async () => {
  const L = await import(tmpUrl);

  // ── 常量契约 ──
  check("STATE_WIDGET 契约", L.STATE_WIDGET === "SFCharacterState");
  check("PROMPT_WIDGET 契约", L.PROMPT_WIDGET === "SFCharacterPrompt");
  check("DOM_WIDGET 契约", L.DOM_WIDGET === "sf_character_panel");
  check("CHARACTERS_API 契约", L.CHARACTERS_API === "/api/sfnodes/characters");
  check("LIB_PREFIX 契约", L.LIB_PREFIX === "character_");
  check("SHOTS 三分镜", JSON.stringify(L.SHOTS) === '["face","half","full"]');

  // ── 状态解析/单选收敛 ──
  check("parseState 正常 JSON", JSON.stringify(L.parseState('["A","B"]')) === '["A","B"]');
  check("parseState 坏 JSON 容错", L.parseState("{bad").length === 0);
  check("parseState 空串容错", L.parseState("").length === 0);
  check("serializeSingle 有名", L.serializeSingle("A") === '["A"]');
  check("serializeSingle 空名", L.serializeSingle("") === "[]");
  check("firstSelected 取首个", L.firstSelected('["A","B"]') === "A");
  check("firstSelected 空", L.firstSelected("[]") === "");
  check("coerceSelection 有效保留", L.coerceSelection(ROLES, '["Hero B"]') === "Hero B");
  check("coerceSelection 空回落首项", L.coerceSelection(ROLES, "[]") === "主角A");
  check("coerceSelection 失效回落首项", L.coerceSelection(ROLES, '["不存在"]') === "主角A");
  check("coerceSelection 空库回落空", L.coerceSelection([], '["A"]') === "");

  // ── 语言化/分镜取值 ──
  check("中文环境优先 name_cn", L.resolveLabel("Hero B", "英雄B", true) === "英雄B");
  check("英文环境用原名", L.resolveLabel("Hero B", "英雄B", false) === "Hero B");
  check("shotUrl 脸部", L.shotUrl(ROLES[0], "face") === "/api/sfnodes/characters/image?path=samples_id_chara/x/face.jpg");
  check("shotUrl 数组取首项", L.shotUrl({ face: ["a.jpg", "b.jpg"] }, "face") === "a.jpg");
  check("shotUrl 缺省空串", L.shotUrl(ROLES[2], "full") === "");
  check("isRemoteThumb http", L.isRemoteThumb("https://h/full.jpg") === true);
  check("isRemoteThumb 本地路由", L.isRemoteThumb("/api/sfnodes/characters/image?path=a.jpg") === false);

  // ── 条目与显示提示词 ──
  check("entryOf 命中", L.entryOf(ROLES, "主角A") === ROLES[0]);
  check("entryOf 未命中 null", L.entryOf(ROLES, "不存在") === null);
  check("displayPrompt 角色原词", L.displayPrompt(ROLES, "主角A", "") === "a girl with black hair");
  check("displayPrompt 草稿优先", L.displayPrompt(ROLES, "主角A", "hand") === "hand");
  check("displayPrompt 无选择空串", L.displayPrompt(ROLES, "", "") === "");

  // ── filterAndSort：单选置顶 + 搜索 ──
  let items = L.filterAndSort(ROLES, "", '["Hero B"]', false);
  check("选中置顶", items[0].name === "Hero B");
  check("选中标记唯一", items.filter((i) => i.selected).length === 1);
  check("raw 携带原条目", items[0].raw === ROLES[1]);
  items = L.filterAndSort(ROLES, "black hair", "[]", false);
  check("搜索 prompt 命中", items.filter((i) => !i.hidden).map((i) => i.name).join(",") === "主角A");
  items = L.filterAndSort(ROLES, "主角", "[]", false);
  check("搜索 name 命中", items.filter((i) => !i.hidden).map((i) => i.name).join(",") === "主角A");
  items = L.filterAndSort(ROLES, "nomatch", '["NoShot"]', false);
  check("选中项搜索时永不隐藏", items.find((i) => i.name === "NoShot").hidden === false);
  items = L.filterAndSort(ROLES, "   ", "[]", false);
  check("空白查询不隐藏", items.every((i) => !i.hidden));

  console.log();
  if (failures.length) {
    console.log(`FAILED: ${failures.length}: ${failures.join(", ")}`);
    process.exit(1);
  }
  console.log("ALL PASS");
})();
