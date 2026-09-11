// SFCharacterSelect lib 纯函数测试（Node 直接运行：node tests/test_character_lib.mjs）
// 覆盖：parseState 新旧两态/serializeSelection/coerceSelection 收敛、
// entryImages/rolePrompt/displayPrompt、filterAndSort 搜索（含图 prompt）
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
  { name: "主角A", name_cn: "主角A", prompt: "hero",
    images: [
      { label: "脸", url: "/api/x/face.jpg", prompt: "p-face" },
      { label: "身", url: "https://h/body.jpg", prompt: "" },
    ] },
  { name: "Hero B", prompt: "a boy", images: [{ label: "脸", url: "f.jpg", prompt: "pb" }] },
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

  // ── 状态解析/序列化/收敛 ──
  check("parseState 新态", JSON.stringify(L.parseState('{"role":"A","shots":["x"]}')) === '{"role":"A","shots":["x"],"_legacy":false}');
  check("parseState 去重不过滤（保序）", L.parseState('{"role":"A","shots":["x","x"]}').shots.join(",") === "x,x");
  check("parseState 旧数组迁移标记", L.parseState('["A"]')._legacy === true && L.parseState('["A"]').role === "A");
  check("parseState 坏输入容错", L.parseState("{bad").role === "" && L.parseState(42).role === "");
  check("serializeSelection 去重保序", L.serializeSelection("A", ["y", "x", "y"]) === '{"role":"A","shots":["y","x"]}');
  check("coerceSelection 有效交集保序", JSON.stringify(L.coerceSelection(ROLES, '{"role":"主角A","shots":["身","脸","无"]}')) === '{"role":"主角A","shots":["脸","身"]}');
  check("coerceSelection 旧数组收敛首图", JSON.stringify(L.coerceSelection(ROLES, '["Hero B"]')) === '{"role":"Hero B","shots":["脸"]}');
  check("coerceSelection 空回落首图", JSON.stringify(L.coerceSelection(ROLES, "[]")) === '{"role":"主角A","shots":["脸"]}');
  check("coerceSelection 失效回落首图", JSON.stringify(L.coerceSelection(ROLES, '{"role":"X","shots":["脸"]}')) === '{"role":"主角A","shots":["脸"]}');
  check("coerceSelection 空库", JSON.stringify(L.coerceSelection([], "[]")) === '{"role":"","shots":[]}');

  // ── 条目与显示提示词 ──
  check("entryImages 顺序", L.entryImages(ROLES[0]).map((i) => i.label).join(",") === "脸,身");
  check("entryImages 缺省空", L.entryImages(ROLES[2]).length === 0);
  check("entryOf/rolePromptOf", L.entryOf(ROLES, "主角A") === ROLES[0] && L.rolePromptOf(ROLES[0]) === "hero");
  check("displayPrompt 拼接回落", L.displayPrompt(ROLES, '{"role":"主角A","shots":["脸","身"]}', "") === "p-face, hero");
  check("displayPrompt 草稿整体覆盖", L.displayPrompt(ROLES, '{"role":"主角A","shots":["脸"]}', "hand") === "hand");
  check("displayPrompt 空选回落首图", L.displayPrompt(ROLES, '{"role":"","shots":[]}', "") === "p-face");

  // ── filterAndSort ──
  let items = L.filterAndSort(ROLES, "", "主角A", false);
  check("选中置顶", items[0].name === "主角A");
  check("raw 携带原条目", items[0].raw === ROLES[0]);
  items = L.filterAndSort(ROLES, "p-face", "[]", false);
  check("搜索图 prompt 命中", items.filter((i) => !i.hidden).map((i) => i.name).join(",") === "主角A");
  items = L.filterAndSort(ROLES, "主角", "[]", false);
  check("搜索 name 命中", items.filter((i) => !i.hidden).length === 1);
  items = L.filterAndSort(ROLES, "nomatch", "NoShot", false);
  check("选中项搜索时永不隐藏", items.find((i) => i.name === "NoShot").hidden === false);
  items = L.filterAndSort(ROLES, "   ", "[]", false);
  check("空白查询不隐藏", items.every((i) => !i.hidden));

  // ── 远程判定 ──
  check("isRemoteThumb http", L.isRemoteThumb("https://h/body.jpg") === true);
  check("isRemoteThumb 本地路由", L.isRemoteThumb("/api/x/face.jpg") === false);

  console.log();
  if (failures.length) {
    console.log(`FAILED: ${failures.length}: ${failures.join(", ")}`);
    process.exit(1);
  }
  console.log("ALL PASS");
})();
