// SF Note lib 纯函数测试（Node 直接运行：node tests/test_note_lib.mjs）
// 覆盖（原版 GoohaiNote.js 引擎逐义移植）：parseSegments 链接三式 /
// buildCharList / wrapCharList（英文断词、CJK 整字换行、行首禁则回拉）/
// layoutText（\n 转义/回车归一）/ normalizeStyle 钳制 / parseStyle 容错 /
// hexToRgba / strokeColor / groupLine。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_note_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_note_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

// 等宽测量（1/字符）：换行位置完全确定，便于锁定引擎语义
const measure = (s) => s.length;
const str = (line) => line.map((c) => c.ch).join("");

(async () => {
  const L = await import(pathToFileURL(tmpMjs).href);

  // ── parseSegments ──
  {
    const segs = L.parseSegments("看[[https://x.com/a]]好");
    check("[[链接]] 切分", segs.length === 3 && segs[1].type === "link"
      && segs[1].url === "https://x.com/a" && segs[1].content === "https://x.com/a");
  }
  {
    const segs = L.parseSegments("去 http://a.bc 看");
    check("裸 http 链接", segs.length === 3 && segs[1].type === "link" && segs[1].url === "http://a.bc");
  }
  {
    const segs = L.parseSegments("上 www.abc.com 看");
    check("www 链接", segs.length === 3 && segs[1].type === "link" && segs[1].url === "www.abc.com");
  }
  {
    const segs = L.parseSegments("纯文本无链接");
    check("纯文本单段", segs.length === 1 && segs[0].type === "text");
  }

  // ── buildCharList ──
  {
    const chars = L.buildCharList([{ type: "link", content: "ab", url: "u" }]);
    check("逐字携带 type/url", chars.length === 2 && chars[0].type === "link" && chars[1].url === "u");
  }

  // ── wrapCharList：英文空格断词 ──
  {
    const lines = L.wrapCharList(measure, L.buildCharList(L.parseSegments("abc def ghi")), 5)
      .map(str);
    check("英文不断词中切分", JSON.stringify(lines) === '["abc ","def ","ghi"]');
  }

  // ── wrapCharList：CJK 整字换行 ──
  {
    const lines = L.wrapCharList(measure, L.buildCharList(L.parseSegments("中文测试换行")), 3)
      .map(str);
    check("CJK 按字换行", JSON.stringify(lines) === '["中文测","试换行"]');
  }

  // ── wrapCharList：行首禁则回拉（，不能居行首）──
  {
    const lines = L.wrapCharList(measure, L.buildCharList(L.parseSegments("ab，cd")), 2);
    const joined = lines.map(str).join("");
    check("禁则回拉无丢字", joined === "ab，cd");
    check("行首无禁则标点", lines.every((l) => l.length === 0 || ![",", "，", "。"].includes(l[0].ch)));
    check("回拉形态", JSON.stringify(lines.map(str)) === '["a","b，","cd"]');
  }

  // ── wrapCharList：显式换行 ──
  {
    const lines = L.wrapCharList(measure, L.buildCharList(L.parseSegments("a\nb")), 10).map(str);
    check("显式换行分行", JSON.stringify(lines) === '["a","b"]');
  }

  // ── layoutText ──
  {
    check("\\n 转义换行", JSON.stringify(L.layoutText("a\\nb", 10, measure).map(str)) === '["a","b"]');
    check("\\r\\n 归一", JSON.stringify(L.layoutText("a\r\nb", 10, measure).map(str)) === '["a","b"]');
    const lines = L.layoutText("看 https://x.com 好", 100, measure);
    const hasLink = lines.some((l) => l.some((c) => c.type === "link"));
    check("排版保留链接标记", hasLink);
  }

  // ── normalizeStyle / parseStyle ──
  {
    const s = L.normalizeStyle({ fontSize: 999, backgroundAlpha: 5, textAlign: "nope", fontColor: "red" });
    check("数值钳制", s.fontSize === 200 && s.backgroundAlpha === 1);
    check("非法枚举/颜色回默认", s.textAlign === "center" && s.fontColor === "#C8C8C8");
  }
  {
    const s = L.normalizeStyle({ textAlign: "right", stroke: 0, locked: 1 });
    check("合法值保留+布尔归一", s.textAlign === "right" && s.stroke === false && s.locked === true);
  }
  {
    const s = L.parseStyle("not-json{");
    check("非法 JSON 回默认", s.fontSize === L.DEFAULT_STYLE.fontSize);
  }
  {
    const s = L.parseStyle('{"fontSize":30}');
    check("部分字段合并默认", s.fontSize === 30 && s.padding === L.DEFAULT_STYLE.padding);
  }

  // ── hexToRgba / strokeColor ──
  check("hexToRgba", L.hexToRgba("#1B4669", 0.25) === "rgba(27,70,105,0.25)");
  check("hexToRgba 非法回退", L.hexToRgba("red", 0.5) === "rgba(120,120,120,0.5)");
  check("strokeColor +80 钳制", L.strokeColor("#1B4669", 0.25) === "rgba(107,150,185,0.25)");

  // ── groupLine ──
  {
    const groups = L.groupLine(L.buildCharList(L.parseSegments("ahttps://x.com b")));
    // "a" + link + " b" → text/link/text 三组（注意空格在链接后）
    check("行内分组", groups.length === 3 && groups[1].type === "link" && groups[2].text === " b");
  }

  if (failures.length) {
    console.log(`\n${failures.length} FAILED`);
    process.exit(1);
  }
  console.log("\nALL PASSED");
})();
