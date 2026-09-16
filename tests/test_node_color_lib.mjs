// sf_node_color_lib 纯逻辑冒烟（Node 直接运行：node tests/test_node_color_lib.mjs）
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_node_color_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_node_color_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(tmpUrl);

    // ── normalizeHexColor ──
    check("6 位小写", L.normalizeHexColor("#a1b2c3") === "#a1b2c3");
    check("6 位大写归一", L.normalizeHexColor("#A1B2C3") === "#a1b2c3");
    check("无 # 前缀", L.normalizeHexColor("A1B2C3") === "#a1b2c3");
    check("3 位展开", L.normalizeHexColor("#abc") === "#aabbcc");
    check("3 位无 # 展开", L.normalizeHexColor("f0a") === "#ff00aa");
    check("空白容忍", L.normalizeHexColor("  #ABC  ") === "#aabbcc");
    check("非法字符 null", L.normalizeHexColor("#gggggg") === null);
    check("长度不符 null", L.normalizeHexColor("#abcd") === null);
    check("8 位带 alpha null", L.normalizeHexColor("#a1b2c3d4") === null);
    check("空串 null", L.normalizeHexColor("") === null);
    check("非字符串 null", L.normalizeHexColor(null) === null && L.normalizeHexColor(123) === null);

    // ── pushRecent ──
    check("空列表推入", JSON.stringify(L.pushRecent([], "#112233")) === '["#112233"]');
    check("非法色不推入", JSON.stringify(L.pushRecent([], "xx")) === "[]");
    check("最新在前", JSON.stringify(L.pushRecent(["#111111"], "#222222")) === '["#222222","#111111"]');
    check("去重并前置", JSON.stringify(L.pushRecent(["#111111", "#222222"], "#111111")) === '["#111111","#222222"]');
    check("去重忽略大小写", L.pushRecent(["#AABBCC"], "#aabbcc").length === 1);
    check("上限截断", L.pushRecent(["#000001", "#000002", "#000003"], "#000004", 3).length === 3
        && L.pushRecent(["#000001", "#000002", "#000003"], "#000004", 3)[0] === "#000004");
    check("不改原数组", (() => { const a = ["#111111"]; L.pushRecent(a, "#222222"); return a.length === 1; })());

    // ── applyNodeColor / resetNodeColor / readNodeColor ──
    const bare = { color: "#000000", bgcolor: "#000000" };
    check("apply 返回计数", L.applyNodeColor([bare], "#123456") === 1);
    check("apply 同时写标题与节点体", bare.color === "#123456" && bare.bgcolor === "#123456");
    check("apply 非法色返回 0 不改", L.applyNodeColor([bare], "nope") === 0 && bare.color === "#123456");
    check("apply 跳过空节点", L.applyNodeColor([null, bare], "#abcdef") === 1);
    check("apply 3 位展开写入", (() => { const n = {}; L.applyNodeColor([n], "#abc"); return n.color === "#aabbcc"; })());

    const withSetter = {
        color: "#111111", bgcolor: "#111111", _last: "unset",
        setColorOption(o) { this._last = o; if (o == null) { this.color = undefined; this.bgcolor = undefined; } else { this.color = o.color; this.bgcolor = o.bgcolor; } },
    };
    check("reset 走 setColorOption(null)", L.resetNodeColor([withSetter]) === 1 && withSetter._last === null);
    check("reset 清空颜色", withSetter.color === undefined && withSetter.bgcolor === undefined);
    const bare2 = { color: "#123456", bgcolor: "#123456" };
    check("reset 无 setter 置 undefined", L.resetNodeColor([bare2]) === 1 && bare2.color === undefined && bare2.bgcolor === undefined);

    check("read 优先 bgcolor", L.readNodeColor([{ color: "#111111", bgcolor: "#222222" }]) === "#222222");
    check("read 回退 color", L.readNodeColor([{ color: "#111111", bgcolor: "zzz" }]) === "#111111");
    check("read 无有效色 null", L.readNodeColor([{ color: "zzz", bgcolor: "zzz" }]) === null);
    check("read 空列表 null", L.readNodeColor([]) === null);

    if (failures.length) {
        console.log("FAILURES:", failures.join(", "));
        process.exit(1);
    }
    console.log("ALL PASS");
})();
