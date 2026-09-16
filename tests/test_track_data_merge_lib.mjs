// SF Track Data Merge lib 纯函数测试（Node 直接运行：
// node tests/test_track_data_merge_lib.mjs）
// 覆盖：parseModes 容错 / getMode 默认 sub / setMode / toggleMode /
// serializeModes / collectSlotNames（节点属性与注入 JSON 的单源约定）。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_track_data_merge_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_track_data_merge_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(pathToFileURL(tmpMjs).href);

    check("常量", L.MODE_ADD === "add" && L.MODE_SUB === "sub"
        && L.DEFAULT_MODE === "sub" && L.SLOT_PREFIX === "track_");

    // parseModes
    check("parse 对象", JSON.stringify(L.parseModes({ track_1: "add" })) === '{"track_1":"add"}');
    check("parse JSON 字符串", JSON.stringify(L.parseModes('{"track_1":"add","track_2":"sub"}'))
        === '{"track_1":"add","track_2":"sub"}');
    check("parse 非法值归 sub", L.parseModes({ track_1: "xxx" }).track_1 === "sub");
    check("parse 垃圾字符串回退", JSON.stringify(L.parseModes("nope")) === "{}");
    check("parse null 回退", JSON.stringify(L.parseModes(null)) === "{}");
    check("parse 数组回退", JSON.stringify(L.parseModes([1, 2])) === "{}");

    // getMode 默认 sub
    check("getMode 缺省 sub", L.getMode({}, "track_1") === "sub");
    check("getMode 未知对象 sub", L.getMode(null, "track_1") === "sub");
    check("getMode add", L.getMode({ track_1: "add" }, "track_1") === "add");

    // setMode 不改原对象
    const base = { track_1: "sub" };
    const next = L.setMode(base, "track_2", "add");
    check("setMode 新对象", next.track_2 === "add" && base.track_2 === undefined);
    check("setMode 非法归 sub", L.setMode(base, "track_2", "wat").track_2 === "sub");

    // toggleMode
    check("toggle add->sub", L.toggleMode("add") === "sub");
    check("toggle sub->add", L.toggleMode("sub") === "add");
    check("toggle 未知->add", L.toggleMode("wat") === "add");

    // serializeModes
    check("serialize 归一", L.serializeModes({ track_1: "add", track_2: "bad" })
        === '{"track_1":"add","track_2":"sub"}');
    check("serialize 空", L.serializeModes(null) === "{}");

    // collectSlotNames：仅匹配 track_ 前缀且保持顺序
    const inputs = [
        { name: "track_data" }, { name: "track_1" }, { name: "add_1" },
        { name: "track_2" }, null, { name: "track_10" },
    ];
    check("collectSlotNames 过滤/保序", JSON.stringify(L.collectSlotNames(inputs))
        === '["track_1","track_2","track_10"]');
    check("collectSlotNames 空输入", JSON.stringify(L.collectSlotNames(undefined)) === "[]");

    // round-trip：序列化注入 → parseModes 还原
    const round = L.parseModes(L.serializeModes({ track_3: "add" }));
    check("序列化往返", round.track_3 === "add");

    console.log();
    if (failures.length) {
        console.log(`FAILED: ${failures.length} -> ${failures}`);
        process.exit(1);
    }
    console.log("ALL PASS");
})();
