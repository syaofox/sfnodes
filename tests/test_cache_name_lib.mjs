// sf_cache_name_lib 纯逻辑测试（Node 直接运行：node tests/test_cache_name_lib.mjs）
// 覆盖：NEW_LABEL / buildNameOptions（保序去重 + 当前值插入 + 新建入口）
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_cache_name_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_cache_name_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const L = await import(pathToFileURL(tmpMjs).href);
const NEW = L.NEW_LABEL;

check("NEW_LABEL 常量", NEW === "＋ 新建缓存…");

check("空列表只留新建入口", JSON.stringify(L.buildNameOptions([], "")) === JSON.stringify([NEW]));
check("已有名保序 + 新建入口",
    JSON.stringify(L.buildNameOptions(["b", "a"], "")) === JSON.stringify(["b", "a", NEW]));
check("当前值不在列表则插入队首",
    JSON.stringify(L.buildNameOptions(["b"], "drive")) === JSON.stringify(["drive", "b", NEW]));
check("当前值已在列表不重复",
    JSON.stringify(L.buildNameOptions(["b", "drive"], "drive")) === JSON.stringify(["b", "drive", NEW]));
check("空/新建入口值不入列表",
    JSON.stringify(L.buildNameOptions(["", NEW, "b"], NEW)) === JSON.stringify(["b", NEW]));
check("非字符串名归一化",
    JSON.stringify(L.buildNameOptions([1, 2], "")) === JSON.stringify(["1", "2", NEW]));

if (failures.length) {
    console.log("FAILED:", failures.length, failures);
    process.exit(1);
}
console.log("ALL PASS");
