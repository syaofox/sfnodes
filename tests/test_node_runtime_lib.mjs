// sf_node_runtime_lib 纯逻辑冒烟（Node 直接运行：node tests/test_node_runtime_lib.mjs）
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_node_runtime_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_node_runtime_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(tmpUrl);

    // ── formatDuration ──
    check("整数补 3 位", L.formatDuration(1) === "1.000s");
    check("四舍五入 3 位", L.formatDuration(1.23456) === "1.235s");
    check("0 值", L.formatDuration(0) === "0.000s");
    check("NaN 归零", L.formatDuration(NaN) === "0.000s");
    check("undefined 归零", L.formatDuration(undefined) === "0.000s");
    check("字符串数字", L.formatDuration("2.5") === "2.500s");

    // ── accumulateSeconds ──
    check("首段毫秒→秒", L.accumulateSeconds(0, 1500) === 1.5);
    check("累加", L.accumulateSeconds(1.5, 500) === 2);
    check("undefined 前值按 0", L.accumulateSeconds(undefined, 1000) === 1);
    check("非法前值按 0", L.accumulateSeconds("x", 1000) === 1);
    check("非法增量按 0", L.accumulateSeconds(3, NaN) === 3);

    // ── resolveNodeId ──
    // 新版前端 event.detail 直接是节点 id（primitive）
    check("primitive 数字 id", L.resolveNodeId(5) === 5);
    check("primitive 字符串 id", L.resolveNodeId("7") === "7");
    check("primitive 0", L.resolveNodeId(0) === 0);
    check("终止 null", L.resolveNodeId(null) === null && L.resolveNodeId(undefined) === null);
    // 对象形态兼容
    check("node 回退", L.resolveNodeId({ node: 5 }) === 5);
    check("display_node 优先", L.resolveNodeId({ display_node: 7, node: 5 }) === 7);
    check("对象终止 null", L.resolveNodeId({ node: null }) === null);
    check("display_node:0 归一为 0", L.resolveNodeId({ display_node: 0, node: 5 }) === 0);

    // ── badge 标记 ──
    const b = L.markRuntimeBadge({ text: "x" });
    check("mark 打标", L.isRuntimeBadge(b) === true);
    check("普通 badge 不误判", L.isRuntimeBadge({ text: "x" }) === false);
    check("null 不误判", L.isRuntimeBadge(null) === false);
    check("标记常量", L.BADGE_FLAG === "_sfRuntimeBadge");

    if (failures.length) {
        console.log("FAILURES:", failures.join(", "));
        process.exit(1);
    }
    console.log("ALL PASS");
})();
