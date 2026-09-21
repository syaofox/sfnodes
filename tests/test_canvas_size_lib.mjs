// SF Canvas Size Preset lib 纯函数测试（Node 直接运行：
// node tests/test_canvas_size_lib.mjs）
// 覆盖：isTierHeader / firstSelectable / validCustomName / validDim /
// normalizeCustomPresets / customOptionValue / mergeResolutionValues。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_canvas_size_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_canvas_size_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(pathToFileURL(tmpMjs).href);

    check("CUSTOM_HEADER", L.CUSTOM_HEADER === "--Custom--");

    // isTierHeader
    check("分组头判定", L.isTierHeader("--1MP--") && L.isTierHeader("--Custom--"));
    check("非分组头", !L.isTierHeader("1024x1024 (1:1)") && !L.isTierHeader("") && !L.isTierHeader(null));

    // firstSelectable
    check("首个可选跳过分组头", L.firstSelectable(["--1MP--", "1024x1024 (1:1)", "x"]) === "1024x1024 (1:1)");
    check("全分组头退首个", L.firstSelectable(["--a--", "--b--"]) === "--a--");
    check("空数组 undefined", L.firstSelectable([]) === undefined);

    // validCustomName
    check("合法名", L.validCustomName("My Wide"));
    check("空名非法", !L.validCustomName("   "));
    check("括号名非法", !L.validCustomName("A(1)") && !L.validCustomName("A)"));
    check("路径名非法", !L.validCustomName("a/b") && !L.validCustomName("a\\b"));
    check("控制字符非法", !L.validCustomName("A\x01"));
    check("超长名非法", !L.validCustomName("A".repeat(201)));
    check("非字符串非法", !L.validCustomName(3) && !L.validCustomName(null));

    // validDim
    check("合法维", L.validDim(1) && L.validDim(32768));
    check("非法维", !L.validDim(0) && !L.validDim(-1) && !L.validDim(32769)
        && !L.validDim(1.5) && !L.validDim(true) && !L.validDim("2"));

    // normalizeCustomPresets
    check("归一化裸数组", JSON.stringify(L.normalizeCustomPresets([{ name: " A ", w: 3, h: 1 }]))
        === '[{"name":"A","w":3,"h":1}]');
    check("归一化 {presets}", JSON.stringify(L.normalizeCustomPresets({ presets: [{ name: "B", w: 2, h: 2 }] }))
        === '[{"name":"B","w":2,"h":2}]');
    check("过滤非法与重名", JSON.stringify(L.normalizeCustomPresets([
        { name: "", w: 1, h: 1 }, "junk",
        { name: "A", w: 3, h: 1 }, { name: "A", w: 9, h: 9 },
        { name: "B", w: 0, h: 1 }, { name: "C", w: 1, h: "x" },
        { name: "D(X)", w: 2, h: 2 }, { name: "E", w: 2.5, h: 4 },
    ])) === '[{"name":"A","w":3,"h":1}]');
    check("垃圾输入回空", JSON.stringify(L.normalizeCustomPresets("nope")) === "[]"
        && JSON.stringify(L.normalizeCustomPresets(null)) === "[]");

    // customOptionValue
    check("选项编码", L.customOptionValue("My Wide", 1600, 900) === "1600x900 (My Wide)");

    // parseCanvasSizeLabel（接线宽高比静态预读，§118）
    check("解析带比例标签", JSON.stringify(L.parseCanvasSizeLabel("1024x1024 (1:1)")) === '{"w":1024,"h":1024}');
    check("解析裸 WxH", JSON.stringify(L.parseCanvasSizeLabel("1024x768")) === '{"w":1024,"h":768}');
    check("解析自定义项编码", JSON.stringify(L.parseCanvasSizeLabel("1600x900 (My Wide)")) === '{"w":1600,"h":900}');
    check("畸形/分组头/非串回 null", L.parseCanvasSizeLabel("abc") === null
        && L.parseCanvasSizeLabel("--1MP--") === null
        && L.parseCanvasSizeLabel("1024") === null
        && L.parseCanvasSizeLabel("1024x0") === null
        && L.parseCanvasSizeLabel(null) === null);

    // readResolutionWidgetSize（上游 resolution combo）
    check("读 resolution widget", JSON.stringify(L.readResolutionWidgetSize(
        { widgets: [{ name: "model", value: "x" }, { name: "resolution", value: "704x1408 (0.5)" }] }))
        === '{"w":704,"h":1408}');
    check("无 resolution widget 回 null", L.readResolutionWidgetSize({ widgets: [{ name: "value", value: 5 }] }) === null
        && L.readResolutionWidgetSize(null) === null);

    // mergeResolutionValues
    const official = ["--1MP--", "1024x1024 (1:1)", "1280x720 (16:9)"];
    check("无自定义保持原样", JSON.stringify(L.mergeResolutionValues(official, [])) === JSON.stringify(official));
    check("加入自定义分组且置顶", JSON.stringify(L.mergeResolutionValues(official, [{ name: "Wide", w: 1600, h: 900 }]))
        === JSON.stringify(["--Custom--", "1600x900 (Wide)", ...official]));
    check("与官方重复跳过", JSON.stringify(L.mergeResolutionValues(official, [{ name: "1:1", w: 1024, h: 1024 }]))
        === JSON.stringify(official));
    check("同名去重不同名并存", JSON.stringify(L.mergeResolutionValues([], [
        { name: "A", w: 1, h: 1 }, { name: "A", w: 9, h: 9 }, { name: "B", w: 1, h: 1 },
    ])) === JSON.stringify(["--Custom--", "1x1 (A)", "1x1 (B)"]));

    console.log();
    if (failures.length) { console.log("FAILED:", failures.length, "项"); process.exit(1); }
    console.log("ALL PASS");
})();
