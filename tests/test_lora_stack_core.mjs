// SFLoraStack core 纯函数库测试（Node 直接运行：node tests/test_lora_stack_core.mjs）
// 覆盖：
//   - 状态 readState/writeState 归一（坏 JSON/垃圾回默认、sc 缺省 = sm、
//     linkStrength 强制 sc=sm、id 去重、MAX_LORAS 截断、prefs 归一）
//   - mutations：add/remove/duplicate/move/patch（换名清 triggers/custom、
//     联动镜像、id 不可变）/setAllOn/countOn
//   - promptState 精简注入（cosmetic 剥掉、cacheMode/sep/loras 保留、
//     custom 剥掉）
//   - loadDefaults/saveDefaults 走 globalThis.app 设置
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_lora_stack_core_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_lora_stack_core.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

function fakeNode(properties = {}) {
    return { properties: { ...properties } };
}

(async () => {
    const C = await import(tmpUrl);
    const {
        BRAND, STATE_PROP, HIDDEN_INPUT, MAX_LORAS, MIN_STRENGTH, MAX_STRENGTH,
        DEFAULT_PREFS, DEFAULT_STATE,
        clampStrength, roundStrength, normalize,
        readState, writeState,
        addLora, removeLora, duplicateLora, moveLora, reorderLora, patchLora,
        setAllOn, countOn, promptState, accentOf,
    } = C;

    // ---- 常量契约（与 Python lora_reader 镜像）----
    check("HIDDEN_INPUT 匹配 Python 键", HIDDEN_INPUT === "LoraLoaderState");
    check("MAX_LORAS = 64", MAX_LORAS === 64);
    check("强度范围 ±10", MIN_STRENGTH === -10 && MAX_STRENGTH === 10);
    check("DEFAULT_PREFS 完整", ["sep", "step", "defStrength", "linkStrength", "civitai", "thumbs", "hideExt", "accent", "cacheMode"].every(
        (k) => k in DEFAULT_PREFS));

    // ---- 归一 ----
    check("readState 无状态回默认", readState(fakeNode()).loras.length === 0);
    check("readState 坏 JSON 回默认", readState(fakeNode({ loraStackState: "{oops" })).loras.length === 0);
    check("readState 空串回默认", readState(fakeNode({ loraStackState: "" })).loras.length === 0);
    check("readState 默认 prefs", readState(fakeNode()).sep === ", " && readState(fakeNode()).cacheMode === "last");
    check("normalize 垃圾根对象", normalize(null).loras.length === 0 && normalize("x").loras.length === 0);

    const n1 = fakeNode();
    writeState(n1, {
        loras: [
            { name: "a.safetensors", sm: 1.0 },           // sc 缺省 = sm
            { name: "b.safetensors", on: false, sm: 0.5, sc: 9.0 }, // link 下 sc 强制 = sm
            { name: "", sm: 1.0 },                        // 空名保留（行级占位允许）
            "not a dict",                                 // 非 dict 丢弃
        ],
        linkStrength: true,
    });
    const st1 = readState(n1);
    check("sc 缺省 = sm", st1.loras[0].sc === 1.0);
    check("linkStrength 强制 sc=sm", st1.loras[1].sc === 0.5);
    check("非 dict 丢弃", st1.loras.length === 3);
    check("on 缺省 true", st1.loras[0].on === true && st1.loras[1].on === false);
    check("id 自动生成", st1.loras.every((e) => typeof e.id === "string" && e.id.length > 0));

    // id 去重
    const n2 = fakeNode();
    writeState(n2, { loras: [{ id: "dup", name: "x", sm: 1 }, { id: "dup", name: "y", sm: 1 }] });
    const ids2 = readState(n2).loras.map((e) => e.id);
    check("重复 id 重分配", ids2[0] !== ids2[1]);

    // 截断
    const many = Array.from({ length: MAX_LORAS + 5 }, (_, i) => ({ name: `f${i}.safetensors`, sm: 1 }));
    check("MAX_LORAS 截断", normalize({ loras: many }).loras.length === MAX_LORAS);

    // ---- clamp/round ----
    check("clampStrength 钳制", clampStrength(999) === MAX_STRENGTH && clampStrength(-999) === MIN_STRENGTH);
    check("clampStrength 垃圾 -> 0", clampStrength("abc") === 0 && clampStrength(NaN) === 0);
    check("roundStrength 2 位", roundStrength(1.234) === 1.23 && roundStrength(0.5) === 0.5 && roundStrength(-1.236) === -1.24);

    // ---- mutations ----
    const n3 = fakeNode();
    const r = addLora(n3, "");
    check("addLora ok", r.ok === true && r.index === 0);
    const id0 = r.state.loras[0].id;
    check("addLora 默认强度 = defStrength", readState(n3).loras[0].sm === 1.0);
    addLora(n3, "b.safetensors");
    check("addLora 计数", readState(n3).loras.length === 2);

    check("removeLora", removeLora(n3, id0) !== null && readState(n3).loras.length === 1);
    check("removeLora 不存在 -> null", removeLora(n3, "nope") === null);

    const idb = readState(n3).loras[0].id;
    check("duplicateLora", duplicateLora(n3, idb) !== null && readState(n3).loras.length === 2);
    check("duplicateLora 新 id", readState(n3).loras[1].id !== idb);

    check("moveLora 下移", moveLora(n3, idb, +1) !== null && readState(n3).loras[0].id !== idb);
    check("moveLora 越界", moveLora(n3, idb, +5) === null);
    check("reorderLora", reorderLora(n3, 0, 1) !== null && reorderLora(n3, 0, 0) === null);

    // patch：联动镜像 + 换名清词 + id 保持
    const n4 = fakeNode();
    addLora(n4, "x.safetensors");
    const e4 = readState(n4).loras[0];
    patchLora(n4, e4.id, { sm: 0.4 });
    const after = readState(n4).loras[0];
    check("patch 联动 sc 跟随", after.sc === 0.4 && after.sm === 0.4);
    patchLora(n4, e4.id, { triggers: ["w1"], custom: ["w2"] });
    patchLora(n4, e4.id, { name: "y.safetensors" });
    const after2 = readState(n4).loras[0];
    check("patch 换名清 triggers/custom", after2.triggers.length === 0 && after2.custom.length === 0);
    check("patch 换名保留 id", after2.id === e4.id);
    patchLora(n4, e4.id, { sm: 0.7, sc: 0.3 }); // link 模式下 normalize 强制 sc=sm（不变量）
    check("patch link 模式显式 sc 被归一", readState(n4).loras[0].sc === 0.7);

    // 未联动时 sc 独立
    const n5 = fakeNode();
    writeState(n5, { linkStrength: false, loras: [{ name: "x", sm: 1, sc: 0.5 }] });
    patchLora(n5, readState(n5).loras[0].id, { sm: 0.2 });
    check("未联动 sc 独立", readState(n5).loras[0].sm === 0.2 && readState(n5).loras[0].sc === 0.5);

    // setAllOn / countOn
    const n6 = fakeNode();
    addLora(n6, "a"); addLora(n6, "b");
    check("countOn 初始", countOn(readState(n6)) === 2);
    patchLora(n6, readState(n6).loras[0].id, { on: false });
    check("countOn 半开", countOn(readState(n6)) === 1);
    setAllOn(n6, false);
    check("setAllOn false", countOn(readState(n6)) === 0);

    // ---- promptState 精简注入 ----
    const n7 = fakeNode();
    writeState(n7, {
        accent: "#123456", thumbs: false, step: 0.5, defStrength: 2.0, linkStrength: false,
        sep: "|", cacheMode: "all",
        loras: [{ name: "a.safetensors", on: true, sm: 1, sc: 0.5, triggers: ["w"], custom: ["secret"] }],
    });
    const ps = promptState(readState(n7));
    check("promptState 剥 cosmetic", !("accent" in ps) && !("thumbs" in ps) && !("step" in ps) &&
        !("defStrength" in ps) && !("linkStrength" in ps));
    check("promptState 保留执行字段", ps.sep === "|" && ps.cacheMode === "all" && ps.version === 1);
    check("promptState loras 形状", ps.loras[0].name === "a.safetensors" && ps.loras[0].sm === 1 &&
        ps.loras[0].sc === 0.5 && ps.loras[0].on === true);
    check("promptState custom 剥掉", !("custom" in ps.loras[0]) && !("id" in ps.loras[0]));

    // ---- loadDefaults/saveDefaults 走 globalThis.app ----
    const saved = {};
    globalThis.app = {
        ui: { settings: {
            getSettingValue: (k) => saved[k],
            setSettingValueAsync: async (k, v) => { saved[k] = v; },
        } },
    };
    check("saveDefaults 只存 prefs 键", await C.saveDefaults({ sep: ";", accent: "#f00", bogus: 1 }) === true
        && Object.keys(JSON.parse(saved["sfnodes.LoraStack.Defaults"])).sort().join(",") === "accent,sep");
    check("loadDefaults 读回", (() => {
        saved["sfnodes.LoraStack.Defaults"] = JSON.stringify({ sep: ";" });
        return C.loadDefaults().sep === ";";
    })());

    // accentOf
    check("accentOf 默认品牌色", accentOf(fakeNode()) === BRAND);
    check("accentOf 节点覆盖", accentOf(fakeNode({ loraStackState: JSON.stringify({ accent: "#123456" }) })) === "#123456");

    console.log();
    if (failures.length) {
        console.log(`${failures.length} FAILURES:`, failures);
        process.exit(1);
    }
    console.log("ALL PASS");
})();
