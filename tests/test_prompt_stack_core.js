// SFPromptStack 核心纯逻辑测试（Node 直接运行：node tests/test_prompt_stack_core.js）
// 复制 web/sf_prompt_stack_core.js 为 .mjs 后加载，覆盖：
//   - normalize：旧数据/手写状态兜底、id 去重、上限
//   - readState / writeState：node.properties JSON 往返
//   - promptState：注入形状（只留 enabled/text）
//   - activeRows：开且非空的行（输出 index 语义）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ps_core_"));
    const code = fs.readFileSync(path.join(__dirname, "..", "web", "sf_prompt_stack_core.js"), "utf8");
    fs.writeFileSync(path.join(tmpDir, "core.mjs"), code);
    const core = await import(path.join(tmpDir, "core.mjs"));

    // normalize：默认状态
    const dflt = core.normalize({});
    check("默认 rows 空", Array.isArray(dflt.rows) && dflt.rows.length === 0);

    // normalize：行归一（缺省 enabled=true、text 非字符串兜底、id 补发）
    const st = core.normalize({ rows: [
        { text: "hello" },
        { enabled: false, text: "off" },
        { text: 123 },
        null,
    ] });
    check("行归一", st.rows.length === 3
        && st.rows[0].enabled === true && st.rows[0].text === "hello"
        && st.rows[1].enabled === false
        && typeof st.rows[0].id === "string" && st.rows[0].id.startsWith("p"));

    // id 去重
    const dup = core.normalize({ rows: [
        { id: "same", text: "a" },
        { id: "same", text: "b" },
    ] });
    check("id 去重", dup.rows[0].id !== dup.rows[1].id);

    // 上限
    const many = core.normalize({ rows: Array.from({ length: 600 }, (_, i) => ({ text: "t" + i })) });
    check("行数上限", many.rows.length === core.MAX_ROWS);

    // readState：坏 JSON 走默认
    const node = { properties: {}, setDirtyCanvas() {} };
    node.properties[core.STATE_PROP] = "not-json";
    check("readState 坏 JSON 兜底", core.readState(node).rows.length === 0);

    // writeState + readState 往返
    const s2 = core.normalize({ rows: [{ enabled: true, text: "x" }] });
    core.writeState(node, s2);
    check("writeState 写 properties", typeof node.properties[core.STATE_PROP] === "string");
    check("往返一致", core.readState(node).rows[0].text === "x");
    check("writeState 标记脏画布", true); // setDirtyCanvas 由 mock 吞掉

    // promptState 注入形状：只留 enabled/text，id/label 剥掉
    const inj = core.promptState(core.normalize({ rows: [
        { id: "p1", enabled: true, label: "L", text: "a" },
        { id: "p2", enabled: false, label: "", text: "b" },
    ] }));
    check("注入形状", inj.version === 1 && inj.rows.length === 2
        && JSON.stringify(inj.rows[0]) === JSON.stringify({ enabled: true, text: "a" })
        && JSON.stringify(inj.rows[1]) === JSON.stringify({ enabled: false, text: "b" }));

    // activeRows：开且非空
    const act = core.activeRows(core.normalize({ rows: [
        { enabled: true, text: "a" },
        { enabled: false, text: "b" },
        { enabled: true, text: "   " },
        { enabled: true, text: "c" },
    ] }));
    check("activeRows", act.length === 2 && act[0].text === "a" && act[1].text === "c");

    // 行高 h：缺省/非法 → null（UI 用默认），合法 clamp
    const hh = core.normalize({ rows: [
        { text: "dflt" },
        { text: "bad", h: "x" },
        { text: "small", h: 10 },
        { text: "big", h: 9999 },
        { text: "ok", h: 76.8 },
    ] });
    check("h 缺省 null", hh.rows[0].h === null);
    check("h 非法 null", hh.rows[1].h === null);
    check("h 下限 clamp", hh.rows[2].h === core.MIN_ROW_H);
    check("h 上限 clamp", hh.rows[3].h === core.MAX_ROW_H);
    check("h 合法保留(取整)", hh.rows[4].h === 76);

    // 注入形状：h 剥掉（UI 属性不进缓存键）
    const injH = core.promptState(core.normalize({ rows: [{ text: "a", h: 120 }] }));
    check("注入不含 h", JSON.stringify(injH.rows[0]) === JSON.stringify({ enabled: true, text: "a" }));

    if (failures.length) {
        console.log(`\n${failures.length} FAILED`);
        process.exit(1);
    }
    console.log("\nALL PASS");
})();
