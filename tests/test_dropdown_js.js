// SFValueDropdown 纯函数库测试（Node 直接运行：node tests/test_dropdown_js.js）
// 覆盖：
//   - coerce 双端同用例（与 tests/test_dropdown.py 相同的数字语法/readable/
//     coerceValue 断言——THE PARITY RULE 的守护）
//   - 状态 readState/writeState 归一（非 dict 行丢弃、index 钳制、模式归一）
//   - 运行游标：fixed 不动；increment 首轮发显示中的条目、之后推进、wrap、
//     _pending 持有（未 commit 不推进）、commitPick 花牌、手工选择压过序列；
//     random 永不连续相同（2+ 条）
//   - injectedState lean 形状（只含 {version,type,value}）
//   - syncOutput 槽类型/名称同步；slotAccepts 多类型/通配兼容
const fs = require("fs");
const os = require("os");
const path = require("path");

const tmpMjs = path.join(os.tmpdir(), "sf_dropdown_lib_test.mjs");
fs.copyFileSync(path.join(__dirname, "..", "web", "sf_dropdown_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(tmpMjs);
    const {
        TYPES, STATE_PROP, HIDDEN_INPUT,
        normalizeType, readable, coerceValue, previewText,
        defaultState, readState, writeState,
        pendingIndex, commitPick, shownIndex, injectedState,
        syncOutput, slotAccepts, SOCKET_TYPES,
    } = L;

    // ---- 数字语法（与 Python 侧同用例）----
    const nums = ["5", "5.", ".5", "5.5", "+5", "-3", "1e3", "1E3", "-1e3"];
    for (const s of nums) check(`asNumber 收 ${s}`, coerceValue(s, "float") === Number(s));
    for (const s of ["0x10", "0b1", "1_0", "1,024", "1024px", "abc", "Infinity", "NaN", "", "   "]) {
        check(`asNumber 拒 ${JSON.stringify(s)}`, coerceValue(s, "int") === 0);
    }

    // ---- readable ----
    check("readable text 恒真", readable("anything", "text") && readable(123, "text") && readable(null, "text"));
    check("readable int 好", readable("1024", "int") === true);
    check("readable int 坏", readable("abc", "int") === false);
    check("readable int 超钳制", readable("1e308", "int") === false);
    check("readable bool 单词", readable("YES", "bool") === true && readable("off", "bool") === true);
    check("readable bool 数字", readable("0", "bool") === true && readable("3", "bool") === true);
    check("readable bool 坏", readable("maybe", "bool") === false);
    check("readable bool 超钳制仍读", readable("1e308", "bool") === true);
    check("readable 未知类型回退 text", readable("x", "bogus") === true);
    check("readable 别名类型", readable("1024", "integer") === true && readable("true", "boolean") === true);

    // ---- coerceValue（与 Python coerce_value 同用例）----
    check("coerce text 原样", coerceValue("hi", "text") === "hi");
    check("coerce text null -> 空", coerceValue(null, "text") === "");
    check("coerce text bool -> 拼写", coerceValue(true, "text") === "true" && coerceValue(false, "text") === "false");
    check("coerce text 整 float 去 .0", coerceValue(2.0, "text") === "2");
    check("coerce text 非整 float", coerceValue(2.5, "text") === "2.5");
    check("coerce int 取整 half-away", coerceValue("2.5", "int") === 3 && coerceValue("-2.5", "int") === -3);
    check("coerce int 钳制", coerceValue("1e308", "int") === 1e12);
    check("coerce int 坏 -> 0", coerceValue("abc", "int") === 0);
    check("coerce float 好", coerceValue("0.35", "float") === 0.35);
    check("coerce float 坏 -> 0", coerceValue("abc", "float") === 0);
    check("coerce float 钳制", coerceValue("1e20", "float") === 1e12);
    check("coerce bool 单词", coerceValue("yes", "bool") === true && coerceValue("No", "bool") === false);
    check("coerce bool 数字", coerceValue("0", "bool") === false && coerceValue("3", "bool") === true);
    check("coerce bool 坏 -> false", coerceValue("maybe", "bool") === false);
    check("coerce bool 真值保持", coerceValue(true, "bool") === true);
    check("previewText 多行截断", previewText("line1\nline2", "text") === "line1…");

    // ---- normalizeType ----
    check("normalizeType 别名", normalizeType("integer") === "int" && normalizeType("string") === "text"
        && normalizeType("boolean") === "bool" && normalizeType("bogus") === "text" && normalizeType(5) === "text");

    // ---- 状态 ----
    const node = { properties: {} };
    let st = readState(node);
    check("readState 空默认", st.type === "text" && st.mode === "fixed" && st.index === 0 && st.options.length === 0);
    writeState(node, { options: [{ name: "a", value: "1" }, { name: "b", value: "2" }], type: "int" });
    st = readState(node);
    check("writeState 写入", st.options.length === 2 && st.type === "int");
    check("STATE_PROP 键", JSON.stringify(node.properties[STATE_PROP]) === JSON.stringify(st));
    check("writeState 容错非对象", readState({ properties: { [STATE_PROP]: "junk" } }).type === "text");
    writeState(node, { options: [{ name: "a", value: "1" }, null, "junk", { name: 5, value: 2 }, { name: "b" }] });
    st = readState(node);
    check("writeState 非对象行归一为空行", st.options.length === 5 && st.options[1].name === "" && st.options[1].value === "");
    check("readState 非对象行丢弃", readState({ properties: { [STATE_PROP]: { options: [{ name: "a", value: "1" }, null] } } }).options.length === 1);
    writeState(node, { index: 99, options: [{ name: "a", value: "1" }] });
    check("index 钳制到列表内", readState(node).index === 0);
    writeState(node, { mode: "bogus" });
    check("模式归一回退 fixed", readState(node).mode === "fixed");

    // ---- 游标：fixed ----
    const mk = (options, mode = "fixed", index = 0) => {
        const n = { properties: {} };
        writeState(n, { options, mode, index });
        return n;
    };
    {
        const n = mk([{ name: "a", value: "1" }, { name: "b", value: "2" }], "fixed", 1);
        check("fixed pendingIndex = 选中", pendingIndex(n) === 1);
        check("fixed 不产生 pending", n._sfDropdownPending == null);
        check("fixed injectedState", JSON.stringify(injectedState(n)) === JSON.stringify({ version: 1, type: "text", value: "2" }));
        commitPick(n);
        check("fixed commit 不动游标", n._sfDropdownCursor == null);
    }

    // ---- 游标：increment（首轮发显示中的，然后推进；pending 持有；commit 花牌；wrap）----
    {
        const n = mk([{ name: "a", value: "1" }, { name: "b", value: "2" }, { name: "c", value: "3" }], "increment", 1);
        check("increment 首轮发显示中的", pendingIndex(n) === 1);
        check("increment 持有 pending", n._sfDropdownPending === 1);
        check("increment 未 commit 再掷同牌", pendingIndex(n) === 1);
        commitPick(n);
        check("increment commit 花牌 -> cursor", n._sfDropdownCursor === 1 && n._sfDropdownPending == null);
        check("increment 次轮推进", pendingIndex(n) === 2);
        commitPick(n);
        check("increment 三轮 wrap", pendingIndex(n) === 0);
        commitPick(n);
        check("increment wrap 后继续", pendingIndex(n) === 1);
        check("increment shownIndex 跟随 cursor", shownIndex(n) === 1);
    }

    // ---- 游标：increment 手工选择压过序列 ----
    {
        const n = mk([{ name: "a", value: "1" }, { name: "b", value: "2" }, { name: "c", value: "3" }], "increment", 0);
        pendingIndex(n);
        commitPick(n);
        check("序列推进到 1", pendingIndex(n) === 1);
        writeState(n, { index: 2 });   // 用户点击选择 c
        n._sfDropdownPending = null;
        n._sfDropdownCursor = null;
        check("手工选择后重新从选中开始", pendingIndex(n) === 2);
        check("手工选择后 shownIndex", shownIndex(n) === 2);
    }

    // ---- 游标：random 永不连续相同 ----
    {
        for (let round = 0; round < 20; round++) {
            const n = mk([{ name: "a", value: "1" }, { name: "b", value: "2" }], "random", 0);
            const first = pendingIndex(n);
            commitPick(n);
            const second = pendingIndex(n);
            if (first === second) { check(`random 20 轮内无连续重复（round ${round}）`, false); break; }
            if (round === 19) check("random 20 轮内无连续重复", true);
        }
        const n = mk([{ name: "a", value: "1" }, { name: "b", value: "2" }], "random", 0);
        const i = pendingIndex(n);
        check("random 界内", i >= 0 && i < 2);
        check("random shownIndex 显示 pending", shownIndex(n) === i);
    }

    // ---- 空列表 ----
    {
        const n = mk([], "increment");
        check("空列表 pendingIndex 0", pendingIndex(n) === 0);
        check("空列表 injectedState value null", injectedState(n).value === null);
    }

    // ---- syncOutput ----
    {
        const n = { outputs: [{ name: "value", label: "", type: "" }] };
        writeState(n, { type: "int" });
        syncOutput(n);
        check("syncOutput int 槽", n.outputs[0].type === "INT" && n.outputs[0].name === "value" && n.outputs[0].label.length > 0);
        writeState(n, { type: "bool" });
        syncOutput(n);
        check("syncOutput bool 槽", n.outputs[0].type === "BOOLEAN");
        writeState(n, { type: "float" });
        syncOutput(n);
        check("syncOutput float 槽", n.outputs[0].type === "FLOAT");
        writeState(n, { type: "text" });
        syncOutput(n);
        check("syncOutput text 槽", n.outputs[0].type === "STRING");
    }

    // ---- slotAccepts ----
    check("slotAccepts 相等", slotAccepts("INT", "INT") === true);
    check("slotAccepts 不等", slotAccepts("FLOAT", "INT") === false);
    check("slotAccepts 通配", slotAccepts("*", "INT") === true && slotAccepts("INT", "*") === true);
    check("slotAccepts 空槽", slotAccepts("", "INT") === true && slotAccepts(null, "INT") === true);
    check("slotAccepts 多类型", slotAccepts("FLOAT,INT,BOOLEAN", "INT") === true);
    check("slotAccepts 多类型不含", slotAccepts("FLOAT,STRING", "INT") === false);
    check("slotAccepts 大小写不敏感", slotAccepts("float", "FLOAT") === true);

    // ---- 常量（Python 契约）----
    check("TYPES 与 Python 一致", JSON.stringify(TYPES) === JSON.stringify(["text", "int", "float", "bool"]));
    check("HIDDEN_INPUT 与 Python 一致", HIDDEN_INPUT === "DropdownState");

    if (failures.length) {
        console.log(`\n${failures.length} FAILED`);
        process.exit(1);
    }
    console.log("\nALL PASS");
})();
