// SFTextFindReplace 纯函数库测试（Node 直接运行：node tests/test_find_replace_js.js）
// 覆盖：state（defaultState/readState 容错/mutators/预览持久化）、tidy、
//       applyRulesJS（literal/whole-word/大小写/regex 反向引用/$ 转义/ReDoS/
//       非法正则/Unicode 折叠）、isCatastrophicRegex、diffTokens、escapeHtml
const fs = require("fs");
const os = require("os");
const path = require("path");

const tmpMjs = path.join(os.tmpdir(), "sf_find_replace_lib_test.mjs");
fs.copyFileSync(path.join(__dirname, "..", "web", "sf_find_replace_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(tmpMjs);
    const {
        STATE_PROP, PREVIEW_PROP, freshRule, defaultState, readState, restoreFromProperties,
        addRule, deleteRule, toggleRuleEnabled, setFind, setReplace, setToggle,
        reorderRules, resetToDefault, getPreviewInput, setPreviewInput,
        tidy, applyRulesJS, isCatastrophicRegex, diffTokens, escapeHtml,
    } = L;

    // ---- defaultState / freshRule ----
    const d = defaultState();
    check("defaultState 形状", d.version === 1 && d.caseSensitive === false &&
        d.wholeWord === false && d.regex === false && d.tidy === true &&
        Array.isArray(d.rules) && d.rules.length === 1);
    check("freshRule 形状", d.rules[0].enabled === true && d.rules[0].find === "" &&
        d.rules[0].replace === "" && typeof d.rules[0].id === "string");
    check("freshRule id 唯一", freshRule().id !== freshRule().id);

    // ---- readState 容错 ----
    check("readState 无状态回默认", readState({ properties: {} }).rules.length === 1);
    check("readState 非对象回默认", readState({ properties: { [STATE_PROP]: "junk" } }).rules.length === 1);
    check("readState 空 rules 回默认", readState({ properties: { [STATE_PROP]: { rules: [] } } }).rules.length === 1);
    const dirty = { properties: { [STATE_PROP]: { rules: [{ find: "a" }, null, "x"] } } };
    const rd = readState(dirty);
    check("readState 丢弃非对象行", rd.rules.length === 1 && rd.rules[0].find === "a");
    check("readState 补缺省字段", rd.rules[0].enabled === true && typeof rd.rules[0].id === "string" && rd.rules[0].replace === "");
    check("readState 修正非法开关", readState({ properties: { [STATE_PROP]: { rules: [{ find: "a" }], tidy: false } } }).tidy === false);

    // ---- mutators ----
    const node = { properties: {} };
    restoreFromProperties(node);
    check("restoreFromProperties 写入初始状态", node.properties[STATE_PROP].rules.length === 1);
    addRule(node);
    addRule(node);
    check("addRule 增行", node.properties[STATE_PROP].rules.length === 3);
    const id0 = node.properties[STATE_PROP].rules[0].id;
    const id1 = node.properties[STATE_PROP].rules[1].id;
    toggleRuleEnabled(node, id0);
    check("toggleRuleEnabled", node.properties[STATE_PROP].rules[0].enabled === false);
    setFind(node, id0, "cat");
    setReplace(node, id0, "dog");
    check("setFind/setReplace", node.properties[STATE_PROP].rules[0].find === "cat" &&
        node.properties[STATE_PROP].rules[0].replace === "dog");
    setToggle(node, "regex");
    check("setToggle", node.properties[STATE_PROP].regex === true);
    deleteRule(node, id1);
    check("deleteRule 删行", node.properties[STATE_PROP].rules.length === 2);
    deleteRule(node, node.properties[STATE_PROP].rules[0].id);
    check("deleteRule 最后一行不删", node.properties[STATE_PROP].rules.length === 1);
    deleteRule(node, node.properties[STATE_PROP].rules[0].id);
    check("deleteRule 删不存在 id 不炸", node.properties[STATE_PROP].rules.length === 1);
    resetToDefault(node);
    check("resetToDefault", node.properties[STATE_PROP].rules.length === 1 &&
        node.properties[STATE_PROP].regex === false);
    addRule(node);
    const st = node.properties[STATE_PROP];
    const a = st.rules[0].id;
    const b = st.rules[1].id;
    reorderRules(node, 0, 1);
    check("reorderRules 前移", st.rules[0].id === b && st.rules[1].id === a);
    reorderRules(node, 0, 0);
    check("reorderRules 同位不变", st.rules.length === 2);

    // ---- 预览持久化 ----
    check("getPreviewInput 无样本", getPreviewInput({ properties: {} }) === null);
    setPreviewInput(node, "hello", false);
    check("setPreviewInput", getPreviewInput(node).input === "hello" && getPreviewInput(node).truncated === false);
    setPreviewInput(node, "x".repeat(5000), false);
    check("setPreviewInput 自我封顶", getPreviewInput(node).input.length === 4000 && getPreviewInput(node).truncated === true);
    check("setPreviewInput 非字符串强转", (setPreviewInput(node, 42), getPreviewInput(node).input === "42"));
    check("PREVIEW_PROP 与规则状态分离", PREVIEW_PROP !== STATE_PROP);

    // ---- tidy（与 Python 同逻辑；注意：tidy 本身不含规则删除）----
    check("tidy 折叠空格", tidy("a   b") === "a b");
    check("tidy 逗号", tidy("a  x , ,  b,") === "a x, b");
    check("tidy 逗号前空格", tidy("a ,b") === "a,b");
    check("tidy 保留换行", tidy("a\n  b") === "a\n b");
    check("tidy 去行首逗号", tidy(", a") === "a");
    check("tidy 去行尾逗号", tidy("a ,") === "a");
    check("tidy 首尾 trim", tidy("  a b  ") === "a b");

    // ---- applyRulesJS：literal ----
    let r = applyRulesJS("a cat", { rules: [{ find: "cat", replace: "dog" }], tidy: false });
    check("literal 替换", r.output === "a dog" && r.warnings.length === 0);
    r = applyRulesJS("Hello world", { rules: [{ find: "hello", replace: "hi" }], tidy: false });
    check("默认忽略大小写", r.output === "hi world");
    r = applyRulesJS("Hello world", { rules: [{ find: "hello", replace: "hi" }], caseSensitive: true, tidy: false });
    check("caseSensitive 不命中", r.output === "Hello world");
    r = applyRulesJS("art artist heart", { rules: [{ find: "art", replace: "X" }], wholeWord: true, tidy: false });
    check("whole word 只命中整词", r.output === "X artist heart");
    r = applyRulesJS("x", { rules: [{ find: "x", replace: "$1" }], tidy: false });
    check("literal 模式 $ 字面转义", r.output === "$1");
    r = applyRulesJS("x", { rules: [{ find: "x", replace: "\\1" }], tidy: false });
    check("literal 模式 \\1 字面保留", r.output === "\\1");

    // ---- applyRulesJS：regex ----
    r = applyRulesJS("a 3 b 42", { rules: [{ find: "\\d+", replace: "N" }], regex: true, tidy: false });
    check("regex \\d+ 替换", r.output === "a N b N");
    r = applyRulesJS("hello world", { rules: [{ find: "(\\w+) (\\w+)", replace: "\\2 \\1" }], regex: true, tidy: false });
    check("regex 反向引用 \\2 \\1（pyTemplateToJs）", r.output === "world hello");
    r = applyRulesJS("aba", { rules: [{ find: "(a)(b)", replace: "\\2\\1" }], regex: true, tidy: false });
    check("regex 相邻组引用", r.output === "baa");
    r = applyRulesJS("(x)", { rules: [{ find: "\\([^)]*\\)", replace: "" }], regex: true, tidy: false });
    check("regex 转义括号", r.output === "");
    r = applyRulesJS("word", { rules: [{ find: "(?P<w>w\\w+)", replace: "[\\g<w>]" }], regex: true, tidy: false });
    check("regex 命名组 (?P<n>) 翻译", r.output === "[word]");

    // ---- applyRulesJS：ReDoS 防护 ----
    r = applyRulesJS("aaaa", { rules: [{ find: "(a+)+", replace: "X" }], regex: true, tidy: false });
    check("嵌套量词跳过", r.output === "aaaa" && r.warnings.length === 1 &&
        r.warnings[0].includes("catastrophically slow"));
    r = applyRulesJS("aab", { rules: [{ find: "(a+){2}b", replace: "X" }], regex: true, tidy: false });
    check("有界量词正常命中", r.output === "X" && r.warnings.length === 0);
    r = applyRulesJS("(((", { rules: [{ find: "[()]+", replace: "X" }], regex: true, tidy: false });
    check("字符类内不误报", r.output === "X" && r.warnings.length === 0);

    // ---- isCatastrophicRegex 与 Python 对照 ----
    check("isCatastrophicRegex 嵌套", isCatastrophicRegex("(a+)+") === true && isCatastrophicRegex("(a*)*") === true &&
        isCatastrophicRegex("(\\w+)+") === true);
    check("isCatastrophicRegex 有界安全", isCatastrophicRegex("(a+){2}") === false && isCatastrophicRegex("a+") === false &&
        isCatastrophicRegex("(ab)+") === false && isCatastrophicRegex("\\d+") === false);
    check("isCatastrophicRegex 转义安全", isCatastrophicRegex("\\(a+\\)+") === false);
    check("isCatastrophicRegex 字符类安全", isCatastrophicRegex("[()]+") === false);

    // ---- applyRulesJS：非法正则 ----
    r = applyRulesJS("abc", { rules: [{ find: "(", replace: "X" }], regex: true, tidy: false });
    check("非法正则警告不炸", r.output === "abc" && r.warnings.length === 1 && r.warnings[0].includes("invalid regex"));

    // ---- applyRulesJS：tidy 默认开 + 禁用规则 ----
    r = applyRulesJS("a  cat", { rules: [{ find: "cat", replace: "" }] });
    check("tidy 默认开", r.output === "a");
    r = applyRulesJS("aaa", { rules: [{ find: "a", replace: "X", enabled: false }], tidy: false });
    check("禁用规则跳过", r.output === "aaa");
    r = applyRulesJS("bbb", { rules: [{ find: "", replace: "X" }], tidy: false });
    check("空 find 跳过", r.output === "bbb");

    // ---- applyRulesJS：Unicode ----
    r = applyRulesJS("画水彩画", { rules: [{ find: "水彩", replace: "油画" }], tidy: false });
    check("中文 literal 替换", r.output === "画油画画");
    r = applyRulesJS("Kelvin", { rules: [{ find: "k", replace: "K" }], tidy: false });
    check("Unicode 大小写折叠（Kelvin 符号 /u）", r.output === "Kelvin");
    r = applyRulesJS("café", { rules: [{ find: "CAFÉ", replace: "tea" }], tidy: false });
    check("重音忽略大小写", r.output === "tea");
    r = applyRulesJS("art artist", { rules: [{ find: "art", replace: "X" }], wholeWord: true, tidy: false });
    check("whole word 中文语境不误伤（拉丁边界）", r.output === "X artist");

    // ---- diffTokens ----
    let df = diffTokens("a b c", "a x c");
    check("diffTokens 划分", df.some((p) => p.t === "del" && p.s === "b") && df.some((p) => p.t === "ins" && p.s === "x"));
    df = diffTokens("same", "same");
    check("diffTokens 全等", df.every((p) => p.t === "eq"));
    df = diffTokens("a b c", "a b c d e f g h i j k l m n o p q r s t u v w x y z");
    check("diffTokens 追加", df.every((p) => p.t === "eq" || p.t === "ins"));
    const big = "a ".repeat(3000);
    df = diffTokens(big, big + "z");
    check("diffTokens 1M 上限退化", df.length === 2 && df[0].t === "del" && df[1].t === "ins");

    // ---- escapeHtml ----
    check("escapeHtml", escapeHtml("<a>&\"b") === "&lt;a&gt;&amp;\"b");

    printResult();
    function printResult() {
        console.log("\nFAILURES:", failures.length);
        process.exit(failures.length ? 1 : 0);
    }
})();
