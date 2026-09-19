// sf_prompt_list.js 纯逻辑测试（Node 直接运行：node tests/test_prompt_list_lines_js.js）
// 覆盖 needsMeasure：软换行镜像测量的"必不换行"判定（字符数 × 12px 最大字符宽 ≤ 容器宽）
// 与 tab 特判（等宽字体下 tab 宽度不可估，强制测量）；selectionToRange：
// 点选模式的逻辑行区间 → 输出索引切片（含 skip_empty 空白行吸附）。
// 函数体从源文件提取（括号计数），保证与 web 实现单真源一致。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const src = fs.readFileSync(path.join(__dirname, "..", "web", "sf_prompt_list.js"), "utf8");

function extractFn(name) {
    const start = src.indexOf(`function ${name}(`);
    if (start < 0) throw new Error(`function ${name} not found in web/sf_prompt_list.js`);
    let depth = 0;
    let i = src.indexOf("{", start);
    for (; i < src.length; i++) {
        if (src[i] === "{") depth++;
        else if (src[i] === "}") {
            depth--;
            if (depth === 0) return src.slice(start, i + 1);
        }
    }
    throw new Error(`unbalanced braces in ${name}`);
}

(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pl_lines_"));
    const modPath = path.join(tmpDir, "lib.mjs");
    fs.writeFileSync(modPath, extractFn("needsMeasure") + "\n" + extractFn("selectionToRange")
        + "\n" + extractFn("effectiveCount")
        + "\nexport { needsMeasure, selectionToRange, effectiveCount };");
    const { needsMeasure, selectionToRange, effectiveCount } = await import(modPath);

    check("短行不测量", !needsMeasure("abc", 100));                       // 36 ≤ 100
    check("恰好一行不测量", !needsMeasure("abcdefghij", 120));            // 120 ≤ 120 边界
    check("超宽行测量", needsMeasure("abcdefghijk", 120));                // 132 > 120
    check("空文本不测量", !needsMeasure("", 100));
    check("tab 行强制测量（窄）", needsMeasure("a\tb", 10000));            // 宽不可估
    check("tab 行强制测量（宽）", needsMeasure("a\tb", 0));
    check("窄容器下普通行也测量", needsMeasure("abcdefghij", 119));
    check("CJK 行按 12px 计", !needsMeasure("测试文本", 60));              // 4×12=48 ≤ 60
    check("CJK 超宽测量", needsMeasure("测试文本文本文本", 60));            // 8×12=96 > 60
    check("超长纯空格行测量", needsMeasure(" ".repeat(100), 500));         // 空白行超宽同样软换行
    check("短纯空格行不测量", !needsMeasure("  ", 500));

    // ── selectionToRange：逻辑行区间 → {start, maxRows} ──
    const off = [0, 1, 2, 3]; // skip_empty 关：逻辑行与输出 index 一一对应
    check("sel 单行", JSON.stringify(selectionToRange(2, 2, off)) === '{"start":2,"maxRows":1}');
    check("sel 多行", JSON.stringify(selectionToRange(1, 3, off)) === '{"start":1,"maxRows":3}');
    check("sel 乱序区间", JSON.stringify(selectionToRange(3, 1, off)) === '{"start":1,"maxRows":3}');
    check("sel 越界 clamp", JSON.stringify(selectionToRange(-5, 99, off)) === '{"start":0,"maxRows":4}');
    const on = [0, -1, 1, -1, 2]; // skip_empty 开：idx 1/3 为空白行
    check("sel 跨空白取首末", JSON.stringify(selectionToRange(0, 4, on)) === '{"start":0,"maxRows":3}');
    check("sel 区间内含有效行", JSON.stringify(selectionToRange(1, 2, on)) === '{"start":1,"maxRows":1}');
    check("sel 单击空白行向下吸附", JSON.stringify(selectionToRange(1, 1, on)) === '{"start":1,"maxRows":1}');
    check("sel 单击末位空白取上方", JSON.stringify(selectionToRange(3, 3, on)) === '{"start":2,"maxRows":1}');
    check("sel 尾部空白向上吸附", JSON.stringify(selectionToRange(3, 4, on)) === '{"start":2,"maxRows":1}');
    check("sel 全空白区间向下吸附", JSON.stringify(selectionToRange(0, 2, [-1, -1, -1, 0])) === '{"start":0,"maxRows":1}');
    check("sel 全空白区间向上吸附", JSON.stringify(selectionToRange(2, 2, [-1, 0, -1])) === '{"start":0,"maxRows":1}');
    check("sel 全空白返回 null", selectionToRange(0, 2, [-1, -1, -1]) === null);
    check("sel 空文本返回 null", selectionToRange(0, 0, []) === null);

    // ── effectiveCount：自动 total 的有效行数（与头部计数同语义）──
    check("count skip 开计非空行", effectiveCount("a\n\nb\nc", true) === 3);
    check("count skip 关计逻辑行", effectiveCount("a\n\nb\nc", false) === 4);
    check("count 全空白 skip 开为 0", effectiveCount("  \n\t\n", true) === 0);
    check("count 全空白 skip 关为 3", effectiveCount("  \n\t\n", false) === 3);
    check("count 空文本 skip 开为 0", effectiveCount("", true) === 0);
    check("count 空文本 skip 关为 1", effectiveCount("", false) === 1);
    check("count null 不炸", effectiveCount(null, true) === 0 && effectiveCount(undefined, false) === 1);
    check("count 首尾空行", effectiveCount("\nfirst\n", true) === 1);

    console.log(failures.length ? `\n${failures.length} FAILED` : "\nAll passed");
    process.exit(failures.length ? 1 : 0);
})();
