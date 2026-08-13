// sf_prompt_list.js 行高测量纯逻辑测试（Node 直接运行：node tests/test_prompt_list_lines_js.js）
// 覆盖 needsMeasure：软换行镜像测量的"必不换行"判定（字符数 × 12px 最大字符宽 ≤ 容器宽）
// 与 tab 特判（等宽字体下 tab 宽度不可估，强制测量）。
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
    fs.writeFileSync(modPath, extractFn("needsMeasure") + "\nexport { needsMeasure };");
    const { needsMeasure } = await import(modPath);

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

    console.log(failures.length ? `\n${failures.length} FAILED` : "\nAll passed");
    process.exit(failures.length ? 1 : 0);
})();
