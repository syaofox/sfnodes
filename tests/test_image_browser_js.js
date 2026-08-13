// image_browser.js 纯逻辑回归测试（Node 直接运行：node tests/test_image_browser_js.js）
// 覆盖：getImageFolderFromValue（当前值 → type+folder 推导）与 folderExists（目录有效性）。
// 函数体从源文件提取（括号计数），保证与 web 实现单真源一致。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const src = fs.readFileSync(path.join(__dirname, "..", "web", "image_browser.js"), "utf8");

function extractFn(name) {
    const start = src.indexOf(`function ${name}(`);
    if (start < 0) throw new Error(`function ${name} not found in web/image_browser.js`);
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
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_imgbrowser_"));
    const mjs = [
        extractFn("getImageFolderFromValue"),
        extractFn("folderExists"),
        "export { getImageFolderFromValue, folderExists };",
    ].join("\n\n");
    const modPath = path.join(tmpDir, "lib.mjs");
    fs.writeFileSync(modPath, mjs);
    const { getImageFolderFromValue, folderExists } = await import(modPath);

    // ── getImageFolderFromValue ──
    check("空值 → input 根", (() => { const r = getImageFolderFromValue(""); return r.type === "input" && r.folder === ""; })());
    check("根目录文件 → input 根", (() => { const r = getImageFolderFromValue("a.png"); return r.type === "input" && r.folder === ""; })());
    check("子目录文件 → input/faces", (() => { const r = getImageFolderFromValue("faces/a.png"); return r.type === "input" && r.folder === "faces"; })());
    check("深层文件 → input/sub/deep", (() => { const r = getImageFolderFromValue("sub/deep/b.png"); return r.type === "input" && r.folder === "sub/deep"; })());
    check("output 后缀 → output 根", (() => { const r = getImageFolderFromValue("x.png [output]"); return r.type === "output" && r.folder === ""; })());
    check("output 后缀子目录 → output/out", (() => { const r = getImageFolderFromValue("out/x.png [output]"); return r.type === "output" && r.folder === "out"; })());
    check("clipspace → input/clipspace", (() => { const r = getImageFolderFromValue("clipspace/msk.png"); return r.type === "input" && r.folder === "clipspace"; })());

    // ── folderExists ──
    const items = [
        { path: "a.png" },
        { path: "faces/b.png" },
        { path: "faces/deep/c.png" },
        { path: "out/d.png" },
    ];
    check("空 folder 恒有效", folderExists([], ""));
    check("根目录恒有效", folderExists(items, ""));
    check("存在的目录有效", folderExists(items, "faces"));
    check("深层目录有效", folderExists(items, "faces/deep"));
    check("不存在的目录无效", !folderExists(items, "sub"));
    check("同名前缀目录不算（face vs faces）", !folderExists(items, "face"));
    check("空列表下子目录无效", !folderExists([], "faces"));

    console.log(failures.length ? `\n${failures.length} FAILED` : "\nAll passed");
    process.exit(failures.length ? 1 : 0);
})();
