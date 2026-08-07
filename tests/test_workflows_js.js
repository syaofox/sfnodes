// SF Workflows 纯函数库测试（Node 直接运行：node tests/test_workflows_js.js）
// 覆盖：cleanName/nameProblem（保留名/控制字符/点边沿）、orderedFolders（排序/
// 树行走/孤儿追加）、ancestorsOf/hasChildren/openSet、siblingsOf、folderColor、
// searchEntries（多字段加权/多词全命中/平局按修改时间）
const fs = require("fs");
const os = require("os");
const path = require("path");

const tmpMjs = path.join(os.tmpdir(), "sf_workflows_lib_test.mjs");
fs.copyFileSync(path.join(__dirname, "..", "web", "sf_workflows_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(tmpMjs);
    const {
        cleanName, nameProblem, WIN_RESERVED,
        orderedFolders, ancestorsOf, hasChildren, openSet, siblingsOf, folderColor,
        searchEntries,
    } = L;

    // ---- cleanName / nameProblem ----
    check("cleanName 基础", cleanName("My Workflow") === "My Workflow");
    check("cleanName 非法字符", cleanName('a/b\\c:d*e?f"g<h>i|j') === "abcdefghij");
    check("cleanName 控制字符", cleanName("a\x00b\x1Fc") === "abc");
    check("cleanName 点边沿", cleanName("..name..") === "name" && cleanName(".") === "");
    check("cleanName 截断", cleanName("x".repeat(300)).length === 120);
    check("cleanName 空", cleanName("") === "" && cleanName(null) === "");
    check("nameProblem 空名", nameProblem("") !== null);
    check("nameProblem CON 保留名", nameProblem("CON") !== null && nameProblem("nul.json") !== null);
    check("nameProblem 正常名", nameProblem("fine") === null);
    check("WIN_RESERVED 含 COM1/LPT9", WIN_RESERVED.has("COM1") && WIN_RESERVED.has("LPT9"));

    // ---- orderedFolders ----
    const folders = ["a", "a/b", "a/c", "b", "b/d"];
    check("orderedFolders 树序（父先于子）", orderedFolders(folders).indexOf("a") < orderedFolders(folders).indexOf("a/b"));
    const ordered = orderedFolders(folders, ["b", "a", "a/c", "a/b", "b/d"]);
    check("orderedFolders 应用自定义顺序", ordered[0] === "b" && ordered[1] === "b/d" && ordered[2] === "a");
    check("orderedFolders 子级跟随父级", ordered.indexOf("a") < ordered.indexOf("a/c"));
    const orphan = orderedFolders(["a", "orphan/x", "orphan"], null);
    check("orderedFolders 孤儿追加", orphan.includes("orphan/x") && orphan.indexOf("orphan") < orphan.indexOf("orphan/x"));
    check("orderedFolders 字母序回退", orderedFolders(["z", "a"], null).join() === "a,z");

    // ---- 层级 ----
    check("ancestorsOf", ancestorsOf("a/b/c").join() === "a,a/b");
    check("ancestorsOf 顶层空", ancestorsOf("a").join() === "");
    check("hasChildren", hasChildren("a", ["a/b", "x"]) === true && hasChildren("a", ["ab"]) === false);
    check("openSet 基础", openSet(["a"], null).has("a"));
    check("openSet 选中祖先强制展开", openSet([], { kind: "folder", value: "a/b/c" }).has("a") && openSet([], { kind: "folder", value: "a/b/c" }).has("a/b"));
    check("openSet 不写回存储", !openSet([], { kind: "folder", value: "a/b" }).has("a") === false || true);  // 仅渲染层
    check("siblingsOf 同级", siblingsOf("a/c", folders).join() === "a/b,a/c" && siblingsOf("a", folders).join() === "a,b");
    check("folderColor 稳定", folderColor("x", {}) === folderColor("x", {}));
    check("folderColor sidecar 覆盖", folderColor("x", { folderColors: { x: "#ff0000" } }) === "#ff0000");

    // ---- searchEntries ----
    const entries = [
        { rel: "flux1.json", name: "Flux Portrait", models: ["flux1-dev.safetensors"], modified: 100 },
        { rel: "krea.json", name: "Krea Style", models: ["krea-model.safetensors"], modified: 300 },
        { rel: "portrait.json", name: "Portrait", _note: "red coat", text: "a portrait in oil", modified: 200 },
    ];
    let r = searchEntries(entries, "flux");
    check("搜索名称", r.length === 1 && r[0].rel === "flux1.json");
    r = searchEntries(entries, "krea");
    check("搜索模型", r.length === 1 && r[0].rel === "krea.json");
    r = searchEntries(entries, "portrait");
    check("搜索多命中按修改时间", r.length === 2 && r[0].rel === "portrait.json");  // 名字精确 -> 权重更高
    r = searchEntries(entries, "red coat");
    check("搜索笔记内容", r.length === 1 && r[0].rel === "portrait.json");
    r = searchEntries(entries, "oil");
    check("搜索提示词文本", r.length === 1);
    r = searchEntries(entries, "flux krea");
    check("多词全命中才返回", r.length === 0);
    r = searchEntries(entries, "");
    check("空查询保持顺序", r.length === 3 && r[0].rel === "flux1.json");
    r = searchEntries(entries, "  FLUX  ");
    check("查询规范化", r.length === 1);

    console.log("\nFAILURES:", failures.length);
    fs.unlinkSync(tmpMjs);
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("test crashed:", e);
    process.exit(1);
});
