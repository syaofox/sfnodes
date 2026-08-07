// SFPromptTags 纯工具库测试（Node 直接运行：node tests/test_prompt_tags_js.js）
// 覆盖：scanTokens（边界：邮件/算式/链式/跨种/Unicode 码点）、expandAll（展开/未知
//       保留/spans/解析器）、normalizeLibrary（清洗/去重/分类补齐/sides/catModes/
//       移桶/旧库兼容）、reorder、导出/导入数据变换、isSameAsStored
const fs = require("fs");
const os = require("os");
const path = require("path");

// sf_prompt_tags_lib.js 是无 app/DOM 依赖的 ES 模块，复制为 .mjs 后直接 import（项目惯例）
const tmpMjs = path.join(os.tmpdir(), "sf_prompt_tags_lib_test.mjs");
fs.copyFileSync(path.join(__dirname, "..", "web", "sf_prompt_tags_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(tmpMjs);
    const {
        scanTokens, expandAll, normalizeLibrary, tagLines, uniqueTagName, cleanTagName,
        isListTag, catOf, sideOfCat, isListCat, catsOnSameSide,
        reorderCategoryStep, canMoveCategory, reorderCategoryTo,
        tagMode, catMode, MODES, DEFAULT_MODE, cleanMode, hasPosition,
        exportLibraryJSON, parseImport, importCategories, subsetImport, applyImportData,
        isSameAsStored, TEXT_BUCKET, LIST_BUCKET,
    } = L;

    // ---- scanTokens ----
    let toks = scanTokens("hello @a world");
    check("@ 基础扫描", toks.length === 1 && toks[0].kind === "tag" && toks[0].name === "a");
    toks = scanTokens("@a @b");
    check("两个 @tag", toks.length === 2 && toks[1].name === "b");
    toks = scanTokens("user@name.com");
    check("邮件不误判", toks.length === 0);
    toks = scanTokens("2*2");
    check("算式不误判", toks.length === 0);
    toks = scanTokens("@a@b");
    check("同种链式 @a@b", toks.length === 2);
    toks = scanTokens("*a*b");
    check("同种链式 *a*b", toks.length === 2 && toks.every((t) => t.kind === "wild"));
    toks = scanTokens("@a#b");
    check("跨种不链式 @a#b", toks.length === 1 && toks[0].name === "a");
    toks = scanTokens("a@b");
    check("单词后 @ 不误判", toks.length === 0);
    toks = scanTokens("@foo_bar-2");
    check("下划线/连字符名称", toks.length === 1 && toks[0].name === "foo_bar-2");
    toks = scanTokens("𠀀@tag");
    check("补充平面字符后不误判", toks.length === 0);
    toks = scanTokens("𠀀 @tag");
    check("空格隔开的补充平面字符后可识别", toks.length === 1);
    toks = scanTokens("*Styles");
    check("* wildcard", toks.length === 1 && toks[0].kind === "wild" && toks[0].name === "Styles");
    toks = scanTokens("#animals");
    check("# list", toks.length === 1 && toks[0].kind === "list" && toks[0].name === "animals");
    toks = scanTokens("no tokens");
    check("无符号不扫描", toks.length === 0);
    toks = scanTokens(null);
    check("null 输入容错", toks.length === 0);

    // ---- expandAll ----
    const tags = [
        { name: "a", cat: "", text: "AAA" },
        { name: "list1", cat: "", kind: "list", text: "one\ntwo\n" },
        { name: "b", cat: "", text: "BBB" },
    ];
    let r = expandAll("@a and @b", { tags });
    check("@tag 展开", r.out === "AAA and BBB" && r.knownTags.join() === "a,b");
    check("@tag 展开 spans", r.spans.length === 2 && r.spans[0].start === 0 && r.spans[0].end === 3 && r.spans[0].known === true);
    r = expandAll("@nope", { tags });
    check("未知 @tag 保留字面量", r.out === "@nope" && r.unknownTags.join() === "nope" && r.spans[0].known === false);
    r = expandAll("*S", { tags, resolveWild: () => "rand" });
    check("* 解析器替换", r.out === "rand" && r.knownWilds.join() === "S");
    r = expandAll("*S", { tags });
    check("无解析器时 * 保留字面量", r.out === "*S");
    r = expandAll("#list1", { tags, resolveList: (n) => (n === "list1" ? "one" : null) });
    check("# 解析器替换", r.out === "one" && r.knownLists.join() === "list1");
    r = expandAll("#nope", { tags, resolveList: (n) => (n === "list1" ? "one" : null) });
    check("未知 # 保留字面量（解析器返回 null）", r.out === "#nope" && r.unknownLists.join() === "nope");
    r = expandAll("plain text", { tags });
    check("无 token 原样输出", r.out === "plain text" && r.spans.length === 0);
    r = expandAll("@a @a", { tags });
    check("重复 @tag 各自展开", r.out === "AAA AAA");
    r = expandAll("x@a", { tags });
    check("token 紧贴单词不展开", r.out === "x@a");
    r = expandAll(null, { tags });
    check("null 文本容错", r.out === "" && r.spans.length === 0);
    r = expandAll("@a", {});
    check("空标签表时保留字面量", r.out === "@a" && r.unknownTags.length === 1);

    // ---- normalizeLibrary：基础 ----
    let lib = normalizeLibrary({
        categories: ["Styles", "styles", ""],
        tags: [
            { name: "ta+g", cat: "Styles", text: "x" },
            { name: "Tag", cat: "", text: "y" },
            { name: null, cat: "", text: "z" },
            { name: "list1", cat: "Styles", kind: "list", text: "a\nb" },
            { name: "text1", cat: "Styles", kind: "text", text: "t" },
            { name: "orphan", cat: "Missing", text: "o" },
            { name: "badtext", cat: "", text: 123 },
        ],
    });
    check("分类大小写去重", lib.categories.filter((c) => c.toLowerCase() === "styles").length === 1 && lib.categories.includes("Styles"));
    check("名称清洗 ta+g -> tag", lib.tags.some((t) => t.name === "tag"));
    check("大小写去重 Tag/tag", lib.tags.filter((t) => t.name === "tag").length === 1);
    check("非法名称丢弃", !lib.tags.some((t) => t.name === ""));
    check("kind 仅 list 保留", lib.tags.find((t) => t.name === "list1").kind === "list" && !("kind" in lib.tags.find((t) => t.name === "text1")));
    check("标签引用分类自动补齐", lib.categories.includes("Missing"));
    check("text 非字符串转空", lib.tags.find((t) => t.name === "badtext").text === "");
    check("规范化输出含 listCats/catModes 键", Array.isArray(lib.listCats) && typeof lib.catModes === "object");
    lib = normalizeLibrary(null);
    check("null 库容错", lib.categories.length === 0 && lib.tags.length === 0);
    lib = normalizeLibrary({ categories: "notarray", tags: "nope" });
    check("非数组字段容错", lib.categories.length === 0 && lib.tags.length === 0);

    // ---- normalizeLibrary：sides（Text/List 分类侧）----
    lib = normalizeLibrary({
        categories: ["Styles", "Animals"],
        listCats: ["Animals"],
        tags: [
            { name: "fox", cat: "Animals", kind: "list", text: "a\nb" },
            { name: "oil", cat: "Styles", text: "x" },
            { name: "rebel", cat: "Animals", text: "plain text in a list cat" },
            { name: "stray-list", cat: "Styles", kind: "list", text: "1\n2" },
        ],
    });
    check("listCats 声明保留", lib.listCats.length === 1 && lib.listCats[0] === "Animals");
    check("sideOfCat 判定", sideOfCat("Animals", lib) === "list" && sideOfCat("Styles", lib) === "text");
    check("isListCat", isListCat("Animals", lib) === true && isListCat("Styles", lib) === false);
    check("kind 与分类侧冲突 -> 移桶", lib.tags.find((t) => t.name === "rebel").cat === "");
    check("List 标签在 Text 分类 -> 移桶", lib.tags.find((t) => t.name === "stray-list").cat === "");
    check("一致者不动", lib.tags.find((t) => t.name === "fox").cat === "Animals");
    // 旧库（无 listCats 声明）：全是 list 标签的分类启发式归 List 侧
    lib = normalizeLibrary({
        categories: ["OldLists"],
        tags: [{ name: "l", cat: "OldLists", kind: "list", text: "1\n2" }],
    });
    check("旧库启发式：全 list 分类 -> List 侧", lib.listCats.includes("OldLists"));
    // 桶名保留：不能成为分类
    lib = normalizeLibrary({ categories: ["Text", "List", "Uncategorized", "Real"], tags: [] });
    check("桶名不能成为分类", lib.categories.length === 1 && lib.categories[0] === "Real");
    // 标签 cat 写桶名 -> 落到桶且 List 桶传递 kind；Text 桶不得反向强制（显式 list 保留）
    lib = normalizeLibrary({ tags: [{ name: "t1", cat: "List", text: "x" }, { name: "t2", cat: "Text", kind: "list", text: "y" }] });
    check("List 桶名传递 kind 且落桶", lib.tags.find((t) => t.name === "t1").kind === "list" && lib.tags.find((t) => t.name === "t1").cat === "");
    check("Text 桶名不强制 kind（显式 list 保留）", lib.tags.find((t) => t.name === "t2").kind === "list" && lib.tags.find((t) => t.name === "t2").cat === "");

    // ---- normalizeLibrary：catModes / tag.mode ----
    lib = normalizeLibrary({
        categories: ["C", "D"],
        catModes: { C: "order", D: "bogus", NOPE: "random" },
        tags: [{ name: "a", cat: "C", text: "x", mode: "random" }, { name: "b", cat: "C", text: "y" }],
    });
    check("catModes 规范化（含大小写 canon）", catMode("C", lib) === "order" && catMode("D", lib) === DEFAULT_MODE && catMode("NOPE", lib) === DEFAULT_MODE);
    check("未知分类模式不进入 catModes", !("NOPE" in lib.catModes));
    check("tag.mode 非默认写入", tagMode(lib.tags.find((t) => t.name === "a")) === "random");
    check("tag.mode 默认不写入", !("mode" in lib.tags.find((t) => t.name === "b")));
    check("模式常量", MODES.join() === "shuffle,random,order" && DEFAULT_MODE === "shuffle");
    check("cleanMode/hasPosition", cleanMode("nope") === "shuffle" && hasPosition("random") === false && hasPosition("order") === true);

    // ---- reorder（同侧排序）----
    lib = normalizeLibrary({
        categories: ["T1", "L1", "T2", "L2"],
        listCats: ["L1", "L2"],
        tags: [],
    });
    check("catsOnSameSide", catsOnSameSide(lib, "T1").join() === "T1,T2" && catsOnSameSide(lib, "L1").join() === "L1,L2");
    let next = reorderCategoryStep(lib, "T2", -1);
    check("同侧上移", next && next.join() === "T2,L1,T1,L2");
    check("canMoveCategory 首行不能上移", canMoveCategory(lib, "T1", -1) === false && canMoveCategory(lib, "T2", -1) === true);
    next = reorderCategoryTo(lib, "L2", "T2", true);
    check("跨侧排序拒绝", next === null);
    next = reorderCategoryTo(lib, "T2", "T1", true);
    check("同侧排序到位", next && next[0] === "T2" && next[1] === "T1");
    next = reorderCategoryTo(lib, "T2", "T1", false);
    check("同侧已就位 -> null", next === null);

    // ---- isSameAsStored ----
    lib = normalizeLibrary({ categories: ["C"], tags: [{ name: "a", cat: "C", text: "x" }] });
    check("isSameAsStored 一致", isSameAsStored(lib, { categories: ["C"], tags: [{ name: "a", cat: "C", text: "x" }] }) === true);
    check("isSameAsStored 不一致", isSameAsStored(lib, { categories: ["C"], tags: [{ name: "a", cat: "C", text: "y" }] }) === false);

    // ---- 导出 ----
    lib = normalizeLibrary({
        categories: ["Styles", "Animals"],
        listCats: ["Animals"],
        catModes: { Styles: "order" },
        tags: [
            { name: "fox", cat: "Animals", kind: "list", text: "a\nb" },
            { name: "oil", cat: "Styles", text: "x" },
            { name: "loose", text: "uncat" },
        ],
    });
    const allJson = exportLibraryJSON(lib, null);
    check("导出全部可回读", JSON.parse(allJson).tags.length === 3);
    const catJson = JSON.parse(exportLibraryJSON(lib, "Styles"));
    check("导出单分类只含该分类", catJson.tags.length === 1 && catJson.tags[0].name === "oil" && catJson.categories.length === 1);
    const listJson = JSON.parse(exportLibraryJSON(lib, "Animals"));
    check("导出 List 分类带 listCats", listJson.listCats.length === 1 && listJson.tags[0].kind === "list");
    const bucketJson = JSON.parse(exportLibraryJSON(lib, TEXT_BUCKET));
    check("导出桶（无分类标签）", bucketJson.tags.length === 1 && bucketJson.tags[0].name === "loose" && bucketJson.categories.length === 0);

    // ---- 导入解析 ----
    let parsed = parseImport(JSON.stringify({ tags: [{ name: "new1", text: "x" }, { name: "new2", text: "y" }] }), lib);
    check("parseImport 无冲突", parsed.error == null && parsed.conflicts.length === 0 && parsed.data.tags.length === 2);
    const clashFile = JSON.stringify({ tags: [{ name: "fox", cat: "Animals", kind: "list", text: "new" }, { name: "brandnew", text: "z" }] });
    parsed = parseImport(clashFile, lib);
    check("parseImport 冲突检测", parsed.conflicts.length === 1 && parsed.conflicts[0] === "fox");
    check("parseImport 非法 JSON", parseImport("not json", lib).error != null);
    check("parseImport 空文件", parseImport("[]", lib).error != null);
    const dupFile = JSON.stringify({ tags: [{ name: "dup", text: "1" }, { name: "dup", text: "2" }, { name: "dup", text: "3" }] });
    parsed = parseImport(dupFile, lib);
    check("parseImport 文件内重名去重", parsed.data.tags.map((t) => t.name).join() === "dup,dup-2,dup-3");
    const cjkFile = JSON.stringify({ tags: [{ name: "中文", text: "x" }, { name: "ok", text: "y" }] });
    parsed = parseImport(cjkFile, lib);
    check("parseImport 不可用名称计数", parsed.dropped === 1 && parsed.data.tags.length === 1);
    check("importCategories 计数", importCategories(parsed).length === 1);

    // ---- 导入收窄 / 应用 ----
    lib = normalizeLibrary({ categories: ["A", "B"], tags: [{ name: "mine", cat: "A", text: "old" }] });
    parsed = parseImport(JSON.stringify({
        categories: ["A", "B"],
        tags: [{ name: "mine", cat: "A", text: "new" }, { name: "fresh", cat: "B", text: "f" }],
    }), lib);
    const sub = subsetImport(parsed, ["A", "B"], lib);
    check("subsetImport 冲突重算", sub.conflicts.length === 1);
    let r2 = applyImportData(lib, parsed, "both");
    check("导入 both：冲突改名保留全部", r2.data.tags.some((t) => t.name === "mine-2") && r2.data.tags.some((t) => t.name === "fresh") && r2.added === 2 && r2.replaced === 0);
    r2 = applyImportData(lib, parsed, "replace");
    check("导入 replace：覆盖我方", r2.replaced === 1 && r2.data.tags.find((t) => t.name === "mine").text === "new");
    r2 = applyImportData(lib, parsed, "skip");
    check("导入 skip：只加不冲突", r2.added === 1 && r2.replaced === 0 && r2.data.tags.some((t) => t.name === "fresh"));
    const catMerge = applyImportData(normalizeLibrary({ tags: [] }), { data: { categories: ["NewCat"], listCats: ["NewCat"], catModes: {}, tags: [] } }, "both");
    check("导入合并分类", catMerge.data.categories.includes("NewCat") && catMerge.data.listCats.includes("NewCat"));

    // ---- tagLines / isListTag / catOf ----
    check("tagLines 按行拆分", tagLines("a\r\n  b  \n\nc").join("|") === "a|b|c");
    check("tagLines 空文本", tagLines("").length === 0);
    check("isListTag", isListTag({ kind: "list" }) === true && isListTag({}) === false);
    check("catOf 无分类返回该侧桶名", catOf({ name: "x" }) === "Text" && catOf({ name: "x", kind: "list" }) === "List");

    // ---- uniqueTagName（新签名：base, tags, ignore）----
    check("uniqueTagName 基础", uniqueTagName("tag", [{ name: "other" }]) === "tag");
    check("uniqueTagName 冲突加后缀", uniqueTagName("tag", [{ name: "tag" }]) === "tag-2");
    check("uniqueTagName 连续冲突", uniqueTagName("tag", [{ name: "tag" }, { name: "tag-2" }]) === "tag-3");
    check("uniqueTagName 大小写不敏感", uniqueTagName("Tag", [{ name: "tag" }]) === "Tag-2");
    check("uniqueTagName 非法名回退", uniqueTagName("!!", [{ name: "other" }]) === "tag");
    const tagRef = { name: "tag" };
    check("uniqueTagName 忽略自身", uniqueTagName("tag", [tagRef], tagRef) === "tag");

    // ---- cleanTagName ----
    check("cleanTagName 清洗", cleanTagName("  a b! ") === "ab");
    check("cleanTagName 非 ASCII 为空", cleanTagName("中文") === "");

    console.log("\nFAILURES:", failures.length);
    fs.unlinkSync(tmpMjs);
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("test crashed:", e);
    process.exit(1);
});
