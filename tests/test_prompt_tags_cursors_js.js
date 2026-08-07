// SFPromptTags Picks 游标系统测试（Node 直接运行：node tests/test_prompt_tags_cursors_js.js）
// 覆盖：nextIndex 三种模式（order 每 run 推进一次 / shuffle 发牌不重复 / random 真随机）、
//       同 build 内多次使用发新牌、_pending 持有语义（未入队 build 不消耗）、
//       commitPicks 只消耗被入队 build、cursorInfo、resetCursor、renameCursor、池变更重启
const fs = require("fs");
const os = require("os");
const path = require("path");

// cursors import ./sf_prompt_tags_lib.js，二者都复制到同一 tmp 目录；
// /scripts/app.js 替换为 globalThis.app（测试先例）
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ptg_cursors_"));
const libCode = fs.readFileSync(path.join(__dirname, "..", "web", "sf_prompt_tags_lib.js"), "utf8");
fs.writeFileSync(path.join(tmpDir, "sf_prompt_tags_lib.mjs"), libCode);
const cursorsCode = fs
    .readFileSync(path.join(__dirname, "..", "web", "sf_prompt_tags_cursors.js"), "utf8")
    .replace('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
    .replace('from "./sf_prompt_tags_lib.js"', 'from "./sf_prompt_tags_lib.mjs"');
fs.writeFileSync(path.join(tmpDir, "sf_prompt_tags_cursors.mjs"), cursorsCode);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// settings mock（内存对象模拟 comfy.settings.json）
const settingsStore = {};
globalThis.app = {
    ui: {
        settings: {
            getSettingValue: (k) => (k in settingsStore ? settingsStore[k] : null),
            setSettingValueAsync: (k, v) => { settingsStore[k] = v; return Promise.resolve(); },
            setSettingValue: (k, v) => { settingsStore[k] = v; },
        },
    },
};
globalThis.window = { addEventListener: () => {} };

(async () => {
    const C = await import(path.join(tmpDir, "sf_prompt_tags_cursors.mjs"));
    const { nextIndex, cursorInfo, resetCursor, renameCursor, beginPickBuild, commitPicks, listKey, catKey, flushCursors } = C;

    const NEW = () => ({});
    const commitFor = (p) => commitPicks(p);

    // ---- random：真随机 + 不落盘 ----
    {
        const p = NEW();
        beginPickBuild(p);
        const i = nextIndex("list:r", 5, "random");
        check("random 返回界内下标", i >= 0 && i < 5);
        commitFor(p);
        check("random 无可存位置（不写设置）", !("sfnodes.PromptTags.Cursors" in settingsStore));
    }

    // ---- order：每 run 推进一次；同一 build 内所有使用同值 ----
    {
        const p = NEW();
        beginPickBuild(p);
        check("order 首次 = 0", nextIndex("list:seq", 3, "order") === 0);
        check("order 同 build 再使用仍 0", nextIndex("list:seq", 3, "order", 7) === 0);
        commitFor(p);
        const p2 = NEW();
        beginPickBuild(p2);
        check("order 第二个 run = 1", nextIndex("list:seq", 3, "order") === 1);
        check("order 位置信息", cursorInfo("list:seq", 3, "order") === "next 2 of 3");
        commitFor(p2);
        const p3 = NEW();
        beginPickBuild(p3);
        check("order 循环回绕", nextIndex("list:seq", 3, "order") === 2);
    }

    // ---- shuffle：同 build 多次使用各发一张新牌，且不重复 ----
    {
        const p = NEW();
        beginPickBuild(p);
        const a = nextIndex("list:deck", 3, "shuffle");
        const b = nextIndex("list:deck", 3, "shuffle", 1);
        const c = nextIndex("list:deck", 3, "shuffle", 2);
        const s = [a, b, c].sort((x, y) => x - y);
        check("shuffle 同 build 三张不同牌", s.join() === "0,1,2");
        commitFor(p);
        check("shuffle 落盘后位置", cursorInfo("list:deck", 3, "shuffle") === "3 left in the deck");
    }

    // ---- 持有语义：未入队 build 的选择不被消耗（反复给同一次 run）----
    {
        const pExp = NEW();
        beginPickBuild(pExp);
        const kept = nextIndex("list:hold", 3, "order");   // 0
        // 该 build 从未入队（无 commitPicks）
        const pRun = NEW();
        beginPickBuild(pRun);
        const got = nextIndex("list:hold", 3, "order");    // 复用持有值
        check("未入队 build 的选择被复用", got === kept && got === 0);
        commitFor(pRun);
        const pNext = NEW();
        beginPickBuild(pNext);
        check("真实 run 后推进", nextIndex("list:hold", 3, "order") === 1);
        commitFor(pNext);
    }

    // ---- 池变更重启 ----
    {
        const p = NEW();
        beginPickBuild(p);
        nextIndex("list:grow", 2, "order");
        commitFor(p);
        const p2 = NEW();
        beginPickBuild(p2);
        check("池尺寸变化后从头", nextIndex("list:grow", 3, "order") === 0);
        commitFor(p2);
    }

    // ---- 单选项 ----
    {
        const p = NEW();
        beginPickBuild(p);
        check("单选项恒为 0", nextIndex("list:one", 1, "shuffle") === 0 && nextIndex("list:one", 1, "order") === 0);
    }

    // ---- resetCursor：从头开始 ----
    {
        const p = NEW();
        beginPickBuild(p);
        nextIndex("list:rst", 3, "order");
        commitFor(p);
        resetCursor("list:rst");
        const p2 = NEW();
        beginPickBuild(p2);
        check("reset 后回到 0", nextIndex("list:rst", 3, "order") === 0);
        commitFor(p2);
    }

    // ---- renameCursor：位置迁移 ----
    {
        const p = NEW();
        beginPickBuild(p);
        nextIndex("list:oldname", 4, "order");
        commitFor(p);
        renameCursor("list:oldname", "list:newname");
        check("改名后位置跟随", cursorInfo("list:newname", 4, "order") === "next 2 of 4");
        check("旧键清除", cursorInfo("list:oldname", 4, "order") === "next 1 of 4");
    }

    // ---- catKey 与 listKey 键形 ----
    check("键形", listKey("Fruit") === "list:fruit" && catKey("Styles") === "cat:styles");

    flushCursors();
    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("test crashed:", e);
    process.exit(1);
});
