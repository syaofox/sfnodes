// sf_cache_name_lib 纯逻辑测试（Node 直接运行：node tests/test_cache_name_lib.mjs）
// 覆盖：NEW_LABEL / buildNameOptions（保序去重 + 当前值插入 + 新建入口）/
// isNameTextLinked（link / links 两形态）/ applyNameOverrideState（置灰与恢复）/
// installCacheNameList（callback patch + 刷新按钮去重 + 文本覆盖 hook）。
// 依赖链拷贝为 .mjs（复用 sf_dynamic_slots.isSlotConnected，相对导入改 .mjs，
// test_lora_browser_smoke.js 同款手法）。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_cache_name_lib_"));
for (const n of ["sf_cache_name_lib.js", "sf_dynamic_slots.js"]) {
    const code = fs
        .readFileSync(path.join(here, "..", "web", n), "utf8")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const L = await import(pathToFileURL(path.join(tmpDir, "sf_cache_name_lib.mjs")).href);
const NEW = L.NEW_LABEL;

check("NEW_LABEL 常量", NEW === "＋ 新建缓存…");

check("空列表只留新建入口", JSON.stringify(L.buildNameOptions([], "")) === JSON.stringify([NEW]));
check("已有名保序 + 新建入口",
    JSON.stringify(L.buildNameOptions(["b", "a"], "")) === JSON.stringify(["b", "a", NEW]));
check("当前值不在列表则插入队首",
    JSON.stringify(L.buildNameOptions(["b"], "drive")) === JSON.stringify(["drive", "b", NEW]));
check("当前值已在列表不重复",
    JSON.stringify(L.buildNameOptions(["b", "drive"], "drive")) === JSON.stringify(["b", "drive", NEW]));
check("空/新建入口值不入列表",
    JSON.stringify(L.buildNameOptions(["", NEW, "b"], NEW)) === JSON.stringify(["b", NEW]));
check("非字符串名归一化",
    JSON.stringify(L.buildNameOptions([1, 2], "")) === JSON.stringify(["1", "2", NEW]));

// ── isNameTextLinked（复用 isSlotConnected：link 数字 / links 数组 / 空值）──
const mkInputNode = (link, links) => ({ inputs: [{ name: "name_text", link, links }] });
check("未连接 link=null", L.isNameTextLinked(mkInputNode(null)) === false);
check("未连接 link=-1", L.isNameTextLinked(mkInputNode(-1)) === false);
check("未连接 link=undefined", L.isNameTextLinked(mkInputNode(undefined)) === false);
check("已连接 link 数字", L.isNameTextLinked(mkInputNode(5)) === true);
check("已连接 links 数组", L.isNameTextLinked(mkInputNode(null, [7])) === true);
check("无输入/空节点不崩", L.isNameTextLinked({ inputs: [] }) === false && L.isNameTextLinked(null) === false);

// ── applyNameOverrideState（连接置灰 / 断开恢复 / 不覆盖原有 disabled）──
const makeNode = (link = null, disabled = false) => {
    const w = { name: "name", disabled };
    const n = {
        widgets: [w],
        inputs: [{ name: "name_text", link }],
        dirty: 0,
        computed: 0,
        updateComputedDisabled() { this.computed++; },
        setDirtyCanvas() { this.dirty++; },
    };
    return { n, w };
};

{
    const { n, w } = makeNode();
    check("未连接不禁用", L.applyNameOverrideState(n) === false && w.disabled === false);
    n.inputs[0].link = 9;
    check("连接后禁用", L.applyNameOverrideState(n) === true && w.disabled === true);
    check("置灰触发重算与重绘", n.computed === 1 && n.dirty === 1);
    L.applyNameOverrideState(n);
    check("状态未变提前返回不重复重算", n.computed === 1);
    n.inputs[0].link = null;
    check("断开恢复可用", L.applyNameOverrideState(n) === false && w.disabled === false);
}
{
    const { n, w } = makeNode(3, true);
    L.applyNameOverrideState(n);
    n.inputs[0].link = null;
    L.applyNameOverrideState(n);
    check("原有 disabled 断开后保留", w.disabled === true);
}
check("无 name widget 不崩", L.applyNameOverrideState({ widgets: [], inputs: [] }) === false);

// ── installCacheNameList（stub fetch + addWidget，幂等 + 覆盖 hook）──
globalThis.fetch = async () => ({ ok: true, json: async () => ({ names: ["a"] }) });
{
    const w = { name: "name", value: "", options: {} };
    const n = {
        widgets: [w],
        inputs: [{ name: "name_text", link: null }],
        addWidget(type, label, value, cb) { this.widgets.push({ name: label, type, cb }); },
        updateComputedDisabled() {},
        setDirtyCanvas() {},
    };
    L.installCacheNameList(n, { api: "/x" });
    check("安装：patch callback", w._sfCacheNamePatched === true);
    check("安装：刷新按钮 1 个", n.widgets.filter((x) => x.name === "↻ 刷新缓存列表").length === 1);
    L.installCacheNameList(n, { api: "/x" });
    check("重复安装幂等（按钮不重复）", n.widgets.filter((x) => x.name === "↻ 刷新缓存列表").length === 1);
    await new Promise((r) => setTimeout(r, 0));
    check("首次拉取重建 options", JSON.stringify(w.options.values) === JSON.stringify(["a", NEW]));
    n.inputs[0].link = 1;
    n.onConnectionsChange();
    check("连线 hook 驱动置灰", w.disabled === true);
}

if (failures.length) {
    console.log("FAILED:", failures.length, failures);
    process.exit(1);
}
console.log("ALL PASS");
