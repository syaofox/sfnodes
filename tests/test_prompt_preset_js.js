// SFPromptPreset 前端逻辑测试（Node 直接运行：node tests/test_prompt_preset_js.js）
// 覆盖：pose/couple 互斥联动、预设 description 动态写入 widget.tooltip
const fs = require("fs");
const path = require("path");

const code = fs
    .readFileSync(path.join(__dirname, "..", "web", "prompt_preset.js"), "utf8")
    .replace('import { app } from "/scripts/app.js";', "")
    .replace('import { api } from "/scripts/api.js";', "");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const capturedExts = [];
const descData = {
    Outfit: { 旗袍: "Form-fitting qipao dress" },
    Pose: { 回眸: "Looking back over the shoulder" },
    Environment: { 现代地铁车厢: "Contemporary subway interior" },
};
const app = {
    graph: { _nodes: [] },
    registerExtension: (ext) => capturedExts.push(ext),
};
const api = { fetchApi: async () => ({ ok: true, json: async () => descData }) };
new Function("app", "api", code)(app, api);

const mainExt = capturedExts.find((e) => e.name === "sfnodes.prompt_preset");
check("存在扩展", mainExt !== undefined);

(async () => {
    await mainExt.setup();
    check("setup 拉取描述映射", true);

    // ---------- 互斥联动 + tooltip 动态化 ----------
    const nodeType = { prototype: {} };
    mainExt.beforeRegisterNodeDef(nodeType, { name: "SFPromptPreset" });
    const otherType = { prototype: {} };
    mainExt.beforeRegisterNodeDef(otherType, { name: "OtherNode" });
    check("其他节点不受影响", otherType.prototype.onNodeCreated === undefined);

    function fakeWidget(name, initial = "禁用") {
        return { name, value: initial, callback: null, tooltip: "initial-tooltip" };
    }
    function makeNode() {
        const n = {
            widgets: [
                fakeWidget("outfit_preset"),
                fakeWidget("pose_preset"),
                fakeWidget("couple_preset"),
                fakeWidget("environment_preset"),
                Object.assign(fakeWidget("input_text"), { type: "string" }),
            ],
        };
        nodeType.prototype.onNodeCreated.call(n);
        return n;
    }

    const n0 = makeNode();
    const outfit0 = n0.widgets.find((w) => w.name === "outfit_preset");
    const pose0 = n0.widgets.find((w) => w.name === "pose_preset");
    const couple0 = n0.widgets.find((w) => w.name === "couple_preset");
    const env0 = n0.widgets.find((w) => w.name === "environment_preset");
    const text0 = n0.widgets.find((w) => w.name === "input_text");
    check("预设 widget 均有 callback 包装", ["outfit_preset", "pose_preset", "couple_preset", "environment_preset"]
        .every((n) => typeof n0.widgets.find((w) => w.name === n).callback === "function"));
    check("非预设 widget 不包装", text0.callback === null);
    check("互斥：pose 与 couple 已联动", typeof pose0.callback === "function" && typeof couple0.callback === "function");

    // 选中值变化 → tooltip 更新为 description
    outfit0.value = "旗袍";
    outfit0.callback();
    check("outfit tooltip 显示 description", outfit0.tooltip === "Form-fitting qipao dress");
    pose0.value = "回眸";
    pose0.callback();
    check("pose tooltip 显示 description", pose0.tooltip === "Looking back over the shoulder");
    check("选 pose 后 couple 置禁用", couple0.value === "禁用");
    env0.value = "现代地铁车厢";
    env0.callback();
    check("environment tooltip 显示 description", env0.tooltip === "Contemporary subway interior");

    // 置回禁用 → tooltip 清空
    pose0.value = "禁用";
    pose0.callback();
    check("置禁用后 tooltip 清空", pose0.tooltip === null);

    // 随机：无 description → tooltip 清空（不残留）
    pose0.value = "随机";
    pose0.callback();
    check("随机无 description 清空 tooltip", pose0.tooltip === null);

    // 互斥不影响 tooltip 链
    couple0.value = "拥抱";
    couple0.callback();
    check("选 couple 后 pose 置禁用", pose0.value === "禁用");

    // ---------- setup 后已有节点的 tooltip 同步 ----------
    app.graph._nodes = [{
        type: "SFPromptPreset",
        widgets: [fakeWidget("pose_preset", "回眸")],
    }];
    await mainExt.setup();
    const synced = app.graph._nodes[0].widgets[0];
    check("setup 同步已有节点 tooltip", synced.tooltip === "Looking back over the shoulder");
    app.graph._nodes = [];

    console.log("\nFAILURES:", failures.length);
    process.exit(failures.length ? 1 : 0);
})();
