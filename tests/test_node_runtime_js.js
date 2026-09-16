// SF Node Runtime 前端冒烟测试（Node 直接运行：node tests/test_node_runtime_js.js）
// 覆盖（.mjs 拷贝链真实加载）：
// - 扩展注册 sfnodes.NodeRuntime + 设置注册（默认关）
// - 关闭时 executing 事件不产生耗时记录 / badge
// - Classic（非 Vue）：间隔结算写入 node.executionDuration + onDrawForeground 绘制 "x.xxx s"
// - Vue Nodes 2.0：间隔结算写入原生 node.badges（LGraphBadge），并保留非运行时 badge
// - execution_start 清空；onChange(false) 立即移除
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── 受控时间 ──
let NOW = 1000;
const realNow = Date.now;
Date.now = () => NOW;

// ── LGraphBadge / LiteGraph ──
class FakeBadge {
    constructor(opts) { Object.assign(this, opts); }
}
globalThis.LGraphBadge = FakeBadge;
globalThis.LiteGraph = {
    vueNodesMode: false,
    NODE_TITLE_COLOR: "#999",
    NODE_DEFAULT_BGCOLOR: "#353535",
    NODE_TITLE_HEIGHT: 30,
};

// ── app / api stubs ──
const SETTING_ID = "sfnodes.NodeRuntime.Enabled";
let settingValue = false;
let registeredSetting = null;
let registeredExt = null;

const nodes = [];
function mkNode(id) { return { id, badges: undefined, executionDuration: undefined, flags: {}, setDirtyCanvas() {}, graph: null }; }
const n1 = mkNode(1);
const n2 = mkNode(2);
nodes.push(n1, n2);

const graph = {
    _nodes: nodes,
    getNodeById(id) { return nodes.find((n) => String(n.id) === String(id)) || null; },
    trigger() {},
};
n1.graph = graph;
n2.graph = graph;

globalThis.app = {
    graph,
    canvas: { setDirty() {} },
    ui: {
        settings: {
            getSettingValue: (id) => (id === SETTING_ID ? settingValue : undefined),
            addSetting: (spec) => { registeredSetting = spec; },
        },
    },
    registerExtension: (e) => { registeredExt = e; },
};

const apiHandlers = {};
globalThis.api = {
    addEventListener(t, fn) { (apiHandlers[t] = apiHandlers[t] || []).push(fn); },
    removeEventListener() {},
};

function fire(type, detail) {
    for (const fn of apiHandlers[type] || []) fn({ detail });
}

// ── 拷贝依赖链为 .mjs 并真实加载 ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_nr_"));
for (const n of ["sf_node_runtime.js", "sf_node_runtime_lib.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

function makeCtx() {
    return {
        font: "", fillStyle: "", textAlign: "", textBaseline: "",
        drawn: [],
        save() {}, restore() {},
        beginPath() {}, fill() {}, rect() {}, roundRect() {},
        measureText: (t) => ({ width: String(t).length * 6 }),
        fillText(t) { this.drawn.push(t); },
    };
}

(async () => {
    await import(path.join(tmpDir, "sf_node_runtime.mjs"));

    check("扩展注册名", !!registeredExt && registeredExt.name === "sfnodes.NodeRuntime");
    registeredExt.init();
    check("设置注册 id + 默认关",
        !!registeredSetting && registeredSetting.id === SETTING_ID
        && registeredSetting.type === "boolean" && registeredSetting.defaultValue === false);
    check("execution_start/executing 监听已挂",
        (apiHandlers.execution_start || []).length === 1 && (apiHandlers.executing || []).length === 1);

    // ── 关闭：不记录 ──
    settingValue = false;
    fire("execution_start");
    NOW = 1000;
    fire("executing", 1);
    NOW = 1250;
    fire("executing", 2);
    check("关闭时不记录耗时", n1.executionDuration == null && n2.executionDuration == null);

    // ── Classic 开启：executionDuration 结算 ──
    settingValue = true;
    globalThis.LiteGraph.vueNodesMode = false;
    fire("execution_start");
    NOW = 1000;
    fire("executing", 1);
    NOW = 1250;
    fire("executing", 2);
    check("Classic 节点1 结算 0.25s", n1.executionDuration === 0.25);
    NOW = 2000;
    fire("executing", null);
    check("Classic 节点2 结算 0.75s", n2.executionDuration === 0.75);
    check("Classic 不加 badge", !(n1.badges || []).length && !(n2.badges || []).length);

    // ── onDrawForeground 绘制 ──
    class FakeNodeType {}
    FakeNodeType.prototype.onDrawForeground = function () { this._origCalled = true; };
    registeredExt.beforeRegisterNodeDef(FakeNodeType);
    const inst = new FakeNodeType();
    inst.flags = {};
    inst.executionDuration = 1.25;
    const ctx = makeCtx();
    inst.onDrawForeground(ctx);
    check("Classic 绘制文案 x.xxx s", ctx.drawn.includes("1.250s"));
    check("Classic 原 onDrawForeground 仍调用", inst._origCalled === true);
    // 折叠时不绘制
    const ctx2 = makeCtx();
    const inst2 = new FakeNodeType();
    inst2.flags = { collapsed: true };
    inst2.executionDuration = 9;
    inst2.onDrawForeground(ctx2);
    check("折叠时跳过绘制", !ctx2.drawn.length);
    // 关闭时不绘制
    settingValue = false;
    const ctx3 = makeCtx();
    const inst3 = new FakeNodeType();
    inst3.flags = {};
    inst3.executionDuration = 9;
    inst3.onDrawForeground(ctx3);
    check("关闭时跳过绘制", !ctx3.drawn.length);
    settingValue = true;

    // ── Vue 模式：node.badges ──
    globalThis.LiteGraph.vueNodesMode = true;
    fire("execution_start");
    NOW = 3000;
    fire("executing", 1);
    NOW = 3300;
    fire("executing", 2);
    check("Vue 节点1 结算 0.3s", n1.executionDuration === 0.3);
    check("Vue 生成运行时 badge", !!(n1.badges && n1.badges[0] && n1.badges[0]._sfRuntimeBadge));
    check("Vue badge 文案", n1.badges[0].text === "0.300s");
    // 保留非运行时 badge
    const other = { text: "#2", _other: true };
    n1.badges.push(other);
    fire("execution_start");
    NOW = 4000;
    fire("executing", 1);
    NOW = 4100;
    fire("executing", null);
    check("Vue 非运行时 badge 保留", n1.badges.includes(other));
    check("Vue 运行时 badge 重建", n1.badges.some((b) => b && b._sfRuntimeBadge && b.text === "0.100s"));

    // ── onChange(false) 清除 ──
    registeredSetting.onChange(false);
    check("onChange(false) 清除运行时 badge",
        !n1.badges.some((b) => b && b._sfRuntimeBadge)
        && !(n2.badges || []).some((b) => b && b._sfRuntimeBadge));
    check("onChange(false) 清 executionDuration", n1.executionDuration == null && n2.executionDuration == null);

    Date.now = realNow;
    if (failures.length) {
        console.log(`\n${failures.length} FAILED: ${failures}`);
        process.exit(1);
    }
    console.log("\nALL PASSED");
})().catch((e) => { console.error("FATAL:", e); process.exit(1); });
