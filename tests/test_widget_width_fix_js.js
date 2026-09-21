// SF legacy widget 宽度修复前端冒烟（Node 直接运行：node tests/test_widget_width_fix_js.js）
// 覆盖（.mjs 拷贝链真实加载 sf_widget_width_fix.js）：
// - 扩展注册名 sfnodes.WidgetWidthFix / init·setup 幂等
// - 模块加载即：工厂包装 + 全图扫描（已有节点·subgraph·节点 .subgraph）
// - 之后 addWidget/addCustomWidget 创建的 widget 即时受控
// - Classic 写丢弃读 undefined（回退 nodeWidth）/ Vue 透传 / 动态切档
// - nodeCreated / loadedGraphNode / afterConfigureGraph 三路补扫
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── LGraphNode / LiteGraph 桩 ──
class FakeLGraphNode {
    constructor() { this.widgets = []; }
    addWidget(type, name, value, callback, options) {
        return this.addCustomWidget({ type, name, value, callback, options: options || {}, y: 0 });
    }
    addCustomWidget(widget) { this.widgets.push(widget); return widget; }
}
globalThis.LGraphNode = FakeLGraphNode;
globalThis.LiteGraph = { vueNodesMode: false, LGraphNode: FakeLGraphNode };

// ── app 图桩：已有 widget 的节点 + subgraph 两形态 ──
const preNode = new FakeLGraphNode();
preNode.widgets.push({ name: "pre-existing" });
const subNode = new FakeLGraphNode();
subNode.widgets.push({ name: "in-subgraph" });
const sub = { _nodes: [subNode], subgraphs: new Map() };
const embedNode = new FakeLGraphNode();
embedNode.widgets.push({ name: "embed" });
embedNode.subgraph = sub;
const root = { nodes: [preNode, embedNode], subgraphs: new Map([["s", sub]]) };

let registeredExt = null;
globalThis.app = {
    graph: root,
    registerExtension: (ext) => { registeredExt = ext; },
};

// ── 拷贝依赖链为 .mjs 并真实加载 ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_ww_"));
for (const n of ["sf_widget_width_fix.js", "sf_widget_width_lib.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

(async () => {
    await import(path.join(tmpDir, "sf_widget_width_fix.mjs"));

    check("扩展注册名", !!registeredExt && registeredExt.name === "sfnodes.WidgetWidthFix");

    // ── 模块加载即包装 + 全图扫描 ──
    check("已有节点 widget 受控", !!preNode.widgets[0]._sfWidgetWidthGuarded);
    check("subgraph Map 内节点受控", !!subNode.widgets[0]._sfWidgetWidthGuarded);
    check("节点 .subgraph 分支受控", !!embedNode.widgets[0]._sfWidgetWidthGuarded);

    // ── 工厂路径：新 widget 即时受控 ──
    const viaAddWidget = preNode.addWidget("combo", "mode", "x", () => {});
    const viaCustom = preNode.addCustomWidget({ name: "custom" });
    check("addWidget 新 widget 受控", !!viaAddWidget._sfWidgetWidthGuarded);
    check("addCustomWidget 新 widget 受控", !!viaCustom._sfWidgetWidthGuarded);

    // ── Classic 写丢弃 / Vue 透传 ──
    globalThis.LiteGraph.vueNodesMode = false;
    viaAddWidget.width = 500;
    check("Classic 写丢弃", viaAddWidget.width === undefined);
    check("回退 nodeWidth", (viaAddWidget.width || 240) === 240);
    globalThis.LiteGraph.vueNodesMode = true;
    viaAddWidget.width = 500;
    check("Vue 写透传", viaAddWidget.width === 500);
    globalThis.LiteGraph.vueNodesMode = false;

    // ── init/setup 幂等（不叠包、不抛错）──
    const factoryBefore = FakeLGraphNode.prototype.addWidget;
    registeredExt.init();
    registeredExt.setup();
    check("init/setup 后工厂不叠包", FakeLGraphNode.prototype.addWidget === factoryBefore);

    // ── 三路补扫 ──
    const createdNode = new FakeLGraphNode();
    createdNode.widgets.push({ name: "created" });
    registeredExt.nodeCreated(createdNode);
    check("nodeCreated 补扫", !!createdNode.widgets[0]._sfWidgetWidthGuarded);

    const loadedNode = new FakeLGraphNode();
    loadedNode.widgets.push({ name: "loaded" });
    registeredExt.loadedGraphNode(loadedNode);
    check("loadedGraphNode 补扫", !!loadedNode.widgets[0]._sfWidgetWidthGuarded);

    const afterNode = new FakeLGraphNode();
    afterNode.widgets.push({ name: "after-config" });
    root.nodes.push(afterNode);
    registeredExt.afterConfigureGraph();
    check("afterConfigureGraph 全图补扫", !!afterNode.widgets[0]._sfWidgetWidthGuarded);

    if (failures.length) {
        console.log(`\n${failures.length} FAILED`);
        process.exit(1);
    }
    console.log("\nALL PASSED");
})();
