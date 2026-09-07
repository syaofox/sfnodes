// SFConvert Anything 前端逻辑测试（Node 直接运行：node tests/test_convert_anything_js.js）
// 覆盖：
// - 扩展注册与热重载防双包装
// - onNodeCreated 按初始 combo 值改型 + callback 改型（callback 参数即新值）
// - onAfterGraphConfigured 按 widget 当前值恢复槽型
// - setSlotType 元素替换（Vue shallowReactive 兼容）与未知值回退 "*"
// 复用 any_pack.js 真实源码（setSlotType 真源），仅 mock app 依赖
const fs = require("fs");
const path = require("path");

const stripImports = (f) =>
    fs.readFileSync(path.join(__dirname, "..", "web", f), "utf8")
        .replace(/import[^;]+;/g, "")
        .replace(/export\s+(?=function|const|let|class|var)/g, "");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ---- mocks ----
const capturedExts = [];
const app = {
    graph: { _nodes: [], links: {} },
    canvas: { setDirty: () => {} },
    registerExtension: (ext) => capturedExts.push(ext),
};

// 加载 any_pack.js（setSlotType 真源），尾部追加一行把导出挂到全局供 convert 模块注入
const anyPackCode = stripImports("any_pack.js") + "\nglobalThis.__sfSetSlotType = setSlotType;";
new Function("app", "isSlotConnected", "uniqueName", anyPackCode)(
    app,
    (slot) => slot && (slot.link != null || (Array.isArray(slot.links) && slot.links.length > 0)),
    (slots, selfIndex, base) => base,
);
check("any_pack 扩展已注册（源码加载成功）", capturedExts.some((e) => e.name === "sfnodes.AnyPack"));

new Function("app", "setSlotType", stripImports("sf_convert_anything.js"))(app, globalThis.__sfSetSlotType);

const ext = capturedExts.find((e) => e.name === "sfnodes.ConvertAnything");
check("扩展已注册", ext !== undefined);

// ---- 假节点与注册 ----
const makeNode = () => ({
    widgets: [{ name: "output_type", value: "string", callback: undefined }],
    outputs: [{ name: "output", type: "*", localized_name: "output" }],
});

const nodeType = { prototype: {} };
ext.beforeRegisterNodeDef(nodeType, { name: "SFConvertAnything" });
check("钩子已包装", typeof nodeType.prototype.onNodeCreated === "function"
    && typeof nodeType.prototype.onAfterGraphConfigured === "function");

const node = makeNode();
nodeType.prototype.onNodeCreated.call(node);
check("创建时按默认值改型 STRING", node.outputs[0].type === "STRING");

node.widgets[0].callback("int");
check("callback 改型 INT", node.outputs[0].type === "INT");
node.widgets[0].callback("float");
check("callback 改型 FLOAT", node.outputs[0].type === "FLOAT");
node.widgets[0].callback("boolean");
check("callback 改型 BOOLEAN", node.outputs[0].type === "BOOLEAN");

// 元素替换：Vue shallowReactive 数组需替换元素而非原地改（platform §2.7）
const before = node.outputs[0];
node.widgets[0].callback("string");
check("槽位元素被替换", node.outputs[0] !== before && node.outputs[0].type === "STRING");
check("槽名/显示名不动（保留 output）", node.outputs[0].name === "output" && node.outputs[0].localized_name === "output");

// 未知 combo 值回退通配
node.widgets[0].callback("nonsense");
check("未知值回退 *", node.outputs[0].type === "*");

// 工作流恢复：nodeCreated 早于 widgets_values 恢复，onAfterGraphConfigured 补挂
node.widgets[0].value = "boolean";
nodeType.prototype.onAfterGraphConfigured.call(node);
check("configure 恢复 BOOLEAN", node.outputs[0].type === "BOOLEAN");

// 热重载防双包装：再次注册不再包装（改型行为不变且 callback 未叠加）
const node2 = makeNode();
ext.beforeRegisterNodeDef(nodeType, { name: "SFConvertAnything" });
nodeType.prototype.onNodeCreated.call(node2);
check("二次注册防双包装", node2.outputs[0].type === "STRING" && node2.widgets[0].callback !== undefined);

// 其他节点不受影响
const otherType = { prototype: {} };
ext.beforeRegisterNodeDef(otherType, { name: "SFAnyPack" });
check("其他节点不包装", otherType.prototype.onNodeCreated === undefined);

if (failures.length) {
    console.log(`\n${failures}`);
    process.exit(1);
}
console.log("\nALL PASS");
