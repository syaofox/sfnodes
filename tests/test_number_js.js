// simple_math.js SFNumber number_type↔value/输出槽 联动冒烟测试
// （Node 直接运行：node tests/test_number_js.js）
// 验证：
//   - nodeCreated 默认 FLOAT：value options step/round/precision = 小数档，输出槽 FLOAT/float
//   - combo 切 INT：options 切整数档 + 当前值取整 + 输出槽 INT/int + 原回调不被吞
//   - 切回 FLOAT：options 恢复 + 输出槽 FLOAT/float（PERCENT 档已移除）
//   - configure 工作流恢复：按保存的 number_type 重应用档位（不触发 callback、不换算存量值）
//   - 非 SFNumber 节点不受影响
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM/app（仅注册期需要）──
globalThis.document = {
    createElement() { return { style: {}, addEventListener() {}, click() {}, remove() {} }; },
    body: { appendChild() {} },
};
globalThis.app = {
    registerExtension(ext) { this._ext = ext; },
};

function stageModule() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sf_number_"));
    // simple_math.js 相对导入 sf_dynamic_slots.js + any_pack.js（后者再依赖 sf_dynamic_slots）
    for (const f of ["simple_math.js", "sf_dynamic_slots.js", "any_pack.js"]) {
        const code = fs
            .readFileSync(path.join(__dirname, "..", "web", f), "utf8")
            .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;");
        fs.writeFileSync(path.join(tmp, f), code);
    }
    return require(path.join(tmp, "simple_math.js"));
}

function makeNumberNode(numberType, value) {
    return {
        comfyClass: "SFNumber",
        widgets: [
            { name: "number_type", type: "combo", value: numberType, options: {}, callback: undefined },
            { name: "value", type: "number", value, options: { step: 0.01, round: 0.01, precision: 2 } },
        ],
        outputs: [{ type: "*", name: "value", localized_name: "value" }],
        setDirtyCanvas() {},
    };
}

function out(n) {
    return n.outputs[0];
}

const mod = stageModule();
const ext = globalThis.app._ext;

// ── 1. nodeCreated：默认 FLOAT 保持小数档 + 输出槽 FLOAT/float ──
{
    const n = makeNumberNode("FLOAT", 0.5);
    ext.nodeCreated(n);
    const v = n.widgets[1];
    check("默认 FLOAT step=0.01", v.options.step === 0.01);
    check("默认 FLOAT precision=2", v.options.precision === 2);
    check("默认 FLOAT 值不动", v.value === 0.5);
    check("默认输出槽 type=FLOAT", out(n).type === "FLOAT");
    check("默认输出槽 name=float", out(n).name === "float" && out(n).localized_name === "float");
}

// ── 2. combo 切 INT：options 切整数档 + 当前值取整 + 输出槽 INT/int + 原回调不被吞 ──
{
    const n = makeNumberNode("FLOAT", 1.5);
    let origCalled = false;
    n.widgets[0].callback = () => { origCalled = true; };
    ext.nodeCreated(n);
    n.widgets[0].value = "INT";
    n.widgets[0].callback.call(n.widgets[0], "INT");
    const v = n.widgets[1];
    check("INT step=1", v.options.step === 1);
    check("INT step2=1（精调步进不残留 0.01）", v.options.step2 === 1);
    check("INT precision=0", v.options.precision === 0);
    check("INT 当前值取整 1.5→2", v.value === 2);
    check("INT 输出槽 type=INT", out(n).type === "INT");
    check("INT 输出槽 name=int", out(n).name === "int" && out(n).localized_name === "int");
    check("原回调不被吞", origCalled);
}

// ── 3. 切回 FLOAT：档位恢复 + 输出槽恢复 ──
{
    const n = makeNumberNode("FLOAT", 1.5);
    ext.nodeCreated(n);
    n.widgets[0].value = "INT";
    n.widgets[0].callback.call(n.widgets[0], "INT");
    n.widgets[0].value = "FLOAT";
    n.widgets[0].callback.call(n.widgets[0], "FLOAT");
    const v = n.widgets[1];
    check("切回 FLOAT step=0.01", v.options.step === 0.01);
    check("切回 FLOAT step2=0.01", v.options.step2 === 0.01);
    check("切回 FLOAT 值保留", v.value === 2);
    check("切回 FLOAT 输出槽 type=FLOAT", out(n).type === "FLOAT");
    check("切回 FLOAT 输出槽 name=float", out(n).name === "float");
}

// ── 4. configure：按保存的 number_type 重应用（工作流恢复不触发 callback、不换算存量值）──
{
    const n = makeNumberNode("FLOAT", 0.5);
    // 先定义还原逻辑再 nodeCreated（wrapper 捕获 nodeCreated 时的 configure）
    n.configure = function () {
        // 模拟 LiteGraph configure 还原 widgets_values
        n.widgets[0].value = "INT";
        n.widgets[1].value = 2.7;
    };
    ext.nodeCreated(n);
    n.configure({});
    const v = n.widgets[1];
    check("configure 后 INT 档生效", v.options.step === 1 && v.options.precision === 0);
    check("configure 后值取整 2.7→3", v.value === 3);
    check("configure 后输出槽 type=INT", out(n).type === "INT");
    check("configure 后输出槽 name=int", out(n).name === "int");
}

// ── 5. 非 SFNumber 节点不受影响 ──
{
    const n = { comfyClass: "SFOther", widgets: [] };
    ext.nodeCreated(n);
    check("非 SFNumber 不挂 configure 包装", typeof n.configure === "undefined");
}

if (failures.length) {
    console.log(`\n${failures.length} FAILED`);
    process.exit(1);
} else {
    console.log("\nALL PASSED");
}
