// simple_math.js SFNumber number_type↔value 联动冒烟测试（Node 直接运行：node tests/test_number_js.js）
// 验证：
//   - nodeCreated 默认 FLOAT：value options step/round/precision = 小数档
//   - combo 切 INT：options 切整数档 + 当前值取整 + 原回调不被吞
//   - FLOAT↔PERCENT 切档按规范值 ×100/÷100 换算；PERCENT 整数书写
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
    // simple_math.js 相对导入 sf_dynamic_slots.js → 两文件同目录暂存
    for (const f of ["simple_math.js", "sf_dynamic_slots.js"]) {
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
        setDirtyCanvas() {},
    };
}

const mod = stageModule();
const ext = globalThis.app._ext;

// ── 1. nodeCreated：默认 FLOAT 保持小数档 ──
{
    const n = makeNumberNode("FLOAT", 0.5);
    ext.nodeCreated(n);
    const v = n.widgets[1];
    check("默认 FLOAT step=0.01", v.options.step === 0.01);
    check("默认 FLOAT precision=2", v.options.precision === 2);
    check("默认 FLOAT 值不动", v.value === 0.5);
}

// ── 2. combo 切 INT：options 切整数档 + 当前值取整 + 原回调不被吞 ──
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
    check("原回调不被吞", origCalled);
}

// ── 3. 切回 FLOAT / PERCENT：档位恢复 + 规范值换算 ──
{
    const n = makeNumberNode("FLOAT", 1.5);
    ext.nodeCreated(n);
    n.widgets[0].value = "INT";
    n.widgets[0].callback.call(n.widgets[0], "INT");
    n.widgets[0].value = "FLOAT";
    n.widgets[0].callback.call(n.widgets[0], "FLOAT");
    let v = n.widgets[1];
    check("切回 FLOAT step=0.01", v.options.step === 0.01);
    check("切回 FLOAT step2=0.01", v.options.step2 === 0.01);
    check("切回 FLOAT 值保留", v.value === 2);

    // INT 2 → PERCENT 200（×100）
    n.widgets[0].value = "PERCENT";
    n.widgets[0].callback.call(n.widgets[0], "PERCENT");
    check("PERCENT 整数书写 step=1", v.options.step === 1 && v.options.precision === 0);
    check("INT 2 → PERCENT 200（×100）", v.value === 200);

    // FLOAT 0.5 → PERCENT 50（×100）
    const p = makeNumberNode("FLOAT", 0.5);
    ext.nodeCreated(p);
    p.widgets[0].value = "PERCENT";
    p.widgets[0].callback.call(p.widgets[0], "PERCENT");
    check("FLOAT 0.5 → PERCENT 50（×100）", p.widgets[1].value === 50);

    // PERCENT 150 → FLOAT 1.5（÷100）
    const m = makeNumberNode("PERCENT", 150);
    ext.nodeCreated(m);
    m.widgets[0].value = "FLOAT";
    m.widgets[0].callback.call(m.widgets[0], "FLOAT");
    check("PERCENT 150 → FLOAT 1.5（÷100）", m.widgets[1].value === 1.5);
    check("切走 PERCENT 恢复小数档", m.widgets[1].options.step === 0.01);

    // PERCENT 150 → INT 2（÷100 后取整）
    const k = makeNumberNode("PERCENT", 150);
    ext.nodeCreated(k);
    k.widgets[0].value = "INT";
    k.widgets[0].callback.call(k.widgets[0], "INT");
    check("PERCENT 150 → INT 2（÷100 取整）", k.widgets[1].value === 2);
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
}
{
    // PERCENT 存量值原样恢复（不 ×100/÷100）
    const n = makeNumberNode("FLOAT", 0.5);
    n.configure = function () {
        n.widgets[0].value = "PERCENT";
        n.widgets[1].value = 150;
    };
    ext.nodeCreated(n);
    n.configure({});
    const v = n.widgets[1];
    check("configure PERCENT 档位整数书写", v.options.step === 1);
    check("configure PERCENT 存量值不换算 150", v.value === 150);
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
