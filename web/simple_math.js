import { app } from "/scripts/app.js";
import { installDynamicSlots } from "./sf_dynamic_slots.js";
import { setSlotType } from "./any_pack.js";

const LETTERS = "abcdefghijklmnopqrstuvwxyz";

// SFNumber：number_type 切换时调整 value widget 的步进/精度（INT 整数、FLOAT 小数）
// ⚠ 前端创建 widget 时会按 step 派生 step2（精调步进，修饰键拖拽用）且不随 step 联动，须一并覆盖
const SF_NUMBER_OPTS = {
    INT: { step: 1, step2: 1, round: 1, precision: 0 },
    FLOAT: { step: 0.01, step2: 0.01, round: 0.01, precision: 2 },
};

// 输出槽改型：随 number_type 同步槽类型+槽名（复用 any_pack.setSlotType，patterns §41 同款）
const SF_NUMBER_SOCKET = {
    INT: { type: "INT", name: "int" },
    FLOAT: { type: "FLOAT", name: "float" },
};

// 切档换算：规范值 q = FLOAT 语义量（INT 档回填取整）
const SF_NUMBER_TO_Q = {
    INT: (v) => v,
    FLOAT: (v) => v,
};
const SF_NUMBER_FROM_Q = {
    INT: (q) => Math.round(q),
    FLOAT: (q) => q,
};

function findNumberWidgets(node) {
    const find = (n) => node.widgets?.find((w) => w.name === n);
    const typeW = find("number_type");
    const valueW = find("value");
    return typeW && valueW ? { typeW, valueW } : null;
}

function syncOutputType(node, mode) {
    // 输出槽类型+槽名随 number_type 同步（元素替换，Vue 渲染生效）
    const m = SF_NUMBER_SOCKET[mode] || SF_NUMBER_SOCKET.FLOAT;
    setSlotType(node, node.outputs, 0, m.type, { name: m.name, localized_name: m.name });
}

function applyNumberMode(node) {
    // 仅重应用 options 档位 + INT 档取整（工作流恢复路径也走这里，不换算数值）
    const w = findNumberWidgets(node);
    if (!w) return;
    const opts = SF_NUMBER_OPTS[w.typeW.value] || SF_NUMBER_OPTS.FLOAT;
    Object.assign(w.valueW.options, opts);
    if (w.typeW.value === "INT" && typeof w.valueW.value === "number") {
        w.valueW.value = Math.round(w.valueW.value);
    }
    syncOutputType(node, w.typeW.value);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
}

function switchNumberMode(node) {
    // 用户显式切档：经规范值 q 换算回填（工作流恢复不走此路径，避免改写存量值）
    const w = findNumberWidgets(node);
    if (!w) return;
    const prevMode = node._sfNumberMode || w.typeW.value;
    if (prevMode !== w.typeW.value && typeof w.valueW.value === "number") {
        const q = (SF_NUMBER_TO_Q[prevMode] || SF_NUMBER_TO_Q.FLOAT)(w.valueW.value);
        w.valueW.value = (SF_NUMBER_FROM_Q[w.typeW.value] || SF_NUMBER_FROM_Q.FLOAT)(q);
    }
    node._sfNumberMode = w.typeW.value;
    applyNumberMode(node);
}

function setupSFNumber(node) {
    const w = findNumberWidgets(node);
    if (!w) return;
    node._sfNumberMode = w.typeW.value;
    // combo 切换联动（包装 callback，不吞原回调）
    const origCb = w.typeW.callback;
    w.typeW.callback = function (...args) {
        if (origCb) origCb.apply(this, args);
        switchNumberMode(node);
    };
    // 工作流恢复不触发 callback：configure 值还原后按保存的 number_type 重应用档位（不换算）
    const origConfigure = node.configure;
    node.configure = function (...args) {
        if (origConfigure) origConfigure.apply(this, args);
        const w2 = findNumberWidgets(node);
        if (w2) node._sfNumberMode = w2.typeW.value;
        applyNumberMode(node);
    };
    // 初始（默认 FLOAT）
    applyNumberMode(node);
}

app.registerExtension({
    name: "sfnodes.SimpleMath",

    nodeCreated(node) {
        if (node.comfyClass === "SFSimpleMath" || node.comfyClass === "SFSimpleMathCondition") {
            installDynamicSlots(node, {
                inputMatch: (name) => /^[a-z]$/.test(name),
                inputCount: 26,
                inputType: "*",
                initialInputs: 2,
                nameFor: (cfg, count) => LETTERS[count],
            });
            return;
        }
        if (node.comfyClass === "SFNumber") {
            setupSFNumber(node);
        }
    },
});
