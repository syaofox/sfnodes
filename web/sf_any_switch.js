// ==========================================================================
// SF Any Switch - 动态任意输入切换（复刻 rgthree Any Switch）
// ==========================================================================
//
// 后端为灵活 optional schema（任意输入名都按 any_type 放行），本扩展负责：
// - 初始 4 个 any_XX 输入槽；全连 → 追加 1 槽、断开 → 回收尾部空槽
//   （installDynamicSlots，保底 4 槽 = 稳态恒有 1 个空闲槽，上限 20）
// - 类型着色：每个输入槽按自身连线类型改型（复用 any_pack.setSlotType，
//   Vue 前端须元素替换才触发重渲染）；输出槽跟随第一个非 "*" 的输入类型，
//   label 同步为类型名；全部断开回退 "*"
// - workflow 加载/粘贴恢复连线不触发 onConnectionsChange，
//   用 onAfterGraphConfigured 重算（any_pack 同款挂点）
//
// 与 rgthree 的有意差异：不做 reroute 穿透类型推断（followConnectionUntilType），
// 直连场景 link.type 连线时已是具体类型，经 reroute 时保持 "*" 不着色
// （any_pack 同样接受此限制）。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installDynamicSlots } from "./sf_dynamic_slots.js";
import { setSlotType, slotLinkTypes, unionType } from "./any_pack.js";

const ANY_PREFIX = "any_";
const INITIAL_INPUTS = 4;
const MAX_INPUTS = 20;
const OUTPUT_NAME = "value";

const anyName = (count) => ANY_PREFIX + String(count + 1).padStart(2, "0");

// 逐槽重算类型：输入 = 自身连线类型并集；输出 = 第一个非 "*" 的输入类型。
function syncSlotTypes(node) {
    if (!node || !node.graph) return;
    (node.inputs || []).forEach((slot, i) => {
        setSlotType(node, node.inputs, i, unionType(slotLinkTypes(node, slot)));
    });
    let outType = "*";
    for (const slot of node.inputs || []) {
        const t = unionType(slotLinkTypes(node, slot));
        if (t !== "*") {
            outType = t;
            break;
        }
    }
    setSlotType(node, node.outputs, 0, outType, {
        label: outType === "*" ? OUTPUT_NAME : outType,
    });
}

app.registerExtension({
    name: "sfnodes.AnySwitch",

    nodeCreated(node) {
        if (node.comfyClass !== "SFAnySwitch") return;

        // 后端 schema 无具体输入（灵活 optional），初始槽位由前端补齐
        for (let i = 0; i < INITIAL_INPUTS; i++) {
            node.addInput(anyName(i), "*");
        }
        node.setSize(node.computeSize());

        // 类型重算在 installDynamicSlots 的槽位增删之后执行
        // （后者包装 node.onConnectionsChange 时会把本函数作为 original 调用）
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info, slot_info) {
            syncSlotTypes(this);
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
        };

        installDynamicSlots(node, {
            inputPrefix: ANY_PREFIX,
            inputStart: 1,
            inputCount: MAX_INPUTS,
            inputType: "*",
            initialInputs: INITIAL_INPUTS,
            nameFor: (cfg, count) => anyName(count),
        });
        syncSlotTypes(node);

        const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            if (originalOnAfterGraphConfigured) {
                originalOnAfterGraphConfigured.apply(this, arguments);
            }
            syncSlotTypes(this);
        };
    },
});
