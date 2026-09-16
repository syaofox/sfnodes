// ==========================================================================
// SF Conditioning Combine - 动态多路输入合并（原生 Conditioning Combine 扩展）
// ==========================================================================
//
// 后端为灵活 optional schema（任意 conditioning_N 输入名都按 CONDITIONING
// 放行），本扩展负责：
// - 初始 2 个 conditioning_N 输入槽（对齐原生 2 路）；全连 → 追加 1 槽、
//   断开 → 回收尾部空槽（installDynamicSlots，保底 2 槽，上限 20）
// - workflow 加载/粘贴恢复连线不触发 onConnectionsChange，
//   用 onAfterGraphConfigured 按实际链接数补齐/回收（any_pack 同款挂点）
//
// 与 SFAnySwitch 的有意差异：CONDITIONING 是固定类型，无需按连接类型改型
// （不复用 any_pack.setSlotType 着色体系）；槽序即拼接语义，不做自动重命名。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installConfiguredSlotRecovery, installDynamicSlots } from "./sf_dynamic_slots.js";

const CONDITIONING_PREFIX = "conditioning_";
const INITIAL_INPUTS = 2;
const MAX_INPUTS = 20;

const condName = (count) => CONDITIONING_PREFIX + (count + 1);

app.registerExtension({
    name: "sfnodes.ConditioningCombine",

    nodeCreated(node) {
        if (node.comfyClass !== "SFConditioningCombine") return;

        // 后端 schema 无具体输入（灵活 optional），初始槽位由前端补齐
        for (let i = 0; i < INITIAL_INPUTS; i++) {
            node.addInput(condName(i), "CONDITIONING");
        }
        node.setSize(node.computeSize());

        installDynamicSlots(node, {
            inputPrefix: CONDITIONING_PREFIX,
            inputStart: 1,
            inputCount: MAX_INPUTS,
            inputType: "CONDITIONING",
            initialInputs: INITIAL_INPUTS,
            nameFor: (cfg, count) => condName(count),
        });

        // 加载/粘贴恢复：configure 直赋 links 不触发 onConnectionsChange，
        // 按实际链接数补齐/回收（sf_dynamic_slots 公共实现）。
        installConfiguredSlotRecovery(node, {
            inputPrefix: CONDITIONING_PREFIX,
            inputStart: 1,
            inputType: "CONDITIONING",
            initialInputs: INITIAL_INPUTS,
            inputCount: MAX_INPUTS,
        });
    },
});
