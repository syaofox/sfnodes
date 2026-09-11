// ==========================================================================
// SF Conditioning Concat - 动态多路拼接（原生 Conditioning Concat 扩展）
// ==========================================================================
//
// 后端为灵活 optional schema（任意 conditioning_N 输入名都按 CONDITIONING
// 放行，复用 conditioning_combine 的槽名前缀与槽数上下限语义），本扩展负责：
// - 初始 2 个 conditioning_N 输入槽（conditioning_1 = 被拼接方 to，
//   conditioning_2 起 = 拼接源 from）；全连 → 追加 1 槽、
//   断开 → 回收尾部空槽（installDynamicSlots，保底 2 槽，上限 20）
// - workflow 加载/粘贴恢复连线不触发 onConnectionsChange，
//   用 onAfterGraphConfigured 按实际链接数补齐/回收（combine 同款挂点）
//
// 与 SFConditioningCombine 前端的唯一差异是 comfyClass 与槽角色注释：
// 槽位增删、固定 CONDITIONING 类型免着色、不自动重命名三条结论延续 §49。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installDynamicSlots, isSlotConnected } from "./sf_dynamic_slots.js";

const CONDITIONING_PREFIX = "conditioning_";
const INITIAL_INPUTS = 2;
const MAX_INPUTS = 20;

const condName = (count) => CONDITIONING_PREFIX + (count + 1);

app.registerExtension({
    name: "sfnodes.ConditioningConcat",

    nodeCreated(node) {
        if (node.comfyClass !== "SFConditioningConcat") return;

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
        // 按实际链接数补齐到 linked+1（上限内），回收多余尾部空槽。
        const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            if (originalOnAfterGraphConfigured) {
                originalOnAfterGraphConfigured.apply(this, arguments);
            }
            if (this.comfyClass !== "SFConditioningConcat") return;
            const inputs = this.inputs || [];
            let linked = 0;
            for (const slot of inputs) {
                if (slot && slot.name && slot.name.startsWith(CONDITIONING_PREFIX) && isSlotConnected(slot)) {
                    linked += 1;
                }
            }
            const want = Math.min(Math.max(linked + 1, INITIAL_INPUTS), MAX_INPUTS);
            // 补齐
            let dynamic = inputs.filter((s) => s && s.name && s.name.startsWith(CONDITIONING_PREFIX));
            while (dynamic.length < want) {
                this.addInput(condName(dynamic.length), "CONDITIONING");
                dynamic = this.inputs.filter((s) => s && s.name && s.name.startsWith(CONDITIONING_PREFIX));
            }
            // 回收尾部空槽
            const reversed = [...this.inputs].reverse();
            for (const slot of reversed) {
                if (!slot || !slot.name || !slot.name.startsWith(CONDITIONING_PREFIX)) break;
                const current = this.inputs.filter((s) => s && s.name && s.name.startsWith(CONDITIONING_PREFIX));
                if (!isSlotConnected(slot) && current.length > want) {
                    this.removeInput(this.inputs.indexOf(slot));
                } else {
                    break;
                }
            }
            this.setSize(this.computeSize());
        };
    },
});
