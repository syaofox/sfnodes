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
import { installDynamicSlots, isSlotConnected } from "./sf_dynamic_slots.js";

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
        // 按实际链接数补齐到 linked+1（上限内），回收多余尾部空槽。
        const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            if (originalOnAfterGraphConfigured) {
                originalOnAfterGraphConfigured.apply(this, arguments);
            }
            if (this.comfyClass !== "SFConditioningCombine") return;
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
            // 回收尾部空槽（保留 name+localized_name 同步：新增槽无需改名）
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
