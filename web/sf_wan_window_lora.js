// ==========================================================================
// SF Wan Window LoRA - 逐窗口位置 LoRA 槽（SFLoraPreset 连线槽动态增删）
// ==========================================================================
//
// 后端为灵活 optional schema（任意 window_N 输入名都按 SF_LORA_PRESET
// 放行，VALIDATE_INPUTS 接管校验），本扩展负责：
// - 初始 2 个 window_N 输入槽；全连 → 追加 1 槽、断开 → 回收尾部空槽
//   （installDynamicSlots，保底 2 槽，上限 10）
// - workflow 加载/粘贴恢复连线不触发 onConnectionsChange，
//   用 onAfterGraphConfigured 按实际链接数补齐/回收（combine 同款挂点）
//
// 位置语义：槽位即窗口位置（slot = 窗口出现顺序 % 槽总数，空槽直通），
// 槽序即 window_N 数字序，不做自动重命名。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installDynamicSlots, isSlotConnected } from "./sf_dynamic_slots.js";

const WINDOW_PREFIX = "window_";
const WINDOW_TYPE = "SF_LORA_PRESET";
const INITIAL_INPUTS = 2;
const MAX_INPUTS = 10;

const winName = (count) => WINDOW_PREFIX + (count + 1);

app.registerExtension({
    name: "sfnodes.WanWindowLoRA",

    nodeCreated(node) {
        if (node.comfyClass !== "SFWanWindowLoRA") return;

        // 后端 schema 无具体输入（灵活 optional），初始槽位由前端补齐
        for (let i = 0; i < INITIAL_INPUTS; i++) {
            node.addInput(winName(i), WINDOW_TYPE);
        }
        node.setSize(node.computeSize());

        installDynamicSlots(node, {
            inputPrefix: WINDOW_PREFIX,
            inputStart: 1,
            inputCount: MAX_INPUTS,
            inputType: WINDOW_TYPE,
            initialInputs: INITIAL_INPUTS,
            nameFor: (cfg, count) => winName(count),
        });

        // 加载/粘贴恢复：configure 直赋 links 不触发 onConnectionsChange，
        // 按实际链接数补齐到 linked+1（上限内），回收多余尾部空槽。
        const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            if (originalOnAfterGraphConfigured) {
                originalOnAfterGraphConfigured.apply(this, arguments);
            }
            if (this.comfyClass !== "SFWanWindowLoRA") return;
            const inputs = this.inputs || [];
            let linked = 0;
            for (const slot of inputs) {
                if (slot && slot.name && slot.name.startsWith(WINDOW_PREFIX) && isSlotConnected(slot)) {
                    linked += 1;
                }
            }
            const want = Math.min(Math.max(linked + 1, INITIAL_INPUTS), MAX_INPUTS);
            // 补齐
            let dynamic = inputs.filter((s) => s && s.name && s.name.startsWith(WINDOW_PREFIX));
            while (dynamic.length < want) {
                this.addInput(winName(dynamic.length), WINDOW_TYPE);
                dynamic = this.inputs.filter((s) => s && s.name && s.name.startsWith(WINDOW_PREFIX));
            }
            // 回收尾部空槽
            const reversed = [...this.inputs].reverse();
            for (const slot of reversed) {
                if (!slot || !slot.name || !slot.name.startsWith(WINDOW_PREFIX)) break;
                const current = this.inputs.filter((s) => s && s.name && s.name.startsWith(WINDOW_PREFIX));
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
