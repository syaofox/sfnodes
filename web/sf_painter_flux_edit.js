// ==========================================================================
// SF Painter Flux Image Edit - 参考图动态槽位
// ==========================================================================
//
// 后端 schema 只声明 image1（+ 固定的 image1_mask），本扩展复用公共库
// sf_dynamic_slots.installDynamicSlots：
// - image1 连接后自动追加 image2、image3……最多 10 张
// - 断开尾部空槽则回收（保底 1 张）
// - image1_mask 不匹配 /^image\d+$/，不参与动态槽
//
// workflow 加载/粘贴恢复连线不触发 onConnectionsChange，用
// onAfterGraphConfigured 按实际链接数补齐/回收（sf_conditioning_combine 同款挂点）。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installDynamicSlots, isSlotConnected } from "./sf_dynamic_slots.js";

const NODE_NAME = "SFPainterFluxImageEdit";
const IMAGE_RE = /^image(\d+)$/;
const INITIAL_IMAGES = 1;
const MAX_IMAGES = 10;

const imageName = (n) => `image${n}`;

function dynamicInputs(node) {
    return (node.inputs || []).filter((s) => s && IMAGE_RE.test(s.name));
}

function linkedImageCount(node) {
    let linked = 0;
    for (const slot of dynamicInputs(node)) {
        if (isSlotConnected(slot)) linked += 1;
    }
    return linked;
}

app.registerExtension({
    name: "sfnodes.PainterFluxImageEdit",

    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) return;

        installDynamicSlots(node, {
            inputMatch: (name) => IMAGE_RE.test(name),
            inputStart: 1,
            inputCount: MAX_IMAGES,
            inputType: "IMAGE",
            initialInputs: INITIAL_IMAGES,
            nameFor: (cfg, count) => imageName(cfg.start + count),
        });

        const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            if (originalOnAfterGraphConfigured) {
                originalOnAfterGraphConfigured.apply(this, arguments);
            }
            if (this.comfyClass !== NODE_NAME) return;

            const want = Math.min(Math.max(linkedImageCount(this) + 1, INITIAL_IMAGES), MAX_IMAGES);
            // 补齐到 want。
            let dynamic = dynamicInputs(this);
            while (dynamic.length < want) {
                this.addInput(imageName(dynamic.length + 1), "IMAGE");
                dynamic = dynamicInputs(this);
            }
            // 回收尾部空槽（保留至少 one 空槽）。
            const reversed = [...(this.inputs || [])].reverse();
            for (const slot of reversed) {
                if (!slot || !IMAGE_RE.test(slot.name)) break;
                const current = dynamicInputs(this);
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
