import { app } from "/scripts/app.js";
import { installDynamicSlots } from "./sf_dynamic_slots.js";

app.registerExtension({
    name: "sfnodes.MaskBatch",

    nodeCreated(node) {
        if (node.comfyClass !== "SFMaskBatch") return;

        installDynamicSlots(node, {
            inputPrefix: "mask_",
            inputStart: 1,
            inputCount: 16,
            inputType: "MASK",
            initialInputs: 2,
        });
    },
});
