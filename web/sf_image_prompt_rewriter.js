import { app } from "/scripts/app.js";
import { installConfiguredSlotRecovery, installDynamicSlots } from "./sf_dynamic_slots.js";

const SLOT_CONFIG = {
    inputPrefix: "image_",
    inputStart: 1,
    inputCount: 10,
    inputType: "IMAGE",
    initialInputs: 1,
};

app.registerExtension({
    name: "sfnodes.ImagePromptRewriter",

    nodeCreated(node) {
        if (node.comfyClass !== "SFImagePromptRewriter") return;

        installDynamicSlots(node, SLOT_CONFIG);
        installConfiguredSlotRecovery(node, SLOT_CONFIG);
    },
});
