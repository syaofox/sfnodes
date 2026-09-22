import { app } from "/scripts/app.js";
import { installConfiguredSlotRecovery, installDynamicSlots } from "./sf_dynamic_slots.js";

const SLOT_CONFIG = {
    inputPrefix: "image_",
    inputStart: 1,
    inputCount: 8,
    inputType: "IMAGE",
    initialInputs: 1,
};

app.registerExtension({
    name: "sfnodes.QwenImage21PromptEnhancer",

    nodeCreated(node) {
        if (node.comfyClass !== "SFQwenImage21PromptEnhancer") return;

        installDynamicSlots(node, SLOT_CONFIG);
        installConfiguredSlotRecovery(node, SLOT_CONFIG);
    },
});
