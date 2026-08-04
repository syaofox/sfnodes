import { app } from "/scripts/app.js";
import { installDynamicSlots } from "./sf_dynamic_slots.js";

app.registerExtension({
    name: "sfnodes.ImageConcatenate",

    nodeCreated(node) {
        if (node.comfyClass !== "SFImageConcatenate") return;

        installDynamicSlots(node, {
            inputPrefix: "image_",
            inputStart: 1,
            inputCount: 16,
            inputType: "IMAGE",
            initialInputs: 2,
        });
    },
});
