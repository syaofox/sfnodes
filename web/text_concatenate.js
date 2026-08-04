import { app } from "/scripts/app.js";
import { installDynamicSlots } from "./sf_dynamic_slots.js";

app.registerExtension({
    name: "sfnodes.TextConcatenate",

    nodeCreated(node) {
        if (node.comfyClass !== "SFTextConcatenate") return;

        installDynamicSlots(node, {
            inputPrefix: "text_",
            inputStart: 1,
            inputCount: 16,
            inputType: "STRING",
            initialInputs: 1,
        });
    },
});
