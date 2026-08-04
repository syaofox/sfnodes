import { app } from "/scripts/app.js";
import { installDynamicSlots } from "./sf_dynamic_slots.js";

const LETTERS = "abcdefghijklmnopqrstuvwxyz";

app.registerExtension({
    name: "sfnodes.SimpleMath",

    nodeCreated(node) {
        if (node.comfyClass !== "SFSimpleMath" && node.comfyClass !== "SFSimpleMathCondition") return;

        installDynamicSlots(node, {
            inputMatch: (name) => /^[a-z]$/.test(name),
            inputCount: 26,
            inputType: "*",
            initialInputs: 2,
            nameFor: (cfg, count) => LETTERS[count],
        });
    },
});
