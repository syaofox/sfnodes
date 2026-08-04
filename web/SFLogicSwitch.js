// ==========================================================================
// SF Logic Switch - Dynamic Input Management
// ==========================================================================
//
// Description:
// JavaScript extension for SFAnythingIndexSwitch that enables dynamic
// input slot management. Initially shows only 2 slots (value0, value1),
// with automatic addition of new slots as connections are made.
//
// Features:
// - Initial display of 2 input slots
// - Auto-add slots when existing slots are connected (max 20)
// - Auto-remove trailing empty slots on disconnect
// - Works for SFAnythingIndexSwitch
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installDynamicSlots } from "./sf_dynamic_slots.js";

app.registerExtension({
    name: "sfnodes.LogicSwitch",

    nodeCreated(node) {
        if (node.comfyClass !== "SFAnythingIndexSwitch") return;

        installDynamicSlots(node, {
            inputPrefix: "value",
            inputStart: 0,
            inputCount: 20,
            inputType: "*",
            initialInputs: 2,
        });
    },
});
