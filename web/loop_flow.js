// ==========================================================================
// SF Loop Flow - Dynamic Input/Output Slot Management
// ==========================================================================
//
// Description:
// JavaScript extension for the sfnodes loop nodes (SFForLoopStart /
// SFForLoopEnd / SFWhileLoopStart / SFWhileLoopEnd) that enables dynamic
// input AND output slot management. Initially shows only 1 value slot,
// with automatic addition of new slots as connections are made.
//
// Features:
// - Initial display of 1 input slot + 1 output slot (value/initial_value)
// - Auto-add slots when existing slots are connected (up to declared count)
// - Auto-remove trailing empty slots on disconnect
// - Fixed slots (flow / index / total / condition) are never touched
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installDynamicSlots } from "./sf_dynamic_slots.js";

const LOOP_NODES = {
    SFWhileLoopStart: {
        inputPrefix: "initial_value",
        inputStart: 0,
        inputCount: 20,
        outputPrefix: "value",
        outputStart: 0,
        outputCount: 20,
    },
    SFWhileLoopEnd: {
        inputPrefix: "initial_value",
        inputStart: 0,
        inputCount: 20,
        outputPrefix: "value",
        outputStart: 0,
        outputCount: 20,
    },
    SFForLoopStart: {
        inputPrefix: "initial_value",
        inputStart: 1,
        inputCount: 19,
        outputPrefix: "value",
        outputStart: 1,
        outputCount: 19,
    },
    SFForLoopEnd: {
        inputPrefix: "initial_value",
        inputStart: 1,
        inputCount: 19,
        outputPrefix: "value",
        outputStart: 1,
        outputCount: 19,
    },
};

app.registerExtension({
    name: "sfnodes.loop_flow",

    nodeCreated(node) {
        const cfg = LOOP_NODES[node.comfyClass];
        if (!cfg) return;

        installDynamicSlots(node, {
            inputPrefix: cfg.inputPrefix,
            inputStart: cfg.inputStart,
            inputCount: cfg.inputCount,
            inputType: "*",
            initialInputs: 1,
            outputPrefix: cfg.outputPrefix,
            outputStart: cfg.outputStart,
            outputCount: cfg.outputCount,
            outputType: "*",
            initialOutputs: 1,
        });
    },
});
