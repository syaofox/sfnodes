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

const INITIAL_SLOTS = 1;

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

const slotName = (slot) => slot?.name ?? "";

const isDynamic = (slot, prefix) => slotName(slot).startsWith(prefix);

const isConnected = (slot) =>
    slot.link !== null && slot.link !== undefined && slot.link !== -1;

const isOutputEmpty = (slot) =>
    slot.links === null ||
    slot.links === undefined ||
    slot.links.length === 0;

const nextName = (prefix, start, count) => prefix + (start + count);

app.registerExtension({
    name: "sfnodes.loop_flow",

    nodeCreated(node) {
        const cfg = LOOP_NODES[node.comfyClass];
        if (!cfg) return;

        const originalOnConnectionsChange = node.onConnectionsChange;

        const getDynamicInputs = () =>
            (node.inputs ?? []).filter((inp) => isDynamic(inp, cfg.inputPrefix));

        const getDynamicOutputs = () =>
            (node.outputs ?? []).filter((out) => isDynamic(out, cfg.outputPrefix));

        const trimInputs = () => {
            const dyn = getDynamicInputs();
            while (dyn.length > INITIAL_SLOTS) {
                const idx = node.inputs.indexOf(dyn[dyn.length - 1]);
                if (idx < 0) break;
                node.removeInput(idx);
                dyn.pop();
            }
        };

        const trimOutputs = () => {
            const dyn = getDynamicOutputs();
            while (dyn.length > INITIAL_SLOTS) {
                const idx = node.outputs.indexOf(dyn[dyn.length - 1]);
                if (idx < 0) break;
                node.removeOutput(idx);
                dyn.pop();
            }
        };

        trimInputs();
        trimOutputs();

        // Recompute node size after trimming hidden slots, otherwise the
        // initial height still accounts for all declared slots
        node.setSize(node.computeSize());

        node.onConnectionsChange = function (type, index, connected, link_info, slot_info) {
            if (type === 1) {
                const dynamicInputs = getDynamicInputs();

                if (connected) {
                    const allConnected =
                        dynamicInputs.length > 0 &&
                        dynamicInputs.every(isConnected);
                    if (allConnected && dynamicInputs.length < cfg.inputCount) {
                        this.addInput(
                            nextName(cfg.inputPrefix, cfg.inputStart, dynamicInputs.length),
                            "*"
                        );
                    }
                } else {
                    const reversed = [...node.inputs].reverse();
                    for (const inp of reversed) {
                        if (!isDynamic(inp, cfg.inputPrefix)) break;
                        if (!isConnected(inp) && getDynamicInputs().length > INITIAL_SLOTS) {
                            const idx = node.inputs.indexOf(inp);
                            node.removeInput(idx);
                        } else {
                            break;
                        }
                    }
                }
            } else if (type === 2) {
                const dynamicOutputs = getDynamicOutputs();

                if (connected) {
                    const allConnected =
                        dynamicOutputs.length > 0 &&
                        dynamicOutputs.every((out) => !isOutputEmpty(out));
                    if (allConnected && dynamicOutputs.length < cfg.outputCount) {
                        this.addOutput(
                            nextName(cfg.outputPrefix, cfg.outputStart, dynamicOutputs.length),
                            "*"
                        );
                    }
                } else {
                    const reversed = [...node.outputs].reverse();
                    for (const out of reversed) {
                        if (!isDynamic(out, cfg.outputPrefix)) break;
                        if (isOutputEmpty(out) && getDynamicOutputs().length > INITIAL_SLOTS) {
                            const idx = node.outputs.indexOf(out);
                            node.removeOutput(idx);
                        } else {
                            break;
                        }
                    }
                }
            }

            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
        };
    },
});
