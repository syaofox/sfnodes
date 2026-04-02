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

const MAX_INPUTS = 20;
const INITIAL_INPUTS = 2;

app.registerExtension({
    name: "sfnodes.LogicSwitch",

    nodeCreated(node) {
        if (node.comfyClass !== "SFAnythingIndexSwitch") return;

        // Trim inputs down to initial count on creation
        // Keep only the index widget input + value0, value1
        const originalOnConnectionsChange = node.onConnectionsChange;

        const getDynamicInputs = () =>
            node.inputs.filter((inp) => inp.name.startsWith("value"));

        const trimInputs = () => {
            const dynamic = getDynamicInputs();
            while (dynamic.length > INITIAL_INPUTS) {
                const idx = node.inputs.indexOf(dynamic[dynamic.length - 1]);
                node.removeInput(idx);
                dynamic.pop();
            }
        };

        // Trim on first load
        trimInputs();

        node.onConnectionsChange = function (type, index, connected, link_info, slot_info) {
            // Only handle input connections (type 1)
            if (type !== 1) {
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }
                return;
            }

            const dynamicInputs = getDynamicInputs();

            if (connected) {
                // All dynamic inputs connected? Add a new one
                const allConnected = dynamicInputs.every((inp) => inp.link !== null && inp.link !== undefined);
                if (allConnected) {
                    if (dynamicInputs.length >= MAX_INPUTS) return;
                    const newName = "value" + dynamicInputs.length;
                    this.addInput(newName, "*");
                }
            } else {
                // On disconnect: remove trailing empty slots beyond INITIAL_INPUTS
                const reversed = [...node.inputs].reverse();
                for (const inp of reversed) {
                    if (!inp.name.startsWith("value")) break;
                    if ((inp.link === null || inp.link === undefined) && getDynamicInputs().length > INITIAL_INPUTS) {
                        const idx = node.inputs.indexOf(inp);
                        node.removeInput(idx);
                    } else {
                        break;
                    }
                }
            }

            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
        };
    },
});
