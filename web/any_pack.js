// ==========================================================================
// SF Any Pack / SF Any Unpack - Dynamic Slot Management, Auto-Naming & Rename
// ==========================================================================
//
// Description:
// JavaScript extension for SFAnyPack and SFAnyUnpack that enables dynamic
// slot management, automatic slot naming on connect, name propagation to the
// unpacker, and right-click slot renaming.
//
// SFAnyPack (packer):
// - Initially shows 2 input slots (value0, value1)
// - Auto-adds input slots when existing ones are connected (max 20)
// - Auto-removes trailing empty slots on disconnect
// - When a wire is connected, the input slot is automatically named after
//   the source node's output slot (unless the slot was manually renamed)
// - Right-click an input slot to rename it manually
//
// SFAnyUnpack (unpacker):
// - Initially shows 1 output slot (out0)
// - Auto-adds output slots when existing ones are connected (max 20)
// - Auto-removes trailing empty slots on disconnect
// - Output slot names are propagated from the connected SFAnyPack input
//   names (by slot index), keeping the unpacked wires identifiable
// - Right-click an output slot to rename it manually
//
// Naming rules:
// - Manually renamed slots (sfManualName flag) are never overwritten
// - Duplicate names get a numeric suffix (image, image_2, ...) to keep the
//   prompt serialization keys unique (the backend parameter name of an
//   SFAnyPack input is always "value{index}")
// - Slot names are kept on disconnect
//
// ==========================================================================

import { app } from "/scripts/app.js";

const MAX_SLOTS = 20;
const PACK_INITIAL_INPUTS = 2;
const UNPACK_INITIAL_OUTPUTS = 1;
const PACK_PREFIX = "value";
const UNPACK_PREFIX = "out";
const MAX_PROPAGATE_DEPTH = 8;

const isSlotConnected = (isInput, slot) => {
    if (isInput) {
        return slot.link !== null && slot.link !== undefined;
    }
    if (Array.isArray(slot.links)) {
        return slot.links.length > 0;
    }
    return slot.link !== null && slot.link !== undefined;
};

// Returns a name that does not collide with any other slot in the same array.
// Duplicate names would break prompt serialization (same key overwritten).
const uniqueName = (slots, selfIndex, base) => {
    if (!Array.isArray(slots) || slots.length === 0) return base;
    let name = base;
    let i = 2;
    while (slots.some((s, idx) => idx !== selfIndex && s && s.name === name)) {
        name = base + "_" + i++;
    }
    return name;
};

// ---------------------------------------------------------------------------
// Chain propagation of slot names:
//   SFAnyPack inputs  -> connected SFAnyUnpack outputs (by slot index)
//   SFAnyUnpack outs  -> connected SFAnyPack inputs (chain scenario)
// Manually renamed slots are skipped. Propagation stops when nothing changes
// or the depth limit is reached.
// ---------------------------------------------------------------------------
function propagateNames(node, depth = 0) {
    if (!node || depth > MAX_PROPAGATE_DEPTH) return;
    const isPack = node.comfyClass === "SFAnyPack";
    const isUnpack = node.comfyClass === "SFAnyUnpack";
    if (!isPack && !isUnpack) return;
    if (!node.graph || !node.graph.links || !node.outputs || !node.outputs.length) return;

    const links = node.outputs[0].links || [];
    const changedTargets = [];

    for (const linkId of links) {
        const link = node.graph.links[linkId];
        if (!link) continue;
        const target = node.graph.getNodeById(link.target_id);
        if (!target) continue;

        if (isPack && target.comfyClass === "SFAnyUnpack") {
            let changed = false;
            (target.outputs || []).forEach((out, i) => {
                const src = node.inputs[i];
                if (src && !out.sfManualName && out.name !== src.name) {
                    out.name = src.name;
                    changed = true;
                }
            });
            if (changed) {
                target.setSize?.(target.computeSize());
                changedTargets.push(target);
            }
        } else if (isUnpack && target.comfyClass === "SFAnyPack") {
            let changed = false;
            (target.inputs || []).forEach((inp, i) => {
                const src = node.outputs[i];
                if (src && !inp.sfManualName && inp.name !== src.name) {
                    inp.name = uniqueName(target.inputs, i, src.name);
                    changed = true;
                }
            });
            if (changed) {
                target.setSize?.(target.computeSize());
                changedTargets.push(target);
            }
        }
    }

    for (const target of changedTargets) {
        propagateNames(target, depth + 1);
    }
}

// ---------------------------------------------------------------------------
// Prompt serialization hook: maps renamed SFAnyPack input names back to the
// backend parameter names ("value{index}") before the prompt is sent.
// Works with both the modern ({ output, workflow }) and legacy
// ([prompt, workflow]) return formats, sync or async.
// ---------------------------------------------------------------------------
function installPromptMapping() {
    const original = app.graphToPrompt;
    if (!original || original.sfAnyPackMapped) return;

    const patchPrompt = (result) => {
        const prompt = Array.isArray(result) ? result[0] : result && result.output;
        if (!prompt || !app.graph || !app.graph._nodes) return result;
        for (const node of app.graph._nodes) {
            if (node.comfyClass !== "SFAnyPack" || !node.inputs) continue;
            const promptNode = prompt[node.id];
            if (!promptNode || !promptNode.inputs) continue;
            node.inputs.forEach((input, i) => {
                const paramName = PACK_PREFIX + i;
                if (input.name !== paramName && promptNode.inputs[input.name] !== undefined) {
                    promptNode.inputs[paramName] = promptNode.inputs[input.name];
                    delete promptNode.inputs[input.name];
                }
            });
        }
        return result;
    };

    const wrapped = function (...args) {
        const result = original.apply(app, args);
        if (result && typeof result.then === "function") {
            return result.then(patchPrompt);
        }
        return patchPrompt(result);
    };
    wrapped.sfAnyPackMapped = true;
    app.graphToPrompt = wrapped;
}

// ---------------------------------------------------------------------------
// Right-click menu: rename the slot under the mouse cursor.
// ---------------------------------------------------------------------------
function setupRenameMenu(node, isInput) {
    const originalGetExtraMenuOptions = node.getExtraMenuOptions;

    node.getExtraMenuOptions = function (canvas, options) {
        if (originalGetExtraMenuOptions) {
            originalGetExtraMenuOptions.apply(this, arguments);
        }
        if (!Array.isArray(options)) return;

        const mouse = canvas && canvas.graph_mouse;
        if (!mouse) return;

        let slot = null;
        let index = -1;
        if (typeof this.getSlotInPosition === "function") {
            const hit = this.getSlotInPosition(mouse[0], mouse[1]);
            if (hit) {
                if (isInput && hit.input) {
                    slot = hit.input;
                    index = hit.slot;
                } else if (!isInput && hit.output) {
                    slot = hit.output;
                    index = hit.slot;
                }
            }
        }
        if (slot === null || index < 0) return;

        options.push({
            content: "Rename " + (isInput ? "Input" : "Output") + " Slot",
            callback: () => {
                const current = slot.name;
                const newName = prompt(
                    'Rename ' + (isInput ? "input" : "output") + ' slot "' + current + '" to:',
                    current
                );
                if (newName === null) return;
                const trimmed = newName.trim();
                if (trimmed === "" || trimmed === current) return;
                const slots = isInput ? this.inputs : this.outputs;
                slot.name = uniqueName(slots, slots.indexOf(slot), trimmed);
                slot.sfManualName = true;
                this.setSize(this.computeSize());
                propagateNames(this);
            }
        });
    };
}

app.registerExtension({
    name: "sfnodes.AnyPack",

    setup() {
        installPromptMapping();
    },

    nodeCreated(node) {
        if (node.comfyClass !== "SFAnyPack" && node.comfyClass !== "SFAnyUnpack") return;

        const isPack = node.comfyClass === "SFAnyPack";
        const isInput = isPack;
        const prefix = isPack ? PACK_PREFIX : UNPACK_PREFIX;
        const initialCount = isPack ? PACK_INITIAL_INPUTS : UNPACK_INITIAL_OUTPUTS;

        // All inputs of SFAnyPack / all outputs of SFAnyUnpack are dynamic
        // slots, so no name-prefix filtering is needed (names may be renamed).
        const getDynamic = () => (isInput ? node.inputs : node.outputs).slice();

        const removeSlot = (slot) => {
            const slots = isInput ? node.inputs : node.outputs;
            const idx = slots.indexOf(slot);
            if (idx >= 0) {
                if (isInput) {
                    node.removeInput(idx);
                } else {
                    node.removeOutput(idx);
                }
            }
        };

        const addSlot = (name) => {
            if (isInput) {
                node.addInput(name, "*");
            } else {
                node.addOutput(name, "*");
            }
        };

        // Auto-name a Pack input slot after the source node's output slot
        // name (skips manually renamed slots).
        const autoNameInput = (index, link_info) => {
            if (!link_info) return false;
            const source = node.graph && node.graph.getNodeById(link_info.origin_id);
            const sourceName = source && source.outputs && source.outputs[link_info.origin_slot] &&
                source.outputs[link_info.origin_slot].name;
            if (!sourceName) return false;
            const target = node.inputs[index];
            if (!target || target.sfManualName || target.name === sourceName) return false;
            target.name = uniqueName(node.inputs, index, sourceName);
            node.setSize(node.computeSize());
            return true;
        };

        // Trim down to initial count on creation
        const trimSlots = () => {
            const dynamic = getDynamic();
            while (dynamic.length > initialCount) {
                removeSlot(dynamic[dynamic.length - 1]);
                dynamic.pop();
            }
        };

        trimSlots();

        // Recompute node size after trimming hidden slots, otherwise the
        // initial height still accounts for all 20 declared slots
        node.setSize(node.computeSize());

        setupRenameMenu(node, isInput);

        const originalOnConnectionsChange = node.onConnectionsChange;

        node.onConnectionsChange = function (type, index, connected, link_info, slot_info) {
            if (type !== (isInput ? 1 : 2)) {
                // For SFAnyUnpack: when its pack input is connected, pull the
                // slot names from the upstream SFAnyPack node.
                if (!isInput && type === 1 && connected && link_info) {
                    const source = node.graph && node.graph.getNodeById(link_info.origin_id);
                    if (source && source.comfyClass === "SFAnyPack") {
                        propagateNames(source);
                    }
                }
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }
                return;
            }

            const dynamic = getDynamic();

            if (connected) {
                // Add a new slot only when the connection lands on the last dynamic slot
                // and all dynamic slots are already connected (prevents extra slots
                // when adding additional links to an already-connected output)
                const allConnected = dynamic.every((s) => isSlotConnected(isInput, s));
                const lastIndex = (isInput ? node.inputs : node.outputs).length - 1;
                if (allConnected && index === lastIndex && dynamic.length < MAX_SLOTS) {
                    addSlot(prefix + dynamic.length);
                }

                // Auto-name the connected Pack input slot and propagate names downstream
                if (isInput && autoNameInput(index, link_info)) {
                    propagateNames(node);
                }
            } else {
                // On disconnect: remove trailing empty slots beyond initial count
                // (slot names are kept for remaining slots)
                const reversed = [...dynamic].reverse();
                for (const slot of reversed) {
                    if (!isSlotConnected(isInput, slot) && getDynamic().length > initialCount) {
                        removeSlot(slot);
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
