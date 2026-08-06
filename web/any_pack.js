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
// - Initially shows 1 input slot (value0)
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
// - When connected to an SFAnyPack, its output slots expand to match the
//   pack's input slot count (grow-only: never removed by the disconnect
//   trim while the pack still has that many slots)
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
// Slot coloring (auto-adapt to the connected data type):
// - SFAnyPack input slots / SFAnyUnpack output slots retype themselves to the
//   connected data type(s), so the slot dot matches the wire color
//   (e.g. IMAGE orange, MASK green; a "IMAGE,MASK" union renders as split
//   colors). Slots revert to "*" (neutral) once disconnected.
// - Types are recomputed on connect/disconnect and after workflow load/paste
//   (onAfterGraphConfigured: configure restores links without firing
//   onConnectionsChange). The backend keeps declaring "*", so the retyping is
//   purely a frontend concern (slot rendering + connection validation).
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { isSlotConnected, uniqueName } from "./sf_dynamic_slots.js";

const MAX_SLOTS = 20;
const PACK_INITIAL_INPUTS = 1;
const UNPACK_INITIAL_OUTPUTS = 1;
const PACK_PREFIX = "value";
const UNPACK_PREFIX = "out";
const MAX_PROPAGATE_DEPTH = 8;

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
            // Expand the unpack outputs to the pack's input slot count
            // (grow-only) before propagating names by slot index.
            const expanded = expandOutputsTo(node, target);
            let changed = false;
            (target.outputs || []).forEach((out, i) => {
                const src = node.inputs[i];
                if (src && !out.sfManualName && out.name !== src.name) {
                    // Replace the slot element instead of mutating it in place
                    // (Vue frontend re-renders slots on element replacement)
                    // and keep localized_name in sync: the renderers read
                    // label ?? localized_name ?? name, and initial slots carry
                    // a localized_name that would otherwise keep the old name.
                    target.outputs[i] = Object.assign({}, out, {
                        name: src.name,
                        localized_name: src.name,
                    });
                    changed = true;
                }
            });
            if (expanded || changed) {
                target.setSize?.(target.computeSize());
                changedTargets.push(target);
            }
        } else if (isUnpack && target.comfyClass === "SFAnyPack") {
            let changed = false;
            (target.inputs || []).forEach((inp, i) => {
                const src = node.outputs[i];
                if (src && !inp.sfManualName && inp.name !== src.name) {
                    const newName = uniqueName(target.inputs, i, src.name);
                    target.inputs[i] = Object.assign({}, inp, {
                        name: newName,
                        localized_name: newName,
                    });
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
// Slot coloring: collect the types of all links attached to a slot (1 for
// inputs, N for outputs) and expose them as a comma-separated union type.
// "*/empty" types are dropped (nothing to color); link types are already
// concrete because the graph computes them with commonType() at connect time.
// ---------------------------------------------------------------------------
function slotLinkTypes(node, slot) {
    if (!node || !node.graph || !slot) return [];
    const ids = slot.link != null
        ? [slot.link]
        : Array.isArray(slot.links)
            ? slot.links
            : [];
    const types = [];
    for (const id of ids) {
        const link = node.graph.links[id];
        const t = link && typeof link.type === "string" ? link.type : "";
        for (const part of t.split(",")) {
            const p = part.trim();
            if (p && p !== "*" && !types.includes(p)) types.push(p);
        }
    }
    return types;
}

function unionType(types) {
    return types && types.length ? types.join(",") : "*";
}

// Replace the slot with a shallow copy carrying the new type. In the Vue
// frontend node.inputs/node.outputs are reactive arrays: mutating slot.type
// in place does not re-render the slot dot, but replacing the array element
// does (same pattern as the official dynamic-type feature).
function setSlotType(node, slots, index, type) {
    const slot = slots[index];
    if (!slot || slot.type === type) return;
    slots[index] = Object.assign({}, slot, { type });
    app.canvas?.setDirty?.(true, true);
}

// Recompute the type of every dynamic slot of a Pack/Unpack node from its
// current links. Runs on connect/disconnect and after workflow load/paste.
function syncSlotTypes(node) {
    if (!node || !node.graph) return;
    const isPack = node.comfyClass === "SFAnyPack";
    const isUnpack = node.comfyClass === "SFAnyUnpack";
    if (!isPack && !isUnpack) return;
    if (isPack) {
        (node.inputs || []).forEach((slot, i) => {
            setSlotType(node, node.inputs, i, unionType(slotLinkTypes(node, slot)));
        });
    } else {
        // Expand outputs to the upstream pack's input slot count (covers
        // workflow load/paste of graphs saved before auto-expansion).
        const packNode = upstreamPackNode(node);
        if (packNode) {
            expandOutputsTo(packNode, node);
        }
        (node.outputs || []).forEach((slot, i) => {
            setSlotType(node, node.outputs, i, unionType(slotLinkTypes(node, slot)));
        });
    }
}

// ---------------------------------------------------------------------------
// Output auto-expansion: an SFAnyUnpack fed by an SFAnyPack shows as many
// output slots as the pack has input slots. Grow-only (never removes slots,
// which could break downstream wires); capped at MAX_SLOTS.
// ---------------------------------------------------------------------------
function expandOutputsTo(packNode, unpackNode) {
    if (!packNode || !unpackNode || typeof unpackNode.addOutput !== "function") return false;
    const needed = Math.min((packNode.inputs || []).length, MAX_SLOTS);
    const cur = (unpackNode.outputs || []).length;
    if (needed <= cur) return false;
    for (let i = cur; i < needed; i++) {
        unpackNode.addOutput(UNPACK_PREFIX + i, "*");
    }
    unpackNode.setSize?.(unpackNode.computeSize());
    return true;
}

// The upstream SFAnyPack feeding this node via its "pack" input (null when
// the input is empty or not fed by an SFAnyPack).
function upstreamPackNode(node) {
    if (!node || !node.graph || !node.inputs || !node.inputs[0]) return null;
    const linkId = node.inputs[0].link;
    if (linkId == null) return null;
    const link = node.graph.links[linkId];
    if (!link) return null;
    const source = node.graph.getNodeById(link.origin_id);
    return source && source.comfyClass === "SFAnyPack" ? source : null;
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
                const idx = slots.indexOf(slot);
                if (idx < 0) return;
                // Replace the slot element so the frontend re-renders the
                // label (mutating slot.name in place is not tracked) and keep
                // localized_name in sync (renderers read label ?? localized_name ?? name).
                const renamed = uniqueName(slots, idx, trimmed);
                slots[idx] = Object.assign({}, slot, {
                    name: renamed,
                    localized_name: renamed,
                    sfManualName: true,
                });
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
        // name (skips manually renamed slots). Replaces the slot element so
        // the Vue frontend re-renders the slot label even when the type does
        // not change (e.g. "*" -> "*" connections), and keeps localized_name
        // in sync (renderers read label ?? localized_name ?? name).
        const autoNameInput = (index, link_info) => {
            if (!link_info) return false;
            const source = node.graph && node.graph.getNodeById(link_info.origin_id);
            const sourceName = source && source.outputs && source.outputs[link_info.origin_slot] &&
                source.outputs[link_info.origin_slot].name;
            if (!sourceName) return false;
            const target = node.inputs[index];
            if (!target || target.sfManualName || target.name === sourceName) return false;
            const newName = uniqueName(node.inputs, index, sourceName);
            node.inputs[index] = Object.assign({}, target, {
                name: newName,
                localized_name: newName,
            });
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

        // After workflow load/paste the links are restored without firing
        // onConnectionsChange, so recompute all slot types from the links.
        const originalOnAfterConfigured = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            if (originalOnAfterConfigured) {
                originalOnAfterConfigured.apply(this, arguments);
            }
            syncSlotTypes(this);
        };

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
                const allConnected = dynamic.every((s) => isSlotConnected(s));
                const lastIndex = (isInput ? node.inputs : node.outputs).length - 1;
                if (allConnected && index === lastIndex && dynamic.length < MAX_SLOTS) {
                    addSlot(prefix + dynamic.length);
                }

                // Auto-name the connected Pack input slot, then propagate
                // names and expand downstream unpack outputs
                if (isInput) {
                    autoNameInput(index, link_info);
                    propagateNames(node);
                }
            } else {
                // On disconnect: remove trailing empty slots beyond the
                // minimum (slot names are kept for remaining slots). When the
                // unpack is fed by a pack, the minimum is the pack's current
                // input slot count so the auto-expansion is preserved.
                const trimFloor = isPack
                    ? initialCount
                    : Math.max(initialCount, (upstreamPackNode(node)?.inputs || []).length);
                const reversed = [...dynamic].reverse();
                for (const slot of reversed) {
                    if (!isSlotConnected(slot) && getDynamic().length > trimFloor) {
                        removeSlot(slot);
                    } else {
                        break;
                    }
                }
            }

            // Recolor dynamic slots from the current link state (connect or
            // disconnect already updated slot.link / slot.links by now).
            syncSlotTypes(this);

            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
        };
    },
});
