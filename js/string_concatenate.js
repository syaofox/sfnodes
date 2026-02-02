// ==========================================================================
// String Concatenate - Dynamic String Slot Management
// ==========================================================================
//
// JavaScript extension for SFStringConcatenate node that enables dynamic
// addition/removal of string input slots. Default 3 visible slots, with
// "Add" and "Remove" buttons to change the visible count (min 1, max 20).
//
// ==========================================================================

import { app } from "/scripts/app.js";

const MAX_SLOTS = 20;

app.registerExtension({
    name: "sfnodes.StringConcatenate",

    nodeCreated(node) {
        if (node.comfyClass !== "SFStringConcatenate") return;

        node.visibleSlotCount = node.visibleSlotCount ?? 1;

        const getSlotWidget = (slotIndex) =>
            node.widgets.find((w) => w.name === `string_${slotIndex}`);

        const setSlotVisibility = (slotIndex, visible) => {
            const widget = getSlotWidget(slotIndex);
            if (widget) {
                widget.hidden = !visible;
                if (!visible) widget.value = "";
            }
        };

        const updateNodeSize = () => {
            const baseHeight = 120;
            const slotHeight = 64;
            const calculatedHeight = baseHeight + node.visibleSlotCount * slotHeight;
            const currentWidth = node.size[0] || 320;
            node.setSize([currentWidth, calculatedHeight]);
        };

        const initializeSlots = () => {
            for (let i = 1; i <= MAX_SLOTS; i++) {
                setSlotVisibility(i, i <= node.visibleSlotCount);
            }
            updateNodeSize();
        };

        const addSlot = () => {
            if (node.visibleSlotCount < MAX_SLOTS) {
                node.visibleSlotCount++;
                setSlotVisibility(node.visibleSlotCount, true);
                updateNodeSize();
                updateButtonStates();
                node.setDirtyCanvas(true, true);
            }
        };

        const removeSlot = () => {
            if (node.visibleSlotCount > 1) {
                setSlotVisibility(node.visibleSlotCount, false);
                node.visibleSlotCount--;
                updateNodeSize();
                updateButtonStates();
                node.setDirtyCanvas(true, true);
            }
        };

        const updateButtonStates = () => {
            if (node.removeSlotButton) {
                node.removeSlotButton.disabled = node.visibleSlotCount <= 1;
            }
        };

        initializeSlots();

        node.addWidget("button", "➕ 添加一项", null, addSlot);
        node.removeSlotButton = node.addWidget("button", "➖ 删除一项", null, removeSlot);
        updateButtonStates();

        const originalSerialize = node.serialize;
        node.serialize = function () {
            const data = originalSerialize.call(this);
            data.visibleSlotCount = this.visibleSlotCount;
            return data;
        };

        const originalConfigure = node.configure;
        node.configure = function (data) {
            originalConfigure.apply(this, arguments);
            if (data.visibleSlotCount !== undefined) {
                this.visibleSlotCount = data.visibleSlotCount;
                initializeSlots();
                updateButtonStates();
            }
        };
    },
});
