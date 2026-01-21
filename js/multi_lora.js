// ==========================================================================
// Multi LoRA Loader Model Only - Dynamic Slot Management
// ==========================================================================
// 
// Description:
// JavaScript extension for MultiLoraLoaderModelOnly node that enables
// dynamic addition of LoRA slots. Initially shows only 1 slot, with
// an "Add Slot" button to reveal additional slots as needed.
// 
// Features:
// - Initial display of 1 LoRA slot
// - "Add Slot" button to dynamically add more slots
// - State persistence (saves visible slot count with workflow)
// - Supports up to 50 slots
// 
// ==========================================================================

import { app } from "/scripts/app.js";

app.registerExtension({
    name: "sfnodes.MultiLoraLoaderModelOnly",

    nodeCreated(node) {
        if (node.comfyClass === "SFMultiLoraLoaderModelOnly") {
            // 初始化可见槽位数量（默认1个）
            node.visibleSlotCount = node.visibleSlotCount || 1;
            const MAX_SLOTS = 50;

            // 获取所有LoRA相关的控件
            const getSlotWidgets = (slotIndex) => {
                return {
                    enabled: node.widgets.find(w => w.name === `lora_${slotIndex}_enabled`),
                    name: node.widgets.find(w => w.name === `lora_${slotIndex}_name`),
                    strength: node.widgets.find(w => w.name === `lora_${slotIndex}_strength`)
                };
            };

            // 显示/隐藏指定槽位的所有控件
            const setSlotVisibility = (slotIndex, visible) => {
                const widgets = getSlotWidgets(slotIndex);
                if (widgets.enabled) widgets.enabled.hidden = !visible;
                if (widgets.name) widgets.name.hidden = !visible;
                if (widgets.strength) widgets.strength.hidden = !visible;
            };

            // 初始化：隐藏除第一个槽位外的所有槽位
            const initializeSlots = () => {
                for (let i = 1; i <= MAX_SLOTS; i++) {
                    setSlotVisibility(i, i <= node.visibleSlotCount);
                }
            };

            // 添加槽位按钮的处理函数
            const addSlot = () => {
                if (node.visibleSlotCount < MAX_SLOTS) {
                    node.visibleSlotCount++;
                    setSlotVisibility(node.visibleSlotCount, true);
                    node.setDirtyCanvas(true, true);
                }
            };

            // 初始化槽位显示状态（在添加按钮之前）
            initializeSlots();

            // 添加"添加槽位"按钮
            node.addWidget("button", "➕ 添加槽位", null, () => {
                addSlot();
            });

            // 保存原始的 serialize 方法
            const originalSerialize = node.serialize;
            node.serialize = function () {
                const data = originalSerialize.call(this);
                // 保存可见槽位数量
                data.visibleSlotCount = this.visibleSlotCount;
                return data;
            };

            // 保存原始的 configure 方法
            const originalConfigure = node.configure;
            node.configure = function (data) {
                originalConfigure.apply(this, arguments);
                
                // 恢复可见槽位数量
                if (data.visibleSlotCount !== undefined) {
                    this.visibleSlotCount = data.visibleSlotCount;
                    // 重新初始化槽位显示状态
                    initializeSlots();
                }
            };
        }
    },
});
