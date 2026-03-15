// ==========================================================================
// Multi LoRA Loader - Dynamic Slot Management
// ==========================================================================
// 
// Description:
// JavaScript extension for MultiLoraLoader and MultiLoraLoaderModelOnly nodes
// that enables dynamic addition of LoRA slots. Initially shows only 1 slot,
// with an "Add Slot" button to reveal additional slots as needed.
// 
// Features:
// - Initial display of 1 LoRA slot
// - "Add Slot" button to dynamically add more slots
// - "Remove Slot" button to remove the last slot (minimum 1 slot)
// - State persistence (saves visible slot count with workflow)
// - Supports up to 50 slots
// - Works for both MultiLoraLoader and MultiLoraLoaderModelOnly nodes
// 
// ==========================================================================

import { app } from "/scripts/app.js";

app.registerExtension({
    name: "sfnodes.MultiLoraLoader",

    nodeCreated(node) {
        // 支持两个节点类：SFMultiLoraLoader 和 SFMultiLoraLoaderModelOnly
        if (node.comfyClass === "SFMultiLoraLoader" || node.comfyClass === "SFMultiLoraLoaderModelOnly") {
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
                if (widgets.enabled) {
                    widgets.enabled.hidden = !visible;
                    if (!visible) {
                        // 隐藏时重置值
                        widgets.enabled.value = false;
                    }
                }
                if (widgets.name) {
                    widgets.name.hidden = !visible;
                    if (!visible) {
                        // 隐藏时重置值
                        widgets.name.value = "[None]";
                    }
                }
                if (widgets.strength) {
                    widgets.strength.hidden = !visible;
                    if (!visible) {
                        // 隐藏时重置值
                        widgets.strength.value = 1.0;
                    }
                }
            };

            // 更新节点大小以适应可见的控件
            const updateNodeSize = () => {
                // 计算可见控件的高度
                // 基础高度：标题 + model输入 + (clip输入，如果是MultiLoraLoader) + normalize_weight + 按钮
                // 每个槽位增加约3个控件的高度（enabled + name + strength）
                const hasClip = node.comfyClass === "SFMultiLoraLoader";
                const baseHeight = hasClip ? 220 : 180; // MultiLoraLoader需要额外空间给CLIP输入
                const slotHeight = 85; // 每个槽位的高度（enabled + name + strength）
                
                const calculatedHeight = baseHeight + (node.visibleSlotCount * slotHeight);
                
                // 设置节点大小，保持宽度不变
                const currentWidth = node.size[0] || 320;
                node.setSize([currentWidth, calculatedHeight]);
            };

            // 初始化：隐藏除第一个槽位外的所有槽位
            const initializeSlots = () => {
                for (let i = 1; i <= MAX_SLOTS; i++) {
                    setSlotVisibility(i, i <= node.visibleSlotCount);
                }
                // 更新节点大小以适应可见的控件
                updateNodeSize();
            };

            // 添加槽位按钮的处理函数
            const addSlot = () => {
                if (node.visibleSlotCount < MAX_SLOTS) {
                    node.visibleSlotCount++;
                    setSlotVisibility(node.visibleSlotCount, true);
                    updateNodeSize(); // 更新节点大小
                    updateButtonStates(); // 更新按钮状态
                    node.setDirtyCanvas(true, true);
                }
            };

            // 删除槽位按钮的处理函数
            const removeSlot = () => {
                if (node.visibleSlotCount > 1) {
                    // 清除最后一个槽位的值
                    setSlotVisibility(node.visibleSlotCount, false);
                    node.visibleSlotCount--;
                    updateNodeSize(); // 更新节点大小
                    updateButtonStates(); // 更新按钮状态
                    node.setDirtyCanvas(true, true);
                }
            };

            // 更新按钮状态（删除按钮在只有1个槽位时应该禁用）
            const updateButtonStates = () => {
                if (node.removeSlotButton) {
                    node.removeSlotButton.disabled = node.visibleSlotCount <= 1;
                }
            };

            // 初始化槽位显示状态（在添加按钮之前）
            initializeSlots();

            // 添加"添加槽位"按钮
            node.addWidget("button", "➕ 添加槽位", null, () => {
                addSlot();
            });

            // 添加"删除槽位"按钮
            node.removeSlotButton = node.addWidget("button", "➖ 删除槽位", null, () => {
                removeSlot();
            });
            
            // 初始化按钮状态
            updateButtonStates();

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
                    // 更新按钮状态
                    updateButtonStates();
                }
            };
        }
    },
});
