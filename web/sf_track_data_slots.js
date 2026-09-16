// ==========================================================================
// SF Track Data Slots - SAM3_TRACK_DATA 加减节点动态槽位
// ==========================================================================
//
// 管两个节点（后端均为静态 20 槽 schema，前端裁剪到初始 4）：
// - SFTrackDataSubtract：exclude_N 排除槽（MASK / SAM3_TRACK_DATA）
// - SFTrackDataAdd：add_N 叠加槽（MASK / SAM3_TRACK_DATA）
//
// 复用 sf_dynamic_slots：全连 → 追加 1 槽、断开 → 回收尾部空槽，
// 加载/粘贴经 installConfiguredSlotRecovery 补齐/回收（configure 不触发
// onConnectionsChange）。槽序无拼接语义（并集/相减均与顺序无关），不做重命名。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { installConfiguredSlotRecovery, installDynamicSlots } from "./sf_dynamic_slots.js";

const INITIAL_INPUTS = 4;
const MAX_INPUTS = 20;
const SLOT_TYPE = "MASK,SAM3_TRACK_DATA";

const PREFIX_BY_CLASS = {
    SFTrackDataSubtract: "exclude_",
    SFTrackDataAdd: "add_",
};

app.registerExtension({
    name: "sfnodes.TrackDataSlots",

    nodeCreated(node) {
        const prefix = PREFIX_BY_CLASS[node.comfyClass];
        if (!prefix) return;

        installDynamicSlots(node, {
            inputPrefix: prefix,
            inputStart: 1,
            inputCount: MAX_INPUTS,
            inputType: SLOT_TYPE,
            initialInputs: INITIAL_INPUTS,
        });

        installConfiguredSlotRecovery(node, {
            inputPrefix: prefix,
            inputStart: 1,
            inputType: SLOT_TYPE,
            initialInputs: INITIAL_INPUTS,
            inputCount: MAX_INPUTS,
        });
    },
});
