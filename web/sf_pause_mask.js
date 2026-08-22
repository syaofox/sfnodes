// ==========================================================================
// sf_pause_mask.js - SFPauseMask 主扩展（薄配置）
// ==========================================================================
//
// 全部机制收敛于 web/sf_pause_kit.js（definePauseGate），与 SFPauseImage 同构。
// MASK 类型，输入键 "mask"：快照是 L 模式灰度 PNG（temp 目录按节点 id 命名），
// Continue 时前端把上游剪出 prompt、Python 读回快照，预览 /view 直接可用。
//
// ==========================================================================

import { definePauseGate } from "./sf_pause_kit.js";

definePauseGate({
    classy: "SFPauseMask",
    extensionName: "sfnodes.PauseMask",
    widgetType: "sf_pause_mask_ui",
    savePrefix: "PauseMask",

    stateProp: "pauseMaskState",
    propPrefix: "_sfPauseMask",

    inputKey: "mask",
    extraInputKeys: null,
    frameEventKey: "sf_pause_mask_frame",

    logTag: "SF Pause Mask",
    injectName: "Pause Mask",
    captureMsg: "Run once to capture a mask",

    cssId: "sf-pm-css",
    cssPrefix: "sf-pm-",
    emptyText: "Press Run to preview the mask here",
    contTitle: "从快照运行工作流其余部分",
    regenTitle: "在此处掷一张新遮罩（尊重你的种子）",
    toolNoun: "遮罩",
});
