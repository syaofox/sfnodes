// ==========================================================================
// sf_pause_image.js - SFPauseImage 主扩展（薄配置）
// ==========================================================================
//
// 全部机制收敛于 web/sf_pause_kit.js（definePauseGate）：节点体 UI、快照
// Copy/Open/Save 链路、双钩子（graphToPrompt 只 INJECT / api.queuePrompt 才
// PRUNE，prune 复用 sf_pause_text_lib.js::applyGateMode）、executed 回填。
//
// 快照机制：Pause 时 Python 把首帧存到 ComfyUI temp 目录（按节点 id 确定性
// 命名），Continue 时前端把上游剪出 prompt、Python 读回快照——只有下游运行，
// 上游模型链完全跳过。Save：POST /api/sfnodes/preview/{save,prepare}。
//
// ⚠ frameEventKey "sf_pause_frame" 是历史遗留键（无 _image_ 段），Python 端
// nodes/image/pause_image.py 硬编码，两端必须一致。
//
// 与 Pixaroma 原件差异（已确认范围）：无 accent 颜色设置、无 Vue
// ResizeObserver 撑高（保留 onResize clamp）、无 canvas zoom 辅助。
//
// ==========================================================================

import { definePauseGate } from "./sf_pause_kit.js";

definePauseGate({
    classy: "SFPauseImage",
    extensionName: "sfnodes.PauseImage",
    widgetType: "sf_pause_image_ui",
    savePrefix: "PauseImage",

    stateProp: "pauseImageState",
    propPrefix: "_sfPauseImage",

    inputKey: "image",
    extraInputKeys: null,
    frameEventKey: "sf_pause_frame",

    logTag: "SF Pause Image",
    injectName: "Pause Image",
    captureMsg: "Run once to capture an image",

    cssId: "sf-pi-css",
    cssPrefix: "sf-pi-",
    emptyText: "Press Run to preview the image here",
    contTitle: "从快照运行工作流其余部分",
    regenTitle: "在此处掷一张新图（尊重你的种子）",
    toolNoun: "图片",
});
