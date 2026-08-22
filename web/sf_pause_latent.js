// ==========================================================================
// sf_pause_latent.js - SFPauseLatent 主扩展（薄配置）
// ==========================================================================
//
// 全部机制收敛于 web/sf_pause_kit.js（definePauseGate），与 SFPauseImage 同构。
// LATENT 类型，分段采样中间暂停：Pause 时 Python 把中间态 latent 整 batch 存
// 成 safetensors（temp 目录按节点 id 命名），Continue 时前端把第一段采样（整条
// 上游链）剪出 prompt、Python 读回快照——第二段采样器从暂停时那份精确的中间态
// 继续，第一段完全不重跑。预览显示 image 输入（VAEDecode 结果），它只在 Pause
// 时在场；Continue 时经 extraInputKeys 连同 latent 链接一并剪掉，否则 VAEDecode
// 仍被消费会把第一段采样器拉活。
//
// ==========================================================================

import { definePauseGate } from "./sf_pause_kit.js";

definePauseGate({
    classy: "SFPauseLatent",
    extensionName: "sfnodes.PauseLatent",
    widgetType: "sf_pause_latent_ui",
    savePrefix: "PauseLatent",

    stateProp: "pauseLatentState",
    propPrefix: "_sfPauseLatent",

    inputKey: "latent",
    extraInputKeys: ["image"],
    frameEventKey: "sf_pause_latent_frame",

    logTag: "SF Pause Latent",
    injectName: "Pause Latent",
    captureMsg: "Run once to capture an image",

    cssId: "sf-pl-css",
    cssPrefix: "sf-pl-",
    emptyText: "Press Run to preview the image here",
    contTitle: "从 latent 快照运行工作流其余部分（第二段采样继续）",
    regenTitle: "重新采样第一段（尊重你的种子）",
    toolNoun: "图片",
});
