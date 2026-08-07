// ==========================================================================
// sf_pause_image_lib.js - SFPauseImage 状态（纯函数）
// ==========================================================================
//
// 状态存 node.properties.pauseImageState（随工作流保存、Vue 标签页切换存活）。
// 形状保持最小（加载路径绝不重写序列化状态，不会误标工作流已修改）：
//   gate:  "pause"（默认）| "pass"
//   frame: { filename, subfolder, type }——最近一次快照（restore 用）
// "hasSnapshot" 与尺寸标签是运行时推导（frame 文件能否真实加载）：
// node._sfPauseImageHasSnapshot，由 ui.mjs showFrame() 设置。绝不能进
// node.properties——否则加载时的图片解析（如重启后 temp 快照已消失）会在打开
// 时改动已存状态、弄脏未编辑的工作流。
//
// prompt 剪枝（applyGateMode 等）复用 sf_pause_text_lib.js——同一份 prune
// 实现，PauseImage 以 {inputKey: "image"} 调用（见主扩展）。
//
// ==========================================================================

export const STATE_PROP = "pauseImageState";

export function getState(node) {
    node.properties = node.properties || {};
    let s = node.properties[STATE_PROP];
    if (!s || typeof s !== "object") {
        s = { gate: "pause", frame: null };
        node.properties[STATE_PROP] = s;
    }
    if (s.gate !== "pause" && s.gate !== "pass") s.gate = "pause";
    return s;
}

export function setGate(node, gate) {
    const s = getState(node);
    s.gate = gate === "pass" ? "pass" : "pause";
}
