// ==========================================================================
// sf_pause_latent_lib.js - SFPauseLatent 状态（纯函数）
// ==========================================================================
//
// 与 sf_pause_image_lib.js 同构（30 行平行实现），仅 STATE_PROP 键名不同
// （pauseLatentState）。状态存 node.properties（随工作流保存）：
//   gate:  "pause"（默认）| "pass"
//   frame: { filename, subfolder, type }——最近一次预览快照（restore 用）
// "hasSnapshot" 是运行时推导（frame 文件能否真实加载），绝不住 properties。
//
// prompt 剪枝复用 sf_pause_text_lib.js::applyGateMode——同一份 prune
// 实现，SFPauseLatent 以 {inputKey: "latent", extraInputKeys: ["image"]}
// 调用（预览 image 链接在 continue 时一并删除，见主扩展）。
//
// ==========================================================================

export const STATE_PROP = "pauseLatentState";

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
