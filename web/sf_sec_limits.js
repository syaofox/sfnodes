// SF SeC 数值上限补丁（纯前端，无需重启）：
// ComfyUI 前端 widget 工厂对未声明 max 的 INT/FLOAT 输入默认上限 2048
// （settingStore-*.js: `max: t.max ?? 2048`）。Comfyui-SecNodes 的
// annotation_frame_idx / object_id / max_frames_to_track 都没写 max，
// 导致长视频（>2048 帧）无法填写标注帧。这里在节点定义阶段把 max 抬高，
// 使 widget 建出时就带大 max；不依赖改第三方包，也不怕其更新。
import { app } from "/scripts/app.js";

const NODE = "SeCVideoSegmentation";
export const RAISE = {
    annotation_frame_idx: 1000000,
    object_id: 1000000,
    max_frames_to_track: 1000000,
};

// 抬高 input spec（[type, opts]）的 max；已有更大的 max 不降级
export function raiseSpec(spec, value) {
    if (Array.isArray(spec) && spec[1] && typeof spec[1] === "object") {
        spec[1].max = Math.max(Number(spec[1].max) || 0, value);
    }
}

function patchNode(node) {
    if (node?.comfyClass !== NODE && node?.type !== NODE) return;
    for (const [name, value] of Object.entries(RAISE)) {
        const w = node.widgets?.find((x) => x?.name === name);
        if (!w) continue;
        w.options = w.options || {};
        w.options.max = Math.max(Number(w.options.max) || 0, value);
    }
}

app.registerExtension({
    name: "sfnodes.SecLimits",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;
        for (const [name, value] of Object.entries(RAISE)) {
            raiseSpec(nodeData.input?.optional?.[name], value);
            raiseSpec(nodeData.input?.required?.[name], value);
        }
        // 兜底：若 widget 已按旧 max 建好（如定义缓存），创建/恢复时再补一次
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            patchNode(this);
            return r;
        };
        const onConfigured = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function () {
            const r = onConfigured?.apply(this, arguments);
            patchNode(this);
            return r;
        };
    },
});
