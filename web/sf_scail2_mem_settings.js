// ==========================================================================
// sf_scail2_mem_settings.js - SCAIL-2 预处理内存优化设置
// ==========================================================================
//
// 注册 ComfyUI Settings 项，控制后端运行时补丁 sf_utils/scail2_mem.py：
// 整段彩色蒙版渲染 / 28ch 提取按帧分块（O(T)→O(chunk)），以及
// WanSCAILToVideo 参考蒙版裁剪（只上采样真正用到的前 n_ref 帧）。
// 后端每节点执行时读服务器 comfy.settings.json，改动即时生效、无需重启。
//
// 设置系统不可用时静默降级（后端回落默认值：开启 / 32 帧 / f16）。
// ==========================================================================

import { app } from "/scripts/app.js";

export const SETTING_ENABLED = "sfnodes.SCAIL2Mem.Enabled";
export const SETTING_CHUNK = "sfnodes.SCAIL2Mem.ChunkFrames";
export const SETTING_HALF = "sfnodes.SCAIL2Mem.HalfPrecision";

let _registered = false;

export function registerScail2MemSettings() {
    if (_registered) return;
    _registered = true;
    try {
        const s = app.ui.settings;
        s.addSetting({
            id: SETTING_ENABLED,
            name: "SF SCAIL-2: memory-efficient mask preprocessing (chunked render/extract + ref trim)",
            defaultValue: true,
            type: "boolean",
        });
        s.addSetting({
            id: SETTING_CHUNK,
            name: "SF SCAIL-2: mask preprocessing chunk frames",
            defaultValue: 32,
            type: "slider",
            attrs: { min: 1, max: 256, step: 1 },
        });
        s.addSetting({
            id: SETTING_HALF,
            name: "SF SCAIL-2: output colored masks as float16 (halves peak memory)",
            defaultValue: true,
            type: "boolean",
        });
    } catch (e) {
        console.warn("[sfnodes] SCAIL-2 memory settings unavailable", e);
    }
}

app.registerExtension({
    name: "sfnodes.SCAIL2Mem",
    init() {
        registerScail2MemSettings();
    },
});
