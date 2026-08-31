// ==========================================================================
// sf_canvas_align.js — 画布多选宽度对齐（主扩展）
// 选中 ≥2 节点时在画布背景右键菜单注入 SF Align Width 子菜单：
//   Widest / Narrowest / First Selected，仅改 size[0]，高度不变。
// ==========================================================================

import { app } from "/scripts/app.js";
import {
    getSelectedNodes,
    calcTargetWidth,
    alignNodesWidth,
} from "./sf_canvas_align_lib.js";

function doAlign(mode) {
    const nodes = getSelectedNodes(app);
    if (nodes.length < 2) return;
    const tw = calcTargetWidth(nodes, mode);
    if (!tw) return;
    const g = app.graph;
    try { g?.beforeChange?.(); } catch { /* ignore */ }
    alignNodesWidth(nodes, tw);
    try { g?.afterChange?.(); } catch { /* ignore */ }
    try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    try { g?.setDirtyCanvas?.(true, true); } catch { /* ignore */ }
}

app.registerExtension({
    name: "sfnodes.CanvasAlign",

    getCanvasMenuItems() {
        const nodes = getSelectedNodes(app);
        if (nodes.length < 2) return [];
        // LiteGraph 菜单项支持 has_submenu + submenu.options（Classic）；
        // ComfyUI 前端对 getCanvasMenuItems 的返回值会透传给 LiteGraph
        // ContextMenu，两种形态均可。has_submenu 显式标记可提升兼容性。
        return [{
            content: "SF Align Width",
            has_submenu: true,
            submenu: {
                options: [
                    { content: "Width \u2192 Widest", callback: () => doAlign("widest") },
                    { content: "Width \u2192 Narrowest", callback: () => doAlign("narrowest") },
                    { content: "Width \u2192 First Selected", callback: () => doAlign("first") },
                ],
            },
        }];
    },
});
