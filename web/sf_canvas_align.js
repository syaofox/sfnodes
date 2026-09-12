// ==========================================================================
// sf_canvas_align.js — 画布多选尺寸对齐（菜单项构建器）
// 选中 ≥2 节点时供聚合菜单（web/sf_canvas_menu.js，📦 SF Menu ▶ SF Align）
// 提供三入口：
//   Width  ▶ Widest / Narrowest / First Selected（仅改 size[0]）
//   Height ▶ Tallest / Shortest / First Selected（仅改 size[1]）
//   Size   ▶ Widest & Tallest / Narrowest & Shortest / First Selected（两维同改）
// 本文件不再自行注册画布菜单（分散入口已收敛到聚合器），只 export 构建器。
// ==========================================================================

import { app } from "/scripts/app.js";
import {
    getSelectedNodes,
    calcTargetWidth,
    calcTargetHeight,
    alignNodesWidth,
    alignNodesHeight,
    alignNodesSize,
} from "./sf_canvas_align_lib.js";

function doAlignWidth(mode) {
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

function doAlignHeight(mode) {
    const nodes = getSelectedNodes(app);
    if (nodes.length < 2) return;
    const th = calcTargetHeight(nodes, mode);
    if (!th) return;
    const g = app.graph;
    try { g?.beforeChange?.(); } catch { /* ignore */ }
    alignNodesHeight(nodes, th);
    try { g?.afterChange?.(); } catch { /* ignore */ }
    try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    try { g?.setDirtyCanvas?.(true, true); } catch { /* ignore */ }
}

function doAlignSize(mode) {
    const nodes = getSelectedNodes(app);
    if (nodes.length < 2) return;
    const tw = calcTargetWidth(nodes, mode === "shortest" ? "narrowest" : mode);
    const th = calcTargetHeight(nodes, mode);
    if (!tw && !th) return;
    const g = app.graph;
    try { g?.beforeChange?.(); } catch { /* ignore */ }
    alignNodesSize(nodes, tw, th);
    try { g?.afterChange?.(); } catch { /* ignore */ }
    try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    try { g?.setDirtyCanvas?.(true, true); } catch { /* ignore */ }
}

export function buildAlignMenuItems() {
    const nodes = getSelectedNodes(app);
    if (nodes.length < 2) return [];
    // LiteGraph 菜单项支持 has_submenu + submenu.options（Classic）；
    // ComfyUI 前端对 getCanvasMenuItems 的返回值会透传给 LiteGraph
    // ContextMenu，两种形态均可。has_submenu 显式标记可提升兼容性。
    return [
        {
            content: "SF Align Width",
            has_submenu: true,
            submenu: {
                options: [
                    { content: "Width \u2192 Widest", callback: () => doAlignWidth("widest") },
                    { content: "Width \u2192 Narrowest", callback: () => doAlignWidth("narrowest") },
                    { content: "Width \u2192 First Selected", callback: () => doAlignWidth("first") },
                ],
            },
        },
        {
            content: "SF Align Height",
            has_submenu: true,
            submenu: {
                options: [
                    { content: "Height \u2192 Tallest", callback: () => doAlignHeight("tallest") },
                    { content: "Height \u2192 Shortest", callback: () => doAlignHeight("shortest") },
                    { content: "Height \u2192 First Selected", callback: () => doAlignHeight("first") },
                ],
            },
        },
        {
            content: "SF Align Size",
            has_submenu: true,
            submenu: {
                options: [
                    { content: "Size \u2192 Widest & Tallest", callback: () => doAlignSize("widest") },
                    { content: "Size \u2192 Narrowest & Shortest", callback: () => doAlignSize("shortest") },
                    { content: "Size \u2192 First Selected", callback: () => doAlignSize("first") },
                ],
            },
        },
    ];
}
