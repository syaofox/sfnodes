// ==========================================================================
// sf_canvas_align.js — 画布多选尺寸对齐（菜单项构建器）
// 选中 ≥2 节点时供聚合菜单（web/sf_canvas_menu.js，📦 SF Menu ▶ SF Align）
// 提供 9 动作（单层平铺，组间 disabled 分组头 Width/Height/Size 隔开）：
//   Width  组：Width: Widest / Narrowest / First Selected（仅改 size[0]）
//   Height 组：Height: Tallest / Shortest / First Selected（仅改 size[1]）
//   Size   组：Size: Widest & Tallest / Narrowest & Shortest / First Selected（两维同改）
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
    // 单层平铺：分组头用 disabled 项（无 callback，点击无操作 fail-safe），
    // 动作标签带组前缀以保唯一可区分（测试按 content 直取）。
    const header = (content) => ({ content, disabled: true });
    return [
        header("Width"),
        { content: "Width: Widest", callback: () => doAlignWidth("widest") },
        { content: "Width: Narrowest", callback: () => doAlignWidth("narrowest") },
        { content: "Width: First Selected", callback: () => doAlignWidth("first") },
        header("Height"),
        { content: "Height: Tallest", callback: () => doAlignHeight("tallest") },
        { content: "Height: Shortest", callback: () => doAlignHeight("shortest") },
        { content: "Height: First Selected", callback: () => doAlignHeight("first") },
        header("Size"),
        { content: "Size: Widest & Tallest", callback: () => doAlignSize("widest") },
        { content: "Size: Narrowest & Shortest", callback: () => doAlignSize("shortest") },
        { content: "Size: First Selected", callback: () => doAlignSize("first") },
    ];
}
