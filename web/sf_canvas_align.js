// ==========================================================================
// sf_canvas_align.js — 画布多选尺寸对齐（菜单项构建器）
// 选中 ≥2 节点时供聚合菜单（web/sf_canvas_menu.js，📦 SF Menu ▶ SF Align）
// 动作（单层平铺，组间 disabled 分组头 Width/Height/Size 隔开）：
//   Width  组：Width: Widest / Narrowest / Mouse Node（仅改 size[0]）
//   Height 组：Height: Tallest / Shortest / Mouse Node（仅改 size[1]）
//   Size   组：Size: Widest & Tallest / Narrowest & Shortest / Mouse Node（两维同改）
// Mouse Node = 打开菜单时鼠标下的节点（节点右键入口传 node；画布空白右键
// 无基准，三项不注入）。基准必须在建菜单时捕获——点击菜单项时鼠标已移开。
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

function doAlignWidth(mode, refNode) {
    const nodes = getSelectedNodes(app);
    if (nodes.length < 2) return;
    const tw = calcTargetWidth(nodes, mode, refNode);
    if (!tw) return;
    const g = app.graph;
    try { g?.beforeChange?.(); } catch { /* ignore */ }
    alignNodesWidth(nodes, tw);
    try { g?.afterChange?.(); } catch { /* ignore */ }
    try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    try { g?.setDirtyCanvas?.(true, true); } catch { /* ignore */ }
}

function doAlignHeight(mode, refNode) {
    const nodes = getSelectedNodes(app);
    if (nodes.length < 2) return;
    const th = calcTargetHeight(nodes, mode, refNode);
    if (!th) return;
    const g = app.graph;
    try { g?.beforeChange?.(); } catch { /* ignore */ }
    alignNodesHeight(nodes, th);
    try { g?.afterChange?.(); } catch { /* ignore */ }
    try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    try { g?.setDirtyCanvas?.(true, true); } catch { /* ignore */ }
}

function doAlignSize(mode, refNode) {
    const nodes = getSelectedNodes(app);
    if (nodes.length < 2) return;
    const tw = calcTargetWidth(nodes, mode === "shortest" ? "narrowest" : mode, refNode);
    const th = calcTargetHeight(nodes, mode, refNode);
    if (!tw && !th) return;
    const g = app.graph;
    try { g?.beforeChange?.(); } catch { /* ignore */ }
    alignNodesSize(nodes, tw, th);
    try { g?.afterChange?.(); } catch { /* ignore */ }
    try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    try { g?.setDirtyCanvas?.(true, true); } catch { /* ignore */ }
}

export function buildAlignMenuItems(refNode) {
    const nodes = getSelectedNodes(app);
    if (nodes.length < 2) return [];
    // LiteGraph 菜单项支持 has_submenu + submenu.options（Classic）；
    // ComfyUI 前端对 getCanvasMenuItems 的返回值会透传给 LiteGraph
    // ContextMenu，两种形态均可。has_submenu 显式标记可提升兼容性。
    // 单层平铺：分组头用 disabled 项（无 callback，点击无操作 fail-safe），
    // 动作标签带组前缀以保唯一可区分（测试按 content 直取）。
    const header = (content) => ({ content, disabled: true });
    // 无鼠标基准（画布空白右键）时不注入 Mouse Node 项，避免死行。
    const mouse = (group, run) => (refNode
        ? [{ content: `${group}: Mouse Node`, callback: () => run("mouse", refNode) }]
        : []);
    return [
        header("Width"),
        { content: "Width: Widest", callback: () => doAlignWidth("widest") },
        { content: "Width: Narrowest", callback: () => doAlignWidth("narrowest") },
        ...mouse("Width", doAlignWidth),
        header("Height"),
        { content: "Height: Tallest", callback: () => doAlignHeight("tallest") },
        { content: "Height: Shortest", callback: () => doAlignHeight("shortest") },
        ...mouse("Height", doAlignHeight),
        header("Size"),
        { content: "Size: Widest & Tallest", callback: () => doAlignSize("widest") },
        { content: "Size: Narrowest & Shortest", callback: () => doAlignSize("shortest") },
        ...mouse("Size", doAlignSize),
    ];
}
