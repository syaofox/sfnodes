// ==========================================================================
// sf_canvas_menu.js — 本包聚合菜单（唯一顶层入口，画布背景右键 + 节点右键）
//   📦 SF Menu ▶ SF LoRA Browser / SF Workflows /
//                 SF Align ▶（≥2 选中节点时）/ SF Node Color…（≥1 选中节点时）/
//                 SF Memory ▶
// 双入口共用同一组装（buildSfMenuOptions）：getCanvasMenuItems（空白处右键）
// 与 getNodeMenuItems(node)（节点右键，前端 GraphView 会收编进节点菜单；
// 右键节点时前端已先行选中该节点，选中集构建器语义不变，见
// experience/platform.md §123）。节点入口把 node 作为 Align 的 Mouse Node
// 基准传入（空白右键无基准，该三项不注入，见 §124）。
// 各动作实现仍在原特性文件（零逻辑复制），本文件只做组装：
//   对齐 buildAlignMenuItems（sf_canvas_align.js）/ 节点任意色 buildNodeColorMenuItem
//   （sf_node_color.js）/ 内存 buildMemoryMenuItem（sf_memory_menu.js）/
//   工作流 openWorkflowsPanel（sf_workflows.js）/ 浏览器 openLoraBrowser
//   （sf_lora_browser.js）。
// （便签 SF Note 节点保留可搜索添加，不再占画布菜单入口。）
// 子菜单结构 has_submenu + submenu.options（sf_canvas_align §11 先例，
// Classic/Vue 双兼容）；SF Align 内为单层平铺（9 动作 + disabled 分组头），
// 组内直达无需再展开，见 experience/platform.md §2.15。
// ==========================================================================

import { app } from "/scripts/app.js";
import { buildAlignMenuItems } from "./sf_canvas_align.js";
import { buildNodeColorMenuItem } from "./sf_node_color.js";
import { buildMemoryMenuItem } from "./sf_memory_menu.js";
import { openWorkflowsPanel } from "./sf_workflows.js";
import { openLoraBrowser } from "./sf_lora_browser.js";

function buildSfMenuOptions(refNode) {
    const options = [
        { content: "SF LoRA Browser", callback: openLoraBrowser },
        { content: "SF Workflows", callback: openWorkflowsPanel },
    ];
    // 对齐是多选操作：<2 节点不注入（sf_canvas_align 原守卫语义）。
    const align = buildAlignMenuItems(refNode);
    if (align.length) {
        options.push({
            content: "SF Align",
            has_submenu: true,
            submenu: { options: align },
        });
    }
    // 任意节点颜色需 ≥1 选中节点（无选中返回 null，不注入）。
    const nodeColor = buildNodeColorMenuItem();
    if (nodeColor) options.push(nodeColor);
    options.push(buildMemoryMenuItem());
    return options;
}

function buildSfMenuItem(refNode) {
    return {
        content: "📦 SF Menu",
        has_submenu: true,
        submenu: { options: buildSfMenuOptions(refNode) },
    };
}

app.registerExtension({
    name: "sfnodes.CanvasMenu",

    getCanvasMenuItems() {
        return [buildSfMenuItem()];
    },

    // 节点右键：同一菜单，node 兼作 Align 的 Mouse Node 基准（任意节点可用）。
    getNodeMenuItems(node) {
        return [buildSfMenuItem(node)];
    },
});
