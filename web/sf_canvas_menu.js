// ==========================================================================
// sf_canvas_menu.js — 本包画布背景右键聚合菜单（唯一顶层入口）
//   📦 SF Menu ▶ SF LoRA Browser / SF Workflows /
//                 SF Align ▶（≥2 选中节点时）/ SF Memory ▶
// 各动作实现仍在原特性文件（零逻辑复制），本文件只做组装：
//   对齐 buildAlignMenuItems（sf_canvas_align.js）/ 内存 buildMemoryMenuItem
//   （sf_memory_menu.js）/ 工作流 openWorkflowsPanel（sf_workflows.js）/
//   浏览器 openLoraBrowser（sf_lora_browser.js）。
// （便签 SF Note 节点保留可搜索添加，不再占画布菜单入口。）
// 子菜单结构 has_submenu + submenu.options（sf_canvas_align §11 先例，
// Classic/Vue 双兼容）；SF Align 内为单层平铺（9 动作 + disabled 分组头），
// 组内直达无需再展开，见 experience/platform.md §2.15。
// ==========================================================================

import { app } from "/scripts/app.js";
import { buildAlignMenuItems } from "./sf_canvas_align.js";
import { buildMemoryMenuItem } from "./sf_memory_menu.js";
import { openWorkflowsPanel } from "./sf_workflows.js";
import { openLoraBrowser } from "./sf_lora_browser.js";

app.registerExtension({
    name: "sfnodes.CanvasMenu",

    getCanvasMenuItems() {
        const options = [
            { content: "SF LoRA Browser", callback: openLoraBrowser },
            { content: "SF Workflows", callback: openWorkflowsPanel },
        ];
        // 对齐是多选操作：<2 节点不注入（sf_canvas_align 原守卫语义）。
        const align = buildAlignMenuItems();
        if (align.length) {
            options.push({
                content: "SF Align",
                has_submenu: true,
                submenu: { options: align },
            });
        }
        options.push(buildMemoryMenuItem());
        return [
            {
                content: "📦 SF Menu",
                has_submenu: true,
                submenu: { options },
            },
        ];
    },
});
