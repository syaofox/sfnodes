// ==========================================================================
// sf_canvas_menu.js — 本包聚合菜单（唯一顶层入口，画布背景右键 + 节点右键）
//   📦 SF Menu ▶ SF Align ▶（≥2 选中；<2 时退化为 disabled 提示行）/
//                 SF Node Color…（≥1 选中时）/ SF LoRA Browser /
//                 SF LoRA Presets / SF Workflows / SF Memory ▶ / Add SF Note
// 排序：上下文相关项（Align/Node Color）在前，全局工具随后（见 §125）。
// 双入口共用同一组装（buildSfMenuOptions）：getCanvasMenuItems（空白处右键，
// 返回 [null, 项] 让原生菜单与本包之间出现分隔线）与 getNodeMenuItems(node)
// （节点右键，前端 GraphView 会收编进节点菜单；右键节点时前端已先行选中该
// 节点，选中集构建器语义不变，见 experience/platform.md §123）。节点入口把
// node 作为 Align 的 Mouse Node 基准传入（空白右键无基准，该三项不注入，
// 见 §124）；节点入口不加分隔线（Vue 单选时 LiteGraph 块前置，会变成菜单
// 顶部横线）。
// 各动作实现仍在原特性文件（零逻辑复制），本文件只做组装：
//   对齐 buildAlignMenuItems（sf_canvas_align.js）/ 节点任意色 buildNodeColorMenuItem
//   （sf_node_color.js）/ 内存 buildMemoryMenuItem（sf_memory_menu.js）/
//   工作流 openWorkflowsPanel（sf_workflows.js）/ 浏览器 openLoraBrowser
//   （sf_lora_browser.js）/ 预设管理 openLoraPresetManager（sf_lora_preset_manager.js，
//   独立模式：无 node 自动隐藏保存栏）/ 便签 addNoteFromMenu（sf_note.js，落点 =
//   菜单构建时捕获的右键位置，见上）。
// 子菜单结构 has_submenu + submenu.options（sf_canvas_align §11 先例，
// Classic/Vue 双兼容）；SF Align 内为单层平铺（动作 + disabled 分组头），
// 组内直达无需再展开，见 experience/platform.md §2.15。
// ==========================================================================

import { app } from "/scripts/app.js";
import { buildAlignMenuItems } from "./sf_canvas_align.js";
import { buildNodeColorMenuItem } from "./sf_node_color.js";
import { buildMemoryMenuItem } from "./sf_memory_menu.js";
import { openWorkflowsPanel } from "./sf_workflows.js";
import { openLoraBrowser } from "./sf_lora_browser.js";
import { openLoraPresetManager } from "./sf_lora_preset_manager.js";
import { addNoteFromMenu } from "./sf_note.js";

// ── 右键位置捕获（Add SF Note 落点）───────────────────────────────────────
// 菜单项点击时鼠标已移到菜单上，落点必须在菜单构建时捕获。pointerdown
// capture 早于前端建菜单（Classic 由 pointerup 的 onClick 触发
// processContextMenu；Vue 由节点 contextmenu 触发），Classic/Vue 通用；
// 坐标换算与前端 adjustMouseEvent 同式（client → canvas）。
let lastRightClickPos = null;

function clientToCanvas(clientX, clientY) {
    const c = app?.canvas;
    const el = c?.canvas;
    if (!c || !el || typeof el.getBoundingClientRect !== "function") return null;
    const rect = el.getBoundingClientRect();
    const scale = c.ds?.scale || 1;
    const off = c.ds?.offset || [0, 0];
    const x = (clientX - rect.left) / scale - (off[0] || 0);
    const y = (clientY - rect.top) / scale - (off[1] || 0);
    return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
    window.addEventListener("pointerdown", (e) => {
        if (e?.button !== 2) return;
        const pos = clientToCanvas(e.clientX, e.clientY);
        if (pos) lastRightClickPos = pos;
    }, true);
}

// 建菜单时取落点：优先最近一次右键位置，兜底前端 graph_mouse
//（键盘开菜单等无 pointerdown 场景），再兜底 null（sf_note 走视口中心）。
function menuOpenPos() {
    if (lastRightClickPos) return lastRightClickPos;
    const gm = app?.canvas?.graph_mouse;
    if (Array.isArray(gm) && Number.isFinite(gm[0]) && Number.isFinite(gm[1])) return [gm[0], gm[1]];
    return null;
}

function buildSfMenuOptions(refNode) {
    const options = [];
    const openPos = menuOpenPos();
    // 对齐是多选操作：<2 节点不注入动作（sf_canvas_align 原守卫语义），
    // 保留 disabled 提示行提升可发现性（无 callback，点击无操作 fail-safe）。
    const align = buildAlignMenuItems(refNode);
    if (align.length) {
        options.push({
            content: "SF Align",
            has_submenu: true,
            submenu: { options: align },
        });
    } else {
        options.push({ content: "SF Align (select ≥2 nodes)", disabled: true });
    }
    // 任意节点颜色需 ≥1 选中节点（无选中返回 null，不注入）。
    const nodeColor = buildNodeColorMenuItem();
    if (nodeColor) options.push(nodeColor);
    options.push(
        { content: "SF LoRA Browser", callback: openLoraBrowser },
        { content: "SF LoRA Presets", callback: () => openLoraPresetManager() },
        { content: "SF Workflows", callback: openWorkflowsPanel },
        buildMemoryMenuItem(),
        { content: "Add SF Note", callback: () => addNoteFromMenu(openPos) }
    );
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

    // 画布空白右键：前导 null = 原生项与本包之间的分隔线（LiteGraph ContextMenu
    // 与 Vue 转换器都认）。
    getCanvasMenuItems() {
        return [null, buildSfMenuItem()];
    },

    // 节点右键：同一菜单，node 兼作 Align 的 Mouse Node 基准（任意节点可用）。
    getNodeMenuItems(node) {
        return [buildSfMenuItem(node)];
    },
});
