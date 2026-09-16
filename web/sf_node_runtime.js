// ==========================================================================
// sf_node_runtime.js — 节点运行时间显示
//
// 复刻 ComfyUI-Easy-Use 的 Comfy.EasyUse.TimeTaken：监听 ComfyUI
// execution_start / executing 事件，用相邻 executing 的墙钟间隔估算节点耗时，
// 显示 "x.xxx s"。开关 sfnodes.NodeRuntime.Enabled，**默认关**（Easy-Use
// 默认开，本包按需求默认关）。
//
// 显示双路（platform.md §2.18）：
//   - Classic 渲染器：`onDrawForeground` 逐节点类型补丁绘制（Easy-Use 原做法，
//     项目内 sf_dropdown/sf_find_replace/sf_image_resize 等已用）。
//   - Vue Nodes 2.0：per-node onDrawForeground 不触发，改用原生 `node.badges`
//     （`window.LGraphBadge` 实例，官方 Comfy.NodeBadge 同款），并以
//     graph.trigger("node:property:changed",{property:"badges"}) 触发刷新。
//
// 耗时估算：记录上一个 executing 节点 id 与开始时间，收到下一个 executing 时
// 结算上一个节点 += 间隔；execution_start 清空全部（每轮重新计时）。
// 纯计算收敛于 sf_node_runtime_lib.js（可 .mjs 直测）。
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import {
    formatDuration,
    accumulateSeconds,
    resolveNodeId,
    isRuntimeBadge,
    markRuntimeBadge,
} from "./sf_node_runtime_lib.js";

const SETTING_ID = "sfnodes.NodeRuntime.Enabled";

function isEnabled() {
    try {
        const v = app.ui?.settings?.getSettingValue?.(SETTING_ID);
        return v === true || v === "true";
    } catch {
        return false;
    }
}

function litegraph() {
    return globalThis.LiteGraph || globalThis.window?.LiteGraph || null;
}

function isVueMode() {
    const LG = litegraph();
    return !!(LG && LG.vueNodesMode);
}

function themeColors() {
    const LG = litegraph();
    return {
        fg: (LG && LG.NODE_TITLE_COLOR) || "#999",
        bg: (LG && LG.NODE_DEFAULT_BGCOLOR) || "#353535",
        titleH: (LG && LG.NODE_TITLE_HEIGHT) || 30,
    };
}

// ── Classic：onDrawForeground 绘制 ────────────────────────────────────────
// 复刻 Easy-Use 的绘制：标题栏上方一个圆角小方块 "x.xxx s"。
function drawDurationLabel(ctx, seconds) {
    const { fg, bg, titleH } = themeColors();
    const text = formatDuration(seconds);
    const height = titleH - 10;
    ctx.save();
    ctx.font = "12px Inter, sans-serif";
    const width = ctx.measureText(text).width + 10;
    const x = 0;
    const y = -titleH - 20;
    ctx.fillStyle = bg;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, width, height, 4);
    else ctx.rect(x, y, width, height);
    ctx.fill();
    ctx.fillStyle = fg;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(text, 8, -titleH - 6);
    ctx.restore();
}

// ── Vue Nodes 2.0：原生 node.badges ───────────────────────────────────────
function makeBadge(seconds) {
    const Badge = globalThis.LGraphBadge || globalThis.window?.LGraphBadge;
    if (typeof Badge !== "function") return null;
    const { fg, bg } = themeColors();
    try {
        return markRuntimeBadge(new Badge({ text: formatDuration(seconds), fgColor: fg, bgColor: bg }));
    } catch {
        return null;
    }
}

function upsertBadge(node, seconds) {
    const badge = makeBadge(seconds);
    if (!badge) return;
    let list = node.badges;
    if (!Array.isArray(list)) list = node.badges = [];
    const idx = list.findIndex(isRuntimeBadge);
    if (idx >= 0) list[idx] = badge;
    else list.push(badge);
}

function allNodes() {
    try {
        return app.graph?._nodes || app.graph?.nodes || [];
    } catch {
        return [];
    }
}

function findNode(id) {
    if (id == null) return null;
    const g = app.graph;
    if (!g || typeof g.getNodeById !== "function") return null;
    try {
        return g.getNodeById(id) || g.getNodeById(String(id)) || null;
    } catch {
        return null;
    }
}

function refreshNode(node) {
    try {
        node.setDirtyCanvas?.(true, true);
    } catch {
        /* ignore */
    }
    try {
        app.canvas?.setDirty?.(true, true);
    } catch {
        /* ignore */
    }
    // Vue Nodes 2.0：nodeManager 依据 node:property:changed 事件刷新 badges
    // （官方 Comfy.NodeBadge 同款触发方式）。
    try {
        const g = node.graph || app.graph;
        g?.trigger?.("node:property:changed", {
            type: "node:property:changed",
            nodeId: node.id,
            property: "badges",
            oldValue: node.badges,
            newValue: node.badges,
        });
    } catch {
        /* ignore */
    }
}

function setNodeDuration(node, seconds) {
    if (!node) return;
    try {
        node.executionDuration = seconds;
    } catch {
        /* ignore */
    }
    if (isVueMode()) upsertBadge(node, seconds);
    refreshNode(node);
}

function clearNodeDuration(node) {
    if (!node) return;
    const list = node.badges;
    const idx = Array.isArray(list) ? list.findIndex(isRuntimeBadge) : -1;
    // 无本特性 badge 且无耗时记录时直接跳过，避免 execution_start 时对
    // 整图每个节点都触发一次 property 事件 / 重绘。
    if (idx < 0 && node.executionDuration == null) return;
    try {
        delete node.executionDuration;
    } catch {
        /* ignore */
    }
    if (idx >= 0) list.splice(idx, 1);
    refreshNode(node);
}

function clearAll() {
    for (const node of allNodes()) clearNodeDuration(node);
}

// ── 计时状态 ──────────────────────────────────────────────────────────────
let _lastNodeId = null;
const _startTimes = new Map();

function recordExecution(detail) {
    const started = _startTimes.get(_lastNodeId);
    _startTimes.delete(_lastNodeId);
    if (_lastNodeId != null && started != null) {
        const node = findNode(_lastNodeId);
        if (node) {
            const elapsedMs = Date.now() - started;
            setNodeDuration(node, accumulateSeconds(node.executionDuration, elapsedMs));
        }
    }
    const id = resolveNodeId(detail);
    _lastNodeId = id;
    if (id != null) _startTimes.set(id, Date.now());
}

api.addEventListener("execution_start", () => {
    _lastNodeId = null;
    _startTimes.clear();
    clearAll();
});

api.addEventListener("executing", (event) => {
    if (!isEnabled()) return;
    recordExecution(event?.detail ?? event);
});

app.registerExtension({
    name: "sfnodes.NodeRuntime",

    init() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "SF: show node execution time badge (seconds)",
                type: "boolean",
                defaultValue: false,
                // 关闭时立即移除已显示的耗时（开启后需等下一轮执行才有数据）。
                onChange: (value) => {
                    if (!value) clearAll();
                },
            });
        } catch {
            /* 设置系统不可用则忽略 */
        }
    },

    // Classic 渲染器：逐节点类型包装 onDrawForeground 绘制耗时（Easy-Use 同款）。
    beforeRegisterNodeDef(nodeType) {
        const orig = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (
                !isVueMode() &&
                isEnabled() &&
                !this.flags?.collapsed &&
                this.executionDuration != null
            ) {
                try {
                    drawDurationLabel(ctx, this.executionDuration);
                } catch {
                    /* ignore */
                }
            }
            return orig?.apply(this, arguments);
        };
    },
});
