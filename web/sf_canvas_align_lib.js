// ==========================================================================
// sf_canvas_align_lib.js — 画布多选尺寸对齐纯逻辑（无 app/ DOM 依赖）
// 可直接拷 .mjs 用 Node 单测。
// ==========================================================================

// ── 选中集收集 ────────────────────────────────────────────────────────────
// 兼容 ComfyUI 多版本 selected_nodes 形态：Object / Array / Map / Set。
// 为空时回退扫描 graph._nodes 上 is_selected / flags.is_selected 标记
//（Vue 前端可能用此标记）。
export function getSelectedNodes(app) {
    const out = [];
    const seen = new Set();
    const push = (n) => {
        if (!n || seen.has(n)) return;
        seen.add(n);
        out.push(n);
    };
    try {
        const c = app?.canvas;
        const sel = c?.selected_nodes;
        if (sel) {
            let iter = null;
            if (Array.isArray(sel)) iter = sel;
            else if (sel instanceof Set) iter = [...sel];
            else if (sel instanceof Map) iter = [...sel.values()];
            else if (typeof sel.values === "function") {
                try { iter = [...sel.values()]; } catch { iter = Object.values(sel); }
            } else if (typeof sel === "object") iter = Object.values(sel);
            if (iter) for (const n of iter) push(n);
        }
        // current_node / node_over 单选兜底不算多选，但若 selected_nodes 为空
        // 且用户框选后仅 flags 标记，仍需回退扫描。
        if (out.length === 0) {
            // 尝试 current_node 作为补充（单选场景保持为空，避免误触发多选菜单）
        }
        // 回退扫描：selected_nodes 为空时才扫描，避免与已收集的重复
        if (out.length === 0) {
            const nodes = app?.graph?._nodes || app?.graph?.nodes || [];
            for (const n of nodes) {
                if (n && (n.is_selected || n.flags?.is_selected)) push(n);
            }
        }
        if (out.length >= 2) return out;
        // 若 selected_nodes 已有内容但 <2，直接返回（不回退扫描，避免把
        // 悬停/单选误判为多选）
        if (sel && out.length > 0) return out;
        return out;
    } catch {
        return out;
    }
}

// ── 目标宽度计算 ─────────────────────────────────────────────────────────
function nodeWidth(n) {
    if (!n) return 0;
    const w = n.size?.[0];
    if (typeof w === "number" && isFinite(w) && w > 0) return w;
    try {
        const cs = n.computeSize?.();
        if (Array.isArray(cs) && typeof cs[0] === "number" && isFinite(cs[0])) return cs[0];
    } catch { /* ignore */ }
    return 0;
}

export function calcTargetWidth(nodes, mode) {
    if (!Array.isArray(nodes) || nodes.length === 0) return 0;
    if (mode === "first") return nodeWidth(nodes[0]);
    if (mode === "narrowest") {
        let min = Infinity;
        for (const n of nodes) {
            const w = nodeWidth(n);
            if (w > 0 && w < min) min = w;
        }
        return min === Infinity ? 0 : min;
    }
    // widest (default)
    let max = 0;
    for (const n of nodes) {
        const w = nodeWidth(n);
        if (w > max) max = w;
    }
    return max;
}

// ── 对齐执行 ─────────────────────────────────────────────────────────────
function nodeHeight(n) {
    if (!n) return 0;
    const h = n.size?.[1];
    if (typeof h === "number" && isFinite(h) && h > 0) return h;
    try {
        const cs = n.computeSize?.();
        if (Array.isArray(cs) && typeof cs[1] === "number" && isFinite(cs[1])) return cs[1];
    } catch { /* ignore */ }
    return 0;
}

export function calcTargetHeight(nodes, mode) {
    if (!Array.isArray(nodes) || nodes.length === 0) return 0;
    if (mode === "first") return nodeHeight(nodes[0]);
    if (mode === "shortest") {
        let min = Infinity;
        for (const n of nodes) {
            const h = nodeHeight(n);
            if (h > 0 && h < min) min = h;
        }
        return min === Infinity ? 0 : min;
    }
    // tallest (default)
    let max = 0;
    for (const n of nodes) {
        const h = nodeHeight(n);
        if (h > max) max = h;
    }
    return max;
}

function minWidth(n) {
    try {
        const cs = n.computeSize?.();
        if (Array.isArray(cs) && typeof cs[0] === "number" && isFinite(cs[0])) return cs[0];
    } catch { /* ignore */ }
    return 0;
}

function minHeight(n) {
    try {
        const cs = n.computeSize?.();
        if (Array.isArray(cs) && typeof cs[1] === "number" && isFinite(cs[1])) return cs[1];
    } catch { /* ignore */ }
    return 0;
}

export function alignNodesWidth(nodes, targetW) {
    if (!Array.isArray(nodes) || nodes.length === 0) return 0;
    const tw = Number(targetW);
    if (!isFinite(tw) || tw <= 0) return 0;
    let count = 0;
    for (const n of nodes) {
        if (!n) continue;
        const mw = minWidth(n);
        const w = Math.max(tw, mw || 0);
        const h = nodeHeight(n) || n.size?.[1] || 0;
        if (!h) continue;
        try {
            if (typeof n.setSize === "function") n.setSize([w, h]);
            else n.size = [w, h];
            count++;
        } catch { /* ignore single node failure */ }
    }
    return count;
}

export function alignNodesHeight(nodes, targetH) {
    if (!Array.isArray(nodes) || nodes.length === 0) return 0;
    const th = Number(targetH);
    if (!isFinite(th) || th <= 0) return 0;
    let count = 0;
    for (const n of nodes) {
        if (!n) continue;
        const mh = minHeight(n);
        const h = Math.max(th, mh || 0);
        const w = nodeWidth(n) || n.size?.[0] || 0;
        if (!w) continue;
        try {
            if (typeof n.setSize === "function") n.setSize([w, h]);
            else n.size = [w, h];
            count++;
        } catch { /* ignore single node failure */ }
    }
    return count;
}

export function alignNodesSize(nodes, targetW, targetH) {
    if (!Array.isArray(nodes) || nodes.length === 0) return 0;
    const tw = Number(targetW);
    const th = Number(targetH);
    if ((!isFinite(tw) || tw <= 0) && (!isFinite(th) || th <= 0)) return 0;
    let count = 0;
    for (const n of nodes) {
        if (!n) continue;
        const mw = minWidth(n);
        const mh = minHeight(n);
        const w = isFinite(tw) && tw > 0 ? Math.max(tw, mw || 0) : (nodeWidth(n) || n.size?.[0] || 0);
        const h = isFinite(th) && th > 0 ? Math.max(th, mh || 0) : (nodeHeight(n) || n.size?.[1] || 0);
        if (!w || !h) continue;
        try {
            if (typeof n.setSize === "function") n.setSize([w, h]);
            else n.size = [w, h];
            count++;
        } catch { /* ignore single node failure */ }
    }
    return count;
}
