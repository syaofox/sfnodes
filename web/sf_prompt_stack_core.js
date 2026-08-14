// ==========================================================================
// sf_prompt_stack_core.js - SFPromptStack 纯逻辑（无 DOM/app 依赖，可 .mjs 测试）
// ==========================================================================
// 状态存 node.properties.promptStackState（JSON 字符串，随工作流保存）。
// 形状对齐 Pixaroma PromptStack（rows/enabled/label/text）——prompt_reader
// 的 _pix_prompt_stack_extract 可直接恢复本节点生成的图。
// ==========================================================================

export const STATE_PROP = "promptStackState";
export const HIDDEN_INPUT = "PromptStackState"; // 匹配 Python INPUT_TYPES 键

export const MAX_ROWS = 500;
const DEFAULT_STATE = { version: 1, rows: [] };

// 行高范围（px）——UI 拖拽角标调节；h 为 null 时 UI 用默认 ROW_H
export const MIN_ROW_H = 40;
export const MAX_ROW_H = 300;

let _idc = 0;
export function newId() {
    try { if (crypto?.randomUUID) return "p" + crypto.randomUUID().slice(0, 8); } catch { /* 忽略 */ }
    return "p" + Date.now().toString(36) + (_idc++).toString(36);
}

function normRow(e) {
    if (!e || typeof e !== "object") return null;
    const text = typeof e.text === "string" ? e.text : "";
    let h = null;
    if (typeof e.h === "number" && Number.isFinite(e.h)) {
        h = Math.max(MIN_ROW_H, Math.min(MAX_ROW_H, Math.floor(e.h)));
    }
    return {
        id: typeof e.id === "string" && e.id ? e.id : newId(),
        enabled: e.enabled == null ? true : !!e.enabled,
        label: typeof e.label === "string" ? e.label : "",
        text,
        h, // 未设置/非法 → null（UI 用默认行高）
    };
}

export function normalize(raw) {
    const st = { ...DEFAULT_STATE, ...(raw && typeof raw === "object" ? raw : {}) };
    st.rows = (Array.isArray(st.rows) ? st.rows : [])
        .map(normRow)
        .filter(Boolean)
        .slice(0, MAX_ROWS);
    // 去重 id（手改/复制的状态不能有按 id 不可达的行）
    const seen = new Set();
    for (const r of st.rows) {
        if (seen.has(r.id)) r.id = newId();
        seen.add(r.id);
    }
    return st;
}

export function readState(node) {
    const v = node.properties?.[STATE_PROP];
    if (typeof v === "string" && v) {
        try { return normalize(JSON.parse(v)); } catch { /* 走默认 */ }
    }
    return normalize({});
}

// 写回 node.properties（序列化随工作流保存）
export function writeState(node, st) {
    node.properties[STATE_PROP] = JSON.stringify(normalize(st));
    node.setDirtyCanvas?.(true, true);
}

// 注入形状：只留执行字段（cosmetic 的 label/id 剥掉——注入字符串即缓存键，
// 改 id 不应触发重跑）。enabled 恒存在（false 也要传给后端过滤）。
export function promptState(st) {
    return {
        version: 1,
        rows: st.rows.map((r) => ({ enabled: !!r.enabled, text: r.text })),
    };
}

// 开且非空的行（与后端过滤语义一致）——输出 index 按此计算
export function activeRows(st) {
    return st.rows.filter((r) => r.enabled && r.text.trim());
}
