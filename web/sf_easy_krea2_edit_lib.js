// ==========================================================================
// sf_easy_krea2_edit_lib.js - SFEasyKrea2Edit 逐图 strength 纯逻辑
// ==========================================================================
//
// 界面真源：node.properties[STRENGTH_PROP]（随工作流保存，按槽位编号存储；
// 动态 number widget 只做显示/编辑，serialize:false 避免 widgets_values 位置漂移）。
// 提交时由 sf_easy_krea2_edit.js 的 graphToPrompt 钩子把 stateJson() 注入
// hidden 输入 HIDDEN_INPUT（与 SFPromptStack / SFTrackDataMerge 同模式）。
//
// 无 app 依赖，可拷 .mjs 直测。

export const STRENGTH_PROP = "sfEasyKrea2Strengths";
export const HIDDEN_INPUT = "SFEasyKrea2EditState";
export const IMAGE_RE = /^image(\d+)$/;

export const strengthLabel = (n) => `强度 ${n}`;

// 当前 imageN 槽位编号（升序）
export function imageNumbers(node) {
    const nums = [];
    for (const inp of node?.inputs || []) {
        const m = IMAGE_RE.exec(inp?.name || "");
        if (m) nums.push(parseInt(m[1], 10));
    }
    return nums.sort((a, b) => a - b);
}

export function readStrengths(node) {
    const raw = node?.properties?.[STRENGTH_PROP];
    return raw && typeof raw === "object" ? raw : {};
}

// 槽位 n 的强度；缺失/非法回退 1.0（与后端默认一致）
export function strengthFor(node, n) {
    const v = readStrengths(node)[String(n)];
    return typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : 1.0;
}

export function setStrength(node, n, v) {
    if (!node.properties) node.properties = {};
    const num = typeof v === "number" ? v : Number(v);
    node.properties[STRENGTH_PROP] = { ...readStrengths(node), [String(n)]: Number.isFinite(num) ? num : 1.0 };
}

// hidden 状态 JSON：只含当前槽位，按编号键控
export function stateJson(node) {
    const strengths = {};
    for (const n of imageNumbers(node)) strengths[String(n)] = strengthFor(node, n);
    return JSON.stringify({ strengths });
}

// 让动态 strength widget 与当前 imageN 槽位一致（增删 + 回填 properties 值）。
// 只处理带 _sfStrengthN 标记的 widget；其他 widget 不动。
export function syncStrengthWidgets(node, onChange) {
    if (!node || !Array.isArray(node.widgets)) return;
    const wanted = new Set(imageNumbers(node).map(String));

    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const w = node.widgets[i];
        if (w?._sfStrengthN != null && !wanted.has(String(w._sfStrengthN))) {
            node.widgets.splice(i, 1);
            w.onRemove?.();
        }
    }

    const existing = new Set(
        node.widgets.filter((w) => w?._sfStrengthN != null).map((w) => String(w._sfStrengthN)));
    if (typeof node.addWidget === "function") {
        for (const n of imageNumbers(node)) {
            if (existing.has(String(n))) continue;
            const w = node.addWidget(
                "number", strengthLabel(n), strengthFor(node, n),
                (v) => { setStrength(node, n, v); onChange?.(n, v); },
                { serialize: false, min: 0, max: 10, step: 0.01 },
            );
            if (w) w._sfStrengthN = n;
        }
    }

    for (const w of node.widgets) {
        if (w?._sfStrengthN == null) continue;
        const v = strengthFor(node, w._sfStrengthN);
        if (w.value !== v) w.value = v;
    }
}
