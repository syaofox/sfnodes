// SF LoRA Preset - 过滤纯逻辑（无 app/DOM 依赖，可拷 .mjs 单测）
// 搜预设名 + 关联 LoRA 文件名（大小写不敏感子串），高亮命中
export function filterPresets(presets, q) {
    const s = (q || "").trim().toLowerCase();
    if (!s) return presets;
    const out = {};
    for (const [name, data] of Object.entries(presets || {})) {
        const hay = [name, ...((data && Array.isArray(data.loras) ? data.loras : []).map((l) => (l && typeof l.lora === "string" ? l.lora : "")))].join("\n").toLowerCase();
        if (hay.includes(s)) out[name] = data;
    }
    return out;
}

function escapeHtml(s) {
    return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

export function highlight(text, q) {
    const t = String(text || "");
    const s = (q || "").trim();
    if (!s) return escapeHtml(t);
    const lowT = t.toLowerCase();
    const lowS = s.toLowerCase();
    const idx = lowT.indexOf(lowS);
    if (idx < 0) return escapeHtml(t);
    return escapeHtml(t.slice(0, idx)) + "<mark>" + escapeHtml(t.slice(idx, idx + s.length)) + "</mark>" + escapeHtml(t.slice(idx + s.length));
}
