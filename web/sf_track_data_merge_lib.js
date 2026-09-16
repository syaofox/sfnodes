// ==========================================================================
// sf_track_data_merge_lib.js - SFTrackDataMerge 槽模式纯逻辑（无 DOM/app 依赖）
// ==========================================================================
//
// 槽模式状态约定：`{ "track_1": "sub", "track_2": "add", ... }`（键=输入槽名）。
// 值为 "add"（并集塌单身份）或 "sub"（逐对象相减，默认）。可拷 .mjs 直测。
// ==========================================================================

export const MODE_ADD = "add";
export const MODE_SUB = "sub";
export const DEFAULT_MODE = MODE_SUB;
export const SLOT_PREFIX = "track_";

export function normalizeMode(mode) {
    return mode === MODE_ADD ? MODE_ADD : MODE_SUB;
}

// 解析状态（对象或 JSON 字符串），非法输入回退空对象；丢弃未知模式值。
export function parseModes(raw) {
    let data = raw;
    if (typeof raw === "string") {
        try {
            data = JSON.parse(raw);
        } catch (e) {
            return {};
        }
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) return {};
    const out = {};
    for (const key of Object.keys(data)) out[key] = normalizeMode(data[key]);
    return out;
}

export function getMode(modes, name) {
    return normalizeMode(modes && modes[name]);
}

export function setMode(modes, name, mode) {
    return Object.assign({}, modes || {}, { [name]: normalizeMode(mode) });
}

export function toggleMode(mode) {
    return normalizeMode(mode) === MODE_ADD ? MODE_SUB : MODE_ADD;
}

export function serializeModes(modes) {
    return JSON.stringify(parseModes(modes));
}

// 输入槽数组中匹配前缀 + 数字后缀的槽名（保持原顺序）。
// 必须排除基础槽 `track_data`（它同样以 `track_` 开头）。
export function collectSlotNames(inputs, prefix = SLOT_PREFIX) {
    const names = [];
    for (const slot of inputs || []) {
        if (!slot || typeof slot.name !== "string" || !slot.name.startsWith(prefix)) continue;
        if (!/^\d+$/.test(slot.name.slice(prefix.length))) continue;
        names.push(slot.name);
    }
    return names;
}
