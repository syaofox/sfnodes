// ==========================================================================
// sf_canvas_size_lib.js - Canvas Size Preset 纯逻辑库
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供 web/canvas_size.js
// 与 tests/ 复制为 .mjs 直接测试。
//
// 自定义分辨率库（后端 sf_utils/canvas_size_presets.py，GET/POST/DELETE
// /api/sfnodes/canvas_size_custom）的归一化与 combo options 合并规则与后端
// _normalize_presets 同口径（前端兜底）。

// 自定义分组标签（与各模型档位头 --1MP-- / --Official-- 同款；部分前端不渲染为
// 原生分组，仅作可读分隔，故自定义项放在标签之后、语义上仍属该分组）。
export const CUSTOM_HEADER = "--Custom--";

const NAME_MAX_LEN = 200;
const DIM_MAX = 32768;

// 分组头判定。
export function isTierHeader(value) {
  return typeof value === "string" && value.startsWith("--") && value.endsWith("--");
}

// 第一个非分组头选项（全为分组头时退首个元素）。
export function firstSelectable(values) {
  if (!Array.isArray(values) || values.length === 0) return undefined;
  return values.find((v) => !isTierHeader(v)) ?? values[0];
}

// 自定义预设名：非空、无路径分隔符/控制字符、无括号（combo 编码 "WxH (name)"
// 依赖首个 '(' / ')'），限长。与后端 canvas_size_presets._valid_name 同口径。
export function validCustomName(name) {
  if (typeof name !== "string") return false;
  const s = name.trim();
  if (!s || s.length > NAME_MAX_LEN) return false;
  if (s.includes("/") || s.includes("\\")) return false;
  if (s.includes("(") || s.includes(")")) return false;
  for (const c of s) {
    if (c.charCodeAt(0) < 32) return false;
  }
  return true;
}

// 像素维：正整数且 ≤ 32768（bool/浮点不算）。
export function validDim(v) {
  return typeof v === "number" && Number.isInteger(v) && v > 0 && v <= DIM_MAX;
}

// normalizeCustomPresets(raw) → [{name, w, h}]
// 接受数组或 {presets:[...]}；过滤非法与重名（保留首个），与后端归一化一致。
export function normalizeCustomPresets(raw) {
  const list = Array.isArray(raw) ? raw : raw && Array.isArray(raw.presets) ? raw.presets : [];
  const out = [];
  const seen = new Set();
  for (const item of list) {
    if (!item || typeof item !== "object") continue;
    const name = String(item.name ?? "").trim();
    if (!validCustomName(name) || seen.has(name)) continue;
    if (!validDim(item.w) || !validDim(item.h)) continue;
    seen.add(name);
    out.push({ name, w: item.w, h: item.h });
  }
  return out;
}

// 自定义条目 -> combo 值（"WxH (name)"，_parse_resolution 可解析出宽高）。
export function customOptionValue(name, w, h) {
  return `${w}x${h} (${name})`;
}

// mergeResolutionValues(officialValues, customPresets) → combo 完整 values
// 常规调用是伪模型场景（officialValues=[]）：`--Custom--` 分组头 + 自定义项。
// 自定义项与官方值重复则跳过（避免 combo 内重复选项）；无自定义项时保持官方表
// 原样（不留悬空分组头）。真实模型不应传 customPresets（自定义库不混入模型列表）。
export function mergeResolutionValues(officialValues, customPresets) {
  const official = Array.isArray(officialValues) ? officialValues.slice() : [];
  const presets = normalizeCustomPresets(customPresets);
  if (presets.length === 0) return official;
  const seen = new Set(official);
  const custom = [];
  for (const p of presets) {
    const value = customOptionValue(p.name, p.w, p.h);
    if (seen.has(value)) continue;
    seen.add(value);
    custom.push(value);
  }
  if (custom.length === 0) return official;
  return [CUSTOM_HEADER, ...custom, ...official];
}
