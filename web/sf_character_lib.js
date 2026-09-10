// SFCharacterSelect 纯逻辑（无 app/DOM 依赖，可拷 .mjs 直测）：
// 单选状态序列化/解析、语言化标签、分镜取值、搜索过滤 + 选中置顶。
// 与后端 nodes/text/character.py 的 SFCharacterState 契约一致（JSON 数组，
// 只取首个有效名）。自包含不跨文件 import，保证单文件可拷测。

export const STATE_WIDGET = "SFCharacterState"; // 隐藏真源 widget 名（Python hidden 声明）
export const PROMPT_WIDGET = "SFCharacterPrompt"; // 手改草稿 widget 名（Python hidden 声明）
export const DOM_WIDGET = "sf_character_panel"; // DOM widget 名（纯交互，不承担值传输）
export const CHARACTERS_API = "/api/sfnodes/characters"; // 角色列表路由
export const LIB_PREFIX = "character_"; // 角色库名前缀（与风格/服装库隔离）
export const SHOTS = ["face", "half", "full"]; // 三分镜：脸部特写/半身像/全身像
export const SHOT_LABELS = { face: "脸部特写", half: "半身像", full: "全身像" };

export function parseState(json) {
  if (Array.isArray(json)) return json.map(String).filter(Boolean);
  if (typeof json !== "string") return [];
  try {
    const v = JSON.parse(json);
    return Array.isArray(v) ? v.map(String).filter(Boolean) : [];
  } catch (e) {
    return [];
  }
}

export function serializeSingle(name) {
  const n = String(name || "");
  return JSON.stringify(n ? [n] : []);
}

export function firstSelected(state) {
  const names = parseState(state);
  return names.length ? names[0] : "";
}

// 选择收敛：当前选择有效则保留，否则回落首个角色（空库回落 ""）。
// 供库加载完成后调用——工作流恢复的有效选择不受影响，仅"空/失效"时自动首选。
export function coerceSelection(roles, state) {
  const names = (roles || []).map((s) => (s && s.name ? String(s.name) : "")).filter(Boolean);
  const cur = firstSelected(state);
  if (cur && names.indexOf(cur) !== -1) return cur;
  return names.length ? names[0] : "";
}

// 中文环境且条目提供 name_cn 时显示中文名（标签），原始 name 恒为值键
export function resolveLabel(name, nameCn, isZh) {
  return isZh && nameCn ? nameCn : name;
}

// 某分镜 URL 取单值（数据可能为数组，前端只展示一张）
export function shotUrl(entry, shot) {
  const t = entry && entry[shot];
  if (Array.isArray(t)) return t.length ? t[0] : "";
  return t || "";
}

export function entryOf(roles, name) {
  if (!name || !Array.isArray(roles)) return null;
  return roles.find((s) => s && s.name === name) || null;
}

// 显示用提示词：草稿非空优先（后端 resolve_prompt 同语义），否则角色原词
export function displayPrompt(roles, name, draft) {
  if (draft != null && String(draft) !== "") return String(draft);
  const e = entryOf(roles, name);
  const p = e && e.prompt;
  return p != null ? String(p) : "";
}

// 搜索过滤 + 选中置顶（稳定排序；选中的项永不隐藏）。
// 搜索范围 = name + label + prompt（角色描述在 prompt 里）。
// selected 传状态字符串或单名均可，内部统一收敛为单名。
// 每项携带原条目引用 raw，供调用方直接取分镜 URL。
export function filterAndSort(roles, query, selected, isZh) {
  const sel = typeof selected === "string" && selected.trim().startsWith("[")
    ? firstSelected(selected)
    : String(selected || "");
  const q = String(query || "").trim().toLowerCase();
  const items = (roles || []).map((s) => {
    const name = String((s && s.name) || "");
    const label = resolveLabel(name, s && s.name_cn, isZh);
    const picked = !!sel && name === sel;
    const promptLower = String((s && s.prompt) || "").toLowerCase();
    const hidden =
      !!q &&
      !picked &&
      name.toLowerCase().indexOf(q) === -1 &&
      label.toLowerCase().indexOf(q) === -1 &&
      promptLower.indexOf(q) === -1;
    return { name, label, selected: picked, hidden, raw: s };
  });
  return items.sort((a, b) => (b.selected ? 1 : 0) - (a.selected ? 1 : 0));
}

// 远程直链判定：本地路由路径（/api/sfnodes/...）需经 sfApiUrl 包裝基址
export function isRemoteThumb(url) {
  return typeof url === "string" && (url.startsWith("http://") || url.startsWith("https://"));
}
