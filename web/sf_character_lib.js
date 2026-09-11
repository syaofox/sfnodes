// SFCharacterSelect 纯逻辑（无 app/DOM 依赖，可拷 .mjs 直测）：
// 多选状态序列化/解析/收敛、语言化标签、搜索过滤 + 选中置顶。
// 与后端 nodes/text/character.py 的 SFCharacterState 契约一致：
// {"role": 名, "shots": [label]}；旧 ["名"] 数组态迁移为首图。
// 前端消费的是归一化角色列表（images: [{label, prompt?, url}]），
// 旧 face/half/full 形状由后端归一，lib 只认新形状。自包含不跨文件 import。

export const STATE_WIDGET = "SFCharacterState"; // 隐藏真源 widget 名（Python hidden 声明）
export const PROMPT_WIDGET = "SFCharacterPrompt"; // 手改草稿 widget 名（Python hidden 声明）
export const DOM_WIDGET = "sf_character_panel"; // DOM widget 名（纯交互，不承担值传输）
export const CHARACTERS_API = "/api/sfnodes/characters"; // 角色列表路由
export const LIB_PREFIX = "character_"; // 角色库名前缀（与风格/服装库隔离）

export function parseState(state) {
  let v = state;
  if (typeof v === "string") {
    try {
      v = v ? JSON.parse(v) : null;
    } catch (e) {
      return { role: "", shots: [], _legacy: false };
    }
  }
  if (v && typeof v === "object" && !Array.isArray(v)) {
    const role = v.role ? String(v.role) : "";
    const shots = Array.isArray(v.shots) ? v.shots.map(String).filter(Boolean) : [];
    return { role, shots, _legacy: false };
  }
  const names = parseNames(v);
  if (names.length) return { role: names[0], shots: [], _legacy: true };
  return { role: "", shots: [], _legacy: false };
}

function parseNames(state) {
  if (Array.isArray(state)) return state.map(String).filter(Boolean);
  if (typeof state !== "string") return [];
  try {
    const v = JSON.parse(state);
    if (Array.isArray(v)) return v.map(String).filter(Boolean);
    return [];
  } catch (e) {
    return [];
  }
}

export function serializeSelection(role, shots) {
  const seen = [];
  for (const s of shots || []) {
    const n = String(s);
    if (n && seen.indexOf(n) === -1) seen.push(n);
  }
  return JSON.stringify({ role: String(role || ""), shots: seen });
}

// 角色条目 → 标准 images 列表 [{label, url, prompt}]（保库内顺序）
export function entryImages(entry) {
  const imgs = entry && entry.images;
  if (!Array.isArray(imgs)) return [];
  const out = [];
  for (const item of imgs) {
    if (!item || !item.label || !item.url) continue;
    out.push({ label: String(item.label), url: String(item.url), prompt: item.prompt != null ? String(item.prompt) : "" });
  }
  return out;
}

export function entryOf(roles, name) {
  if (!name || !Array.isArray(roles)) return null;
  return roles.find((s) => s && s.name === name) || null;
}

export function rolePromptOf(entry) {
  if (!entry) return "";
  return entry.prompt != null ? String(entry.prompt) : "";
}

// 选择收敛：角色有效则保留并取交集分镜，否则回落首个角色首图。
// 空 shots（含旧数组态）一律收敛为该角色首图；空库回落空选择。
export function coerceSelection(roles, state) {
  const parsed = parseState(state);
  const list = Array.isArray(roles) ? roles : [];
  const names = list.filter((d) => d && d.name).map((d) => d.name);
  const roleValid = parsed.role && names.indexOf(parsed.role) !== -1;
  const role = roleValid ? parsed.role : (names.length ? names[0] : "");
  const entry = entryOf(list, role);
  const valid = entryImages(entry).map((i) => i.label);
  let shots;
  if (!roleValid || parsed.shots.length === 0) {
    shots = valid.slice(0, 1);
  } else {
    const wanted = {};
    for (const s of parsed.shots) wanted[s] = true;
    shots = valid.filter((l) => wanted[l]);
    if (!shots.length) shots = valid.slice(0, 1);
  }
  return { role, shots };
}

// role_prompt 路显示值：草稿非空优先，否则角色原词（后端 resolve_role_prompt 同语义）
export function displayRolePrompt(roles, state, draft) {
  if (draft != null && String(draft) !== "") return String(draft);
  const sel = coerceSelection(roles, state);
  return rolePromptOf(entryOf(Array.isArray(roles) ? roles : [], sel.role));
}

// 显示用提示词：草稿非空整体覆盖，否则选中分镜 prompt 逗号拼接
// （单项空回落角色级 prompt，保库内顺序）
export function displayPrompt(roles, state, draft) {
  if (draft != null && String(draft) !== "") return String(draft);
  const sel = coerceSelection(roles, state);
  const entry = entryOf(Array.isArray(roles) ? roles : [], sel.role);
  if (!entry) return "";
  const fallback = rolePromptOf(entry);
  const byLabel = {};
  for (const i of entryImages(entry)) byLabel[i.label] = i;
  const parts = [];
  for (const label of sel.shots) {
    const item = byLabel[label];
    if (!item) continue;
    const text = item.prompt || fallback;
    if (text) parts.push(text);
  }
  return parts.join(", ");
}

// 中文环境且条目提供 name_cn 时显示中文名（标签），原始 name 恒为值键
export function resolveLabel(name, nameCn, isZh) {
  return isZh && nameCn ? nameCn : name;
}

// 搜索过滤 + 选中置顶（稳定排序；选中的项永不隐藏）。
// 搜索范围 = 角色名 + label + 角色 prompt + 各图 label/prompt。
// 每项携带原条目引用 raw。
export function filterAndSort(roles, query, selectedRole, isZh) {
  const sel = String(selectedRole || "");
  const q = String(query || "").trim().toLowerCase();
  const items = (roles || []).map((s) => {
    const name = String((s && s.name) || "");
    const label = resolveLabel(name, s && s.name_cn, isZh);
    const picked = !!sel && name === sel;
    const hay = [name, label, (s && s.prompt) || ""];
    for (const i of entryImages(s)) {
      hay.push(i.label);
      hay.push(i.prompt);
    }
    const hit = hay.some((h) => String(h).toLowerCase().indexOf(q) !== -1);
    return { name, label, selected: picked, hidden: !!q && !picked && !hit, raw: s };
  });
  return items.sort((a, b) => (b.selected ? 1 : 0) - (a.selected ? 1 : 0));
}

// 远程直链判定：本地路由路径（/api/sfnodes/...）需经 sfApiUrl 包裝基址
export function isRemoteThumb(url) {
  return typeof url === "string" && (url.startsWith("http://") || url.startsWith("https://"));
}
