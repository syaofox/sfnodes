// SFIDClothingSelector 纯逻辑（无 app/DOM 依赖，可拷 .mjs 直测）：
// 单选状态序列化/解析、语言化标签、缩略图取值、搜索过滤 + 选中置顶。
// 与后端 nodes/text/id_clothing.py 的 SFIDClothingState 契约一致（JSON 数组，
// 只取首个有效名；与 SFStylesState 同形，便于心智复用，但本模块自包含，
// 不 import sf_styles_selector_lib.js，保证单文件可拷测）。

export const STATE_WIDGET = "SFIDClothingState"; // 隐藏真源 widget 名（Python hidden 声明）
export const PROMPT_WIDGET = "SFIDClothingPrompt"; // 手改草稿 widget 名（Python hidden 声明）
export const DOM_WIDGET = "sf_id_clothing_panel"; // DOM widget 名（纯交互，不承担值传输）
export const STYLES_API = "/api/sfnodes/styles"; // 列表路由复用 styles（id_ 前缀子集）
export const LIB_PREFIX = "id_"; // 服装库名前缀（与 styles 风格库隔离）

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

// 中文环境且条目提供 name_cn 时显示中文名（标签），原始 name 恒为值键
export function resolveLabel(name, nameCn, isZh) {
  return isZh && nameCn ? nameCn : name;
}

// 缩略图取单值（数据可能为数组，前端只展示一张）
export function thumbnailOf(entry) {
  const t = entry && entry.thumbnail;
  if (Array.isArray(t)) return t.length ? t[0] : "";
  return t || "";
}

export function entryOf(styles, name) {
  if (!name || !Array.isArray(styles)) return null;
  return styles.find((s) => s && s.name === name) || null;
}

// 显示用提示词：草稿非空优先（后端 resolve_prompt 同语义），否则模板原词
export function displayPrompt(styles, name, draft) {
  if (draft != null && String(draft) !== "") return String(draft);
  const e = entryOf(styles, name);
  const p = e && e.prompt;
  return p != null ? String(p) : "";
}

// 搜索过滤 + 选中置顶（稳定排序；选中的项永不隐藏——对齐 styles 行为）。
// selected 传状态字符串或单名均可，内部统一收敛为单名。
// 每项携带原条目引用 raw，供调用方直接取缩略图等扩展字段。
export function filterAndSort(styles, query, selected, isZh) {
  const sel = typeof selected === "string" && selected.trim().startsWith("[")
    ? firstSelected(selected)
    : String(selected || "");
  const q = String(query || "").trim().toLowerCase();
  const items = (styles || []).map((s) => {
    const name = String((s && s.name) || "");
    const label = resolveLabel(name, s && s.name_cn, isZh);
    const picked = !!sel && name === sel;
    // 搜索范围 = name + label + prompt（与 styles 库只搜 name/label 不同：
    // 服装模板的可检索英文描述全在 prompt 里，标题只是短中文名）
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

// hover 预览图 URL：本地路由路径（/api/sfnodes/...）或远程 http 直链
export function isRemoteThumb(url) {
  return typeof url === "string" && (url.startsWith("http://") || url.startsWith("https://"));
}
