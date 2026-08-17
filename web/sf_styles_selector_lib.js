// SFStylesSelector 纯逻辑（无 app/DOM 依赖，可拷 .mjs 直测）：
// 状态序列化/解析、语言化标签、缩略图取值、搜索过滤 + 选中置顶排序。
// 与后端 nodes/text/styles_selector.py 的 SFStylesState 契约一致（JSON 数组）。

export const STATE_WIDGET = "SFStylesState"; // 隐藏真源 widget 名（Python hidden 声明）
export const DOM_WIDGET = "sf_styles_panel"; // DOM widget 名（纯交互，不承担值传输）
export const STYLES_API = "/api/sfnodes/styles";

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

export function serializeState(names) {
  return JSON.stringify([...new Set(names.filter(Boolean))]);
}

// 中文环境且样式提供 name_cn 时显示中文名（标签），原始 name 恒为值键
export function resolveLabel(name, nameCn, isZh) {
  return isZh && nameCn ? nameCn : name;
}

// 缩略图取单值（数据可能为数组，前端只展示一张）
export function thumbnailOf(entry) {
  const t = entry && entry.thumbnail;
  if (Array.isArray(t)) return t.length ? t[0] : "";
  return t || "";
}

// 搜索过滤 + 选中置顶（稳定排序；选中的项永不隐藏——对齐 Easy-Use 原版行为）。
// 每项携带原条目引用 raw，供调用方直接取缩略图等扩展字段。
export function filterAndSort(styles, query, selected, isZh) {
  const q = String(query || "").trim().toLowerCase();
  const sel = new Set(parseState(selected));
  const items = styles.map((s) => {
    const name = String(s.name || "");
    const label = resolveLabel(name, s.name_cn, isZh);
    const picked = sel.has(name);
    const hidden =
      !!q &&
      !picked &&
      name.toLowerCase().indexOf(q) === -1 &&
      label.toLowerCase().indexOf(q) === -1;
    return { name, label, selected: picked, hidden, raw: s };
  });
  return items.sort((a, b) => (b.selected ? 1 : 0) - (a.selected ? 1 : 0));
}

// hover 预览图 URL：本地路由路径（/api/sfnodes/...）或远程 http 直链
export function isRemoteThumb(url) {
  return typeof url === "string" && (url.startsWith("http://") || url.startsWith("https://"));
}
