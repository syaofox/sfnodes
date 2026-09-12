// ==========================================================================
// sf_ignore_groups_lib.js - SF Ignore Groups 纯逻辑库（复刻孤海忽略多组）
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_ignore_groups.js 使用，也供 tests/ 复制为 .mjs 直接测试。
// 原版 "nodes pass.js" 的几何/状态/切换逻辑逐义迁移：LiteGraph 原始对象
// 只在 gBounds/nBounds/getGroupColor 三处接触，其余函数一律操作 plain
// 数据（group:{title,bounds,color}、node:{bounds,mode,disabled}），可直测。
// properties 键原版 guhai_ig_* → sf_ig_*（与布尔/滑条复刻口径一致）。
// ==========================================================================

// LiteGraph 节点模式（原版手写 4/2/0，具名收敛）
export const MODE_ALWAYS = 0; // 正常执行
export const MODE_NEVER = 2; // 禁用（ctrl+m）
export const MODE_BYPASS = 4; // 旁路（ctrl+b）

// 切换模式
export const SWITCH_DEFAULT = "default"; // 默认：多选开关
export const SWITCH_ALWAYS_ONE = "always_one"; // 始终开启 1 个
export const SWITCH_AT_MOST_ONE = "at_most_one"; // 最多开启 1 个

// 状态默认值（properties 缺键回落）
export const STATE_DEFAULTS = {
  filter: "",
  mode: SWITCH_DEFAULT,
  active: null,
  activeSet: null, // default 模式惰性初始化（首次 refresh 按组实态填充）
  nameColor: null,
  disable: false, // false=旁路(true=禁用)
  sortOrder: "position",
  colorFilter: "none",
  uiScale: 1.0,
};

// properties 键 ↔ 状态字段映射
const KEYMAP = [
  ["sf_ig_filter", "filter"],
  ["sf_ig_mode", "mode"],
  ["sf_ig_active", "active"],
  ["sf_ig_active_set", "activeSet"],
  ["sf_ig_name_color", "nameColor"],
  ["sf_ig_disable", "disable"],
  ["sf_ig_sort_order", "sortOrder"],
  ["sf_ig_color_filter", "colorFilter"],
  ["sf_ig_ui_scale", "uiScale"],
];

export function readState(properties) {
  const p = properties || {};
  const st = { ...STATE_DEFAULTS };
  for (const [key, field] of KEYMAP) {
    if (p[key] != null) st[field] = p[key];
  }
  if (typeof st.uiScale !== "number") {
    const v = Number(st.uiScale);
    st.uiScale = v > 0 ? v : 1.0;
  }
  st.disable = !!st.disable;
  return st;
}

export function writeState(properties, state) {
  const p = properties || {};
  for (const [key, field] of KEYMAP) {
    p[key] = state[field];
  }
  return p;
}

// ── 原始对象 → plain 数据 ──
export function groupBounds(g) {
  if (g._bounding) return [...g._bounding];
  if (g.bounding) return [...g.bounding];
  const p = g.pos || [0, 0];
  const s = g.size || [0, 0];
  return [p[0], p[1], s[0], s[1]];
}

export function nodeBounds(n, titleH = 30) {
  const p = n.pos || [0, 0];
  const s = n.size || [100, 60];
  if (n.collapsed || n._collapsed) {
    let collapsedW = n._collapsed_width;
    if (!collapsedW || collapsedW <= 0) {
      const title = (typeof n.getTitle === "function" ? n.getTitle() : n.title) || "";
      collapsedW = Math.max(80, title.length * 7 + 50);
    }
    return [p[0], p[1], collapsedW, titleH];
  }
  return [p[0], p[1], s[0], s[1]];
}

// 组颜色归一（number→#rrggbb，#rgb→#rrggbb，其余 неизвест→""）
export function normalizeColor(color) {
  if (typeof color === "number") {
    return "#" + color.toString(16).padStart(6, "0");
  }
  if (typeof color === "string" && color.startsWith("#") && (color.length === 4 || color.length === 7)) {
    if (color.length === 4) {
      return "#" + color[1] + color[1] + color[2] + color[2] + color[3] + color[3];
    }
    return color;
  }
  return "";
}

// ── 几何 ──
export function hit(a, b) {
  return !(
    a[0] + a[2] <= b[0] ||
    a[0] >= b[0] + b[2] ||
    a[1] + a[3] <= b[1] ||
    a[1] >= b[1] + b[3]
  );
}

export function inside(inner, outer) {
  return (
    inner[0] >= outer[0] &&
    inner[1] >= outer[1] &&
    inner[0] + inner[2] <= outer[0] + outer[2] &&
    inner[1] + inner[3] <= outer[1] + outer[3]
  );
}

// 组内节点：命中本组或被本组完全包含的子组矩形的节点
export function collectNodes(group, allGroups, nodes) {
  const rects = [group.bounds];
  for (const ag of allGroups) {
    if (ag !== group && inside(ag.bounds, group.bounds)) {
      rects.push(ag.bounds);
    }
  }
  return nodes.filter((n) => rects.some((r) => hit(n.bounds, r)));
}

// 被 parent 完全包含的嵌套组（递归，含多层）
export function nestedGroups(parent, allGroups) {
  const result = [];
  const visited = new Set([parent]);
  (function findNested(p) {
    for (const ag of allGroups) {
      if (!visited.has(ag) && inside(ag.bounds, p.bounds)) {
        visited.add(ag);
        result.push(ag);
        findNested(ag);
      }
    }
  })(parent);
  return result;
}

// ── 状态 ──
export function isNodeActive(n) {
  return n.mode !== MODE_BYPASS && n.mode !== MODE_NEVER && !n.disabled;
}

// 组态：全开 true、全关 false、混合 null；空组按 true（原版同款）
export function groupState(group, allGroups, nodes) {
  const members = collectNodes(group, allGroups, nodes);
  if (members.length === 0) return true;
  if (members.every(isNodeActive)) return true;
  if (members.every((n) => !isNodeActive(n))) return false;
  return null;
}

export function stateSig(entries) {
  return entries
    .map((e) => e.title + ":" + (e.state === true ? "1" : e.state === false ? "0" : "m"))
    .join("\x00");
}

// ── 列表：空组过滤 + 关键词 + 颜色 + 排序 ──
export function filterSortGroups(groups, allGroups, nodes, { filter, colorFilter, sortOrder }) {
  let list = groups.filter((g) => collectNodes(g, allGroups, nodes).length > 0);
  const kw = (filter || "").trim().toLowerCase();
  if (kw) {
    list = list.filter((g) => g.title.toLowerCase().includes(kw));
  }
  if (colorFilter && colorFilter !== "none") {
    list = list.filter((g) => {
      if (colorFilter === "__transparent__") return !g.color;
      return !!g.color && g.color.toLowerCase() === colorFilter.toLowerCase();
    });
  }
  list = list.slice();
  if (sortOrder === "position") {
    list.sort((a, b) => {
      if (a.bounds[1] !== b.bounds[1]) return a.bounds[1] - b.bounds[1];
      return a.bounds[0] - b.bounds[0];
    });
  } else {
    list.sort((a, b) => a.title.localeCompare(b.title, undefined, { sensitivity: "base" }));
  }
  return list;
}

// ── 切换纯函数：输入 {mode, active, activeSet} + 点选 title (+嵌套 titles），
// 返回新的 {active, activeSet}（原版 handleToggle 一致，调用方负责 save/refresh）──
export function toggleTransition(switchMode, active, activeSet, title, nestedTitles = []) {
  if (switchMode === SWITCH_DEFAULT) {
    let set = Array.isArray(activeSet) ? activeSet.slice() : [];
    if (set.includes(title)) {
      const off = new Set([title, ...nestedTitles]);
      set = set.filter((t) => !off.has(t));
    } else {
      if (!set.includes(title)) set.push(title);
      for (const t of nestedTitles) {
        if (!set.includes(t)) set.push(t);
      }
    }
    return { active, activeSet: set };
  }
  if (switchMode === SWITCH_ALWAYS_ONE) {
    if (active === title) return { active, activeSet };
    return { active: title, activeSet };
  }
  // at_most_one：点已选项则全关，否则独占
  if (active === title) return { active: null, activeSet };
  return { active: title, activeSet };
}
