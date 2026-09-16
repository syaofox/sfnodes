// ==========================================================================
// sf_node_color_lib.js — 任意节点颜色纯逻辑（无 app/DOM 依赖）
// 可拷 .mjs 用 Node 直测（test_node_color_lib.mjs）。
//
// 背景：ComfyUI 前端 LGraphCanvas.node_colors 调色板仅 9 个预设，
// 节点右键 Colors 子菜单无任意色入口；节点自身按 node.color（标题）/
// node.bgcolor（节点体）渲染，二者在序列化白名单内（见 platform.md §2.17），
// 故直接赋任意 hex 即可生效并随工作流保存。
// ==========================================================================

// 最近颜色条容量（localStorage 记忆，机器私有）
export const RECENT_LIMIT = 8;

// 规范化任意输入为小写 #rrggbb；非法返回 null。
// 接受 "#RGB" / "#RRGGBB" / "RGB" / "RRGGBB"（大小写不敏感，忽略首尾空白）。
export function normalizeHexColor(value) {
  if (typeof value !== "string") return null;
  let s = value.trim().toLowerCase();
  if (!s) return null;
  if (s[0] === "#") s = s.slice(1);
  if (/^[0-9a-f]{3}$/.test(s)) {
    s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
  }
  if (!/^[0-9a-f]{6}$/.test(s)) return null;
  return "#" + s;
}

// FIFO 去重最近色列表（最新的在最前，超上限截断）。非法色不入列表。
export function pushRecent(list, hex, max = RECENT_LIMIT) {
  const out = Array.isArray(list) ? list.slice() : [];
  const n = normalizeHexColor(hex);
  if (!n) return out;
  const filtered = out.filter((c) => normalizeHexColor(c) !== n);
  filtered.unshift(n);
  return filtered.slice(0, Math.max(0, max | 0));
}

// 批量把节点标题/节点体设为同一任意色，返回成功设置数。
// 直接赋值 node.color/bgcolor（LGraphNode.setColorOption 亦接受任意值，
// 但直接赋值对 FakeNode / 各前端版本最稳）。
export function applyNodeColor(nodes, hex) {
  const n = normalizeHexColor(hex);
  if (!n) return 0;
  let count = 0;
  for (const node of nodes || []) {
    if (!node) continue;
    try {
      node.color = n;
      node.bgcolor = n;
      count++;
    } catch {
      /* 忽略单个节点失败 */
    }
  }
  return count;
}

// 清除自定义色，回落主题默认（等价原生「No Color」：setColorOption(null)）。
export function resetNodeColor(nodes) {
  let count = 0;
  for (const node of nodes || []) {
    if (!node) continue;
    try {
      if (typeof node.setColorOption === "function") node.setColorOption(null);
      else {
        node.color = undefined;
        node.bgcolor = undefined;
      }
      count++;
    } catch {
      /* 忽略单个节点失败 */
    }
  }
  return count;
}

// 取选中节点当前色（优先节点体），无有效色返回 null。
export function readNodeColor(nodes) {
  for (const node of nodes || []) {
    if (!node) continue;
    const c = normalizeHexColor(node.bgcolor) || normalizeHexColor(node.color);
    if (c) return c;
  }
  return null;
}
