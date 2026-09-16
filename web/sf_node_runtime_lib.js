// ==========================================================================
// sf_node_runtime_lib.js — 节点运行时间显示纯逻辑（无 app/DOM 依赖）
// 可拷 .mjs 用 Node 直测（test_node_runtime_lib.mjs）。
//
// 背景：复刻 ComfyUI-Easy-Use 的 Comfy.EasyUse.TimeTaken——监听 ComfyUI
// `/scripts/api.js` 的 execution_start / executing 事件，用相邻 executing
// 事件的间隔估算每个节点耗时，作为节点 badge 显示（见 platform.md §2.18）。
// 本模块只收敛可测的纯计算，事件/画布交互留在主模块。
// ==========================================================================

// 运行时 badge 标记：仅用于识别/替换本特性自己的 badge，不影响其它扩展
// （如官方 Comfy.NodeBadge、API 价格）推送的 badge。
export const BADGE_FLAG = "_sfRuntimeBadge";

// 秒 → "x.xxx s" 文案（Easy-Use 同款 toFixed(3)）；非法值按 0 处理。
export function formatDuration(seconds) {
  const s = Number(seconds);
  return `${(Number.isFinite(s) ? s : 0).toFixed(3)}s`;
}

// 累加耗时：上一段秒数 + 本次毫秒增量（支持节点在循环/子图中多次执行）。
export function accumulateSeconds(prevSeconds, elapsedMs) {
  const prev = Number(prevSeconds);
  const ms = Number(elapsedMs);
  return (Number.isFinite(prev) ? prev : 0) + (Number.isFinite(ms) ? ms : 0) / 1000;
}

// 从 executing 事件 detail 解析节点 id。
// 新版前端 `/scripts/api.js` 对 executing 事件是
// `dispatchCustomEvent("executing", data.display_node || data.node)`，
// 即 **event.detail 直接就是节点 id**（数字/字符串），整轮结束为 null；
// 旧版/其它来源可能传 `{node, display_node}` 对象，这里两种形态都兼容。
// 对象形态优先 display_node（子图内节点归到可见的子图节点），回退 node。
export function resolveNodeId(detail) {
  if (detail == null) return null;
  if (typeof detail === "object") {
    const id = detail.display_node ?? detail.node;
    return id == null ? null : id;
  }
  return detail;
}

export function isRuntimeBadge(badge) {
  return !!badge && badge[BADGE_FLAG] === true;
}

export function markRuntimeBadge(badge) {
  if (badge) {
    try {
      badge[BADGE_FLAG] = true;
    } catch {
      /* ignore */
    }
  }
  return badge;
}
