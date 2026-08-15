// ==========================================================================
// sf_popup.js — 浮动弹层公共三件套（外部点击 / Esc / 滚轮三关闭 + viewport 钳位）
// ==========================================================================
//
// 项目里 13+ 个浮动弹层（下拉列表 / 信息面板 / 编辑器 / 右键菜单）各自重复
// 实现"外部点击关闭 / Esc 分层关闭 / 滚轮关闭 / 画布 transform 缩放定位"，
// 踩过的坑记录在 doc/experience.md（§15.6 弹层定位与 canvas 缩放、§19 面板风
// 确认框必须豁免宿主面板的 document 捕获监听——确认框挂在 body、不在面板 DOM
// 内，不豁免则其事件会穿透到面板监听，Esc 连关面板、外部点击误关）。
// 本模块收敛这三件套，**新弹层优先使用**（存量弹层按需迁移）。
//
// 无 app/ComfyUI 依赖（纯 DOM），可拷为 .mjs 冒烟测试。canvas scale 由调用
// 方从参数传入（各节点从不同来源拿 scale，本模块不猜）。
//
// ==========================================================================

// ── 三关闭（外部 pointerdown / Esc / wheel）─────────────────────────────
// 挂到 overlay 元素上，返回清理函数（幂等，可安全多次调用）。外部点击与滚轮
// 用 capture 阶段 document 监听；Esc 用 keydown。`exempt(e)` 返回 true 时跳过
// 关闭（用于豁免宿主面板自身的 document 捕获监听，见 experience.md §19）。
// `onClose` 在真正关闭时调用一次（三路共用，监听移除后天然去重）。
export function attachPopupDismiss(overlay, { onClose, exempt } = {}) {
  if (!overlay || typeof overlay.addEventListener !== "function") return () => {};
  const isInside = (e) => overlay.contains(e.target);
  const onPointer = (e) => {
    if (isInside(e)) return;
    if (exempt && exempt(e)) return;
    detach();
    if (onClose) onClose();
  };
  const onKey = (e) => {
    if (e.key !== "Escape") return;
    if (exempt && exempt(e)) return;
    detach();
    if (onClose) onClose();
  };
  const detach = () => {
    document.removeEventListener("pointerdown", onPointer, true);
    document.removeEventListener("mousedown", onPointer, true); // 旧浏览器兜底
    document.removeEventListener("keydown", onKey, true);
    document.removeEventListener("wheel", onPointer, true);
  };
  document.addEventListener("pointerdown", onPointer, true);
  document.addEventListener("mousedown", onPointer, true);
  document.addEventListener("keydown", onKey, true);
  document.addEventListener("wheel", onPointer, true);
  return detach;
}

// ── viewport 钳位 ──────────────────────────────────────────────────────
// 把已定位（position:fixed）的弹层元素钳回视口内，四周留 margin。`scale` 传入
// 画布缩放系数时边距按比例折算（ComfyUI canvas 缩放下 position:fixed 弹层
// 的 root font-size 已按 scale 缩放，见 experience.md §15.6）。返回修正后的
// {left, top}；未越界时不动元素、返回当前值（可安全每帧调用）。
export function clampToViewport(el, { margin = 8, scale = 1 } = {}) {
  if (!el) return null;
  const rect = el.getBoundingClientRect();
  const m = margin * scale;
  let left = rect.left;
  let top = rect.top;
  if (left < m) left = m;
  if (top < m) top = m;
  if (left + rect.width > window.innerWidth - m) left = Math.max(m, window.innerWidth - rect.width - m);
  if (top + rect.height > window.innerHeight - m) top = Math.max(m, window.innerHeight - rect.height - m);
  if (left !== rect.left) el.style.left = `${left}px`;
  if (top !== rect.top) el.style.top = `${top}px`;
  return { left, top };
}
