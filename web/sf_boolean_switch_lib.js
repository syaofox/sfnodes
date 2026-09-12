// ==========================================================================
// sf_boolean_switch_lib.js - SF Boolean Switch 纯逻辑库（复刻孤海布尔开关）
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_boolean_switch.js 使用，也供 tests/ 复制为 .mjs 直接测试。
// 原版 BOOLEAN.js 的魔法数字收敛为 TOGGLE 常量（绘制与命中判定同源，
// 原版两处手写 72/10/6 与 72/10/20 各写一份）；标签归一化与省略号截断
// 抽为纯函数（measure 注入，canvas 相关只剩一处调用）。
// ==========================================================================

// 开关几何（与原版一致）：轨道 72×28，边距 m=10，右偏移 xOff=6；
// 点击区 = 轨道左沿再向左 clickPad=14（原版 mouse 侧写死的 _w-72-10-20）。
export const TOGGLE = { tw: 72, th: 28, m: 10, xOff: 6, clickPad: 14 };

export const DEFAULT_LABEL = "value";

// 标签归一化：去首尾空，空回落默认（原版 finishEdit 逻辑）
export function normalizeLabel(text, fallback = DEFAULT_LABEL) {
  const t = (text || "").trim();
  return t || fallback;
}

// 省略号截断（原版 ellipsis 一致；measure 由调用方注入 ctx.measureText）
export function ellipsisText(text, maxW, measure) {
  if (measure(text).width <= maxW) return text;
  let t = text;
  while (t.length > 1 && measure(t + "…").width > maxW) {
    t = t.slice(0, -1);
  }
  return t + "…";
}

// 开关轨道左沿 x（绘制与命中同源）
export function trackX(width) {
  return width - TOGGLE.tw - TOGGLE.m - TOGGLE.xOff;
}

// 点击是否落在开关区（含左侧 clickPad 容差，原版 toggleStartX 等价）
export function toggleHit(posX, width) {
  return posX > trackX(width) - TOGGLE.clickPad;
}
