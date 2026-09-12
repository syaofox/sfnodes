// ==========================================================================
// sf_universal_slider_lib.js - SF Universal Slider 纯逻辑库（复刻孤海万能滑条）
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_universal_slider.js 使用，也供 tests/ 复制为 .mjs 直接测试。
// 原版 goohai_universal_slider.js 的 pct/clamp/snap/fmt/castVal/calcValue
// 逐行同义迁移 + 设置归一化 normalizeSliderSettings（原版分散在设置弹窗
// bOk.onclick，集中后前后端同测：min/max 对调、step 非法回退、int 档取整）。
// ==========================================================================

export const DEFAULTS = {
  sliderType: "float",
  sliderMin: 0,
  sliderMax: 1,
  sliderStep: 0.01,
  sliderLabel: "value",
  sliderColor: "#e8c547",
};

export function pct(v, mn, mx) {
  const r = mx - mn;
  return r > 0 ? ((v - mn) / r) * 100 : 0;
}

export function clamp(v, mn, mx) {
  return Math.max(mn, Math.min(mx, v));
}

export function snap(v, mn, step) {
  return step > 0 ? Math.round((v - mn) / step) * step + mn : v;
}

export function fmtVal(v, isInt) {
  return isInt ? String(Math.round(v)) : Number(v).toFixed(2);
}

export function castVal(v, isInt) {
  if (isInt) return parseInt(Math.round(v), 10);
  return parseFloat(Number(v).toFixed(2));
}

// 统一值计算：先 snap 再钳制再按类型取整（与原版 calcValue 一致）
export function calcValue(v, mn, mx, step, isInt) {
  v = snap(v, mn, step);
  v = clamp(v, mn, mx);
  return castVal(v, isInt);
}

// 设置归一化（原版 bOk.onclick 逻辑集中）：返回 {type, min, max, step, label}
export function normalizeSliderSettings({ type, min, max, step, label }) {
  let t = type === "int" ? "int" : "float";
  let mn = parseFloat(min);
  let mx = parseFloat(max);
  let st = parseFloat(step);
  let lb = (label || "").trim() || "value";
  if (isNaN(mn)) mn = 0;
  if (isNaN(mx)) mx = 1;
  if (mn > mx) {
    const tmp = mn;
    mn = mx;
    mx = tmp;
  }
  if (isNaN(st) || st <= 0) st = t === "int" ? 1 : 0.01;
  if (t === "int") {
    mn = Math.round(mn);
    mx = Math.round(mx);
    st = Math.max(1, Math.round(st));
  }
  return { type: t, min: mn, max: mx, step: st, label: lb };
}

// 输出槽类型/槽名（静态 value + 前端槽名动态 int/float，见 patterns §41 同款）
// int 档 → INT / float 档 → FLOAT（后端始终 any，直通渲染与连线校验）
export function outputSlotForType(sliderType) {
  const isInt = sliderType === "int";
  return isInt
    ? { type: "INT", name: "int" }
    : { type: "FLOAT", name: "float" };
}
