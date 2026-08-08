// ============================================================
// SF Inpaint Crop — 几何（JS 镜像 sf_utils/inpaint_helpers.py 的 compute_region）
// 保持 computeRegion() 与 Python compute_region 1:1，编辑器实时预览的裁剪框
// 与节点实际输出一致。唯一允许的漂移是亚像素矩形放置（Python 用 banker's
// rounding，JS 原本 round-half-up）——预览最多差 1px，输出尺寸永远一致。
// ============================================================

export const GEO_DEFAULTS = {
  size_mode: "keep", target: 1024, target_w: 1024, target_h: 1024,
  multiple: 8, context_px: 24, context_pct: 10, mask_grow: 4, mask_blur: 4,
  blend: 16, min_size: 256, max_size: 2048, allow_upscale: true,
};

// round half to EVEN，对齐 Python 内置 round()（banker's rounding，_round_mult
// 用）。Math.round 是 half-UP，不加这个编辑器裁剪框尺寸在精确 .5 边界时会
// 与真实输出差一个 multiple 步长（如 1056/64 = 16.5 -> Python 16 -> 1024，
// half-up 17 -> 1088）。只有精确 .5 的情况不同，其余值舍入完全一致。
const roundHalfEven = (x) => {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
};
const roundMult = (v, m) => {
  m = Math.max(1, Math.round(m));
  return Math.max(m, roundHalfEven(v / m) * m);
};
const clampi = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(v)));

export function computeRegion(bbox, W, H, params) {
  const p = { ...GEO_DEFAULTS, ...(params || {}) };
  W = Math.round(W); H = Math.round(H);
  let x0, y0, x1, y1;
  if (!bbox) { x0 = 0; y0 = 0; x1 = W; y1 = H; } else { [x0, y0, x1, y1] = bbox; }
  const bw = Math.max(1, x1 - x0), bh = Math.max(1, y1 - y0);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;

  // 上下文扩张取 max(context_px, blend)，让接缝羽化有空间在裁剪图内部衰减
  // 到 0（Option B——compute_region 的镜像；大 softness 会扩张裁剪图）。
  const ctx = Math.max(p.context_px, p.blend || 0);
  let rw = bw + 2 * ctx + bw * p.context_pct / 100;
  let rh = bh + 2 * ctx + bh * p.context_pct / 100;

  const mult = p.multiple, mode = p.size_mode;

  // force 模式：先把期望区域扩到目标宽高比（输出为下方设定的固定目标尺寸）。
  let tw = 0, th = 0;
  if (mode === "force") {
    tw = Math.max(mult, roundMult(p.target_w, mult));
    th = Math.max(mult, roundMult(p.target_h, mult));
    const ta = tw / th;
    if (rw / rh < ta) rw = rh * ta; else rh = rw / ta;
  }

  // 在图像内放置并夹紧源区域
  let rw_i = Math.max(1, Math.min(Math.round(rw), W));
  let rh_i = Math.max(1, Math.min(Math.round(rh), H));
  if (mode === "force") {
    // 图像边界夹紧后保持源宽高比 == 目标宽高比（镜像）。
    const aspect = tw / th;
    if (rw_i > rh_i * aspect) rw_i = Math.max(1, Math.round(rh_i * aspect));
    else rh_i = Math.max(1, Math.round(rw_i / aspect));
  }
  const rx = clampi(cx - rw_i / 2, 0, W - rw_i);
  const ry = clampi(cy - rh_i / 2, 0, H - rh_i);

  // 输出尺寸由**夹紧后的**源矩形（rw_i, rh_i）推导，绝不用未夹紧的区域——
  // 图像边缘裁掉区域时裁剪图也不会被拉伸（compute_region 的镜像）。
  let out_w, out_h;
  if (mode === "force") {
    out_w = tw; out_h = th;
  } else if (mode === "free") {
    // 保持裁剪矩形自身尺寸；长边超过 max_size 时**两轴乘同一系数**缩放以
    // 保持宽高比（compute_region 的镜像）。
    let ow = rw_i, oh = rh_i;
    const big = Math.max(ow, oh);
    if (big > p.max_size) { const k = p.max_size / big; ow *= k; oh *= k; }
    out_w = roundMult(ow, mult);
    out_h = roundMult(oh, mult);
  } else {
    const long = Math.max(rw_i, rh_i);
    let s = long > 0 ? p.target / long : 1;
    if (!p.allow_upscale) s = Math.min(s, 1);
    let ow = rw_i * s, oh = rh_i * s;
    const small = Math.min(ow, oh);
    if (small < p.min_size) { const k = p.min_size / small; ow *= k; oh *= k; }
    const big = Math.max(ow, oh);
    if (big > p.max_size) { const k = p.max_size / big; ow *= k; oh *= k; }
    out_w = roundMult(ow, mult); out_h = roundMult(oh, mult);
  }
  out_w = Math.max(mult, Math.round(out_w));
  out_h = Math.max(mult, Math.round(out_h));
  return { rx, ry, rw: rw_i, rh: rh_i, out_w, out_h };
}

// 从遮罩 canvas 的 alpha 通道找画过像素的外接框。
// 返回 [x0,y0,x1,y1]（x1/y1 开区间）或没有画过时返回 null。
export function maskBBoxFromImageData(data, w, h, thresh = 8) {
  let x0 = w, y0 = h, x1 = 0, y1 = 0, found = false;
  for (let y = 0; y < h; y++) {
    let row = y * w * 4;
    for (let x = 0; x < w; x++) {
      if (data[row + x * 4 + 3] > thresh) {
        found = true;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return found ? [x0, y0, x1 + 1, y1 + 1] : null;
}

// bbox 每侧外扩 `px` 像素（夹紧）——镜像 Python 遮罩膨胀（grow）对 bbox 的效果。
export function growBBox(bbox, px, W, H) {
  if (!bbox) return null;
  return [
    Math.max(0, bbox[0] - px), Math.max(0, bbox[1] - px),
    Math.min(W, bbox[2] + px), Math.min(H, bbox[3] + px),
  ];
}

// 向外接缝羽化 alpha——Python `_blur_alpha` 的 scipy 路径的 canvas 镜像
// （sf_utils/inpaint_helpers.py）。把 RGBA `data` 缓冲（w x h）的 ALPHA 通道
// 当遮罩读，返回 Float32Array alpha：遮罩**内部**（及边缘）1.0，向外 `k` 像素
// 经与节点相同的 smoothstep 衰减到 0。纯函数、无 DOM，可单测。
//
// 节点用精确的欧氏有符号距离变换；这里用快速 2 趟 (1, sqrt2) chamfer 变换
// 计算外侧距离（内部就是 1）。chamfer 与欧氏相差几个百分点，软接缝预览上
// 不可见。这就是"预览 == 结果"的对齐：适中的 softness 在编辑器里不再显得
// 比真实缝合接缝更紧。
export function seamAlphaFromAlpha(data, w, h, k) {
  const n = w * h;
  const INF = 1e9;
  const d = new Float32Array(n);
  for (let i = 0, p = 3; i < n; i++, p += 4) d[i] = data[p] > 127 ? 0 : INF;
  const d1 = 1, d2 = Math.SQRT2;
  // 前向趟：左上 -> 右下（已访问的邻居）
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let v = d[i];
      if (v === 0) continue;
      if (x > 0) v = Math.min(v, d[i - 1] + d1);
      if (y > 0) {
        v = Math.min(v, d[i - w] + d1);
        if (x > 0) v = Math.min(v, d[i - w - 1] + d2);
        if (x < w - 1) v = Math.min(v, d[i - w + 1] + d2);
      }
      d[i] = v;
    }
  }
  // 后向趟：右下 -> 左上
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      let v = d[i];
      if (v === 0) continue;
      if (x < w - 1) v = Math.min(v, d[i + 1] + d1);
      if (y < h - 1) {
        v = Math.min(v, d[i + w] + d1);
        if (x < w - 1) v = Math.min(v, d[i + w + 1] + d2);
        if (x > 0) v = Math.min(v, d[i + w - 1] + d2);
      }
      d[i] = v;
    }
  }
  const out = new Float32Array(n);
  const kk = Math.max(1e-3, k);
  for (let i = 0; i < n; i++) {
    if (d[i] === 0) { out[i] = 1; continue; }      // 遮罩内部 + 边缘
    const t = Math.max(0, Math.min(1, 1 - d[i] / kk));
    out[i] = t * t * (3 - 2 * t);                  // smoothstep，与节点一致
  }
  return out;
}
