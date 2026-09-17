// ==========================================================================
// sf_brush_mask_lib.js - SF Image Brush Mask 纯几何/纯逻辑库
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_brush_mask.js 使用，也供 tests/ 复制为 .mjs 直接测试。
// 布局向 SFImageCropExpand 看齐（§37/§44 同款）：左侧工具竖列（模式/
// 破坏性操作/数值步进/取色，节点顶直通画布区底）+ 底行（Load/Browse 与
// 信息文本同排）；显示坐标系公式与 sf_crop_expand_lib.js 同形。
// 原版 YCNodes 的横向 Size/Opacity 拖拽滑块在 30px 竖列里放不下，改为
// 步进器（Size± 步长 2 / Opa± 步长 5%，纯函数 stepBrushSize/stepOpacity），
// 实时数值进底行信息文本。
// ==========================================================================

// 布局常量（与 sf_crop_expand_lib.js LAYOUT 同形）：左侧工具列
// （toolColW + toolColGap 由图片区让宽）从节点顶直通画布区底；底行
//（Load/Browse + 信息文本）占 bottomH。
export const LAYOUT = { shiftLeft: 10, shiftRight: 80, toolColW: 34, toolColGap: 6, bottomH: 26 };

// 左侧竖列按钮（从上到下）：模式 → 破坏性操作 → 数值步进 → 取色。
// 列按钮几何：w=30、h=18、步进 22、列顶 = shiftLeft+6（主扩展 buildControls）。
// eraserColor 已移除（真擦除预览无红色可染，见 §45.9）。
export const TOOL_COL = [
  "brush",
  "erase",
  "clear",
  "undo",
  "sizeMinus",
  "sizePlus",
  "opaMinus",
  "opaPlus",
  "brushColor",
];

export const COL_TOP = 16;
export const COL_W = 30;
export const COL_H = 18;
export const COL_GAP = 4;
export const COL_STEP = COL_H + COL_GAP;

// 步进器步长/边界（与后端域一致：size 1..200，opacity 0.1..1.0）。
export const SIZE_MIN = 1;
export const SIZE_MAX = 200;
export const SIZE_STEP = 2;
export const OPA_MIN = 0.1;
export const OPA_MAX = 1.0;
export const OPA_STEP = 0.05;

// 滚轮快调命中的按钮 id（左竖列四个步进器；取色/模式/底行按钮不响应滚轮）。
export const WHEEL_STEPPERS = ["sizeMinus", "sizePlus", "opaMinus", "opaPlus"];

// hitStepper(buttons, lx, ly) → 步进器 id | null
// buttons: buildControls 几何（含 x/y/w/h，步进器 y 均为数值）；纯函数，
// 主扩展的全局 wheel 监听与测试共用。
export function hitStepper(buttons, lx, ly) {
  for (const b of buttons || []) {
    if (!WHEEL_STEPPERS.includes(b.id)) continue;
    if (lx >= b.x && lx <= b.x + b.w && ly >= b.y && ly <= b.y + b.h) return b.id;
  }
  return null;
}

// wheelDir(deltaY) → +1（上滚增大）/ -1（下滚减小）/ 0（水平滚忽略）
export function wheelDir(deltaY) {
  if (deltaY < 0) return 1;
  if (deltaY > 0) return -1;
  return 0;
}

// wheelAction(id, dir) → 步进后的按钮行为 id（上滚走 Plus、下滚走 Minus）
export function wheelAction(id, dir) {
  if (id === "sizeMinus" || id === "sizePlus") return dir >= 0 ? "sizePlus" : "sizeMinus";
  if (id === "opaMinus" || id === "opaPlus") return dir >= 0 ? "opaPlus" : "opaMinus";
  return null;
}

// stepBrushSize(cur, dir, step) → 新笔刷直径（dir=+1/-1，钳制 1..200 取整；
// step 默认 SIZE_STEP，设置页 sfnodes.BrushMask.SizeStep 覆盖）
export function stepBrushSize(cur, dir, step = SIZE_STEP) {
  const st = Number(step);
  const dv = Number.isFinite(st) && st > 0 ? st : SIZE_STEP;
  const v = (Number(cur) || 0) + (dir >= 0 ? dv : -dv);
  return Math.max(SIZE_MIN, Math.min(SIZE_MAX, Math.round(v)));
}

// stepOpacity(cur, dir, step) → 新预览透明度（dir=+1/-1，钳制 0.1..1.0，
// 保留 2 位小数；step 默认 OPA_STEP，设置页 sfnodes.BrushMask.OpacityStep
// 以整数百分比存取，调用方除以 100 传入）
export function stepOpacity(cur, dir, step = OPA_STEP) {
  const st = Number(step);
  const dv = Number.isFinite(st) && st > 0 ? st : OPA_STEP;
  const v = (Number(cur) || 0) + (dir >= 0 ? dv : -dv);
  return Math.max(OPA_MIN, Math.min(OPA_MAX, Math.round(v * 100) / 100));
}

// 节点最小宽高（控件不溢出前提下的下限）：
// - 宽度：底行 Load(44)/Browse(48)（右缘 106）+ 最小文本窗 ~100（信息文本
//   溢出时截断 "…"，节点拉宽即恢复全文）+ shiftRight/边距 → 取整 320；
//   推导：10（左缩进）+ 106（按钮）+ 6（间隙）+ 100（文本）+ 86（右槽区）≈ 308。
// - 高度：竖列 9 项（列顶 16 起，步进 22，底 =16+9*22-4=210）+ 底行 26 +
//   上下边距 → 取整 320（竖列缩短后仍保持，与旧存量 size 兼容）。
// 双端拖拽 resize 的最小值都取自 node.computeSize()（前端包实测 onDrag 里
// clamp 到 computeSize）——主扩展包装 computeSize 返回 ensureMinSize 结果
// 钳住拖拽；创建/恢复两处 clampNodeSize 兜底。
export const MIN_NODE_WIDTH = 320;
export const MIN_NODE_HEIGHT = 320;

// ensureMinSize(w, h) → [w, h]（低于下限抬升，非 0 数值原样放行）。
// computeSize 包装与 clampNodeSize 共用。
export function ensureMinSize(w, h) {
  return [Math.max(w || 0, MIN_NODE_WIDTH), Math.max(h || 0, MIN_NODE_HEIGHT)];
}

// 右下角 resize cursor 视觉修正：原生命中区（resizeHandleSize 15×15）本身
// 可用，无需扩大；主扩展注册后执行的 mousemove listener 在原生区内直接写
// style.cursor（绕过 pointer.resizeDirection 被清空的间接链路，见 §44）。
export const RESIZE_HANDLE_SIZE = 15;

// hitResizeCornerSE(localX, localY, w, h) → 右下角 handleSize×handleSize 内。
export function hitResizeCornerSE(localX, localY, w, h, handleSize = RESIZE_HANDLE_SIZE) {
  return localX >= w - handleSize && localX <= w && localY >= h - handleSize && localY <= h;
}

// computeDisplayMetrics(state, nodeW, nodeH) → 显示坐标系
// state: {srcW, srcH}；显示区为节点内让出左工具列与底信息行后的矩形，
// 源图按 min(scaleX, scaleY) 等比居中（公式与 sf_crop_expand_lib.js 同形）。
export function computeDisplayMetrics(state, nodeW, nodeH) {
  const { shiftLeft, shiftRight, toolColW, toolColGap, bottomH } = LAYOUT;
  const areaW = Math.max(1, nodeW - shiftRight - shiftLeft - toolColW - toolColGap);
  const areaH = Math.max(1, nodeH - shiftLeft - shiftLeft - bottomH);
  const srcW = Math.max(1, state.srcW || 512);
  const srcH = Math.max(1, state.srcH || 512);
  const scale = Math.min(areaW / srcW, areaH / srcH);
  const scaledW = srcW * scale;
  const scaledH = srcH * scale;
  const offsetX = shiftLeft + toolColW + toolColGap + (areaW - scaledW) / 2;
  const offsetY = shiftLeft + (areaH - scaledH) / 2;
  return { scale, offsetX, offsetY, scaledW, scaledH, areaW, areaH };
}

// 屏幕局部坐标 → 图片坐标（metrics 来自 computeDisplayMetrics）
export function localToImage(localX, localY, m) {
  return { x: (localX - m.offsetX) / m.scale, y: (localY - m.offsetY) / m.scale };
}

// 图片坐标 → 屏幕局部坐标（绘制笔触用）
export function imageToLocal(imgX, imgY, m) {
  return { x: m.offsetX + imgX * m.scale, y: m.offsetY + imgY * m.scale };
}

// clampToImage(x, y, srcW, srcH) → 钳制到 [0, srcW-1]×[0, srcH-1]
// （与原版 valueUpdate 落笔钳制一致）。
export function clampToImage(x, y, srcW, srcH) {
  return {
    x: Math.max(0, Math.min(x, (srcW || 512) - 1)),
    y: Math.max(0, Math.min(y, (srcH || 512) - 1)),
  };
}

// colorTextStyle(color) → 取色按钮文字色（背景亮度 > 128 取黑，否则白）。
// 同时接受 "#rrggbb" 与 "r,g,b" 两种形态（Fill/笔刷取色按钮共用）。
export function colorTextStyle(color) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(String(color || ""));
  if (!m) {
    const rgb = String(color || "255,255,255").split(",").map((v) => parseInt(String(v).trim(), 10));
    if (rgb.length === 3 && rgb.every((v) => Number.isFinite(v))) {
      const brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000;
      return brightness > 128 ? "rgba(0,0,0,0.9)" : "rgba(255,255,255,0.9)";
    }
    return "rgba(255,255,255,0.9)";
  }
  const brightness = (parseInt(m[1], 16) * 299 + parseInt(m[2], 16) * 587 + parseInt(m[3], 16) * 114) / 1000;
  return brightness > 128 ? "rgba(0,0,0,0.9)" : "rgba(255,255,255,0.9)";
}

// ── 画布绘制（ctx 由调用方提供；SFImageBrushMask 与 SFImageCropExpandBrushMask
// 共用，原 sf_brush_mask.js 内联实现提升）──────────────────────────────────

// drawStrokePath(ctx, pts, m, lineW, style, fill)：单笔笔触绘制。
// fill=true 时整体填充闭合多边形（SAM 并入的 fill 笔触）；否则按 lineW 描边
// （单点画直径=lineW 的圆盘，与后端印章同语义）。m 需提供 scale/offset（源图
// 局部画布传 unit = {scale:1, offsetX:0, offsetY:0}）。
export function drawStrokePath(ctx, pts, m, lineW, style, fill) {
  if (!pts || pts.length === 0) return;
  // fill 笔触（SAM 并入）：整体填充闭合多边形
  if (fill) {
    if (pts.length === 1) {
      const p = imageToLocal(pts[0][0], pts[0][1], m);
      ctx.fillStyle = style;
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(1, lineW / 2), 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    ctx.fillStyle = style;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const p = imageToLocal(pts[i][0], pts[i][1], m);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fill();
    return;
  }
  ctx.lineWidth = Math.max(1, lineW);
  ctx.strokeStyle = style;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (pts.length === 1) {
    const p = imageToLocal(pts[0][0], pts[0][1], m);
    // 单点：画一个直径=线宽的圆盘（后端同款：单点印章）
    ctx.fillStyle = style;
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(1, lineW / 2), 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const p = imageToLocal(pts[i][0], pts[i][1], m);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
}

// paintStrokeMask(cvs, strokes, opts) → ctx：离屏画布按画序合成二值遮罩笔触
// （真擦除预览）——brush/fill 以 opts.paintStyle 覆盖，erase 以 destination-out
// 打洞（主画布禁用 destination-out，会连照片一起擦掉）。离屏按源图像素绘制
// （lineW 取源图笔刷直径），后端 rasterize_strokes 为同款画序，预览与输出
// 逐像素一致。opts.current 为进行中笔触 {mode, size, points}（可选）。
export function paintStrokeMask(cvs, strokes, opts = {}) {
  const defaultSize = opts.defaultSize || 80;
  const paintStyle = opts.paintStyle || "rgba(255,255,255,1)";
  const mx = cvs.getContext("2d");
  mx.save();
  mx.setTransform(1, 0, 0, 1, 0, 0);
  mx.clearRect(0, 0, cvs.width, cvs.height);
  mx.globalCompositeOperation = "source-over";
  const unit = { scale: 1, offsetX: 0, offsetY: 0 };
  const paint = (s) => {
    const mode = s.mode || "brush";
    if (mode === "erase") {
      mx.globalCompositeOperation = "destination-out";
      drawStrokePath(mx, s.points, unit, s.size || defaultSize, "rgba(0,0,0,1)", false);
      mx.globalCompositeOperation = "source-over";
    } else {
      drawStrokePath(mx, s.points, unit, s.size || defaultSize, paintStyle, mode === "fill");
    }
  };
  for (const s of strokes || []) paint(s);
  if (opts.current && opts.current.points && opts.current.points.length > 0) paint(opts.current);
  mx.restore();
  return mx;
}

// parseStroke(stroke, defaultSize) → {points, mode, size}
// 与 sf_utils/brush_mask.py::_parse_one_stroke 同语义（双端镜像）：
// mode:size:opacity[:r,g,b]:points / mode:points / 裸点列；opacity 与颜色
// 只解析不使用（后端二值，预览语义）。points 为 [{x, y}]（float，未裁剪）。
export function parseStroke(stroke, defaultSize = 80) {
  let mode = "brush";
  let size = Number(defaultSize) || 80;
  let pointsStr = String(stroke ?? "");
  if (pointsStr.includes(":")) {
    const parts = pointsStr.split(":");
    if (parts[0] === "brush" || parts[0] === "erase") {
      mode = parts[0];
      if (parts.length >= 4) {
        const part3 = parts[3];
        // 严格颜色判定：三段必须都是纯整数（"255,0,0" 是颜色，
        // "10,10;20,20" 含 ";" 是旧格式点列——parseInt 会静默截断
        // "10;20"→10 造成误判，原版 YCNodes 同款 bug，此处修复）。
        const segs = part3 ? part3.split(",") : [];
        const isColor = segs.length === 3 && segs.every((c) => /^\s*\d+\s*$/.test(c));
        if (isColor) {
          const [r, g, b] = segs.map((c) => parseInt(c.trim(), 10));
          if ([r, g, b].every((v) => Number.isFinite(v) && v >= 0 && v <= 255)) {
            size = parseFloat(parts[1]) || size;
            pointsStr = parts.slice(4).join(":");
          } else {
            size = parseFloat(parts[1]) || size;
            pointsStr = parts.slice(3).join(":");
          }
        } else {
          size = parseFloat(parts[1]) || size;
          pointsStr = parts.slice(3).join(":");
        }
      } else {
        pointsStr = parts.slice(1).join(":");
      }
    }
  }
  const points = [];
  for (const seg of pointsStr.split(";")) {
    if (!seg.trim()) continue;
    const coords = seg.split(",");
    if (coords.length === 2) {
      const x = parseFloat(coords[0]);
      const y = parseFloat(coords[1]);
      if (Number.isFinite(x) && Number.isFinite(y)) points.push({ x, y });
    }
  }
  return { points, mode, size };
}

// parseBrushData(brushData, defaultSize) → strokes 数组（未裁剪，与 Python
// parse_strokes 裁剪前形状一致，供往返测试）。
export function parseBrushData(brushData, defaultSize = 80) {
  if (typeof brushData !== "string" || !brushData.trim()) return [];
  const out = [];
  for (const raw of brushData.split("|")) {
    if (!raw.trim()) continue;
    const parsed = parseStroke(raw, defaultSize);
    if (parsed.points.length > 0) out.push(parsed);
  }
  return out;
}

// buildBrushData(strokes, opacity) → 原版 brush_data 串
// strokes: [{mode, size, points:[{x,y}]}]；opacity 默认 1.0（后端忽略）。
export function buildBrushData(strokes, opacity = 1.0) {
  return (strokes || [])
    .map((s) => {
      const pts = (s.points || []).map((p) => `${p.x},${p.y}`).join(";");
      return `${s.mode || "brush"}:${s.size ?? 80}:${opacity}:${pts}`;
    })
    .join("|");
}
