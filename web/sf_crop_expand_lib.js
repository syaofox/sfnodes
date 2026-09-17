// ==========================================================================
// sf_crop_expand_lib.js - SF Image Crop Expand 纯几何/纯逻辑库
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_crop_expand.js 使用，也供 tests/ 复制为 .mjs 直接测试。
// 复刻 ComfyUI-YCNodes_Toolkit js/ImageCrop.js 的交互数学——含"拖拽冻结
// 快照"防自反馈飘移机制（裁剪框扩张 → 显示区扩张 → scale 变小 → 鼠标反算
// 漂移的循环），拖拽期间一律使用 onMouseDown 时冻结的 scale/offset/边界。
// ==========================================================================

// 预设比例（key 与原版一致；ratio 为 w/h，null = 不约束）
export const ASPECT_RATIOS = [
  { key: "free", label: "Free", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "4:3", label: "4:3", ratio: 4 / 3 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "3:2", label: "3:2", ratio: 3 / 2 },
  { key: "2:3", label: "2:3", ratio: 2 / 3 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
  { key: "9:16", label: "9:16", ratio: 9 / 16 },
  { key: "21:9", label: "21:9", ratio: 21 / 9 },
  { key: "2:1", label: "2:1", ratio: 2 },
  { key: "1:2", label: "1:2", ratio: 0.5 },
  { key: "custom", label: "Custom", ratio: null },
];

// 竖列预设比例（画布区左侧一列，free 置顶；预设 key 集合与原版行2 一致）
export const RATIO_PRESETS_COL = ["free", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"];

// 节点内边距布局。左侧竖列（Free 置顶 + 预设 + Custom/Reset/Color 收尾）
// 从节点顶直通画布区底（ratioColW + ratioColGap 由图片区让出）；Load Image/
// Browse 按钮贴节点底部一行左侧，与信息文本同排（bottomH 为底行高度）。
export const LAYOUT = { shiftLeft: 10, shiftRight: 80, ratioColW: 34, ratioColGap: 6, bottomH: 26 };

// 底部行高度由 LAYOUT.bottomH 承担（Load Image/Browse 按钮行 h21 + 余量，
// 按钮与信息文本同排）。画布区高度必须让出这一行，否则显示区底缘压到按钮。

export const MIN_SIZE = 10;
export const HANDLE_SIZE = 10;

// 节点最小宽高（控件不溢出前提下的下限）：
// - 高度：竖列 11 项（colTop=16 起，步进 22，底 ≈254）+ 底行 26 + 上下边距
// - 宽度：底行 Load(44)/Browse(48)（右缘 106）+ 最小文本窗 ~100（信息文本
//   溢出时截断 "…"，节点拉宽即恢复全文）+ 间隙与 shiftRight → 取整 320；
//   推导：10（左缩进）+ 106（按钮）+ 6（间隙）+ 100（文本）+ 86（右槽区）≈ 308。
// 双端拖拽 resize 的最小值都取自 node.computeSize()（前端包实测 onDrag 里
// clamp 到 computeSize）——主扩展包装 computeSize 返回 ensureMinSize 结果
// 钳住拖拽；创建/恢复两处 clampNodeSize 兜底。加载图片不改节点大小（显示
// 区 scale 动态适配现有画布区域）。
export const MIN_NODE_WIDTH = 320;
export const MIN_NODE_HEIGHT = 300;

// ensureMinSize(w, h, minW, minH) → [w, h]（低于下限抬升，非 0 数值原样放行）。
// computeSize 包装与 clampNodeSize 共用；minW/minH 可选（组合节点双列布局取更大
// 下限，见 sf_crop_expand_brush_mask_lib.js），默认即本库 MIN。
export function ensureMinSize(w, h, minW = MIN_NODE_WIDTH, minH = MIN_NODE_HEIGHT) {
  return [Math.max(w || 0, minW), Math.max(h || 0, minH)];
}

// 右下角 resize cursor 视觉修正：原生命中区（resizeHandleSize 15×15）本身
// 可用，无需扩大；但 pointer.resizeDirection 会被 hover 判定切换与第三方扩
// 展重放 processMouseMove 反复清空（帧尾 updateCursorStyle 读到空 → cursor
// 闪回 default，视觉上几乎不触发）。主扩展注册后执行的 mousemove listener
// 在原生 15×15 区内补写 dir——判定区与原生命中区一致（cursor 与拖动匹配）。
export const RESIZE_HANDLE_SIZE = 15;

// hitResizeCornerSE(localX, localY, w, h) → 右下角 handleSize×handleSize 内
// （调用点已保证坐标在节点 boundingRect 内；x=右缘/y=底缘含边界）。
export function hitResizeCornerSE(localX, localY, w, h, handleSize = RESIZE_HANDLE_SIZE) {
  return localX >= w - handleSize && localX <= w && localY >= h - handleSize && localY <= h;
}

// ratioFromAspect(key, customW, customH) → number | null
// custom 比例取 customW/customH，非法输入退化为不约束。
export function ratioFromAspect(key, customW, customH) {
  if (key === "custom") {
    const w = Number(customW);
    const h = Number(customH);
    if (w > 0 && h > 0) return w / h;
    return null;
  }
  const entry = ASPECT_RATIOS.find((r) => r.key === key);
  return entry ? entry.ratio : null;
}

// computeDisplayMetrics(state, nodeW, nodeH, frozen, extraLeft) → 显示坐标系
// state: {cropX, cropY, cropW, cropH, srcW, srcH}
// frozen: 拖拽期快照（onMouseDown 时保存的 displayMin/scale/offset/scaled 尺寸），
//         非拖拽传 null 走动态计算（视图自适应）；frozen 已含 extraLeft 影响，
//         原样透传。
// extraLeft: 显示区左侧额外让出的宽度（组合节点第二列；默认 0 = 原行为）。
export function computeDisplayMetrics(state, nodeW, nodeH, frozen, extraLeft = 0) {
  if (frozen) {
    return {
      displayMinX: frozen.displayMinX,
      displayMinY: frozen.displayMinY,
      scale: frozen.scale,
      offsetX: frozen.offsetX,
      offsetY: frozen.offsetY,
      scaledDisplayWidth: frozen.scaledDisplayWidth,
      scaledDisplayHeight: frozen.scaledDisplayHeight,
    };
  }
  const { shiftLeft, shiftRight } = LAYOUT;
  const displayMinX = Math.min(0, state.cropX);
  const displayMinY = Math.min(0, state.cropY);
  const displayMaxX = Math.max(state.srcW, state.cropX + state.cropW);
  const displayMaxY = Math.max(state.srcH, state.cropY + state.cropH);
  const displayWidth = Math.max(1, displayMaxX - displayMinX);
  const displayHeight = Math.max(1, displayMaxY - displayMinY);

  const extra = Number(extraLeft) || 0;
  const areaW = nodeW - shiftRight - shiftLeft - LAYOUT.ratioColW - LAYOUT.ratioColGap - extra;
  const areaH = nodeH - shiftLeft - shiftLeft - LAYOUT.bottomH;
  const scale = Math.min(areaW / displayWidth, areaH / displayHeight);
  const scaledDisplayWidth = displayWidth * scale;
  const scaledDisplayHeight = displayHeight * scale;
  const offsetX = shiftLeft + LAYOUT.ratioColW + LAYOUT.ratioColGap + extra + (areaW - scaledDisplayWidth) / 2;
  const offsetY = shiftLeft + (areaH - scaledDisplayHeight) / 2;

  return { displayMinX, displayMinY, scale, offsetX, offsetY, scaledDisplayWidth, scaledDisplayHeight };
}

// 屏幕局部坐标 → 图片坐标（metrics 来自 computeDisplayMetrics）
export function localToImage(localX, localY, m) {
  return {
    x: (localX - m.offsetX) / m.scale + m.displayMinX,
    y: (localY - m.offsetY) / m.scale + m.displayMinY,
  };
}

// 图片坐标 → 屏幕局部坐标（localToImage 的互逆对；绘制覆盖层用。
// 注意与 sf_brush_mask_lib.imageToLocal 的区别：本函数含 displayMin 偏移，
// 后者是针对源图局部画布（displayMin=0）的简化版）
export function imageToLocal(imgX, imgY, m) {
  return {
    x: m.offsetX + (imgX - m.displayMinX) * m.scale,
    y: m.offsetY + (imgY - m.displayMinY) * m.scale,
  };
}

// ── 画布绘制（ctx 与主题色由调用方提供；函数本身无 app/DOM 依赖）──────────
// SFImageCropExpand 与 SFImageCropExpandBrushMask 共用（原 crop_expand.js 内联
// 实现提升；去 node 依赖改为参数传入，画序/视觉与原实现逐行一致）。

// 未加载/绘制失败时的占位网格。
export function drawPlaceholder(ctx, x, y, width, height, scale) {
  ctx.fillStyle = "rgba(100,100,100,0.3)";
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = "rgba(150,150,150,0.2)";
  ctx.lineWidth = 1;
  const gridSize = 32 * scale;
  for (let gx = x; gx <= x + width; gx += gridSize) {
    ctx.beginPath();
    ctx.moveTo(gx, y);
    ctx.lineTo(gx, y + height);
    ctx.stroke();
  }
  for (let gy = y; gy <= y + height; gy += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, gy);
    ctx.lineTo(x + width, gy);
    ctx.stroke();
  }
}

// drawCropBox(ctx, rect, srcW, srcH, m)：框外（原图内）半透明压暗 + 框线 +
// 九宫格 + 8 控制点。rect: {x, y, w, h}（图片坐标），m 来自 computeDisplayMetrics。
export function drawCropBox(ctx, rect, srcW, srcH, m) {
  const x1 = m.offsetX + (rect.x - m.displayMinX) * m.scale;
  const y1 = m.offsetY + (rect.y - m.displayMinY) * m.scale;
  const x2 = x1 + rect.w * m.scale;
  const y2 = y1 + rect.h * m.scale;

  const imgX1 = m.offsetX + (0 - m.displayMinX) * m.scale;
  const imgY1 = m.offsetY + (0 - m.displayMinY) * m.scale;
  const imgX2 = imgX1 + srcW * m.scale;
  const imgY2 = imgY1 + srcH * m.scale;

  // 裁切框外（原图内）的半透明遮罩
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  if (y1 > imgY1) ctx.fillRect(imgX1, imgY1, imgX2 - imgX1, y1 - imgY1);
  if (y2 < imgY2) ctx.fillRect(imgX1, y2, imgX2 - imgX1, imgY2 - y2);
  if (x1 > imgX1) ctx.fillRect(imgX1, Math.max(y1, imgY1), x1 - imgX1, Math.min(y2, imgY2) - Math.max(y1, imgY1));
  if (x2 < imgX2) ctx.fillRect(x2, Math.max(y1, imgY1), imgX2 - x2, Math.min(y2, imgY2) - Math.max(y1, imgY1));

  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // 九宫格辅助线
  ctx.strokeStyle = "rgba(255,255,255,0.4)";
  ctx.lineWidth = 1;
  const w3 = (x2 - x1) / 3;
  const h3 = (y2 - y1) / 3;
  for (let i = 1; i < 3; i++) {
    ctx.beginPath();
    ctx.moveTo(x1 + w3 * i, y1);
    ctx.lineTo(x1 + w3 * i, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1 + h3 * i);
    ctx.lineTo(x2, y1 + h3 * i);
    ctx.stroke();
  }

  // 控制点
  const hs = HANDLE_SIZE;
  const handles = [
    { x: x1, y: y1 }, { x: x2, y: y1 }, { x: x1, y: y2 }, { x: x2, y: y2 },
    { x: (x1 + x2) / 2, y: y1 }, { x: (x1 + x2) / 2, y: y2 },
    { x: x1, y: (y1 + y2) / 2 }, { x: x2, y: (y1 + y2) / 2 },
  ];
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.strokeStyle = "rgba(0,0,0,0.8)";
  ctx.lineWidth = 1;
  for (const p of handles) {
    ctx.fillRect(p.x - hs / 2, p.y - hs / 2, hs, hs);
    ctx.strokeRect(p.x - hs / 2, p.y - hs / 2, hs, hs);
  }
}

// getHandleAtPoint(imgX, imgY, rect, scale) → 'nw'|'ne'|'sw'|'se'|'n'|'s'|'w'|'e'|'move'|null
// handleSize 以图片坐标系计（屏幕 10px / scale）。
export function getHandleAtPoint(imgX, imgY, rect, scale) {
  const hs = HANDLE_SIZE / scale;
  const x1 = rect.x;
  const y1 = rect.y;
  const x2 = rect.x + rect.w;
  const y2 = rect.y + rect.h;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;

  if (Math.abs(imgX - x1) < hs && Math.abs(imgY - y1) < hs) return "nw";
  if (Math.abs(imgX - x2) < hs && Math.abs(imgY - y1) < hs) return "ne";
  if (Math.abs(imgX - x1) < hs && Math.abs(imgY - y2) < hs) return "sw";
  if (Math.abs(imgX - x2) < hs && Math.abs(imgY - y2) < hs) return "se";
  if (Math.abs(imgX - cx) < hs && Math.abs(imgY - y1) < hs) return "n";
  if (Math.abs(imgX - cx) < hs && Math.abs(imgY - y2) < hs) return "s";
  if (Math.abs(imgX - x1) < hs && Math.abs(imgY - cy) < hs) return "w";
  if (Math.abs(imgX - x2) < hs && Math.abs(imgY - cy) < hs) return "e";
  if (imgX >= x1 && imgX <= x2 && imgY >= y1 && imgY <= y2) return "move";
  return null;
}

export function getCursorForHandle(handle) {
  const cursors = {
    nw: "nw-resize",
    ne: "ne-resize",
    sw: "sw-resize",
    se: "se-resize",
    n: "n-resize",
    s: "s-resize",
    w: "w-resize",
    e: "e-resize",
    move: "move",
  };
  return cursors[handle] || "default";
}

// updateCropByDrag(drag, handle, imgX, imgY, ratio) → 新矩形 {x, y, w, h}
// drag: {startImgX, startImgY, startRect:{x,y,w,h}}（onMouseDown 快照）
// ratio: number | null（非 move 手柄应用比例约束，逻辑与原版一致）
export function updateCropByDrag(drag, handle, imgX, imgY, ratio) {
  const dx = imgX - drag.startImgX;
  const dy = imgY - drag.startImgY;
  const s = drag.startRect;

  let newX = s.x;
  let newY = s.y;
  let newW = s.w;
  let newH = s.h;

  if (handle === "move") {
    newX = s.x + dx;
    newY = s.y + dy;
  } else if (handle === "nw") {
    newX = s.x + dx;
    newY = s.y + dy;
    newW = s.w - dx;
    newH = s.h - dy;
  } else if (handle === "ne") {
    newY = s.y + dy;
    newW = s.w + dx;
    newH = s.h - dy;
  } else if (handle === "sw") {
    newX = s.x + dx;
    newW = s.w - dx;
    newH = s.h + dy;
  } else if (handle === "se") {
    newW = s.w + dx;
    newH = s.h + dy;
  } else if (handle === "n") {
    newY = s.y + dy;
    newH = s.h - dy;
  } else if (handle === "s") {
    newH = s.h + dy;
  } else if (handle === "w") {
    newX = s.x + dx;
    newW = s.w - dx;
  } else if (handle === "e") {
    newW = s.w + dx;
  }

  if (ratio && handle !== "move") {
    if (handle === "n" || handle === "s") {
      newW = Math.round(newH * ratio);
      if (handle === "n") newX = s.x + (s.w - newW) / 2;
    } else if (handle === "w" || handle === "e") {
      newH = Math.round(newW / ratio);
      if (handle === "w") newY = s.y + (s.h - newH) / 2;
    } else {
      // 角点：保持比例，以宽度为准
      newH = Math.round(newW / ratio);
    }
  }

  if (newW < MIN_SIZE) newW = MIN_SIZE;
  if (newH < MIN_SIZE) newH = MIN_SIZE;

  return { x: newX, y: newY, w: newW, h: newH };
}

// roundRect(rect) → 各分量取整（MouseUp 落定）
export function roundRect(rect) {
  return {
    x: Math.round(rect.x),
    y: Math.round(rect.y),
    w: Math.round(rect.w),
    h: Math.round(rect.h),
  };
}

// applyRatioToRect(rect, ratio) → 以宽度为准调整高度，保持中心点
// （点击预设比例按钮 / 应用自定义比例）
export function applyRatioToRect(rect, ratio) {
  if (!ratio || ratio <= 0) return { ...rect };
  const centerX = rect.x + rect.w / 2;
  const centerY = rect.y + rect.h / 2;
  const newH = Math.max(MIN_SIZE, Math.round(rect.w / ratio));
  return { x: rect.x, w: rect.w, y: Math.round(centerY - newH / 2), h: newH };
}

// 裁剪框是否超出源图边界（信息栏 "(Extended)" 标注）
export function isExtended(rect, srcW, srcH) {
  return rect.x < 0 || rect.y < 0 || rect.x + rect.w > srcW || rect.y + rect.h > srcH;
}

// normalizeRatioPresets(raw) → [{name, w, h}]
// 已保存自定义比例定义的前端归一化（与后端 crop_expand_presets._normalize_presets
// 同口径，前端兜底）：接受数组或 {presets:[...]}；name 非空、w/h 为正有限数且
// ≤ 10000；重名保留首个。供 Custom 管理窗渲染列表（纯函数，可 .mjs 直测）。
export function normalizeRatioPresets(raw) {
  const RATIO_MAX = 10000;
  const list = Array.isArray(raw) ? raw : raw && Array.isArray(raw.presets) ? raw.presets : [];
  const valid = (v) => typeof v === "number" && isFinite(v) && v > 0 && v <= RATIO_MAX;
  const out = [];
  const seen = new Set();
  for (const item of list) {
    if (!item || typeof item !== "object") continue;
    const name = String(item.name ?? "").trim();
    if (!name || seen.has(name)) continue;
    if (!valid(item.w) || !valid(item.h)) continue;
    seen.add(name);
    out.push({ name, w: item.w, h: item.h });
  }
  return out;
}
