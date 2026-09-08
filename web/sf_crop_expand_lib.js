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

export const RATIO_PRESETS_ROW2 = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"];

// 节点内边距布局（与原版 DEFAULT_LAYOUT 一致）
export const LAYOUT = { shiftLeft: 10, shiftRight: 80, panelHeight: 58 };

export const MIN_SIZE = 10;
export const HANDLE_SIZE = 10;

// 节点最小宽高：面板按钮行（row1 到 x≈290）+ 画布区 + 信息文本所需空间。
// 创建/恢复/尺寸自适应三处统一钳制（LiteGraph 默认按 schema 算的初始尺寸
// 偏小，按钮会外溢）。
export const MIN_NODE_WIDTH = 400;
export const MIN_NODE_HEIGHT = 360;

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

// computeDisplayMetrics(state, nodeW, nodeH, frozen) → 显示坐标系
// state: {cropX, cropY, cropW, cropH, srcW, srcH}
// frozen: 拖拽期快照（onMouseDown 时保存的 displayMin/scale/offset/scaled 尺寸），
//         非拖拽传 null 走动态计算（视图自适应）。
export function computeDisplayMetrics(state, nodeW, nodeH, frozen) {
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
  const { shiftLeft, shiftRight, panelHeight } = LAYOUT;
  const displayMinX = Math.min(0, state.cropX);
  const displayMinY = Math.min(0, state.cropY);
  const displayMaxX = Math.max(state.srcW, state.cropX + state.cropW);
  const displayMaxY = Math.max(state.srcH, state.cropY + state.cropH);
  const displayWidth = Math.max(1, displayMaxX - displayMinX);
  const displayHeight = Math.max(1, displayMaxY - displayMinY);

  const areaW = nodeW - shiftRight - shiftLeft;
  const areaH = nodeH - shiftLeft - shiftLeft - panelHeight;
  const scale = Math.min(areaW / displayWidth, areaH / displayHeight);
  const scaledDisplayWidth = displayWidth * scale;
  const scaledDisplayHeight = displayHeight * scale;
  const offsetX = shiftLeft + (areaW - scaledDisplayWidth) / 2;
  const offsetY = shiftLeft + panelHeight + (areaH - scaledDisplayHeight) / 2;

  return { displayMinX, displayMinY, scale, offsetX, offsetY, scaledDisplayWidth, scaledDisplayHeight };
}

// 屏幕局部坐标 → 图片坐标（metrics 来自 computeDisplayMetrics）
export function localToImage(localX, localY, m) {
  return {
    x: (localX - m.offsetX) / m.scale + m.displayMinX,
    y: (localY - m.offsetY) / m.scale + m.displayMinY,
  };
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
