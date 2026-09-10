// ==========================================================================
// sf_brush_mask_lib.js - SF Image Brush Mask 纯几何/纯逻辑库
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_brush_mask.js 使用，也供 tests/ 复制为 .mjs 直接测试。
// 复刻 ComfyUI-YCNodes_Toolkit js/Loadimage_brushmask.*.js 的面板/绘制数学，
// 并按 SFImageCropExpand（sf_crop_expand_lib.js §37/§44）收敛三处修复：
// 最小尺寸钳制（computeSize 包装一处生效）、右下角 cursor 命中判定、
// 显示坐标系与图片坐标系互算。
// ==========================================================================

// 顶部控制面板（按钮行 + 滑块行）与底部信息行高度。面板内容：
// 行1 Load/Browse/Clear/Undo/Eraser + 双色块，行2 Size/Opacity 滑块。
export const LAYOUT = { shiftLeft: 10, shiftRight: 80, panelH: 58, bottomH: 22 };

// 节点最小宽高（控件不溢出前提下的下限）：
// - 宽度：行1 五按钮（Load 56/Browse 52/Clear 44/Undo 44/Eraser 52 + 间隙）
//   + 右侧色块列 40 ≈ 300，面板可用宽 = W-82 → W≥400 取整 420；
// - 高度：panelH 58 + bottomH 22 + 上下边距 20 + 画布最小 200 → 300，取 320。
// 双端拖拽 resize 的最小值都取自 node.computeSize()（前端包实测 onDrag 里
// clamp 到 computeSize）——主扩展包装 computeSize 返回 ensureMinSize 结果
// 钳住拖拽；创建/恢复两处 clampNodeSize 兜底。
export const MIN_NODE_WIDTH = 420;
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
// state: {srcW, srcH}；显示区为节点内让出顶面板与底信息行后的矩形，
// 源图按 min(scaleX, scaleY) 等比居中。
export function computeDisplayMetrics(state, nodeW, nodeH) {
  const { shiftLeft, shiftRight, panelH, bottomH } = LAYOUT;
  const areaW = Math.max(1, nodeW - shiftRight - shiftLeft);
  const areaH = Math.max(1, nodeH - shiftLeft - panelH - bottomH - shiftLeft);
  const srcW = Math.max(1, state.srcW || 512);
  const srcH = Math.max(1, state.srcH || 512);
  const scale = Math.min(areaW / srcW, areaH / srcH);
  const scaledW = srcW * scale;
  const scaledH = srcH * scale;
  const offsetX = shiftLeft + (areaW - scaledW) / 2;
  const offsetY = shiftLeft + panelH + (areaH - scaledH) / 2;
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
