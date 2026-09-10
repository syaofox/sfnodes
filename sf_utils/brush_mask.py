"""SF brush mask 纯逻辑：画笔串解析 + 栅格化为遮罩数组。

移植自 ComfyUI-YCNodes_Toolkit ``py/Loadimage_brushmask.py`` 的绘制语义
（圆形印章 brush 置 1 / erase 置 0、线段按印章步进、mask 恒二值——前端
Opacity/颜色仅预览语义，后端忽略），抽为无 torch/ComfyUI 依赖的纯函数
（仅 numpy），供 ``nodes/image/brush_mask.py`` 与测试共用。

串格式（与原版双向兼容）：多笔以 ``|`` 分隔，每笔为 ::

    mode:size:opacity:r,g,b:points   （新格式，r,g,b 仅解析不使用）
    mode:size:opacity:points          （旧格式，无颜色）
    mode:points                      （旧格式，无尺寸）
    x1,y1;x2,y2;...                  （裸点列，默认 brush）

其中 ``points`` 为 ``x,y`` 以 ``;`` 分隔。坐标以源图像素为单位，
越界点丢弃（与原版一致）。

结构化 state 笔触另有第三种模式 ``fill``（SAM 等算法写入）：``points`` 为
闭合多边形顶点（像素整数），后端整体填充、前端整体填充绘制——与
brush/erase 同一列表统一管理（添加/擦除/撤销全通用，见 §45.9）。
"""

import numpy as np


def _parse_one_stroke(stroke, default_size=80):
    """Parse a single stroke string into a dict.

    Returns ``{"mode": "brush"|"erase", "size": int, "points": [(x, y)]}``
    with out-of-range filtering left to the caller (needs image dims).
    """
    mode = "brush"
    size = int(default_size) if default_size else 80
    points_str = stroke
    try:
        if ":" in stroke:
            parts = stroke.split(":")
            if parts[0] in ("brush", "erase"):
                mode = parts[0]
                if len(parts) >= 4:
                    part3 = parts[3]
                    if part3 and "," in part3:
                        color_parts = part3.split(",")
                        if len(color_parts) == 3:
                            try:
                                r = int(float(color_parts[0]))
                                g = int(float(color_parts[1]))
                                b = int(float(color_parts[2]))
                            except (ValueError, IndexError):
                                r = g = b = -1
                            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                                try:
                                    size = int(float(parts[1]))
                                except (ValueError, IndexError):
                                    pass
                                # opacity 解析后丢弃（预览语义，后端二值）
                                points_str = ":".join(parts[4:])
                            else:
                                try:
                                    size = int(float(parts[1]))
                                except (ValueError, IndexError):
                                    pass
                                points_str = ":".join(parts[3:])
                        else:
                            try:
                                size = int(float(parts[1]))
                            except (ValueError, IndexError):
                                pass
                            points_str = ":".join(parts[3:])
                    else:
                        try:
                            size = int(float(parts[1]))
                        except (ValueError, IndexError):
                            pass
                        points_str = ":".join(parts[3:])
                else:
                    points_str = ":".join(parts[1:])
    except Exception:
        pass
    if size is None or not isinstance(size, int):
        try:
            size = int(float(size))
        except Exception:
            size = int(default_size) if default_size else 80
    size = max(1, size)

    points = []
    for point_str in points_str.split(";"):
        if not point_str.strip():
            continue
        try:
            coords = point_str.split(",", 1)
            if len(coords) == 2:
                points.append((float(coords[0]), float(coords[1])))
        except (ValueError, IndexError):
            continue
    return {"mode": mode, "size": size, "points": points}


def parse_strokes(brush_data, width, height, default_size=80):
    """Parse a full brush_data string into clipped strokes.

    越界点丢弃（原版语义：``0 <= x < width`` 且 ``0 <= y < height`` 才保留，
    取整方式与原版一致 ``int(float(v))``）。空输入返回 ``[]``。
    """
    if not isinstance(brush_data, str) or not brush_data.strip():
        return []
    try:
        w = int(width)
        h = int(height)
    except Exception:
        return []
    if w <= 0 or h <= 0:
        return []
    strokes = []
    for raw in brush_data.split("|"):
        if not raw.strip():
            continue
        parsed = _parse_one_stroke(raw, default_size)
        pts = []
        for fx, fy in parsed["points"]:
            try:
                x = int(fx)
                y = int(fy)
            except Exception:
                continue
            if 0 <= x < w and 0 <= y < h:
                pts.append((x, y))
        if pts:
            strokes.append({"mode": parsed["mode"], "size": parsed["size"], "points": pts})
    return strokes


def parse_state_strokes(state, default_size=80):
    """Parse the ``strokes`` list of the hidden JSON state.

    State strokes are ``{"mode", "size", "points": [[x, y], ...]}`` (already
    numeric, from the frontend). Clips to ``(src_w, src_h)`` and drops empty
    strokes so backend/test share one code path with the string format.
    """
    if not isinstance(state, dict):
        return []
    try:
        w = int(state.get("src_w", 0))
        h = int(state.get("src_h", 0))
    except Exception:
        return []
    if w <= 0 or h <= 0:
        return []
    try:
        size_default = int(state.get("brush_size", default_size) or default_size)
    except Exception:
        size_default = int(default_size) if default_size else 80
    out = []
    raw_strokes = state.get("strokes", [])
    if not isinstance(raw_strokes, list):
        return []
    for item in raw_strokes:
        if not isinstance(item, dict):
            continue
        mode = item.get("mode", "brush")
        if mode not in ("brush", "erase", "fill"):
            mode = "brush"
        try:
            size = int(float(item.get("size", size_default)))
        except Exception:
            size = size_default
        size = max(1, size)
        pts = []
        for pt in item.get("points", []) or []:
            try:
                x = int(float(pt[0]))
                y = int(float(pt[1]))
            except Exception:
                continue
            if 0 <= x < w and 0 <= y < h:
                pts.append((x, y))
        if pts:
            out.append({"mode": mode, "size": size, "points": pts})
    return out


def build_brush_data(strokes):
    """Serialize strokes back to the legacy brush_data string (opacity=1.0).

    前端以结构化 strokes 为真源；本函数仅供测试往返断言与旧格式兼容。
    """
    parts = []
    for st in strokes:
        pts = ";".join(f"{x},{y}" for x, y in st.get("points", []))
        parts.append(f"{st.get('mode', 'brush')}:{int(st.get('size', 80))}:1.0:{pts}")
    return "|".join(parts)


def _fill_polygon(mask, pts):
    """Fill a closed polygon with 1 (PIL, no cv2 needed — locally testable)."""
    from PIL import Image as _PILImage
    from PIL import ImageDraw as _ImageDraw
    h, w = mask.shape
    img = _PILImage.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
    _ImageDraw.Draw(img).polygon([(int(x), int(y)) for x, y in pts], fill=255)
    filled = np.array(img).astype(np.float32) / 255.0
    mask[:, :] = np.maximum(mask, filled)


def mask_to_fill_strokes(mask_arr, eps=1.5, min_area=16.0, max_contours=64):
    """Trace a binary (H, W) float mask into fill strokes (one per contour).

    每个轮廓独立一笔（Undo 以物体为粒度）。小碎斑（面积 < min_area）丢弃，
    只保留最大的 max_contours 个。顶点为像素整数。cv2 缺席时返回 []。
    """
    try:
        import cv2 as _cv2
    except Exception:
        return []
    m = np.asarray(mask_arr)
    if m.ndim != 2 or m.shape[0] <= 0 or m.shape[1] <= 0:
        return []
    bw = ((m > 0.5).astype(np.uint8)) * 255
    if not bw.any():
        return []
    contours, _ = _cv2.findContours(bw, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for cnt in contours:
        try:
            area = float(_cv2.contourArea(cnt))
        except Exception:
            continue
        if area < min_area:
            continue
        approx = _cv2.approxPolyDP(cnt, eps, True)
        poly = approx.reshape(-1, 2)
        if len(poly) < 3:
            continue
        items.append((area, [[int(x), int(y)] for x, y in poly]))
    items.sort(key=lambda t: t[0], reverse=True)
    items = items[:max(1, int(max_contours))]
    return [{"mode": "fill", "size": 0, "points": pts} for _, pts in items]


def _draw_circle(mask, x, y, radius):
    """Stamp a binary disc (brush → max 1). Vectorized, ported from YCNodes."""
    h, w = mask.shape
    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)
    if x_max <= x_min or y_max <= y_min:
        return
    y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
    dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
    mask[y_min:y_max, x_min:x_max] = np.maximum(
        mask[y_min:y_max, x_min:x_max],
        (dist_sq <= radius * radius).astype(np.float32),
    )


def _erase_circle(mask, x, y, radius):
    """Stamp a binary disc (erase → 0). Vectorized, ported from YCNodes."""
    h, w = mask.shape
    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)
    if x_max <= x_min or y_max <= y_min:
        return
    y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
    dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
    erase = dist_sq <= radius * radius
    mask[y_min:y_max, x_min:x_max] = np.where(
        erase, 0.0, mask[y_min:y_max, x_min:x_max]
    )


def _stamp_line(mask, x1, y1, x2, y2, radius, erase):
    """Walk a segment stamping discs (dedup via unique, ported from YCNodes)."""
    circle = _erase_circle if erase else _draw_circle
    if x1 == x2 and y1 == y2:
        circle(mask, x1, y1, radius)
        return
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.sqrt(dx * dx + dy * dy))
    step_size = max(1, radius // 3) if radius > 10 else 1
    steps = max(1, int(length / step_size) + 1)
    if steps <= 0:
        circle(mask, x1, y1, radius)
        return
    t_values = np.linspace(0, 1, steps + 1)
    x_coords = (x1 + dx * t_values).astype(np.int32)
    y_coords = (y1 + dy * t_values).astype(np.int32)
    h, w = mask.shape
    valid = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
    x_coords = x_coords[valid]
    y_coords = y_coords[valid]
    if len(x_coords) == 0:
        return
    unique = np.unique(np.column_stack((y_coords, x_coords)), axis=0)
    for y, x in unique:
        circle(mask, int(x), int(y), radius)


def rasterize_strokes(strokes, width, height):
    """Rasterize parsed strokes into a (H, W) float32 mask (0..1, binary).

    brush 笔画按自身 size 置 1（含线段插值），erase 笔画置 0。空笔画
    返回全黑遮罩。
    """
    try:
        w = int(width)
        h = int(height)
    except Exception:
        return np.zeros((1, 1), dtype=np.float32)
    if w <= 0 or h <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    for st in strokes or []:
        pts = st.get("points", []) or []
        if not pts:
            continue
        # fill 笔触：整体填充多边形（size 无意义，SAM 写入 0）
        if st.get("mode") == "fill":
            if len(pts) >= 3:
                _fill_polygon(mask, pts)
            continue
        try:
            radius = max(1, int(st.get("size", 80)) // 2)
        except Exception:
            radius = 40
        erase = st.get("mode") == "erase"
        for i, (x, y) in enumerate(pts):
            if i > 0:
                px, py = pts[i - 1]
                _stamp_line(mask, int(px), int(py), int(x), int(y), radius, erase)
            else:
                if erase:
                    _erase_circle(mask, int(x), int(y), radius)
                else:
                    _draw_circle(mask, int(x), int(y), radius)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)
