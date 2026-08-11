"""图片切块（Tile）纯逻辑：行/列/重叠 → 每块矩形坐标。

SFImageTile 与 SFImageUntile 共用本模块的几何计算，保证两侧契约一致
（移植自 cubiq/ComfyUI_essentials image.py 的 ImageTile / ImageUntile）。
无 torch / ComfyUI 依赖，可独立测试。

- resolve_overlap: FLOAT 比例 + INT 像素相加出实际重叠量，钳制到块尺寸一半，
  单行/列对应方向强制清零
- tile_rects:      行优先返回每块矩形（Tile 与 Untile 共用；Untile 传入
  反推出的净块尺寸与画布尺寸）
- tile_plan:       Tile 侧完整规划（净块尺寸 + 实际重叠 + 完整块尺寸 + 矩形列表）
"""


def resolve_overlap(tile_h, tile_w, rows, cols, overlap, overlap_x, overlap_y):
    """计算实际使用的重叠量（像素）。返回 (overlap_h, overlap_w)。

    规则与原版一致：
      - overlap_h = int(tile_h * overlap) + overlap_y，钳制到 tile_h // 2
      - rows == 1 时垂直重叠无意义，强制为 0（cols == 1 同理水平）
    """
    overlap_h = int(tile_h * overlap) + overlap_y
    overlap_w = int(tile_w * overlap) + overlap_x
    overlap_h = min(tile_h // 2, overlap_h)
    overlap_w = min(tile_w // 2, overlap_w)
    if rows == 1:
        overlap_h = 0
    if cols == 1:
        overlap_w = 0
    return overlap_h, overlap_w


def tile_rects(rows, cols, cell_h, cell_w, overlap_h, overlap_w, out_h, out_w):
    """行优先（先 i 后 j）返回每块矩形 [(y1, x1, y2, x2), ...]。

    cell_h/cell_w: 净块尺寸（不含重叠）；overlap_h/overlap_w: 实际重叠量；
    out_h/out_w: 输出画布尺寸（末尾越界时钳制回画布，块保持完整尺寸）。
    """
    rects = []
    for i in range(rows):
        for j in range(cols):
            y1 = i * cell_h
            x1 = j * cell_w
            if i > 0:
                y1 -= overlap_h
            if j > 0:
                x1 -= overlap_w
            y2 = y1 + cell_h + overlap_h
            x2 = x1 + cell_w + overlap_w
            if y2 > out_h:
                y2 = out_h
                y1 = y2 - cell_h - overlap_h
            if x2 > out_w:
                x2 = out_w
                x1 = x2 - cell_w - overlap_w
            rects.append((y1, x1, y2, x2))
    return rects


def tile_plan(h, w, rows, cols, overlap=0.0, overlap_x=0, overlap_y=0):
    """Tile 侧完整规划。返回 dict：

      tile_h / tile_w       净块尺寸（h//rows、w//cols 向下取整，不可整除部分丢弃）
      overlap_h / overlap_w 实际重叠量（已钳制；Untile 必须使用这些值还原）
      tile_height/width     含重叠的完整块尺寸（Tile 输出的 tile_width/height）
      rects                 行优先的矩形列表
    """
    tile_h = h // rows
    tile_w = w // cols
    out_h = tile_h * rows
    out_w = tile_w * cols
    overlap_h, overlap_w = resolve_overlap(
        tile_h, tile_w, rows, cols, overlap, overlap_x, overlap_y
    )
    rects = tile_rects(
        rows, cols, tile_h, tile_w, overlap_h, overlap_w, out_h, out_w
    )
    return {
        "tile_h": tile_h,
        "tile_w": tile_w,
        "overlap_h": overlap_h,
        "overlap_w": overlap_w,
        "tile_height": tile_h + overlap_h,
        "tile_width": tile_w + overlap_w,
        "rects": rects,
    }
