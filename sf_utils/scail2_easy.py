"""SF SCAIL-2 纯逻辑工具（无 torch / ComfyUI 依赖，可 mock 直测）。

从 ComfyUI-SCAIL2-Easy 的 nodes.py 抽出：常量表、帧数/尺寸数学、多主体拼图
布局算法。节点实现在 nodes/video/scail2.py，编排与张量运算留在节点文件。
"""

RESOLUTION_PRESETS = ("512p", "704p", "custom")
LONG_VIDEO_MODES = ("chunk", "context_sampling")
CONTEXT_SCHEDULES = ("standard_static", "standard_uniform", "looped_uniform", "batched")
REFERENCE_PACK_TYPE = "SCAIL2_REFERENCE_PACK"
MAX_REFERENCE_SUBJECTS = 6
MAX_LEGACY_REFERENCE_IMAGES_PER_SUBJECT = 6
MAX_PREFIX_REFERENCE_IMAGES = 5
MAX_MIXED_REFERENCE_IMAGES = 5
MAX_STAGE_REFERENCE_SOURCE_HEIGHT = 2048
SCAIL_COLOR_PALETTE = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
)


def clamp_int(value, low, high):
    return max(low, min(high, int(value)))


def is_reference_pack(value):
    return isinstance(value, dict) and value.get("type") == REFERENCE_PACK_TYPE


def round_32(value):
    return max(32, (int(value) // 32) * 32)


def round_nearest_32(value):
    return max(32, int((float(value) + 16) // 32) * 32)


def wan_frame_count_cover(value):
    value = max(1, int(value))
    if value == 1:
        return 1
    return 1 + ((value - 1 + 3) // 4) * 4


def wan_frame_count_floor(value):
    value = max(1, int(value))
    return 1 + ((value - 1) // 4) * 4


def infer_generation_size(height, width):
    """按源图高宽推断 SCAIL-2 生成尺寸（各维对齐 32，向下取整）。"""
    return round_32(int(width)), round_32(int(height))


def target_size_for_video(height, width, resolution, custom_width=832, custom_height=480):
    """复刻 Fit Video 尺寸策略：短边缩放到 512/704（长边按比例），custom 直接取参。"""
    height = int(height)
    width = int(width)
    if resolution == "custom":
        return round_nearest_32(max(32, int(custom_width))), round_nearest_32(max(32, int(custom_height)))

    target_short = 704 if resolution == "704p" else 512
    source_short = max(1, min(width, height))
    scale = target_short / source_short

    target_width = round_nearest_32(width * scale)
    target_height = round_nearest_32(height * scale)
    return target_width, target_height


def subject_color(subject_index):
    return SCAIL_COLOR_PALETTE[int(subject_index) % len(SCAIL_COLOR_PALETTE)]


def reference_canvas_aspect(width, height):
    ratio = float(width) / max(1.0, float(height))
    if ratio < 0.84:
        return "portrait"
    if ratio > 1.18:
        return "landscape"
    return "square"


def visual_stage_rows(count, row_count, max_per_row):
    count = max(1, int(count))
    row_count = max(1, int(row_count))
    max_per_row = max(1, int(max_per_row))
    rows = [[] for _ in range(row_count)]
    for index in range(count):
        rows[index % row_count].append(index)
    if any(len(row) == 0 or len(row) > max_per_row for row in rows):
        return None
    return rows


def stage_row_candidates(count, aspect):
    count = max(1, int(count))
    if count <= 1:
        return [[[0]]]

    if aspect == "portrait":
        if count == 3:
            return [[[0, 2], [1]]]
        max_rows = 1 if count <= 2 else (3 if count >= 5 else 2)
        max_per_row = 3
    elif aspect == "square":
        max_rows = 1 if count <= 3 else (3 if count >= 5 else 2)
        max_per_row = 4
    else:
        max_rows = 1 if count <= 4 else 2
        max_per_row = 6

    candidates = []
    for row_count in range(1, min(max_rows, count) + 1):
        rows = visual_stage_rows(count, row_count, max_per_row)
        if rows is not None:
            candidates.append(rows)
    return candidates or [[list(range(count))]]


def stage_row_profile(row_count, row_index, aspect):
    if row_count <= 1:
        max_h = 0.96 if aspect != "square" else 0.92
        return {"base": 0.985, "max_h": max_h, "row_w": 0.94, "overlap": 0.10}

    if row_count == 2 and aspect == "portrait":
        profiles = [
            {"base": 0.74, "max_h": 0.78, "row_w": 1.04, "overlap": 0.24},
            {"base": 1.02, "max_h": 0.92, "row_w": 0.96, "overlap": 0.16},
        ]
    elif row_count == 2 and aspect == "square":
        profiles = [
            {"base": 0.78, "max_h": 0.66, "row_w": 0.92, "overlap": 0.14},
            {"base": 0.995, "max_h": 0.74, "row_w": 0.88, "overlap": 0.12},
        ]
    elif row_count == 2:
        profiles = [
            {"base": 0.74, "max_h": 0.58, "row_w": 0.94, "overlap": 0.14},
            {"base": 0.99, "max_h": 0.66, "row_w": 0.90, "overlap": 0.12},
        ]
    elif aspect == "portrait":
        profiles = [
            {"base": 0.56, "max_h": 0.48, "row_w": 0.92, "overlap": 0.16},
            {"base": 0.78, "max_h": 0.56, "row_w": 0.94, "overlap": 0.15},
            {"base": 1.02, "max_h": 0.64, "row_w": 0.90, "overlap": 0.13},
        ]
    elif aspect == "square":
        profiles = [
            {"base": 0.54, "max_h": 0.44, "row_w": 0.90, "overlap": 0.16},
            {"base": 0.76, "max_h": 0.52, "row_w": 0.92, "overlap": 0.15},
            {"base": 0.99, "max_h": 0.60, "row_w": 0.88, "overlap": 0.13},
        ]
    else:
        profiles = [
            {"base": 0.50, "max_h": 0.40, "row_w": 0.94, "overlap": 0.16},
            {"base": 0.73, "max_h": 0.48, "row_w": 0.94, "overlap": 0.15},
            {"base": 0.99, "max_h": 0.56, "row_w": 0.90, "overlap": 0.13},
        ]
    return profiles[min(int(row_index), len(profiles) - 1)]


def effective_row_width(widths, overlap):
    if not widths:
        return 1.0
    effective = float(sum(widths))
    for index in range(1, len(widths)):
        effective -= min(widths[index - 1], widths[index]) * float(overlap)
    return max(1.0, effective)


def stage_layout_from_rows(entries, rows, canvas_w, canvas_h, aspect):
    fit_scales = []
    row_profiles = []
    for row_index, row in enumerate(rows):
        profile = stage_row_profile(len(rows), row_index, aspect)
        row_profiles.append(profile)
        row_widths = [entries[index]["metrics"]["crop_w"] for index in row]
        fit_scales.append((float(canvas_w) * float(profile["row_w"])) / effective_row_width(row_widths, profile["overlap"]))
        for index in row:
            metrics = entries[index]["metrics"]
            fit_scales.append((float(canvas_h) * float(profile["max_h"])) / max(1.0, metrics["crop_h"]))

    group_scale = max(0.001, min(fit_scales) if fit_scales else 1.0)
    specs = [None for _ in entries]
    for row_index, row in enumerate(rows):
        profile = row_profiles[row_index]
        overlap = float(profile["overlap"])
        scaled_widths = [entries[index]["metrics"]["crop_w"] * group_scale for index in row]
        total_width = effective_row_width(scaled_widths, overlap)
        left = (float(canvas_w) - total_width) * 0.5
        for local_index, index in enumerate(row):
            width = scaled_widths[local_index]
            center_x = (left + width * 0.5) / max(1.0, float(canvas_w))
            specs[index] = {
                "x": max(0.0, min(1.0, center_x)),
                "base": float(profile["base"]),
                "max_w": 1.0,
                "max_h": 1.0,
                "row_index": float(row_index),
                "row_count": float(len(rows)),
                "row_w": float(profile["row_w"]),
                "row_max_h": float(profile["max_h"]),
                "overlap": overlap,
            }
            if local_index + 1 < len(row):
                next_width = scaled_widths[local_index + 1]
                left += width - min(width, next_width) * overlap

    row_lengths = [len(row) for row in rows]
    balance_penalty = (max(row_lengths) - min(row_lengths)) * 0.002 if row_lengths else 0.0
    row_penalty = max(0, len(rows) - 1) * 0.012
    layout_score = group_scale * max(0.90, 1.0 - row_penalty) - balance_penalty
    return {
        "specs": [spec if spec is not None else {"x": 0.5, "base": 0.985, "max_w": 1.0, "max_h": 1.0} for spec in specs],
        "scales": [group_scale for _ in entries],
        "rows": rows,
        "group_scale": group_scale,
        "score": layout_score,
    }


def layout_stage_entries(entries, canvas_w, canvas_h):
    count = len(entries)
    if count <= 0:
        return [], []
    if count == 1:
        return [{"x": 0.5, "base": 0.985, "max_w": 0.86, "max_h": 0.96, "row_index": 0.0}], [None]

    aspect = reference_canvas_aspect(canvas_w, canvas_h)
    candidates = stage_row_candidates(count, aspect)
    best = None
    for rows in candidates:
        layout = stage_layout_from_rows(entries, rows, canvas_w, canvas_h, aspect)
        if best is None or (layout["score"], layout["group_scale"], -len(layout["rows"])) > (
            best["score"],
            best["group_scale"],
            -len(best["rows"]),
        ):
            best = layout

    if best is None:
        return [{"x": 0.5, "base": 0.985, "max_w": 1.0, "max_h": 1.0} for _ in entries], [None for _ in entries]
    return best["specs"], best["scales"]
