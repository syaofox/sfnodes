"""SFVideoConcat 纯逻辑：从混合结构解析段视频文件路径。

循环分段落盘时 `VHS_VideoCombine` 每轮返回 `VHS_FILENAMES`（Python 值
`(save_output: bool, [path, ...])`，其中含首帧元数据 PNG，接音频时还可能
有无音轨中间 mp4 与 `-audio.mp4` 终版）；循环状态经 `SFBatchAnything`
累加后可能是 list/tuple/str 的混合拼接结构（tuple + tuple 直接拼接）。

本模块递归提取全部段视频路径：保序去重、跳过元数据 PNG 与被 `-audio`
终版覆盖的中间视频。无 torch / ComfyUI 依赖，可直接单测。
"""

_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")


def _walk(value, depth, out):
    if depth > 8:
        return
    if isinstance(value, str):
        out.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _walk(item, depth + 1, out)


def _is_video_path(path):
    return path.lower().endswith(_VIDEO_EXTENSIONS)


def _audio_override_name(path):
    """返回同段带音轨终版文件名（`X.mp4` → `X-audio.mp4`），非视频返回 None。"""
    for ext in _VIDEO_EXTENSIONS:
        if path.lower().endswith(ext):
            return path[: -len(ext)] + "-audio" + path[-len(ext):]
    return None


def collect_segment_paths(value):
    """递归提取段视频文件路径（保序去重，跳过 PNG 与被 `-audio` 覆盖的中间视频）。"""
    raw = []
    _walk(value, 0, raw)
    candidates = [p for p in raw if p and _is_video_path(p)]
    if not candidates:
        return []
    present_lower = {p.lower() for p in candidates}
    out = []
    for path in dict.fromkeys(candidates):
        override = _audio_override_name(path)
        if override is not None and override.lower() in present_lower:
            continue
        out.append(path)
    return out
