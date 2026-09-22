"""SFVideoCompare 纯逻辑与预览探针：temp 预览命名 + av 元数据读取 + ui 条目拼装。

TE_MAN「TE MAN 视频对比」的干净室复刻配套（能力自 README 与其 .pyd 字符串表
还原，实现自写）：后端把两路 VIDEO 各落盘 temp mp4（H264，浏览器可播；源为
h264/mp4 时 save_to 内部纯 remux 不重编码），再用 av 读帧数/帧率/时长，供前端
进度条与「同步帧」使用。

无 ComfyUI 依赖（av 为 ComfyUI 运行时自带第三方库），可直接单测（测试以桩 av
注入 sys.modules）。
"""

import os
import uuid

PREVIEW_PREFIX = "sf_video_compare"
PREVIEW_EXT = ".mp4"


def preview_filename():
    """temp 预览文件名（每次执行唯一，避免浏览器缓存旧内容）。"""
    return f"{PREVIEW_PREFIX}_{uuid.uuid4().hex}{PREVIEW_EXT}"


def fallback_frame_count(frame_count, frame_rate, duration):
    """容器未写帧数时用 时长×帧率 估算（mp4 正常有 stsz 帧数，webm 常缺）。"""
    if frame_count > 0 or frame_rate <= 0 or duration <= 0:
        return frame_count
    return int(round(duration * frame_rate))


def probe_video_meta(path):
    """读视频元数据 {frame_count, frame_rate, duration}；无视频流返回全 0。"""
    import av

    with av.open(path, mode="r") as container:
        streams = getattr(container.streams, "video", None) or []
        if not streams:
            return {"frame_count": 0, "frame_rate": 0.0, "duration": 0.0}
        stream = streams[0]
        rate = getattr(stream, "average_rate", None)
        frame_rate = float(rate) if rate else 0.0
        duration = 0.0
        if getattr(stream, "duration", None) and getattr(stream, "time_base", None):
            duration = float(stream.duration * stream.time_base)
        elif getattr(container, "duration", None):
            duration = float(container.duration / av.time_base)
        frame_count = fallback_frame_count(
            int(getattr(stream, "frames", 0) or 0), frame_rate, duration
        )
        return {
            "frame_count": frame_count,
            "frame_rate": frame_rate,
            "duration": duration,
        }


def build_entry(filename, subfolder, folder_type, meta):
    """拼装前端 /view 恢复条目（结构部分 + 同步元数据，字段类型归一）。"""
    meta = meta or {}
    return {
        "filename": filename,
        "subfolder": subfolder or "",
        "type": folder_type or "temp",
        "frame_count": int(meta.get("frame_count", 0) or 0),
        "frame_rate": float(meta.get("frame_rate", 0.0) or 0.0),
        "duration": float(meta.get("duration", 0.0) or 0.0),
    }


def preview_path(temp_dir, filename):
    """预览文件绝对路径（temp 根目录，subfolder 恒空）。"""
    return os.path.join(temp_dir, filename)
