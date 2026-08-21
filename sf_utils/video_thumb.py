"""视频首帧提取纯逻辑（SFLoraStack Civitai 缩略 + Sample 缩略共用）。

仅标准库 + 可选 cv2 / PIL，无 comfy / folder_paths。
任何失败返回 None，永不抛错，可在测试环境直接跑。
"""
import os
import tempfile


def _is_video_bytes(raw):
    """按 magic 粗判视频（mp4/mov 的 ftyp，webm/mkv 的 EBML）。"""
    if not raw or len(raw) < 12:
        return False
    if raw[4:8] == b"ftyp":
        return True
    if raw[:4] == b"\x1a\x45\xdf\xa3":
        return True
    return False


def _ext_for_video_bytes(raw):
    """视频字节 -> 后缀，供临时文件命名。"""
    if not raw:
        return ".mp4"
    if raw[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    # ftyp / 未知一律 mp4（cv2 按内容而非后缀解码，后缀仅辅助）
    return ".mp4"


def extract_first_frame_from_bytes(raw, quality=92):
    """视频字节 -> 首帧 jpeg 字节，失败返回 None。永不抛错。

    quality: jpeg 质量 1-100。
    依赖 cv2（opencv-contrib-python，已在 requirements.txt），缺库/解码失败返回 None。
    """
    if not raw or not isinstance(raw, (bytes, bytearray)):
        return None
    # 非视频直接拒绝（调用方已判，这里双保险）
    if not _is_video_bytes(raw):
        return None
    try:
        import cv2  # 延迟导入，缺库回退
    except Exception:
        return None
    ext = _ext_for_video_bytes(raw)
    tmp = None
    try:
        # NamedTemporaryFile 在 Windows 上被占用无法二次打开，用 mkstemp
        fd, tmp = tempfile.mkstemp(suffix=ext)
        try:
            os.write(fd, bytes(raw))
        finally:
            try:
                os.close(fd)
            except Exception:
                pass
        cap = None
        try:
            cap = cv2.VideoCapture(tmp)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            # cv2 默认 BGR，imencode 按 BGR 存 jpeg 即正确色彩
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok2 or buf is None:
                return None
            return bytes(buf.tobytes())
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    except Exception:
        return None
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass


def extract_first_frame_from_path(path, quality=92):
    """视频文件路径 -> 首帧 jpeg 字节，失败返回 None。永不抛错。"""
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        return None
    try:
        import cv2
    except Exception:
        return None
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok2 or buf is None:
            return None
        return bytes(buf.tobytes())
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
