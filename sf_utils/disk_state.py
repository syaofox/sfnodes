"""共享的磁盘状态工具：input/<子目录>/ 下的项目文件安全读取与解码。

crop.py 与 inpaint_editor.py 曾各自内联一份（_safe_join 语义分叉正是
SFImageCrop 粘贴上传输出白图 bug 的温床），现收敛为单一实现。
纯函数，无 ComfyUI 依赖（PIL 属运行时依赖），可独立测试。
"""
import base64
import io
import os
import re

from PIL import Image


def safe_join(root_dir: str, rel, strip_prefix: str = None) -> str:
    """把保存的相对路径解析为 root_dir 下的绝对路径，越界或不存在返回 None。

    strip_prefix: 上传/保存路由返回的 path 是 "subdir/<file>"（ComfyUI
    惯例 subfolder/filename）；当解析根是子目录本身时剥掉该前缀，否则会
    双重拼接（input/<subdir>/<subdir>/... 文件不存在）。

    词法层面先拒绝绝对路径 / 盘符 / UNC 值（UNC 路径仅解析就会打开 SMB
    连接），再 realpath + startswith 包含性检查。"""
    if not rel or not isinstance(rel, str):
        return None
    q = rel.strip().strip('"').strip("'")
    if not q:
        return None
    if q.replace("/", "\\").startswith("\\\\"):
        return None
    try:
        if os.path.splitdrive(q)[0]:
            return None
        if os.path.isabs(q):
            return None
    except (ValueError, TypeError):
        return None
    if strip_prefix:
        for _prefix in (strip_prefix + "/", strip_prefix + "\\", "./"):
            if q.startswith(_prefix):
                q = q[len(_prefix):]
                break
    root = os.path.realpath(root_dir)
    try:
        full = os.path.realpath(os.path.join(root, q))
    except (OSError, ValueError, TypeError):
        return None
    if not full.startswith(root + os.sep):
        return None
    if not os.path.exists(full):
        return None
    return full


def sanitize_id(raw, fallback: str) -> str:
    """仅保留单词字符 / 连字符，构造的 project_id 无法夹带路径分隔符。"""
    s = str(raw or "")
    s = re.sub(r"[^A-Za-z0-9_-]", "", s)
    return s[:64] or fallback


_ILLEGAL_FN_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_FN_MAX_LEN = 128
_WIN_RESERVED_NAMES = frozenset((
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
))


def sanitize_filename(raw, fallback: str = "file") -> str:
    """把任意用户输入净化成安全的单段文件名（保留 Unicode，拒绝路径逃逸）。

    先检查 leading '/' 与 '..'/'.' 段（在任何清洗之前——清洗会把 '..' 吃掉，
    让路径穿越检查失效）；再把路径分隔符与 Windows 非法字符替换为 '_'、
    剥离边沿空白/点、拒绝隐藏文件、保留设备名加 '_' 后缀、截断限长。
    不可恢复时返回 fallback（调用方兜底默认名）。"""
    if not isinstance(raw, str):
        return fallback
    s = raw.strip().replace("\\", "/")
    if not s or s.startswith("/"):
        return fallback
    parts = s.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return fallback
    cleaned = _ILLEGAL_FN_CHARS.sub("_", s)
    # 循环到稳定：边沿空白、边沿下划线、尾点会互相遮蔽
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = cleaned.strip().strip("_").rstrip(". ")
    if not cleaned or cleaned.startswith("."):
        return fallback
    if cleaned.split(".", 1)[0].upper() in _WIN_RESERVED_NAMES:
        cleaned += "_"
    return cleaned[:_FN_MAX_LEN]


def decode_image(b64: str):
    """把 dataURL（或裸 base64）解码为 PIL Image，失败返回 None。"""
    if not isinstance(b64, str) or not b64:
        return None
    try:
        payload = b64.split(",", 1)[-1] if "," in b64 else b64
        raw = base64.b64decode(payload)
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception:
        return None
