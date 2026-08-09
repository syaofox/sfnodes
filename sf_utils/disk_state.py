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
