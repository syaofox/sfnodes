"""共享的磁盘状态工具：input/<子目录>/ 下的项目文件安全读取与解码。

crop.py 与 inpaint_editor.py 曾各自内联一份（_safe_join 语义分叉正是
SFImageCrop 粘贴上传输出白图 bug 的温床），现收敛为单一实现。
纯函数，无 ComfyUI 依赖（PIL 属运行时依赖），可独立测试。
"""
import base64
import io
import json
import os
import re
import threading

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


_PREFIX_MAX_LEN = 256    # 文件名前缀输入上限（尽早拒绝明显的垃圾）
_PREFIX_OUTPUT_MAX = 100  # 清洗后输出上限

_SEGMENT_ILLEGAL_RE = re.compile(r'[<>:"|?*\x00-\x1f\x7f]')
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def sanitize_segment(seg):
    """把单段文件名中的 Windows 非法字符换成 '_'，整理边沿，守卫保留设备名。

    调用方须在调用前拒绝 '..'。全部不可用时返回 ""。尾点/尾空格剥离——
    Windows 创建时本就静默剥离，这里剥离让报告路径与磁盘实际落盘一致。
    （preview_routes / save_image_exact 曾各持一份，现收敛。）"""
    cleaned = _SEGMENT_ILLEGAL_RE.sub("_", seg)
    cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned)
    # 循环到稳定：边沿空白、边沿下划线、尾点/空格会互相遮蔽
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = cleaned.strip().strip("_").rstrip(". ")
    if cleaned and cleaned.split(".", 1)[0].upper() in _WIN_RESERVED_NAMES:
        cleaned += "_"
    return cleaned


def safe_prefix(raw):
    """清洗可含子目录的文件名前缀；不可恢复时返回 ""（调用方兜底默认名）。

    管道：逐段只替换 Windows 非法字符为 '_'、折叠重复 '_'、剥离边沿空白/
    下划线/尾点、Windows 保留设备名加 '_' 后缀。其余（非拉丁文字、重音、
    空格）原样通过，与原生 SaveImage 一致。段以 '/' 分隔。先检查 leading
    '/' 与 '..' 段（在任何清洗之前——清洗会把 '..' 吃掉，让路径穿越检查失效）。
    （省略了原版 %date:FMT% token 展开：路由场景前端从不传带 token 的前缀。）
    （preview_routes._safe_prefix 单源提升；save_image_exact._safe_filename
    的前缀清洗段与之 1:1，现委托本函数再做扩展名剥离。）"""
    if not isinstance(raw, str):
        return ""
    s = raw.strip().replace("\\", "/")
    if not s or len(s) > _PREFIX_MAX_LEN:
        return ""
    if s.startswith("/"):
        return ""
    parts = s.split("/")
    if any(p == ".." for p in parts):
        return ""
    cleaned_parts = [sanitize_segment(p) for p in parts if p]
    cleaned_parts = [p for p in cleaned_parts if p]
    if not cleaned_parts:
        return ""
    result = "/".join(cleaned_parts)
    if len(result) > _PREFIX_OUTPUT_MAX:
        result = result[:_PREFIX_OUTPUT_MAX].rstrip("/_-")
        if not result:
            return ""
    return result


def sf_user_dir() -> str:
    """<ComfyUI user dir>/sfnodes —— 本项目用户数据统一目录。

    krea2_presets / text_presets / lora_routes / character / styles_selector
    曾各持一份逻辑相同的 _sf_user_dir（注释互相点名"同款"），现收敛为单一
    实现。folder_paths 惰性导入（测试环境无 ComfyUI 时回落包内 user/）。
    """
    base = None
    try:
        import folder_paths

        base = folder_paths.get_user_directory()
    except Exception:
        base = None
    if not base:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user")
    d = os.path.join(base, "sfnodes")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def mtime_size_sig(path):
    """文件 (mtime, size) 签名；缺失/不可读返回 None。

    缓存失效键用（text/krea2_presets 的存储缓存曾各持一份逐字相同的
    stat 块；character/styles_selector 是 (path, mtime, size) 三元组键，
    形状不同不硬合）。
    """
    try:
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def atomic_write_bytes(path, data: bytes) -> None:
    """bytes 原子写盘：pid+tid 临时名 + os.replace；失败清理临时文件后原样抛错。

    临时名带 pid + 线程 id：并发写同一文件时抢同一个 .tmp 会写出混杂内容
    （tid 版在跨进程并发时仍可撞名，pid+tid 严格更安全）。异常向上传播，
    makedirs / 日志 / 返回值约定由调用方保留（各存储点的 try 形状不同，
    不在本函数内统一吞错）。
    """
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path, obj, *, ensure_ascii=False, indent=2) -> None:
    """JSON 原子写盘（presets/缓存/sidecar 各存储的 tmp+replace 同构收敛）。

    默认序列化选项对齐 presets/lora_reader（ensure_ascii=False + indent=2）；
    用标准库默认（ASCII + 无缩进）的调用方显式传 ensure_ascii=True, indent=None，
    与原 json.dump(data, f) 字节一致。
    """
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent).encode("utf-8"))
