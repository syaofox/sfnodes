"""SFLoraStack 后端路由（模块导入时注册，见 lora_notes._register_routes 先例）。

前缀统一 /api/sfnodes/。支撑多 LoRA 栈节点：文件列表、离线 info + 触发词、
预览缩略图、可选的（用户点击的）Civitai 查询、Civitai 账户、用户自定义
触发词/预览图。除 /lora/civitai 外全部离线。每个路由对配置的 loras 目录
做 realpath 守卫，构造的 ?name= 无法读到目录外。

用户数据存储位置（与 lora_presets.json 同目录惯例）：
  <user>/sfnodes/civitai.json          Civitai API key + 偏好
  <user>/sfnodes/lora_triggers.json    用户自定义触发词（按 LoRA 名）
  <user>/sfnodes/lora_previews/        用户自定义预览图
"""
import asyncio
import base64
import json
import os

import folder_paths
from aiohttp import web

# LoRA 自定义存储（lora_triggers.json）协程级互斥：保证 fp+write 单事务
# 线程级 RLock 已在 lora_reader 内保证 read→write 原子，此处再串行化
# 协程的 run_in_executor 调度，避免两协程在 await 间交错
_notes_async_lock = asyncio.Lock()

from .logger import get_logger
from . import lora_reader as R

logger = get_logger(__name__)


def _is_path_under(p, *roots):
    """绝对路径是否位于任一根目录内。

    严格 realpath 双端检查是主测试（同盘 symlink 逃逸仍被拒绝）；LEXICAL
    （abspath）检查仅作一个窄场景的回退：严格比较因两侧解析到不同盘而无法
    进行时——junction/挂载把 loras 子目录指向另一块盘（realpath 落点不同、
    commonpath 抛 ValueError），此时若仅 fail-closed，所有 junction 后的 LoRA
    都会误报 "LoRA not found"。abspath 折叠 ".."，任何分支都逃不出去。
    """
    if not p or not isinstance(p, str):
        return False
    try:
        child_abs = os.path.abspath(p)
        child_real = os.path.realpath(p)
    except (OSError, ValueError, TypeError):
        return False
    for root in roots:
        if not root or not isinstance(root, str):
            continue
        # 1) STRICT：两侧 realpath。所有普通路径的唯一测试，不放大任何面。
        cross_drive = False
        try:
            parent_real = os.path.realpath(root)
            if os.path.commonpath([child_real, parent_real]) == parent_real:
                return True
        except ValueError:
            # "Paths don't have the same drive"——无关根，或 junction 场景。
            # 只有这个结果解锁 lexical 回退。
            cross_drive = True
        except (OSError, TypeError):
            continue
        if not cross_drive:
            continue
        try:
            c = os.path.normcase(child_abs)
            pa = os.path.normcase(os.path.abspath(root))
            if os.path.commonpath([c, pa]) == pa:
                return True
        except (OSError, ValueError, TypeError):
            continue
    return False


def _lora_dirs():
    try:
        return list(folder_paths.get_folder_paths("loras"))
    except Exception:
        return []


def _sf_user_dir():
    """<ComfyUI user dir>/sfnodes —— 本项目用户数据统一目录。"""
    base = None
    try:
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


def _civitai_account_file():
    return os.path.join(_sf_user_dir(), "civitai.json")


def _custom_triggers_file():
    return os.path.join(_sf_user_dir(), "lora_triggers.json")


def _previews_dir():
    return os.path.join(_sf_user_dir(), "lora_previews")


def _civitai_account():
    return R.read_civitai_account(_civitai_account_file())


def _civitai_public_account(acc):
    """浏览器唯一可见的形状。key 本身绝不离开服务器：`configured` 说有没有，
    `hint` 显示后 4 位让用户能区分是哪个 key。"""
    return {
        "ok": True,
        "configured": bool(acc.get("key")),
        "hint": R.mask_civitai_key(acc.get("key")),
        "host": acc.get("host", "com"),
        "adultThumbs": bool(acc.get("adult_thumbs")),
    }


def _resolve_lora_path(name):
    """LoRA 文件名（含子文件夹前缀）-> 保证位于配置的 loras 目录内的真实
    路径，或 None。失败 CLOSED：loras 目录无法确定时拒绝服务未验证路径。"""
    if not name or not isinstance(name, str):
        return None
    try:
        p = folder_paths.get_full_path("loras", name)
    except Exception:
        p = None
    if not p or not os.path.isfile(p):
        return None
    roots = _lora_dirs()
    if not roots or not _is_path_under(p, *roots):
        return None
    return p


# ---------------------------------------------------------------------------
# 数据域分派：SF Load Diffusion Model（web/sf_load_diffusion_model.js）复用
# 本模块的 Civitai 查询 / 描述 / 预览图 / 孤儿迁移路由，仅换存储域——
# 同一实现服务两个域，绝不内联副本。别名路由见 _register_routes 尾部：
# /api/sfnodes/dmodel_* 指向同一 handler，handler 内按请求路径取域。
#   loras 域            diffusion_models 域
#   lora_triggers.json  dmodels.json        （用户词/描述，R 函数按文件参数化）
#   lora_previews/      previews_model/     （sha1 槽位独立目录，防撞键）
#   models/loras        models/unet + models/diffusion_models
# ---------------------------------------------------------------------------

def _is_dmodel_req(request):
    try:
        return (request.path or "").startswith("/api/sfnodes/dmodel")
    except Exception:
        return False


def _model_dirs():
    try:
        return list(folder_paths.get_folder_paths("diffusion_models"))
    except Exception:
        return []


def _dmodels_file():
    """diffusion 域的用户数据文件（与 lora_triggers.json 同构、分域）。"""
    return os.path.join(_sf_user_dir(), "dmodels.json")


def _previews_model_dir():
    """diffusion 域的用户预览图目录（独立于 lora_previews，槽位不冲突）。"""
    d = os.path.join(_sf_user_dir(), "previews_model")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _resolve_model_path(name):
    """diffusion 模型文件名 -> 保证位于配置的 diffusion_models 目录内
    （models/unet + models/diffusion_models）的真实路径，或 None。
    fail-closed 与 _resolve_lora_path 同款。"""
    if not name or not isinstance(name, str):
        return None
    try:
        p = folder_paths.get_full_path("diffusion_models", name)
    except Exception:
        p = None
    if not p or not os.path.isfile(p):
        return None
    roots = _model_dirs()
    if not roots or not _is_path_under(p, *roots):
        return None
    return p


def _dom_resolve(request, name):
    """按请求路径所属数据域解析并守卫模型文件路径。"""
    if _is_dmodel_req(request):
        return _resolve_model_path(name)
    return _resolve_lora_path(name)


def _dom_dirs(request):
    return _model_dirs() if _is_dmodel_req(request) else _lora_dirs()


def _dom_notes_file(request):
    return _dmodels_file() if _is_dmodel_req(request) else _custom_triggers_file()


def _dom_previews_dir(request):
    return _previews_model_dir() if _is_dmodel_req(request) else _previews_dir()


def _looks_like_image(raw):
    """magic bytes 判定图片（jpg/png/webp/gif/bmp）。该文件会被直接回给浏览器
    当图片渲染，不是图的就是坏功能。"""
    if not raw:
        return False
    if raw[:3] == b"\xff\xd8\xff":
        return True
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return True
    if raw[:4] in (b"GIF8",):
        return True
    if raw[:2] == b"BM":
        return True
    return False


def _looks_like_video(raw):
    """magic bytes 判定视频（mp4/mov/webm/mkv）。mp4/mov 为 ftyp，webm/mkv 为 EBML。"""
    if not raw or len(raw) < 12:
        return False
    if raw[4:8] == b"ftyp":
        return True
    if raw[:4] == b"\x1a\x45\xdf\xa3":
        return True
    return False


def _looks_like_media(raw):
    return _looks_like_image(raw) or _looks_like_video(raw)


def _ext_from_image(raw, headers=None, url=""):
    """从 magic / 响应头推断媒体扩展名（Civitai 原图 URL 的扩展名不可信，如
    .../original=true/xxx.jpeg 实际为 PNG，见 modelVersionId=3103403；视频同理）。
    优先级：magic > Content-Type > Content-Disposition > URL。永不抛错。"""
    try:
        if raw:
            if raw[:8] == b"\x89PNG\r\n\x1a\n":
                return ".png"
            if raw[:3] == b"\xff\xd8\xff":
                return ".jpg"
            if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
                return ".webp"
            if raw[:4] == b"GIF8":
                return ".gif"
            if raw[:2] == b"BM":
                return ".bmp"
            if len(raw) >= 12 and raw[4:8] == b"ftyp":
                # mp4/mov，细分：若含 isom/mp42 则 mp4，否则 mov 亦存为 mp4
                return ".mp4"
            if raw[:4] == b"\x1a\x45\xdf\xa3":
                return ".webm"
        if headers:
            ct = (headers.get("Content-Type") or headers.get("content-type") or "").split(";")[0].strip().lower()
            if ct == "image/png":
                return ".png"
            if ct in ("image/jpeg", "image/jpg"):
                return ".jpg"
            if ct == "image/webp":
                return ".webp"
            if ct == "image/gif":
                return ".gif"
            if ct == "image/bmp":
                return ".bmp"
            if ct in ("video/mp4", "video/quicktime"):
                return ".mp4"
            if ct in ("video/webm",):
                return ".webm"
            if ct in ("video/x-matroska",):
                return ".mkv"
            cd = headers.get("Content-Disposition") or headers.get("content-disposition") or ""
            import re as _re
            m = _re.search(r'filename="([^"]+)"', cd)
            if m:
                import os as _os
                e = _os.path.splitext(m.group(1))[1].lower()
                if e in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".mp4", ".mov", ".webm", ".mkv"):
                    return ".jpg" if e == ".jpeg" else (".mp4" if e == ".mov" else e)
        if url:
            import os as _os
            import urllib.parse as _up
            try:
                e = _os.path.splitext(_up.urlparse(url).path)[1].lower()
                if e in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".mp4", ".mov", ".webm", ".mkv"):
                    if e == ".jpeg":
                        return ".jpg"
                    if e == ".mov":
                        return ".mp4"
                    return e
            except Exception:
                pass
    except Exception:
        pass
    return ".jpg"


# 预览图上限：浏览器先降采样到 512px 再上传，接近这个值的都是没走我们代码的。
# Civitai 缩略图（width=256）远小于此，但流量守卫统一走它。
_PREVIEW_MAX_BYTES = 4 * 1024 * 1024
# 视频缩略图上限：视频首帧提取需下载完整视频前段，阈值放宽至 50MB（与 sample 单张上限对齐）。
_PREVIEW_VIDEO_MAX_BYTES = 50 * 1024 * 1024


_THUMB_ALLOWED_HOSTS = (
    "image.civitai.com",
    "image-cache.civitai.com",
    "cdn.civitai.com",
    "civitai.com",
    "civitai.red",
    "image.civitai.red",
    "orchestration.civitai.com",
)

def _thumb_url_safe(url):
    """Civitai 缩略图 URL 是否值得下载。URL 来自 Civitai API 响应——只信
    https 且域名在白名单（防侧车投毒 SSRF 到内网 https 服务）。纯函数，可单测。"""
    if not isinstance(url, str) or not url.startswith("https://"):
        return False
    try:
        import urllib.parse as _up
        host = (_up.urlparse(url).hostname or "").lower()
        if not host:
            return False
        for allowed in _THUMB_ALLOWED_HOSTS:
            if host == allowed or host.endswith("." + allowed):
                return True
        logger.debug(f"Blocked thumb URL host not in whitelist: {host} ({url[:80]})")
        return False
    except Exception:
        return False


async def _download_thumb(url):
    """下载一张缩略图到 bytes，任何问题返回 None。

    支持图片与视频：图片直返，视频抽首帧为 jpeg（50MB 上限）。
    流式读、magic bytes 校验——写出的文件会以图片身份直接回浏览器，
    不是图/视频的就是坏功能。永不抛错。
    """
    try:
        import aiohttp
    except Exception:
        return None
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"User-Agent": "ComfyUI-sfnodes",
                                                 "Accept": "image/*,video/*"}) as resp:
                if resp.status != 200:
                    return None
                chunks = []
                total = 0
                # 先按视频上限读，图片超 4MB 后续再拒（避免视频被 4MB 截断无法解码）
                async for chunk in resp.content.iter_chunked(65536):
                    total += len(chunk)
                    if total > _PREVIEW_VIDEO_MAX_BYTES:
                        return None
                    chunks.append(chunk)
    except Exception:
        return None
    raw = b"".join(chunks)
    if _looks_like_image(raw):
        if len(raw) > _PREVIEW_MAX_BYTES:
            return None
        return raw
    if _looks_like_video(raw):
        # 视频抽首帧为 jpeg，耗时操作放线程池
        try:
            loop = asyncio.get_running_loop()
            from .video_thumb import extract_first_frame_from_bytes
            jpeg = await loop.run_in_executor(None, extract_first_frame_from_bytes, raw)
            if jpeg and _looks_like_image(jpeg):
                return jpeg
        except Exception:
            pass
        return None
    return None


# 模型页 HTML 上限：页面（SSR + __NEXT_DATA__）实测约 130KB，2MB 只挡
# 异常/改版膨胀，不设限可能让一次性请求拖进 GB 级垃圾。
_PAGE_MAX_BYTES = 2 * 1024 * 1024

# 模型页抓取必须模拟浏览器：Cloudflare 既按 UA 也按 TLS 握手指纹（JA3）
# 拦截——"ComfyUI-sfnodes" UA 直接 403；连带 Chrome UA 的 aiohttp 请求也
# 实测 403（Python 默认 TLS 指纹被识别），curl 与 curl_cffi 的指纹才放行。
_PAGE_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def _page_fetch_curl_cffi(url):
    """curl_cffi 同步抓取（模拟浏览器 TLS 指纹 + HTTP/2 过 Cloudflare）。

    请求库自身带 libcurl 轮子，不需要系统 curl。在 executor 线程运行。
    任何失败返回 None，永不抛错。"""
    try:
        from curl_cffi import requests as cr
    except Exception:
        return None
    try:
        r = cr.get(url, impersonate="chrome", timeout=(10, 15))
        if r.status_code != 200:
            return None
        body = r.content
        if not body or len(body) > _PAGE_MAX_BYTES:
            return None
        return body
    except Exception:
        return None


async def _download_page(url):
    """抓一个 HTML 页面到 bytes，任何问题返回 None。永不抛错。

    curl_cffi（浏览器 TLS 指纹，实测过 CF）优先，aiohttp 兜底（部分直连
    网络不需要指纹伪装）。2MB 上限。失败一律返回 None 由调用方降级——
    页面只是描述的补充来源，绝不拖垮主查询。"""
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, _page_fetch_curl_cffi, url)
    if raw:
        return raw
    try:
        import aiohttp
    except Exception:
        return None
    timeout = aiohttp.ClientTimeout(total=15, connect=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"User-Agent": _PAGE_UA,
                                                 "Accept": "text/html,application/xhtml+xml"}) as resp:
                if resp.status != 200:
                    return None
                chunks = []
                total = 0
                async for chunk in resp.content.iter_chunked(65536):
                    total += len(chunk)
                    if total > _PAGE_MAX_BYTES:
                        return None
                    chunks.append(chunk)
    except Exception:
        return None
    return b"".join(chunks)


def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, routes not registered")
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/lora_list")
        async def api_lora_list(request):
            """ComfyUI 知道的每个 LoRA 文件名（名字含子文件夹前缀）。"""
            # no-store：这个 JSON 否则不带缓存头，浏览器启发式缓存它正是
            # "改名后文件永不出现"的 bug。
            hdrs = {"Cache-Control": "no-store"}
            try:
                files = list(folder_paths.get_filename_list("loras"))
            except Exception:
                # 扫描失败不是空文件夹：前端把干净 [] 当真相，会把每行都标
                # "missing"（网络盘/锁文件的瞬时故障引发全网误报）。说清楚。
                return web.json_response({"loras": [], "error": True}, headers=hdrs)
            # 分页与搜索（Phase D2）：q/limit/offset 均为可选，缺省全量以兼容旧前端
            q = request.query.get("q", "") or request.query.get("query", "")
            if q:
                ql = q.lower()
                files = [f for f in files if ql in f.lower()]
            total = len(files)
            try:
                limit = request.query.get("limit")
                offset = request.query.get("offset")
                if limit is not None:
                    lim = int(limit)
                    off = int(offset or 0)
                    if lim >= 0 and off >= 0:
                        files = files[off: off + lim]
            except Exception:
                pass
            # total 供新前端分页，旧前端忽略
            return web.json_response({"loras": files, "total": total}, headers=hdrs)

        @routes.get("/api/sfnodes/lora_info")
        async def api_lora_info(request):
            """一个 LoRA 的离线 info + 触发词（信息面板）。恒 200 让前端不必按
            HTTP 状态分支；只读文件头部 + 侧车。"""
            name = request.query.get("name", "")
            path = _resolve_lora_path(name)
            if not path:
                # 文件不存在（改名/移动后旧路径行）：统一存储按基名孤儿兜底
                # （无文件可算指纹，只做基名唯一匹配）。命中返回 store 数据
                # + _file_missing/orphan_key，面板提示用户重新选择路径。
                try:
                    store = R.read_custom_store(_custom_triggers_file())
                    orphan = R.find_orphan_key(store, name)
                    if orphan:
                        entry = store.get(orphan) or {}
                        info = {
                            "title": name.rsplit("/", 1)[-1] or name,
                            "base_model": "",
                            "description": entry.get("description", ""),
                            "triggers": entry.get("words", []),
                            "file_triggers": [],
                            "sidecar_triggers": [],
                            "source": "custom",
                            "has_preview": False,
                            "custom_triggers": entry.get("words", []),
                            "custom_description": entry.get("description", ""),
                            "custom_selected": entry.get("selected", []),
                            "orphan_key": orphan,
                            "orphan_triggers": entry.get("words", []),
                            "orphan_description": entry.get("description", ""),
                            "orphan_selected": entry.get("selected", []),
                            "orphan_preview": False,
                            "preview_v": 0,
                            "custom_preview": False,
                            "restorable_thumb": False,
                            "civitai_host": _civitai_account().get("host", "com"),
                            "_file_missing": True,
                        }
                        return web.json_response({"ok": True, "info": info},
                                                 headers={"Cache-Control": "no-store"})
                except Exception:
                    pass
                return web.json_response({"ok": False, "message": "LoRA not found."})
            try:
                loop = asyncio.get_running_loop()
                # 头部读取虽小，但哈希/侧车 I/O 是磁盘绑定的——移出 aiohttp
                # 事件循环。
                info = await loop.run_in_executor(None, R.build_lora_info, path)
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not read: {}".format(exc)})
            # 用户自己的词随文件词和 Civitai 词一起返回，面板打开瞬间三源齐备。
            # 统一存储真源。旧
            # lora_notes 侧车（<base>.sf.json）在任一读取入口首次惰性迁移
            # 并入后删除（幂等：store 已有该 LoRA 数据则跳过）。
            try:
                await loop.run_in_executor(
                    None, R.migrate_legacy_sidecar, _custom_triggers_file(), path, name
                )
            except Exception:
                pass
            try:
                info["custom_triggers"] = R.get_custom_triggers(_custom_triggers_file(), name)
            except Exception:
                info["custom_triggers"] = []
            # 用户自己的描述同理：custom_description 存在则面板优先展示它，
            # 否则展示文件/Civitai 的 description。
            try:
                info["custom_description"] = R.get_custom_description(_custom_triggers_file(), name)
            except Exception:
                info["custom_description"] = ""
            # 全局默认勾选（新建行默认值，工作流覆盖它）
            try:
                info["custom_selected"] = R.get_custom_selected(_custom_triggers_file(), name)
            except Exception:
                info["custom_selected"] = []
            # 面板头部 "View on Civitai" 链接按账户主机偏好生成（偏好 red 的
            # 用户打开 civitai.red 页，成人模型在 com 网页可能访问受限）。
            try:
                info["civitai_host"] = _civitai_account().get("host", "com")
            except Exception:
                info["civitai_host"] = "com"
            # 孤儿数据检测：文件被移动/改名后，新键下没有自定义数据，但存储
            # 里还有旧键的数据（自定义词/描述/预览图仍在）。指纹优先（内容
            # 级证据，文件改名也匹配），基名兜底（存量无指纹数据）。附字段
            # 让前端显示迁移提示条；迁移由用户确认后执行（不自动，防误配）。
            try:
                # 无论新键是否有数据都检测孤儿：无数据时为迁移，有数据时为合并提示
                store = R.read_custom_store(_custom_triggers_file())
                orphan = None
                fp = await loop.run_in_executor(None, R.file_fingerprint, path)
                if fp:
                    orphan = R.find_orphan_by_fingerprint(store, fp, exclude=R.custom_trigger_key(name))
                if orphan is None:
                    orphan = R.find_orphan_key(store, name)
                if orphan:
                    entry = store.get(orphan, {})
                    info["orphan_key"] = orphan
                    info["orphan_triggers"] = entry.get("words", [])
                    info["orphan_description"] = entry.get("description", "")
                    info["orphan_selected"] = entry.get("selected", [])
                    info["orphan_preview"] = bool(R.find_custom_preview(_previews_dir(), orphan))
            except Exception:
                pass
            # ...以及他们自己的预览图。custom_preview 驱动面板的 "remove"
            # 开关；preview_v 是 mtime，让浏览器越过缩略图路由的一小时缓存
            # （别的节点/别的会话换过图而本面板没看见的情况）。
            try:
                folder = _previews_dir()
                info["preview_v"] = R.custom_preview_version(folder, name)
                info["custom_preview"] = bool(info["preview_v"])
                if info["custom_preview"]:
                    info["has_preview"] = True
            except Exception:
                info["custom_preview"] = False
                info["preview_v"] = 0
            # 封面恢复：自动保存的封面图以 LoRA 路径 hash 命名，文件移动/
            # 改名后 hash 失配、本地找不到。侧车（跟随文件）里仍有缩略图
            # URL——附标志让前端静默重下载到新 hash 名下。仅当本地确无预览
            # 时附 True，避免打扰本就有封面的 LoRA（此处 custom_preview 已定）。
            try:
                info["restorable_thumb"] = False
                if not info.get("custom_preview"):
                    acc = _civitai_account()
                    if R.sidecar_thumbnail(path, allow_adult=bool(acc.get("adult_thumbs"))):
                        info["restorable_thumb"] = True
            except Exception:
                info["restorable_thumb"] = False
            return web.json_response({"ok": True, "info": info},
                                     headers={"Cache-Control": "no-store"})

        @routes.get("/api/sfnodes/lora_thumb")
        async def api_lora_thumb(request):
            """提供 LoRA/扩散模型的预览图，404 则无。

            用户自己的图胜过模型旁的图：面板和未来任何缩略图都读这里，
            覆盖必须在此兑现而非只在显示处。（dmodel 别名路径同 handler，
            域分派见模块头注释。）"""
            name = request.query.get("name", "")
            path = _dom_resolve(request, name)
            if not path:
                return web.Response(status=404)
            try:
                own = R.find_custom_preview(_dom_previews_dir(request), name)
            except Exception:
                own = None
            if own:
                return web.FileResponse(own, headers={"Cache-Control": "public, max-age=3600"})
            prev = R.find_preview_path(path)
            roots = _dom_dirs(request)
            if not prev or not roots or not _is_path_under(prev, *roots):
                return web.Response(status=404)
            return web.FileResponse(prev, headers={"Cache-Control": "public, max-age=3600"})

        @routes.get("/api/sfnodes/lora/civitai")
        async def api_lora_civitai(request):
            """可选在线查询（用户点击，或 dmodel 面板打开时自动）。

            给文件取指纹（SHA256），向 Civitai 请求精确文件匹配，把原始响应
            缓存在模型旁（侧车跟随文件），未来读取即时且离线。恒 200；
            `reason` 告诉前端显示哪张卡：found / notfound / offline。
            （dmodel 别名路径同 handler，域分派见模块头注释。）"""
            name = request.query.get("name", "")
            path = _dom_resolve(request, name)
            if not path:
                return web.json_response({"ok": False, "reason": "notfound", "message": "Model not found."})
            loop = asyncio.get_running_loop()
            try:
                sha = await loop.run_in_executor(None, R.file_sha256, path)
            except Exception as exc:
                return web.json_response({"ok": False, "reason": "offline",
                                          "message": "Could not read the file: {}".format(exc)})
            try:
                import aiohttp
            except Exception:
                return web.json_response({"ok": False, "reason": "offline",
                                          "message": "Could not reach Civitai."})
            # 30s 而非 12s：Civitai API 负载时经常慢，提前放弃读起来像
            # "这功能不工作"。哈希已算完，这预算纯粹是 HTTP 往返。
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            acc = _civitai_account()
            hosts = R.civitai_hosts(acc.get("host"))
            logger.info("[SFLoraStack] civitai lookup for {}: hosts={} key={}".format(
                name, ",".join(hosts), "yes" if acc.get("key") else "no"))
            headers = {
                "User-Agent": "ComfyUI-sfnodes",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }
            if acc.get("key"):
                headers["Authorization"] = "Bearer {}".format(acc["key"])
            data = None
            last_note = "Could not reach Civitai."
            # 针对 key 的拒绝是最可操作的报告，单独留着，不被第二个主机随后
            # 的话（那里的超时会埋掉它）覆盖。
            key_note = None
            by_hash_notfound = False
            for i, host in enumerate(hosts):
                last = i == len(hosts) - 1
                url = "https://{}/api/v1/model-versions/by-hash/{}".format(host, sha)
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(url, headers=headers) as resp:
                            if resp.status == 404:
                                # 404 只在最后一个主机上定论。曾经立即结束查找，
                                # 理由是两主机服务同一目录——对公开模型成立，但
                                # 对成人评级模型不成立：主站用这同一个 404 隐藏
                                # 它，而 unrestricted 主机正常返回。
                                if not last:
                                    last_note = "Not found on {}.".format(host)
                                    continue
                                by_hash_notfound = True
                                last_note = "Not found on Civitai (by hash)."
                                break
                            if resp.status in (401, 403):
                                # 绝不在循环内返回：401/403 是最主机特有的失败
                                # （一个域名的 Cloudflare/公司/ISP 屏蔽页，另一
                                # 域名正常），备胎主机正是为此存在。
                                if acc.get("key"):
                                    key_note = ("Civitai refused the API key ({}). Check it in the node "
                                                "settings.".format(resp.status))
                                    last_note = key_note
                                else:
                                    last_note = ("Civitai refused the request ({}). Your network may be "
                                                 "blocking Civitai, or this model may need an API key - "
                                                 "add one in the node settings.".format(resp.status))
                                continue
                            if resp.status != 200:
                                last_note = "Civitai returned {}.".format(resp.status)
                                continue
                            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
                            # 循环读整个 body，保留内存上限。
                            chunks = []
                            total = 0
                            async for chunk in resp.content.iter_chunked(65536):
                                total += len(chunk)
                                if total > 4 * 1024 * 1024:
                                    return web.json_response({"ok": False, "reason": "offline",
                                                              "message": "Civitai response too large."})
                                chunks.append(chunk)
                            body = b"".join(chunks)
                            try:
                                data = json.loads(body)
                            except Exception:
                                # 200 但不是 JSON = 网络/防护层的屏蔽或登录页，
                                # 不是 Civitai 说没有。命名 content type 让下次
                                # bug 报告可诊断。continue，绝不 return：另一
                                # 主机正是按域名屏蔽的备胎。
                                data = None
                                last_note = ("Civitai replied with {} instead of data - most likely a "
                                             "block or sign-in page from your network or its protection "
                                             "layer.".format(ctype or "an unknown format"))
                                continue
                            break
                except Exception as exc:
                    # 保留失败原因：超时/DNS/TLS/代理拒绝和屏蔽页曾全部坍缩成
                    # 一行通用文案，失去了给用户看原因的意义。
                    kind = type(exc).__name__
                    if "Timeout" in kind or "Cancelled" in kind:
                        last_note = "Civitai timed out."
                    elif "ContentEncoding" in kind or "Decompress" in kind:
                        last_note = ("Civitai sent a compressed reply this install cannot read ({}). "
                                     "Please report this.".format(kind))
                    elif "JSON" in kind or "Decode" in kind or "Value" in kind:
                        last_note = "Civitai sent an unreadable reply (a login or block page?)."
                    else:
                        last_note = "Could not reach Civitai ({}).".format(kind)
                    continue
            # by-hash 未命中或网络失败后的回退：Archive 模型（zip）本地解压后指纹与归档不一致，
            # 需通过用户粘贴的 Civitai 链接/ID 二次查询
            if by_hash_notfound or data is None:
                ids = {"model_id": None, "version_id": None}
                for k in ["modelId", "model_id"]:
                    v = request.query.get(k)
                    cid = R._clean_id(v) if v else None
                    if cid is not None:
                        ids["model_id"] = cid
                for k in ["versionId", "version_id"]:
                    v = request.query.get(k)
                    cid = R._clean_id(v) if v else None
                    if cid is not None:
                        ids["version_id"] = cid
                for k in ["civitaiUrl", "url", "link", "civitai_url"]:
                    v = request.query.get(k)
                    if v:
                        ex = R.extract_civitai_ids(v)
                        if ex.get("model_id") and not ids["model_id"]:
                            ids["model_id"] = ex["model_id"]
                        if ex.get("version_id") and not ids["version_id"]:
                            ids["version_id"] = ex["version_id"]
                if ids["version_id"] or ids["model_id"]:
                    fallback_data = None
                    fallback_host = hosts[0] if hosts else "civitai.com"
                    if ids["version_id"]:
                        for h in hosts:
                            try:
                                f_url = "https://{}/api/v1/model-versions/{}".format(h, ids["version_id"])
                                async with aiohttp.ClientSession(timeout=timeout) as session:
                                    async with session.get(f_url, headers=headers) as resp:
                                        if resp.status != 200:
                                            continue
                                        chunks = []
                                        total = 0
                                        async for chunk in resp.content.iter_chunked(65536):
                                            total += len(chunk)
                                            if total > 4 * 1024 * 1024:
                                                break
                                            chunks.append(chunk)
                                        body = b"".join(chunks)
                                        try:
                                            fallback_data = json.loads(body)
                                        except Exception:
                                            continue
                                        fallback_host = h
                                        break
                            except Exception:
                                continue
                    if fallback_data is None and ids["model_id"]:
                        for h in hosts:
                            try:
                                f_url = "https://{}/api/v1/models/{}".format(h, ids["model_id"])
                                async with aiohttp.ClientSession(timeout=timeout) as session:
                                    async with session.get(f_url, headers=headers) as resp:
                                        if resp.status != 200:
                                            continue
                                        chunks = []
                                        total = 0
                                        async for chunk in resp.content.iter_chunked(65536):
                                            total += len(chunk)
                                            if total > 4 * 1024 * 1024:
                                                break
                                            chunks.append(chunk)
                                        body = b"".join(chunks)
                                        try:
                                            m_data = json.loads(body)
                                        except Exception:
                                            continue
                                        vers = m_data.get("modelVersions") or []
                                        target = None
                                        if ids["version_id"]:
                                            for v in vers:
                                                if str(v.get("id")) == str(ids["version_id"]):
                                                    target = v
                                                    break
                                        if target is None and vers:
                                            target = vers[0]
                                        if target:
                                            if not isinstance(target.get("model"), dict):
                                                target["model"] = {"name": m_data.get("name"), "type": m_data.get("type")}
                                            fallback_data = target
                                            fallback_host = h
                                            break
                            except Exception:
                                continue
                    if fallback_data is not None:
                        data = fallback_data
                        host = fallback_host
                        by_hash_notfound = False
                    else:
                        if by_hash_notfound:
                            return web.json_response({"ok": True, "found": False, "reason": "notfound", "hint": "未找到对应的 Civitai 版本，请检查链接或 ID 是否正确。"})
                        return web.json_response({"ok": False, "reason": "offline", "message": key_note or last_note})
                if by_hash_notfound:
                    return web.json_response({"ok": True, "found": False, "reason": "archive", "hint": "该版本为 .zip 归档（含多个文件），本地文件的指纹与归档不一致。请粘贴 Civitai 链接或 Version ID 后重试。"})
            if data is None:
                # key 拒绝优于后面的主机说的话："检查你的 key"是用户能行动的，
                # 尾随超时不是。
                return web.json_response({"ok": False, "reason": "offline",
                                          "message": key_note or last_note})
            parsed = R.parse_civitai_modelversion(data, allow_adult=bool(acc.get("adult_thumbs")))
            # Civitai 回了 200 + 可用记录 -> FOUND，即使这个版本恰好没有
            # trainedWords 和 model.name（很多都没有）。
            if not parsed:
                return web.json_response({"ok": True, "found": False, "reason": "notfound"})
            # 页面主体描述补充：API 的 version 级 description 实测常常为空，
            # 而模型页 Description 卡显示模型级完整描述（SSR __NEXT_DATA__）。
            # 抓页面提取后拼接——API 在前、页面在后。任何失败降级为仅有 API
            # 描述，绝不破坏查询结果。
            page_desc = ""
            if parsed.get("model_id"):
                page_url = "https://{}/models/{}".format(host, parsed["model_id"])
                if parsed.get("version_id"):
                    page_url += "?modelVersionId={}".format(parsed["version_id"])
                try:
                    raw = await _download_page(page_url)
                    if raw:
                        page_desc = R.extract_page_description(
                            raw.decode("utf-8", errors="replace"))
                except Exception as exc:
                    logger.warning("[SFLoraStack] page description fetch failed for {}: {}".format(
                        name, exc))
            merged = R.merge_descriptions(parsed.get("description"), page_desc)
            # 若开启批量下载开关，追加全部样例的生成信息（原样保留，全部图片）
            download_flag = request.query.get("downloadSamples") == "1" or request.query.get("download_samples") == "1"
            if download_flag:
                sp = parsed.get("sample_prompts")
                if isinstance(sp, str) and sp.strip():
                    merged = "\n\n".join([s for s in (merged, sp) if isinstance(s, str) and s.strip()])
            if merged:
                parsed["description"] = merged
                # 侧车同步存拼接版：未来读取/其他节点（lora_notes）
                # 从侧车解析即拿到完整描述，无需重新抓取。
                data["description"] = merged
            # 写入侧归一化：description 统一以 markdown 缓存并打 sfnodes_description_md
            # 标记——读侧（read_sidecar_info）凭标记确定性透传，不再猜 HTML/markdown。
            # 页面成功时 merged 已是 markdown；失败时把原始 API 描述转一次。
            raw_desc = data.get("description")
            if not merged and isinstance(raw_desc, str):
                data["description"] = R._html_to_markdown(raw_desc)
            data["sfnodes_description_md"] = True
            await loop.run_in_executor(None, R.save_sidecar_cache, path, data)
            resp = {"ok": True, "found": True, "info": parsed}
            # 封面自动保存到本地（与用户自定义预览同目录同名规则）：
            # 无现有自定义预览 -> 静默下载保存；有且未确认覆盖 -> 跳过；
            # 下载/校验失败不致命——文本信息照常返回，前端在状态条提示。
            thumbnail = parsed.get("thumbnail")
            if thumbnail:
                try:
                    folder = _dom_previews_dir(request)
                    existing = R.find_custom_preview(folder, name)
                    overwrite = request.query.get("overwrite") == "1"
                    if existing and not overwrite:
                        resp["thumb_skipped"] = True
                    elif not _thumb_url_safe(thumbnail):
                        resp["thumb_error"] = "The picture URL is not a secure https link."
                    else:
                        raw = await _download_thumb(thumbnail)
                        if raw is None:
                            resp["thumb_error"] = "Could not download the picture from Civitai."
                        else:
                            written = await loop.run_in_executor(
                                None, R.write_custom_preview, folder, name, raw
                            )
                            if not written:
                                resp["thumb_error"] = "Could not save the picture locally."
                            else:
                                resp["thumb_v"] = R.custom_preview_version(folder, name)
                except Exception as exc:
                    resp["thumb_error"] = "Could not save the picture: {}".format(exc)
            # 批量下载全部原图样例到 sample/（开关 sfnodes.Civitai.DownloadSamples，默认关；不压缩、不做 NSFW 过滤）
            download_flag = request.query.get("downloadSamples") == "1" or request.query.get("download_samples") == "1"
            original_images = parsed.get("original_images") or []
            if download_flag and original_images:
                try:
                    sample_dir = os.path.join(os.path.dirname(path), "sample")
                    await loop.run_in_executor(None, lambda: os.makedirs(sample_dir, exist_ok=True))
                    # 并发 3，总量 500MB，单张 100MB（视频原图可达数十 MB，用户要求不限制/加大限制）
                    import hashlib as _hashlib
                    import urllib.parse as _urlparse

                    sem = asyncio.Semaphore(3)
                    total_downloaded = 0
                    downloaded = 0
                    skipped = 0
                    failed = 0
                    total_bytes = 0
                    BATCH_MAX = 500 * 1024 * 1024
                    SINGLE_MAX = 100 * 1024 * 1024

                    async def _dl_one(idx, url):
                        nonlocal total_bytes, downloaded, skipped, failed
                        if not isinstance(url, str) or not url.startswith("https://"):
                            failed += 1
                            return
                        # 按 URL hash 命名，扩展名由实际内容推断（URL 的 .jpeg 不可信，见 3103403 仅为 PNG）
                        try:
                            h = _hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
                        except Exception:
                            failed += 1
                            return
                        if total_bytes >= BATCH_MAX:
                            skipped += 1
                            return
                        async with sem:
                            try:
                                import aiohttp as _aio
                                tout = _aio.ClientTimeout(total=20, connect=10)
                                async with _aio.ClientSession(timeout=tout) as sess:
                                    async with sess.get(url, headers={"User-Agent": "ComfyUI-sfnodes", "Accept": "image/*,video/*"}) as r:
                                        if r.status != 200:
                                            failed += 1
                                            return
                                        # 响应头用于扩展名推断（重定向后 B2 的 content-type/content-disposition 才是真值）
                                        hdrs = r.headers
                                        chunks = []
                                        cur = 0
                                        async for chunk in r.content.iter_chunked(65536):
                                            cur += len(chunk)
                                            if cur > SINGLE_MAX:
                                                failed += 1
                                                return
                                            if total_bytes + cur > BATCH_MAX:
                                                failed += 1
                                                return
                                            chunks.append(chunk)
                                        raw = b"".join(chunks)
                                        if not _looks_like_media(raw):
                                            failed += 1
                                            return
                                        ext = _ext_from_image(raw, hdrs, url)
                                        fname = f"civitai_{idx:02d}_{h}{ext}"
                                        fpath = os.path.join(sample_dir, fname)
                                        # A 方案：按 hash 去重（共享目录多 LoRA 复用，忽略 idx 差异）
                                        # 同目录如 wan22/.../All-In-One NSFW 下两 LoRA 的同 URL 在不同 idx 会产生
                                        # civitai_00_H vs civitai_05_H，旧逻辑仅判同 idx 不同 ext 会重复下载；
                                        # 现扫描目录内任意含 _<H>. 的文件即视为已存，兼容旧误标与 idx 差异。
                                        try:
                                            for _existing in os.listdir(sample_dir):
                                                # 快速过滤：须含 _<hash>.
                                                if f"_{h}." not in _existing:
                                                    continue
                                                # 确保 hash 段在扩展名前（civitai_??_H.ext）
                                                name_wo_ext = os.path.splitext(_existing)[0]
                                                if name_wo_ext.endswith(f"_{h}"):
                                                    skipped += 1
                                                    return
                                        except Exception:
                                            pass
                                        # 原图直存，不压缩
                                        def _write():
                                            tmp = fpath + f".{os.getpid()}.{hash(fname) & 0xffff}.tmp"
                                            try:
                                                with open(tmp, "wb") as f:
                                                    f.write(raw)
                                                os.replace(tmp, fpath)
                                                return True
                                            except Exception:
                                                try:
                                                    os.remove(tmp)
                                                except Exception:
                                                    pass
                                                return False
                                        ok = await loop.run_in_executor(None, _write)
                                        if ok:
                                            total_bytes += len(raw)
                                            downloaded += 1
                                        else:
                                            failed += 1
                            except Exception:
                                failed += 1
                                return

                    # 依次调度（受 sem 限流）
                    await asyncio.gather(*[_dl_one(i, u) for i, u in enumerate(original_images)])
                    if downloaded:
                        resp["samples_downloaded"] = downloaded
                    if skipped:
                        resp["samples_skipped"] = skipped
                    if failed:
                        resp["samples_failed"] = failed
                except Exception as exc:
                    resp["samples_error"] = "Could not download samples: {}".format(exc)
            elif not download_flag and original_images:
                resp["samples_note"] = "Sample images not downloaded — enable sfnodes.Civitai.DownloadSamples in settings."
            return web.json_response(resp)

        @routes.get("/api/sfnodes/civitai/account")
        async def api_civitai_account_get(request):
            """是否配置了 key，以及两个查询偏好。绝不是 key 本身。"""
            return web.json_response(_civitai_public_account(_civitai_account()),
                                     headers={"Cache-Control": "no-store"})

        @routes.post("/api/sfnodes/civitai/account")
        async def api_civitai_account_set(request):
            """设置 key 和/或偏好。缺席字段不动；`key: ""` 清 key。以同样的
            公开形状应答，面板按服务器实际存储重绘。"""
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                body = {}
            acc = _civitai_account()
            if "key" in body:
                raw = body.get("key")
                if isinstance(raw, str) and raw.strip() == "":
                    acc["key"] = ""
                else:
                    k = R.sanitize_civitai_key(raw)
                    if not k:
                        return web.json_response({
                            "ok": False,
                            "message": "That does not look like an API key - it should be one "
                                       "run of ordinary characters with no spaces.",
                        }, headers={"Cache-Control": "no-store"})
                    acc["key"] = k
            if body.get("host") in ("com", "red"):
                acc["host"] = body["host"]
            if "adultThumbs" in body:
                acc["adult_thumbs"] = bool(body.get("adultThumbs"))
            if not R.write_civitai_account(_civitai_account_file(), acc):
                return web.json_response({"ok": False, "message": "Could not save the settings file."},
                                         headers={"Cache-Control": "no-store"})
            return web.json_response(_civitai_public_account(acc), headers={"Cache-Control": "no-store"})

        @routes.post("/api/sfnodes/lora/custom_triggers")
        async def api_lora_custom_triggers(request):
            """保存一个 LoRA 的用户自定义触发词。POST {name, words}。

            名字是存储键，绝非文件路径——custom_trigger_key 归一化后当 dict 键
            用，够不到磁盘。仍先对 loras 目录解析它，存储只会积累真实存在的
            LoRA（拼错的/敌意名字被拒而非悄悄积累）。恒 200。"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            name = data.get("name", "") or request.query.get("name", "")
            words = data.get("words", [])
            path = _resolve_lora_path(name)
            roots = _lora_dirs()
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False, "message": "LoRA not found."})
            loop = asyncio.get_running_loop()
            try:
                async with _notes_async_lock:
                    # 内容指纹随条目记录：文件日后改名/移动，孤儿匹配靠它找回。
                    def _set_with_fp():
                        fp = R.file_fingerprint(path)
                        return R.set_custom_triggers(_custom_triggers_file(), name, words, fp)
                    stored = await loop.run_in_executor(None, _set_with_fp)
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not save: {}".format(exc)})
            return web.json_response({"ok": True, "words": stored})

        @routes.post("/api/sfnodes/lora/custom_description")
        async def api_lora_custom_description(request):
            """保存一个 LoRA 的用户自定义描述（覆盖 Civitai/文件的说明）。
            POST {name, description}。

            名字是存储键，绝非文件路径（同 custom_triggers 规则）——仍先对
            模型目录解析它，存储只积累真实存在的模型。空字符串清除自定义
            描述（回到 Civitai/文件原文）。恒 200。（dmodel 别名路径同
            handler，域分派见模块头注释。）"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            name = data.get("name", "") or request.query.get("name", "")
            description = data.get("description", "")
            path = _dom_resolve(request, name)
            roots = _dom_dirs(request)
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False, "message": "Model not found."})
            loop = asyncio.get_running_loop()
            try:
                async with _notes_async_lock:
                    # 内容指纹随条目记录（同 custom_triggers）。
                    def _set_with_fp():
                        fp = R.file_fingerprint(path)
                        return R.set_custom_description(_dom_notes_file(request), name, description, fp)
                    stored = await loop.run_in_executor(None, _set_with_fp)
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not save: {}".format(exc)})
            return web.json_response({"ok": True, "description": stored})

        @routes.post("/api/sfnodes/lora/custom_selected")
        async def api_lora_custom_selected(request):
            """保存一个 LoRA 的全局默认勾选（新建行默认值）。POST {name, selected}。

            工作流覆盖它，仅空行回填。与 custom_triggers 同守卫/锁/指纹语义，
            名字仍先对 loras 目录解析以防存储积累敌意键。恒 200。"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            name = data.get("name", "") or request.query.get("name", "")
            selected = data.get("selected", [])
            path = _resolve_lora_path(name)
            roots = _lora_dirs()
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False, "message": "LoRA not found."})
            loop = asyncio.get_running_loop()
            try:
                async with _notes_async_lock:
                    def _set_with_fp():
                        fp = R.file_fingerprint(path)
                        return R.set_custom_selected(_custom_triggers_file(), name, selected, fp)
                    stored = await loop.run_in_executor(None, _set_with_fp)
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not save: {}".format(exc)})
            return web.json_response({"ok": True, "selected": stored})

        @routes.post("/api/sfnodes/lora/migrate")
        async def api_lora_migrate(request):
            """把旧路径键下的自定义数据（词/描述/预览图）迁移到当前 LoRA 名。
            POST {name}。基名唯一匹配由纯逻辑决定；新键已有数据时不迁移。
            恒 200。"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            name = data.get("name", "") or request.query.get("name", "")
            old_key = data.get("old_key", "") or None
            path = _dom_resolve(request, name)
            roots = _dom_dirs(request)
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False, "message": "Model not found."})
            loop = asyncio.get_running_loop()
            try:
                async with _notes_async_lock:
                    # old_key 来自孤儿检测（指纹或基名命中）；fp 随迁移写入新键。
                    def _migrate_with_fp():
                        fp = R.file_fingerprint(path)
                        return R.migrate_custom_data(_dom_notes_file(request), name, fp, old_key)
                    res = await loop.run_in_executor(None, _migrate_with_fp)
                    if not res.get("ok"):
                        return web.json_response({"ok": False, "message": "Nothing to migrate."})
                    old = res["old_key"]
                    moved_pv = await loop.run_in_executor(
                        None, R.migrate_custom_preview, _dom_previews_dir(request), name, old
                    )
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not migrate: {}".format(exc)})
            return web.json_response({"ok": True, "old_key": old,
                                      "preview_moved": bool(moved_pv)})

        @routes.post("/api/sfnodes/lora/merge")
        async def api_lora_merge(request):
            """把旧路径键下的自定义数据合并到当前 LoRA 名（新键已有数据时用）。

            词并集去重、描述拼接（新在前旧在后）、预览图按需迁移。POST {name, old_key}。恒 200。"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            name = data.get("name", "") or request.query.get("name", "")
            old_key = data.get("old_key", "") or None
            path = _dom_resolve(request, name)
            roots = _dom_dirs(request)
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False, "message": "Model not found."})
            loop = asyncio.get_running_loop()
            try:
                async with _notes_async_lock:
                    def _merge_with_fp():
                        fp = R.file_fingerprint(path)
                        return R.merge_custom_data(_dom_notes_file(request), name, fp, old_key)
                    res = await loop.run_in_executor(None, _merge_with_fp)
                    if not res.get("ok"):
                        return web.json_response({"ok": False, "message": "Nothing to merge."})
                    old = res["old_key"]
                    # 预览图：旧有新无时迁移，旧有新有则保留新
                    folder = _dom_previews_dir(request)
                    has_new = bool(R.find_custom_preview(folder, name))
                    moved_pv = False
                    if not has_new:
                        moved_pv = await loop.run_in_executor(
                            None, R.migrate_custom_preview, folder, name, old
                        )
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not merge: {}".format(exc)})
            return web.json_response({"ok": True, "old_key": old, "merged": True,
                                      "preview_moved": bool(moved_pv)})

        @routes.post("/api/sfnodes/lora/civitai_thumb_save")
        async def api_lora_civitai_thumb_save(request):
            """用户确认后，把侧车里的 Civitai 缩略图下载并覆盖保存为本地预览。
            POST {name}。

            查询时若已有用户自定义预览会跳过保存（thumb_skipped）；前端确认
            替换后调这里——读侧车拿同一张图（无需重新查询），覆盖写本地预览。
            恒 200。"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            name = data.get("name", "") or request.query.get("name", "")
            path = _dom_resolve(request, name)
            roots = _dom_dirs(request)
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False,
                                          "message": "Model not found."})
            loop = asyncio.get_running_loop()
            thumbnail = R.sidecar_thumbnail(path, allow_adult=bool(_civitai_account().get("adult_thumbs")))
            if not thumbnail:
                return web.json_response({"ok": False,
                                          "message": "No Civitai picture saved for this model - run the lookup first."})
            if not _thumb_url_safe(thumbnail):
                return web.json_response({"ok": False,
                                          "message": "The picture URL is not a secure https link."})
            raw = await _download_thumb(thumbnail)
            if raw is None:
                return web.json_response({"ok": False,
                                          "message": "Could not download the picture from Civitai."})
            folder = _dom_previews_dir(request)
            written = await loop.run_in_executor(None, R.write_custom_preview, folder, name, raw)
            if not written:
                return web.json_response({"ok": False, "message": "Could not save the picture locally."})
            return web.json_response({"ok": True, "v": R.custom_preview_version(folder, name)})

        @routes.post("/api/sfnodes/lora/preview")
        async def api_lora_preview_set(request):
            """存储一个 LoRA 的用户自定义预览图。POST {name, dataUrl}。

            解码前先查大小（base64 膨胀三分之一，先解码会让超大载荷用解码
            所需内存来拒绝它）；字节必须长得像图（此文件直接以图片回浏览器）。
            恒 200。"""
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                body = {}
            name = str(body.get("name", "") or "")
            data_url = str(body.get("dataUrl", "") or "")
            path = _dom_resolve(request, name)
            roots = _dom_dirs(request)
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False, "message": "Model not found."})
            if "," not in data_url:
                return web.json_response({"ok": False, "message": "Nothing to save."})
            payload = data_url.split(",", 1)[1]
            if len(payload) > _PREVIEW_MAX_BYTES * 4 // 3 + 8:
                return web.json_response({"ok": False, "message": "That picture is too large."})
            try:
                raw = base64.b64decode(payload)
            except Exception:
                return web.json_response({"ok": False, "message": "That picture could not be read."})
            if not raw or len(raw) > _PREVIEW_MAX_BYTES:
                return web.json_response({"ok": False, "message": "That picture is too large."})
            if not _looks_like_image(raw):
                return web.json_response(
                    {"ok": False, "message": "That file is not a picture the browser can show."})
            loop = asyncio.get_running_loop()
            folder = _dom_previews_dir(request)
            try:
                written = await loop.run_in_executor(None, R.write_custom_preview, folder, name, raw)
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not save: {}".format(exc)})
            if not written:
                return web.json_response({"ok": False, "message": "Could not save that picture."})
            return web.json_response({"ok": True, "v": R.custom_preview_version(folder, name)})

        @routes.post("/api/sfnodes/lora/preview_delete")
        async def api_lora_preview_delete(request):
            """删除用户自定义预览，自动图回来。POST {name}。

            文件名从 LoRA 名推导并先检查我们可能写出的形状，手改请求无法把
            os.remove 指向别处。恒 200。"""
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                body = {}
            name = str(body.get("name", "") or "") or request.query.get("name", "")
            path = _dom_resolve(request, name)
            roots = _dom_dirs(request)
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False, "message": "Model not found."})
            try:
                removed = R.delete_custom_preview(_dom_previews_dir(request), name)
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not remove: {}".format(exc)})
            return web.json_response({"ok": True, "removed": bool(removed)})

        @routes.post("/api/sfnodes/lora/civitai_delete")
        async def api_lora_civitai_delete(request):
            """删除缓存的 Civitai 侧车（<base>.civitai.info），信息回到文件自己的
            词。POST {name}。路径守卫到 loras 目录；恒 200。"""
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            name = data.get("name", "") or request.query.get("name", "")
            path = _dom_resolve(request, name)
            roots = _dom_dirs(request)
            if not path or not roots or not _is_path_under(path, *roots):
                return web.json_response({"ok": False, "message": "Model not found."})
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(None, R.delete_sidecar_cache, path)
            return web.json_response({"ok": bool(ok)})

        # ------------------------------------------------------------------
        # SF Load Diffusion Model 别名路由：同一 handler 注册到 dmodel 前缀，
        # handler 内按请求路径分派数据域（见模块头注释）。触发词是 LoRA 专属
        # 概念，custom_triggers / lora_info / lora_list 不设别名——dmodel 域
        # 的 info 组装在 sf_utils/diffusion_routes.py。
        # ------------------------------------------------------------------
        routes.get("/api/sfnodes/dmodel_thumb")(api_lora_thumb)
        routes.get("/api/sfnodes/dmodel/civitai")(api_lora_civitai)
        routes.post("/api/sfnodes/dmodel/custom_description")(api_lora_custom_description)
        routes.post("/api/sfnodes/dmodel/migrate")(api_lora_migrate)
        routes.post("/api/sfnodes/dmodel/merge")(api_lora_merge)
        routes.post("/api/sfnodes/dmodel/civitai_thumb_save")(api_lora_civitai_thumb_save)
        routes.post("/api/sfnodes/dmodel/preview")(api_lora_preview_set)
        routes.post("/api/sfnodes/dmodel/preview_delete")(api_lora_preview_delete)
        routes.post("/api/sfnodes/dmodel/civitai_delete")(api_lora_civitai_delete)

        logger.info("LoRA stack API routes registered")

    except Exception as e:
        logger.error(f"Failed to register LoRA stack routes: {e}")


_register_routes()
