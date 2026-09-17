"""SFCanvasSizePreset 自定义分辨率预设持久化（user/sfnodes/canvas_size_presets.json）。

设计（对齐 crop_expand_presets.py / text_presets.py 范式）：
- 全局库为真源：结构 {"presets": [{"name": str, "w": int, "h": int}]}，数组
  保序；跨工作流共享（选中即把该分辨率用作本节点 resolution，不随工作流保存库）。
- 候选名编码约定：前端把条目编码为 "WxH (name)" 塞进 resolution combo，
  故名称不得含 '(' / ')'（否则 _parse_resolution 截断），保存时拒绝。
- 路由 /api/sfnodes/canvas_size_custom GET/POST/DELETE，import 时注册。
- 读改写 asyncio.Lock 互斥、临时名 + os.replace 原子写、mtime+size 缓存。

纯逻辑（load/save/校验/归一化）无 ComfyUI 节点依赖、可独立 mock 测试
（folder_paths 在 disk_state.sf_user_dir 内惰性 import）。
"""

import asyncio
import json
import os

from aiohttp import web

from .logger import get_logger
from .common import valid_name
from .disk_state import atomic_write_json, mtime_size_sig  # 原子写盘/文件签名（单源，见 disk_state）
from .disk_state import sf_user_dir as _sf_user_dir  # 用户数据统一目录（单源，见 disk_state）

logger = get_logger(__name__)

# 预设 read-modify-write 互斥锁：两个并发请求同时读-改-写会互相覆盖。
_lock = asyncio.Lock()

# 轻量缓存：{"sig": (mtime, size) | None, "data": [...]}，文件变化自动重载。
_cache = {"sig": None, "data": []}

_NAME_MAX_LEN = 200
_DIM_MAX = 32768  # 像素上限，挡荒谬值


def _store_path():
    return os.path.join(_sf_user_dir(), "canvas_size_presets.json")


def _valid_dim(v):
    """正整数（bool 不算数）且在 (0, _DIM_MAX] 内。"""
    if isinstance(v, bool) or not isinstance(v, int):
        return False
    return 0 < v <= _DIM_MAX


def _valid_name(name) -> bool:
    """valid_name + 禁括号（combo 编码 'WxH (name)' 依赖首个 '(' / ')'）。"""
    if not valid_name(name, max_len=_NAME_MAX_LEN):
        return False
    return "(" not in name and ")" not in name


def _normalize_presets(data):
    """归一化预设列表：接受文件级 dict（{"presets": [...]}）或裸列表；
    只保留 name 非空合规、w/h 为正整数的条目，重名保留首个。"""
    raw = data.get("presets", []) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return []
    out = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen or not _valid_name(name):
            continue
        w = item.get("w")
        h = item.get("h")
        if not _valid_dim(w) or not _valid_dim(h):
            continue
        seen.add(name)
        out.append({"name": name, "w": int(w), "h": int(h)})
    return out


def load_presets():
    """读取全局预设列表（mtime+size 变化自动重载，crop_expand_presets 范式）。"""
    path = _store_path()
    sig = mtime_size_sig(path)
    if _cache["sig"] == sig:
        return _cache["data"]
    data = []
    if sig is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _normalize_presets(json.load(f))
        except Exception as e:
            logger.warning("Failed to load canvas size presets: %s (%s)", path, e)
    _cache["sig"] = sig
    _cache["data"] = data
    return data


def save_presets(presets):
    """落盘（disk_state.atomic_write_json 临时文件 + os.replace 原子替换）。"""
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_write_json(path, {"presets": presets})
    try:
        st = os.stat(path)
        _cache["sig"] = (st.st_mtime, st.st_size)
    except OSError:
        pass
    _cache["data"] = _normalize_presets(presets)
    logger.info("Saved canvas size presets: %s (%s presets)", path, len(presets))


def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, canvas size preset routes not registered")
            return
        routes = ins.routes
        path = "/api/sfnodes/canvas_size_custom"

        @routes.get(path)
        async def _list(request: web.Request) -> web.Response:
            try:
                return web.json_response({"presets": load_presets()})
            except Exception as e:
                logger.error("GET %s failed: %s", path, e)
                return web.json_response({"error": "internal error"}, status=500)

        @routes.post(path)
        async def _save(request: web.Request) -> web.Response:
            try:
                try:
                    body = await request.json()
                except Exception:
                    return web.json_response({"error": "invalid json"}, status=400)
                name = (body or {}).get("name", "")
                w = (body or {}).get("w")
                h = (body or {}).get("h")
                if not _valid_name(name):
                    return web.json_response({"error": "invalid name"}, status=400)
                if not _valid_dim(w) or not _valid_dim(h):
                    return web.json_response({"error": "invalid dimensions"}, status=400)
                name = name.strip()
                entry = {"name": name, "w": int(w), "h": int(h)}
                async with _lock:
                    presets = load_presets()
                    for i, item in enumerate(presets):
                        if item["name"] == name:
                            presets[i] = entry
                            break
                    else:
                        presets.append(entry)
                    save_presets(presets)
                return web.json_response({"ok": True, "name": name})
            except Exception as e:
                logger.error("POST %s failed: %s", path, e)
                return web.json_response({"error": "internal error"}, status=500)

        @routes.delete(path)
        async def _delete(request: web.Request) -> web.Response:
            try:
                name = request.rel_url.query.get("name", "")
                if not _valid_name(name):
                    return web.json_response({"error": "invalid name"}, status=400)
                name = name.strip()
                async with _lock:
                    presets = load_presets()
                    remaining = [item for item in presets if item["name"] != name]
                    if len(remaining) == len(presets):
                        return web.json_response({"error": "not found"}, status=404)
                    save_presets(remaining)
                return web.json_response({"deleted": name})
            except Exception as e:
                logger.error("DELETE %s failed: %s", path, e)
                return web.json_response({"error": "internal error"}, status=500)

        logger.info("Canvas size preset API routes registered")

    except Exception as e:
        logger.error(f"Failed to register canvas size preset routes: {e}")


_register_routes()
