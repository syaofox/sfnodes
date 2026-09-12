"""SFImageCropExpand 自定义比例预设持久化（user/sfnodes/crop_expand_presets.json）。

设计（对齐 text_presets.py 范式）：
- 全局库为真源：结构 {"presets": [{"name": str, "w": number, "h": number}]}，
  数组保序；跨工作流共享（点击已存定义=套用到本节点状态，不随工作流保存库）。
- 路由 /api/sfnodes/crop_expand_presets GET/POST/DELETE，import 时注册；
- 读改写 asyncio.Lock 互斥（text/lora_presets 同款）、临时名 + os.replace 原子写；
- 加载 mtime+size 缓存（text/krea2_presets 同款），避免反复磁盘 IO。

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
_RATIO_MAX = 10000.0  # w/h 上限，挡荒谬值


def _store_path():
    return os.path.join(_sf_user_dir(), "crop_expand_presets.json")


def _valid_ratio(v):
    """正有限数（bool 不算数）且在 (0, _RATIO_MAX] 内。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf")) and 0 < f <= _RATIO_MAX


def _normalize_presets(data):
    """归一化预设列表：接受文件级 dict（{"presets": [...]}）或裸列表；
    只保留 name 非空、w/h 为正有限数的条目，重名保留首个。"""
    raw = data.get("presets", []) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return []
    out = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen:
            continue
        w = item.get("w")
        h = item.get("h")
        if not _valid_ratio(w) or not _valid_ratio(h):
            continue
        seen.add(name)
        out.append({"name": name, "w": float(w), "h": float(h)})
    return out


def load_presets():
    """读取全局预设列表（mtime+size 变化自动重载，text_presets.load_presets 范式）。"""
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
            logger.warning("Failed to load crop expand presets: %s (%s)", path, e)
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
    logger.info("Saved crop expand presets: %s (%s presets)", path, len(presets))


def _valid_name(name) -> bool:
    return valid_name(name, max_len=_NAME_MAX_LEN)


def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, crop expand preset routes not registered")
            return
        routes = ins.routes
        path = "/api/sfnodes/crop_expand_presets"

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
                if not _valid_ratio(w) or not _valid_ratio(h):
                    return web.json_response({"error": "invalid ratio"}, status=400)
                name = name.strip()
                entry = {"name": name, "w": float(w), "h": float(h)}
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

        logger.info("Crop expand preset API routes registered")

    except Exception as e:
        logger.error(f"Failed to register crop expand preset routes: {e}")


_register_routes()
