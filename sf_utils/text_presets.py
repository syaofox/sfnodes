"""SFTextPreset 预设持久化（user/sfnodes/text_presets.json）。

设计：
- 全局库为真源：结构 {"presets": [{"name": str, "text": str}]}，数组保序；
- 旧工作流兼容：节点 execute 未命中全局库时回退解析节点隐藏输入 presets_json
  （nodes/text/text_preset.py），前端 combo 同时显示全局库与工作流残留项；
- 路由 /api/sfnodes/text_presets GET/POST/DELETE，import 时注册（lora_presets.py 同款）；
- 读改写 asyncio.Lock 互斥（lora_presets.py 同款）、线程 id 临时名 + os.replace 原子写；
- 加载 mtime+size 缓存（krea2_presets.py 范式），execute 每次排队都要读，避免反复磁盘 IO。

纯逻辑（load/save/校验/查找）无 ComfyUI 节点依赖、可独立 mock 测试
（folder_paths 在 _sf_user_dir 内惰性 import，krea2_presets.py 同款）。
"""

import asyncio
import json
import os
import threading

from aiohttp import web

from .logger import get_logger
from .common import valid_name
from .disk_state import sf_user_dir as _sf_user_dir  # 用户数据统一目录（单源，见 disk_state）

logger = get_logger(__name__)

# 预设 read-modify-write 互斥锁：两个并发请求同时读-改-写会互相覆盖
# （lora_presets._presets_lock 同款）。
_lock = asyncio.Lock()

# 轻量缓存：{"sig": (mtime, size) | None, "data": [...]}，文件变化自动重载。
_cache = {"sig": None, "data": []}

_NAME_MAX_LEN = 200
_TEXT_MAX_LEN = 20000


def _store_path():
    return os.path.join(_sf_user_dir(), "text_presets.json")


def _normalize_presets(data):
    """归一化预设列表：接受文件级 dict（{"presets": [...]}) 或裸列表；
    只保留 {name, text} 且 name 非空的条目，重名保留首个。"""
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
        seen.add(name)
        out.append({"name": name, "text": str(item.get("text", ""))})
    return out


def load_presets():
    """读取全局预设列表（mtime+size 变化自动重载，krea2_presets.load_store 范式）。"""
    path = _store_path()
    try:
        st = os.stat(path)
        sig = (st.st_mtime, st.st_size)
    except OSError:
        sig = None
    if _cache["sig"] == sig:
        return _cache["data"]
    data = []
    if sig is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _normalize_presets(json.load(f))
        except Exception as e:
            logger.warning("Failed to load text presets: %s (%s)", path, e)
    _cache["sig"] = sig
    _cache["data"] = data
    return data


def save_presets(presets):
    """落盘（线程安全：临时名带线程 id，os.replace 原子替换，lora_presets 同款）。"""
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "{}.{}.tmp".format(path, threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"presets": presets}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    try:
        st = os.stat(path)
        _cache["sig"] = (st.st_mtime, st.st_size)
    except OSError:
        pass
    _cache["data"] = _normalize_presets(presets)
    logger.info("Saved text presets: %s (%s presets)", path, len(presets))


def find_text(name):
    """按名查全局预设文本；未命中返回 None（与"存在但文本为空"区分）。"""
    if not name:
        return None
    for item in load_presets():
        if item["name"] == name:
            return item["text"]
    return None


def _valid_name(name) -> bool:
    return valid_name(name, max_len=_NAME_MAX_LEN)


def _valid_text(text) -> bool:
    return isinstance(text, str) and len(text) <= _TEXT_MAX_LEN


def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, text preset routes not registered")
            return
        routes = ins.routes
        path = "/api/sfnodes/text_presets"

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
                text = (body or {}).get("text", "")
                if not _valid_name(name):
                    return web.json_response({"error": "invalid name"}, status=400)
                if not _valid_text(text):
                    return web.json_response({"error": "invalid text"}, status=400)
                name = name.strip()
                async with _lock:
                    presets = load_presets()
                    for item in presets:
                        if item["name"] == name:
                            item["text"] = text
                            break
                    else:
                        presets.append({"name": name, "text": text})
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

        logger.info("Text presets API routes registered")

    except Exception as e:
        logger.error(f"Failed to register text presets routes: {e}")


_register_routes()
