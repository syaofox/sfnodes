"""Krea2 预设管理（SFImageInterrogator 反推预设 + SFKrea2SystemPrompt 系统指令预设）。

设计（见 doc/experience.md §31）：
- 内置默认：krea2.py 硬编码的 INTERROGATOR_PRESETS / KREA2_PRESETS 为默认源；
- 用户覆盖：<user>/sfnodes/{kind}_presets.json 存用户改动，结构
      {"overrides": {"<名>": "文本"}, "deleted": ["<内置名>"]}
  overrides 既用于修改内置（按名覆盖、保持内置位置）也用于新增（追加到末尾）；
  deleted 是墓碑，标记被删除的内置预设（复位=清除墓碑还原）。
- merge() 纯函数把内置 + 用户存储合并为最终 {name: text}（combo 与 API 暴露它）。

纯逻辑（merge/校验/读写）无节点依赖、可独立 mock 测试（对齐 lora_presets.py 范式）；
路由由 krea2.py 在 import 时调用 register(kind, builtin) 注册（内置 dict 此刻才齐全）。
"""

import asyncio
import json
import os
import threading

from aiohttp import web

from .logger import get_logger

logger = get_logger(__name__)

_KINDS = ("interrogator", "krea2")

# 每 kind 一把读写锁：并发读-改-写互相覆盖（lora_presets._presets_lock 同款）。
_locks = {kind: asyncio.Lock() for kind in _KINDS}

# 已注册的内置预设（register() 写入），供路由读取。
_builtin = {}

# 受保护预设名（不可删除/复位，如 Krea2SystemPrompt 的 "none" 虚拟项）。
_protected = {}

# 轻量缓存：{kind: ((mtime, size), store)}，文件变化自动重载（prompt_preset 范式）。
_store_cache = {}


def _sf_user_dir():
    """<ComfyUI user dir>/sfnodes —— 本项目用户数据统一目录（styles_selector 同款本地镜像，
    避免拉入 lora_routes 的重依赖）。"""
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


def _store_path(kind):
    return os.path.join(_sf_user_dir(), "{}_presets.json".format(kind))


def _normalize_store(data):
    if not isinstance(data, dict):
        return {"overrides": {}, "deleted": []}
    overrides = data.get("overrides", {})
    deleted = data.get("deleted", [])
    if not isinstance(overrides, dict):
        overrides = {}
    if not isinstance(deleted, list):
        deleted = []
    return {
        "overrides": {str(k): v for k, v in overrides.items() if isinstance(k, str) and isinstance(v, str)},
        "deleted": [n for n in deleted if isinstance(n, str)],
    }


def load_store(kind):
    """读取用户存储（线程安全；mtime+size 变化自动重载）。"""
    path = _store_path(kind)
    try:
        st = os.stat(path)
        sig = (st.st_mtime, st.st_size)
    except OSError:
        sig = None
    cached = _store_cache.get(kind)
    if cached and cached[0] == sig:
        return cached[1]
    store = {"overrides": {}, "deleted": []}
    if sig is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                store = _normalize_store(json.load(f))
        except Exception:
            logger.warning("Failed to load krea2 presets store: %s", path)
    _store_cache[kind] = (sig, store)
    return store


def save_store(kind, store):
    """落盘（线程安全：临时名带线程 id，os.replace 原子替换）。"""
    path = _store_path(kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "{}.{}.tmp".format(path, threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    try:
        st = os.stat(path)
        _store_cache[kind] = ((st.st_mtime, st.st_size), _normalize_store(store))
    except OSError:
        pass
    logger.info("Saved krea2 presets: %s (%s)", path, kind)


def _valid_name(name) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    if "/" in name or "\\" in name:
        return False
    if any(ord(c) < 32 for c in name):
        return False
    return True


def _valid_text(text) -> bool:
    return isinstance(text, str) and len(text) <= 20000


def merge(builtin, store):
    """合并内置 + 用户存储为最终 {name: text}。

    - 以内置顺序为基准：墓碑删除；overrides 按名覆盖文本（保持内置位置）；
    - overrides 中内置没有的名字（用户新增）按插入序追加到末尾。
    """
    overrides = store.get("overrides", {}) if isinstance(store, dict) else {}
    deleted = set(store.get("deleted", [])) if isinstance(store, dict) else set()
    out = {}
    for name, text in builtin.items():
        if name in deleted:
            continue
        out[name] = overrides.get(name, text)
    for name, text in overrides.items():
        if name not in builtin and name not in out:
            out[name] = text
    return out


def merged(kind):
    """当前生效的合并预设 {name: text}（节点执行/前端 API 用）。"""
    return merge(_builtin.get(kind, {}), load_store(kind))


def _register_routes(kind):
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, krea2 %s preset routes not registered", kind)
            return
        routes = ins.routes
        path = "/api/sfnodes/{}_presets".format(kind)

        @routes.get(path)
        async def _list(request: web.Request) -> web.Response:
            try:
                builtin = _builtin.get(kind, {})
                store = load_store(kind)
                return web.json_response({
                    "presets": merge(builtin, store),
                    "builtin": builtin,
                    "user": store.get("overrides", {}),
                    "deleted": store.get("deleted", []),
                })
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
                async with _locks[kind]:
                    store = load_store(kind)
                    store.setdefault("overrides", {})[name] = text
                    # 修改/新增后清除墓碑（复活被删的内置）
                    deleted = store.get("deleted", [])
                    if name in deleted:
                        store["deleted"] = [n for n in deleted if n != name]
                    save_store(kind, store)
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
                if name in _protected.get(kind, ()):
                    return web.json_response({"error": "protected"}, status=400)
                async with _locks[kind]:
                    builtin = _builtin.get(kind, {})
                    store = load_store(kind)
                    if name in builtin:
                        # 内置：记墓碑并移除 override（改文本+删并存时以删为准）
                        deleted = store.setdefault("deleted", [])
                        if name not in deleted:
                            deleted.append(name)
                        store.setdefault("overrides", {}).pop(name, None)
                        save_store(kind, store)
                        return web.json_response({"deleted": name, "builtin": True})
                    overrides = store.setdefault("overrides", {})
                    if name not in overrides:
                        return web.json_response({"error": "not found"}, status=404)
                    del overrides[name]
                    save_store(kind, store)
                    return web.json_response({"deleted": name, "builtin": False})
            except Exception as e:
                logger.error("DELETE %s failed: %s", path, e)
                return web.json_response({"error": "internal error"}, status=500)

        @routes.post(path + "/reset")
        async def _reset(request: web.Request) -> web.Response:
            try:
                try:
                    body = await request.json()
                except Exception:
                    body = {}
                all_ = bool((body or {}).get("all", False))
                async with _locks[kind]:
                    store = load_store(kind)
                    if all_:
                        if store.get("overrides") or store.get("deleted"):
                            save_store(kind, {"overrides": {}, "deleted": []})
                        return web.json_response({"reset": "all"})
                    name = (body or {}).get("name", "")
                    if not _valid_name(name):
                        return web.json_response({"error": "invalid name"}, status=400)
                    if name in _protected.get(kind, ()):
                        return web.json_response({"error": "protected"}, status=400)
                    changed = False
                    if store.setdefault("overrides", {}).pop(name, None) is not None:
                        changed = True
                    deleted = store.get("deleted", [])
                    if name in deleted:
                        store["deleted"] = [n for n in deleted if n != name]
                        changed = True
                    if changed:
                        save_store(kind, store)
                    return web.json_response({"reset": name})
            except Exception as e:
                logger.error("POST %s/reset failed: %s", path, e)
                return web.json_response({"error": "internal error"}, status=500)

        logger.info("Krea2 %s preset API routes registered", kind)
    except Exception as e:
        logger.error("Failed to register krea2 %s preset routes: %s", kind, e)


def register(kind, builtin, protected=()):
    """注册某 kind 的预设管理路由，并记录内置默认源与受保护名（如 "none"）。"""
    if kind not in _KINDS:
        return
    _builtin[kind] = dict(builtin)
    _protected[kind] = tuple(protected)
    _register_routes(kind)
