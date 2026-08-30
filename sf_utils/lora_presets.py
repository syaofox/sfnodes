import asyncio
import json
import os
import threading

import folder_paths
from aiohttp import web

from .logger import get_logger

logger = get_logger(__name__)

# 预设 read-modify-write 互斥锁：两个并发请求同时读-改-写会互相覆盖
# （workflows meta 的 _WF_META_LOCK 同款）。
_presets_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# 约定：LoRA 预设（顺序 + 强度 + 提示词）统一存于
#     user/sfnodes/lora_presets.json
# 结构：{"presets": {"<名称>": {loras: [{lora,on,strength,strengthTwo}], positive?: string}}}
# positive 为可选正向提示词（与 triggers 分离保存，不自动拼接），旧预设缺省视 ""。
# ---------------------------------------------------------------------------

_PRESETS_PATH = os.path.join(
    folder_paths.get_user_directory(), "sfnodes", "lora_presets.json"
)


def _load_presets() -> dict:
    try:
        if not os.path.isfile(_PRESETS_PATH):
            return {}
        with open(_PRESETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        presets = data.get("presets", {}) if isinstance(data, dict) else {}
        return presets if isinstance(presets, dict) else {}
    except Exception:
        logger.warning(f"Failed to load lora presets: {_PRESETS_PATH}")
        return {}


def _save_presets(presets: dict) -> None:
    d = os.path.dirname(_PRESETS_PATH)
    os.makedirs(d, exist_ok=True)
    # 临时名带线程 id：并发写同一文件时抢同一个 .tmp 会写出混杂内容
    # （lora_reader.write_custom_store 同款做法）。
    tmp = f"{_PRESETS_PATH}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"presets": presets}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _PRESETS_PATH)
    logger.info(f"Saved lora presets: {_PRESETS_PATH} ({len(presets)} presets)")


def _valid_preset_name(name) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    if "/" in name or "\\" in name:
        return False
    if any(ord(c) < 32 for c in name):
        return False
    return True


_POSITIVE_MAX_LEN = 8000


def _sanitize_positive(v) -> str:
    if not isinstance(v, str):
        return ""
    s = v.strip()
    if len(s) > _POSITIVE_MAX_LEN:
        s = s[:_POSITIVE_MAX_LEN]
    return s


def _valid_preset_data(data) -> bool:
    if not isinstance(data, dict):
        return False
    loras = data.get("loras")
    if not isinstance(loras, list):
        return False
    for item in loras:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("lora"), str) or not item.get("lora"):
            return False
    # positive 可选；存在时必须为字符串
    if "positive" in data and not isinstance(data.get("positive"), str):
        return False
    return True


def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, routes not registered")
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/lora_presets")
        async def _list_presets(request: web.Request) -> web.Response:
            try:
                return web.json_response({"presets": _load_presets()})
            except Exception as e:
                logger.error(f"GET /api/sfnodes/lora_presets failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        @routes.post("/api/sfnodes/lora_presets")
        async def _save_preset(request: web.Request) -> web.Response:
            try:
                try:
                    body = await request.json()
                except Exception:
                    return web.json_response({"error": "invalid json"}, status=400)
                name = body.get("name", "")
                data = body.get("data")
                if not _valid_preset_name(name):
                    return web.json_response({"error": "invalid name"}, status=400)
                if not _valid_preset_data(data):
                    return web.json_response({"error": "invalid data"}, status=400)
                # 归一化 positive（截断、去首尾空白，空串不存储以保持旧预设精简）
                if isinstance(data.get("positive"), str):
                    data["positive"] = _sanitize_positive(data["positive"])
                    if not data["positive"]:
                        data.pop("positive", None)
                async with _presets_lock:
                    presets = _load_presets()
                    presets[name.strip()] = data
                    _save_presets(presets)
                return web.json_response({"ok": True, "name": name.strip()})
            except Exception as e:
                logger.error(f"POST /api/sfnodes/lora_presets failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        @routes.delete("/api/sfnodes/lora_presets")
        async def _delete_preset(request: web.Request) -> web.Response:
            try:
                name = request.rel_url.query.get("name", "")
                if not _valid_preset_name(name):
                    return web.json_response({"error": "invalid name"}, status=400)
                async with _presets_lock:
                    presets = _load_presets()
                    if name not in presets:
                        return web.json_response({"error": "not found"}, status=404)
                    del presets[name]
                    _save_presets(presets)
                return web.json_response({"deleted": name})
            except Exception as e:
                logger.error(f"DELETE /api/sfnodes/lora_presets failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        logger.info("LoRA presets API routes registered")

    except Exception as e:
        logger.error(f"Failed to register LoRA presets routes: {e}")


_register_routes()
