"""LoRA 用户数据网关（SFPowerLoraLoader / SFLoraLoader / SFLoraLoaderModelOnly 共用）。

2026-08 起与 SFLoraStack 统一存储：用户自定义触发词/描述的单一真源为
<user>/sfnodes/lora_triggers.json（路径见 lora_routes._custom_triggers_file），
本模块只是 Power 系（旧信息对话框 + loader 节点 execute 输出）的读写网关：

  - 读（get_merged_metadata）：文件内嵌元数据（safetensors 头部）+ Civitai
    侧车（<base>.civitai.info）+ 统一存储（优先）三源合并，返回形状与旧
    /lora_notes 兼容 {trigger_words, description, base_model, source_url,
    _has_custom, _has_embedded}——前端对话框与 SFLoraLoader 节点的输出不变。
  - 写（set_custom_notes）：trigger_words 字符串按逗号/换行拆分为 words
    数组（SFLoraStack 的 chips 模型），连同 description 与内容指纹写入
    统一存储；空数据清空条目。
  - 旧 <模型>.sf.json 侧车：任一读取入口首次读到该 LoRA 时惰性迁移并入
    统一存储后删除（lora_reader.migrate_legacy_sidecar），本模块常态路径
    不再读写侧车。

仅支持 loras 类型（旧 ?type= 泛化移除——三个消费节点本就用 loras；
统一存储的 key 空间无类型维度，混入其它目录类型会撞 key）。
"""
import os

import folder_paths
from aiohttp import web

from . import lora_reader as R
from .logger import get_logger
from .lora_routes import _custom_triggers_file  # 统一存储路径（单一实现，杜绝双真源）

logger = get_logger(__name__)


def _resolve_lora_path(filename):
    """LoRA 相对路径 -> 位于 loras 根目录内的真实路径，或 None。"""
    if not filename or not isinstance(filename, str):
        return None
    try:
        full = folder_paths.get_full_path("loras", filename)
    except Exception:
        return None
    if not full or not os.path.isfile(full):
        return None
    # 防御：确认落在 loras 已注册的根目录内
    for root in folder_paths.get_folder_paths("loras"):
        if full.startswith(os.path.normpath(root) + os.sep):
            return full
    return None


def _embedded_trigger_words(meta):
    """内嵌元数据 -> 触发词字符串。推导逻辑与 SFLoraStack 同源
    （lora_reader.derive_trigger_words：显式短语优先、标签频率兜底）。"""
    if not isinstance(meta, dict):
        return ""
    return ", ".join(R.derive_trigger_words(meta))


def _embedded_description(meta):
    if not isinstance(meta, dict):
        return ""
    return (
        meta.get("modelspec.description")
        or meta.get("modelspec.title")
        or meta.get("modelspec_description")
        or meta.get("modelspec_title")
        or ""
    )


def _embedded_base_model(meta):
    if not isinstance(meta, dict):
        return ""
    return (
        meta.get("ss_base_model_version")
        or meta.get("modelspec.base_model")
        or meta.get("modelspec_base_model")
        or meta.get("base_model")
        or ""
    )


def get_merged_metadata(filename):
    """一个 LoRA 的合并元数据（前端对话框 + loader 节点 execute 输出共用）。

    形状与旧版 /lora_notes 兼容。自定义数据优先级：统一存储
    （lora_triggers.json）> Civitai 侧车（.civitai.info）> 文件内嵌元数据。
    文件缺失返回 {"_not_found": True}。永不抛错。
    """
    path = _resolve_lora_path(filename)
    if path is None:
        return {"_not_found": True}
    # 惰性迁移旧 .sf.json 侧车（store 已有该 LoRA 数据时幂等跳过）
    try:
        R.migrate_legacy_sidecar(_custom_triggers_file(), path, filename)
    except Exception:
        pass
    meta = R.read_safetensors_metadata(path)
    side = R.read_sidecar_info(path) or {}
    try:
        entry = R.read_custom_store(_custom_triggers_file()).get(
            R.custom_trigger_key(filename), {}
        ) or {}
    except Exception:
        entry = {}

    words = list(entry.get("words") or [])
    # 自定义描述先单独留存：_has_custom 只看统一存储里有没有用户数据，
    # desc 随后会被 sidecar/embedded 兜底覆盖，不能复用同一变量。
    entry_desc = entry.get("description") or ""
    desc = entry_desc
    if words:
        trigger_words = ", ".join(words)
    elif side.get("triggers"):
        trigger_words = ", ".join(side["triggers"])
    else:
        trigger_words = _embedded_trigger_words(meta)
    if not desc:
        desc = side.get("description") or _embedded_description(meta)
    return {
        "trigger_words": trigger_words,
        "description": desc,
        "base_model": _embedded_base_model(meta) or side.get("base_model", ""),
        "source_url": meta.get("source_url", "") if isinstance(meta, dict) else "",
        # 高亮语义（i 图标"有信息"标记）：统一存储有用户词/描述，或
        # .civitai.info 侧车有词/描述（用户主动查过 Civitai 获取的信息）。
        # 刻意不含 embedded——文件自带词/描述几乎人人都有，无区分度。
        "_has_custom": bool(words or entry_desc)
        or bool(side.get("triggers"))
        or bool(side.get("description")),
        "_has_embedded": bool(meta),
    }


def set_custom_notes(filename, data):
    """保存/清空一个 LoRA 的用户自定义触发词与描述（统一存储网关）。

    `data` 是旧 /lora_notes 形状 {trigger_words: str, description: str}；
    trigger_words 按逗号/换行拆分为数组写入（SFLoraStack 的 chips 模型，
    lora_reader.split_trigger_text 同源）。空数据清空条目（删除存储键）。
    返回 get_merged_metadata 形状（前端回填兼容）。永不抛错。
    """
    path = _resolve_lora_path(filename)
    if path is None:
        return {}
    data = data if isinstance(data, dict) else {}
    words = R.split_trigger_text(data.get("trigger_words"))
    desc = data.get("description")
    desc = desc if isinstance(desc, str) else ""
    fp = R.file_fingerprint(path)
    store_path = _custom_triggers_file()
    R.set_custom_triggers(store_path, filename, words, fp)
    R.set_custom_description(store_path, filename, desc, fp)
    return get_merged_metadata(filename)


# ---------------------------------------------------------------------------
# HTTP route registration
# ---------------------------------------------------------------------------

def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, routes not registered")
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/lora_notes")
        async def _get_notes(request: web.Request) -> web.Response:
            try:
                filename = request.query.get("filename", "")
                if not filename:
                    return web.json_response({"error": "filename required"}, status=400)
                return web.json_response(get_merged_metadata(filename))
            except Exception as e:
                logger.error(f"GET /api/sfnodes/lora_notes failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        @routes.post("/api/sfnodes/lora_notes")
        async def _save_notes(request: web.Request) -> web.Response:
            try:
                filename = request.query.get("filename", "")
                if not filename:
                    return web.json_response({"error": "filename required"}, status=400)
                body = await request.json()
                if not isinstance(body, dict):
                    return web.json_response({"error": "json object required"}, status=400)
                data = set_custom_notes(filename, body)
                logger.info(f"Saved notes for {filename}")
                return web.json_response(data)
            except Exception as e:
                logger.error(f"POST /api/sfnodes/lora_notes failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        @routes.delete("/api/sfnodes/lora_notes")
        async def _delete_notes(request: web.Request) -> web.Response:
            try:
                filename = request.query.get("filename", "")
                if not filename:
                    return web.json_response({"error": "filename required"}, status=400)
                return web.json_response(set_custom_notes(filename, {}))
            except Exception as e:
                logger.error(f"DELETE /api/sfnodes/lora_notes failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        logger.info("LoRA notes API routes registered")

    except Exception as e:
        logger.error(f"Failed to register LoRA notes routes: {e}")


_register_routes()

# 导入以触发 LoRA 示例图（sample）API 路由注册
from . import lora_samples  # noqa: F401, E402
