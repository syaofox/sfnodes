"""LoRA 用户数据网关（SFLoraStack / SFLoraLoader / SFLoraLoaderModelOnly 共用）。

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
from .lora_routes import _custom_triggers_file, _is_path_under, _previews_dir  # 统一存储/预览路径（单一实现，杜绝双真源）

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
    # 防御：确认落在 loras 已注册的根目录内（realpath + 跨盘 lexical 回退，见 lora_routes._is_path_under）
    try:
        roots = folder_paths.get_folder_paths("loras")
    except Exception:
        roots = []
    if roots and not _is_path_under(full, *roots):
        return None
    if not roots:
        # 无根目录配置时 fail-closed
        return None
    return full


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


def _find_orphan_entry(store, filename, path=None):
    """store 中与当前 LoRA 唯一匹配的非空孤儿条目（key 不同、数据非空）。

    文件存在时指纹优先（内容级证据：改名/移动后内容不变）、基名兜底；
    文件不存在（path=None）时仅基名（无文件可算指纹）。同名多目录歧义
    由 lora_reader.find_orphan_* 放弃（返回 None）。永不抛错。
    返回 (entry, orphan_key) 或 None。
    """
    key = R.custom_trigger_key(filename)
    if not key:
        return None
    if path is not None:
        try:
            fp = R.file_fingerprint(path)
        except Exception:
            fp = None
        if fp:
            ok = R.find_orphan_by_fingerprint(store, fp, exclude=key)
            if ok:
                e = store.get(ok)
                if e and (e.get("words") or e.get("description")):
                    return (e, ok)
    ok = R.find_orphan_key(store, filename)
    if ok:
        e = store.get(ok)
        if e and (e.get("words") or e.get("description")):
            return (e, ok)
    return None


def _orphan_meta(entry, orphan_key):
    """孤儿条目 -> merged 形状（文件数据缺失时只有 store 的用户数据）。"""
    return {
        "trigger_words": ", ".join(entry.get("words") or []),
        "description": entry.get("description", ""),
        "base_model": "",
        "source_url": "",
        "_has_custom": True,
        "_has_embedded": False,
        "_file_missing": True,
        "orphan_key": orphan_key,
    }


def get_merged_metadata(filename):
    """一个 LoRA 的合并元数据（前端对话框 + loader 节点 execute 输出共用）。

    形状与旧版 /lora_notes 兼容。自定义数据优先级：统一存储
    （lora_triggers.json）> Civitai 侧车（.civitai.info）> 文件内嵌元数据。
    文件不存在时按基名从统一存储孤儿兜底（改名/移动后数据仍可读，
    附 orphan_key/_file_missing 让前端提示重新选择路径）。永不抛错。
    """
    path = _resolve_lora_path(filename)
    if path is None:
        # 文件不存在：统一存储基名孤儿兜底（指纹不可用——无文件可算内容指纹）
        try:
            store = R.read_custom_store(_custom_triggers_file())
        except Exception:
            store = {}
        found = _find_orphan_entry(store, filename, None)
        if found:
            return _orphan_meta(*found)
        return {"_not_found": True}
    # 惰性迁移旧 .sf.json 侧车（store 已有该 LoRA 数据时幂等跳过）
    try:
        R.migrate_legacy_sidecar(_custom_triggers_file(), path, filename)
    except Exception:
        pass
    meta = R.read_safetensors_metadata(path)
    side = R.read_sidecar_info(path) or {}
    try:
        store = R.read_custom_store(_custom_triggers_file())
    except Exception:
        store = {}
    entry = store.get(R.custom_trigger_key(filename), {}) or {}

    words = list(entry.get("words") or [])
    # 自定义描述先单独留存：_has_custom 只看统一存储里有没有用户数据，
    # desc 随后会被 sidecar/embedded 兜底覆盖，不能复用同一变量。
    entry_desc = entry.get("description") or ""
    orphan_key = ""
    # 孤儿明细（对齐 SFLoraStack 面板 /lora_info 形状）：迁移提示条显示
    # 旧路径下有什么可迁移（词/描述/预览图）。仅孤儿命中时非空。
    orphan_triggers = []
    orphan_description = ""
    orphan_preview = False
    if not words and not entry_desc:
        # 本 key 无数据：孤儿检测找回改名/移动前的数据（文件存在 -> 指纹+基名）
        found = _find_orphan_entry(store, filename, path)
        if found:
            entry, orphan_key = found
            words = list(entry.get("words") or [])
            entry_desc = entry.get("description") or ""
            orphan_triggers = list(entry.get("words") or [])
            orphan_description = entry.get("description") or ""
            orphan_preview = bool(R.find_custom_preview(_previews_dir(), orphan_key))
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
        "orphan_key": orphan_key,
        "orphan_triggers": orphan_triggers,
        "orphan_description": orphan_description,
        "orphan_preview": orphan_preview,
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
