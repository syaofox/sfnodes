"""SF Load Diffusion Model 后端路由（模块导入时注册，lora_routes 先例）。

只实现 info 组装与 LoRA 域真正不同的那一个：GET /api/sfnodes/dmodel_info。
Civitai 查询（by-hash + 页面描述）、用户自定义描述、预览图保存/删除、孤儿
迁移/合并等与 LoRA 完全同构的路由不在此重复——由 lora_routes._register_routes
以别名注册到 /api/sfnodes/dmodel/*，handler 内按请求路径分派存储域
（dmodels.json / previews_model/ / diffusion_models 目录，见 lora_routes
模块头"数据域分派"注释）。

info 组装差异：扩散模型没有触发词与 LoRA 训练参数概念——三组触发词恒为
空数组（前端 hideTriggers 整块隐藏），meta 行改为架构字符串 + 文件大小；
其余字段（title/description 三档/source/has_preview/model_id 等）与
R.build_lora_info 同形状，让 sf_lora_stack_info.js 面板零分支消费。
侧车缓存（<base>.civitai.info）跟随模型文件本身，两域天然隔离。
"""
import asyncio
import os

from aiohttp import web

from .logger import get_logger
from . import lora_reader as R
from .lora_routes import (
    _resolve_model_path,
    _dmodels_file,
    _previews_model_dir,
    _civitai_account,
)

logger = get_logger(__name__)


def _human_size(n):
    """字节数 -> 人读尺寸字符串；非法输入返回 ""。"""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return ""
    if n < 0:
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            if unit == "B":
                return "{} {}".format(int(n), unit)
            return "{:.1f} {}".format(n, unit)
        n /= 1024.0
    return ""


def _arch_from_meta(meta):
    """从 safetensors __metadata__ 猜架构显示串；猜不到返回 ""。

    来源按可靠性排序：modelspec.architecture（kohya 系）>
    ss_base_model_version（SD 训练脚本）> config JSON（ComfyUI 系把模型
    config 存在头部，含 model_type/architecture 字段）。永不抛错。"""
    if not isinstance(meta, dict):
        return ""
    for k in ("modelspec.architecture", "ss_base_model_version", "architecture"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raw = meta.get("config")
    if isinstance(raw, str) and raw.strip():
        try:
            import json as _json

            cfg = _json.loads(raw)
        except Exception:
            return ""
        if isinstance(cfg, dict):
            for k in ("architecture", "_class_name", "model_type"):
                v = cfg.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ""


def _build_dmodel_info(path, name):
    """一个扩散模型的统一离线 info（形状对齐 R.build_lora_info）。永不抛错。

    触发词恒空：diffusion 模型没有触发词，面板经 ctx.hideTriggers 隐藏该
    区块，这里保留键只为响应形状一致。size_h 是本域新增（meta 行展示）。"""
    meta = R.read_safetensors_metadata(path)
    file_desc = R._html_to_markdown(meta.get("modelspec.description"))
    try:
        st = os.stat(path)
        size_h = _human_size(st.st_size)
        mtime = int(st.st_mtime)
    except Exception:
        size_h = ""
        mtime = 0
    info = {
        "title": R._title_from_meta(meta, path),
        "base_model": _arch_from_meta(meta),
        "rank": "",
        "alpha": "",
        "num_images": "",
        "date": meta.get("modelspec.date", "") or "",
        "description": file_desc,
        "file_description": file_desc,
        "civitai_description": "",
        "triggers": [],
        "file_triggers": [],
        "sidecar_triggers": [],
        "source": "file",
        "has_preview": R.find_preview_path(path) is not None,
        "size": size_h,
        "mtime": mtime,
    }
    # 侧车（先前 Civitai 查询缓存，跟随文件）：标题/架构/描述胜出，语义同 build_lora_info。
    side = R.read_sidecar_info(path)
    if side.get("name"):
        info["title"] = side["name"]
    if side.get("base_model"):
        info["base_model"] = side["base_model"]
    if side.get("description"):
        info["description"] = side["description"]
        info["civitai_description"] = side["description"]
        info["source"] = "sidecar"
    if side.get("model_id") is not None:
        info["model_id"] = side["model_id"]
    if side.get("version_id") is not None:
        info["version_id"] = side["version_id"]
    return info


def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, dmodel routes not registered")
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/dmodel_info")
        async def api_dmodel_info(request):
            """一个扩散模型的离线 info（信息面板）。恒 200 让前端不必按
            HTTP 状态分支；只读文件头部 + 侧车，绝不加载权重。"""
            name = request.query.get("name", "")
            path = _resolve_model_path(name)
            if not path:
                # 文件不存在（改名/移动后旧值）：同 lora_info 的孤儿兜底，
                # 返回旧键数据 + _file_missing 让面板提示重选。
                try:
                    store = R.read_custom_store(_dmodels_file())
                    orphan = R.find_orphan_key(store, name)
                    if orphan:
                        entry = store.get(orphan) or {}
                        info = {
                            "title": name.rsplit("/", 1)[-1] or name,
                            "base_model": "",
                            "description": entry.get("description", ""),
                            "triggers": [],
                            "file_triggers": [],
                            "sidecar_triggers": [],
                            "source": "custom",
                            "has_preview": False,
                            "custom_triggers": [],
                            "custom_description": entry.get("description", ""),
                            "orphan_key": orphan,
                            "orphan_description": entry.get("description", ""),
                            "orphan_preview": False,
                            "preview_v": 0,
                            "custom_preview": False,
                            "restorable_thumb": False,
                            "civitai_host": _civitai_account().get("host", "com"),
                            "size": "",
                            "_file_missing": True,
                        }
                        return web.json_response({"ok": True, "info": info},
                                                 headers={"Cache-Control": "no-store"})
                except Exception:
                    pass
                return web.json_response({"ok": False, "message": "Model not found."})
            loop = asyncio.get_running_loop()
            try:
                info = await loop.run_in_executor(None, _build_dmodel_info, path, name)
            except Exception as exc:
                return web.json_response({"ok": False, "message": "Could not read: {}".format(exc)})
            # 用户自定义描述（dmodels.json 单源）。触发词域不适用，键留空数组。
            try:
                info["custom_description"] = R.get_custom_description(_dmodels_file(), name)
            except Exception:
                info["custom_description"] = ""
            info["custom_triggers"] = []
            # 面板头部 "View on Civitai" 链接按账户主机偏好生成。
            try:
                info["civitai_host"] = _civitai_account().get("host", "com")
            except Exception:
                info["civitai_host"] = "com"
            # 孤儿数据检测（指纹优先基名兜底），语义同 lora_info。
            try:
                store = R.read_custom_store(_dmodels_file())
                orphan = None
                fp = await loop.run_in_executor(None, R.file_fingerprint, path)
                if fp:
                    orphan = R.find_orphan_by_fingerprint(store, fp, exclude=R.custom_trigger_key(name))
                if orphan is None:
                    orphan = R.find_orphan_key(store, name)
                if orphan:
                    entry = store.get(orphan, {})
                    info["orphan_key"] = orphan
                    info["orphan_description"] = entry.get("description", "")
                    info["orphan_preview"] = bool(R.find_custom_preview(_previews_model_dir(), orphan))
            except Exception:
                pass
            # 用户预览图状态（previews_model/ 独立槽位目录）。
            try:
                folder = _previews_model_dir()
                info["preview_v"] = R.custom_preview_version(folder, name)
                info["custom_preview"] = bool(info["preview_v"])
                if info["custom_preview"]:
                    info["has_preview"] = True
            except Exception:
                info["custom_preview"] = False
                info["preview_v"] = 0
            # 封面恢复标志（侧车有缩略图 URL 而本地无预览时前端静默重下载）。
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

        logger.info("Diffusion model API routes registered")

    except Exception as e:
        logger.error(f"Failed to register diffusion model routes: {e}")


# lora_routes/lora_samples 同款：模块尾调用 = 导入时副作用注册。
# 触发点：nodes/model/load_diffusion_model.py 尾行（节点总被根 __init__
# 加载，路由随之必注册）。
_register_routes()
