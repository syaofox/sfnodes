"""SFPauseImage 保存路由（/api/sfnodes/preview/save + /prepare + /flip + /pause/load）。

复刻 Pixaroma 的 /pixaroma/api/preview/save 与 /prepare（拖回重建元数据用）：
- save：把 base64 PNG 存到 ComfyUI output/ 目录，嵌入 workflow/prompt PNG 块
  （保存的图片可拖回 ComfyUI 重建同一张图）
- prepare：嵌入元数据后返回 data URI + 建议文件名（Save to Disk 用）
- flip：把 temp 闸门快照 PNG 原地水平镜像（SF Pause Image 的 Flip 开关；就地
  镜像让预览 / Continue / Copy / Save / Open 全部一致，无需重跑上游）
- pause/load：把 input/ 里的外载图物化成 temp 快照（Load/Browse/拖放/粘贴，
  见 web/sf_pause_source.js；Pause/Pass 仍优先接线图）

注册方式沿用 sf_utils/lora_notes.py 的 _register_routes 先例：模块导入时
（__init__.py import）副作用注册，try/except 包裹，环境异常时降级不注册。
"""

import base64
import io
import json
import os

import folder_paths
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from ...sf_utils.common import json_safe as _json_safe  # NaN/Inf 清洗（单源，见 common）
from ...sf_utils.disk_state import decode_image as _decode_image  # dataURL 解码（单源，见 disk_state）
from ...sf_utils.disk_state import safe_prefix as _safe_prefix  # 文件名前缀清洗（单源，见 disk_state）
from .pause_image import _snapshot_path  # 外载图物化目标（temp 快照路径单源）


def _metadata_disabled():
    """尊重 ComfyUI 全局 --disable-metadata（与原生 SaveImage 一致）。

    每次调用实时读取（不缓存）：args 在启动期间填充，我们与 comfy 的导入顺序
    不保证，import 时快照可能是解析前的默认值。失败开放（返回 False）：
    flag 或整个模块缺失时保持现状，而不是静默丢失每个嵌入的工作流。
    """
    try:
        from comfy.cli_args import args as _comfy_cli_args
        return bool(getattr(_comfy_cli_args, "disable_metadata", False))
    except Exception:
        return False


def _build_pnginfo(prompt=None, workflow=None, parameters=None):
    """按 ComfyUI SaveImage 惯例构建 PngInfo（prompt/workflow 块）。

    两个约定同时支持：路由侧传 dict（app.graphToPrompt() 的 JSON 可序列化
    对象）；任意参数可为 None（对应块跳过）。--disable-metadata 时返回空
    PngInfo（PIL 不写任何 tEXt 块）。NaN/Inf 经 _json_safe 清洗，保证拖回
    ComfyUI 时 prompt/workflow 是合法 JSON。
    """
    pnginfo = PngInfo()
    if _metadata_disabled():
        return pnginfo
    if prompt is not None:
        try:
            pnginfo.add_text("prompt", prompt if isinstance(prompt, str) else json.dumps(_json_safe(prompt)))
        except Exception:
            pass
    if workflow is not None:
        try:
            pnginfo.add_text("workflow", workflow if isinstance(workflow, str) else json.dumps(_json_safe(workflow)))
        except Exception:
            pass
    if parameters:
        pnginfo.add_text("parameters", str(parameters))
    return pnginfo


def _mirror_png(path):
    """把 PNG 文件原地水平镜像（左右翻转）。

    先用 with 打开取出像素与文本块（PIL 惰性解码，需要显式 load 到内存镜像
    对象），句柄释放后再覆盖保存——否则 Windows 文件锁会让保存失败。文本块
    经 _build_pnginfo 重建（尊重 --disable-metadata）；闸门快照通常无文本块。
    """
    with Image.open(path) as snap:
        info = dict(snap.info)
        mirrored = ImageOps.mirror(snap)
    pnginfo = _build_pnginfo(
        prompt=info.get("prompt"),
        workflow=info.get("workflow"),
        parameters=info.get("parameters"),
    )
    mirrored.save(path, "PNG", pnginfo=pnginfo)


def _resolve_input_path(src_path):
    """把 "subfolder/file.png" 安全解析到 input/ 下的绝对路径；越界返回 None。

    只接受相对路径（拒绝绝对路径与 ".."）；realpath + commonpath 双重包含检查，
    防符号链接逃逸（folder_paths.is_within_directory 同款语义，独立实现以兼容
    旧运行时）。
    """
    rel = str(src_path or "").replace("\\", "/").strip()
    if not rel or rel.startswith("/"):
        return None
    parts = [p for p in rel.split("/") if p]
    if not parts or any(p == ".." for p in parts):
        return None
    try:
        input_dir = os.path.realpath(folder_paths.get_input_directory())
        path = os.path.realpath(os.path.join(input_dir, *parts))
        if os.path.commonpath((input_dir, path)) != input_dir:
            return None
    except ValueError:
        return None
    return path


def materialize_pause_snapshot(unique_id, src_path, flip=False):
    """读 input/ 源图 →（flip 时水平镜像）→ 写入该节点的 temp 快照。

    返回 frame dict（Continue/预览链路直接消费）。源图不存在或路径越界抛
    FileNotFoundError。文本块经 _build_pnginfo 保留（尊重 --disable-metadata）。
    """
    path = _resolve_input_path(src_path)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("source not found")
    with Image.open(path) as snap:
        info = dict(snap.info)
        img = ImageOps.mirror(snap) if flip else snap.copy()
    dest = _snapshot_path(unique_id)
    pnginfo = _build_pnginfo(
        prompt=info.get("prompt"),
        workflow=info.get("workflow"),
        parameters=info.get("parameters"),
    )
    img.save(dest, "PNG", pnginfo=pnginfo)
    return {"filename": os.path.basename(dest), "subfolder": "", "type": "temp"}


def _register_routes():
    try:
        from server import PromptServer
        from aiohttp import web

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            print("[sfnodes] PromptServer instance not available, preview routes not registered")
            return
        routes = ins.routes

        @routes.post("/api/sfnodes/preview/save")
        async def api_preview_save(request):
            """把 base64 PNG 存到 ComfyUI output/ 目录并嵌入工作流元数据。

            Request JSON: { image_b64, filename_prefix, workflow, prompt }
            Response: { status, filename, subfolder } 或 { error }
            """
            try:
                data = await request.json()
            except Exception:
                return web.json_response({"error": "invalid JSON"}, status=400)
            if not isinstance(data, dict):
                data = {}
            image_b64 = data.get("image_b64", "")
            prefix = _safe_prefix(data.get("filename_prefix", "PauseImage")) or "PauseImage"
            workflow = data.get("workflow")
            prompt = data.get("prompt")

            pil = _decode_image(image_b64)
            if pil is None:
                return web.json_response({"error": "invalid image data"}, status=400)
            try:
                output_dir = folder_paths.get_output_directory()
                full_folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
                    prefix, output_dir, pil.width, pil.height
                )
                os.makedirs(full_folder, exist_ok=True)
                fname = f"{name}_{counter:05}_.png"
                full_path = os.path.join(full_folder, fname)
                # parameters 块（Civitai/A1111）穿过重编码——浏览器 POST 的是本
                # 插件自己写的 PNG 原始字节，该块（如有）是我们自己的输出随行；
                # 从零重建 PngInfo 会静默丢弃它。None 在 _build_pnginfo 里是
                # no-op，没有该块的文件字节一致。
                pnginfo = _build_pnginfo(prompt=prompt, workflow=workflow,
                                         parameters=pil.info.get("parameters"))
                pil.save(full_path, "PNG", pnginfo=pnginfo)
            except Exception as e:
                return web.json_response({"error": f"save failed: {e}"}, status=500)
            return web.json_response(
                {"status": "success", "filename": fname, "subfolder": subfolder}
            )

        @routes.post("/api/sfnodes/preview/prepare")
        async def api_preview_prepare(request):
            """把工作流元数据嵌入 PNG 并返回 data URI + 自增建议文件名。

            Request JSON: { image_b64, filename_prefix, workflow, prompt }
            Response: { image_b64, suggested_filename } 或 { error }
            """
            try:
                data = await request.json()
            except Exception:
                return web.json_response({"error": "invalid JSON"}, status=400)
            if not isinstance(data, dict):
                data = {}
            image_b64 = data.get("image_b64", "")
            prefix = _safe_prefix(data.get("filename_prefix", "PauseImage")) or "PauseImage"
            workflow = data.get("workflow")
            prompt = data.get("prompt")

            pil = _decode_image(image_b64)
            if pil is None:
                return web.json_response({"error": "invalid image data"}, status=400)
            try:
                pnginfo = _build_pnginfo(prompt=prompt, workflow=workflow,
                                         parameters=pil.info.get("parameters"))
                buf = io.BytesIO()
                pil.save(buf, "PNG", pnginfo=pnginfo)
                out_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                # 建议文件名取下一个空闲计数（不实际写入）
                output_dir = folder_paths.get_output_directory()
                _, name, counter, _, _ = folder_paths.get_save_image_path(
                    prefix, output_dir, pil.width, pil.height
                )
                suggested = f"{name}_{counter:05}_.png"
            except Exception as e:
                return web.json_response({"error": f"prepare failed: {e}"}, status=500)
            return web.json_response({"image_b64": out_b64, "suggested_filename": suggested})

        @routes.post("/api/sfnodes/preview/flip")
        async def api_preview_flip(request):
            """把 temp 目录里的闸门快照 PNG 原地水平镜像（Flip 开关用）。

            Request JSON: { filename }
            Response: { status } 或 { error }

            只接受 temp 目录下的 basename 且限定 sf_pause 前缀（图片/遮罩闸门
            快照命名），避免该路由被用来改写任意文件；原图已被打开的句柄在 with
            退出后释放，再覆盖保存（Windows 文件锁）。
            """
            try:
                data = await request.json()
            except Exception:
                return web.json_response({"error": "invalid JSON"}, status=400)
            if not isinstance(data, dict):
                data = {}
            raw = str(data.get("filename", ""))
            filename = os.path.basename(raw)
            if not filename or filename != raw or not filename.startswith("sf_pause"):
                return web.json_response({"error": "invalid filename"}, status=400)
            try:
                temp_dir = folder_paths.get_temp_directory()
                path = os.path.join(temp_dir, filename)
                if not os.path.isfile(path):
                    return web.json_response({"error": "snapshot not found"}, status=404)
                _mirror_png(path)
            except Exception as e:
                return web.json_response({"error": f"flip failed: {e}"}, status=500)
            return web.json_response({"status": "success"})

        @routes.post("/api/sfnodes/pause/load")
        async def api_pause_load(request):
            """把 input/ 里的外载图物化成本节点的 temp 快照（Load/Browse/拖放/粘贴）。

            Request JSON: { unique_id, src_path, flip }
            Response: { status, frame:{filename,subfolder,type} } 或 { error }

            Pause/Pass 仍优先接线图；本路由产物供 Continue 提交与预览显示。
            """
            try:
                data = await request.json()
            except Exception:
                return web.json_response({"error": "invalid JSON"}, status=400)
            if not isinstance(data, dict):
                data = {}
            unique_id = data.get("unique_id")
            if unique_id is None:
                return web.json_response({"error": "missing unique_id"}, status=400)
            try:
                frame = materialize_pause_snapshot(
                    unique_id, data.get("src_path", ""), bool(data.get("flip"))
                )
            except FileNotFoundError:
                return web.json_response({"error": "source not found"}, status=404)
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)
            except Exception as e:
                return web.json_response({"error": f"load failed: {e}"}, status=500)
            return web.json_response({"status": "success", "frame": frame})

        print("[sfnodes] preview routes registered (/api/sfnodes/preview/save, /prepare, /flip, /pause/load)")
    except Exception as e:
        print(f"[sfnodes] preview routes registration failed: {e}")


_register_routes()
