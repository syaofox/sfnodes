"""SFPauseImage 保存路由（/api/sfnodes/preview/save + /prepare）。

复刻 Pixaroma 的 /pixaroma/api/preview/save 与 /prepare（拖回重建元数据用）：
- save：把 base64 PNG 存到 ComfyUI output/ 目录，嵌入 workflow/prompt PNG 块
  （保存的图片可拖回 ComfyUI 重建同一张图）
- prepare：嵌入元数据后返回 data URI + 建议文件名（Save to Disk 用）

注册方式沿用 sf_utils/lora_notes.py 的 _register_routes 先例：模块导入时
（__init__.py import）副作用注册，try/except 包裹，环境异常时降级不注册。
"""

import base64
import io
import json
import os
import re

import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo


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


def _json_safe(obj):
    """清洗 NaN/Inf 使嵌入 JSON 合法（拖回 ComfyUI 时 JSON.parse 不炸）。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
    return obj


_DISALLOWED_CHAR_RE = re.compile(r'[<>:"|?*\x00-\x1f\x7f]')
_MULTI_UNDERSCORE_RE = re.compile(r"_+")
_PREFIX_MAX_LEN = 256       # 输入上限（尽早拒绝明显的垃圾）
_PREFIX_OUTPUT_MAX = 100    # 输出上限
_WIN_RESERVED_NAMES = frozenset((
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
))


def _sanitize_segment(seg):
    """把 Windows 非法字符换成 '_'，整理边沿，守卫保留设备名。

    调用方须在调用前拒绝 '..'。全部不可用时返回 ""。尾点/尾空格剥离——
    Windows 创建时本就静默剥离，这里剥离让报告路径与磁盘实际落盘一致。
    """
    cleaned = _DISALLOWED_CHAR_RE.sub("_", seg)
    cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned)
    # 循环到稳定：边沿空白、边沿下划线、尾点/空格会互相遮蔽
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = cleaned.strip().strip("_").rstrip(". ")
    if cleaned and cleaned.split(".", 1)[0].upper() in _WIN_RESERVED_NAMES:
        cleaned += "_"
    return cleaned


def _safe_prefix(raw):
    """清洗文件名前缀；不可恢复时返回 ""（调用方兜底默认名）。

    管道：逐段只替换 Windows 非法字符（<>:"|?* 与控制字符）为 '_'、折叠重复
    '_'、剥离边沿空白/下划线/尾点、Windows 保留设备名加 '_' 后缀。其余
    （非拉丁文字、重音、空格）原样通过，与原生 SaveImage 一致。段以 '/' 分隔。
    先检查 leading '/' 与 '..' 段（在任何清洗之前——清洗会把 '..' 吃掉，
    让路径穿越检查失效）。
    （省略了原版 %date:FMT% token 展开：路由场景前端从不传带 token 的前缀。）
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().replace("\\", "/")
    if not s or len(s) > _PREFIX_MAX_LEN:
        return ""
    if s.startswith("/"):
        return ""
    parts = s.split("/")
    if any(p == ".." for p in parts):
        return ""
    cleaned_parts = [_sanitize_segment(p) for p in parts if p]
    cleaned_parts = [p for p in cleaned_parts if p]
    if not cleaned_parts:
        return ""
    result = "/".join(cleaned_parts)
    if len(result) > _PREFIX_OUTPUT_MAX:
        result = result[:_PREFIX_OUTPUT_MAX].rstrip("/_-")
    return result


def _decode_image(image_b64):
    """data URI base64 PNG -> PIL.Image，失败返回 None。"""
    if not isinstance(image_b64, str) or not image_b64:
        return None
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


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

        print("[sfnodes] preview routes registered (/api/sfnodes/preview/save, /prepare)")
    except Exception as e:
        print(f"[sfnodes] preview routes registration failed: {e}")


_register_routes()
