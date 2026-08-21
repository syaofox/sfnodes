import asyncio
import hashlib
import os

import folder_paths
from aiohttp import web
from PIL import Image, ImageOps

from .disk_state import sanitize_filename
from .logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 约定：LoRA 示例图存放在该 LoRA 所在目录的 sample/ 子目录下，
# 例：lora "krea2/InnieVagina/xxx.safetensors" →
#     models/loras/krea2/InnieVagina/sample/*.png
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif"}
_VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".webm", ".mkv"}
_MEDIA_EXTS = _IMAGE_EXTS | _VIDEO_EXTS
_LORA_EXTS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft"}
_THUMB_SIZE = 256


def _get_loras_roots() -> list[str]:
    return folder_paths.get_folder_paths("loras")


def _is_under_root(abs_path: str, root: str) -> bool:
    # 复用 lora_routes 的健壮检查（realpath + 跨盘 lexical 回退），但此处为避免
    # 循环导入（lora_routes 依赖 folder_paths），就地实现同款逻辑的轻量版。
    # 仅模块内使用；对外仍通过 _is_path_under 语义保证 symlink 逃逸被挡。
    try:
        from .lora_routes import _is_path_under as _check
        return _check(abs_path, root)
    except Exception:
        pass
    # 回退：兼容尚未加载时的最简检查（不抛错）
    try:
        return os.path.commonpath([os.path.realpath(abs_path), os.path.realpath(root)]) == os.path.realpath(root)
    except Exception:
        try:
            return abs_path.startswith(os.path.normpath(root) + os.sep)
        except Exception:
            return False


def _resolve_lora_dir(lora_name: str) -> str | None:
    """lora 相对路径 → 其所在目录的绝对路径；找不到/非法返回 None"""
    if not lora_name or not isinstance(lora_name, str):
        return None
    if os.path.splitext(lora_name)[1].lower() not in _LORA_EXTS:
        return None
    full = folder_paths.get_full_path("loras", lora_name)
    if not full or not os.path.isfile(full):
        return None
    return os.path.dirname(full)


def _rel_to_root(abs_path: str) -> str | None:
    """绝对路径 → 相对于 loras 根的路径；不在根内返回 None"""
    for root in _get_loras_roots():
        if _is_under_root(abs_path, root):
            return os.path.relpath(abs_path, root)
    return None


def _rel_to_sample_dir(lora_name: str) -> str | None:
    """lora 相对路径 → sample 目录相对 loras 根的路径（如 krea2/x/sample）"""
    lora_dir = _resolve_lora_dir(lora_name)
    if lora_dir is None:
        return None
    rel = _rel_to_root(os.path.join(lora_dir, "sample"))
    return rel


def _list_sample_images(lora_name: str) -> tuple[list[str], str | None]:
    """返回 (相对 loras 根的图片路径列表, sample 目录相对路径)"""
    lora_dir = _resolve_lora_dir(lora_name)
    if lora_dir is None:
        return [], None
    sample_dir = os.path.join(lora_dir, "sample")
    images = []
    if os.path.isdir(sample_dir):
        for f in sorted(os.listdir(sample_dir)):
            if f.startswith(".") or os.path.splitext(f)[1].lower() not in _MEDIA_EXTS:
                continue
            full = os.path.join(sample_dir, f)
            if not os.path.isfile(full):
                continue
            rel = _rel_to_root(full)
            if rel:
                images.append(rel)
    return images, _rel_to_root(sample_dir)


def _resolve_sample_image(abs_path: str) -> str | None:
    """解析 sample 图片的绝对路径（须在 loras 根内且位于某个 sample/ 目录下）"""
    rel = abs_path
    if not rel or not isinstance(rel, str) or os.path.isabs(rel):
        return None
    # 任意 ".." 段即拒（词法层面先于 realpath，防 lev+realpath 绕过）
    # sanitize 语义：按 "/" 切分后检查每段
    if any(p == ".." for p in rel.replace("\\", "/").split("/")):
        return None
    # 拒绝绝对盘符 / UNC（safe_join 同款）
    try:
        if os.path.splitdrive(rel)[0]:
            return None
    except Exception:
        return None
    parts = rel.replace("\\", "/").split("/")
    if "sample" not in parts[:-1]:
        return None
    for root in _get_loras_roots():
        candidate = os.path.normpath(os.path.join(root, rel))
        # realpath 守卫：symlink 逃逸在此被 _is_under_root (realpath) 挡住
        try:
            real_candidate = os.path.realpath(candidate)
            real_root = os.path.realpath(root)
        except Exception:
            continue
        if not _is_under_root(real_candidate, real_root):
            continue
        if not os.path.isfile(real_candidate):
            continue
        # 额外确认：sample 目录确在路径中（realpath 后仍含 sample 段）
        if "sample" not in real_candidate.replace("\\", "/").split(os.sep):
            # realpath 解析后 sample 可能被 symlink 隐藏，仍要求原始 rel 有 sample
            pass
        return real_candidate
    return None


def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, routes not registered")
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/lora_samples")
        async def _list_samples(request: web.Request) -> web.Response:
            try:
                lora_name = request.rel_url.query.get("filename", "")
                images, sample_dir = _list_sample_images(lora_name)
                if sample_dir is None:
                    return web.json_response({"error": "lora not found"}, status=404)
                return web.json_response({"images": images, "sample_dir": sample_dir})
            except Exception as e:
                logger.error(f"GET /api/sfnodes/lora_samples failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        @routes.get("/api/sfnodes/lora_samples/image")
        async def _sample_image(request: web.Request) -> web.Response:
            try:
                path = request.rel_url.query.get("path", "")
                full = _resolve_sample_image(path)
                if full is None:
                    return web.Response(status=404)

                w = request.rel_url.query.get("w")
                if w:
                    # 视频缩略：抽首帧为图片（与 Civitai 缩略同源 video_thumb），失败回退原视频
                    if os.path.splitext(full)[1].lower() in _VIDEO_EXTS:
                        try:
                            size = min(max(int(w), 16), 1024)
                        except Exception:
                            size = _THUMB_SIZE
                        cache_dir = os.path.join(
                            folder_paths.get_temp_directory(), ".sf_lora_samples"
                        )
                        os.makedirs(cache_dir, exist_ok=True)
                        key = hashlib.md5(
                            f"{path}:{os.path.getmtime(full)}:{size}:vthumb".encode()
                        ).hexdigest()
                        cache_path = os.path.join(cache_dir, f"{key}.webp")
                        if not os.path.isfile(cache_path):
                            loop = asyncio.get_running_loop()

                            def _make_vthumb():
                                try:
                                    from .video_thumb import extract_first_frame_from_path
                                    jpeg = extract_first_frame_from_path(full)
                                    if not jpeg:
                                        return False
                                    # jpeg -> PIL -> webp thumb
                                    import io
                                    img = Image.open(io.BytesIO(jpeg))
                                    img = ImageOps.exif_transpose(img)
                                    img.thumbnail((size, size), Image.LANCZOS)
                                    if img.mode not in ("RGB", "L"):
                                        img = img.convert("RGBA")
                                    else:
                                        img = img.convert("RGB")
                                    img.save(cache_path, format="WEBP", quality=85)
                                    return True
                                except Exception as e:
                                    logger.warning(f"video thumb failed for {path}: {e}")
                                    return False

                            ok = await loop.run_in_executor(None, _make_vthumb)
                            if not ok:
                                return web.FileResponse(full)
                        return web.FileResponse(cache_path)
                    try:
                        size = min(max(int(w), 16), 1024)
                    except Exception:
                        size = _THUMB_SIZE
                    cache_dir = os.path.join(
                        folder_paths.get_temp_directory(), ".sf_lora_samples"
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    key = hashlib.md5(
                        f"{path}:{os.path.getmtime(full)}:{size}".encode()
                    ).hexdigest()
                    cache_path = os.path.join(cache_dir, f"{key}.webp")
                    if not os.path.isfile(cache_path):
                        try:
                            img = Image.open(full)
                        except Exception:
                            return web.FileResponse(full)
                        img = ImageOps.exif_transpose(img)
                        img.thumbnail((size, size), Image.LANCZOS)
                        if img.mode not in ("RGB", "L"):
                            img = img.convert("RGBA")
                        else:
                            img = img.convert("RGB")
                        img.save(cache_path, format="WEBP", quality=85)
                    return web.FileResponse(cache_path)

                return web.FileResponse(full)
            except Exception as e:
                logger.error(f"GET /api/sfnodes/lora_samples/image failed: {e}")
                return web.Response(status=500)

        @routes.get("/api/sfnodes/lora_samples/prompt")
        async def _sample_prompt(request: web.Request) -> web.Response:
            """读取 sample 目录下图片/视频的 prompt（复用 sf_utils.prompt_reader）。"""
            try:
                path = request.rel_url.query.get("path", "")
                full = _resolve_sample_image(path)
                if full is None:
                    return web.json_response({"found": False, "message": "Sample not found."})
                try:
                    from .prompt_reader import read_prompt_from_image
                except Exception as e:
                    logger.error(f"prompt_reader import failed: {e}")
                    return web.json_response({"found": False, "message": "Prompt reader unavailable."})
                loop = asyncio.get_running_loop()
                try:
                    result = await loop.run_in_executor(None, read_prompt_from_image, full)
                except Exception as e:
                    logger.error(f"GET /api/sfnodes/lora_samples/prompt failed: {e}")
                    return web.json_response({"found": False, "message": f"Could not read prompt: {e}"})
                return web.json_response(result)
            except Exception as e:
                logger.error(f"GET /api/sfnodes/lora_samples/prompt failed: {e}")
                return web.json_response({"found": False, "message": "internal error"})

        @routes.post("/api/sfnodes/lora_samples/upload")
        async def _upload_sample(request: web.Request) -> web.Response:
            try:
                try:
                    post = await request.post()
                except Exception:
                    return web.Response(status=400)
                lora_name = post.get("filename", "")
                lora_dir = _resolve_lora_dir(lora_name)
                if lora_dir is None:
                    return web.json_response({"error": "lora not found"}, status=404)

                image = post.get("image")
                if image is None or not image.file:
                    return web.Response(status=400)
                ext = os.path.splitext(image.filename or "")[1].lower()
                if ext not in _MEDIA_EXTS:
                    return web.json_response({"error": "unsupported image type"}, status=400)

                sample_dir = os.path.join(lora_dir, "sample")
                os.makedirs(sample_dir, exist_ok=True)

                # 净化文件名：防保留名 / 非法字符 / 路径穿越
                raw_name = os.path.basename(image.filename or "upload.png")
                # 先按扩展名校验，再整体净化（保留扩展名）
                ext_check = os.path.splitext(raw_name)[1].lower()
                if ext_check not in _MEDIA_EXTS:
                    return web.json_response({"error": "unsupported image type"}, status=400)
                safe = sanitize_filename(raw_name, fallback="sample.png")
                # sanitize 可能改变扩展名，二次校验
                if os.path.splitext(safe)[1].lower() not in _MEDIA_EXTS:
                    safe = os.path.splitext(safe)[0] + ext_check
                filename = safe
                split = os.path.splitext(filename)
                filepath = os.path.join(sample_dir, filename)
                i = 1
                while os.path.exists(filepath):
                    filename = f"{split[0]} ({i}){split[1]}"
                    filepath = os.path.join(sample_dir, filename)
                    i += 1

                # 限流：单文件 50MB（与预览视频上限对齐），防 DoS
                data = image.file.read()
                if len(data) > 50 * 1024 * 1024:
                    return web.json_response({"error": "file too large (50MB max)"}, status=413)
                with open(filepath, "wb") as f:
                    f.write(data)
                logger.info(f"Saved lora sample: {filepath}")

                rel = _rel_to_root(filepath)
                if rel is None:
                    return web.Response(status=400)
                return web.json_response({"name": filename, "path": rel})
            except Exception as e:
                logger.error(f"POST /api/sfnodes/lora_samples/upload failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        @routes.delete("/api/sfnodes/lora_samples")
        async def _delete_sample(request: web.Request) -> web.Response:
            try:
                path = request.rel_url.query.get("path", "")
                full = _resolve_sample_image(path)
                if full is None:
                    return web.json_response({"error": "not found"}, status=404)
                os.remove(full)
                logger.info(f"Deleted lora sample: {full}")
                return web.json_response({"deleted": path})
            except Exception as e:
                logger.error(f"DELETE /api/sfnodes/lora_samples failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        logger.info("LoRA samples API routes registered")

    except Exception as e:
        logger.error(f"Failed to register LoRA samples routes: {e}")


_register_routes()
