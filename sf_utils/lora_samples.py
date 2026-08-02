import hashlib
import os

import folder_paths
from aiohttp import web
from PIL import Image, ImageOps

from .logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 约定：LoRA 示例图存放在该 LoRA 所在目录的 sample/ 子目录下，
# 例：lora "krea2/InnieVagina/xxx.safetensors" →
#     models/loras/krea2/InnieVagina/sample/*.png
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif"}
_LORA_EXTS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft"}
_THUMB_SIZE = 256


def _get_loras_roots() -> list[str]:
    return folder_paths.get_folder_paths("loras")


def _is_under_root(abs_path: str, root: str) -> bool:
    return abs_path.startswith(os.path.normpath(root) + os.sep)


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
            if f.startswith(".") or os.path.splitext(f)[1].lower() not in _IMAGE_EXTS:
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
    if not rel or not isinstance(rel, str) or os.path.isabs(rel) or ".." in rel:
        return None
    parts = rel.replace("\\", "/").split("/")
    if "sample" not in parts[:-1]:
        return None
    for root in _get_loras_roots():
        candidate = os.path.normpath(os.path.join(root, rel))
        if _is_under_root(candidate, root) and os.path.isfile(candidate):
            return candidate
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
                        img = Image.open(full)
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
                if ext not in _IMAGE_EXTS:
                    return web.json_response({"error": "unsupported image type"}, status=400)

                sample_dir = os.path.join(lora_dir, "sample")
                os.makedirs(sample_dir, exist_ok=True)

                filename = os.path.basename(image.filename)
                split = os.path.splitext(filename)
                filepath = os.path.join(sample_dir, filename)
                i = 1
                while os.path.exists(filepath):
                    filename = f"{split[0]} ({i}){split[1]}"
                    filepath = os.path.join(sample_dir, filename)
                    i += 1

                with open(filepath, "wb") as f:
                    f.write(image.file.read())
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
