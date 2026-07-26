import os
import hashlib
import folder_paths
from aiohttp import web
from nodes import LoadImage
from PIL import Image, ImageOps

THUMB_SIZE = 256

_CATEGORY = "sfnodes/image"


def _get_base_dir(source_type="input"):
    if source_type == "output":
        return folder_paths.get_output_directory()
    return folder_paths.get_input_directory()


def _list_images_recursive(source_type="input"):
    base_dir = _get_base_dir(source_type)
    files = []
    for root, _, filenames in os.walk(base_dir):
        for f in filenames:
            full_path = os.path.join(root, f)
            if os.path.isfile(full_path):
                rel_path = os.path.relpath(full_path, base_dir)
                files.append(rel_path)
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


class SFLoadImageBrowser(LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_images_recursive()
        return {
            "required": {
                "image": (files, {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "load_image"
    CATEGORY = _CATEGORY
    DESCRIPTION = "浏览 input/output 目录（含子文件夹）中的图片并加载"

    def load_image(self, image):
        image_output, mask = super().load_image(image)
        return (image_output, mask, image)


def _register_routes():
    try:
        from server import PromptServer
        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/images/list")
        async def _list_images(request: web.Request) -> web.Response:
            try:
                source_type = request.rel_url.query.get("type", "input")
                base_dir = _get_base_dir(source_type)
                files = _list_images_recursive(source_type)
                result = []
                for rel_path in files:
                    full_path = os.path.join(base_dir, rel_path)
                    stat = os.stat(full_path)
                    result.append({
                        "filename": os.path.basename(rel_path),
                        "subfolder": os.path.dirname(rel_path),
                        "path": rel_path,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    })
                return web.json_response(result)
            except Exception:
                return web.Response(status=500)

        @routes.get("/api/sfnodes/images/thumb")
        async def _image_thumb(request: web.Request) -> web.Response:
            try:
                image_path = request.rel_url.query.get("path", "")
                source_type = request.rel_url.query.get("type", "input")
                if not image_path or ".." in image_path:
                    return web.Response(status=400)

                base_dir = _get_base_dir(source_type)
                full_path = os.path.normpath(os.path.join(base_dir, image_path))
                if not full_path.startswith(os.path.normpath(base_dir)):
                    return web.Response(status=403)
                if not os.path.isfile(full_path):
                    return web.Response(status=404)

                cache_dir = os.path.join(folder_paths.get_temp_directory(), ".sf_thumb_cache")
                os.makedirs(cache_dir, exist_ok=True)

                cache_key = hashlib.md5(
                    f"{source_type}:{image_path}:{os.path.getmtime(full_path)}".encode()
                ).hexdigest()
                cache_path = os.path.join(cache_dir, f"{cache_key}.webp")

                if os.path.isfile(cache_path):
                    return web.FileResponse(cache_path)

                img = Image.open(full_path)
                img = ImageOps.exif_transpose(img)
                img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                img.save(cache_path, "WEBP", quality=85)

                return web.FileResponse(cache_path)
            except Exception:
                return web.Response(status=500)

        @routes.delete("/api/sfnodes/images/delete")
        async def _delete_image(request: web.Request) -> web.Response:
            try:
                image_path = request.rel_url.query.get("path", "")
                source_type = request.rel_url.query.get("type", "input")
                if source_type != "input":
                    return web.Response(status=403, text="只能删除 input 目录中的文件")
                if not image_path or ".." in image_path:
                    return web.Response(status=400)

                base_dir = _get_base_dir(source_type)
                full_path = os.path.normpath(os.path.join(base_dir, image_path))
                if not full_path.startswith(os.path.normpath(base_dir)):
                    return web.Response(status=403)
                if not os.path.isfile(full_path):
                    return web.Response(status=404)

                os.remove(full_path)
                return web.Response(status=200)
            except Exception:
                return web.Response(status=500)
    except Exception:
        pass


_register_routes()
