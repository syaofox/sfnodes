import hashlib
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.utils
import folder_paths
from aiohttp import web

_CATEGORY = "sfnodes/image"

_DEFAULT_FOLDER = "default"


def _get_images_base_dir() -> str:
    base = os.path.join(folder_paths.get_user_directory(), "sfnodes", "images")
    os.makedirs(base, exist_ok=True)
    return base


def _list_one_level_subdirs(root: str) -> list:
    try:
        return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    except OSError:
        return []


def _list_folders() -> list:
    folders = [_DEFAULT_FOLDER]
    images_base = _get_images_base_dir()
    folders += ["images/" + d for d in _list_one_level_subdirs(images_base) if d != _DEFAULT_FOLDER]
    for prefix, root in (
        ("input", folder_paths.get_input_directory()),
        ("output", folder_paths.get_output_directory()),
    ):
        if not os.path.isdir(root):
            continue
        folders.append(prefix)
        folders += [prefix + "/" + d for d in _list_one_level_subdirs(root)]
    return folders


def _resolve_under(root: str, rel: str) -> str:
    root = os.path.normpath(root)
    rel = (rel or "").strip()
    if not rel or os.path.isabs(rel):
        rel = _DEFAULT_FOLDER
    else:
        rel = rel.lstrip("/\\")
    target = os.path.normpath(os.path.join(root, rel))
    if target != root and not target.startswith(root + os.sep):
        return os.path.join(_get_images_base_dir(), _DEFAULT_FOLDER)
    return target


def _resolve_folder(folder: str) -> str:
    name = (folder or _DEFAULT_FOLDER).strip()
    if not name or name == _DEFAULT_FOLDER:
        return os.path.join(_get_images_base_dir(), _DEFAULT_FOLDER)

    if name == "input":
        return os.path.normpath(folder_paths.get_input_directory())
    if name == "output":
        return os.path.normpath(folder_paths.get_output_directory())
    if name.startswith("images/"):
        return _resolve_under(_get_images_base_dir(), name[len("images/"):])
    if name.startswith("input/"):
        return _resolve_under(folder_paths.get_input_directory(), name[len("input/"):])
    if name.startswith("output/"):
        return _resolve_under(folder_paths.get_output_directory(), name[len("output/"):])
    return _resolve_under(_get_images_base_dir(), name)


def _sort_key(filename):
    match = re.search(r'\d+', filename)
    if match:
        return (0, int(match.group()))
    return (1, filename)


def _sorted_image_files(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1) -> list:
    dir_files = [os.path.join(directory, x) for x in sorted(os.listdir(directory), key=_sort_key)]
    dir_files = [f for f in dir_files if os.path.isfile(f)]
    dir_files = folder_paths.filter_files_content_types(dir_files, ["image"])
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]
    return dir_files


class SFLoadImagesPath:
    @classmethod
    def INPUT_TYPES(cls):
        folders = _list_folders()
        return {
            "required": {
                "folder": (folders, {"tooltip": "选择图片目录：input / output 目录及其子目录，或 user/sfnodes/images/ 下子目录，批量加载其中全部图片"}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "限制加载的图片数量（0 = 无限制）"}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "跳过前 N 张图片"}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "tooltip": "每隔 N 张选取 1 张"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count")
    FUNCTION = "load_images"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从 input / output 目录或其子目录、user/sfnodes/images/ 下子目录批量加载图片，统一尺寸后输出图片批次与遮罩，用作反推等批量图片输入"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, folder, image_load_cap=0, skip_first_images=0, select_every_nth=1):
        directory = _resolve_folder(folder)
        if not os.path.isdir(directory):
            return False
        dir_files = _sorted_image_files(directory, image_load_cap, skip_first_images, select_every_nth)
        h = hashlib.sha256()
        for filepath in dir_files:
            h.update(filepath.encode())
            h.update(str(os.path.getmtime(filepath)).encode())
        return h.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, folder, **kwargs):
        if not os.path.isdir(_resolve_folder(folder)):
            return f"Directory '{_resolve_folder(folder)}' cannot be found."
        return True

    def load_images(self, folder, image_load_cap=0, skip_first_images=0, select_every_nth=1):
        directory = _resolve_folder(folder)
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory '{directory}' does not exist. "
                "Please create the folder and add image files, "
                "then click the refresh button on the node to update the folder list."
            )

        dir_files = _sorted_image_files(directory, image_load_cap, skip_first_images, select_every_nth)
        if len(dir_files) == 0:
            entries = sorted(os.listdir(directory))[:5]
            hint = ", ".join(repr(e) for e in entries) if entries else "directory is empty"
            raise FileNotFoundError(
                f"No images could be loaded from directory '{directory}' "
                f"(contents: {hint}). Supported formats include png/jpg/jpeg/webp/gif/bmp/tiff."
            )

        sizes = {}
        has_alpha = False
        for image_path in dir_files:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            has_alpha |= 'A' in i.getbands()
            count = sizes.get(i.size, 0)
            sizes[i.size] = count + 1
        width, height = max(sizes.items(), key=lambda x: x[1])[0]

        iformat = "RGBA" if has_alpha else "RGB"
        pbar = comfy.utils.ProgressBar(len(dir_files))
        images = []
        for idx, image_path in enumerate(dir_files):
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            img = img.convert(iformat)
            arr = np.array(img, dtype=np.float32)
            t = torch.from_numpy(arr).div_(255)
            if t.shape[0] != height or t.shape[1] != width:
                t = t.movedim(-1, 0).unsqueeze(0)
                t = comfy.utils.common_upscale(t, width, height, "lanczos", "center")
                t = t.squeeze(0).movedim(0, -1)
            if has_alpha:
                t[:, :, -1] = 1 - t[:, :, -1]
            images.append(t)
            pbar.update_absolute(idx + 1, len(dir_files))

        batch = torch.stack(images, dim=0)
        if has_alpha:
            masks = batch[:, :, :, 3]
            images_out = batch[:, :, :, :3]
        else:
            images_out = batch
            masks = torch.zeros((batch.size(0), 64, 64), dtype=torch.float32, device="cpu")

        return (images_out, masks, len(dir_files))


def _register_routes():
    try:
        from server import PromptServer
        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/images_path/folders")
        async def _list_folders_route(request: web.Request) -> web.Response:
            try:
                return web.json_response({"folders": _list_folders()})
            except Exception:
                return web.Response(status=500)
    except Exception:
        pass


_register_routes()
