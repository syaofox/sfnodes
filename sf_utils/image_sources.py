"""图片目录源与解码共享纯逻辑：目录解析（input/output 前缀 + 绝对路径）、
自然序图片列表、EXIF 旋转与 alpha 通道提取。

原为 nodes/image/load_images_path.py 的私有实现（目录解析/排序/解码三块），
SFLoadImagesCursor 需要同一套语义，收敛为单一实现（见
experience/nodes-image.md §147）。无状态、无 torch 依赖。
"""
import os
import re

from PIL import Image, ImageOps

import folder_paths

DEFAULT_FOLDER = "default"


def _get_input_base_dir() -> str:
    return os.path.normpath(folder_paths.get_input_directory())


def list_one_level_subdirs(root: str) -> list:
    try:
        return sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
        )
    except OSError:
        return []


def list_folders() -> list:
    folders = [DEFAULT_FOLDER]
    for prefix, root in (
        ("input", folder_paths.get_input_directory()),
        ("output", folder_paths.get_output_directory()),
    ):
        if not os.path.isdir(root):
            continue
        folders.append(prefix)
        folders += [prefix + "/" + d for d in list_one_level_subdirs(root)]
    return folders


def _resolve_under(root: str, rel: str) -> str:
    root = os.path.normpath(root)
    rel = (rel or "").strip()
    if not rel or os.path.isabs(rel):
        rel = DEFAULT_FOLDER
    else:
        rel = rel.lstrip("/\\")
    target = os.path.normpath(os.path.join(root, rel))
    if target != root and not target.startswith(root + os.sep):
        return _get_input_base_dir()
    return target


def resolve_folder(folder: str) -> str:
    name = (folder or DEFAULT_FOLDER).strip()
    if not name or name == DEFAULT_FOLDER:
        return _get_input_base_dir()

    # 直接输入路径模式：绝对路径原样使用（用户主动输入的任意目录）。
    if os.path.isabs(name):
        return os.path.normpath(name)

    if name == "input":
        return os.path.normpath(folder_paths.get_input_directory())
    if name == "output":
        return os.path.normpath(folder_paths.get_output_directory())
    if name.startswith("input/"):
        return _resolve_under(folder_paths.get_input_directory(), name[len("input/"):])
    if name.startswith("output/"):
        return _resolve_under(folder_paths.get_output_directory(), name[len("output/"):])
    return _resolve_under(_get_input_base_dir(), name)


def sort_key(filename):
    match = re.search(r'\d+', filename)
    if match:
        # 末位文件名兜底：纯数字相同（a1.png vs b1.png）时按文件名定序，
        # 不依赖 os.listdir 的返回顺序（不同文件系统顺序不同）。
        return (0, int(match.group()), filename)
    return (1, filename)


def sorted_image_files(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1) -> list:
    dir_files = [os.path.join(directory, x) for x in sorted(os.listdir(directory), key=sort_key)]
    dir_files = [f for f in dir_files if os.path.isfile(f)]
    dir_files = folder_paths.filter_files_content_types(dir_files, ["image"])
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]
    return dir_files


def open_image(image_path: str):
    """打开图片并应用 EXIF 旋转；不做模式转换（调用方决定 RGB/RGBA）。"""
    return ImageOps.exif_transpose(Image.open(image_path))


def image_alpha(img):
    """取 alpha 通道 L 图（0..255）；无 A 通道且无调色板 transparency 时返回 None。"""
    if "A" in img.getbands():
        return img.getchannel("A")
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA").getchannel("A")
    return None
