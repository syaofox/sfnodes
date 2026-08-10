import os
import re

import torch
from PIL import Image, ImageOps

import comfy.utils

from ...sf_utils.image_convert import pil2tensor

_CATEGORY = "sfnodes/image"


def _load_images_from_folder(folder_path, start_index=0, max_index=None):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹未找到: {folder_path}")

    if not os.listdir(folder_path):
        raise ValueError(f"文件夹为空: {folder_path}")

    file_list = sorted(
        os.listdir(folder_path),
        key=lambda s: sum(
            ((s, int(n)) for s, n in re.findall(r"(\D+)(\d+)", "a%s0" % s)), ()
        ),
    )

    if max_index is not None:
        start_index = max(0, min(start_index, len(file_list) - 1))
        end_index = min(start_index + max_index, len(file_list))
    else:
        start_index = 0
        end_index = len(file_list)

    image_list = []
    ref_image = None

    for num in range(start_index, end_index):
        fname = os.path.join(folder_path, file_list[num])
        img = Image.open(fname)
        img = ImageOps.exif_transpose(img)
        if img is None:
            raise ValueError(f"无法从文件中读取有效图像: {fname}")
        image = img.convert("RGB")

        t_image = pil2tensor(image)
        if ref_image is None:
            ref_image = t_image
        else:
            if t_image.shape[1:] != ref_image.shape[1:]:
                t_image = comfy.utils.common_upscale(
                    t_image.movedim(-1, 1),
                    ref_image.shape[2],
                    ref_image.shape[1],
                    "lanczos",
                    "center",
                ).movedim(1, -1)

        image_list.append(t_image)

    if not image_list:
        raise ValueError("未找到有效图像")

    image_batch = torch.cat(image_list, dim=0)
    images_out = [image_batch[i : i + 1, ...] for i in range(image_batch.shape[0])]

    file_list = [os.path.join(folder_path, file_list[i]) for i in range(start_index, end_index)]

    return image_batch, images_out, file_list


class LoadImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "LIST",
        "INT",
    )
    RETURN_NAMES = (
        "images_list",
        "image_batch",
        "file_list",
        "count",
    )
    OUTPUT_IS_LIST = (
        True,
        False,
        True,
        False,
    )
    FUNCTION = "make_list"
    CATEGORY = _CATEGORY
    DESCRIPTION = "读取文件夹中的图片，返回图片列表和图片批次"

    def make_list(self, input_path):
        image_batch, images_out, file_list = _load_images_from_folder(input_path)
        return (images_out, image_batch, file_list, len(file_list))
