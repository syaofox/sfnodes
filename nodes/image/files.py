import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.utils

from ...sf_utils.image_convert import pil2tensor

_CATEGORY = "sfnodes/image"


class LoadImageFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "images"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_full_path")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "读取指定路径图片，返回图片和图片名称"

    def execute(self, image_path):
        # 去掉可能存在的双引号
        image_path = image_path.strip('"')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"文件未找到: {image_path}")

        file_full_path = str(Path(image_path).absolute())

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if img is None:
            raise ValueError(f"无法从文件中读取有效图像: {image_path}")

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert("RGB")

        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image, file_full_path)


class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "", "multiline": False}),
                "start_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "tooltip": "设置开始索引"},
                ),
                "max_index": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999, "tooltip": "设置最大索引"},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "LIST",
    )
    RETURN_NAMES = (
        "images_list",
        "image_batch",
        "file_list",
    )
    OUTPUT_IS_LIST = (
        True,
        False,
        True,
    )
    FUNCTION = "make_list"
    CATEGORY = _CATEGORY
    DESCRIPTION = "读取文件夹中的图片，返回图片列表和图片批次"

    def make_list(self, start_index, max_index, input_path):
        # 检查输入路径是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件夹未找到: {input_path}")

        # 检查文件夹是否为空
        if not os.listdir(input_path):
            raise ValueError(f"文件夹为空: {input_path}")

        # 对文件列表进行排序
        file_list = sorted(
            os.listdir(input_path),
            key=lambda s: sum(
                ((s, int(n)) for s, n in re.findall(r"(\D+)(\d+)", "a%s0" % s)), ()
            ),
        )

        image_list = []

        # 确保 start_index 在列表范围内
        start_index = max(0, min(start_index, len(file_list) - 1))

        # 计算结束索引
        end_index = min(start_index + max_index, len(file_list))

        ref_image = None

        for num in range(start_index, end_index):
            fname = os.path.join(input_path, file_list[num])
            img = Image.open(fname)
            img = ImageOps.exif_transpose(img)
            if img is None:
                raise ValueError(f"无法从文件中读取有效图像: {fname}")
            image = img.convert("RGB")

            t_image = pil2tensor(image)
            # 确保所有图像的尺寸相同
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

        # 完整路径
        file_list = [
            os.path.join(input_path, file_list[i]) for i in range(len(file_list))
        ][start_index:end_index]
        return (
            images_out,
            image_batch,
            file_list,
        )


class SelectFace:
    dir_dict = {}

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        target_dir = os.path.join(current_dir, "data", "face_pieces")
        if os.path.exists(target_dir):
            for d in Path(target_dir).iterdir():
                if d.is_dir():
                    cls.dir_dict[d.name] = d
        else:
            os.makedirs(target_dir, exist_ok=True)

        return {
            "required": {"face_name": (list(cls.dir_dict.keys()), {"default": ""})},
            "optional": {"char_name": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "face_path",
        "face_name",
    )
    FUNCTION = "execute"

    CATEGORY = _CATEGORY
    DESCRIPTION = "从特定路径选择人脸，子文件夹名即为人脸名称，路径在facepath文件中设置"

    def execute(self, face_name, char_name=None):
        if char_name:
            face_name = char_name
        return (
            str(self.dir_dict[face_name]),
            face_name,
        )


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
        # 检查输入路径是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件夹未找到: {input_path}")

        # 检查文件夹是否为空
        if not os.listdir(input_path):
            raise ValueError(f"文件夹为空: {input_path}")

        # 对文件列表进行排序
        file_list = sorted(
            os.listdir(input_path),
            key=lambda s: sum(
                ((s, int(n)) for s, n in re.findall(r"(\D+)(\d+)", "a%s0" % s)), ()
            ),
        )

        image_list = []

        ref_image = None

        for file in file_list:
            fname = os.path.join(input_path, file)
            img = Image.open(fname)
            img = ImageOps.exif_transpose(img)
            if img is None:
                raise ValueError(f"无法从文件中读取有效图像: {fname}")
            image = img.convert("RGB")

            t_image = pil2tensor(image)
            # 确保所有图像的尺寸相同
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

        # 完整路径
        file_list = [os.path.join(input_path, file) for file in file_list]
        return (
            images_out,
            image_batch,
            file_list,
            len(file_list),
        )
