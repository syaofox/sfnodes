import os
import re
import torch

import comfy.utils

from PIL import Image, ImageOps
from .utils.image_convert import pil2tensor

_CATEGORY = 'sfnodes/files'


class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'input_path': ('STRING', {'default': '', 'multiline': False}),
                'start_index': ('INT', {'default': 0, 'min': 0, 'max': 9999}),
                'max_index': ('INT', {'default': 1, 'min': 1, 'max': 9999}),
            }
        }

    RETURN_TYPES = (
        'IMAGE',
        'IMAGE',
        'LIST',
    )
    RETURN_NAMES = (
        'images_list',
        'image_batch',
        'file_list',
    )
    OUTPUT_IS_LIST = (
        True,
        False,
        True,
    )
    FUNCTION = 'make_list'
    CATEGORY = _CATEGORY
    DESCRIPTION = '读取文件夹中的图片，返回图片列表和图片批次'

    def make_list(self, start_index, max_index, input_path):
        # 检查输入路径是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'文件夹未找到: {input_path}')

        # 检查文件夹是否为空
        if not os.listdir(input_path):
            raise ValueError(f'文件夹为空: {input_path}')

        # 对文件列表进行排序
        file_list = sorted(
            os.listdir(input_path),
            key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()),
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
                raise ValueError(f'无法从文件中读取有效图像: {fname}')
            image = img.convert('RGB')

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
                        'lanczos',
                        'center',
                    ).movedim(1, -1)

            image_list.append(t_image)

        if not image_list:
            raise ValueError('未找到有效图像')

        image_batch = torch.cat(image_list, dim=0)
        images_out = [image_batch[i : i + 1, ...] for i in range(image_batch.shape[0])]

        # 完整路径 
        file_list = [os.path.join(input_path, file_list[i]) for i in range(len(file_list))][start_index:end_index]
        return (
            images_out,
            image_batch,
            file_list,
        )


