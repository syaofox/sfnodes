import math
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from comfy.utils import common_upscale
from ...sf_utils.image_convert import mask2tensor, np2tensor, tensor2mask, tensor2np
from ...sf_utils.mask_utils import solid_mask
from ...sf_utils.image_convert import contrast_adaptive_sharpening
from nodes import LoadImage
import folder_paths
import comfy.utils
from nodes import MAX_RESOLUTION
import os
import torch.nn.functional as F


import json
from datetime import datetime


_CATEGORY = "sfnodes/image"
UPSCALE_METHODS = ["lanczos", "nearest-exact", "bilinear", "area", "bicubic"]

from ...sf_utils.resize_engine import (
    floor_divisible,
    make_divisible,
    make_even,
    total_pixels_to_wh,
)


class GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "width",
        "height",
        "count",
        "min_dimension",
        "max_dimension",
        "megapixels",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "获取图像的宽、高、数量、最小和最大边长，以及单张图像的像素数量（百万像素，保留两位小数）"

    def execute(self, image):
        width, height = image.shape[2], image.shape[1]
        megapixels = round(width * height / 1_000_000, 2)
        return {
            "ui": {
                "width": (width,),
                "height": (height,),
                "count": (image.shape[0],),
                "min_dimension": (min(width, height),),
                "max_dimension": (max(width, height),),
                "megapixels": (megapixels,),
            },
            "result": (
                width,
                height,
                image.shape[0],
                min(width, height),
                max(width, height),
                megapixels,
            ),
        }


class BaseImageScaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (UPSCALE_METHODS,),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height", "min_dimension")
    CATEGORY = _CATEGORY

    def scale_image(self, image, width, height, upscale_method, mask=None, divisible_by=16):
        width = floor_divisible(width, divisible_by)
        height = floor_divisible(height, divisible_by)

        image_tensor = image.movedim(-1, 1)
        scaled_image = common_upscale(
            image_tensor, width, height, upscale_method, "disabled"
        )
        scaled_image = scaled_image.movedim(1, -1)

        result_mask = solid_mask(width, height)
        if mask is not None:
            mask_image = mask2tensor(mask)
            mask_image = mask_image.movedim(-1, 1)
            mask_image = common_upscale(
                mask_image, width, height, upscale_method, "disabled"
            )
            mask_image = mask_image.movedim(1, -1)
            result_mask = tensor2mask(mask_image)

        return scaled_image, result_mask

    def prepare_result(self, scaled_image, result_mask, width, height):
        return {
            "ui": {
                "width": (width,),
                "height": (height,),
            },
            "result": (
                scaled_image,
                result_mask,
                width,
                height,
                min(width, height),
            ),
        }


class ImageScalerByPixels(BaseImageScaler):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()

        base_inputs["required"].update(
            {
                "total_pixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 16.0,
                        "step": 0.01,
                        "tooltip": "设置缩放比例，范围为0.01到16.0，步长为0.01",
                    },
                ),
                "limit": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "限制缩放比例，如果图像的像素数小于目标像素数，则不缩放图像",
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "向下取整到该数的倍数，1 表示不约束，默认 8",
                    },
                ),
            }
        )

        return base_inputs

    FUNCTION = "execute"
    DESCRIPTION = """
    将图片缩放到指定像素数，total_pixels为缩放比例，limit为True时，如果图像的像素数小于目标像素数，则不缩放图像
    divisible_by 会将最终宽高向下取整到该数的倍数（默认 8）
    """

    def execute(self, image, upscale_method, total_pixels, limit=True, divisible_by=8, mask=None):
        samples = image.movedim(-1, 1)
        total = int(total_pixels * 1024 * 1024)
        current_pixels = samples.shape[3] * samples.shape[2]

        # Only upscale if current pixels is less than target total, when limit is True
        if limit and current_pixels <= total:
            result_mask = (
                mask if mask is not None else solid_mask(image.shape[2], image.shape[1])
            )
            return self.prepare_result(
                image, result_mask, image.shape[2], image.shape[1]
            )

        computed = total_pixels_to_wh(samples.shape[3], samples.shape[2], total_pixels)
        width = floor_divisible(computed[0], divisible_by)
        height = floor_divisible(computed[1], divisible_by)

        scaled_image, result_mask = self.scale_image(
            image, width, height, upscale_method, mask, divisible_by
        )
        # scale_image 已按 divisible_by 取整，width/height 取最终张量尺寸
        width, height = scaled_image.shape[2], scaled_image.shape[1]
        return self.prepare_result(scaled_image, result_mask, width, height)


class ImageScaleBySpecifiedSide(BaseImageScaler):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update(
            {
                "size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "step": 1,
                        "max": 99999,
                        "tooltip": "设置缩放目标像素数，范围为0到99999，步长为1",
                    },
                ),
                "shorter": ("BOOLEAN", {"default": False, "tooltip": "参照短边缩放"}),
                "limit": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "限制缩放比例，如果图像的最短边小于size，则不缩放图像",
                    },
                ),
                "crop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "如果较长边超过阈值，则裁剪图像",
                    },
                ),
                "crop_threshold": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "step": 1,
                        "max": 99999,
                        "tooltip": "裁剪阈值，当较长边超过此值时触发裁剪",
                    },
                ),
                "crop_position": (
                    ["top", "bottom", "left", "right", "center"],
                    {
                        "default": "center",
                        "tooltip": "指定裁剪位置",
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "step": 1,
                        "max": 512,
                        "tooltip": "向下取整到该数的倍数，1 表示不约束，默认 8",
                    },
                ),
            }
        )
        return base_inputs

    FUNCTION = "execute"
    DESCRIPTION = """
    根据指定边长缩放图片，shorter为True时参照短边，否则参照长边
    limit为True时，如果图像的最短边小于size，则不缩放图像
    crop为True时，如果较长边超过阈值，则根据crop_position裁剪图像
    divisible_by 会将最终宽高向下取整到该数的倍数（默认 8）
    """

    def execute(
        self,
        image,
        size,
        upscale_method,
        shorter,
        limit,
        crop,
        crop_threshold,
        crop_position,
        divisible_by,
        mask=None,
    ):
        # Check if we should skip scaling
        min_side = min(image.shape[2], image.shape[1])
        if limit and min_side < size:
            width = floor_divisible(image.shape[2], divisible_by)
            height = floor_divisible(image.shape[1], divisible_by)
            # 如果尺寸发生变化，需要缩放
            if width != image.shape[2] or height != image.shape[1]:
                scaled_image, result_mask = self.scale_image(
                    image, width, height, upscale_method, mask, divisible_by
                )
                return self.prepare_result(scaled_image, result_mask, scaled_image.shape[2], scaled_image.shape[1])
            else:
                return self.prepare_result(
                    image,
                    mask
                    if mask is not None
                    else solid_mask(image.shape[2], image.shape[1]),
                    image.shape[2],
                    image.shape[1],
                )

        if shorter:
            reference_side_length = min(image.shape[2], image.shape[1])
        else:
            reference_side_length = max(image.shape[2], image.shape[1])

        scale_by = reference_side_length / size
        width = floor_divisible(round(image.shape[2] / scale_by), divisible_by)
        height = floor_divisible(round(image.shape[1] / scale_by), divisible_by)

        # Apply cropping if enabled and needed
        if crop:
            scaled_image, result_mask = self.scale_image(
                image, width, height, upscale_method, mask, divisible_by
            )

            # Check if cropping is needed (one dimension exceeds the crop threshold)
            if (shorter and max(width, height) > crop_threshold) or (
                not shorter and min(width, height) > crop_threshold
            ):
                scaled_image, result_mask = self._crop_image(
                    scaled_image, result_mask, crop_threshold, crop_position, shorter
                )
            width, height = scaled_image.shape[2], scaled_image.shape[1]
            # 裁剪后也确保能被divisible_by整除（向下取整）
            width = floor_divisible(width, divisible_by)
            height = floor_divisible(height, divisible_by)
            # 如果尺寸发生变化，需要重新缩放
            if width != scaled_image.shape[2] or height != scaled_image.shape[1]:
                scaled_image, result_mask = self.scale_image(
                    scaled_image, width, height, upscale_method, result_mask, divisible_by
                )
        else:
            scaled_image, result_mask = self.scale_image(
                image, width, height, upscale_method, mask, divisible_by
            )

        return self.prepare_result(scaled_image, result_mask, scaled_image.shape[2], scaled_image.shape[1])

    def _crop_image(self, image, mask, target_size, crop_position, shorter):
        """Crop image to target size based on specified position"""
        width, height = image.shape[2], image.shape[1]

        if shorter:
            # When shorter=True, we want to crop the longer side
            if width > height:
                # Landscape image - crop width
                crop_width = target_size
                crop_height = height
                x = self._get_crop_coordinate(width, crop_width, crop_position)
                y = 0
            else:
                # Portrait image - crop height
                crop_width = width
                crop_height = target_size
                x = 0
                y = self._get_crop_coordinate(height, crop_height, crop_position)
        else:
            # When shorter=False, we want to crop the shorter side
            if width > height:
                # Landscape image - crop height
                crop_width = width
                crop_height = target_size
                x = 0
                y = self._get_crop_coordinate(height, crop_height, crop_position)
            else:
                # Portrait image - crop width
                crop_width = target_size
                crop_height = height
                x = self._get_crop_coordinate(width, crop_width, crop_position)
                y = 0

        # Perform cropping
        cropped_image = image[:, y : y + crop_height, x : x + crop_width, :]

        if mask is not None:
            cropped_mask = mask[y : y + crop_height, x : x + crop_width]
        else:
            cropped_mask = solid_mask(crop_width, crop_height)

        return cropped_image, cropped_mask

    def _get_crop_coordinate(self, dimension_size, crop_size, position):
        """Calculate crop coordinate based on position"""
        if position == "top" or position == "left":
            return 0
        elif position == "bottom" or position == "right":
            return dimension_size - crop_size
        elif position == "center":
            return (dimension_size - crop_size) // 2
        else:
            return 0  # Default to top/left


class ComputeImageScaleRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_max_size": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 0,
                        "step": 1,
                        "max": 99999,
                        "tooltip": "设置目标最大尺寸，范围为0到99999，步长为1",
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "向下取整到该数的倍数，1 表示不约束，默认 8",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "rescale_ratio",
        "width",
        "height",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据引用图片的大小和目标最大尺寸，返回缩放比例和缩放后的宽高；divisible_by 会将宽高向下取整到该数的倍数（默认 8）"

    def execute(self, image, target_max_size, divisible_by=8):
        samples = image.movedim(-1, 1)
        width, height = samples.shape[3], samples.shape[2]

        rescale_ratio = target_max_size / max(width, height)

        new_width = floor_divisible(round(width * rescale_ratio), divisible_by)
        new_height = floor_divisible(round(height * rescale_ratio), divisible_by)

        return {
            "ui": {
                "rescale_ratio": (rescale_ratio,),
                "width": (new_width,),
                "height": (new_height,),
            },
            "result": (
                rescale_ratio,
                new_width,
                new_height,
            ),
        }





class ScaleImageToSquare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size_length": (
                    "INT",
                    {"default": 1024, "min": 224, "max": 10000, "step": 1},
                ),
                "interpolation": (
                    ["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],
                ),
                "crop_position": (["top", "bottom", "left", "right", "center", "pad"],),
                "sharpening": (
                    "FLOAT",
                    {"default": 0.0, "min": 0, "max": 1, "step": 0.05},
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "向下取整到该数的倍数，1 表示不约束，默认 8",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "prep_image"

    CATEGORY = _CATEGORY
    DESCRIPTION = "将图片缩放为正方形，可选择裁剪或填充方式，并输出填充区域的mask；divisible_by 会将最终尺寸向下取整（默认 8）"

    def prep_image(
        self,
        image,
        size_length,
        interpolation="LANCZOS",
        crop_position="center",
        sharpening=0.0,
        divisible_by=8,
    ):
        size_length = floor_divisible(size_length, divisible_by)
        size = (size_length, size_length)
        _, oh, ow, _ = image.shape
        output = image.permute([0, 3, 1, 2])

        if crop_position == "pad":
            if oh != ow:
                if oh > ow:
                    pad = (oh - ow) // 2
                    pad = (pad, 0, pad, 0)
                elif ow > oh:
                    pad = (ow - oh) // 2
                    pad = (0, pad, 0, pad)
                output = T.functional.pad(output, pad, fill=0)  # type: ignore
        else:
            crop_size = min(oh, ow)
            x = (ow - crop_size) // 2
            y = (oh - crop_size) // 2
            if "top" in crop_position:
                y = 0
            elif "bottom" in crop_position:
                y = oh - crop_size
            elif "left" in crop_position:
                x = 0
            elif "right" in crop_position:
                x = ow - crop_size

            x2 = x + crop_size
            y2 = y + crop_size

            output = output[:, :, y:y2, x:x2]

        imgs = []
        for img in output:
            img = T.ToPILImage()(img)  # using PIL for better results
            img = img.resize(size, resample=Image.Resampling[interpolation])
            imgs.append(T.ToTensor()(img))
        output = torch.stack(imgs, dim=0)
        del imgs, img

        if sharpening > 0:
            output = contrast_adaptive_sharpening(output, sharpening)

        output = output.permute([0, 2, 3, 1])

        # 创建mask，标记填充区域（填充区域为1，原图区域为0）
        # 默认情况下，如果不是pad模式或图像已经是正方形，mask应该全为0（表示没有填充区域）
        mask = torch.zeros(
            (output.shape[0], size_length, size_length), dtype=torch.float32
        )

        # 如果使用pad模式且图像不是正方形，创建对应的mask
        if crop_position == "pad" and oh != ow:
            if oh > ow:
                # 计算填充后的总宽度
                padded_width = oh
                # 计算原始图像在填充后的宽度比例
                original_ratio = ow / padded_width
                # 计算缩放后的原始图像宽度
                scaled_original_width = int(original_ratio * size_length)
                # 计算填充区域宽度
                pad_width = (size_length - scaled_original_width) // 2
                mask[:, :, :pad_width] = 1.0  # 左侧填充区域
                mask[:, :, size_length - pad_width :] = 1.0  # 右侧填充区域
            elif ow > oh:
                # 计算填充后的总高度
                padded_height = ow
                # 计算原始图像在填充后的高度比例
                original_ratio = oh / padded_height
                # 计算缩放后的原始图像高度
                scaled_original_height = int(original_ratio * size_length)
                # 计算填充区域高度
                pad_height = (size_length - scaled_original_height) // 2
                mask[:, :pad_height, :] = 1.0  # 上方填充区域
                mask[:, size_length - pad_height :, :] = 1.0  # 下方填充区域

        return (output, mask)


class ImageResizePlus:
    DESCRIPTION = "高级图片缩放，支持拉伸、保持比例、填充裁剪和条件缩放；divisible_by 会将最终宽高向下取整到该数的倍数（默认 8）"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size_mode": (
                    ["width & height", "total pixels"],
                    {
                        "default": "width & height",
                        "tooltip": "目标尺寸模式：width & height=按宽高，total pixels=按总像素数（忽略 width/height，保持源图宽高比）",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                "total_pixels": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": 0.01,
                        "max": 16.0,
                        "step": 0.01,
                        "tooltip": "目标总像素（百万像素），1.00 = 1024×1024 = 1,048,576 像素（与原生 ImageScaleToTotalPixels 一致）；仅 size_mode=total pixels 生效",
                    },
                ),
                "interpolation": (
                    [
                        "nearest",
                        "bilinear",
                        "bicubic",
                        "area",
                        "nearest-exact",
                        "lanczos",
                    ],
                    {"default": "lanczos"},
                ),
                "method": (
                    ["stretch", "keep proportion", "fill / crop", "pad"],
                    {"default": "keep proportion"},
                ),
                "condition": (
                    [
                        "always",
                        "downscale if bigger",
                        "upscale if smaller",
                        "if bigger area",
                        "if smaller area",
                    ],
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "向下取整到该数的倍数，1 表示不约束，默认 8",
                    },
                ),
                "crop_position": (
                    ["center", "top", "bottom"],
                    {"default": "center"},
                ),
                "pad_color": (
                    "COLOR",
                    {
                        "default": "#000000",
                        "tooltip": "pad 模式的填充颜色，hex 格式（如 #000000）",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "可选的遮罩，将应用相同的缩放变换"}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "width",
        "height",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "高级图片缩放，支持拉伸、保持比例、填充裁剪和条件缩放；divisible_by 会将最终宽高向下取整到该数的倍数（默认 8）"

    def execute(
        self,
        image,
        width,
        height,
        size_mode="width & height",
        total_pixels=1.0,
        method="keep proportion",
        interpolation="lanczos",
        condition="always",
        divisible_by=8,
        keep_proportion=False,
        crop_position="center",
        pad_color="#000000",
        mask=None,
    ):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if size_mode == "total pixels":
            computed = total_pixels_to_wh(ow, oh, total_pixels)
            if computed is not None:
                width, height = computed

        if divisible_by > 1:
            width = floor_divisible(width, divisible_by)
            height = floor_divisible(height, divisible_by)

        if method == "keep proportion" or method == "pad":
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = oh

            ratio = min(width / ow, height / oh)
            new_width = round(ow * ratio)
            new_height = round(oh * ratio)

            if method == "pad":
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith("fill"):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow * ratio)
            new_height = round(oh * ratio)
            x = (new_width - width) // 2
            if crop_position == "top":
                y = 0
            elif crop_position == "bottom":
                y = new_height - height
            else:
                y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= x2 - new_width
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= y2 - new_height
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if mask is not None:
            mask_tensor = mask2tensor(mask)

        do_resize = (
            "always" in condition
            or ("downscale if bigger" == condition and (oh > height or ow > width))
            or ("upscale if smaller" == condition and (oh < height or ow < width))
            or ("bigger area" in condition and (oh * ow > height * width))
            or ("smaller area" in condition and (oh * ow < height * width))
        )

        if do_resize:
            outputs = image.permute(0, 3, 1, 2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(
                    outputs, size=(height, width), mode=interpolation
                )

            if mask is not None:
                mask_tensor = mask_tensor.permute(0, 3, 1, 2)
                mask_tensor = F.interpolate(
                    mask_tensor, size=(height, width), mode="nearest"
                )

            if method == "pad":
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    if isinstance(pad_color, str):
                        hex_color = pad_color.lstrip("#")
                        pad_color_r = int(hex_color[0:2], 16)
                        pad_color_g = int(hex_color[2:4], 16)
                        pad_color_b = int(hex_color[4:6], 16)
                    else:
                        pad_color_r, pad_color_g, pad_color_b = pad_color

                    if (pad_color_r, pad_color_g, pad_color_b) == (0, 0, 0):
                        outputs = F.pad(
                            outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0
                        )
                    else:
                        b, c, h, w = outputs.shape
                        canvas = torch.zeros(
                            (
                                b,
                                c,
                                h + pad_top + pad_bottom,
                                w + pad_left + pad_right,
                            ),
                            dtype=outputs.dtype,
                            device=outputs.device,
                        )
                        canvas[:, 0] = pad_color_r / 255.0
                        canvas[:, 1] = pad_color_g / 255.0
                        canvas[:, 2] = pad_color_b / 255.0
                        if c > 3:
                            canvas[:, 3] = 1.0
                        canvas[
                            :, :, pad_top : pad_top + h, pad_left : pad_left + w
                        ] = outputs
                        outputs = canvas

                    if mask is not None:
                        mask_tensor = F.pad(
                            mask_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=1
                        )

            outputs = outputs.permute(0, 2, 3, 1)
            if mask is not None:
                mask_tensor = mask_tensor.permute(0, 2, 3, 1)

            if method.startswith("fill"):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
                    if mask is not None:
                        mask_tensor = mask_tensor[:, y:y2, x:x2, :]
        else:
            outputs = image

        if divisible_by > 1 and (
            outputs.shape[2] % divisible_by != 0 or outputs.shape[1] % divisible_by != 0
        ):
            w = outputs.shape[2]
            h = outputs.shape[1]
            cx = (w % divisible_by) // 2
            cy = (h % divisible_by) // 2
            cx2 = w - ((w % divisible_by) - cx)
            cy2 = h - ((h % divisible_by) - cy)
            outputs = outputs[:, cy:cy2, cx:cx2, :]
            if mask is not None:
                mask_tensor = mask_tensor[:, cy:cy2, cx:cx2, :]

        outputs = torch.clamp(outputs, 0, 1)

        out_mask = tensor2mask(mask_tensor) if mask is not None else None

        return (
            outputs,
            out_mask,
            outputs.shape[2],
            outputs.shape[1],
        )


class ApexSmartResize:
    """
    Apex Smart Resize - Automatically snaps to closest compatible resolution
    Intelligent resolution detection and scaling with proportion preservation
    """

    def __init__(self):
        # Define compatible resolutions for different AI models
        self.resolution_sets = {
            "Standard": [
                (1024, 1024),
                (1152, 896),
                (896, 1152),
                (1216, 832),
                (832, 1216),
                (1344, 768),
                (768, 1344),
                (1536, 640),
                (640, 1536),
                (832, 1280),
                (1280, 832),
                (704, 1504),
                (1504, 704),
                (896, 1344),
                (1344, 896),
                (960, 1280),
                (1280, 960),
                (512, 512),
                (768, 768),
                (640, 640),
            ],
            "Extended": [
                (1024, 1024),
                (1152, 896),
                (896, 1152),
                (1216, 832),
                (832, 1216),
                (1344, 768),
                (768, 1344),
                (1536, 640),
                (640, 1536),
                (1728, 576),
                (576, 1728),
                (1920, 512),
                (512, 1920),
                (2048, 512),
                (512, 2048),
                (832, 1280),
                (1280, 832),
                (704, 1504),
                (1504, 704),
                (960, 1536),
                (1536, 960),
                (1088, 1472),
                (1472, 1088),
            ],
            "Flux": [
                (1024, 1024),
                (768, 1344),
                (832, 1216),
                (896, 1152),
                (1152, 896),
                (1216, 832),
                (1344, 768),
                (512, 512),
                (640, 1536),
                (1536, 640),
                (704, 1504),
                (1504, 704),
                (832, 1280),
                (1280, 832),
            ],
            "Portrait": [
                (832, 1216),
                (768, 1344),
                (640, 1536),
                (896, 1152),
                (832, 1280),
                (704, 1504),
                (512, 768),
                (576, 1024),
                (640, 960),
                (720, 1280),
                (768, 1024),
                (896, 1344),
            ],
            "Landscape": [
                (1216, 832),
                (1344, 768),
                (1536, 640),
                (1152, 896),
                (1280, 832),
                (1504, 704),
                (768, 512),
                (1024, 576),
                (960, 640),
                (1280, 720),
                (1024, 768),
                (1344, 896),
            ],
            "Square": [
                (512, 512),
                (640, 640),
                (768, 768),
                (832, 832),
                (896, 896),
                (1024, 1024),
                (1152, 1152),
                (1216, 1216),
                (1280, 1280),
                (1344, 1344),
            ],
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution_set": (
                    [
                        "Standard",  # Core SDXL/Flux resolutions
                        "Extended",  # Extra experimental sizes
                        "Flux",  # Flux-optimized
                        "Portrait",  # Tall formats
                        "Landscape",  # Wide formats
                        "Square",  # Square only
                    ],
                    {"default": "Standard"},
                ),
                "snap_method": (
                    [
                        "keep_proportion",  # Scale largest side first, maintain aspect ratio
                        "closest_area",  # Snap to closest total pixel count
                        "closest_ratio",  # Snap to closest aspect ratio
                        "prefer_larger",  # Prefer larger resolutions
                        "prefer_smaller",  # Prefer smaller resolutions
                    ],
                    {"default": "keep_proportion"},
                ),
                "resize_mode": (
                    [
                        "crop_center",  # Crop from center
                        "stretch",  # Stretch to exact dimensions
                        "fit_pad_black",  # Fit with black padding
                        "fit_pad_white",  # Fit with white padding
                        "fit_pad_edge",  # Fit with edge extension
                    ],
                    {"default": "crop_center"},
                ),
                "interpolation": (
                    ["lanczos", "bicubic", "bilinear", "nearest"],
                    {"default": "lanczos"},
                ),
                "show_candidates": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "在控制台显示候选分辨率列表",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "可选的遮罩，将应用相同的缩放变换"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "image",
        "mask",
        "width",
        "height",
        "scale_factor",
        "resolution_info",
        "console_log",
    )
    FUNCTION = "smart_resize"
    CATEGORY = _CATEGORY
    DESCRIPTION = "智能分辨率缩放，自动匹配最佳兼容分辨率，支持多策略"

    def smart_resize(
        self,
        image,
        resolution_set,
        snap_method,
        resize_mode,
        interpolation,
        show_candidates,
        mask=None,
    ):
        start_time = datetime.now()

        try:
            # Get input dimensions
            if len(image.shape) == 4:
                batch_size, orig_h, orig_w, channels = image.shape
            else:
                image = image.unsqueeze(0)
                batch_size, orig_h, orig_w, channels = image.shape

            orig_area = orig_w * orig_h
            orig_aspect = orig_w / orig_h

            # Find best target resolution
            target_w, target_h, info, candidates_info = self._find_best_resolution(
                orig_w, orig_h, resolution_set, snap_method, show_candidates
            )

            target_area = target_w * target_h
            scale_factor = math.sqrt(target_area / orig_area)

            # Generate console data
            console_data = self._create_console_data(
                orig_w,
                orig_h,
                target_w,
                target_h,
                scale_factor,
                resize_mode,
                resolution_set,
                snap_method,
                candidates_info,
                start_time,
            )

            # Resize the image and mask
            resized_image, resized_mask = self._apply_resize(
                image, target_w, target_h, resize_mode, interpolation, mask
            )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            console_data["processing_time_seconds"] = round(processing_time, 3)

            # Format console output
            console_output = json.dumps(console_data, indent=2)

            return (
                resized_image,
                resized_mask,
                target_w,
                target_h,
                scale_factor,
                info,
                console_output,
            )

        except Exception as e:
            error_console = json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                    "original_size": f"{orig_w}x{orig_h}",
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )
            out_mask = tensor2mask(mask2tensor(mask)) if mask is not None else None
            return (image, out_mask, orig_w, orig_h, 1.0, f"Error: {str(e)}", error_console)

    def _create_console_data(
        self,
        orig_w,
        orig_h,
        target_w,
        target_h,
        scale_factor,
        resize_mode,
        resolution_set,
        snap_method,
        candidates_info,
        start_time,
    ):
        """Create structured data for Apex Console"""

        orig_area = orig_w * orig_h
        target_area = target_w * target_h
        memory_change_mb = ((target_area - orig_area) * 4 * 3) / (
            1024 * 1024
        )  # Assume RGB float32

        return {
            "action": "Smart Resize Complete",
            "status": "success",
            "timestamp": start_time.isoformat(),
            "input": {
                "size": f"{orig_w}x{orig_h}",
                "aspect_ratio": round(orig_w / orig_h, 3),
                "total_pixels": f"{orig_area:,}",
                "estimated_memory_mb": round((orig_area * 4 * 3) / (1024 * 1024), 1),
            },
            "output": {
                "size": f"{target_w}x{target_h}",
                "aspect_ratio": round(target_w / target_h, 3),
                "total_pixels": f"{target_area:,}",
                "estimated_memory_mb": round((target_area * 4 * 3) / (1024 * 1024), 1),
            },
            "processing": {
                "resolution_set": resolution_set,
                "snap_method": snap_method,
                "resize_mode": resize_mode,
                "scale_factor": round(scale_factor, 3),
                "size_change_percent": round(
                    ((scale_factor * scale_factor - 1) * 100), 1
                ),
                "memory_change_mb": round(memory_change_mb, 1),
            },
            "candidates": candidates_info,
        }

    def _find_best_resolution(
        self, orig_w, orig_h, resolution_set, snap_method, show_candidates
    ):
        """Find the best target resolution based on method"""

        resolutions = self.resolution_sets[resolution_set]
        orig_area = orig_w * orig_h
        orig_aspect = orig_w / orig_h

        if snap_method == "keep_proportion":
            target_w, target_h, info, candidates = self._keep_proportion_snap(
                orig_w, orig_h, resolutions, show_candidates
            )
            return target_w, target_h, info, candidates

        # Other methods
        candidates = []

        for w, h in resolutions:
            area = w * h
            aspect = w / h
            scale_factor = math.sqrt(area / orig_area)
            aspect_diff = abs(aspect - orig_aspect)
            area_diff = abs(area - orig_area)

            candidates.append(
                {
                    "resolution": f"{w}x{h}",
                    "scale_factor": round(scale_factor, 3),
                    "aspect_ratio": round(aspect, 3),
                    "aspect_diff": round(aspect_diff, 3),
                    "area_diff": area_diff,
                    "total_pixels": f"{area:,}",
                }
            )

        # Sort candidates based on method
        if snap_method == "closest_area":
            candidates.sort(key=lambda x: x["area_diff"])
            best = candidates[0]
            info = f"Closest area match from {resolution_set}"

        elif snap_method == "closest_ratio":
            candidates.sort(key=lambda x: x["aspect_diff"])
            best = candidates[0]
            info = f"Closest aspect ratio from {resolution_set}"

        elif snap_method == "prefer_larger":
            larger_candidates = [c for c in candidates if c["area_diff"] >= 0]
            if larger_candidates:
                larger_candidates.sort(key=lambda x: x["area_diff"])
                best = larger_candidates[0]
            else:
                candidates.sort(key=lambda x: x["area_diff"], reverse=True)
                best = candidates[0]
            info = f"Prefer larger from {resolution_set}"

        else:  # prefer_smaller
            smaller_candidates = [c for c in candidates if c["area_diff"] <= 0]
            if smaller_candidates:
                smaller_candidates.sort(key=lambda x: x["area_diff"], reverse=True)
                best = smaller_candidates[0]
            else:
                candidates.sort(key=lambda x: x["area_diff"])
                best = candidates[0]
            info = f"Prefer smaller from {resolution_set}"

        # Extract target dimensions
        w, h = map(int, best["resolution"].split("x"))
        candidates_info = {
            "method": snap_method,
            "total_evaluated": len(candidates),
            "top_5": sorted(candidates, key=lambda x: x["area_diff"])[:5],
        }

        return w, h, info, candidates_info

    def _keep_proportion_snap(self, orig_w, orig_h, resolutions, show_candidates):
        """Scale by largest dimension while maintaining aspect ratio"""

        orig_aspect = orig_w / orig_h
        is_portrait = orig_h > orig_w

        best_match = None
        best_score = float("inf")
        candidates = []

        for target_w, target_h in resolutions:
            target_is_portrait = target_h > target_w

            # Only consider resolutions with same orientation
            if is_portrait == target_is_portrait:
                if is_portrait:
                    # Scale by height (largest dimension)
                    scale_factor = target_h / orig_h
                    calculated_w = orig_w * scale_factor

                    # Round to nearest multiple of 64 for better compatibility
                    snapped_w = round(calculated_w / 64) * 64

                    # Check if this creates a valid resolution
                    if abs(snapped_w - target_w) <= 64:  # Allow some tolerance
                        aspect_diff = abs((target_w / target_h) - orig_aspect)
                        scale_diff = abs(scale_factor - 1.0)

                        # Scoring: prefer similar aspect ratio and reasonable scaling
                        score = aspect_diff * 10 + scale_diff * 2

                        candidates.append(
                            {
                                "resolution": f"{target_w}x{target_h}",
                                "scale_factor": round(scale_factor, 3),
                                "aspect_diff": round(aspect_diff, 3),
                                "score": round(score, 3),
                            }
                        )

                        if score < best_score:
                            best_score = score
                            best_match = (target_w, target_h)

                else:  # Landscape
                    # Scale by width (largest dimension)
                    scale_factor = target_w / orig_w
                    calculated_h = orig_h * scale_factor

                    snapped_h = round(calculated_h / 64) * 64

                    if abs(snapped_h - target_h) <= 64:
                        aspect_diff = abs((target_w / target_h) - orig_aspect)
                        scale_diff = abs(scale_factor - 1.0)

                        score = aspect_diff * 10 + scale_diff * 2

                        candidates.append(
                            {
                                "resolution": f"{target_w}x{target_h}",
                                "scale_factor": round(scale_factor, 3),
                                "aspect_diff": round(aspect_diff, 3),
                                "score": round(score, 3),
                            }
                        )

                        if score < best_score:
                            best_score = score
                            best_match = (target_w, target_h)

        # Fallback to closest aspect ratio if no good match
        if best_match is None:
            best_aspect_diff = float("inf")
            for w, h in resolutions:
                aspect_diff = abs((w / h) - orig_aspect)
                if aspect_diff < best_aspect_diff:
                    best_aspect_diff = aspect_diff
                    best_match = (w, h)

        target_w, target_h = best_match
        info = f"Keep proportion snap from {len(resolutions)} resolutions"

        candidates_info = {
            "method": "keep_proportion",
            "orientation": "portrait" if orig_h > orig_w else "landscape",
            "total_evaluated": len(candidates),
            "top_5": sorted(candidates, key=lambda x: x["score"])[:5]
            if candidates
            else [],
        }

        return target_w, target_h, info, candidates_info

    def _apply_resize(self, image, target_w, target_h, resize_mode, interpolation, mask=None):
        """Apply the actual resizing with specified method"""

        if mask is not None:
            mask_tensor = mask2tensor(mask)

        if resize_mode == "stretch":
            img_out = self._resize_tensor(image, target_w, target_h, interpolation)
            mask_out = self._resize_mask(mask_tensor, target_w, target_h) if mask is not None else None

        elif resize_mode == "crop_center":
            img_out, mask_out = self._crop_center_resize(
                image, target_w, target_h, interpolation, mask_tensor if mask is not None else None
            )

        elif resize_mode == "fit_pad_black":
            img_out, mask_out = self._fit_pad_resize(
                image, target_w, target_h, interpolation, pad_color=0.0,
                mask_tensor=mask_tensor if mask is not None else None
            )

        elif resize_mode == "fit_pad_white":
            img_out, mask_out = self._fit_pad_resize(
                image, target_w, target_h, interpolation, pad_color=1.0,
                mask_tensor=mask_tensor if mask is not None else None
            )

        elif resize_mode == "fit_pad_edge":
            img_out, mask_out = self._fit_pad_edge_resize(
                image, target_w, target_h, interpolation,
                mask_tensor=mask_tensor if mask is not None else None
            )

        else:
            img_out = self._resize_tensor(image, target_w, target_h, interpolation)
            mask_out = self._resize_mask(mask_tensor, target_w, target_h) if mask is not None else None

        out_mask = tensor2mask(mask_out) if mask_out is not None else None
        return img_out, out_mask

    def _resize_mask(self, mask_tensor, width, height):
        mask_bchw = mask_tensor.permute(0, 3, 1, 2)
        resized = F.interpolate(mask_bchw, size=(height, width), mode="nearest")
        return resized.permute(0, 2, 3, 1)

    def _resize_tensor(self, image, width, height, interpolation):
        """Core tensor resize function"""

        image_bchw = image.permute(0, 3, 1, 2)

        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "lanczos": "bicubic",  # PyTorch fallback
        }

        mode = mode_map.get(interpolation, "bicubic")
        antialias = mode in ["bilinear", "bicubic"]

        resized = F.interpolate(
            image_bchw, size=(height, width), mode=mode, antialias=antialias
        )

        return resized.permute(0, 2, 3, 1)

    def _crop_center_resize(self, image, target_w, target_h, interpolation, mask_tensor=None):
        """Resize to cover target, then center crop"""

        orig_h, orig_w = image.shape[1], image.shape[2]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h

        if orig_aspect > target_aspect:
            # Scale by height, crop width
            new_h = target_h
            new_w = int(target_h * orig_aspect)
        else:
            # Scale by width, crop height
            new_w = target_w
            new_h = int(target_w / orig_aspect)

        # Resize to cover
        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        mask_resized = self._resize_mask(mask_tensor, new_w, new_h) if mask_tensor is not None else None

        # Center crop
        crop_x = max(0, (new_w - target_w) // 2)
        crop_y = max(0, (new_h - target_h) // 2)

        cropped = resized[:, crop_y : crop_y + target_h, crop_x : crop_x + target_w, :]
        cropped_mask = mask_resized[:, crop_y : crop_y + target_h, crop_x : crop_x + target_w, :] if mask_resized is not None else None

        return cropped, cropped_mask

    def _fit_pad_resize(self, image, target_w, target_h, interpolation, pad_color, mask_tensor=None):
        """Fit image with solid color padding"""

        orig_h, orig_w = image.shape[1], image.shape[2]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h

        if orig_aspect > target_aspect:
            # Fit to width
            new_w = target_w
            new_h = int(target_w / orig_aspect)
        else:
            # Fit to height
            new_h = target_h
            new_w = int(target_h * orig_aspect)

        # Resize to fit
        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        mask_resized = self._resize_mask(mask_tensor, new_w, new_h) if mask_tensor is not None else None

        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if pad_w > 0 or pad_h > 0:
            image_bchw = resized.permute(0, 3, 1, 2)
            padded = F.pad(
                image_bchw,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=pad_color,
            )
            result = padded.permute(0, 2, 3, 1)

            if mask_resized is not None:
                mask_bchw = mask_resized.permute(0, 3, 1, 2)
                mask_padded = F.pad(
                    mask_bchw,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
                mask_resized = mask_padded.permute(0, 2, 3, 1)
        else:
            result = resized

        return result, mask_resized

    def _fit_pad_edge_resize(self, image, target_w, target_h, interpolation, mask_tensor=None):
        """Fit image with edge replication padding"""

        orig_h, orig_w = image.shape[1], image.shape[2]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h

        if orig_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / orig_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * orig_aspect)

        # Resize to fit
        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        mask_resized = self._resize_mask(mask_tensor, new_w, new_h) if mask_tensor is not None else None

        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if pad_w > 0 or pad_h > 0:
            image_bchw = resized.permute(0, 3, 1, 2)
            padded = F.pad(
                image_bchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
            )
            result = padded.permute(0, 2, 3, 1)

            if mask_resized is not None:
                mask_bchw = mask_resized.permute(0, 3, 1, 2)
                mask_padded = F.pad(
                    mask_bchw,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
                mask_resized = mask_padded.permute(0, 2, 3, 1)
        else:
            result = resized

        return result, mask_resized



