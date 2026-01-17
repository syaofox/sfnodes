import math
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from comfy.utils import common_upscale
from .utils.image_convert import mask2tensor, np2tensor, tensor2mask, tensor2np
from .utils.mask_utils import solid_mask
from .utils.image_convert import contrast_adaptive_sharpening
from nodes import LoadImage
import folder_paths
import comfy.utils
from nodes import MAX_RESOLUTION
import os
import torch.nn.functional as F


import json
from datetime import datetime


_CATEGORY = "sfnodes/image_processing"
UPSCALE_METHODS = ["lanczos", "nearest-exact", "bilinear", "area", "bicubic"]


def make_even(number):
    _, remainder = divmod(number, 2)
    return number + remainder


def make_divisible(number, divisor):
    """确保数字能被divisor整除，向上取整到最近的倍数"""
    return ((number + divisor - 1) // divisor) * divisor


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
    )
    RETURN_NAMES = (
        "width",
        "height",
        "count",
        "min_dimension",
        "max_dimension",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, image):
        return {
            "ui": {
                "width": (image.shape[2],),
                "height": (image.shape[1],),
                "count": (image.shape[0],),
                "min_dimension": (min(image.shape[2], image.shape[1]),),
                "max_dimension": (max(image.shape[2], image.shape[1]),),
            },
            "result": (
                image.shape[2],
                image.shape[1],
                image.shape[0],
                min(image.shape[2], image.shape[1]),
                max(image.shape[2], image.shape[1]),
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

    def scale_image(self, image, width, height, upscale_method, mask=None):
        
        width = width - (width % 16)
        height = height - (height % 16)

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


class ImageScalerForSDModels(BaseImageScaler):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["sd_model_type"] = (
            ["sdxl", "sd15", "sd15+", "sdxl+", "custom"],
            {
                "tooltip": "根据SD模型类型缩放图片到指定像素数，sd15为512x512，sd15+为512x768，sdxl为1024x1024，sdxl+为1280x1280"
            },
            
        )
        base_inputs["required"].update(
            {
            "custom_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 16.0,
                        "step": 0.01,
                        "tooltip": "设置自定义的像素数，如果选择custom，则使用自定义的像素数",
                    },
                ),
            }
        )
        return base_inputs

    FUNCTION = "execute"
    DESCRIPTION = """
    根据SD模型类型缩放图片到指定像素数，sd15为512x512，sd15+为512x768，sdxl为1024x1024，sdxl+为1280x1280
    """

    def execute(self, image, upscale_method, sd_model_type, custom_megapixels, mask=None):
        total_pixels = self._get_target_size(sd_model_type, custom_megapixels)
        scale_by = math.sqrt(total_pixels / (image.shape[2] * image.shape[1]))
        width = round(image.shape[2] * scale_by)
        height = round(image.shape[1] * scale_by)

        scaled_image, result_mask = self.scale_image(
            image, width, height, upscale_method, mask
        )
        return self.prepare_result(scaled_image, result_mask, width, height)

    @staticmethod
    def _get_target_size(rescale_mode, custom_megapixels):
        if rescale_mode == "custom":
            return int(custom_megapixels * 1024 * 1024)
        size_map = {
            "sd15": 512 * 512,
            "sd15+": 512 * 768,
            "sdxl": 1024 * 1024,
            "sdxl+": 1024 * 1280,
            "none": -1,
        }
        return size_map.get(rescale_mode, -1)


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
                        "default": True,
                        "tooltip": "限制缩放比例，如果图像的像素数小于目标像素数，则不缩放图像",
                    },
                ),
            }
        )

        return base_inputs

    FUNCTION = "execute"
    DESCRIPTION = """
    将图片缩放到指定像素数，total_pixels为缩放比例，limit为True时，如果图像的像素数小于目标像素数，则不缩放图像
    """

    def execute(self, image, upscale_method, total_pixels, limit=True, mask=None):
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

        scale_by = math.sqrt(total / current_pixels)
        # 计算缩放后的宽高 确保宽高为偶数
        width = make_even(round(samples.shape[3] * scale_by))
        height = make_even(round(samples.shape[2] * scale_by))

        scaled_image, result_mask = self.scale_image(
            image, width, height, upscale_method, mask
        )
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
                        "default": 16,
                        "min": 1,
                        "step": 1,
                        "max": 128,
                        "tooltip": "确保最终图像分辨率能被此数字整除，默认16",
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
    """

    def execute(self, image, size, upscale_method, shorter, limit, crop, crop_threshold, crop_position, divisible_by, mask=None):
        # Check if we should skip scaling
        min_side = min(image.shape[2], image.shape[1])
        if limit and min_side < size:
            width = make_divisible(image.shape[2], divisible_by)
            height = make_divisible(image.shape[1], divisible_by)
            # 如果尺寸发生变化，需要缩放
            if width != image.shape[2] or height != image.shape[1]:
                scaled_image, result_mask = self.scale_image(
                    image, width, height, upscale_method, mask
                )
                return self.prepare_result(scaled_image, result_mask, width, height)
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
        width = make_even(round(image.shape[2] / scale_by))
        height = make_even(round(image.shape[1] / scale_by))
        
        # 确保宽度和高度能被divisible_by整除
        width = make_divisible(width, divisible_by)
        height = make_divisible(height, divisible_by)

        # Apply cropping if enabled and needed
        if crop:
            scaled_image, result_mask = self.scale_image(
                image, width, height, upscale_method, mask
            )
            
            # Check if cropping is needed (one dimension exceeds the crop threshold)
            if (shorter and max(width, height) > crop_threshold) or (not shorter and min(width, height) > crop_threshold):
                 scaled_image, result_mask = self._crop_image(
                     scaled_image, result_mask, crop_threshold, crop_position, shorter
                )
            width, height = scaled_image.shape[2], scaled_image.shape[1]
            # 裁剪后也确保能被divisible_by整除
            width = make_divisible(width, divisible_by)
            height = make_divisible(height, divisible_by)
            # 如果尺寸发生变化，需要重新缩放
            if width != scaled_image.shape[2] or height != scaled_image.shape[1]:
                scaled_image, result_mask = self.scale_image(
                    scaled_image, width, height, upscale_method, result_mask
                )
        else:
            scaled_image, result_mask = self.scale_image(
                image, width, height, upscale_method, mask
            )

        return self.prepare_result(scaled_image, result_mask, width, height)

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
        cropped_image = image[:, y:y+crop_height, x:x+crop_width, :]
        
        if mask is not None:
            cropped_mask = mask[y:y+crop_height, x:x+crop_width]
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
    DESCRIPTION = "根据引用图片的大小和目标最大尺寸，返回缩放比例和缩放后的宽高"

    def execute(self, image, target_max_size):
        samples = image.movedim(-1, 1)
        width, height = samples.shape[3], samples.shape[2]

        rescale_ratio = target_max_size / max(width, height)

        new_width = make_even(round(width * rescale_ratio))
        new_height = make_even(round(height * rescale_ratio))

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


class ImageRotate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_from": ("IMAGE",),
                "angle": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -14096,
                        "max": 14096,
                        "step": 0.01,
                        "tooltip": "设置旋转角度，范围为-14096到14096，步长为0.01",
                    },
                ),
                "expand": ("BOOLEAN", {"default": True, "tooltip": "扩展图像尺寸"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rotated_image",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, image_from, angle, expand):
        image_np = tensor2np(image_from[0])

        height, width = image_np.shape[:2]
        center = (width / 2, height / 2)

        if expand:
            # 计算新图像的尺寸
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            new_width = int(height * abs_sin + width * abs_cos)
            new_height = int(height * abs_cos + width * abs_sin)

            # 调整旋转矩阵
            rot_mat[0, 2] += (new_width / 2) - center[0]
            rot_mat[1, 2] += (new_height / 2) - center[1]

            # 执行旋转
            rotated_image = cv2.warpAffine(
                image_np, rot_mat, (new_width, new_height), flags=cv2.INTER_CUBIC
            )
        else:
            # 不扩展图像尺寸的旋转
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(
                image_np, rot_mat, (width, height), flags=cv2.INTER_CUBIC
            )

        # 转换回tensor格式
        rotated_tensor = np2tensor(rotated_image).unsqueeze(0)

        return (rotated_tensor,)


class TrimImageBorders:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 14096,
                        "step": 1,
                        "tooltip": "设置阈值，范围为0到14096，步长为1",
                    },
                ),
                "border_color": (
                    ["black", "white"],
                    {"default": "black", "tooltip": "选择要移除的边框颜色"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    DESCRIPTION = "图片去黑边或白边"

    def run(self, image, threshold, border_color="black"):
        img = tensor2np(image[0])
        img = Image.fromarray(img)
        gray_image = img.convert("L")

        # 根据选择的边框颜色调整二值化逻辑
        if border_color == "white":
            # 对于白色边框，将高于阈值的像素设为0（黑色），低于阈值的设为255（白色）
            binary_image = gray_image.point(lambda x: 0 if x > (255 - threshold) else 255)
        else:
            # 对于黑色边框，将高于阈值的像素设为255（白色），低于阈值的设为0（黑色）
            binary_image = gray_image.point(lambda x: 255 if x > threshold else 0)
        
        bbox = binary_image.getbbox()

        if bbox:
            cropped_image = img.crop(bbox)
        else:
            cropped_image = img

        cropped_image = np2tensor(cropped_image).unsqueeze(0)
        return (cropped_image,)


class AddImageBorder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_width": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "设置边框宽度，范围为0到1000，步长为1",
                    },
                ),
                "border_ratio": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "设置边框比例，范围为0.0到1.0，步长为0.01",
                    },
                ),
                "r": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "g": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("bordered_image", "border_mask")
    FUNCTION = "add_border"
    CATEGORY = _CATEGORY
    DESCRIPTION = "给图片增加指定RGB颜色的边框,可以通过绝对像素值或相对比率设置边框宽度,并输出边框部分的mask"

    def add_border(self, image, border_width, border_ratio, r, g, b):
        # 将输入图像从 PyTorch 张量转换为 NumPy 数组
        img_np = tensor2np(image[0])

        # 获取原始图像的尺寸
        h, w, c = img_np.shape

        # 计算边框宽度
        ratio_width = int(min(h, w) * border_ratio)
        final_border_width = max(border_width, ratio_width)

        # 创建新的带边框的图像
        new_h, new_w = h + 2 * final_border_width, w + 2 * final_border_width
        bordered_img = np.full((new_h, new_w, c), [b, g, r], dtype=np.uint8)

        # 将原始图像放置在边框中央
        bordered_img[
            final_border_width : final_border_width + h,
            final_border_width : final_border_width + w,
        ] = img_np

        # 创建边框mask
        border_mask = np.ones((new_h, new_w), dtype=np.float32)
        border_mask[
            final_border_width : final_border_width + h,
            final_border_width : final_border_width + w,
        ] = 0

        # 将结果转换回 PyTorch 张量
        bordered_tensor = np2tensor(bordered_img).unsqueeze(0)
        mask_tensor = torch.from_numpy(border_mask).unsqueeze(0)

        return (bordered_tensor, mask_tensor)





class ScaleImageToSquare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "size_length": ("INT", {"default": 1024, "min": 224, "max": 10000, "step": 1}),

            "interpolation": (["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],),
            "crop_position": (["top", "bottom", "left", "right", "center", "pad"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "prep_image"

    CATEGORY = "ipadapter/utils"
    DESCRIPTION = "将图片缩放为正方形，可选择裁剪或填充方式，并输出填充区域的mask"

    def prep_image(self, image, size_length, interpolation="LANCZOS", crop_position="center", sharpening=0.0):
        size = (size_length, size_length)
        _, oh, ow, _ = image.shape
        output = image.permute([0,3,1,2])

        if crop_position == "pad":
            if oh != ow:
                if oh > ow:
                    pad = (oh - ow) // 2
                    pad = (pad, 0, pad, 0)
                elif ow > oh:
                    pad = (ow - oh) // 2
                    pad = (0, pad, 0, pad)
                output = T.functional.pad(output, pad, fill=0) # type: ignore
        else:
            crop_size = min(oh, ow)
            x = (ow-crop_size) // 2
            y = (oh-crop_size) // 2
            if "top" in crop_position:
                y = 0
            elif "bottom" in crop_position:
                y = oh-crop_size
            elif "left" in crop_position:
                x = 0
            elif "right" in crop_position:
                x = ow-crop_size

            x2 = x+crop_size
            y2 = y+crop_size

            output = output[:, :, y:y2, x:x2]

        imgs = []
        for img in output:
            img = T.ToPILImage()(img) # using PIL for better results
            img = img.resize(size, resample=Image.Resampling[interpolation])
            imgs.append(T.ToTensor()(img))
        output = torch.stack(imgs, dim=0)
        del imgs, img

        if sharpening > 0:
            output = contrast_adaptive_sharpening(output, sharpening)

        output = output.permute([0,2,3,1])
        
        # 创建mask，标记填充区域（填充区域为1，原图区域为0）
        # 默认情况下，如果不是pad模式或图像已经是正方形，mask应该全为0（表示没有填充区域）
        mask = torch.zeros((output.shape[0], size_length, size_length), dtype=torch.float32)
        
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
                mask[:, :, size_length-pad_width:] = 1.0  # 右侧填充区域
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
                mask[:, size_length-pad_height:, :] = 1.0  # 下方填充区域

        return (output, mask)
    

    
class SFLoadImage(LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()

        # 创建一个新的required字典，保留原始的image参数
        required = {
            "image": base_inputs["required"]["image"],
            "upscale_method": (UPSCALE_METHODS,),
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
        }

        return {"required": required}
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height", "min_dimension")
    CATEGORY = "sfnodes/image_processing"
    
    def load_image(self, image, upscale_method, total_pixels, limit):
        # 首先调用父类的load_image方法加载图像
        image_output, mask = super().load_image(image)
        
        # 然后使用ImageScalerByPixels进行缩放
        image_scaler_by_pixels = ImageScalerByPixels()
        result_dict = image_scaler_by_pixels.execute(image_output, upscale_method, total_pixels, limit, mask)
        

        return result_dict
            
            



   
class SFLoadImageSubfolder(LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for root, _, filenames in os.walk(input_dir):
            for f in filenames:
                full_path = os.path.join(root, f)
                if os.path.isfile(full_path):
                    rel_path = os.path.relpath(full_path, input_dir)
                    files.append(rel_path)
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
        
    CATEGORY = _CATEGORY
    
    def load_image(self, image):
        # 调用父类的load_image方法加载图像
        image_output, mask = super().load_image(image)
        
        # 提取文件名（不含路径和扩展名）
        filename = os.path.splitext(os.path.basename(image))[0]
        
        return (image_output, mask, filename)


class SFLoadImageSubfolderSortedByMtime(LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for root, _, filenames in os.walk(input_dir):
            for f in filenames:
                full_path = os.path.join(root, f)
                if os.path.isfile(full_path):
                    rel_path = os.path.relpath(full_path, input_dir)
                    files.append(rel_path)
        
        # 过滤图片文件
        files = folder_paths.filter_files_content_types(files, ["image"])
        
        # 按修改时间从新到旧排序（降序：最新的在前）
        files_with_mtime = []
        for rel_path in files:
            full_path = os.path.join(input_dir, rel_path)
            mtime = os.path.getmtime(full_path)
            files_with_mtime.append((rel_path, mtime))
        
        files_with_mtime.sort(key=lambda x: x[1], reverse=True)
        files = [rel_path for rel_path, _ in files_with_mtime]
        
        return {"required":
                    {"image": (files, {"image_upload": True})},
                }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
        
    CATEGORY = _CATEGORY
    
    def load_image(self, image):
        # 调用父类的load_image方法加载图像
        image_output, mask = super().load_image(image)
        
        # 提取文件名（不含路径和扩展名）
        filename = os.path.splitext(os.path.basename(image))[0]
        
        return (image_output, mask, filename)
    
            
   
class ImageResizePlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = oh

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]
        
        outputs = torch.clamp(outputs, 0, 1)

        return(outputs, outputs.shape[2], outputs.shape[1],)



class ApexSmartResize:
    """
    Apex Smart Resize - Automatically snaps to closest compatible resolution
    Intelligent resolution detection and scaling with proportion preservation
    """
    
    def __init__(self):
        # Define compatible resolutions for different AI models
        self.resolution_sets = {
            "Standard": [
                (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
                (1344, 768), (768, 1344), (1536, 640), (640, 1536), 
                (832, 1280), (1280, 832), (704, 1504), (1504, 704),
                (896, 1344), (1344, 896), (960, 1280), (1280, 960),
                (512, 512), (768, 768), (640, 640)
            ],
            "Extended": [
                (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
                (1344, 768), (768, 1344), (1536, 640), (640, 1536), (1728, 576),
                (576, 1728), (1920, 512), (512, 1920), (2048, 512), (512, 2048),
                (832, 1280), (1280, 832), (704, 1504), (1504, 704),
                (960, 1536), (1536, 960), (1088, 1472), (1472, 1088)
            ],
            "Flux": [
                (1024, 1024), (768, 1344), (832, 1216), (896, 1152), (1152, 896),
                (1216, 832), (1344, 768), (512, 512), (640, 1536), (1536, 640),
                (704, 1504), (1504, 704), (832, 1280), (1280, 832)
            ],
            "Portrait": [
                (832, 1216), (768, 1344), (640, 1536), (896, 1152), 
                (832, 1280), (704, 1504), (512, 768), (576, 1024),
                (640, 960), (720, 1280), (768, 1024), (896, 1344)
            ],
            "Landscape": [
                (1216, 832), (1344, 768), (1536, 640), (1152, 896),
                (1280, 832), (1504, 704), (768, 512), (1024, 576),
                (960, 640), (1280, 720), (1024, 768), (1344, 896)
            ],
            "Square": [
                (512, 512), (640, 640), (768, 768), (832, 832), (896, 896),
                (1024, 1024), (1152, 1152), (1216, 1216), (1280, 1280), (1344, 1344)
            ]
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution_set": ([
                    "Standard",      # Core SDXL/Flux resolutions
                    "Extended",      # Extra experimental sizes  
                    "Flux",          # Flux-optimized
                    "Portrait",      # Tall formats
                    "Landscape",     # Wide formats
                    "Square"         # Square only
                ], {"default": "Standard"}),
                "snap_method": ([
                    "keep_proportion",   # Scale largest side first, maintain aspect ratio
                    "closest_area",      # Snap to closest total pixel count
                    "closest_ratio",     # Snap to closest aspect ratio
                    "prefer_larger",     # Prefer larger resolutions
                    "prefer_smaller",    # Prefer smaller resolutions
                ], {"default": "keep_proportion"}),
                "resize_mode": ([
                    "crop_center",       # Crop from center
                    "stretch",           # Stretch to exact dimensions
                    "fit_pad_black",     # Fit with black padding
                    "fit_pad_white",     # Fit with white padding  
                    "fit_pad_edge"       # Fit with edge extension
                ], {"default": "crop_center"}),
                "interpolation": ([
                    "lanczos",
                    "bicubic",
                    "bilinear", 
                    "nearest"
                ], {"default": "lanczos"}),
                "show_candidates": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show resolution candidates in console"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("image", "width", "height", "scale_factor", "resolution_info", "console_log")
    FUNCTION = "smart_resize"
    CATEGORY = _CATEGORY

    def smart_resize(self, image, resolution_set, snap_method, resize_mode, interpolation, show_candidates):
        
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
                orig_w, orig_h, target_w, target_h, scale_factor, resize_mode, 
                resolution_set, snap_method, candidates_info, start_time
            )
            
            # Resize the image
            resized_image = self._apply_resize(image, target_w, target_h, resize_mode, interpolation)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            console_data["processing_time_seconds"] = round(processing_time, 3)
            
            # Format console output
            console_output = json.dumps(console_data, indent=2)
            
            return (resized_image, target_w, target_h, scale_factor, info, console_output)
            
        except Exception as e:
            error_console = json.dumps({
                "status": "error",
                "message": str(e),
                "original_size": f"{orig_w}x{orig_h}",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
            
            return (image, orig_w, orig_h, 1.0, f"Error: {str(e)}", error_console)
    
    def _create_console_data(self, orig_w, orig_h, target_w, target_h, scale_factor, resize_mode, 
                           resolution_set, snap_method, candidates_info, start_time):
        """Create structured data for Apex Console"""
        
        orig_area = orig_w * orig_h
        target_area = target_w * target_h
        memory_change_mb = ((target_area - orig_area) * 4 * 3) / (1024 * 1024)  # Assume RGB float32
        
        return {
            "action": "Smart Resize Complete",
            "status": "success",
            "timestamp": start_time.isoformat(),
            "input": {
                "size": f"{orig_w}x{orig_h}",
                "aspect_ratio": round(orig_w / orig_h, 3),
                "total_pixels": f"{orig_area:,}",
                "estimated_memory_mb": round((orig_area * 4 * 3) / (1024 * 1024), 1)
            },
            "output": {
                "size": f"{target_w}x{target_h}",
                "aspect_ratio": round(target_w / target_h, 3),
                "total_pixels": f"{target_area:,}",
                "estimated_memory_mb": round((target_area * 4 * 3) / (1024 * 1024), 1)
            },
            "processing": {
                "resolution_set": resolution_set,
                "snap_method": snap_method,
                "resize_mode": resize_mode,
                "scale_factor": round(scale_factor, 3),
                "size_change_percent": round(((scale_factor * scale_factor - 1) * 100), 1),
                "memory_change_mb": round(memory_change_mb, 1)
            },
            "candidates": candidates_info
        }
    
    def _find_best_resolution(self, orig_w, orig_h, resolution_set, snap_method, show_candidates):
        """Find the best target resolution based on method"""
        
        resolutions = self.resolution_sets[resolution_set]
        orig_area = orig_w * orig_h
        orig_aspect = orig_w / orig_h
        
        if snap_method == "keep_proportion":
            target_w, target_h, info, candidates = self._keep_proportion_snap(orig_w, orig_h, resolutions, show_candidates)
            return target_w, target_h, info, candidates
        
        # Other methods
        candidates = []
        
        for w, h in resolutions:
            area = w * h
            aspect = w / h
            scale_factor = math.sqrt(area / orig_area)
            aspect_diff = abs(aspect - orig_aspect)
            area_diff = abs(area - orig_area)
            
            candidates.append({
                'resolution': f"{w}x{h}",
                'scale_factor': round(scale_factor, 3),
                'aspect_ratio': round(aspect, 3),
                'aspect_diff': round(aspect_diff, 3),
                'area_diff': area_diff,
                'total_pixels': f"{area:,}"
            })
        
        # Sort candidates based on method
        if snap_method == "closest_area":
            candidates.sort(key=lambda x: x['area_diff'])
            best = candidates[0]
            info = f"Closest area match from {resolution_set}"
            
        elif snap_method == "closest_ratio":
            candidates.sort(key=lambda x: x['aspect_diff'])
            best = candidates[0]
            info = f"Closest aspect ratio from {resolution_set}"
            
        elif snap_method == "prefer_larger":
            larger_candidates = [c for c in candidates if c['area_diff'] >= 0]
            if larger_candidates:
                larger_candidates.sort(key=lambda x: x['area_diff'])
                best = larger_candidates[0]
            else:
                candidates.sort(key=lambda x: x['area_diff'], reverse=True)
                best = candidates[0]
            info = f"Prefer larger from {resolution_set}"
            
        else:  # prefer_smaller
            smaller_candidates = [c for c in candidates if c['area_diff'] <= 0]
            if smaller_candidates:
                smaller_candidates.sort(key=lambda x: x['area_diff'], reverse=True)
                best = smaller_candidates[0]
            else:
                candidates.sort(key=lambda x: x['area_diff'])
                best = candidates[0]
            info = f"Prefer smaller from {resolution_set}"
        
        # Extract target dimensions
        w, h = map(int, best['resolution'].split('x'))
        candidates_info = {
            "method": snap_method,
            "total_evaluated": len(candidates),
            "top_5": sorted(candidates, key=lambda x: x['area_diff'])[:5]
        }
        
        return w, h, info, candidates_info
    
    def _keep_proportion_snap(self, orig_w, orig_h, resolutions, show_candidates):
        """Scale by largest dimension while maintaining aspect ratio"""
        
        orig_aspect = orig_w / orig_h
        is_portrait = orig_h > orig_w
        
        best_match = None
        best_score = float('inf')
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
                        
                        candidates.append({
                            'resolution': f"{target_w}x{target_h}",
                            'scale_factor': round(scale_factor, 3),
                            'aspect_diff': round(aspect_diff, 3),
                            'score': round(score, 3)
                        })
                        
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
                        
                        candidates.append({
                            'resolution': f"{target_w}x{target_h}",
                            'scale_factor': round(scale_factor, 3),
                            'aspect_diff': round(aspect_diff, 3),
                            'score': round(score, 3)
                        })
                        
                        if score < best_score:
                            best_score = score
                            best_match = (target_w, target_h)
        
        # Fallback to closest aspect ratio if no good match
        if best_match is None:
            best_aspect_diff = float('inf')
            for w, h in resolutions:
                aspect_diff = abs((w/h) - orig_aspect)
                if aspect_diff < best_aspect_diff:
                    best_aspect_diff = aspect_diff
                    best_match = (w, h)
        
        target_w, target_h = best_match
        info = f"Keep proportion snap from {len(resolutions)} resolutions"
        
        candidates_info = {
            "method": "keep_proportion",
            "orientation": "portrait" if orig_h > orig_w else "landscape",
            "total_evaluated": len(candidates),
            "top_5": sorted(candidates, key=lambda x: x['score'])[:5] if candidates else []
        }
        
        return target_w, target_h, info, candidates_info
    
    def _apply_resize(self, image, target_w, target_h, resize_mode, interpolation):
        """Apply the actual resizing with specified method"""
        
        if resize_mode == "stretch":
            return self._resize_tensor(image, target_w, target_h, interpolation)
        
        elif resize_mode == "crop_center":
            return self._crop_center_resize(image, target_w, target_h, interpolation)
        
        elif resize_mode == "fit_pad_black":
            return self._fit_pad_resize(image, target_w, target_h, interpolation, pad_color=0.0)
        
        elif resize_mode == "fit_pad_white":
            return self._fit_pad_resize(image, target_w, target_h, interpolation, pad_color=1.0)
        
        elif resize_mode == "fit_pad_edge":
            return self._fit_pad_edge_resize(image, target_w, target_h, interpolation)
        
        else:
            return self._resize_tensor(image, target_w, target_h, interpolation)
    
    def _resize_tensor(self, image, width, height, interpolation):
        """Core tensor resize function"""
        
        image_bchw = image.permute(0, 3, 1, 2)
        
        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear", 
            "bicubic": "bicubic",
            "lanczos": "bicubic"  # PyTorch fallback
        }
        
        mode = mode_map.get(interpolation, "bicubic")
        antialias = mode in ["bilinear", "bicubic"]
        
        resized = F.interpolate(image_bchw, size=(height, width), 
                              mode=mode, antialias=antialias)
        
        return resized.permute(0, 2, 3, 1)
    
    def _crop_center_resize(self, image, target_w, target_h, interpolation):
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
        
        # Center crop
        crop_x = max(0, (new_w - target_w) // 2)
        crop_y = max(0, (new_h - target_h) // 2)
        
        cropped = resized[:, crop_y:crop_y+target_h, crop_x:crop_x+target_w, :]
        
        return cropped
    
    def _fit_pad_resize(self, image, target_w, target_h, interpolation, pad_color):
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
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        if pad_w > 0 or pad_h > 0:
            image_bchw = resized.permute(0, 3, 1, 2)
            padded = F.pad(image_bchw, (pad_left, pad_right, pad_top, pad_bottom), 
                          mode='constant', value=pad_color)
            result = padded.permute(0, 2, 3, 1)
        else:
            result = resized
        
        return result
    
    def _fit_pad_edge_resize(self, image, target_w, target_h, interpolation):
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
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        if pad_w > 0 or pad_h > 0:
            image_bchw = resized.permute(0, 3, 1, 2)
            padded = F.pad(image_bchw, (pad_left, pad_right, pad_top, pad_bottom), 
                          mode='replicate')
            result = padded.permute(0, 2, 3, 1)
        else:
            result = resized
        
        return result

# Remove the old NODE_CLASS_MAPPINGS from here since it's now in __init__.py