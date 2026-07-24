import cv2
import numpy as np
import torch

from PIL import Image
from ...sf_utils.image_convert import np2tensor, tensor2np

_CATEGORY = "sfnodes/image"


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
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            new_width = int(height * abs_sin + width * abs_cos)
            new_height = int(height * abs_cos + width * abs_sin)

            rot_mat[0, 2] += (new_width / 2) - center[0]
            rot_mat[1, 2] += (new_height / 2) - center[1]

            rotated_image = cv2.warpAffine(
                image_np, rot_mat, (new_width, new_height), flags=cv2.INTER_CUBIC
            )
        else:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(
                image_np, rot_mat, (width, height), flags=cv2.INTER_CUBIC
            )

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

        if border_color == "white":
            binary_image = gray_image.point(
                lambda x: 0 if x > (255 - threshold) else 255
            )
        else:
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
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("bordered_image", "border_mask")
    FUNCTION = "add_border"
    CATEGORY = _CATEGORY
    DESCRIPTION = "给图片增加指定RGB颜色的边框,可以通过绝对像素值或相对比率设置边框宽度,并输出边框部分的mask"

    def add_border(self, image, border_width, border_ratio, r, g, b):
        img_np = tensor2np(image[0])

        h, w, c = img_np.shape

        ratio_width = int(min(h, w) * border_ratio)
        final_border_width = max(border_width, ratio_width)

        new_h, new_w = h + 2 * final_border_width, w + 2 * final_border_width
        bordered_img = np.full((new_h, new_w, c), [b, g, r], dtype=np.uint8)

        bordered_img[
            final_border_width : final_border_width + h,
            final_border_width : final_border_width + w,
        ] = img_np

        border_mask = np.ones((new_h, new_w), dtype=np.float32)
        border_mask[
            final_border_width : final_border_width + h,
            final_border_width : final_border_width + w,
        ] = 0

        bordered_tensor = np2tensor(bordered_img).unsqueeze(0)
        mask_tensor = torch.from_numpy(border_mask).unsqueeze(0)

        return (bordered_tensor, mask_tensor)
