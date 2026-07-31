import torch
from PIL import Image

from ...sf_utils.cutpaste import (
    calc_target_dimensions,
    get_target_size,
    mask_bbox,
    resize_tensor,
)
from ...sf_utils.image_convert import (
    mask2pil,
    pil2mask,
    pil2tensor,
    tensor2pil,
)
from ...sf_utils.mask_utils import mask_process

_CATEGORY = "sfnodes/inpaint"


def _build_inputs():
    return {
        "required": {
            "image": ("IMAGE",),
            "mode": (
                ["auto", "mask", "face"],
                {
                    "default": "auto",
                    "tooltip": "auto: 有非零遮罩时用遮罩模式，否则用人脸检测模式; mask: 使用遮罩边界; face: 使用人脸检测",
                },
            ),
            "padding": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "设置图像的填充像素数",
                },
            ),
            "padding_percent": (
                "FLOAT",
                {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "设置图像的填充百分比",
                },
            ),
            "rescale_mode": (
                ["sdxl", "sd15", "sdxl+", "sd15+", "none", "custom"],
                {
                    "default": "sdxl",
                    "tooltip": "选择缩放模式，sdxl: 缩放到1024x1024像素; sd15: 缩放到512x512像素; sdxl+: 缩放到1024x1280像素; sd15+: 缩放到512x768像素; none: 不缩放; custom: 使用自定义的像素数",
                },
            ),
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
            "force_square": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "如果开启，将扩展短边成为正方形裁剪区域",
                },
            ),
            "face_index": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "指定要使用的人脸索引，从0开始，仅在face模式下生效",
                },
            ),
        },
        "optional": {
            "mask": ("MASK",),
            "analysis_models": ("ANALYSIS_MODELS",),
            "mask_params": ("MASKPARAMS",),
        },
    }


class SFCutout:
    @classmethod
    def INPUT_TYPES(cls):
        return _build_inputs()

    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "根据遮罩或人脸检测裁剪图像，缩放并生成用于修复的裁剪信息，"
        "支持遮罩/人脸双模式，遮罩后处理由 SFMaskParams 控制"
    )

    RETURN_TYPES = ("CUTINFO", "IMAGE", "MASK")
    RETURN_NAMES = (
        "cutinfo",
        "cutout_image",
        "cutout_mask",
    )

    FUNCTION = "cutout"

    def cutout(
        self,
        image,
        mode,
        padding,
        padding_percent,
        rescale_mode,
        custom_megapixels,
        force_square=False,
        face_index=0,
        mask=None,
        analysis_models=None,
        mask_params=None,
    ):
        if mode == "auto":
            if mask is not None and torch.any(mask > 0):
                mode = "mask"
            elif analysis_models is not None:
                mode = "face"
            else:
                raise Exception(
                    "未提供非零 mask 或 analysis_models，无法确定裁剪区域"
                )

        img = image[0]
        pil_image = tensor2pil(img)

        if mode == "mask":
            x, y, width, height, origin_crop_pil, region_mask_pil = (
                self._cutout_from_mask(
                    pil_image, mask, padding, padding_percent, force_square
                )
            )
        else:
            x, y, width, height, origin_crop_pil = self._cutout_from_face(
                pil_image,
                analysis_models,
                padding,
                padding_percent,
                face_index,
                force_square,
            )
            region_mask_pil = Image.new("L", (width, height), 255)

        target_size = get_target_size(rescale_mode, custom_megapixels)
        if target_size > 0:
            new_width, new_height = calc_target_dimensions(
                width, height, target_size
            )
            cutout_pil = origin_crop_pil.resize(
                (new_width, new_height), resample=Image.Resampling.LANCZOS
            )
            region_mask_pil = region_mask_pil.resize(
                (new_width, new_height), resample=Image.Resampling.LANCZOS
            )
        else:
            new_width, new_height = width, height
            cutout_pil = origin_crop_pil

        cutout_image = pil2tensor(cutout_pil)
        cutout_mask = pil2mask(region_mask_pil)
        if mask_params is not None:
            cutout_mask = mask_process(cutout_mask, mask_params, unqueeze=False)

        cutinfo = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "mask": cutout_mask,
            "original_image": image,
            "cutout_image": cutout_image,
            "origin_face": pil2tensor(origin_crop_pil),
            "new_width": new_width,
            "new_height": new_height,
        }

        return (
            cutinfo,
            cutout_image,
            cutout_mask,
        )

    @staticmethod
    def _cutout_from_mask(pil_image, mask, padding, padding_percent, force_square):
        mask_image = mask2pil(mask)
        bbox = mask_bbox(mask_image)
        if bbox is None:
            raise Exception("Mask没有非零区域，无法裁剪图像")

        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        padding_x = int(width * padding_percent) + padding
        padding_y = int(height * padding_percent) + padding

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(pil_image.width, x_max + padding_x)
        y_max = min(pil_image.height, y_max + padding_y)

        width = x_max - x_min
        height = y_max - y_min

        if force_square:
            x_min, y_min, width, height = SFCutout._expand_to_square(
                x_min, y_min, width, height, pil_image.width, pil_image.height
            )
            x_max = x_min + width
            y_max = y_min + height

        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
        cropped_mask = mask_image.crop((x_min, y_min, x_max, y_max))

        return x_min, y_min, width, height, cropped_image, cropped_mask

    @staticmethod
    def _cutout_from_face(
        pil_image,
        analysis_models,
        padding,
        padding_percent,
        face_index,
        force_square,
    ):
        if analysis_models is None:
            raise Exception("face 模式需要提供 analysis_models")
        face, x, y, width, height = analysis_models.get_single_bbox(
            pil_image, padding, padding_percent, face_index
        )
        if face is None:
            raise Exception("未在图像中检测到人脸。")
        if force_square:
            x, y, width, height = SFCutout._expand_to_square(
                x, y, width, height, pil_image.width, pil_image.height
            )
        return x, y, width, height, pil_image.crop((x, y, x + width, y + height))

    @staticmethod
    def _expand_to_square(x, y, width, height, img_w, img_h):
        square_size = max(width, height)
        if square_size > img_w or square_size > img_h:
            return x, y, width, height
        center_x = x + width // 2
        center_y = y + height // 2
        new_x = max(0, center_x - square_size // 2)
        new_y = max(0, center_y - square_size // 2)
        if new_x + square_size > img_w:
            new_x = img_w - square_size
        if new_y + square_size > img_h:
            new_y = img_h - square_size
        return new_x, new_y, square_size, square_size


class SFPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cutinfo": ("CUTINFO",),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "destination_image": ("IMAGE",),
                "upscale_method": (
                    ["lanczos", "bilinear", "bicubic", "nearest"],
                    {"default": "lanczos", "tooltip": "设置图像缩放的方法"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "paste"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将裁剪信息中的图像贴回原图，支持自定义源图和目标图"

    def paste(
        self,
        cutinfo,
        source_image=None,
        destination_image=None,
        upscale_method="lanczos",
    ):
        if source_image is None:
            source_image = cutinfo["cutout_image"]
        if destination_image is None:
            destination_image = cutinfo["original_image"]

        x = cutinfo["x"]
        y = cutinfo["y"]
        width = cutinfo["width"]
        height = cutinfo["height"]
        mask = cutinfo["mask"]

        source = resize_tensor(source_image, width, height, upscale_method)
        mask = resize_tensor(mask, width, height, upscale_method)

        source_pil = tensor2pil(source)
        destination_pil = tensor2pil(destination_image)
        mask_pil = mask2pil(mask)

        destination_pil.paste(source_pil, (x, y), mask_pil)

        return pil2tensor(destination_pil), pil2mask(mask_pil)


class SFExtractCutInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cutinfo": ("CUTINFO",),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "x",
        "y",
        "width",
        "height",
        "mask",
        "original_image",
        "cutout_image",
        "origin_face",
        "new_width",
        "new_height",
    )
    INPUT_IS_LIST = (True,)
    CATEGORY = _CATEGORY
    FUNCTION = "extract"
    DESCRIPTION = "从裁剪信息中提取坐标、尺寸、遮罩和图像"

    def extract(self, cutinfo):
        if not isinstance(cutinfo, list) or len(cutinfo) <= 0:
            raise Exception(f"裁剪信息不是预期的列表格式: {type(cutinfo)}")

        if len(cutinfo) > 0:
            cutinfo = cutinfo[0]

        return (
            cutinfo.get("x", 0),
            cutinfo.get("y", 0),
            cutinfo.get("width", 0),
            cutinfo.get("height", 0),
            cutinfo.get("mask", None),
            cutinfo.get("original_image", None),
            cutinfo.get("cutout_image", None),
            cutinfo.get("origin_face", None),
            cutinfo.get("new_width", 0),
            cutinfo.get("new_height", 0),
        )
