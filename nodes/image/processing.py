import cv2
import numpy as np
import torch
import comfy.utils
import comfy.model_management
import kornia

from PIL import Image, ImageEnhance
from comfy_extras.nodes_post_processing import Blend, Blur, Quantize
from ...sf_utils.image_convert import image_posterize

_CATEGORY = "sfnodes/image"


class ColorAdjustment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": -100,
                        "max": 100,
                        "step": 5,
                        "tooltip": "设置温度值，范围为-100到100，步长为5",
                    },
                ),
                "hue": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": -90,
                        "max": 90,
                        "step": 5,
                        "tooltip": "设置色调值，范围为-90到90，步长为5",
                    },
                ),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": -100,
                        "max": 100,
                        "step": 5,
                        "tooltip": "设置亮度值，范围为-100到100，步长为5",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": -100,
                        "max": 100,
                        "step": 5,
                        "tooltip": "设置对比度值，范围为-100到100，步长为5",
                    },
                ),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": -100,
                        "max": 100,
                        "step": 5,
                        "tooltip": "设置饱和度值，范围为-100到100，步长为5",
                    },
                ),
                "gamma": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": 0.2,
                        "max": 2.2,
                        "step": 0.1,
                        "tooltip": "设置伽马值，范围为0.2到2.2，步长为0.1",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "对图片进行色彩校正"

    def execute(
        self,
        image: torch.Tensor,
        temperature: float,
        hue: float,
        brightness: float,
        contrast: float,
        saturation: float,
        gamma: float,
    ):
        batch_size, _, _, _ = image.shape
        result = torch.zeros_like(image)

        brightness /= 100
        contrast /= 100
        saturation /= 100
        temperature /= 100

        brightness = 1 + brightness
        contrast = 1 + contrast
        saturation = 1 + saturation

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))

            # brightness
            modified_image = ImageEnhance.Brightness(modified_image).enhance(brightness)

            # contrast
            modified_image = ImageEnhance.Contrast(modified_image).enhance(contrast)
            modified_image = np.array(modified_image).astype(np.float32)

            # temperature
            if temperature > 0:
                modified_image[:, :, 0] *= 1 + temperature
                modified_image[:, :, 1] *= 1 + temperature * 0.4
            elif temperature < 0:
                modified_image[:, :, 2] *= 1 - temperature
            modified_image = np.clip(modified_image, 0, 255) / 255

            # gamma
            modified_image = np.clip(np.power(modified_image, gamma), 0, 1)

            # saturation
            hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
            hls_img[:, :, 2] = np.clip(saturation * hls_img[:, :, 2], 0, 1)
            modified_image = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB) * 255

            # hue
            hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue) % 360
            modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

            modified_image = modified_image.astype(np.uint8)
            modified_image = modified_image / 255
            modified_image = torch.from_numpy(modified_image).unsqueeze(0)
            result[b] = modified_image

        return (result,)


class ColorTint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "设置强度值，范围为0.1到1.0，步长为0.1",
                    },
                ),
                "mode": (
                    [
                        "sepia",
                        "red",
                        "green",
                        "blue",
                        "cyan",
                        "magenta",
                        "yellow",
                        "purple",
                        "orange",
                        "warm",
                        "cool",
                        "lime",
                        "navy",
                        "vintage",
                        "rose",
                        "teal",
                        "maroon",
                        "peach",
                        "lavender",
                        "olive",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "图片颜色滤镜"

    def execute(self, image: torch.Tensor, strength: float, mode: str = "sepia"):
        if strength == 0:
            return (image,)

        sepia_weights = (
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 1, 3).to(image.device)
        )

        mode_filters = {
            "sepia": torch.tensor([1.0, 0.8, 0.6]),
            "red": torch.tensor([1.0, 0.6, 0.6]),
            "green": torch.tensor([0.6, 1.0, 0.6]),
            "blue": torch.tensor([0.6, 0.8, 1.0]),
            "cyan": torch.tensor([0.6, 1.0, 1.0]),
            "magenta": torch.tensor([1.0, 0.6, 1.0]),
            "yellow": torch.tensor([1.0, 1.0, 0.6]),
            "purple": torch.tensor([0.8, 0.6, 1.0]),
            "orange": torch.tensor([1.0, 0.7, 0.3]),
            "warm": torch.tensor([1.0, 0.9, 0.7]),
            "cool": torch.tensor([0.7, 0.9, 1.0]),
            "lime": torch.tensor([0.7, 1.0, 0.3]),
            "navy": torch.tensor([0.3, 0.4, 0.7]),
            "vintage": torch.tensor([0.9, 0.85, 0.7]),
            "rose": torch.tensor([1.0, 0.8, 0.9]),
            "teal": torch.tensor([0.3, 0.8, 0.8]),
            "maroon": torch.tensor([0.7, 0.3, 0.5]),
            "peach": torch.tensor([1.0, 0.8, 0.6]),
            "lavender": torch.tensor([0.8, 0.6, 1.0]),
            "olive": torch.tensor([0.6, 0.7, 0.4]),
        }

        scale_filter = mode_filters[mode].view(1, 1, 1, 3).to(image.device)

        grayscale = torch.sum(image * sepia_weights, dim=-1, keepdim=True)
        tinted = grayscale * scale_filter

        result = tinted * strength + image * (1 - strength)
        return (result,)


class ColorBlockEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "设置强度值，范围为1到100，步长为1",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "图片色块化"

    def execute(self, image: torch.Tensor, strength: int):
        color_adjustment = ColorAdjustment()
        blur = Blur()
        quantize_node = Quantize()
        blender = Blend()

        blurred_image = blur.blur(image, blur_radius=strength, sigma=1.0)
        blurred_image = torch.cat(blurred_image, dim=1)

        quantized_image = quantize_node.quantize(
            blurred_image, colors=5, dither="bayer-2"
        )
        quantized_image = torch.cat(quantized_image, dim=1)

        color_adjusted_image = color_adjustment.execute(
            quantized_image,
            temperature=0,
            hue=0,
            brightness=5,
            contrast=0,
            saturation=-100,
            gamma=1,
        )
        color_adjusted_image = torch.cat(color_adjusted_image, dim=1)

        blender_image = blender.blend_images(
            color_adjusted_image, image, blend_factor=1, blend_mode="overlay"
        )
        blender_image = torch.cat(blender_image, dim=1)

        flat_image = color_adjustment.execute(
            blender_image,
            temperature=0,
            hue=0,
            brightness=5,
            contrast=5,
            saturation=50,
            gamma=1.2,
        )
        flat_image = torch.cat(flat_image, dim=1)
        return (flat_image,)


class FlatteningEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "high_threshold": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "设置高阈值，范围为0.01到10.0，步长为0.01",
                    },
                ),
                "mid_threshold": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "设置中阈值，范围为0.01到10.0，步长为0.01",
                    },
                ),
                "low_threshold": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "设置低阈值，范围为0.01到10.0，步长为0.01",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "图片平面化"

    def execute(
        self,
        image: torch.Tensor,
        high_threshold: float,
        mid_threshold: float,
        low_threshold: float,
    ):
        color_adjustment = ColorAdjustment()
        blender = Blend()

        color_adjusted_image = color_adjustment.execute(
            image,
            temperature=0,
            hue=0,
            brightness=-5,
            contrast=10,
            saturation=65,
            gamma=1.3,
        )
        color_adjusted_image = torch.cat(color_adjusted_image, dim=1)

        posterized_image1 = image_posterize(
            color_adjusted_image, threshold=high_threshold
        )
        posterized_image2 = image_posterize(
            color_adjusted_image, threshold=mid_threshold
        )
        posterized_image3 = image_posterize(
            color_adjusted_image, threshold=low_threshold
        )

        blender_image1 = blender.blend_images(
            posterized_image1, posterized_image2, blend_factor=0.5, blend_mode="screen"
        )
        blender_image1 = torch.cat(blender_image1, dim=1)
        blender_image2 = blender.blend_images(
            blender_image1, posterized_image3, blend_factor=0.5, blend_mode="screen"
        )
        blender_image2 = torch.cat(blender_image2, dim=1)

        flat_image = blender.blend_images(
            blender_image2,
            color_adjusted_image,
            blend_factor=1,
            blend_mode="soft_light",
        )
        flat_image = torch.cat(flat_image, dim=1)
        flat_image = color_adjustment.execute(
            flat_image,
            temperature=0,
            hue=0,
            brightness=-20,
            contrast=45,
            saturation=25,
            gamma=1.0,
        )
        flat_image = torch.cat(flat_image, dim=1)
        return (flat_image,)


class ImageColorMatch:
    DESCRIPTION = "将目标图像的颜色统计分布匹配到参考图像，支持多种色彩空间和遮罩控制"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "目标图像（将被修改颜色分布）"}),
                "reference": ("IMAGE", {"tooltip": "参考图像（提供目标颜色分布）"}),
                "color_space": (
                    ["LAB", "Linear RGB", "YCbCr", "RGB", "LUV", "YUV", "XYZ"],
                    {"tooltip": "用于统计匹配的色彩空间，LAB 最常用（L=光照, a/b=色彩）；Linear RGB 为物理线性空间，光照迁移更准确"},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "色彩迁移强度，0=完全保留原图，1=完全匹配，>1=过度拉伸（创意效果）",
                    },
                ),
                "device": (
                    ["auto", "cpu", "gpu"],
                    {"tooltip": "计算设备，auto=自动选择"},
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "批处理大小，0=一次处理全部帧",
                    },
                ),
            },
            "optional": {
                "reference_mask": ("MASK", {"tooltip": "参考图的遮罩，仅统计遮罩区域的色彩分布"}),
                "target_mask": ("MASK", {"tooltip": "目标图的遮罩，仅对遮罩区域应用色彩迁移"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(
        self,
        image,
        reference,
        color_space,
        strength,
        device,
        batch_size,
        reference_mask=None,
        target_mask=None,
    ):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = "cpu"

        image = image.permute([0, 3, 1, 2])
        reference = reference.permute([0, 3, 1, 2]).to(device)

        # Ensure reference_mask is in the correct format and on the right device
        if reference_mask is not None:
            assert reference_mask.ndim == 3, (
                f"Expected reference_mask to have 3 dimensions, but got {reference_mask.ndim}"
            )
            assert reference_mask.shape[0] == reference.shape[0], (
                f"Frame count mismatch: reference_mask has {reference_mask.shape[0]} frames, but reference has {reference.shape[0]}"
            )

            reference_mask = reference_mask.unsqueeze(1).to(device)
            reference_mask = (reference_mask > 0.5).float()

            if reference_mask.shape[2:] != reference.shape[2:]:
                reference_mask = comfy.utils.common_upscale(
                    reference_mask,
                    reference.shape[3],
                    reference.shape[2],
                    upscale_method="bicubic",
                    crop="center",
                )

        # target_mask: 保持原始值以实现软混合（不二值化）
        if target_mask is not None:
            target_mask = target_mask.unsqueeze(1).to(device)

            # 匹配 spatial 维度
            if target_mask.shape[2:] != image.shape[2:]:
                target_mask = comfy.utils.common_upscale(
                    target_mask,
                    image.shape[3],
                    image.shape[2],
                    upscale_method="bicubic",
                    crop="center",
                )

            # 匹配 batch 维度：不足时重复最后一帧，超出时截断
            if target_mask.shape[0] < image.shape[0]:
                repeats = image.shape[0] - target_mask.shape[0]
                target_mask = torch.cat([target_mask, target_mask[-1:].repeat(repeats, 1, 1, 1)], dim=0)
            elif target_mask.shape[0] > image.shape[0]:
                target_mask = target_mask[:image.shape[0]]

        if batch_size == 0 or batch_size > image.shape[0]:
            batch_size = image.shape[0]

        if "LAB" == color_space:
            reference = kornia.color.rgb_to_lab(reference)
        elif "Linear RGB" == color_space:
            reference = kornia.color.rgb_to_linear_rgb(reference)
        elif "YCbCr" == color_space:
            reference = kornia.color.rgb_to_ycbcr(reference)
        elif "LUV" == color_space:
            reference = kornia.color.rgb_to_luv(reference)
        elif "YUV" == color_space:
            reference = kornia.color.rgb_to_yuv(reference)
        elif "XYZ" == color_space:
            reference = kornia.color.rgb_to_xyz(reference)

        reference_mean, reference_std = self.compute_mean_std(reference, reference_mask)

        # 多帧参考图时聚合为单帧统计，防止与 image batch 维度不匹配
        if reference_mean.shape[0] > 1:
            reference_mean = reference_mean.mean(dim=0, keepdim=True)
            reference_std = reference_std.mean(dim=0, keepdim=True)

        image_batch = torch.split(image, batch_size, dim=0)
        output = []

        offset = 0
        for image in image_batch:
            cur_batch = image.shape[0]
            image = image.to(device)

            if color_space == "LAB":
                image = kornia.color.rgb_to_lab(image)
            elif color_space == "Linear RGB":
                image = kornia.color.rgb_to_linear_rgb(image)
            elif color_space == "YCbCr":
                image = kornia.color.rgb_to_ycbcr(image)
            elif color_space == "LUV":
                image = kornia.color.rgb_to_luv(image)
            elif color_space == "YUV":
                image = kornia.color.rgb_to_yuv(image)
            elif color_space == "XYZ":
                image = kornia.color.rgb_to_xyz(image)

            image_mean, image_std = self.compute_mean_std(image)

            matched = (
                torch.nan_to_num((image - image_mean) / image_std)
                * torch.nan_to_num(reference_std)
                + reference_mean
            )
            matched = strength * matched + (1.0 - strength) * image

            # 应用 target_mask：仅在遮罩区域内做迁移，其余保持原图
            if target_mask is not None:
                mask_slice = target_mask[offset:offset + cur_batch]
                matched = mask_slice * matched + (1.0 - mask_slice) * image

            offset += cur_batch

            if color_space == "LAB":
                matched = kornia.color.lab_to_rgb(matched)
            elif color_space == "Linear RGB":
                matched = kornia.color.linear_rgb_to_rgb(matched)
            elif color_space == "YCbCr":
                matched = kornia.color.ycbcr_to_rgb(matched)
            elif color_space == "LUV":
                matched = kornia.color.luv_to_rgb(matched)
            elif color_space == "YUV":
                matched = kornia.color.yuv_to_rgb(matched)
            elif color_space == "XYZ":
                matched = kornia.color.xyz_to_rgb(matched)

            out = (
                matched.permute([0, 2, 3, 1])
                .clamp(0, 1)
                .to(comfy.model_management.intermediate_device())
            )
            output.append(out)

        output = torch.cat(output, dim=0)
        return (output,)

    def compute_mean_std(self, tensor, mask=None):
        if mask is not None:
            masked_tensor = tensor * mask

            mask_sum = mask.sum(dim=[2, 3], keepdim=True)
            mask_sum = torch.clamp(mask_sum, min=1e-6)

            mean = torch.nan_to_num(
                masked_tensor.sum(dim=[2, 3], keepdim=True) / mask_sum
            )
            std = torch.sqrt(
                torch.nan_to_num(
                    ((masked_tensor - mean) ** 2 * mask).sum(dim=[2, 3], keepdim=True)
                    / mask_sum
                )
            )
        else:
            mean = tensor.mean(dim=[2, 3], keepdim=True)
            std = tensor.std(dim=[2, 3], keepdim=True)
        return mean, std


IMAGE_PROCESSING_CLASS_MAPPINGS = {
    "ColorAdjustment-": ColorAdjustment,
    "ColorTint-": ColorTint,
    "ColorBlockEffect-": ColorBlockEffect,
    "FlatteningEffect-": FlatteningEffect,
}

IMAGE_PROCESSING_NAME_MAPPINGS = {
    "ColorAdjustment-": "Image Color Adjustment",
    "ColorTint-": "Image Color Tint",
    "ColorBlockEffect-": "Image Color Block Effect",
    "FlatteningEffect-": "Image Flattening Effect",
}
