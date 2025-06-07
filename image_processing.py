import cv2
import numpy as np
import torch
import comfy.utils
import comfy.model_management
import kornia

from PIL import Image, ImageEnhance
from comfy_extras.nodes_post_processing import Blend, Blur, Quantize
from .utils.image_convert import image_posterize

_CATEGORY = "sfnodes/image_processing"


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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_space": (["LAB", "YCbCr", "RGB", "LUV", "YUV", "XYZ"],),
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "device": (["auto", "cpu", "gpu"],),
                "batch_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "reference_mask": ("MASK",),
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
        factor,
        device,
        batch_size,
        reference_mask=None,
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

            # Reshape mask to (batch, 1, height, width)
            reference_mask = reference_mask.unsqueeze(1).to(device)

            # Ensure the mask is binary (0 or 1)
            reference_mask = (reference_mask > 0.5).float()

            # Ensure spatial dimensions match
            if reference_mask.shape[2:] != reference.shape[2:]:
                reference_mask = comfy.utils.common_upscale(
                    reference_mask,
                    reference.shape[3],
                    reference.shape[2],
                    upscale_method="bicubic",
                    crop="center",
                )

        if batch_size == 0 or batch_size > image.shape[0]:
            batch_size = image.shape[0]

        if "LAB" == color_space:
            reference = kornia.color.rgb_to_lab(reference)
        elif "YCbCr" == color_space:
            reference = kornia.color.rgb_to_ycbcr(reference)
        elif "LUV" == color_space:
            reference = kornia.color.rgb_to_luv(reference)
        elif "YUV" == color_space:
            reference = kornia.color.rgb_to_yuv(reference)
        elif "XYZ" == color_space:
            reference = kornia.color.rgb_to_xyz(reference)

        reference_mean, reference_std = self.compute_mean_std(reference, reference_mask)

        image_batch = torch.split(image, batch_size, dim=0)
        output = []

        for image in image_batch:
            image = image.to(device)

            if color_space == "LAB":
                image = kornia.color.rgb_to_lab(image)
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
            matched = factor * matched + (1 - factor) * image

            if color_space == "LAB":
                matched = kornia.color.lab_to_rgb(matched)
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

        out = None
        output = torch.cat(output, dim=0)
        return (output,)

    def compute_mean_std(self, tensor, mask=None):
        if mask is not None:
            # Apply mask to the tensor
            masked_tensor = tensor * mask

            # Calculate the sum of the mask for each channel
            mask_sum = mask.sum(dim=[2, 3], keepdim=True)

            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-6)

            # Calculate mean and std only for masked area
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
