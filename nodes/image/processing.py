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
    DESCRIPTION = (
        "将目标图像的颜色统计分布匹配到参考图像。"
        "Statistical：基于所选色彩空间的逐通道统计匹配（均值+标准差）；"
        "Mean：仅平移均值到参考色，保留原图对比度（RGB 域）；"
        "MKL：全协方差矩阵匹配（Monge-Kantorovich 线性最优传输），校正通道间相关性偏移（RGB 域）；"
        "Wavelet：低频差补偿，适合同构图白平衡修复，保留高频细节（RGB 域）。"
        "reference_mask 限定参考统计区域，target_sample_mask 限定目标统计区域，target_mask 限定应用区域（软混合）"
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_image": ("IMAGE", {"tooltip": "目标图像（将被修改颜色分布）"}),
                "reference_image": ("IMAGE", {"tooltip": "参考图像（提供目标颜色分布）"}),
                "color_space": (
                    ["LAB", "Linear RGB", "YCbCr", "RGB", "LUV", "YUV", "XYZ"],
                    {"tooltip": "用于统计匹配的色彩空间，LAB 最常用（L=光照, a/b=色彩）；Linear RGB 为物理线性空间，光照迁移更准确；仅 Statistical 方法使用"},
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
                "method": (
                    ["Statistical", "Mean", "MKL", "Wavelet"],
                    {
                        "default": "Statistical",
                        "tooltip": "匹配算法：Statistical=所选色彩空间的逐通道均值+标准差匹配；Mean=仅平移均值，保留原图对比度；MKL=全协方差匹配（Monge-Kantorovich 线性最优传输），校正通道间相关性偏移；Wavelet=低频差补偿，适合同构图白平衡修复。Mean/MKL/Wavelet 固定于 RGB 域执行",
                    },
                ),
            },
            "optional": {
                "reference_mask": ("MASK", {"tooltip": "参考图的遮罩，仅统计遮罩区域的色彩分布"}),
                "target_sample_mask": ("MASK", {"tooltip": "目标图的统计采样遮罩，仅统计遮罩区域的色彩分布（Statistical/Mean/MKL 的目标统计来源），迁移仍应用于全图"}),
                "target_mask": ("MASK", {"tooltip": "目标图的遮罩，仅对遮罩区域应用色彩迁移（软混合）"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(
        self,
        target_image,
        reference_image,
        color_space,
        strength,
        device,
        batch_size,
        method,
        reference_mask=None,
        target_mask=None,
        target_sample_mask=None,
    ):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = "cpu"

        target_image = target_image.permute([0, 3, 1, 2])
        reference_image = reference_image.permute([0, 3, 1, 2]).to(device)

        reference_mask = self._prepare_mask(
            reference_mask, reference_image.shape[0], reference_image.shape[2], reference_image.shape[3], device, binarize=True
        )
        target_mask = self._prepare_mask(
            target_mask, target_image.shape[0], target_image.shape[2], target_image.shape[3], device, binarize=False
        )
        target_sample_mask = self._prepare_mask(
            target_sample_mask, target_image.shape[0], target_image.shape[2], target_image.shape[3], device, binarize=True
        )

        # 循环外的参考侧统计/变换
        if method == "Statistical":
            reference_image = self._to_color_space(reference_image, color_space)
            reference_mean, reference_std = self.compute_mean_std(reference_image, reference_mask)

            # 多帧参考图时聚合为单帧统计：合并方差（而非平均 std），防止与 image batch 维度不匹配
            if reference_mean.shape[0] > 1:
                ref_var = (
                    (reference_std ** 2 + reference_mean ** 2).mean(dim=0, keepdim=True)
                    - reference_mean.mean(dim=0, keepdim=True) ** 2
                )
                reference_mean = reference_mean.mean(dim=0, keepdim=True)
                reference_std = torch.sqrt(torch.clamp(ref_var, min=0))

            # 部分色彩空间（如 LUV）黑像素会传播 nan，防脏点
            reference_mean = torch.nan_to_num(reference_mean)
            reference_std = torch.nan_to_num(reference_std)
        elif method == "Mean":
            reference_mean, _ = self.compute_mean_std(reference_image, reference_mask)
            if reference_mean.shape[0] > 1:
                reference_mean = reference_mean.mean(dim=0, keepdim=True)
        elif method == "MKL":
            ref_mu, ref_evals, ref_evecs = self._mkl_components(reference_image, reference_mask)
            sqrt_r = ref_evecs @ torch.diag(torch.sqrt(ref_evals.clamp(min=0))) @ ref_evecs.T
        elif method == "Wavelet":
            # 参考低频：多帧聚合后缩放到目标尺寸
            ref_low = kornia.filters.gaussian_blur2d(reference_image, (91, 91), (15.0, 15.0))
            if ref_low.shape[0] > 1:
                ref_low = ref_low.mean(dim=0, keepdim=True)
            ref_low = comfy.utils.common_upscale(
                ref_low, target_image.shape[3], target_image.shape[2], upscale_method="bicubic", crop="center"
            )

        if batch_size == 0 or batch_size > target_image.shape[0]:
            batch_size = target_image.shape[0]

        image_batch = torch.split(target_image, batch_size, dim=0)
        output = []

        offset = 0
        for target in image_batch:
            cur_batch = target.shape[0]
            target = target.to(device)

            sample_slice = (
                target_sample_mask[offset:offset + cur_batch] if target_sample_mask is not None else None
            )

            if method == "Statistical":
                target = self._to_color_space(target, color_space)
                target_mean, target_std = self.compute_mean_std(target, sample_slice)

                matched = (
                    torch.nan_to_num((target - target_mean) / target_std, posinf=0.0, neginf=0.0)
                    * torch.nan_to_num(reference_std)
                    + reference_mean
                )
                matched = strength * matched + (1.0 - strength) * target

                # 应用 target_mask：仅在遮罩区域内做迁移，其余保持原图（软混合，色彩空间域内，与原版一致）
                if target_mask is not None:
                    mask_slice = target_mask[offset:offset + cur_batch]
                    matched = mask_slice * matched + (1.0 - mask_slice) * target

                matched = self._from_color_space(matched, color_space)
            elif method == "Mean":
                target_mean, _ = self.compute_mean_std(target, sample_slice)
                matched = target + (reference_mean - target_mean)
                matched = strength * matched + (1.0 - strength) * target
            elif method == "MKL":
                target_mu, target_evals, target_evecs = self._mkl_components(target, sample_slice)
                inv_sqrt_t = (
                    target_evecs
                    @ torch.diag(1.0 / torch.sqrt(target_evals.clamp(min=1e-6)))
                    @ target_evecs.T
                )
                transform = sqrt_r @ inv_sqrt_t
                pix = target.permute(0, 2, 3, 1)
                matched = (pix - target_mu) @ transform.T + ref_mu
                matched = matched.permute(0, 3, 1, 2)
                matched = strength * matched + (1.0 - strength) * target
            elif method == "Wavelet":
                target_low = kornia.filters.gaussian_blur2d(target, (91, 91), (15.0, 15.0))
                matched = target + (ref_low - target_low)
                matched = strength * matched + (1.0 - strength) * target

            # 非 Statistical 分支的 target_mask 应用（RGB 域）
            if method != "Statistical" and target_mask is not None:
                mask_slice = target_mask[offset:offset + cur_batch]
                matched = mask_slice * matched + (1.0 - mask_slice) * target

            offset += cur_batch

            out = (
                matched.permute([0, 2, 3, 1])
                .clamp(0, 1)
                .to(comfy.model_management.intermediate_device())
            )
            output.append(out)

        output = torch.cat(output, dim=0)
        return (output,)

    def _prepare_mask(self, mask, batch, height, width, device, binarize=False):
        """统一遮罩预处理：加通道维、对齐空间尺寸、广播 batch 维，可选二值化。"""
        if mask is None:
            return None
        mask = mask.unsqueeze(1).to(device)

        if mask.shape[2:] != (height, width):
            mask = comfy.utils.common_upscale(
                mask, width, height, upscale_method="bicubic", crop="center"
            )

        # 匹配 batch 维度：不足时重复最后一帧，超出时截断
        if mask.shape[0] < batch:
            repeats = batch - mask.shape[0]
            mask = torch.cat([mask, mask[-1:].repeat(repeats, 1, 1, 1)], dim=0)
        elif mask.shape[0] > batch:
            mask = mask[:batch]

        if binarize:
            mask = (mask > 0.1).float()
        return mask

    def _to_color_space(self, tensor, color_space):
        if "LAB" == color_space:
            return kornia.color.rgb_to_lab(tensor)
        elif "Linear RGB" == color_space:
            return kornia.color.rgb_to_linear_rgb(tensor)
        elif "YCbCr" == color_space:
            return kornia.color.rgb_to_ycbcr(tensor)
        elif "LUV" == color_space:
            return kornia.color.rgb_to_luv(tensor)
        elif "YUV" == color_space:
            return kornia.color.rgb_to_yuv(tensor)
        elif "XYZ" == color_space:
            return kornia.color.rgb_to_xyz(tensor)
        return tensor

    def _from_color_space(self, tensor, color_space):
        if "LAB" == color_space:
            return kornia.color.lab_to_rgb(tensor)
        elif "Linear RGB" == color_space:
            return kornia.color.linear_rgb_to_rgb(tensor)
        elif "YCbCr" == color_space:
            return kornia.color.ycbcr_to_rgb(tensor)
        elif "LUV" == color_space:
            return kornia.color.luv_to_rgb(tensor)
        elif "YUV" == color_space:
            return kornia.color.yuv_to_rgb(tensor)
        elif "XYZ" == color_space:
            return kornia.color.xyz_to_rgb(tensor)
        return tensor

    def _mkl_components(self, tensor, mask):
        """MKL（Monge-Kantorovich 线性）统计：mask(>0.1) 区域像素的均值与协方差特征分解。"""
        # 先 permute 到 HWC 再索引：CHW 域 bool 索引展平后 reshape(-1, C) 会错位（同通道像素误作像素三元组）
        hw_pixels = tensor.permute(0, 2, 3, 1)  # [B, H, W, C]
        if mask is not None:
            mask_b = mask.permute(0, 2, 3, 1).bool().expand(-1, -1, -1, tensor.shape[1])
            pixels = hw_pixels[mask_b].reshape(-1, tensor.shape[1])
            # 遮罩区域像素过少（如全零遮罩）时兜底全图
            if pixels.shape[0] < tensor.shape[1]:
                pixels = hw_pixels.reshape(-1, tensor.shape[1])
        else:
            pixels = hw_pixels.reshape(-1, tensor.shape[1])

        mu = pixels.mean(dim=0)
        cov = torch.cov(pixels.T) + torch.eye(tensor.shape[1], device=pixels.device, dtype=pixels.dtype) * 1e-6
        evals, evecs = torch.linalg.eigh(cov)
        return mu, evals, evecs

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
