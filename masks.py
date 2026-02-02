import torch
import folder_paths
import random
import numpy as np

from PIL import Image, ImageFilter, ImageOps
from comfy.utils import common_upscale
from nodes import SaveImage
from .utils.image_convert import mask2tensor, np2tensor, tensor2mask, rescale_image
from .utils.mask_utils import (
    blur_mask,
    combine_mask,
    fill_holes,
    expand_mask,
    invert_mask,
    apply_mask_area,
    mask_unsqueeze,
    mask_floor,
    make_odd,
    binary_erosion,
    gaussian_blur,
    mask_process,
)

_CATEGORY = "sfnodes/masks"


class OutlineMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "outer_width": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "设置外轮廓宽度，范围为0到16384，步长为1",
                    },
                ),
                "inner_width": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "设置内轮廓宽度，范围为0到16384，步长为1",
                    },
                ),
                "tapered_corners": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "是否使用锥形角"},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "execute"

    CATEGORY = _CATEGORY
    DESCRIPTION = "给遮罩添加内外轮廓线"

    def execute(self, mask, outer_width, inner_width, tapered_corners):
        m1 = expand_mask(mask, outer_width, tapered_corners)
        m2 = expand_mask(mask, -inner_width, tapered_corners)

        m3 = combine_mask(m1, m2, 0, 0)

        return (m3,)


class CreateBlurredEdgeMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 0, "max": 14096, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 14096, "step": 1}),
                "border": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "border_percent": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "blur_radius": (
                    "INT",
                    {"default": 10, "min": 0, "max": 4096, "step": 1},
                ),
                "blur_radius_percent": (
                    "FLOAT",
                    {"default": 0.00, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltips": "如果未提供图像，将使用输入的宽度和高度创建一个白色图像。"
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据指定图片增加模糊边框"

    def execute(
        self,
        width,
        height,
        border,
        border_percent,
        blur_radius,
        blur_radius_percent,
        image=None,
    ):
        if image is not None:
            _, height, width, _ = image.shape

        # 计算边框宽度
        border_width = int(min(width, height) * border_percent + border)

        # 计算内部图像的尺寸
        inner_width = width - 2 * border_width
        inner_height = height - 2 * border_width

        # 创建内部白色图像
        inner_image = Image.new("RGB", (inner_width, inner_height), "white")

        # 扩展图像，添加黑色边框
        image_with_border = ImageOps.expand(
            inner_image, border=border_width, fill="black"
        )

        # 计算模糊半径
        blur_radius = int(min(width, height) * blur_radius_percent + blur_radius)

        # 应用高斯模糊
        blurred_image = image_with_border.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )

        # 转换为张量
        blurred_tensor = np2tensor(blurred_image)
        blurred_image = blurred_tensor.unsqueeze(0)

        return (tensor2mask(blurred_image),)


class MaskChange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),

                "grow": (
                    "INT",
                    {
                        "default": 0,
                        "min": -4096,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置生长值，范围为-4096到4096，步长为1",
                    },
                ),
                "grow_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置生长百分比，范围为-2.0到2.0，步长为0.01",
                    },
                ),
                "grow_tapered": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否使用锥形角"},
                ),
                "blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置模糊值，范围为0到4096，步长为1",
                    },
                ),
                "blur_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置模糊百分比，范围为0.0到2.0，步长为0.01",
                    },
                ),
                "fill": ("BOOLEAN", {"default": False, "tooltip": "是否填充孔洞"}),
                "pre_invert": ("BOOLEAN", {"default": False, "tooltip": "是否预反转(先反转,再其他操作)"}),



            },
        }

    RETURN_TYPES = ("MASK", "MASK", )
    RETURN_NAMES = ("mask", "inverted_mask",)

    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "修改和处理遮罩"

    def execute(self, mask, grow, grow_percent, grow_tapered, blur, blur_percent, fill, pre_invert):
        print(mask.shape)

        mask_params = {
                "pre_invert": False,
                "mask": mask,
                "grow": grow,
                "grow_percent": grow_percent,
                "grow_tapered": grow_tapered,
                "blur": blur,
                "blur_percent": blur_percent,
                "fill": fill,
                "invert": False,

            }
        mask = mask_process(mask, mask_params, unqueeze=False)  


        if not pre_invert:
            mask_inverted = invert_mask(mask)
        else:
            mask_params_invert = {
            "pre_invert": True,
            "mask": mask,
            "grow": grow,
            "grow_percent": grow_percent,
            "grow_tapered": grow_tapered,
            "blur": blur,
            "blur_percent": blur_percent,
            "fill": fill,
            "invert": False,

        }
            mask_inverted = mask_process(mask, mask_params_invert, unqueeze=False)    


        return (mask, mask_inverted)


class Depth2Mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_depth": ("IMAGE",),
                "depth": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "number",
                        "tooltip": "设置深度阈值，范围为0.0到1.0，步长为0.01",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "mask_inverted")

    FUNCTION = "execute"

    CATEGORY = _CATEGORY
    DESCRIPTION = "将深度图像转换为遮罩"

    def execute(self, image_depth, depth):
        def upscale(image, upscale_method, width, height):
            samples = image.movedim(-1, 1)
            s = common_upscale(samples, width, height, upscale_method, "disabled")
            s = s.movedim(1, -1)
            return (s,)

        bs, height, width = (
            image_depth.size()[0],
            image_depth.size()[1],
            image_depth.size()[2],
        )

        mask1 = torch.zeros((bs, height, width))

        image_depth = upscale(image_depth, "lanczos", width, height)[0]

        mask1 = (image_depth[..., 0] < depth).float()

        return mask1, 1.0 - mask1


class MaskScaleBy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "scale_by": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "设置缩放比例，范围为0.01到8.0，步长为0.01",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "upscale"

    CATEGORY = _CATEGORY
    DESCRIPTION = "根据指定比例缩放遮罩"

    def upscale(self, mask, scale_by):
        image = mask2tensor(mask)
        samples = image.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = common_upscale(samples, width, height, "lanczos", "disabled")
        s = s.movedim(1, -1)
        return (tensor2mask(s),)


class MaskScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "upscale"

    CATEGORY = _CATEGORY
    DESCRIPTION = "根据指定宽高缩放遮罩"

    def upscale(self, mask, width, height):
        image = mask2tensor(mask)
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1, 1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = common_upscale(samples, width, height, "lanczos", "disabled")
            s = s.movedim(1, -1)
        return (tensor2mask(s),)


class MaskPaintArea:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_mask": ("MASK", {"tooltip": "目标遮罩"}),
                "area_mask": ("MASK", {"tooltip": "区域遮罩"}),
                "paint_mode": (
                    ["白色 (1.0)", "黑色 (0.0)", "自定义值"],
                    {"default": "白色 (1.0)", "tooltip": "选择涂黑或涂白模式"},
                ),
                "custom_value": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "设置自定义值，范围为0.0到1.0，步长为0.01",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据区域遮罩对目标遮罩进行涂黑或涂白"

    def execute(self, target_mask, area_mask, paint_mode, custom_value):
        # 根据选择的模式确定要应用的值
        if paint_mode == "白色 (1.0)":
            paint_value = 1.0
        elif paint_mode == "黑色 (0.0)":
            paint_value = 0.0
        else:  # 自定义值
            paint_value = custom_value

        # 应用区域遮罩
        result_mask = apply_mask_area(target_mask, area_mask, paint_value)

        return (result_mask,)


class MaskAdjustGrayscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "输入遮罩"}),
                "gray_value": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "设置灰度值，范围为0.0(黑)到1.0(白)",
                    },
                ),
                "apply_to": (
                    ["整个遮罩", "仅非零区域"],
                    {"default": "仅非零区域", "tooltip": "选择应用灰度值的区域"},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "调整强度，1.0为完全应用新灰度，0.0为保持原样",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将遮罩设置为指定的灰度值"

    def execute(self, mask, gray_value, apply_to, strength):
        import torch

        # 创建一个新的张量以避免修改原始数据
        result_mask = mask.clone()

        if apply_to == "整个遮罩":
            # 应用到整个遮罩
            if strength >= 1.0:
                # 直接设置为指定灰度值
                result_mask = torch.ones_like(mask) * gray_value
            else:
                # 根据强度混合原始遮罩和新灰度值
                result_mask = mask * (1.0 - strength) + (
                    torch.ones_like(mask) * gray_value * strength
                )
        else:
            # 只应用到非零区域
            non_zero_mask = (mask > 0).float()

            if strength >= 1.0:
                # 直接设置非零区域为指定灰度值
                result_mask = torch.where(
                    non_zero_mask > 0, torch.ones_like(mask) * gray_value, mask
                )
            else:
                # 根据强度混合原始遮罩和新灰度值，只在非零区域
                new_values = mask * (1.0 - strength) + (
                    torch.ones_like(mask) * gray_value * strength
                )
                result_mask = torch.where(non_zero_mask > 0, new_values, mask)

        return (result_mask,)


class PreviewMask(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)


class MaskedFill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill": (["neutral", "telea", "navier-stokes"],),
                "falloff": ("INT", {"default": 0, "min": 0, "max": 8191, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = _CATEGORY
    FUNCTION = "fill"

    def fill(self, image, mask, fill: str, falloff: int):
        image = image.detach().clone()
        alpha = mask_unsqueeze(mask_floor(mask))
        assert alpha.shape[0] == image.shape[0], (
            "Image and mask batch size does not match"
        )

        falloff = make_odd(falloff)
        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)

        if fill == "neutral":
            m = (1.0 - alpha).squeeze(1)
            for i in range(3):
                image[:, :, :, i] -= 0.5
                image[:, :, :, i] *= m
                image[:, :, :, i] += 0.5
        else:
            import cv2

            method = cv2.INPAINT_TELEA if fill == "telea" else cv2.INPAINT_NS
            for slice, alpha_slice in zip(image, alpha):
                alpha_np = alpha_slice.squeeze().cpu().numpy()
                alpha_bc = alpha_np.reshape(*alpha_np.shape, 1)
                image_np = slice.cpu().numpy()
                filled_np = cv2.inpaint(
                    (255.0 * image_np).astype(np.uint8),
                    (255.0 * alpha_np).astype(np.uint8),
                    3,
                    method,
                )
                filled_np = filled_np.astype(np.float32) / 255.0
                filled_np = image_np * (1.0 - alpha_bc) + filled_np * alpha_bc
                slice.copy_(torch.from_numpy(filled_np))

        return (image,)


class ImageMaskToTransparency:
    """输入图片和 mask，输出带透明通道的图片，mask 遮盖区域透明。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图片"}),
                "mask": ("MASK", {"tooltip": "遮罩，遮盖区域在输出中为透明"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "输入图片和 mask，输出带透明通道的图片(RGBA)，mask 遮盖区域透明"

    def execute(self, image, mask):
        batch_size = min(image.shape[0], mask.shape[0])
        image = image[:batch_size]
        mask = mask[:batch_size]

        # Resize mask to image spatial size (H, W)
        mask_bchw = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        target_h, target_w = image.shape[1], image.shape[2]
        mask_resized = torch.nn.functional.interpolate(
            mask_bchw, size=(target_h, target_w), mode="bilinear"
        )
        mask_resized = mask_resized.squeeze(1)  # (B, H, W)

        # alpha = 1 - mask (mask=1 -> transparent)
        alpha = 1.0 - mask_resized
        alpha = alpha.unsqueeze(-1)  # (B, H, W, 1)

        # Concat RGB + alpha -> (B, H, W, 4) RGBA
        rgb = image[:, :, :, :3]
        rgba = torch.cat([rgb, alpha], dim=-1)

        return (rgba,)


class FillWithReferenceColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE", {"tooltip": "要填充的目标图像"}),
                "reference_image": ("IMAGE", {"tooltip": "用于获取平均颜色的参考图像"}),
                "mask": ("MASK", {"tooltip": "指定填充区域的蒙版"}),
                "falloff": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8191,
                        "step": 1,
                        "tooltip": "边缘过渡范围，值越大过渡越平滑",
                    },
                ),
                "opacity": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "填充颜色的不透明度",
                    },
                ),
            },
            "optional": {
                "reference_mask": (
                    "MASK",
                    {"tooltip": "可选的参考图像蒙版，只计算蒙版区域内的平均颜色"},
                )
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = _CATEGORY
    FUNCTION = "fill_with_color"
    DESCRIPTION = "使用参考图像的平均颜色填充目标图像蒙版区域"

    def fill_with_color(
        self,
        target_image,
        reference_image,
        mask,
        falloff: int,
        opacity: float,
        reference_mask=None,
    ):
        import torch

        # 克隆目标图像，避免修改原始数据
        result_image = target_image.detach().clone()

        # 处理蒙版
        alpha = mask_unsqueeze(mask_floor(mask))
        assert alpha.shape[0] == result_image.shape[0], "图像和蒙版的批次大小不匹配"

        # 应用falloff效果
        falloff = make_odd(falloff)
        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)

        # 计算参考图像的平均颜色
        if reference_mask is not None:
            # 重新实现参考蒙版的颜色计算逻辑
            # 确保参考蒙版是二维的(batch_size, height, width)
            # 获取参考蒙版的索引
            batch_size = reference_image.shape[0]

            # 初始化存储平均颜色的张量
            avg_colors = []

            # 对批次中的每个图像分别计算平均颜色
            for batch_idx in range(batch_size):
                # 获取当前批次的图像和蒙版
                curr_img = reference_image[batch_idx]
                curr_mask = (
                    reference_mask[batch_idx]
                    if reference_mask.dim() > 2
                    else reference_mask
                )

                # 确保蒙版是二维的(height, width)
                if curr_mask.dim() > 2:
                    curr_mask = curr_mask.squeeze()

                # 创建布尔掩码并添加通道维度
                bool_mask = (curr_mask > 0.0).unsqueeze(-1)

                # 将布尔掩码扩展到三个通道
                bool_mask_expanded = bool_mask.expand(*curr_img.shape)

                # 计算蒙版区域内像素的总数
                mask_pixel_count = bool_mask.sum().item()

                if mask_pixel_count > 0:
                    # 如果蒙版内有像素，计算区域平均颜色
                    masked_img = curr_img * bool_mask_expanded
                    sum_color = masked_img.sum(dim=(0, 1))
                    avg_color = sum_color / mask_pixel_count
                else:
                    # 如果蒙版为空，使用整个图像的平均颜色
                    avg_color = curr_img.mean(dim=(0, 1))

                avg_colors.append(avg_color)

            # 将收集到的平均颜色转换为张量
            avg_color = torch.stack(avg_colors, dim=0)

            # 如果是单一批次，则去除批次维度
            if avg_color.dim() > 1 and avg_color.shape[0] == 1:
                avg_color = avg_color.squeeze(0)
        else:
            # 使用整个参考图像计算平均颜色
            avg_color = torch.mean(
                reference_image.reshape(-1, reference_image.shape[-1]), dim=0
            )

        # 将平均颜色应用到蒙版区域
        for i in range(result_image.shape[0]):  # 处理每一批图像
            # 创建填充颜色图像
            batch_avg_color = (
                avg_color[i]
                if isinstance(avg_color, torch.Tensor) and avg_color.dim() > 1
                else avg_color
            )
            color_fill = torch.ones_like(result_image[i]) * batch_avg_color

            # 根据不透明度和alpha混合颜色
            alpha_i = alpha[i].squeeze(0)
            alpha_with_opacity = alpha_i * opacity

            # 扩展alpha维度以匹配图像通道
            alpha_expanded = alpha_with_opacity.unsqueeze(-1).expand(-1, -1, 3)

            # 混合原始图像和填充颜色
            result_image[i] = (
                result_image[i] * (1 - alpha_expanded) + color_fill * alpha_expanded
            )

        return (result_image,)


class MaskParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pre_invert": ("BOOLEAN", {"default": False, "tooltip": "是否反转遮罩(注意:先反转,再其他操作)"}),

                "grow": (
                    "INT",
                    {
                        "default": 0,
                        "min": -4096,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置生长值，范围为-4096到4096，步长为1",
                    },
                ),
                "grow_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置生长百分比，范围为-2.0到2.0，步长为0.01",
                    },
                ),
                "grow_tapered": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否使用锥形角"},
                ),
                "blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置模糊值，范围为0到4096，步长为1",
                    },
                ),
                "blur_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置模糊百分比，范围为0.0到2.0，步长为0.01",
                    },
                ),
                "fill": ("BOOLEAN", {"default": False, "tooltip": "是否填充孔洞"}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "输出结果反转"}),


            },
        }

    RETURN_TYPES = ("MASKPARAMS",)
    RETURN_NAMES = ("mask_params",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "设置遮罩参数"

    def execute(self, pre_invert, grow, grow_percent, grow_tapered, blur, blur_percent, fill, invert):

        mask_params = {
            "pre_invert": pre_invert,
            "grow": grow,
            "grow_percent": grow_percent,
            "grow_tapered": grow_tapered,
            "blur": blur,
            "blur_percent": blur_percent,
            "fill": fill,
            "invert": invert,
        }

        return (mask_params,)


class MaskParamsEdges:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pre_invert": ("BOOLEAN", {"default": False, "tooltip": "是否反转遮罩(注意:先反转,再其他操作)"}),
                
                # 上边增长参数
                "grow_top": (
                    "INT",
                    {
                        "default": 0,
                        "min": -4096,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置上边生长值，范围为-4096到4096，步长为1",
                    },
                ),
                "grow_top_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置上边生长百分比，范围为-2.0到2.0，步长为0.01",
                    },
                ),
                
                # 下边增长参数
                "grow_bottom": (
                    "INT",
                    {
                        "default": 0,
                        "min": -4096,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置下边生长值，范围为-4096到4096，步长为1",
                    },
                ),
                "grow_bottom_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置下边生长百分比，范围为-2.0到2.0，步长为0.01",
                    },
                ),
                
                # 左边增长参数
                "grow_left": (
                    "INT",
                    {
                        "default": 0,
                        "min": -4096,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置左边生长值，范围为-4096到4096，步长为1",
                    },
                ),
                "grow_left_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置左边生长百分比，范围为-2.0到2.0，步长为0.01",
                    },
                ),
                
                # 右边增长参数
                "grow_right": (
                    "INT",
                    {
                        "default": 0,
                        "min": -4096,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置右边生长值，范围为-4096到4096，步长为1",
                    },
                ),
                "grow_right_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置右边生长百分比，范围为-2.0到2.0，步长为0.01",
                    },
                ),
                
                "grow_tapered": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否使用锥形角"},
                ),
                
                # 上边模糊参数
                "blur_top": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置上边模糊值，范围为0到4096，步长为1",
                    },
                ),
                "blur_top_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置上边模糊百分比，范围为0.0到2.0，步长为0.01",
                    },
                ),
                
                # 下边模糊参数
                "blur_bottom": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置下边模糊值，范围为0到4096，步长为1",
                    },
                ),
                "blur_bottom_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置下边模糊百分比，范围为0.0到2.0，步长为0.01",
                    },
                ),
                
                # 左边模糊参数
                "blur_left": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置左边模糊值，范围为0到4096，步长为1",
                    },
                ),
                "blur_left_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置左边模糊百分比，范围为0.0到2.0，步长为0.01",
                    },
                ),
                
                # 右边模糊参数
                "blur_right": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置右边模糊值，范围为0到4096，步长为1",
                    },
                ),
                "blur_right_percent": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置右边模糊百分比，范围为0.0到2.0，步长为0.01",
                    },
                ),
                
                "fill": ("BOOLEAN", {"default": False, "tooltip": "是否填充孔洞"}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "输出结果反转"}),
            },
        }

    RETURN_TYPES = ("MASKPARAMS",)
    RETURN_NAMES = ("mask_params",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "设置遮罩参数（支持四个边单独定义增长和模糊参数）"

    def execute(
        self,
        pre_invert,
        grow_top, grow_top_percent,
        grow_bottom, grow_bottom_percent,
        grow_left, grow_left_percent,
        grow_right, grow_right_percent,
        grow_tapered,
        blur_top, blur_top_percent,
        blur_bottom, blur_bottom_percent,
        blur_left, blur_left_percent,
        blur_right, blur_right_percent,
        fill,
        invert,
    ):
        mask_params = {
            "pre_invert": pre_invert,
            "grow_top": grow_top,
            "grow_top_percent": grow_top_percent,
            "grow_bottom": grow_bottom,
            "grow_bottom_percent": grow_bottom_percent,
            "grow_left": grow_left,
            "grow_left_percent": grow_left_percent,
            "grow_right": grow_right,
            "grow_right_percent": grow_right_percent,
            "grow_tapered": grow_tapered,
            "blur_top": blur_top,
            "blur_top_percent": blur_top_percent,
            "blur_bottom": blur_bottom,
            "blur_bottom_percent": blur_bottom_percent,
            "blur_left": blur_left,
            "blur_left_percent": blur_left_percent,
            "blur_right": blur_right,
            "blur_right_percent": blur_right_percent,
            "fill": fill,
            "invert": invert,
        }

        return (mask_params,)


class MaskCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像"}),
                "mask": ("MASK", {"tooltip": "用于裁剪的遮罩，白色区域将被裁剪掉"}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "反转遮罩，裁剪黑色区域而保留白色区域"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "遮罩阈值，大于此值的区域被视为白色区域"}),
                "actual_crop": ("BOOLEAN", {"default": True, "tooltip": "是否实际裁剪图像。如果为True，将裁剪图像；如果为False，只会将遮罩区域变黑"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "裁剪掉图像中的遮罩区域"

    def execute(self, image, mask, invert_mask, threshold, actual_crop):
        import torch
        import numpy as np
        from PIL import Image
        
        # 确保图像和遮罩的批次大小匹配
        assert image.shape[0] == mask.shape[0] or mask.shape[0] == 1, "图像和遮罩的批次大小不匹配"
        
        # 如果遮罩和图像尺寸不同，调整遮罩大小
        if image.shape[1:3] != mask.shape[1:3]:
            # 将遮罩转换为张量格式以便于缩放
            mask_tensor = mask2tensor(mask)
            # 调整遮罩大小以匹配图像
            mask_tensor = rescale_image(mask_tensor, image.shape[2], image.shape[1])
            # 转换回遮罩格式
            mask = tensor2mask(mask_tensor)
        
        # 如果需要反转遮罩
        if invert_mask:
            mask = 1.0 - mask
        
        # 如果不需要实际裁剪，只是将遮罩区域变黑
        if not actual_crop:
            # 克隆图像以避免修改原始数据
            result_image = image.detach().clone()
            
            # 将遮罩调整为适合图像处理的格式
            alpha = mask_unsqueeze(mask)
            
            # 确保遮罩和图像的批次大小匹配
            if alpha.shape[0] == 1 and result_image.shape[0] > 1:
                alpha = alpha.repeat(result_image.shape[0], 1, 1, 1)
            
            # 应用遮罩（将遮罩区域设为黑色）
            for i in range(result_image.shape[0]):  # 处理每一批图像
                # 扩展alpha维度以匹配图像通道
                alpha_expanded = alpha[i].squeeze(0).unsqueeze(-1).expand(-1, -1, 3)
                # 将遮罩区域设为黑色（0）
                result_image[i] = result_image[i] * (1.0 - alpha_expanded)
            
            return (result_image,)
        
        # 实际裁剪图像的逻辑
        # 创建结果图像列表
        result_images = []
        
        # 处理每一批图像
        for i in range(image.shape[0]):
            # 获取当前批次的图像和遮罩
            curr_img = image[i].cpu().numpy()
            curr_mask = mask[i if mask.shape[0] > 1 else 0].cpu().numpy()
            
            # 将遮罩二值化
            binary_mask = (curr_mask > threshold).astype(np.uint8)
            
            # 将图像转换为PIL格式以便处理
            pil_img = Image.fromarray((curr_img * 255).astype(np.uint8))
            pil_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
            
            # 获取非零区域的边界框（非遮罩区域）
            non_zero = np.where(binary_mask == 0)
            if len(non_zero[0]) > 0 and len(non_zero[1]) > 0:
                # 计算边界框
                min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
                min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])
                
                # 裁剪图像
                cropped_img = pil_img.crop((min_x, min_y, max_x + 1, max_y + 1)) # type: ignore
                
                # 转换回numpy格式
                cropped_np = np.array(cropped_img).astype(np.float32) / 255.0
                
                # 添加到结果列表
                result_images.append(torch.from_numpy(cropped_np))
            else:
                # 如果没有非遮罩区域，返回空图像（1x1像素）
                empty_img = torch.zeros((1, 1, 3), dtype=torch.float32)
                result_images.append(empty_img)
        
        # 将结果列表转换为批次张量
        # 注意：由于裁剪后的图像可能大小不同，我们需要调整它们到相同大小
        if len(result_images) == 1:
            return (result_images[0].unsqueeze(0),)
        else:
            # 找出最大的宽度和高度
            max_height = max([img.shape[0] for img in result_images])
            max_width = max([img.shape[1] for img in result_images])
            
            # 调整所有图像到相同大小
            resized_images = []
            for img in result_images:
                if img.shape[0] != max_height or img.shape[1] != max_width:
                    # 创建新的空白图像
                    resized = torch.zeros((max_height, max_width, 3), dtype=torch.float32)
                    # 将原始图像复制到新图像的左上角
                    h, w = img.shape[0], img.shape[1]
                    resized[:h, :w, :] = img
                    resized_images.append(resized)
                else:
                    resized_images.append(img)
            
            # 堆叠所有调整大小后的图像
            return (torch.stack(resized_images),)


class MaskFillPercentArea:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "输入遮罩"}),
                "fill_direction": (
                    ["横向", "纵向"],
                    {"default": "横向", "tooltip": "选择填充方向"},
                ),
                "fill_start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "设置填充起始百分比，范围为0.0到1.0，步长为0.01",
                    },
                ),
                "fill_end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "设置填充结束百分比，范围为0.0到1.0，步长为0.01",
                    },
                ),
                "fill_mode": (
                    ["白色 (1.0)", "黑色 (0.0)"],
                    {"default": "白色 (1.0)", "tooltip": "选择填充颜色"},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "填充遮罩的横向或纵向百分比区域为白色或黑色"

    def execute(self, mask, fill_direction, fill_start_percent, fill_end_percent, fill_mode):
        # 确保起始百分比小于等于结束百分比
        if fill_start_percent > fill_end_percent:
            fill_start_percent, fill_end_percent = fill_end_percent, fill_start_percent

        # 根据选择的模式确定要应用的值
        fill_value = 1.0 if fill_mode == "白色 (1.0)" else 0.0

        # 获取mask的形状
        # 如果mask有4个维度，去掉通道维度
        if len(mask.shape) == 4:
            mask = mask.squeeze(1)  # 去掉通道维度
        batch_size, height, width = mask.shape

        # 创建区域遮罩
        area_mask = torch.zeros_like(mask)

        if fill_direction == "横向":
            # 计算填充的起始和结束列
            start_col = int(width * fill_start_percent)
            end_col = int(width * fill_end_percent)
            # 设置区域遮罩
            area_mask[:, :, start_col:end_col] = 1.0
        else:  # 纵向
            # 计算填充的起始和结束行
            start_row = int(height * fill_start_percent)
            end_row = int(height * fill_end_percent)
            # 设置区域遮罩
            area_mask[:, start_row:end_row, :] = 1.0

        # 应用区域遮罩
        result_mask = apply_mask_area(mask, area_mask, fill_value)

        return (result_mask,)


class MaskFillColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图片"}),
                "mask": ("MASK", {"tooltip": "输入遮罩，白色区域将被填充"}),
                "fill_color_r": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "填充颜色的红色通道，范围为0.0到1.0",
                    },
                ),
                "fill_color_g": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "填充颜色的绿色通道，范围为0.0到1.0",
                    },
                ),
                "fill_color_b": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "填充颜色的蓝色通道，范围为0.0到1.0",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "用指定颜色填充图片中mask遮住的部分（默认白色）"

    def execute(self, image, mask, fill_color_r, fill_color_g, fill_color_b):
        import torch
        
        # 克隆图像以避免修改原始数据
        result_image = image.detach().clone()
        
        # 处理遮罩
        alpha = mask_unsqueeze(mask_floor(mask))
        assert alpha.shape[0] == result_image.shape[0] or alpha.shape[0] == 1, (
            "图像和遮罩的批次大小不匹配"
        )
        
        # 如果遮罩和图像尺寸不同，调整遮罩大小
        if image.shape[1:3] != alpha.shape[2:4]:
            # 将遮罩转换为张量格式以便于缩放
            mask_tensor = mask2tensor(mask)
            # 调整遮罩大小以匹配图像
            mask_tensor = rescale_image(mask_tensor, image.shape[2], image.shape[1])
            # 转换回遮罩格式并重新处理
            mask_rescaled = tensor2mask(mask_tensor)
            alpha = mask_unsqueeze(mask_floor(mask_rescaled))
        
        # 创建填充颜色张量 [R, G, B]
        fill_color = torch.tensor([fill_color_r, fill_color_g, fill_color_b], 
                                  dtype=result_image.dtype, 
                                  device=result_image.device)
        
        # 处理每一批图像
        for i in range(result_image.shape[0]):
            # 获取当前批次的alpha遮罩
            alpha_i = alpha[i if alpha.shape[0] > 1 else 0].squeeze(0)
            
            # 扩展alpha维度以匹配图像通道 [H, W] -> [H, W, 3]
            alpha_expanded = alpha_i.unsqueeze(-1).expand(-1, -1, 3)
            
            # 创建填充颜色图像
            color_fill = torch.ones_like(result_image[i]) * fill_color
            
            # 混合原始图像和填充颜色
            # mask区域用填充颜色，非mask区域保持原图
            result_image[i] = (
                result_image[i] * (1.0 - alpha_expanded) + 
                color_fill * alpha_expanded
            )
        
        return (result_image,)

