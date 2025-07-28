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
import os

_CATEGORY = "sfnodes/image_processing"
UPSCALE_METHODS = ["lanczos", "nearest-exact", "bilinear", "area", "bicubic"]


def make_even(number):
    _, remainder = divmod(number, 2)
    return number + remainder


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
            }
        )
        return base_inputs

    FUNCTION = "execute"
    DESCRIPTION = """
    根据指定边长缩放图片，shorter为True时参照短边，否则参照长边
    limit为True时，如果图像的最短边小于size，则不缩放图像
    """

    def execute(self, image, size, upscale_method, shorter, limit, mask=None):
        # Check if we should skip scaling
        min_side = min(image.shape[2], image.shape[1])
        if limit and min_side < size:
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

        scaled_image, result_mask = self.scale_image(
            image, width, height, upscale_method, mask
        )
        return self.prepare_result(scaled_image, result_mask, width, height)


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

        
    CATEGORY = _CATEGORY
    
            
   