import math
from PIL import Image, ImageDraw, ImageFilter
from comfy.utils import common_upscale
from .utils.image_convert import np2tensor, pil2mask, pil2tensor, tensor2np, tensor2pil, mask2pil

_CATEGORY = "sfnodes/inpaint"

class InpaintCutOut:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
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
                "margin": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置贴回去图像的边距像素数",
                    },
                ),
                "margin_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置贴回去图像的边距百分比",
                    },
                ),
                "blur_radius": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置贴回去图像的模糊半径",
                    },
                ),
                "blur_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置贴回去图像的模糊百分比",
                    },
                )
            }
        }

    CATEGORY = _CATEGORY

    RETURN_TYPES = ("IMAGE", "MASK","IMAGE","CUTINFO")
    RETURN_NAMES = ("cutout_image", "cutout_mask","cutout_origin_image","cutinfo")

    FUNCTION = "inpaint_cutout"

    def inpaint_cutout(self, image, mask, padding, padding_percent, rescale_mode, custom_megapixels, margin, margin_percent, blur_radius, blur_percent):
        # 将输入转换为适当的格式
        img = image[0]
        pil_image = tensor2pil(img)
        mask_image = mask2pil(mask)
        
        # 查找mask中非零区域的边界框
        non_zero_coords = []
        for y in range(mask_image.height):
            for x in range(mask_image.width):
                pixel_value = mask_image.getpixel((x, y))
                if pixel_value is None:
                    continue
                if isinstance(pixel_value, tuple):
                    # 如果返回值是元组（例如RGB或RGBA值），检查任何通道是否大于0
                    if any(v > 0 for v in pixel_value):
                        non_zero_coords.append((x, y))
                elif isinstance(pixel_value, (int, float)) and pixel_value > 0:  # 如果是单个值（例如灰度图像）
                    non_zero_coords.append((x, y))
        
        if len(non_zero_coords) == 0:
            raise Exception("Mask没有非零区域，无法裁剪图像")
            
        # 计算边界框
        x_coords, y_coords = zip(*non_zero_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 应用padding
        width = x_max - x_min
        height = y_max - y_min
        
        padding_x = int(width * padding_percent) + padding
        padding_y = int(height * padding_percent) + padding
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(pil_image.width, x_max + padding_x)
        y_max = min(pil_image.height, y_max + padding_y)
        
        # 更新宽度和高度
        width = x_max - x_min
        height = y_max - y_min
        
        # 裁剪图像和mask
        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
        cropped_mask = mask_image.crop((x_min, y_min, x_max, y_max))
        
        # 计算目标大小
        target_size = self._get_target_size(rescale_mode, custom_megapixels)
        
        # 缩放图像和mask（如果需要）
        if target_size > 0:
            scale_factor = math.sqrt(target_size / (width * height))
            new_width = round(width * scale_factor)
            new_height = round(height * scale_factor)
            cropped_image = cropped_image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            cropped_mask = cropped_mask.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        
        # 为边缘创建模糊mask
        ref_size = max(width, height)
        margin_size = int(ref_size * margin_percent) + margin
        blur_size = int(ref_size * blur_percent) + blur_radius
        
        edge_mask = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(edge_mask)
        draw.rectangle(((0, 0), (width, height)), outline="black", width=margin_size)
        edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(blur_size))
        
        if target_size > 0:
            edge_mask = edge_mask.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        
        # 创建cutinfo字典
        cutinfo = {
            "x": x_min,
            "y": y_min,
            "width": width,
            "height": height,
            "mask": pil2mask(edge_mask),
            "source_image": image,
        }
        
        # 转换回tensor格式
        cutout_image = pil2tensor(cropped_image)
        cutout_mask = pil2mask(cropped_mask)
        cutout_origin_image = pil2tensor(pil_image)
        return (cutout_image, cutout_mask, cutout_origin_image, cutinfo)
        
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


class InpaintPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cutinfo": ("CUTINFO",),
                "source_image": ("IMAGE",),

            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "paste"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将cutinfo中的图像贴回原图"

    
    def paste(self, cutinfo, source_image):
        # 从cutinfo中获取图像和位置信息
        x = cutinfo["x"]
        y = cutinfo["y"]
        width = cutinfo["width"]
        height = cutinfo["height"]
        mask = cutinfo["mask"]
        destination_image = cutinfo["source_image"]
        
        destination = tensor2pil(destination_image[0])
        source = tensor2pil(source_image[0])

        mask_image = mask2pil(mask)
        
        # 如果源图像尺寸与目标区域尺寸不匹配，进行调整
        if source.width != width or source.height != height:
            source = source.resize((width, height), resample=Image.Resampling.LANCZOS)
            mask_image = mask_image.resize((width, height), resample=Image.Resampling.LANCZOS)

        position = (x, y)
        destination.paste(source, position, mask_image)

        return pil2tensor(destination), pil2mask(mask_image)

class ExtractCutInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cutinfo": ("CUTINFO",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT","MASK","IMAGE")
    RETURN_NAMES = ("x", "y", "width", "height", "mask","source_image")
    FUNCTION = "extract"

    def extract(self, cutinfo):
        return (cutinfo["x"], cutinfo["y"], cutinfo["width"], cutinfo["height"], cutinfo["mask"], cutinfo["source_image"])
