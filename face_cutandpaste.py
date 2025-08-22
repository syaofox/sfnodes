import math

from PIL import Image, ImageDraw, ImageFilter
from comfy.utils import common_upscale
from .utils.image_convert import np2tensor, pil2mask, pil2tensor, tensor2np, tensor2pil, mask2pil, mask2tensor, tensor2mask

_CATEGORY = "sfnodes/face_analysis"

def create_soft_edge_mask(size, margin, blur_radius):
        mask = Image.new("L", size, 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(((0, 0), size), outline="black", width=margin)
        return mask.filter(ImageFilter.GaussianBlur(blur_radius))


class FaceCutout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "image": ("IMAGE",),
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
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置贴回去图像的边距像素数",
                    },
                ),
                "margin_percent": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "设置贴回去图像的边距百分比",
                    },
                ),
                "blur_radius": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "设置贴回去图像的模糊半径",
                    },
                ),
                "blur_percent": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "设置贴回去图像的模糊百分比",
                    },
                ),
                "is_square": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否将图像裁剪为正方形"},
                ),
                "face_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "指定要使用的人脸索引，从0开始",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",       
        "MASK",    
        "BOUNDINGINFO",
    )
    RETURN_NAMES = (
        "face_image",        
        "mask",
        "bounding_info",
    )
  
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "切下图像中所有人脸并进行缩放，返回所有人脸信息"

    def execute(
        self,
        analysis_models,
        image,
        padding,
        padding_percent,
        rescale_mode,
        custom_megapixels,
        margin,
        margin_percent,
        blur_radius,
        blur_percent,
        is_square=False,
        face_index=0,
    ):
        target_size = self._get_target_size(rescale_mode, custom_megapixels)

        img = image[0]

        pil_image = tensor2pil(img)
        face, x, y, width, height = analysis_models.get_single_bbox(
            pil_image, padding, padding_percent, face_index
        )

        if face is None:
            raise Exception("未在图像中检测到人脸。")


        scale_factor = 1

        # 如果需要正方形，调整宽度和高度
        if is_square:
            # 计算正方形边长，取宽高的最大值
            square_size = max(width, height)
            # 计算新的x和y坐标，使人脸居中
            center_x = x + width // 2
            center_y = y + height // 2
            new_x = center_x - square_size // 2
            new_y = center_y - square_size // 2

            # 确保新坐标不超出图像边界
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            if new_x + square_size > pil_image.width:
                new_x = pil_image.width - square_size
            if new_y + square_size > pil_image.height:
                new_y = pil_image.height - square_size

            # 更新坐标和尺寸
            x = new_x
            y = new_y
            width = square_size
            height = square_size

            # 重新裁剪人脸区域为正方形
            face_np = tensor2np(image[0])
            face_crop = face_np[
                y : y + square_size, x : x + square_size
            ]
            face = np2tensor(face_crop).unsqueeze(0)

        if target_size > 0:
            scale_factor = math.sqrt(target_size / (width * height))
            new_width = round(width * scale_factor)
            new_height = round(height * scale_factor)
            scaled_face = self._rescale_image(face, new_width, new_height)
        else:
            new_width = width
            new_height = height
            scaled_face = face


        ref_size = max(new_width, new_height)
        margin_size = int(ref_size * margin_percent) + margin
        blur_size = int(ref_size * blur_percent) + blur_radius
        print(f"[FacePaste] margin_size: {margin_size}, blur_size: {blur_size}")

        mask_image = create_soft_edge_mask((new_width, new_height), margin_size, blur_size)
        mask_tensor = pil2mask(mask_image)

        bounding_info = {            
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "mask": mask_tensor,
            "origin_image": image, 
            "origin_face":face,
            "new_face":scaled_face,
            "new_width":new_width,
            "new_height":new_height,

        }

        return (
            scaled_face,            
            mask_tensor,            
            bounding_info,
        )

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

    @staticmethod
    def _rescale_image(image, width, height):
        samples = image.movedim(-1, 1)
        resized = common_upscale(samples, width, height, "lanczos", "disabled")
        return resized.movedim(1, -1)


class FacePaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bounding_info": ("BOUNDINGINFO",),
                "source_image": ("IMAGE",),                
                "upscale_method": (
                    ["lanczos", "bilinear", "bicubic", "nearest"],
                    {
                        "default": "lanczos",
                        "tooltip": "设置图像缩放的方法"
                    }
                ),
            },
            "optional": {
                "destination_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "paste"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将bounding_info中的人脸图像贴回原图"

    def paste(self, bounding_info, source_image, destination_image=None, upscale_method="lanczos"):
        # 从bounding_info中获取人脸图像和位置信息
        x = bounding_info["x"]
        y = bounding_info["y"]
        width = bounding_info["width"]
        height = bounding_info["height"]

        if destination_image is None:
            destination_image = bounding_info["origin_image"]


        mask_image = mask2tensor(bounding_info["mask"])                   
        source_image = self._rescale_image(source_image, width, height, upscale_method)        
        mask_image = self._rescale_image(mask_image, width, height, upscale_method)


        source_image = tensor2pil(source_image)
        destination_image = tensor2pil(destination_image)

        mask_image = tensor2mask(mask_image)
        mask_image = mask2pil(mask_image)

        position = (x, y)
        print(f"[FacePaste] destination shape: {destination_image.size}, source shape: {source_image.size}, mask shape: {mask_image.size}")
        destination_image.paste(source_image, position, mask_image)

        return pil2tensor(destination_image), pil2mask(mask_image)
    
    @staticmethod
    def _rescale_image(image, width, height, upscale_method="lanczos"):
        samples = image.movedim(-1, 1)
        source_width, source_height = samples.shape[3], samples.shape[2]
        if source_width != width or source_height != height:
            resized = common_upscale(samples, width, height, upscale_method, "disabled")
        else:
            resized = samples
        return resized.movedim(1, -1)

class ExtractBoundingBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bounding_info": ("BOUNDINGINFO",),               
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "MASK", "IMAGE","IMAGE","IMAGE","INT","INT")
    RETURN_NAMES = ("x", "y", "width", "height", "mask", "origin_image","origin_face","new_face","new_width","new_height")
    INPUT_IS_LIST = (
        True,
        False,
    )
    FUNCTION = "extract"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从边界框信息中提取人脸坐标、尺寸和图像"

    def extract(self, bounding_info):

        

        # 确保bounding_info是字典类型
        if not isinstance(bounding_info, list) and len(bounding_info) <= 0:
            raise Exception(f"边界框信息不是预期的列表格式: {type(bounding_info)}")
        
        if len(bounding_info) > 0:
            bounding_info = bounding_info[0]

        # 从bounding_info中提取信息
        x = bounding_info.get("x", 0)
        y = bounding_info.get("y", 0)
        width = bounding_info.get("width", 0)
        height = bounding_info.get("height", 0)
        mask = bounding_info.get("mask", None)
        origin_face = bounding_info.get("origin_face", None)
        origin_image = bounding_info.get("origin_image", None)
        new_face = bounding_info.get("new_face", None)
        new_width = bounding_info.get("new_width", 0)
        new_height = bounding_info.get("new_height", 0)
        return (x, y, width, height, mask, origin_image,origin_face,new_face,new_width,new_height)
