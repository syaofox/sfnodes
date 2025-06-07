import math

from PIL import Image, ImageDraw, ImageFilter
from comfy.utils import common_upscale
from .utils.image_convert import np2tensor, pil2mask, pil2tensor, tensor2np, tensor2pil

_CATEGORY = "sfnodes/face_analysis"


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
                "is_square": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否将图像裁剪为正方形"},
                ),
            },
        }

    RETURN_TYPES = (
        "BOUNDINGINFOS",
        "IMAGES",
        "INT",
    )
    RETURN_NAMES = (
        "bounding_infos",
        "crop_images",
        "face_count",
    )
    OUTPUT_IS_LIST = (
        True,
        True,
        False,
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
        is_square=False,
    ):
        target_size = self._get_target_size(rescale_mode, custom_megapixels)

        img = image[0]

        pil_image = tensor2pil(img)

        faces, x_coords, y_coords, widths, heights = analysis_models.get_bbox(
            pil_image, padding, padding_percent
        )

        face_count = len(faces)
        if face_count == 0:
            raise Exception("未在图像中检测到人脸。")

        bounding_infos = []
        crop_images = []

        for i, face in enumerate(faces):
            scale_factor = 1

            # 如果需要正方形，调整宽度和高度
            if is_square:
                # 计算正方形边长，取宽高的最大值
                square_size = max(widths[i], heights[i])
                # 计算新的x和y坐标，使人脸居中
                center_x = x_coords[i] + widths[i] // 2
                center_y = y_coords[i] + heights[i] // 2
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
                x_coords[i] = new_x
                y_coords[i] = new_y
                widths[i] = square_size
                heights[i] = square_size

                # 重新裁剪人脸区域为正方形
                face_np = tensor2np(image[0])
                face_crop = face_np[
                    new_y : new_y + square_size, new_x : new_x + square_size
                ]
                face = np2tensor(face_crop).unsqueeze(0)

            if target_size > 0:
                scale_factor = math.sqrt(target_size / (widths[i] * heights[i]))
                new_width = round(widths[i] * scale_factor)
                new_height = round(heights[i] * scale_factor)
                scaled_face = self._rescale_image(face, new_width, new_height)
            else:
                scaled_face = face

            # 为每个人脸创建单独的 BOUNDINGINFO 结构
            bounding_info = {
                "x": x_coords[i],
                "y": y_coords[i],
                "width": widths[i],
                "height": heights[i],
                "scale_factor": scale_factor,
                "margin": margin,
                "margin_percent": margin_percent,
                "blur_radius": blur_radius,
            }
            crop_images.append(scaled_face)

            bounding_infos.append(bounding_info)

        return (
            bounding_infos,
            crop_images,
            face_count,
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
                "distination_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "paste"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将bounding_info中的人脸图像贴回原图"

    @staticmethod
    def create_soft_edge_mask(size, margin, blur_radius):
        mask = Image.new("L", size, 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(((0, 0), size), outline="black", width=margin)
        return mask.filter(ImageFilter.GaussianBlur(blur_radius))

    def paste(self, bounding_info, source_image, distination_image):
        # 从bounding_info中获取人脸图像和位置信息
        x = bounding_info["x"]
        y = bounding_info["y"]
        width = bounding_info["width"]
        height = bounding_info["height"]
        margin = bounding_info["margin"]
        margin_percent = bounding_info["margin_percent"]
        blur_radius = bounding_info["blur_radius"]

        destination = tensor2pil(distination_image[0])
        source = tensor2pil(source_image[0])

        # 如果源图像尺寸与目标区域尺寸不匹配，进行调整
        if source.width != width or source.height != height:
            source = source.resize((width, height), resample=Image.Resampling.LANCZOS)

        ref_size = max(source.width, source.height)
        margin_border = int(ref_size * margin_percent) + margin

        mask = self.create_soft_edge_mask(source.size, margin_border, blur_radius)

        position = (x, y)
        destination.paste(source, position, mask)

        return pil2tensor(destination), pil2mask(mask)


class ExtractBoundingBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bounding_infos": ("BOUNDINGINFOS",),
                "crop_images": ("IMAGES",),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "选择要解析的人脸索引",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "IMAGE", "BOUNDINGINFO")
    RETURN_NAMES = ("x", "y", "width", "height", "crop_image", "bounding_info")
    INPUT_IS_LIST = (
        True,
        False,
    )
    FUNCTION = "extract"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从边界框信息中提取指定索引的人脸坐标、尺寸和图像"

    def extract(self, bounding_infos, crop_images, index=0):
        # 确保bounding_infos是列表
        if not isinstance(bounding_infos, list):
            bounding_infos = [bounding_infos]

        # 确保index是整数
        if isinstance(index, list):
            if len(index) > 0:
                index = index[0]  # 如果index是列表，取第一个元素
            else:
                index = 0

        # 确保index在有效范围内
        if len(bounding_infos) == 0:
            raise Exception("边界框信息列表为空")

        if index >= len(bounding_infos):
            print(
                f"警告：索引 {index} 超出了bounding_infos的范围 {len(bounding_infos)}，使用默认索引0"
            )
            index = 0

        bounding_info = bounding_infos[index]

        # 确保bounding_info是字典类型
        if not isinstance(bounding_info, dict):
            raise Exception(f"边界框信息不是预期的字典格式: {type(bounding_info)}")

        # 从bounding_info中提取信息
        x = bounding_info.get("x", 0)
        y = bounding_info.get("y", 0)
        width = bounding_info.get("width", 0)
        height = bounding_info.get("height", 0)
        crop_image = crop_images[index]
        return (x, y, width, height, crop_image, bounding_info)
