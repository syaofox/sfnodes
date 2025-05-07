import math

import cv2
import numpy as np
import torch
from PIL import Image

from comfy.utils import common_upscale

from .utils.image_convert import mask2tensor, np2tensor, tensor2mask, tensor2np
from .utils.mask_utils import solid_mask


_CATEGORY = 'sfnodes/image scale'
UPSCALE_METHODS = ['lanczos', 'nearest-exact', 'bilinear', 'area', 'bicubic']

def make_even(number):
    _, remainder = divmod(number, 2)
    return number + remainder


class GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
            }
        }

    RETURN_TYPES = (
        'INT',
        'INT',
        'INT',
        'INT',
    )
    RETURN_NAMES = (
        'width',
        'height',
        'count',
        'min_dimension',
    )
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY

    def execute(self, image):
        return {
            'ui': {
                'width': (image.shape[2],),
                'height': (image.shape[1],),
                'count': (image.shape[0],),
            },
            'result': (
                image.shape[2],
                image.shape[1],
                image.shape[0],
                min(image.shape[2], image.shape[1]),
            ),
        }


class BaseImageScaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'upscale_method': (UPSCALE_METHODS,),
            },
            'optional': {
                'mask': ('MASK',),
            },
        }

    RETURN_TYPES = ('IMAGE', 'MASK', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('image', 'mask', 'width', 'height', 'min_dimension')
    CATEGORY = _CATEGORY

    def scale_image(self, image, width, height, upscale_method, mask=None):
        image_tensor = image.movedim(-1, 1)
        scaled_image = common_upscale(image_tensor, width, height, upscale_method, 'disabled')
        scaled_image = scaled_image.movedim(1, -1)

        result_mask = solid_mask(width, height)
        if mask is not None:
            mask_image = mask2tensor(mask)
            mask_image = mask_image.movedim(-1, 1)
            mask_image = common_upscale(mask_image, width, height, upscale_method, 'disabled')
            mask_image = mask_image.movedim(1, -1)
            result_mask = tensor2mask(mask_image)

        return scaled_image, result_mask

    def prepare_result(self, scaled_image, result_mask, width, height):
        return {
            'ui': {
                'width': (width,),
                'height': (height,),
            },
            'result': (
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
        base_inputs['required']['sd_model_type'] = (['sdxl', 'sd15', 'sd15+', 'sdxl+'], {'tooltip': '根据SD模型类型缩放图片到指定像素数，sd15为512x512，sd15+为512x768，sdxl为1024x1024，sdxl+为1024x1280'})
        return base_inputs

    FUNCTION = 'execute'
    DESCRIPTION = """
    根据SD模型类型缩放图片到指定像素数，sd15为512x512，sd15+为512x768，sdxl为1024x1024，sdxl+为1024x1280
    """

    def execute(self, image, upscale_method, sd_model_type, mask=None):
        sd_dimensions = {'sd15': (512, 512), 'sd15+': (512, 768), 'sdxl': (1024, 1024), 'sdxl+': (1024, 1280)}
        target_width, target_height = sd_dimensions.get(sd_model_type, (1024, 1024))
        total_pixels = target_width * target_height

        scale_by = math.sqrt(total_pixels / (image.shape[2] * image.shape[1]))
        width = round(image.shape[2] * scale_by)
        height = round(image.shape[1] * scale_by)

        scaled_image, result_mask = self.scale_image(image, width, height, upscale_method, mask)
        return self.prepare_result(scaled_image, result_mask, width, height)


class ImageScalerByPixels(BaseImageScaler):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()

        base_inputs['required'].update(
            {
                'total_pixels': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 16.0, 'step': 0.01, 'tooltip': '设置缩放比例，范围为0.01到16.0，步长为0.01'}),
                'limit': ('BOOLEAN', {'default': True, 'tooltip': '限制缩放比例，如果图像的像素数小于目标像素数，则不缩放图像'}),
            }
        )

        return base_inputs

    FUNCTION = 'execute'
    DESCRIPTION = """
    将图片缩放到指定像素数，total_pixels为缩放比例，limit为True时，如果图像的像素数小于目标像素数，则不缩放图像
    """

    def execute(self, image, upscale_method, total_pixels, limit=True, mask=None):
        samples = image.movedim(-1, 1)
        total = int(total_pixels * 1024 * 1024)
        current_pixels = samples.shape[3] * samples.shape[2]

        # Only upscale if current pixels is less than target total, when limit is True
        if limit and current_pixels <= total:
            result_mask = mask if mask is not None else solid_mask(image.shape[2], image.shape[1])
            return self.prepare_result(image, result_mask, image.shape[2], image.shape[1])

        scale_by = math.sqrt(total / current_pixels)
        # 计算缩放后的宽高 确保宽高为偶数
        width = make_even(round(samples.shape[3] * scale_by))
        height = make_even(round(samples.shape[2] * scale_by))

        scaled_image, result_mask = self.scale_image(image, width, height, upscale_method, mask)
        return self.prepare_result(scaled_image, result_mask, width, height)


class ImageScaleBySpecifiedSide(BaseImageScaler):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs['required'].update(
            {
                'size': ('INT', {'default': 512, 'min': 0, 'step': 1, 'max': 99999, 'tooltip': '设置缩放目标像素数，范围为0到99999，步长为1'}),
                'shorter': ('BOOLEAN', {'default': False, 'tooltip': '参照短边缩放'}),
                'limit': ('BOOLEAN', {'default': False, 'tooltip': '限制缩放比例，如果图像的最短边小于size，则不缩放图像'}),
            }
        )
        return base_inputs

    FUNCTION = 'execute'
    DESCRIPTION = """
    根据指定边长缩放图片，shorter为True时参照短边，否则参照长边
    limit为True时，如果图像的最短边小于size，则不缩放图像
    """

    def execute(self, image, size, upscale_method, shorter, limit, mask=None):
        # Check if we should skip scaling
        min_side = min(image.shape[2], image.shape[1])
        if limit and min_side < size:
            return self.prepare_result(image, mask if mask is not None else solid_mask(image.shape[2], image.shape[1]), image.shape[2], image.shape[1])

        if shorter:
            reference_side_length = min(image.shape[2], image.shape[1])
        else:
            reference_side_length = max(image.shape[2], image.shape[1])

        scale_by = reference_side_length / size
        width = make_even(round(image.shape[2] / scale_by))
        height = make_even(round(image.shape[1] / scale_by))

        scaled_image, result_mask = self.scale_image(image, width, height, upscale_method, mask)
        return self.prepare_result(scaled_image, result_mask, width, height)


class ComputeImageScaleRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'target_max_size': (
                    'INT',
                    {'default': 1920, 'min': 0, 'step': 1, 'max': 99999, 'tooltip': '设置目标最大尺寸，范围为0到99999，步长为1'},
                ),
            },
        }

    RETURN_TYPES = (
        'FLOAT',
        'INT',
        'INT',
    )
    RETURN_NAMES = (
        'rescale_ratio',
        'width',
        'height',
    )
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据引用图片的大小和目标最大尺寸，返回缩放比例和缩放后的宽高'

    def execute(self, image, target_max_size):
        samples = image.movedim(-1, 1)
        width, height = samples.shape[3], samples.shape[2]

        rescale_ratio = target_max_size / max(width, height)

        new_width = make_even(round(width * rescale_ratio))
        new_height = make_even(round(height * rescale_ratio))

        return {
            'ui': {
                'rescale_ratio': (rescale_ratio,),
                'width': (new_width,),
                'height': (new_height,),
            },
            'result': (
                rescale_ratio,
                new_width,
                new_height,
            ),
        }


class ImageRotate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image_from': ('IMAGE',),
                'angle': (
                    'FLOAT',
                    {'default': 0.1, 'min': -14096, 'max': 14096, 'step': 0.01, 'tooltip': '设置旋转角度，范围为-14096到14096，步长为0.01'},
                ),
                'expand': ('BOOLEAN', {'default': True, 'tooltip': '扩展图像尺寸'}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('rotated_image',)
    FUNCTION = 'run'
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
            rotated_image = cv2.warpAffine(image_np, rot_mat, (new_width, new_height), flags=cv2.INTER_CUBIC)
        else:
            # 不扩展图像尺寸的旋转
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image_np, rot_mat, (width, height), flags=cv2.INTER_CUBIC)

        # 转换回tensor格式
        rotated_tensor = np2tensor(rotated_image).unsqueeze(0)

        return (rotated_tensor,)


class TrimImageBorders:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'threshold': (
                    'INT',
                    {'default': 10, 'min': 0, 'max': 14096, 'step': 1, 'tooltip': '设置阈值，范围为0到14096，步长为1'},
                ),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    CATEGORY = _CATEGORY
    DESCRIPTION = '图片去黑边'

    def run(self, image, threshold):
        img = tensor2np(image[0])
        img = Image.fromarray(img)
        gray_image = img.convert('L')

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
            'required': {
                'image': ('IMAGE',),
                'border_width': ('INT', {'default': 10, 'min': 0, 'max': 1000, 'step': 1, 'tooltip': '设置边框宽度，范围为0到1000，步长为1'}),
                'border_ratio': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'tooltip': '设置边框比例，范围为0.0到1.0，步长为0.01'}),
                'r': ('INT', {'default': 0, 'min': 0, 'max': 255, 'step': 1}),
                'g': ('INT', {'default': 0, 'min': 0, 'max': 255, 'step': 1}),
                'b': ('INT', {'default': 0, 'min': 0, 'max': 255, 'step': 1}),
            }
        }

    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('bordered_image', 'border_mask')
    FUNCTION = 'add_border'
    CATEGORY = _CATEGORY
    DESCRIPTION = '给图片增加指定RGB颜色的边框,可以通过绝对像素值或相对比率设置边框宽度,并输出边框部分的mask'

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
        bordered_img[final_border_width : final_border_width + h, final_border_width : final_border_width + w] = img_np

        # 创建边框mask
        border_mask = np.ones((new_h, new_w), dtype=np.float32)
        border_mask[final_border_width : final_border_width + h, final_border_width : final_border_width + w] = 0

        # 将结果转换回 PyTorch 张量
        bordered_tensor = np2tensor(bordered_img).unsqueeze(0)
        mask_tensor = torch.from_numpy(border_mask).unsqueeze(0)

        return (bordered_tensor, mask_tensor)
