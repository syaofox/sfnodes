import torch
from PIL import Image, ImageFilter, ImageOps
import folder_paths
import random

from comfy.utils import common_upscale
from nodes import SaveImage
from .utils.image_convert import mask2tensor, np2tensor, tensor2mask
from .utils.mask_utils import blur_mask, combine_mask, expand_mask, fill_holes, grow_mask, invert_mask, apply_mask_area

_CATEGORY = 'sfnodes/masks'


class OutlineMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                'outer_width': (
                    'INT',
                    {'default': 10, 'min': 0, 'max': 16384, 'step': 1, 'tooltip': '设置外轮廓宽度，范围为0到16384，步长为1'},
                ),
                'inner_width': (
                    'INT',
                    {'default': 10, 'min': 0, 'max': 16384, 'step': 1, 'tooltip': '设置内轮廓宽度，范围为0到16384，步长为1'},
                ),
                'tapered_corners': ('BOOLEAN', {'default': True, 'tooltip': '是否使用锥形角'}),
            }
        }

    RETURN_TYPES = ('MASK',)

    FUNCTION = 'execute'

    CATEGORY = _CATEGORY
    DESCRIPTION = '给遮罩添加内外轮廓线'

    def execute(self, mask, outer_width, inner_width, tapered_corners):
        m1 = grow_mask(mask, outer_width, tapered_corners)
        m2 = grow_mask(mask, -inner_width, tapered_corners)

        m3 = combine_mask(m1, m2, 0, 0)

        return (m3,)


class CreateBlurredEdgeMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'width': ('INT', {'default': 1024, 'min': 0, 'max': 14096, 'step': 1}),
                'height': ('INT', {'default': 1024, 'min': 0, 'max': 14096, 'step': 1}),
                'border': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1}),
                'border_percent': (
                    'FLOAT',
                    {'default': 0.05, 'min': 0.0, 'max': 2.0, 'step': 0.01},
                ),
                'blur_radius': (
                    'INT',
                    {'default': 10, 'min': 0, 'max': 4096, 'step': 1},
                ),
                'blur_radius_percent': (
                    'FLOAT',
                    {'default': 0.00, 'min': 0.0, 'max': 2.0, 'step': 0.01},
                ),
            },
            'optional': {
                'image': ('IMAGE', {'tooltips': '如果未提供图像，将使用输入的宽度和高度创建一个白色图像。'}),
            },
        }

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据指定图片增加模糊边框'

    def execute(self, width, height, border, border_percent, blur_radius, blur_radius_percent, image=None):
        if image is not None:
            _, height, width, _ = image.shape

        # 计算边框宽度
        border_width = int(min(width, height) * border_percent + border)

        # 计算内部图像的尺寸
        inner_width = width - 2 * border_width
        inner_height = height - 2 * border_width

        # 创建内部白色图像
        inner_image = Image.new('RGB', (inner_width, inner_height), 'white')

        # 扩展图像，添加黑色边框
        image_with_border = ImageOps.expand(inner_image, border=border_width, fill='black')

        # 计算模糊半径
        blur_radius = int(min(width, height) * blur_radius_percent + blur_radius)

        # 应用高斯模糊
        blurred_image = image_with_border.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 转换为张量
        blurred_tensor = np2tensor(blurred_image)
        blurred_image = blurred_tensor.unsqueeze(0)

        return (tensor2mask(blurred_image),)


class MaskChange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                'grow': ('INT', {'default': 0, 'min': -4096, 'max': 4096, 'step': 1, 'tooltip': '设置生长值，范围为-4096到4096，步长为1'}),
                'grow_percent': (
                    'FLOAT',
                    {'default': 0.00, 'min': -2.0, 'max': 2.0, 'step': 0.01, 'tooltip': '设置生长百分比，范围为-2.0到2.0，步长为0.01'},
                ),
                'grow_tapered': ('BOOLEAN', {'default': False, 'tooltip': '是否使用锥形角'}),
                'blur': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1, 'tooltip': '设置模糊值，范围为0到4096，步长为1'}),
                'fill': ('BOOLEAN', {'default': False, 'tooltip': '是否填充孔洞'}),
            },
        }

    RETURN_TYPES = ('MASK', 'MASK')
    RETURN_NAMES = ('mask', 'inverted_mask')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '修改和处理遮罩'

    def execute(self, mask, grow, grow_percent, grow_tapered, blur, fill):
        grow_count = int(grow_percent * max(mask.shape)) + grow
        if grow_count != 0:  # 改为检查是否不等于0
            mask = expand_mask(mask, grow_count, grow_tapered)

        if fill:
            mask = fill_holes(mask)

        if blur > 0:
            mask = blur_mask(mask, blur)

        # mask = mask.squeeze(0).unsqueeze(-1)

        return (mask, invert_mask(mask))


class Depth2Mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image_depth': ('IMAGE',),
                'depth': (
                    'FLOAT',
                    {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'round': 0.001, 'display': 'number', 'tooltip': '设置深度阈值，范围为0.0到1.0，步长为0.01'},
                ),
            },
        }

    RETURN_TYPES = ('MASK', 'MASK')
    RETURN_NAMES = ('mask', 'mask_inverted')

    FUNCTION = 'execute'

    CATEGORY = _CATEGORY
    DESCRIPTION = '将深度图像转换为遮罩'

    def execute(self, image_depth, depth):
        def upscale(image, upscale_method, width, height):
            samples = image.movedim(-1, 1)
            s = common_upscale(samples, width, height, upscale_method, 'disabled')
            s = s.movedim(1, -1)
            return (s,)

        bs, height, width = image_depth.size()[0], image_depth.size()[1], image_depth.size()[2]

        mask1 = torch.zeros((bs, height, width))

        image_depth = upscale(image_depth, 'lanczos', width, height)[0]

        mask1 = (image_depth[..., 0] < depth).float()

        return mask1, 1.0 - mask1


class MaskScaleBy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                'scale_by': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 8.0, 'step': 0.01, 'tooltip': '设置缩放比例，范围为0.01到8.0，步长为0.01'}),
            }
        }

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'upscale'

    CATEGORY = _CATEGORY
    DESCRIPTION = '根据指定比例缩放遮罩'

    def upscale(self, mask, scale_by):
        image = mask2tensor(mask)
        samples = image.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = common_upscale(samples, width, height, 'lanczos', 'disabled')
        s = s.movedim(1, -1)
        return (tensor2mask(s),)


class MaskScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                'width': ('INT', {'default': 512, 'min': 0, 'max': 16384, 'step': 1}),
                'height': ('INT', {'default': 512, 'min': 0, 'max': 16384, 'step': 1}),
            }
        }

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'upscale'

    CATEGORY = _CATEGORY
    DESCRIPTION = '根据指定宽高缩放遮罩'

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

            s = common_upscale(samples, width, height, 'lanczos', 'disabled')
            s = s.movedim(1, -1)
        return (tensor2mask(s),)


class MaskPaintArea:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'target_mask': ('MASK', {'tooltip': '目标遮罩'}),
                'area_mask': ('MASK', {'tooltip': '区域遮罩'}),
                'paint_mode': (
                    ['白色 (1.0)', '黑色 (0.0)', '自定义值'],
                    {'default': '白色 (1.0)', 'tooltip': '选择涂黑或涂白模式'}
                ),
                'custom_value': (
                    'FLOAT',
                    {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'tooltip': '设置自定义值，范围为0.0到1.0，步长为0.01'}
                ),
            }
        }

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据区域遮罩对目标遮罩进行涂黑或涂白'

    def execute(self, target_mask, area_mask, paint_mode, custom_value):
        # 根据选择的模式确定要应用的值
        if paint_mode == '白色 (1.0)':
            paint_value = 1.0
        elif paint_mode == '黑色 (0.0)':
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
            'required': {
                'mask': ('MASK', {'tooltip': '输入遮罩'}),
                'gray_value': (
                    'FLOAT',
                    {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'tooltip': '设置灰度值，范围为0.0(黑)到1.0(白)'}
                ),
                'apply_to': (
                    ['整个遮罩', '仅非零区域'],
                    {'default': '仅非零区域', 'tooltip': '选择应用灰度值的区域'}
                ),
                'strength': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'tooltip': '调整强度，1.0为完全应用新灰度，0.0为保持原样'}
                )
            }
        }

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '将遮罩设置为指定的灰度值'

    def execute(self, mask, gray_value, apply_to, strength):
        import torch
        
        # 创建一个新的张量以避免修改原始数据
        result_mask = mask.clone()
        
        if apply_to == '整个遮罩':
            # 应用到整个遮罩
            if strength >= 1.0:
                # 直接设置为指定灰度值
                result_mask = torch.ones_like(mask) * gray_value
            else:
                # 根据强度混合原始遮罩和新灰度值
                result_mask = mask * (1.0 - strength) + (torch.ones_like(mask) * gray_value * strength)
        else:
            # 只应用到非零区域
            non_zero_mask = (mask > 0).float()
            
            if strength >= 1.0:
                # 直接设置非零区域为指定灰度值
                result_mask = torch.where(non_zero_mask > 0, torch.ones_like(mask) * gray_value, mask)
            else:
                # 根据强度混合原始遮罩和新灰度值，只在非零区域
                new_values = mask * (1.0 - strength) + (torch.ones_like(mask) * gray_value * strength)
                result_mask = torch.where(non_zero_mask > 0, new_values, mask)
        
        return (result_mask,)


class PreviewMask(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mask": ("MASK",), },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)