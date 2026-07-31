import math

import numpy as np
from comfy.utils import common_upscale


def get_target_size(rescale_mode, custom_megapixels):
    """根据缩放模式计算目标像素数，custom 使用自定义百万像素数"""
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


def mask_bbox(mask_pil):
    """查找遮罩非零区域的边界框，返回 (x_min, y_min, x_max, y_max)；无非零区域返回 None"""
    arr = np.asarray(mask_pil)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    non_zero = np.nonzero(arr)
    if len(non_zero[0]) == 0:
        return None
    y_min, y_max = int(non_zero[0].min()), int(non_zero[0].max())
    x_min, x_max = int(non_zero[1].min()), int(non_zero[1].max())
    return x_min, y_min, x_max, y_max


def calc_target_dimensions(width, height, target_size):
    """按目标像素数计算缩放后的尺寸，保持宽高比；target_size <= 0 时返回原尺寸"""
    if target_size <= 0:
        return width, height
    scale_factor = math.sqrt(target_size / (width * height))
    new_width = round(width * scale_factor)
    new_height = round(height * scale_factor)
    return new_width, new_height


def resize_tensor(tensor, width, height, method="lanczos"):
    """缩放张量：IMAGE [B, H, W, C] 或 MASK [B, H, W] 均可"""
    if tensor.ndim == 3:
        samples = tensor.unsqueeze(1)
        resized = common_upscale(samples, width, height, method, "disabled")
        return resized.squeeze(1)
    samples = tensor.movedim(-1, 1)
    resized = common_upscale(samples, width, height, method, "disabled")
    return resized.movedim(1, -1)
