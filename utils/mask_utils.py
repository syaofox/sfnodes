import numpy as np
import torch
from PIL import ImageFilter
from scipy.ndimage import binary_closing, binary_fill_holes

from .image_convert import pil2tensor, tensor2pil


def combine_mask(destination, source, x, y):
    output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
    source = source.reshape((-1, source.shape[-2], source.shape[-1]))

    left, top = (
        x,
        y,
    )
    right, bottom = (
        min(left + source.shape[-1], destination.shape[-1]),
        min(top + source.shape[-2], destination.shape[-2]),
    )
    visible_width, visible_height = (
        right - left,
        bottom - top,
    )

    source_portion = source[:, :visible_height, :visible_width]
    destination_portion = destination[:, top:bottom, left:right]

    # operation == "subtract":
    output[:, top:bottom, left:right] = destination_portion - source_portion

    output = torch.clamp(output, 0.0, 1.0)

    return output


def grow_mask(mask, expand, tapered_corners):
    if expand == 0:
        return mask

    device = mask.device
    mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))  # 添加通道维度

    # 创建卷积核
    c = 0.0 if tapered_corners else 1.0
    kernel = torch.tensor([[c, 1.0, c], [1.0, 1.0, 1.0], [c, 1.0, c]], device=device).unsqueeze(0).unsqueeze(0)

    # 计算填充
    padding = abs(expand)

    if expand > 0:
        # 膨胀操作
        mask = torch.nn.functional.pad(mask, (padding, padding, padding, padding), mode='constant', value=0)
        mask = torch.nn.functional.conv2d(mask, kernel, padding=1, dilation=expand)
    else:
        # 腐蚀操作
        mask = 1 - mask
        mask = torch.nn.functional.pad(mask, (padding, padding, padding, padding), mode='constant', value=1)
        mask = torch.nn.functional.conv2d(mask, kernel, padding=1, dilation=-expand)
        mask = 1 - mask

    # 移除额外的padding
    if padding > 0:
        mask = mask[:, :, padding:-padding, padding:-padding]

    # 将结果转回原始形状
    output = mask.squeeze(1)
    return torch.clamp(output, 0.0, 1.0)


def fill_holes(mask):
    holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
    out = []
    for m in holemask:
        mask_np = m.numpy()
        binary_mask = mask_np > 0
        struct = np.ones((5, 5))
        closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
        filled_mask = binary_fill_holes(closed_mask)
        output = filled_mask.astype(np.float32) * 255  # type: ignore
        output = torch.from_numpy(output)
        out.append(output)
    mask = torch.stack(out, dim=0)
    mask = torch.clamp(mask, 0.0, 1.0)
    return mask


def invert_mask(mask):
    return 1.0 - mask


def expand_mask(mask, expand, tapered_corners):
    import scipy
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
    device = mask.device
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)

    return torch.stack(out, dim=0).to(device)


def blur_mask(mask, radius):
    pil_image = tensor2pil(mask)
    return pil2tensor(pil_image.filter(ImageFilter.GaussianBlur(radius)))


def solid_mask(width, height, value=1):
    return torch.full((1, height, width), value, dtype=torch.float32, device='cpu')


def mask_floor(mask, threshold: float = 0.99):
    # 将遮罩二值化，大于等于阈值的设为1，小于阈值的设为0
    return (mask >= threshold).to(mask.dtype)


def mask_unsqueeze(mask):
    # 调整遮罩的维度，确保输出的遮罩形状为 B1HW
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def apply_mask_area(target_mask, area_mask, paint_value=1.0):
    """
    根据区域mask将目标mask的特定区域涂成指定的值（黑或白）
    
    参数:
        target_mask: 目标遮罩，将被修改
        area_mask: 区域遮罩，定义要修改的区域
        paint_value: 要应用的值，0.0表示涂黑，1.0表示涂白
        
    返回:
        修改后的遮罩
    """
    # 确保mask形状正确
    target_mask = target_mask.reshape((-1, target_mask.shape[-2], target_mask.shape[-1]))
    area_mask = area_mask.reshape((-1, area_mask.shape[-2], area_mask.shape[-1]))
    
    # 确保两个mask尺寸相同
    if target_mask.shape[-2:] != area_mask.shape[-2:]:
        raise ValueError("目标遮罩和区域遮罩的尺寸必须相同")
    
    # 创建修改后的mask副本
    result_mask = target_mask.clone()
    
    # 将area_mask中非零位置的值在target_mask中设置为指定值
    # 使用布尔索引进行操作
    area_bool = area_mask > 0.5  # 将area_mask二值化
    result_mask[area_bool] = paint_value
    
    return result_mask
