import re
import numpy as np
import torch
from PIL import ImageFilter
from scipy.ndimage import binary_closing, binary_fill_holes
import kornia
import torch.nn.functional as F
from .image_convert import pil2tensor, tensor2pil

def combine_mask(destination, source, x, y):
    output = destination.reshape(
        (-1, destination.shape[-2], destination.shape[-1])
    ).clone()
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
    return torch.full((1, height, width), value, dtype=torch.float32, device="cpu")


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
    target_mask = target_mask.reshape(
        (-1, target_mask.shape[-2], target_mask.shape[-1])
    )
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


def binary_dilation(mask, radius: int):
    kernel = torch.ones(1, radius * 2 + 1, device=mask.device)
    mask = kornia.filters.filter2d_separable(
        mask, kernel, kernel, border_type="constant"
    )
    mask = (mask > 0).to(mask.dtype)
    return mask


def make_odd(x):
    if x > 0 and x % 2 == 0:
        return x + 1
    return x


def gaussian_blur(image, radius: int, sigma: float = 0):
    c = image.shape[-3]
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8
    return kornia.filters.gaussian_blur2d(image, (radius, radius), (sigma, sigma))


def binary_erosion(mask, radius: int):
    kernel = torch.ones(1, 1, radius * 2 + 1, radius * 2 + 1, device=mask.device)
    mask = F.pad(mask, (radius, radius, radius, radius), mode="constant", value=1)
    mask = F.conv2d(mask, kernel, groups=1)
    mask = (mask == kernel.numel()).to(mask.dtype)
    return mask


def expand_mask_edges(mask, grow_top, grow_bottom, grow_left, grow_right, grow_tapered):
    """
    对四个边分别进行扩张/收缩操作
    
    参数:
        mask: 输入遮罩
        grow_top: 上边增长值
        grow_bottom: 下边增长值
        grow_left: 左边增长值
        grow_right: 右边增长值
        grow_tapered: 是否使用锥形角
    """
    import scipy
    
    if grow_top == 0 and grow_bottom == 0 and grow_left == 0 and grow_right == 0:
        return mask
    
    c = 0 if grow_tapered else 1
    base_kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
    device = mask.device
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
    out = []
    
    for m in mask:
        output = m.numpy()
        height, width = output.shape
        
        # 分别处理每个边，使用定向的形态学操作
        # 对于每个边，我们创建一个只影响该边的区域mask，然后在该区域内应用形态学操作
        
        # 处理上边
        if grow_top != 0:
            edge_width = abs(grow_top) + 1
            top_region = output[:min(height, edge_width), :].copy()
            for _ in range(abs(grow_top)):
                if grow_top < 0:
                    top_region = scipy.ndimage.grey_erosion(top_region, footprint=base_kernel)
                else:
                    top_region = scipy.ndimage.grey_dilation(top_region, footprint=base_kernel)
            output[:min(height, edge_width), :] = top_region
        
        # 处理下边
        if grow_bottom != 0:
            edge_width = abs(grow_bottom) + 1
            bottom_start = max(0, height - edge_width)
            bottom_region = output[bottom_start:, :].copy()
            for _ in range(abs(grow_bottom)):
                if grow_bottom < 0:
                    bottom_region = scipy.ndimage.grey_erosion(bottom_region, footprint=base_kernel)
                else:
                    bottom_region = scipy.ndimage.grey_dilation(bottom_region, footprint=base_kernel)
            output[bottom_start:, :] = bottom_region
        
        # 处理左边
        if grow_left != 0:
            edge_width = abs(grow_left) + 1
            left_region = output[:, :min(width, edge_width)].copy()
            for _ in range(abs(grow_left)):
                if grow_left < 0:
                    left_region = scipy.ndimage.grey_erosion(left_region, footprint=base_kernel)
                else:
                    left_region = scipy.ndimage.grey_dilation(left_region, footprint=base_kernel)
            output[:, :min(width, edge_width)] = left_region
        
        # 处理右边
        if grow_right != 0:
            edge_width = abs(grow_right) + 1
            right_start = max(0, width - edge_width)
            right_region = output[:, right_start:].copy()
            for _ in range(abs(grow_right)):
                if grow_right < 0:
                    right_region = scipy.ndimage.grey_erosion(right_region, footprint=base_kernel)
                else:
                    right_region = scipy.ndimage.grey_dilation(right_region, footprint=base_kernel)
            output[:, right_start:] = right_region
        
        output = torch.from_numpy(output)
        out.append(output)
    
    return torch.stack(out, dim=0).to(device)


def blur_mask_edges(mask, blur_top, blur_bottom, blur_left, blur_right):
    """
    对四个边分别进行模糊操作
    
    参数:
        mask: 输入遮罩
        blur_top: 上边模糊值
        blur_bottom: 下边模糊值
        blur_left: 左边模糊值
        blur_right: 右边模糊值
    """
    if blur_top == 0 and blur_bottom == 0 and blur_left == 0 and blur_right == 0:
        return mask
    
    device = mask.device
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
    out = []
    
    from PIL import Image
    
    for m in mask:
        output = m.numpy()
        height, width = output.shape
        
        # 转换为PIL图像
        pil_mask = Image.fromarray((output * 255).astype(np.uint8), mode='L')
        
        # 计算最大模糊值
        max_blur = max(blur_top, blur_bottom, blur_left, blur_right)
        if max_blur <= 0:
            out.append(torch.from_numpy(output))
            continue
        
        # 创建模糊半径图（使用向量化操作提高效率）
        blur_radius_map = np.zeros((height, width), dtype=np.float32)
        
        # 上边模糊半径（从顶部向中心递减）
        if blur_top > 0:
            edge_width = min(height, blur_top * 2 + 1)
            i_indices = np.arange(edge_width)
            weights = 1.0 - (i_indices / max(1, edge_width - 1))
            blur_radius_map[:edge_width, :] = np.maximum(blur_radius_map[:edge_width, :], 
                                                          weights[:, np.newaxis] * blur_top)
        
        # 下边模糊半径（从底部向中心递减）
        if blur_bottom > 0:
            edge_width = min(height, blur_bottom * 2 + 1)
            bottom_start = max(0, height - edge_width)
            i_indices = np.arange(edge_width)
            weights = 1.0 - (i_indices / max(1, edge_width - 1))
            blur_radius_map[bottom_start:, :] = np.maximum(blur_radius_map[bottom_start:, :], 
                                                             weights[::-1, np.newaxis] * blur_bottom)
        
        # 左边模糊半径（从左侧向中心递减）
        if blur_left > 0:
            edge_width = min(width, blur_left * 2 + 1)
            j_indices = np.arange(edge_width)
            weights = 1.0 - (j_indices / max(1, edge_width - 1))
            blur_radius_map[:, :edge_width] = np.maximum(blur_radius_map[:, :edge_width], 
                                                          weights[np.newaxis, :] * blur_left)
        
        # 右边模糊半径（从右侧向中心递减）
        if blur_right > 0:
            edge_width = min(width, blur_right * 2 + 1)
            right_start = max(0, width - edge_width)
            j_indices = np.arange(edge_width)
            weights = 1.0 - (j_indices / max(1, edge_width - 1))
            blur_radius_map[:, right_start:] = np.maximum(blur_radius_map[:, right_start:], 
                                                            weights[::-1, np.newaxis] * blur_right)
        
        # 使用多个模糊级别并混合
        unique_radii = sorted(set([int(r) for r in blur_radius_map.flatten() if r > 0]))
        if not unique_radii:
            out.append(torch.from_numpy(output))
            continue
        
        # 预计算不同模糊级别的结果
        blurred_cache = {}
        for radius in unique_radii:
            if radius > 0:
                blurred = pil_mask.filter(ImageFilter.GaussianBlur(radius=radius))
                blurred_cache[radius] = np.array(blurred).astype(np.float32) / 255.0
        
        # 应用渐变模糊（使用向量化操作）
        result = output.copy()
        mask_needs_blur = blur_radius_map > 0
        
        if mask_needs_blur.any():
            # 对需要模糊的区域进行处理
            for radius_int in unique_radii:
                # 找到需要这个模糊级别的像素
                radius_mask = (blur_radius_map >= radius_int) & (blur_radius_map < radius_int + 1)
                if radius_mask.any():
                    if radius_int in blurred_cache:
                        if radius_int + 1 in blurred_cache:
                            # 在两个级别之间插值
                            frac = blur_radius_map - radius_int
                            result[radius_mask] = (
                                blurred_cache[radius_int][radius_mask] * (1 - frac[radius_mask]) +
                                blurred_cache[radius_int + 1][radius_mask] * frac[radius_mask]
                            )
                        else:
                            result[radius_mask] = blurred_cache[radius_int][radius_mask]
                    elif radius_int + 1 in blurred_cache:
                        result[radius_mask] = blurred_cache[radius_int + 1][radius_mask]
        
        output = torch.from_numpy(result)
        out.append(output)
    
    return torch.stack(out, dim=0).to(device)


def mask_process(mask, mask_params=None, unqueeze=True):


    if mask_params is None:
        if unqueeze:
            mask = mask.squeeze(0).unsqueeze(-1)
        return mask


    pre_invert = mask_params["pre_invert"]
    fill = mask_params["fill"]
    invert = mask_params["invert"]
    grow_tapered = mask_params.get("grow_tapered", False)

    if pre_invert:
        mask = 1 - mask

    # 检查是否使用四个边参数
    use_edges = ("grow_top" in mask_params or "grow_bottom" in mask_params or 
                 "grow_left" in mask_params or "grow_right" in mask_params or
                 "blur_top" in mask_params or "blur_bottom" in mask_params or
                 "blur_left" in mask_params or "blur_right" in mask_params)
    
    if use_edges:
        # 使用四个边分别处理
        # 获取增长参数
        grow_top = mask_params.get("grow_top", 0)
        grow_top_percent = mask_params.get("grow_top_percent", 0.0)
        grow_bottom = mask_params.get("grow_bottom", 0)
        grow_bottom_percent = mask_params.get("grow_bottom_percent", 0.0)
        grow_left = mask_params.get("grow_left", 0)
        grow_left_percent = mask_params.get("grow_left_percent", 0.0)
        grow_right = mask_params.get("grow_right", 0)
        grow_right_percent = mask_params.get("grow_right_percent", 0.0)
        
        # 计算每个边的实际增长值
        height, width = mask.shape[-2], mask.shape[-1]
        grow_top_count = int(grow_top_percent * height) + grow_top
        grow_bottom_count = int(grow_bottom_percent * height) + grow_bottom
        grow_left_count = int(grow_left_percent * width) + grow_left
        grow_right_count = int(grow_right_percent * width) + grow_right
        
        if grow_top_count != 0 or grow_bottom_count != 0 or grow_left_count != 0 or grow_right_count != 0:
            mask = expand_mask_edges(mask, grow_top_count, grow_bottom_count, 
                                     grow_left_count, grow_right_count, grow_tapered)
        
        # 获取模糊参数
        blur_top = mask_params.get("blur_top", 0)
        blur_top_percent = mask_params.get("blur_top_percent", 0.0)
        blur_bottom = mask_params.get("blur_bottom", 0)
        blur_bottom_percent = mask_params.get("blur_bottom_percent", 0.0)
        blur_left = mask_params.get("blur_left", 0)
        blur_left_percent = mask_params.get("blur_left_percent", 0.0)
        blur_right = mask_params.get("blur_right", 0)
        blur_right_percent = mask_params.get("blur_right_percent", 0.0)
        
        # 计算每个边的实际模糊值
        blur_top_count = int(blur_top_percent * height) + blur_top
        blur_bottom_count = int(blur_bottom_percent * height) + blur_bottom
        blur_left_count = int(blur_left_percent * width) + blur_left
        blur_right_count = int(blur_right_percent * width) + blur_right
        
        if blur_top_count != 0 or blur_bottom_count != 0 or blur_left_count != 0 or blur_right_count != 0:
            mask = blur_mask_edges(mask, blur_top_count, blur_bottom_count, 
                                   blur_left_count, blur_right_count)
    else:
        # 使用原有的全局参数（向后兼容）
        grow = mask_params.get("grow", 0)
        grow_percent = mask_params.get("grow_percent", 0.0)
        blur = mask_params.get("blur", 0)
        blur_percent = mask_params.get("blur_percent", 0.0)
        
        grow_count = int(grow_percent * max(mask.shape)) + grow
        if grow_count != 0:
            mask = expand_mask(mask, grow_count, grow_tapered)
        
        blur_count = int(blur_percent * max(mask.shape)) + blur 
        if blur_count > 0:
            mask = blur_mask(mask, blur_count)

    if fill:
        mask = fill_holes(mask)

    if invert:
        mask = 1 - mask

    if unqueeze:
        mask = mask.squeeze(0).unsqueeze(-1)
    return mask


