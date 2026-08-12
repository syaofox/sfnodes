import numpy as np


def rgb_to_lab(rgb):
    """sRGB → CIELAB（纯 numpy，无 cv2 依赖）。

    rgb: uint8 或 [0,1] float 数组，末维为 RGB 通道
    返回: float32 数组，L 0-100，a/b 约 -128~127
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    def f(t):
        return np.where(t > 0.04045, ((t + 0.055) / 1.055) ** 2.4, t / 12.92)

    r_, g_, b_ = f(r), f(g), f(b)

    x = 0.4124564 * r_ + 0.3575761 * g_ + 0.1804375 * b_
    y = 0.2126729 * r_ + 0.7151522 * g_ + 0.0721750 * b_
    z = 0.0193339 * r_ + 0.1191920 * g_ + 0.9503041 * b_
    x = x / 0.95047
    z = z / 1.08883

    def g(t):
        delta = 6.0 / 29.0
        return np.where(
            t > delta ** 3, t ** (1.0 / 3.0), t / (3.0 * delta ** 2) + 4.0 / 29.0
        )

    fx, fy, fz = g(x), g(y), g(z)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([l, a, b], axis=-1)


def estimate_skin_color(image, mask, min_ratio=0.05):
    """估计图像 mask 区域内的近似肤色（LAB 肤色过滤后的像素均值）。

    肤色阈值由 imitation_hue.is_skin_or_lips 的 cv2 量化值等值换算：
    L∈(7.8,98)、a∈(-8,52)、b∈(-8,62)。

    参数:
        image: uint8 RGB 数组 [H,W,3]
        mask: 0-1 数组 [H,W]（多余维度会被压缩）
        min_ratio: 肤色像素占 mask 面积的最小比例，不足则回退区域全像素均值

    返回:
        float32 [0,1] (3,)；mask 区域为空时返回 None
    """
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    region = mask > 0.5
    if not region.any():
        return None

    pixels = np.asarray(image)[region]
    lab = rgb_to_lab(pixels)

    skin = (
        (lab[..., 0] > 7.8)
        & (lab[..., 0] < 98.0)
        & (lab[..., 1] > -8.0)
        & (lab[..., 1] < 52.0)
        & (lab[..., 2] > -8.0)
        & (lab[..., 2] < 62.0)
    )
    if skin.sum() >= max(1, int(region.sum() * min_ratio)):
        color = pixels[skin].mean(axis=0)
    else:
        color = pixels.mean(axis=0)
    return (color / 255.0).astype(np.float32)
