"""三点色彩匹配纯逻辑（无 ComfyUI/torch 依赖，可独立测试）。

SFImageColorMatchByPoints 的算法核心：从目标图与参考图各提取三个关键色
（暗部/灰部/亮部，按亮度分位自动定位），逐通道构建三点分段线性映射 LUT
（类似 PS 曲线/色阶的黑灰白场吸管），再应用到目标图。

约定：
- 图像数组 [H, W, 3]，float32，值域 [0, 1]（ComfyUI 标准）
- 颜色数组 [3]（R, G, B）
"""
import numpy as np

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def extract_points(image, dark_percentile=5.0, mid_percentile=50.0, light_percentile=95.0, band_frac=0.25):
    """按亮度分位提取暗/灰/亮三点。

    每个点取分位附近区间内所有像素的 RGB 均值（区间宽度 = 相邻分位差的
    band_frac 倍），既抗噪又保留图像真实颜色（而非每通道独立分位的合成色）。
    区间内无像素（如全图同亮度）时退回亮度最接近分位中心的单个像素。

    Args:
        image: [H, W, 3] float32 [0, 1]。非有限（NaN/Inf）像素被忽略；
            全图无非有限像素时返回黑色三点。
        dark_percentile / mid_percentile / light_percentile: 亮度分位（0-100）。
        band_frac: 采样区间占相邻分位差的比值。

    Returns:
        (dark, mid, light)，各为 [3] float32 颜色。
    """
    pixels = image.reshape(-1, 3)
    ok = np.isfinite(pixels).all(axis=1)
    if not ok.all():
        pixels = pixels[ok]
    if pixels.size == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    flat = (pixels * LUMA_WEIGHTS).sum(axis=1)
    q_dark = float(np.quantile(flat, dark_percentile / 100.0))
    q_mid = float(np.quantile(flat, mid_percentile / 100.0))
    q_light = float(np.quantile(flat, light_percentile / 100.0))

    ranges = [
        (q_dark, q_dark + (q_mid - q_dark) * band_frac),
        (q_mid - (q_mid - q_dark) * band_frac, q_mid + (q_light - q_mid) * band_frac),
        (q_light - (q_light - q_mid) * band_frac, q_light),
    ]

    points = []
    for lo, hi in ranges:
        lo = min(max(lo, 0.0), 1.0)
        hi = min(max(hi, 0.0), 1.0)
        sel = (flat >= lo) & (flat <= hi)
        if sel.any():
            points.append(pixels[sel].mean(axis=0))
        else:
            idx = int(np.argmin(np.abs(flat - (lo + hi) / 2.0)))
            points.append(pixels[idx])
    return points[0], points[1], points[2]


def build_lut(target_points, ref_points, levels=256):
    """逐通道三点分段线性映射 LUT。

    曲线经过点：0→0、目标暗点→参考暗点、目标灰点→参考灰点、目标亮点→
    参考亮点、1→1（逐通道独立）。x 轴按分位提取天然单调，仍做排序防御
    （手动构造非单调三点时退化为正确的单调函数）。

    Args:
        target_points / ref_points: (dark, mid, light) 各 [3] 颜色。
        levels: LUT 级数。

    Returns:
        [3, levels] float32，行对应 R/G/B。
    """
    t = np.stack(
        [np.zeros(3), target_points[0], target_points[1], target_points[2], np.ones(3)], axis=0
    )
    r = np.stack(
        [np.zeros(3), ref_points[0], ref_points[1], ref_points[2], np.ones(3)], axis=0
    )
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)

    # 防御非单调：逐通道按 x 排序，y 跟随（函数值不变，仅归一化输入顺序）
    order = np.argsort(t, axis=0)
    t = np.take_along_axis(t, order, axis=0)
    r = np.take_along_axis(r, order, axis=0)

    xs = np.linspace(0.0, 1.0, levels)
    lut = np.empty((3, levels), dtype=np.float32)
    for c in range(3):
        lut[c] = np.interp(xs, t[:, c], r[:, c]).astype(np.float32)
    return lut


def apply_lut(image, lut):
    """查表应用 LUT（逐通道取整索引）。

    Args:
        image: [H, W, 3] float32 [0, 1]。
        lut: [3, levels] float32。

    Returns:
        与 image 同形状的 float32 数组。
    """
    idx = np.clip(
        np.round(image * (lut.shape[1] - 1)), 0, lut.shape[1] - 1
    ).astype(np.int32)
    out = np.empty_like(image)
    for c in range(3):
        out[..., c] = lut[c][idx[..., c]]
    return out
