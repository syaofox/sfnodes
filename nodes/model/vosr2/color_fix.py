"""Torch-native 色彩对齐 — 移植自 VOSR 的 inference_vosr_onestep.py（Apache-2.0），
经 ylchen333/ComfyUI-VOSR2 的 color.py 改写为纯张量实现（无 PIL/uint8 量化、无 cv2）。

上游用 PIL 往返 + cv2.GaussianBlur；这里直接对 BCHW float [0,1] 张量操作，
可分离高斯核替代 cv2.GaussianBlur。`downsample` 档（TE-Speed-VOSR2 的做法）：
色彩是低频信号，先在 1/N 分辨率上模糊再上采样回原尺寸，高频细节仍取 SR 输出。
"""
import math

import torch
import torch.nn.functional as F

ADAIN_EPS = 1e-5
WAVELET_SIGMA = 5.0


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.to(dtype)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """可分离高斯模糊（reflect padding，逐通道）。x: (B, C, H, W)。

    reflect padding 要求 pad 宽度小于该维尺寸，故每个轴的核半径按尺寸钳制——
    只会收窄（本已高斯衰减的）核尾，不会报错。
    """
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    desired_radius = max(int(math.ceil(sigma * 4.0)), 1)
    radius_h = max(min(desired_radius, h - 1), 0)
    radius_w = max(min(desired_radius, w - 1), 0)

    if radius_w > 0:
        kernel_h = _gaussian_kernel1d(sigma, radius_w, x.device, x.dtype).view(1, 1, 1, -1).expand(c, 1, 1, -1)
        x = F.pad(x, (radius_w, radius_w, 0, 0), mode="reflect")
        x = F.conv2d(x, kernel_h, groups=c)
    if radius_h > 0:
        kernel_v = _gaussian_kernel1d(sigma, radius_h, x.device, x.dtype).view(1, 1, -1, 1).expand(c, 1, -1, 1)
        x = F.pad(x, (0, 0, radius_h, radius_h), mode="reflect")
        x = F.conv2d(x, kernel_v, groups=c)
    return x


def adain_color_fix(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """把 target 的逐通道空间均值/标准差对齐到 source。两者 (B, C, H, W) ∈ [0, 1]。"""
    target_mean = target.mean(dim=[2, 3], keepdim=True)
    target_std = target.std(dim=[2, 3], keepdim=True) + ADAIN_EPS
    source_mean = source.mean(dim=[2, 3], keepdim=True)
    source_std = source.std(dim=[2, 3], keepdim=True) + ADAIN_EPS
    result = (target - target_mean) / target_std * source_std + source_mean
    return torch.clamp(result, 0.0, 1.0)


def wavelet_color_fix(target: torch.Tensor, source: torch.Tensor, downsample: int = 1) -> torch.Tensor:
    """用 source 的低频 + target 的高频合成。两者 (B, C, H, W) ∈ [0, 1]。"""
    downsample = max(1, int(downsample))
    if downsample > 1:
        small = (max(1, target.shape[2] // downsample), max(1, target.shape[3] // downsample))
        source_small = F.interpolate(source, size=small, mode="area")
        target_small = F.interpolate(target, size=small, mode="area")
        source_low = gaussian_blur(source_small, WAVELET_SIGMA)
        target_low = gaussian_blur(target_small, WAVELET_SIGMA)
        size = target.shape[-2:]
        source_low = F.interpolate(source_low, size=size, mode="bilinear", align_corners=False)
        target_low = F.interpolate(target_low, size=size, mode="bilinear", align_corners=False)
    else:
        source_low = gaussian_blur(source, WAVELET_SIGMA)
        target_low = gaussian_blur(target, WAVELET_SIGMA)
    target_high = target - target_low
    result = source_low + target_high
    return torch.clamp(result, 0.0, 1.0)


def apply_color_alignment(decoded: torch.Tensor, reference: torch.Tensor, mode: str, downsample: int = 1) -> torch.Tensor:
    """decoded/reference: (B, C, H, W) float ∈ [0, 1]，空间尺寸一致。"""
    if mode == "none":
        return torch.clamp(decoded, 0.0, 1.0)
    if mode == "adain":
        return adain_color_fix(decoded, reference)
    if mode == "wavelet":
        return wavelet_color_fix(decoded, reference, downsample=downsample)
    raise ValueError(f"Unknown color_alignment mode: {mode!r}")
