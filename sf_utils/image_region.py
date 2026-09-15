"""参考图区域中和纯逻辑（numpy + PIL，无 torch/ComfyUI 依赖）。

把图像指定区域（典型：人脸）做模糊/均值/填充中和，使 Krea2 参考 latent 在该区域
不再携带原人物细节；用于在保留姿势/背景/服装的同时把面部交给角色 LoRA。
见 experience/nodes-image.md §74。

模糊复用 sf_utils/inpaint_helpers.gaussian_blur_np（惰性 import，保持本模块顶层
无 torch 依赖，可直接 numpy 单测）。
"""

import numpy as np

_FEATHER_MAX_PX_RATIO_DEFAULT = 0.05


def _blur01(arr, px):
    """对 [0,1] 的 2D 或 HWC 数组做 PIL 高斯模糊（复用 inpaint_helpers 单源实现）。"""
    a = np.clip(arr, 0.0, 1.0).astype(np.float32)
    if px <= 0:
        return a
    from .inpaint_helpers import gaussian_blur_np
    if a.ndim == 2:
        return gaussian_blur_np(a, px)
    return np.stack([gaussian_blur_np(a[..., k], px) for k in range(a.shape[-1])], axis=-1)


def neutralize_region(image, mask, mode="blur", strength=0.75, blur_radius=48,
                      feather=0.05, fill_value=0.5):
    """中和 image 中 mask 覆盖的区域。

    image: (H,W,C) 或 (B,H,W,C)，float [0,1]。
    mask:  (H,W) 或 (B,H,W)，float [0,1]（软遮罩亦可）。
    mode:  "blur" 区域高斯模糊 | "mean" 区域均值填充 | "fill" 常量填充。
    strength: 0=不变，1=完全替换为中和结果（按羽化后的 mask 混合）。
    feather:  羽化宽度，相对 max(H,W) 的比例。
    fill_value: fill 模式的常量值。

    返回与 image 同形状、同 (3D/4D) 的 float32 结果。
    """
    img = np.asarray(image, dtype=np.float32)
    single = img.ndim == 3
    if single:
        img = img[None, ...]
    if img.ndim != 4:
        raise ValueError(f"neutralize_region: image must be (H,W,C) or (B,H,W,C), got {img.shape}")

    b, h, w, _ = img.shape
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim != 3 or m.shape[1] != h or m.shape[2] != w:
        raise ValueError(
            f"neutralize_region: mask must be (H,W)/(B,H,W) matching image HxW={h}x{w}, got {m.shape}"
        )

    strength = float(min(1.0, max(0.0, strength)))
    if strength <= 0.0:
        out = np.clip(img, 0.0, 1.0).astype(np.float32)
        return out[0] if single else out
    mode = str(mode or "blur").lower()
    feather_px = max(0.0, float(feather)) * max(h, w)

    out = img.copy()
    for i in range(b):
        mi = np.clip(m[i if i < m.shape[0] else m.shape[0] - 1], 0.0, 1.0)
        alpha = _blur01(mi, feather_px) * strength
        if float(alpha.max()) <= 0.0:
            continue
        a3 = alpha[..., None]
        frame = img[i]
        if mode == "mean":
            wsum = float(alpha.sum())
            region_mean = (frame * a3).sum(axis=(0, 1)) / wsum if wsum > 0.0 \
                else frame.mean(axis=(0, 1))
            base = np.broadcast_to(region_mean, frame.shape)
        elif mode == "fill":
            base = np.full_like(frame, float(fill_value))
        else:  # blur
            base = _blur01(frame, blur_radius)
        out[i] = frame * (1.0 - a3) + base * a3

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out[0] if single else out
