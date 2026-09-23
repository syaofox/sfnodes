"""SFReferenceRegionNeutralize — 参考图区域中和（典型：Krea2 洗图时中和人脸）。

把参考图（如深度图）指定区域做模糊/均值/填充中和，使 Krea2 编码出的 reference
latent（同时也是初始 latent）在该区域不再携带原人物面部几何；姿势/背景/服装仍由
参考约束，脸交给角色 LoRA。配合可选的 SFRegionalLoRA 脸区加强使用。

纯逻辑在 sf_utils/image_region.py，见 experience/nodes-image.md §76。
"""

import numpy as np
import torch
import torch.nn.functional as F

from ...sf_utils import image_region as region

_CATEGORY = "sfnodes/image"

_MODES = ["blur", "mean", "fill"]


class SFReferenceRegionNeutralize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK", {
                    "tooltip": "要中和的区域（脸）。尺寸不符会 bilinear 缩放到图像大小；"
                               "批量小于图像时复用末帧",
                }),
                "mode": (_MODES, {
                    "default": "blur",
                    "tooltip": "blur=区域高斯模糊（保留大致深度/头姿，推荐）；"
                               "mean=用区域均值覆盖；fill=用 fill_value 常量覆盖",
                }),
                "strength": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0=原样不改；1=区域完全替换为中和结果",
                }),
                "blur_radius": ("INT", {
                    "default": 48, "min": 1, "max": 256,
                    "tooltip": "blur 模式的高斯模糊半径（像素）",
                }),
                "feather": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "遮罩边缘羽化宽度（相对图像长边比例），0=硬边界",
                }),
                "fill_value": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "fill 模式的常量值（其余模式忽略）",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "neutralize"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("参考图区域中和：把参考图指定区域（如人脸）模糊/均值/填充，"
                   "让 Krea2 参考 latent 在该区域不再携带原人物面部几何，"
                   "从而保留姿势/背景/服装而把脸交给角色 LoRA")

    def neutralize(self, image, mask, mode="blur", strength=0.75, blur_radius=48,
                   feather=0.05, fill_value=0.5):
        h, w = int(image.shape[1]), int(image.shape[2])
        m = self._fit_mask(mask, h, w)
        img_np = image.detach().cpu().numpy().astype(np.float32)
        m_np = m.detach().cpu().numpy().astype(np.float32)
        out = region.neutralize_region(
            img_np, m_np, mode=mode, strength=strength,
            blur_radius=int(blur_radius), feather=feather, fill_value=fill_value,
        )
        return (torch.from_numpy(out),)

    @staticmethod
    def _fit_mask(mask, h, w):
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[1] != h or m.shape[2] != w:
            m = F.interpolate(
                m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False,
            ).squeeze(1)
        return m
