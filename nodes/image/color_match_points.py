import numpy as np
import torch
import comfy.utils

from ...sf_utils.color_match_points import extract_points, build_lut, apply_lut

_CATEGORY = "sfnodes/image"


class ImageColorMatchByPoints:
    DESCRIPTION = (
        "通过三点（暗部/灰部/亮部）使目标图像的颜色匹配参考图像，类似 PS 曲线的黑/灰/白场吸管。"
        "三点按亮度分位自动提取（参考图多帧时逐帧提取后取平均），"
        "逐通道构建三点分段线性映射（LUT）应用到目标图；"
        "target_mask 限定应用区域（软混合）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE", {"tooltip": "目标图像（将被修改颜色）"}),
                "reference_image": ("IMAGE", {"tooltip": "参考图像（提供三点颜色，多帧时三点取平均）"}),
                "dark_percentile": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.5,
                        "max": 49.0,
                        "step": 0.5,
                        "tooltip": "暗点亮度分位（越小越暗）",
                    },
                ),
                "mid_percentile": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 1.0,
                        "max": 99.0,
                        "step": 0.5,
                        "tooltip": "灰点亮度分位，运行时自动夹在暗/亮分位之间",
                    },
                ),
                "light_percentile": (
                    "FLOAT",
                    {
                        "default": 99.5,
                        "min": 51.0,
                        "max": 99.5,
                        "step": 0.5,
                        "tooltip": "亮点亮度分位（越大越亮）",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "匹配强度，0=完全保留原图，1=完全匹配，>1=过度拉伸",
                    },
                ),
            },
            "optional": {
                "target_mask": ("MASK", {"tooltip": "目标图的遮罩，仅对遮罩区域应用色彩匹配（软混合）"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(
        self,
        target_image,
        reference_image,
        dark_percentile,
        mid_percentile,
        light_percentile,
        strength,
        target_mask=None,
    ):
        # 灰点分位夹在暗/亮之间（越界输入防呆）
        mid_percentile = min(max(mid_percentile, dark_percentile + 0.1), light_percentile - 0.1)

        ref_points = self._mean_points(reference_image, dark_percentile, mid_percentile, light_percentile)
        target_np = self._to_np(target_image)
        mask_np = self._prepare_mask(target_mask, target_np.shape) if target_mask is not None else None

        out = np.empty_like(target_np)
        for i in range(target_np.shape[0]):
            t_points = extract_points(target_np[i], dark_percentile, mid_percentile, light_percentile)
            lut = build_lut(t_points, ref_points)
            matched = apply_lut(target_np[i], lut)
            matched = strength * matched + (1.0 - strength) * target_np[i]
            if mask_np is not None:
                m = mask_np[i]
                matched = m * matched + (1.0 - m) * target_np[i]
            out[i] = np.clip(matched, 0.0, 1.0)

        return (torch.from_numpy(out).to(target_image.device),)

    def _to_np(self, tensor):
        """张量 → float32 [B,H,W,C] numpy（detach 防梯度泄漏、contiguous 防切片视图）。"""
        return tensor.float().detach().contiguous().cpu().numpy()

    def _mean_points(self, image, dark_percentile, mid_percentile, light_percentile):
        """参考图多帧时逐帧提取三点后平均（单帧时直接返回）。"""
        img_np = self._to_np(image)
        acc = None
        for i in range(img_np.shape[0]):
            stacked = np.stack(
                extract_points(img_np[i], dark_percentile, mid_percentile, light_percentile), axis=0
            )
            acc = stacked if acc is None else acc + stacked
        return tuple(acc / img_np.shape[0])

    def _prepare_mask(self, mask, target_shape):
        """对齐遮罩：加通道维、空间尺寸对齐目标、广播/截断 batch。返回 [B,H,W,1] float32。"""
        mask = mask.float().unsqueeze(1)
        if mask.shape[2:] != (target_shape[1], target_shape[2]):
            mask = comfy.utils.common_upscale(
                mask, target_shape[2], target_shape[1], upscale_method="bicubic", crop="center"
            )
        if mask.shape[0] < target_shape[0]:
            mask = torch.cat([mask, mask[-1:].repeat(target_shape[0] - mask.shape[0], 1, 1, 1)], dim=0)
        elif mask.shape[0] > target_shape[0]:
            mask = mask[: target_shape[0]]
        return np.expand_dims(mask[:, 0].float().detach().contiguous().cpu().numpy(), -1)
