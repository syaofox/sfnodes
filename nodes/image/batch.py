import torch

from ...sf_utils.common import ordered_slot_items

_CATEGORY = "sfnodes/image"
_MAX_IMAGE_SLOTS = 16


class SFImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_IMAGE_SLOTS + 1):
            optional[f"image_{i}"] = ("IMAGE",)
        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将多路图像沿批次维（batch 维）合并为一个大批次，输入端口随连接自动增减，要求各路 H/W/C 尺寸一致"

    @staticmethod
    def _check_image_dimensions(tensors, names):
        reference = tensors[0].shape[1:]
        mismatched = [names[i] for i, t in enumerate(tensors) if t.shape[1:] != reference]
        if mismatched:
            raise ValueError(f"SF Image Batch: 输入图像尺寸不一致: {mismatched}")

    def execute(self, **kwargs):
        ordered = ordered_slot_items(kwargs, "image_")
        names = [k for k, _ in ordered]
        tensors = [v for _, v in ordered]

        if not tensors:
            raise ValueError("SF Image Batch: 至少需要一路输入图像")

        self._check_image_dimensions(tensors, names)
        return (torch.cat(tensors, dim=0),)


class SFMaskBatch:
    """沿批次维合并多路 MASK；未连接的端口跳过（分段数可少于端口数）。"""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_IMAGE_SLOTS + 1):
            optional[f"mask_{i}"] = ("MASK",)
        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将多路遮罩沿批次维合并为一个大批次，未连接的端口自动跳过（便于分段结果合并），要求各路 H/W 尺寸一致"

    def execute(self, **kwargs):
        ordered = ordered_slot_items(kwargs, "mask_")
        tensors = [t.unsqueeze(0) if t.dim() == 2 else t for _, t in ordered]
        if not tensors:
            raise ValueError("SF Mask Batch: 至少需要一路输入遮罩")
        reference = tensors[0].shape[1:]
        mismatched = [i + 1 for i, t in enumerate(tensors) if t.shape[1:] != reference]
        if mismatched:
            raise ValueError(f"SF Mask Batch: 输入遮罩尺寸不一致: {mismatched}")
        return (torch.cat(tensors, dim=0),)
