import torch

_CATEGORY = "sfnodes/image"
_MAX_IMAGE_SLOTS = 16


def _ordered_pairs(kwargs, prefix):
    """按端口编号（数字，非字典序）收集有序 (名, 值)；None 槽跳过。

    `sorted(kwargs.keys())` 是字典序，槽数 >9 时 `image_10` 会排到 `image_2`
    之前，导致合并顺序错误——这里按 `prefix` 后的整数排序。
    """
    pairs = []
    for k, v in kwargs.items():
        if v is None or not k.startswith(prefix):
            continue
        suffix = k[len(prefix):]
        pairs.append((int(suffix) if suffix.isdigit() else 1 << 30, k, v))
    pairs.sort(key=lambda p: p[0])
    return [(k, v) for _, k, v in pairs]


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
        ordered = _ordered_pairs(kwargs, "image_")
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
        ordered = _ordered_pairs(kwargs, "mask_")
        tensors = [t.unsqueeze(0) if t.dim() == 2 else t for _, t in ordered]
        if not tensors:
            raise ValueError("SF Mask Batch: 至少需要一路输入遮罩")
        reference = tensors[0].shape[1:]
        mismatched = [i + 1 for i, t in enumerate(tensors) if t.shape[1:] != reference]
        if mismatched:
            raise ValueError(f"SF Mask Batch: 输入遮罩尺寸不一致: {mismatched}")
        return (torch.cat(tensors, dim=0),)
