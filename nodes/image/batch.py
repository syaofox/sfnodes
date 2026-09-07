import torch

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
        keys = sorted(kwargs.keys())
        tensors = [kwargs[k] for k in keys if kwargs[k] is not None]
        names = [k for k in keys if kwargs[k] is not None]

        if not tensors:
            raise ValueError("SF Image Batch: 至少需要一路输入图像")

        self._check_image_dimensions(tensors, names)
        return (torch.cat(tensors, dim=0),)
