"""SFCropWithPadInfo — 按 pad_info 反向裁掉 padding（复刻 EditUtils CropWithPadInfo_EditUtils）。

典型用法：Krea2 管线里 SF Krea2 Edit Text Encode 对主图做 pad 画布编码并输出 pad_info，
采样/解码后的图像仍是带 padding 的画布尺寸，用本节点裁回原始内容区（附带 scale_by）。

纯逻辑在 sf_utils/krea2_edit.py::crop_with_pad_info。
"""

from ...sf_utils.common import AnyType
from ...sf_utils import krea2_edit as ke

_CATEGORY = "sfnodes/image"

any_type = AnyType("*")


class SFCropWithPadInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pad_info": (any_type, {
                    "tooltip": "由 Krea2 Edit Text Encode 输出的 pad_info 字典 "
                               "（x/y/width/height/scale_by），空或非法时原样返回图像",
                }),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("cropped_image", "scale_by",)
    FUNCTION = "crop_image"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "按 pad_info 反向裁掉右侧/底部 padding，还原编码前的原始内容区，"
        "并输出原图→缩放图的尺寸比 scale_by"
    )

    def crop_image(self, image, pad_info):
        if not isinstance(pad_info, dict):
            print("SFCropWithPadInfo: pad_info 非法（非 dict），原样返回图像")
            return (image, 1.0)
        return ke.crop_with_pad_info(image, pad_info)
