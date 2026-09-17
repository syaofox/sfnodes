import torch
import numpy as np

from PIL import Image
from ...sf_utils import person_mask as _person_mask
from ...sf_utils.mask_utils import mask_process

_CATEGORY = "sfnodes/face"


class SFPersonMask:
    """人物部位遮罩（MediaPipe selfie multiclass）。

    分割核心在 sf_utils/person_mask.py（与画笔菜单人物部位路由单源）；本类只做
    张量↔PIL 转换、模型缓冲缓存与 mask_params 后处理。
    """

    def __init__(self):
        self.model_buffer = None

    @classmethod
    def INPUT_TYPES(cls):
        bool_widget = lambda default: ("BOOLEAN", {"default": default, "label_on": "启用", "label_off": "禁用"})
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像"}),
            },
            "optional": {
                "include_face": bool_widget(True),
                "include_hair": bool_widget(True),
                "include_body": bool_widget(False),
                "include_clothes": bool_widget(False),
                "include_background": bool_widget(False),
                "confidence": ("FLOAT", {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "分割置信度阈值"}),
                "refine_mask": ("BOOLEAN", {"default": False, "tooltip": "对检测区域进行二次分割以提高边缘质量"}),
                "mask_params": ("MASKPARAMS",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "masked_image")
    FUNCTION = "generate_mask"
    CATEGORY = _CATEGORY
    DESCRIPTION = "使用 MediaPipe 生成人物遮罩，支持面部、头发、身体、衣服、背景等区域的自由组合"

    def generate_mask(self, image, include_face=True, include_hair=True, include_body=False, include_clothes=False, include_background=False, confidence=0.40, refine_mask=False, mask_params=None):
        B = image.shape[0]
        masks = []

        for i in range(B):
            np_img = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img)

            parts = []
            if include_face:
                parts.append("face")
            if include_hair:
                parts.append("hair")
            if include_body:
                parts.append("body")
            if include_clothes:
                parts.append("clothes")
            if include_background:
                parts.append("background")
            if self.model_buffer is None:
                self.model_buffer = _person_mask.load_model_buffer()
            mask_arr = _person_mask.segment_mask(
                pil_img, parts, confidence, refine_mask, self.model_buffer)
            mask_np = (mask_arr * 255.0 + 0.5).astype(np.uint8)

            mask_t = torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(device=image.device)

            if mask_params is not None:
                mask_t = mask_process(mask_t.unsqueeze(0).unsqueeze(-1), mask_params, unqueeze=False).squeeze(0)

            masks.append(mask_t)

        result_mask = torch.stack(masks, dim=0)
        result_masked = image * result_mask.unsqueeze(-1)

        return (result_mask, result_masked)
