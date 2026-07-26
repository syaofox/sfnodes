import os
import torch
import numpy as np
import mediapipe as mp

from PIL import Image
from ...sf_utils.model_manager import ModelManager
from ...sf_utils.mask_utils import mask_process
from ...sf_utils.logger import get_logger

logger = get_logger(__name__)

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HEAD_MASK_MODELS = {
    "selfie_multiclass_256x256": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/selfie_multiclass_256x256.tflite",
        "filename": "selfie_multiclass_256x256.tflite",
        "description": "MediaPipe 自拍多类分割模型，支持面部、头发、身体、衣服等区域分割",
    }
}

CLASS_INDICES = {
    "background": 0,
    "hair": 1,
    "body": 2,
    "face": 3,
    "clothes": 4,
}

_CATEGORY = "sfnodes/face"


class SFPersonMask:
    def __init__(self):
        self.model_manager = ModelManager(HEAD_MASK_MODELS)
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

    def _load_model(self):
        if self.model_buffer is not None:
            return
        model_path = self.model_manager.get_model_path(
            "selfie_multiclass_256x256", sub_dir="person_mask"
        )
        with open(model_path, "rb") as f:
            self.model_buffer = f.read()
        logger.info(f"MediaPipe 分割模型已加载: {os.path.basename(model_path)}")

    def _to_mediapipe_image(self, image: Image.Image) -> mp.Image:
        numpy_image = np.asarray(image)
        if numpy_image.shape[-1] == 4:
            return mp.Image(image_format=mp.ImageFormat.SRGBA, data=numpy_image)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    def _segment(self, image: Image.Image):
        self._load_model()
        base_options = BaseOptions(model_asset_buffer=self.model_buffer)
        options = ImageSegmenterOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
        )
        mp_image = self._to_mediapipe_image(image)
        with ImageSegmenter.create_from_options(options) as segmenter:
            return segmenter.segment(mp_image)

    def _build_mask(self, result, img_shape, include_face, include_hair, include_body, include_clothes, include_background, confidence):
        H, W = img_shape
        mask_np = np.zeros((H, W), dtype=np.uint8)
        selected = []
        if include_face:
            selected.append("face")
        if include_hair:
            selected.append("hair")
        if include_body:
            selected.append("body")
        if include_clothes:
            selected.append("clothes")
        if include_background:
            selected.append("background")
        for name in selected:
            conf = result.confidence_masks[CLASS_INDICES[name]].numpy_view().squeeze()
            mask_np = np.maximum(mask_np, (conf > confidence).astype(np.uint8) * 255)
        return mask_np

    def _refine(self, pil_img, mask_np, include_face, include_hair, include_body, include_clothes, include_background, confidence):
        mask_pil = Image.fromarray(mask_np)
        bbox = mask_pil.getbbox()
        if bbox is None:
            return mask_np
        left, upper, right, lower = bbox
        bw, bh = right - left, lower - upper
        pad_x = int(bw * 0.2) + 1
        pad_y = int(bh * 0.2) + 1
        left = max(0, left - pad_x)
        upper = max(0, upper - pad_y)
        right = min(pil_img.width, right + pad_x)
        lower = min(pil_img.height, lower + pad_y)
        crop = pil_img.crop((left, upper, right, lower))
        result = self._segment(crop)
        crop_mask = self._build_mask(result, (lower - upper, right - left), include_face, include_hair, include_body, include_clothes, include_background, confidence)
        mask_np[upper:lower, left:right] = np.maximum(mask_np[upper:lower, left:right], crop_mask)
        return mask_np

    def generate_mask(self, image, include_face=True, include_hair=True, include_body=False, include_clothes=False, include_background=False, confidence=0.40, refine_mask=False, mask_params=None):
        B = image.shape[0]
        masks = []

        for i in range(B):
            np_img = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img)

            result = self._segment(pil_img)
            mask_np = self._build_mask(result, np_img.shape[:2], include_face, include_hair, include_body, include_clothes, include_background, confidence)

            if refine_mask:
                mask_np = self._refine(pil_img, mask_np, include_face, include_hair, include_body, include_clothes, include_background, confidence)

            mask_t = torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(device=image.device)

            if mask_params is not None:
                mask_t = mask_process(mask_t.unsqueeze(0).unsqueeze(-1), mask_params, unqueeze=False).squeeze(0)

            masks.append(mask_t)

        result_mask = torch.stack(masks, dim=0)
        result_masked = image * result_mask.unsqueeze(-1)

        return (result_mask, result_masked)
