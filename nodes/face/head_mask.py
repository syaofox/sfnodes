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
        "description": "MediaPipe 自拍多类分割模型，支持面部、头发等区域分割",
    }
}

_CATEGORY = "sfnodes/face"


class SFHeadMask:
    def __init__(self):
        self.model_manager = ModelManager(HEAD_MASK_MODELS)
        self.model_buffer = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像"}),
            },
            "optional": {
                "include_face": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用", "tooltip": "是否包含面部区域"}),
                "include_hair": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用", "tooltip": "是否包含头发区域"}),
                "confidence": ("FLOAT", {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "分割置信度阈值"}),
                "mask_params": ("MASKPARAMS",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "masked_image")
    FUNCTION = "generate_head_mask"
    CATEGORY = _CATEGORY
    DESCRIPTION = "使用 MediaPipe 生成头部遮罩，支持面部和头发区域的分割组合"

    def _load_model(self):
        if self.model_buffer is not None:
            return
        model_path = self.model_manager.get_model_path(
            "selfie_multiclass_256x256", sub_dir="person_mask"
        )
        with open(model_path, "rb") as f:
            self.model_buffer = f.read()
        logger.info(f"MediaPipe 头部分割模型已加载: {os.path.basename(model_path)}")

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

    def generate_head_mask(self, image, include_face=True, include_hair=True, confidence=0.40, mask_params=None):
        B = image.shape[0]
        masks = []

        for i in range(B):
            np_img = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img)

            result = self._segment(pil_img)

            H, W = np_img.shape[:2]
            mask_np = np.zeros((H, W), dtype=np.uint8)

            if include_face:
                face_conf = result.confidence_masks[3].numpy_view().squeeze()
                mask_np = np.maximum(mask_np, (face_conf > confidence).astype(np.uint8) * 255)

            if include_hair:
                hair_conf = result.confidence_masks[1].numpy_view().squeeze()
                mask_np = np.maximum(mask_np, (hair_conf > confidence).astype(np.uint8) * 255)

            mask_t = torch.from_numpy(mask_np.astype(np.float32) / 255.0)

            if mask_params is not None:
                mask_t = mask_process(mask_t.unsqueeze(0).unsqueeze(-1), mask_params, unqueeze=False).squeeze(0)

            masks.append(mask_t)

        result_mask = torch.stack(masks, dim=0)
        result_masked = image * result_mask.unsqueeze(-1)

        return (result_mask, result_masked)
