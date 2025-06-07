import cv2
import numpy as np
import onnxruntime
import torch

from comfy.utils import ProgressBar

from .utils.image_convert import np2tensor, tensor2np
from .utils.mask_utils import blur_mask, fill_holes, invert_mask, expand_mask

# from .utils.xseg_models import get_model_path, list_available_models, get_model_description
from .utils.model_manager import ModelManager

_CATEGORY = "sfnodes/face_analysis"

# 模型配置
XSEG_MODELS = {
    "xseg_1": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/xseg_1.onnx",
        "filename": "xseg_1.onnx",
        "description": "原始DFL-XSEG模型，针对人脸分割进行优化",
    },
    "xseg_2": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/xseg_2.onnx",
        "filename": "xseg_2.onnx",
        "description": "改进的XSEG模型，提供更精确的人脸分割",
    },
    "xseg_3": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/xseg_3.onnx",
        "filename": "xseg_3.onnx",
        "description": "改进的XSEG模型，提供更精确的人脸分割",
    },
}


class Occluder:
    def __init__(self, occluder_model_path):
        self.occluder_model_path = occluder_model_path
        self.face_occluder = self.get_face_occluder()

    def get_face_occluder(self):
        return onnxruntime.InferenceSession(
            self.occluder_model_path,
            providers=["CPUExecutionProvider"],
        )

    def create_occlusion_mask(self, crop_vision_frame, threshold=0.5):
        prepare_vision_frame = cv2.resize(
            crop_vision_frame, self.face_occluder.get_inputs()[0].shape[1:3][::-1]
        )
        prepare_vision_frame = (
            np.expand_dims(prepare_vision_frame, axis=0).astype(np.float32) / 255
        )
        prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
        occlusion_mask = self.face_occluder.run(
            None, {self.face_occluder.get_inputs()[0].name: prepare_vision_frame}
        )[0][0]
        occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
        occlusion_mask = (occlusion_mask > threshold).astype(np.float32)  # 应用阈值
        # occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return occlusion_mask


class OccluderLoader:
    def __init__(self):
        self.selected_model = "xseg_2"
        self.occluder_model = None
        self.model_manager = ModelManager(XSEG_MODELS)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_choice": (
                    list(XSEG_MODELS.keys()),
                    {"default": "xseg_2", "tooltip": "选择要加载的模型"},
                )
            }
        }

    RETURN_TYPES = ("OCCLUDER",)
    RETURN_NAMES = ("occluder",)
    FUNCTION = "get_occluder"
    CATEGORY = _CATEGORY
    DESCRIPTION = "加载人脸遮挡模型"

    def get_occluder(self, model_choice="xseg_2"):
        self.selected_model = model_choice
        model_path = self.model_manager.get_model_path(
            self.selected_model, sub_dir="occluder"
        )
        self.occluder_model = Occluder(model_path)
        print(
            f"已加载XSEG模型: {self.selected_model} - {self.model_manager.get_model_description(self.selected_model)}"
        )
        return (self.occluder_model,)


class GeneratePreciseFaceMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "occluder": ("OCCLUDER",),
                "input_image": ("IMAGE",),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "设置遮罩阈值",
                    },
                ),
            },
            "optional": {
                "mask_params": ("MASKPARAMS",),
            },
        }

    RETURN_TYPES = (
        "MASK",
        "MASK",
        "IMAGE",
    )
    RETURN_NAMES = (
        "mask",
        "inverted_mask",
        "image",
    )
    FUNCTION = "generate_mask"
    CATEGORY = _CATEGORY
    DESCRIPTION = "生成精确人脸遮罩 (支持xseg_1/xseg_2模型)"

    def generate_mask(
        self,
        occluder,
        input_image,
        mask_threshold,
        mask_params=None,
    ):
        if mask_params is None:
            mask_params = {
                "grow": 0,
                "grow_percent": 0.0,
                "grow_tapered": False,
                "blur": 0,
                "fill": False,
            }

        grow = mask_params["grow"]
        grow_percent = mask_params["grow_percent"]
        grow_tapered = mask_params["grow_tapered"]
        blur = mask_params["blur"]
        fill = mask_params["fill"]

        face_occluder_model = occluder

        out_mask, out_inverted_mask, out_image = [], [], []

        steps = input_image.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        for i in range(steps):
            mask, processed_img = self._process_single_image(
                input_image[i],
                face_occluder_model,
                mask_threshold,
                grow,
                grow_percent,
                grow_tapered,
                blur,
                fill,
            )
            out_mask.append(mask)
            out_inverted_mask.append(invert_mask(mask))
            out_image.append(processed_img)
            if steps > 1:
                pbar.update(1)

        return (
            torch.stack(out_mask).squeeze(-1),
            torch.stack(out_inverted_mask).squeeze(-1),
            torch.stack(out_image),
        )

    def _process_single_image(
        self,
        img,
        face_occluder_model,
        mask_threshold,
        grow,
        grow_percent,
        grow_tapered,
        blur,
        fill,
    ):
        """处理单张图像"""
        face = tensor2np(img)
        if face is None:
            print("\033[96m没有检测到人脸\033[0m")
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        cv2_image = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
        occlusion_mask = face_occluder_model.create_occlusion_mask(
            cv2_image, mask_threshold
        )

        if occlusion_mask is None:
            print("\033[96m没有检测到人脸特征\033[0m")
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        mask = self._process_mask(
            occlusion_mask, img, grow, grow_percent, grow_tapered, blur, fill
        )
        processed_img = img * mask.repeat(1, 1, 3)
        return mask, processed_img

    def _process_mask(
        self, occlusion_mask, img, grow, grow_percent, grow_tapered, blur, fill
    ):
        """处理遮罩"""
        mask = (
            np2tensor(occlusion_mask)
            .unsqueeze(0)
            .squeeze(-1)
            .clamp(0, 1)
            .to(device=img.device)
        )

        grow_count = int(grow_percent * max(mask.shape)) + grow
        if grow_count > 0:
            mask = expand_mask(mask, grow_count, grow_tapered)

        if fill:
            mask = fill_holes(mask)

        if blur > 0:
            mask = blur_mask(mask, blur)

        return mask.squeeze(0).unsqueeze(-1)
