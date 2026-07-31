from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch

from PIL import Image
from comfy.utils import ProgressBar
from insightface.app import FaceAnalysis

from ...sf_utils.image_convert import np2tensor, tensor2np
from ...sf_utils.mask_utils import invert_mask, mask_process
from ...sf_utils.logger import get_logger
from ...sf_utils.model_manager import ModelManager
from ...sf_utils.insightface_utils import InsightFace, INSIGHTFACE_DIR
from ...sf_utils.downloader import download_model

logger = get_logger(__name__)

_CATEGORY = "sfnodes/face"

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

# 人脸检测模型配置（仅需检测模块，用于先裁剪人脸区域再推理）
DET_MODEL_NAME = "buffalo_l"
DET_MODEL_FILE = "det_10g.onnx"
DET_MODEL_URL = "https://huggingface.co/Syaofox/sfnodes/resolve/main/buffalo_l/det_10g.onnx"


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
        return occlusion_mask


class GeneratePreciseFaceMask:
    def __init__(self):
        self.selected_model = "xseg_3"
        self.occluder_model = None
        self.model_manager = ModelManager(XSEG_MODELS)
        self.detector = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_choice": (
                    list(XSEG_MODELS.keys()),
                    {"default": "xseg_3", "tooltip": "选择要加载的模型"},
                ),
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
                "detect_face_first": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "先检测人脸并裁剪放大后再推理，小脸/非主体人脸识别更准",
                    },
                ),
                "bbox_padding_percent": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "人脸框外扩比例，用于包裹头发",
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
    DESCRIPTION = "生成精确人脸遮罩 (支持xseg_1/xseg_2/xseg_3模型，模型按需自动下载并缓存；可先检测裁剪人脸区域再推理，小脸识别更准)"

    def _load_detector(self):
        if self.detector is not None:
            return
        model_dir = Path(INSIGHTFACE_DIR) / "models" / DET_MODEL_NAME
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / DET_MODEL_FILE
        if not model_path.is_file():
            download_model(DET_MODEL_URL, model_dir, DET_MODEL_FILE)
        face_analysis = FaceAnalysis(
            name=DET_MODEL_NAME,
            root=INSIGHTFACE_DIR,
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.detector = InsightFace(face_analysis)
        logger.info(f"已加载人脸检测模型: {DET_MODEL_NAME}/{DET_MODEL_FILE}")

    def _load_occluder(self, model_choice):
        if self.occluder_model is not None and self.selected_model == model_choice:
            return
        self.selected_model = model_choice
        model_path = self.model_manager.get_model_path(
            self.selected_model, sub_dir="occluder"
        )
        self.occluder_model = Occluder(model_path)
        logger.info(
            f"已加载XSEG模型: {self.selected_model} - {self.model_manager.get_model_description(self.selected_model)}"
        )

    def generate_mask(
        self,
        model_choice,
        input_image,
        mask_threshold,
        detect_face_first=True,
        bbox_padding_percent=0.4,
        mask_params=None,
    ):
        self._load_occluder(model_choice)
        face_occluder_model = self.occluder_model

        out_mask, out_inverted_mask, out_image = [], [], []

        steps = input_image.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        for i in range(steps):
            mask, processed_img = self._process_single_image(
                input_image[i],
                face_occluder_model,
                mask_threshold,
                mask_params,
                detect_face_first,
                bbox_padding_percent,
            )
            out_mask.append(mask)
            out_inverted_mask.append(invert_mask(mask))
            out_image.append(processed_img)
            if steps > 1:
                pbar.update(1) # type: ignore

        return (
            torch.stack(out_mask).squeeze(-1),
            torch.stack(out_inverted_mask).squeeze(-1),
            torch.stack(out_image),
        )

    def _create_cropped_occlusion_mask(
        self, cv2_image, face_occluder_model, mask_threshold, bbox_padding_percent
    ):
        """先检测人脸并裁剪（适当外扩包裹头发），在裁剪区域推理后贴回原图，其余区域为黑"""
        H, W = cv2_image.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        self._load_detector()
        _, x, y, w, h = self.detector.get_single_bbox(
            pil_image, padding=0, padding_percent=bbox_padding_percent, face_index=0
        )
        if w <= 0 or h <= 0:
            logger.warning("未检测到人脸，回退整图推理")
            return face_occluder_model.create_occlusion_mask(cv2_image, mask_threshold)
        crop = cv2_image[y : y + h, x : x + w]
        crop_mask = face_occluder_model.create_occlusion_mask(crop, mask_threshold)
        full_mask = np.zeros((H, W), dtype=crop_mask.dtype)
        full_mask[y : y + h, x : x + w] = crop_mask
        return full_mask

    def _process_single_image(
        self,
        img,
        face_occluder_model,
        mask_threshold,
        mask_params=None,
        detect_face_first=True,
        bbox_padding_percent=0.4,
    ):
        """处理单张图像"""
        face = tensor2np(img)
        if face is None:
            logger.warning("没有检测到人脸")
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        cv2_image = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
        if detect_face_first:
            occlusion_mask = self._create_cropped_occlusion_mask(
                cv2_image, face_occluder_model, mask_threshold, bbox_padding_percent
            )
        else:
            occlusion_mask = face_occluder_model.create_occlusion_mask(
                cv2_image, mask_threshold
            )

        if occlusion_mask is None:
            logger.warning("没有检测到人脸特征")
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        mask = (
            np2tensor(occlusion_mask)
            .unsqueeze(0)
            .squeeze(-1)
            .clamp(0, 1)
            .to(device=img.device)
        )

        mask = mask_process(mask, mask_params, unqueeze=True)
        processed_img = img * mask.repeat(1, 1, 3)
        return mask, processed_img
