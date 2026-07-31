import cv2
import numpy as np
import onnxruntime
import torch

from comfy.utils import ProgressBar

from ...sf_utils.image_convert import np2tensor, tensor2np
from ...sf_utils.mask_utils import invert_mask, mask_process
from ...sf_utils.logger import get_logger
from ...sf_utils.model_manager import ModelManager
from ...sf_utils.face_detector import FaceDetector

logger = get_logger(__name__)

_CATEGORY = "sfnodes/face"


# 模型配置
REGION_MODELS = {
    "bisenet_resnet_18": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/bisenet_resnet_18.onnx",
        "filename": "bisenet_resnet_18.onnx",
        "description": "轻量级BiSeNet模型，使用ResNet-18作为骨干网络",
    },
    "bisenet_resnet_34": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/bisenet_resnet_34.onnx",
        "filename": "bisenet_resnet_34.onnx",
        "description": "高精度BiSeNet模型，使用ResNet-34作为骨干网络",
    },
}


# 面部区域集合
FACE_MASK_REGION_SET = {
    "skin": 1,
    "left-eyebrow": 2,
    "right-eyebrow": 3,
    "left-eye": 4,
    "right-eye": 5,
    "glasses": 6,
    "nose": 10,
    "mouth": 11,
    "upper-lip": 12,
    "lower-lip": 13,
}


class RegionExtractor:
    def __init__(self, region_model_path):
        self.region_model_path = region_model_path
        self.region_model = self.load_region_model()

    def load_region_model(self):
        available_providers = onnxruntime.get_available_providers()
        preferred_providers = []

        # 尝试使用GPU如果可用
        if "CUDAExecutionProvider" in available_providers:
            preferred_providers.append("CUDAExecutionProvider")

        # 总是添加CPU作为备选
        preferred_providers.append("CPUExecutionProvider")

        return onnxruntime.InferenceSession(
            self.region_model_path,
            providers=preferred_providers,
        )

    def create_region_mask(self, image, region_indices, threshold=0.5):
        """创建面部区域遮罩"""
        if len(region_indices) == 0:
            logger.warning("没有选择有效的面部区域")
            return np.zeros(image.shape[:2], dtype=np.float32)

        # 准备输入数据
        model_size = (512, 512)  # BiSeNet模型的标准输入大小
        prepare_image = cv2.resize(image, model_size)
        prepare_image = (
            prepare_image[:, :, ::-1].astype(np.float32) / 255
        )  # BGR转RGB并归一化
        prepare_image = np.subtract(
            prepare_image, np.array([0.485, 0.456, 0.406]).astype(np.float32)
        )  # 减去均值
        prepare_image = np.divide(
            prepare_image, np.array([0.229, 0.224, 0.225]).astype(np.float32)
        )  # 除以标准差
        prepare_image = np.expand_dims(prepare_image, axis=0)
        prepare_image = prepare_image.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        # 运行推理
        try:
            region_mask = self.region_model.run(None, {"input": prepare_image})[0][0]
        except Exception as e:
            logger.error(f"运行模型推理时出错: {e}")
            return np.zeros(image.shape[:2], dtype=np.float32)

        # 处理输出 - 为选定的区域创建二值遮罩
        region_mask = np.isin(region_mask.argmax(0), region_indices)
        region_mask = cv2.resize(region_mask.astype(np.float32), image.shape[:2][::-1])

        return region_mask


class GenerateRegionFaceMask:
    def __init__(self):
        self.selected_model = "bisenet_resnet_34"
        self.region_extractor = None
        self.model_manager = ModelManager(REGION_MODELS)
        self.face_detector = FaceDetector()

    @classmethod
    def INPUT_TYPES(cls):
        region_checkboxes = {}
        for region in FACE_MASK_REGION_SET.keys():
            region_checkboxes[f"use_{region}"] = ("BOOLEAN", {"default": False})
        return {
            "required": {
                "model_choice": (
                    list(REGION_MODELS.keys()),
                    {"default": "bisenet_resnet_34", "tooltip": "选择要加载的模型"},
                ),
                "input_image": ("IMAGE", {"tooltip": "输入图像"}),
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
                **region_checkboxes,
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
    DESCRIPTION = "生成精确面部区域遮罩 (支持bisenet模型，模型按需自动下载并缓存；可先检测裁剪人脸区域再推理，小脸识别更准)"

    def _load_region_extractor(self, model_choice):
        if self.region_extractor is not None and self.selected_model == model_choice:
            return
        self.selected_model = model_choice
        model_path = self.model_manager.get_model_path(
            self.selected_model, sub_dir="region"
        )
        self.region_extractor = RegionExtractor(model_path)
        logger.info(
            f"已加载面部区域分割模型: {self.selected_model} - {self.model_manager.get_model_description(self.selected_model)}"
        )

    def _create_cropped_region_mask(
        self, cv2_image, region_indices, bbox_padding_percent
    ):
        """先检测人脸并裁剪（适当外扩包裹头发），在裁剪区域推理后贴回原图，其余区域为黑"""
        H, W = cv2_image.shape[:2]
        crop_result = self.face_detector.detect_crop(cv2_image, bbox_padding_percent)
        if crop_result is None:
            logger.warning("未检测到人脸，回退整图推理")
            return self.region_extractor.create_region_mask(cv2_image, region_indices)
        crop, x, y, w, h = crop_result
        crop_mask = self.region_extractor.create_region_mask(crop, region_indices)
        full_mask = np.zeros((H, W), dtype=crop_mask.dtype)
        full_mask[y : y + h, x : x + w] = crop_mask
        return full_mask

    def _process_single_image(
        self,
        img,
        region_indices,
        mask_params,
        detect_face_first=True,
        bbox_padding_percent=0.4,
    ):
        """处理单张图像"""
        face = tensor2np(img)
        if face is None:
            logger.warning("无效的输入图像")
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        cv2_image = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
        if detect_face_first:
            region_mask = self._create_cropped_region_mask(
                cv2_image, region_indices, bbox_padding_percent
            )
        else:
            region_mask = self.region_extractor.create_region_mask(
                cv2_image, region_indices
            )

        if region_mask is None or np.max(region_mask) == 0:
            logger.warning("未能创建有效的区域遮罩")
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        mask = (
            np2tensor(region_mask)
            .unsqueeze(0)
            .squeeze(-1)
            .clamp(0, 1)
            .to(device=img.device)
        )

        mask = mask_process(mask, mask_params, unqueeze=True)
        processed_img = img * mask.repeat(1, 1, 3)
        return mask, processed_img

    def generate_mask(
        self,
        model_choice,
        input_image,
        detect_face_first=True,
        bbox_padding_percent=0.4,
        mask_params=None,
        **kwargs,
    ):
        self._load_region_extractor(model_choice)

        selected_names = [
            region
            for region in FACE_MASK_REGION_SET.keys()
            if kwargs.get(f"use_{region}", False)
        ]

        # 如果没有选择任何区域，默认选择皮肤
        if not selected_names:
            selected_names = ["skin"]
        region_indices = [FACE_MASK_REGION_SET[r] for r in selected_names]
        logger.info(f"已选择的面部区域: {', '.join(selected_names)}")

        out_mask, out_inverted_mask, out_image = [], [], []

        steps = input_image.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        for i in range(steps):
            mask, processed_img = self._process_single_image(
                input_image[i],
                region_indices,
                mask_params,
                detect_face_first,
                bbox_padding_percent,
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
