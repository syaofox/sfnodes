from pathlib import Path

import cv2

from PIL import Image
from insightface.app import FaceAnalysis

from .insightface_utils import InsightFace, INSIGHTFACE_DIR
from .downloader import download_model
from .logger import get_logger

logger = get_logger(__name__)

# 人脸检测模型配置（仅需检测模块，用于先裁剪人脸区域再推理）
DET_MODEL_NAME = "buffalo_l"
DET_MODEL_FILE = "det_10g.onnx"
DET_MODEL_URL = "https://huggingface.co/Syaofox/sfnodes/resolve/main/buffalo_l/det_10g.onnx"


class FaceDetector:
    """人脸检测器：懒加载 det_10g 模型，提供检测 + 外扩 + 裁剪"""

    def __init__(self):
        self.detector = None

    def _load(self):
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

    def detect_crop(self, cv2_image, padding_percent=0.4):
        """检测最大人脸并裁剪（外扩包裹头发，边界自动收敛），返回 (crop, x, y, w, h)；检测失败返回 None"""
        pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        self._load()
        _, x, y, w, h = self.detector.get_single_bbox(
            pil_image, padding=0, padding_percent=padding_percent, face_index=0
        )
        if w <= 0 or h <= 0:
            return None
        return cv2_image[y : y + h, x : x + w], x, y, w, h
