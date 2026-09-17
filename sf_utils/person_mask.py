"""人物部位分割共享核心（MediaPipe selfie multiclass，tflite）。

从 ``nodes/face/person_mask.py`` 抽出（SFPersonMask 节点与画笔菜单
``brush_mask_tools`` 人物部位路由单源）：模型缓冲加载（懒 import ModelManager，
测试可注入）、分割（懒 import mediapipe，缺失时给出清晰错误）、部位掩码合成、
可选二次 refine。输入/输出为 PIL/numpy，无 torch 依赖。
"""

import numpy as np

CLASS_INDICES = {
    "background": 0,
    "hair": 1,
    "body": 2,
    "face": 3,
    "clothes": 4,
}

MODEL_NAME = "selfie_multiclass_256x256"

MODELS = {
    MODEL_NAME: {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/selfie_multiclass_256x256.tflite",
        "filename": "selfie_multiclass_256x256.tflite",
        "description": "MediaPipe 自拍多类分割模型，支持面部、头发、身体、衣服等区域分割",
    }
}


def load_model_buffer(manager=None):
    """读取 tflite 字节（缺省经 ModelManager 自动下载/缓存，sub_dir=person_mask）。"""
    if manager is None:
        from .model_manager import ModelManager  # 懒 import：纯模块不依赖 folder_paths/网络栈
        manager = ModelManager(MODELS)
    path = manager.get_model_path(MODEL_NAME, sub_dir="person_mask")
    with open(path, "rb") as f:
        return f.read()


def normalize_parts(parts):
    """勾选部位归一：名列表或逗号串；空/全非法回退 ["face"]（与节点默认一致）。"""
    if isinstance(parts, str):
        parts = [p.strip() for p in parts.split(",")]
    if not isinstance(parts, (list, tuple)):
        return ["face"]
    out = [p for p in parts if p in CLASS_INDICES]
    return out or ["face"]


def _mediapipe():
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError(f"未安装 mediapipe：{e}")
    return mp


def _to_mediapipe_image(mp, pil_img):
    numpy_image = np.asarray(pil_img)
    if numpy_image.shape[-1] == 4:
        return mp.Image(image_format=mp.ImageFormat.SRGBA, data=numpy_image)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)


def segment(pil_img, model_buffer):
    """对 PIL 图像跑 MediaPipe 分割，返回原生 result（调用方随后 build_mask）。"""
    mp = _mediapipe()
    base_options = mp.tasks.BaseOptions(model_asset_buffer=model_buffer)
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=base_options,
        # 注意：mediapipe 的属性名是 RunningMode（原节点局部别名 VisionRunningMode
        # 曾误导——容器内实测无 vision.VisionRunningMode）
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    mp_image = _to_mediapipe_image(mp, pil_img)
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        return segmenter.segment(mp_image)


def build_mask(result, img_shape, parts, confidence=0.40):
    """按勾选部位把 confidence_masks 并集成 uint8 (H, W)（0/255）。"""
    H, W = img_shape
    mask_np = np.zeros((H, W), dtype=np.uint8)
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.40
    conf = max(0.01, min(1.0, conf))
    for name in normalize_parts(parts):
        cm = result.confidence_masks[CLASS_INDICES[name]].numpy_view().squeeze()
        mask_np = np.maximum(mask_np, (cm > conf).astype(np.uint8) * 255)
    return mask_np


def refine(pil_img, mask_np, parts, confidence, model_buffer):
    """按掩码包围盒外扩 20% 裁剪后再分割一次，提升边缘质量（与原节点同语义）。"""
    from PIL import Image

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
    result = segment(crop, model_buffer)
    crop_mask = build_mask(result, (lower - upper, right - left), parts, confidence)
    mask_np[upper:lower, left:right] = np.maximum(mask_np[upper:lower, left:right], crop_mask)
    return mask_np


def segment_mask(pil_img, parts, confidence=0.40, refine_flag=False, model_buffer=None):
    """PIL RGB → float32 (H, W) 0..1 并集掩码（单图；model_buffer 缺省自动加载）。"""
    if model_buffer is None:
        model_buffer = load_model_buffer()
    result = segment(pil_img, model_buffer)
    mask_np = build_mask(result, (pil_img.height, pil_img.width), parts, confidence)
    if refine_flag:
        mask_np = refine(pil_img, mask_np, parts, confidence, model_buffer)
    return mask_np.astype(np.float32) / 255.0
