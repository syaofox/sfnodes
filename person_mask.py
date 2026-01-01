import os
import cv2
import torch
import numpy as np
import mediapipe as mp

from functools import reduce
from PIL import Image
from .utils.model_manager import ModelManager
from .utils.mask_utils import mask_process,pil2tensor

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# 模型配置
PERSON_MASK_MODELS = {
    "selfie_multiclass_256x256": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/selfie_multiclass_256x256.tflite",
        "filename": "selfie_multiclass_256x256.tflite",
        "description": "OpenVINO 多类自拍分割模型",
    }
}


_CATEGORY = "sfnodes/masks"


class PersonSegmenter:
    """人像分割模型的封装类"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model_buffer = None

        # 加载模型到内存
        with open(self.model_path, "rb") as f:
            self.model_buffer = f.read()

        print(f"模型已加载: {os.path.basename(self.model_path)}")

    def create_segmenter(self):
        """创建分割器实例"""
        image_segmenter_base_options = BaseOptions(model_asset_buffer=self.model_buffer)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=image_segmenter_base_options,
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
        )

        return ImageSegmenter.create_from_options(options)


class PersonSegmenterLoader:
    """加载人像分割模型的节点"""

    def __init__(self):
        self.segmenter = None
        self.model_manager = ModelManager(PERSON_MASK_MODELS)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("PERSON_SEGMENTER",)
    RETURN_NAMES = ("segmenter",)
    FUNCTION = "load_segmenter"
    CATEGORY = _CATEGORY
    DESCRIPTION = "加载人像分割模型"

    def load_segmenter(self):
        model_path = self.model_manager.get_model_path(
            "selfie_multiclass_256x256", sub_dir="person_mask"
        )
        segmenter = PersonSegmenter(model_path)
        return (segmenter,)


class PersonMaskGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        false_widget = (
            "BOOLEAN",
            {"default": False, "label_on": "启用", "label_off": "禁用"},
        )
        true_widget = (
            "BOOLEAN",
            {"default": True, "label_on": "启用", "label_off": "禁用"},
        )

        return {
            "required": {
                "segmenter": ("PERSON_SEGMENTER",),
                "images": ("IMAGE",),
            },
            "optional": {
                "face_mask": true_widget,
                "background_mask": false_widget,
                "hair_mask": false_widget,
                "body_mask": false_widget,
                "clothes_mask": false_widget,
                "confidence": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "refine_mask": true_widget,
                 "mask_params": ("MASKPARAMS",),

            },
        }

    CATEGORY = _CATEGORY
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)

    FUNCTION = "generate_mask"

    def get_mediapipe_image(self, image: Image.Image) -> mp.Image:
        # 将图像转换为NumPy数组
        numpy_image = np.asarray(image)

        image_format = mp.ImageFormat.SRGB

        # 如有必要，将BGR转换为RGB
        if numpy_image.shape[-1] == 4:
            image_format = mp.ImageFormat.SRGBA
        elif numpy_image.shape[-1] == 3:
            image_format = mp.ImageFormat.SRGB
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=image_format, data=numpy_image)

    def get_bbox_for_mask(self, mask_image: Image.Image):
        # 将图像转换为灰度图
        grayscale = mask_image.convert("L")

        # 创建二进制掩码，非黑色像素为白色(255)
        mask_for_bbox = grayscale.point(lambda p: 255 if p > 0 else 0)  # type: ignore

        # 获取非黑色区域的边界框
        bbox = mask_for_bbox.getbbox()

        if bbox != None:
            left = bbox[0]
            upper = bbox[1]
            right = bbox[2]
            lower = bbox[3]

            bbox_width = right - left
            bbox_height = lower - upper

            # 在每个方向扩展边界框20%（如果可能）
            bbox_padding_x = round(bbox_width * 0.2)
            bbox_padding_y = round(bbox_height * 0.2)

            # left, upper, right, lower
            bbox = (
                # left
                left - bbox_padding_x if left > bbox_padding_x else 0,
                # upper
                upper - bbox_padding_y if upper > bbox_padding_y else 0,
                # right
                right + bbox_padding_x
                if right < grayscale.width - bbox_padding_x
                else grayscale.width,
                # lower
                lower + bbox_padding_y
                if lower < grayscale.height - bbox_padding_y
                else grayscale.height,
            )

        return bbox

    def __get_mask(
        self,
        image: Image.Image,
        segmenter,
        face_mask: bool,
        background_mask: bool,
        hair_mask: bool,
        body_mask: bool,
        clothes_mask: bool,
        confidence: float,
        refine_mask: bool,
    ) -> Image.Image:
        # 检索分割图像的掩码
        media_pipe_image = self.get_mediapipe_image(image=image)
        segmented_masks = None
        if any([face_mask, background_mask, hair_mask, body_mask, clothes_mask]):
            segmented_masks = segmenter.segment(media_pipe_image)

        # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
        # 0 - 背景
        # 1 - 头发
        # 2 - 身体皮肤
        # 3 - 脸部皮肤
        # 4 - 衣服
        # 5 - 其他(配件)
        masks = []
        if segmented_masks:
            if background_mask:
                masks.append(segmented_masks.confidence_masks[0])
            if hair_mask:
                masks.append(segmented_masks.confidence_masks[1])
            if body_mask:
                masks.append(segmented_masks.confidence_masks[2])
            if face_mask:
                masks.append(segmented_masks.confidence_masks[3])
            if clothes_mask:
                masks.append(segmented_masks.confidence_masks[4])

        image_data = media_pipe_image.numpy_view()
        image_shape = image_data.shape

        # 将图像形状从"rgb"转换为"rgba"，即添加alpha通道
        if image_shape[-1] == 3:
            image_shape = (image_shape[0], image_shape[1], 4)

        mask_background_array = np.zeros(image_shape, dtype=np.uint8)
        mask_background_array[:] = (0, 0, 0, 255)

        mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
        mask_foreground_array[:] = (255, 255, 255, 255)

        mask_arrays = []

        if len(masks) == 0:
            mask_arrays.append(mask_background_array)
        else:
            for mask in masks:
                mask_view = mask.numpy_view()
                # 确保 mask_view 是 2D 数组 (H, W)
                if len(mask_view.shape) > 2:
                    mask_view = mask_view.squeeze()
                # 为每个通道创建条件
                condition = (
                    np.stack((mask_view,) * image_shape[-1], axis=-1) > confidence
                )
                mask_array = np.where(
                    condition, mask_foreground_array, mask_background_array
                )
                mask_arrays.append(mask_array)

        # 合并掩码，取每个掩码的最大值
        merged_mask_arrays = reduce(np.maximum, mask_arrays)

        # 创建图像
        mask_image = Image.fromarray(merged_mask_arrays)

        # 通过放大检测到分割区域的区域来细化掩码
        if refine_mask:
            bbox = self.get_bbox_for_mask(mask_image=mask_image)
            if bbox != None:
                cropped_image_pil = image.crop(bbox)

                cropped_mask_image = self.__get_mask(
                    image=cropped_image_pil,
                    segmenter=segmenter,
                    face_mask=face_mask,
                    background_mask=background_mask,
                    hair_mask=hair_mask,
                    body_mask=body_mask,
                    clothes_mask=clothes_mask,
                    confidence=confidence,
                    refine_mask=False,
                )

                updated_mask_image = Image.new("RGBA", image.size, (0, 0, 0))
                updated_mask_image.paste(cropped_mask_image, bbox)
                mask_image = updated_mask_image

        return mask_image

    def get_mask_images(
        self,
        person_segmenter,
        images,  # tensors
        face_mask: bool,
        background_mask: bool,
        hair_mask: bool,
        body_mask: bool,
        clothes_mask: bool,
        confidence: float,
        refine_mask: bool,
    ) -> list[Image.Image]:
        mask_images: list[Image.Image] = []

        # 使用传入的分割器创建分割实例
        with person_segmenter.create_segmenter() as segmenter:
            for tensor_image in images:
                # 将张量转换为PIL图像
                i = 255.0 * tensor_image.cpu().numpy()

                # mediapipe库对带有alpha通道的图像处理效果更好
                if i.shape[-1] == 3:  # 如果图像是RGB
                    # 添加完全透明的alpha通道(255)
                    i = np.dstack(
                        (i, np.full((i.shape[0], i.shape[1]), 255))
                    )  # 创建RGBA图像

                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                mask_image = self.__get_mask(
                    image=image_pil,
                    segmenter=segmenter,
                    face_mask=face_mask,
                    background_mask=background_mask,
                    hair_mask=hair_mask,
                    body_mask=body_mask,
                    clothes_mask=clothes_mask,
                    confidence=confidence,
                    refine_mask=refine_mask,
                )
                mask_images.append(mask_image)

        return mask_images

    def generate_mask(
        self,
        segmenter,
        images,
        face_mask: bool,
        background_mask: bool,
        hair_mask: bool,
        body_mask: bool,
        clothes_mask: bool,
        confidence: float,
        refine_mask: bool,
        mask_params = None,

    ):
        """从图像创建分割掩码

        Args:
            segmenter: 已加载的人像分割模型
            images (torch.Tensor): 要为其创建掩码的图像。
            face_mask (bool): 创建脸部掩码。
            background_mask (bool): 创建背景掩码。
            hair_mask (bool): 创建头发掩码。
            body_mask (bool): 创建身体掩码。
            clothes_mask (bool): 创建衣服掩码。
            confidence (float): 模型对检测到的项目存在的置信度。
            refine_mask (bool): 是否细化掩码。

        Returns:
            torch.Tensor: 分割掩码。
        """

        mask_images = self.get_mask_images(
            person_segmenter=segmenter,
            images=images,
            face_mask=face_mask,
            background_mask=background_mask,
            hair_mask=hair_mask,
            body_mask=body_mask,
            clothes_mask=clothes_mask,
            confidence=confidence,
            refine_mask=refine_mask,
        )

        tensor_masks = []
        for mask_image in mask_images:          
            tensor_mask = pil2tensor(mask_image)[:, :, :, 0] 
            processed_mask = mask_process(tensor_mask, mask_params, unqueeze=False)
            tensor_masks.append(processed_mask)

        # 合并所有mask并确保正确的维度格式 [B, H, W]
        result_masks = torch.stack(tensor_masks, dim=0)

        if len(result_masks.shape) == 4:
            result_masks = result_masks.squeeze(1)  # 去掉通道维度
        
        return (result_masks,)
