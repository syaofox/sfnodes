import cv2
import numpy as np
import torch

from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer
from comfy.utils import ProgressBar
from ...sf_utils.image_convert import tensor2np, image_to_tensor
from ...sf_utils.mask_utils import mask_process, mask_from_landmarks
from ...sf_utils.logger import get_logger

logger = get_logger(__name__)


_CATEGORY = "sfnodes/face"


class FaceWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_models": (
                    "ANALYSIS_MODELS",
                    {"tooltip": "人脸分析模型，由 SF Face Analysis Models 输出"},
                ),
                "image_from": ("IMAGE", {"tooltip": "源图像：其人脸将被扭曲变形"}),
                "image_to": ("IMAGE", {"tooltip": "目标图像：人脸特征要对齐的目标"}),
                "keypoints": (
                    ["main features", "full face", "full face+forehead (if available)"],
                    {
                        "tooltip": "用于估计变形的关键点区域：main features 仅内部五官；full face 含脸部轮廓；full face+forehead 额外包含前额近似点（基于检测框生成）"
                    },
                ),
            },
            "optional": {
                "mask_from": (
                    "MASK",
                    {"tooltip": "源图像遮罩，限定参与变形的区域；不提供时按源关键点凸包自动生成"},
                ),
                "mask_to": (
                    "MASK",
                    {"tooltip": "目标图像遮罩，限定变形结果的范围；不提供时按目标关键点凸包自动生成"},
                ),
                "mask_params": (
                    "MASKPARAMS",
                    {"tooltip": "遮罩处理参数（生长/模糊/填充/反转等），作用于最终输出遮罩"},
                ),
                "is_mathcolor": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "是否对变形区域做颜色匹配，使其色调与目标图像一致"},
                ),
                "include_background": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "为 True 时输出保留源图像除人脸外的其他部分（整体按同一仿射变换对齐）；为 False 时非变形区域使用目标图像",
                    },
                ),
                "match_strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "颜色匹配强度：1.0 完全采用匹配结果，0.0 保持变形后的原始颜色",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "变形强度：1.0 完全匹配目标人脸特征，0.0 保持源图像不变",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "image",
        "mask",
    )
    FUNCTION = "warp"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将源图像的人脸扭曲变形以匹配目标图像的人脸特征，支持变形强度与颜色匹配强度控制"

    def warp(
        self,
        analysis_models,
        image_from,
        image_to,
        keypoints,
        mask_from=None,
        mask_to=None,
        mask_params=None,
        is_mathcolor=True,
        include_background=False,
        match_strength=0.8,
        strength=1.0,
    ):
        if image_from.shape[0] < image_to.shape[0]:
            image_from = torch.cat(
                [
                    image_from,
                    image_from[-1:].repeat(
                        (image_to.shape[0] - image_from.shape[0], 1, 1, 1)
                    ),
                ],
                dim=0,
            )
        elif image_from.shape[0] > image_to.shape[0]:
            image_from = image_from[: image_to.shape[0]]
            logger.info(
                f"image_from has more frames than image_to; truncated to {image_to.shape[0]} frame(s)"
            )

        if mask_from is not None and mask_from.shape[0] < image_from.shape[0]:
            logger.info(
                f"mask_from has fewer frames ({mask_from.shape[0]}) than image_from; last frame will be reused for the remaining frames"
            )
        if mask_to is not None and mask_to.shape[0] < image_to.shape[0]:
            logger.info(
                f"mask_to has fewer frames ({mask_to.shape[0]}) than image_to; last frame will be reused for the remaining frames"
            )

        steps = image_from.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        cm = ColorMatcher()

        result_image = []
        result_mask = []

        for i in range(steps):
            img_from = tensor2np(image_from[i])
            img_to = tensor2np(image_to[i])

            shape_from = analysis_models.get_landmarks(
                img_from, extended_landmarks=("forehead" in keypoints)
            )
            shape_to = analysis_models.get_landmarks(
                img_to, extended_landmarks=("forehead" in keypoints)
            )

            if shape_from is None or shape_to is None:
                logger.warning(f"No landmarks detected at frame {i}")
                img = image_to[i].unsqueeze(0)
                mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)
                result_image.append(img)
                result_mask.append(mask)
                if steps > 1:
                    pbar.update(1)
                continue

            if keypoints == "main features":
                shape_from = shape_from[1]
                shape_to = shape_to[1]
            elif "forehead" in keypoints:
                # 全部 106 点 + 前额近似弧线点（outline_forehead）
                shape_from = np.vstack([shape_from[0], shape_from[-1]])
                shape_to = np.vstack([shape_to[0], shape_to[-1]])
            else:
                shape_from = shape_from[0]
                shape_to = shape_to[0]

            # get the transformation matrix
            from_points = np.array(shape_from, dtype=np.float64)
            to_points = np.array(shape_to, dtype=np.float64)

            matrix = cv2.estimateAffine2D(from_points, to_points)[0]
            if matrix is None:
                logger.warning(f"Could not estimate affine transform at frame {i}")
                img = image_to[i].unsqueeze(0)
                mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)
                result_image.append(img)
                result_mask.append(mask)
                if steps > 1:
                    pbar.update(1)
                continue

            if strength < 1.0:
                # 向恒等变换插值，实现部分变形
                identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                matrix = strength * matrix + (1.0 - strength) * identity

            output = cv2.warpAffine(
                img_from,
                matrix,
                (img_to.shape[1], img_to.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # 处理mask_from和mask_to（帧数不足时复用最后一帧）
            if mask_from is not None and mask_from.shape[0] > 0:
                mask_from_tensor = mask_from[min(i, mask_from.shape[0] - 1)]
                # 确保mask_from是二维的
                if len(mask_from_tensor.shape) == 3 and mask_from_tensor.shape[2] == 1:
                    mask_from_tensor = mask_from_tensor.squeeze(-1)
                mask_from_np = mask_from_tensor.cpu().numpy().astype(np.float64)

                # 确保mask_from与img_from尺寸一致，如果不一致则调整大小
                if (
                    mask_from_np.shape[0] != img_from.shape[0]
                    or mask_from_np.shape[1] != img_from.shape[1]
                ):
                    mask_from_np = cv2.resize(
                        mask_from_np,
                        (img_from.shape[1], img_from.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
            else:
                # 计算mask_from
                mask_from_np = mask_from_landmarks(img_from, shape_from)

            if mask_to is not None and mask_to.shape[0] > 0:
                mask_to_tensor = mask_to[min(i, mask_to.shape[0] - 1)]
                # 确保mask_to是二维的
                if len(mask_to_tensor.shape) == 3 and mask_to_tensor.shape[2] == 1:
                    mask_to_tensor = mask_to_tensor.squeeze(-1)
                mask_to_np = mask_to_tensor.cpu().numpy().astype(np.float64)

                # 确保mask_to与img_to尺寸一致，如果不一致则调整大小
                if (
                    mask_to_np.shape[0] != img_to.shape[0]
                    or mask_to_np.shape[1] != img_to.shape[1]
                ):
                    mask_to_np = cv2.resize(
                        mask_to_np,
                        (img_to.shape[1], img_to.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
            else:
                # 计算mask_to
                mask_to_np = mask_from_landmarks(img_to, shape_to)

            # 对mask_from进行仿射变换
            output_mask = cv2.warpAffine(
                mask_from_np, matrix, (img_to.shape[1], img_to.shape[0])
            )

            output_mask = (
                torch.from_numpy(output_mask).unsqueeze(0).unsqueeze(-1).float()
            )
            mask_to_local = (
                torch.from_numpy(mask_to_np).unsqueeze(0).unsqueeze(-1).float()
            )
            output_mask = torch.min(output_mask, mask_to_local)

            output = image_to_tensor(output).unsqueeze(0)
            img_to = image_to_tensor(img_to).unsqueeze(0)

            # 处理mask维度：[B,H,W,1] -> [B,H,W] 用于mask_process
            output_mask_2d = output_mask.squeeze(-1)

            # 使用mask_process处理
            processed_mask = mask_process(output_mask_2d, mask_params, unqueeze=False)

            # 恢复维度：[B,H,W] -> [B,H,W,1]
            output_mask = processed_mask.unsqueeze(-1)

            if is_mathcolor:
                cm_ref = None
                cm_image = None
                cm_region = None

                if torch.any(mask_to_local):
                    _, y, x, _ = torch.where(mask_to_local)
                    x1 = max(0, x.min().item())
                    y1 = max(0, y.min().item())
                    x2 = min(img_to.shape[2], x.max().item())
                    y2 = min(img_to.shape[1], y.max().item())
                    cm_ref = img_to[:, y1:y2, x1:x2, :]

                if torch.any(output_mask):
                    _, y, x, _ = torch.where(output_mask)
                    x1 = max(0, x.min().item())
                    y1 = max(0, y.min().item())
                    x2 = min(output.shape[2], x.max().item())
                    y2 = min(output.shape[1], y.max().item())
                    cm_image = output[:, y1:y2, x1:x2, :]
                    cm_region = (y1, y2, x1, x2)

                if (
                    cm_ref is not None
                    and cm_image is not None
                    and cm_image.numel() > 0
                    and cm_ref.numel() > 0
                ):
                    normalized = cm.transfer(
                        src=Normalizer(cm_image[0].numpy()).type_norm(),
                        ref=Normalizer(cm_ref[0].numpy()).type_norm(),
                        method="mkl",
                    )
                    normalized = torch.from_numpy(normalized).unsqueeze(0)
                    y1, y2, x1, x2 = cm_region
                    output[:, y1 : y1 + cm_image.shape[1], x1 : x1 + cm_image.shape[2], :] = (
                        match_strength * normalized + (1 - match_strength) * cm_image
                    )

            if include_background:
                # 背景同样经过同一仿射变换（warped 源图），与变形后的人脸保持同一坐标系
                output_image = output
            else:
                # 原有逻辑：使用目标图像的其他部分
                output_image = output * output_mask + img_to * (1 - output_mask)

            output_image = output_image.clamp(0, 1)
            output_mask = output_mask.clamp(0, 1).squeeze(-1)

            result_image.append(output_image)
            result_mask.append(output_mask)

            if steps > 1:
                pbar.update(1)

        result_image = torch.cat(result_image, dim=0)
        result_mask = torch.cat(result_mask, dim=0)

        return (result_image, result_mask)
