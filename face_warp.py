import cv2
import numpy as np
import torch
import torch.nn.functional as F

from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer
from comfy.utils import ProgressBar
from .utils.image_convert import tensor_to_image, image_to_tensor
from .utils.mask_utils import mask_process


_CATEGORY = "sfnodes/face_analysis"

def mask_from_landmarks(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.float64)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, points, color=(1,))
    return mask


class FaceWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "image_from": ("IMAGE",),
                "image_to": ("IMAGE",),
                "keypoints": (
                    ["main features", "full face", "full face+forehead (if available)"],
                ),
            },
            "optional": {
                "mask_from": ("MASK",),
                "mask_to": ("MASK",),
                "mask_params": ("MASKPARAMS",),
                "is_mathcolor": ("BOOLEAN", {"default": True}),
                "include_background": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    FUNCTION = "warp"
    CATEGORY = _CATEGORY

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

        steps = image_from.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        cm = ColorMatcher()

        result_image = []
        result_mask = []

        for i in range(steps):
            img_from = tensor_to_image(image_from[i])
            img_to = tensor_to_image(image_to[i])

            shape_from = analysis_models.get_landmarks(
                img_from, extended_landmarks=("forehead" in keypoints)
            )
            shape_to = analysis_models.get_landmarks(
                img_to, extended_landmarks=("forehead" in keypoints)
            )

            if shape_from is None or shape_to is None:
                print(f"\033[96mNo landmarks detected at frame {i}\033[0m")
                img = image_to[i].unsqueeze(0)
                mask = torch.zeros_like(img)[:, :, :1]
                result_image.append(img)
                result_mask.append(mask)
                continue

            if keypoints == "main features":
                shape_from = shape_from[1]
                shape_to = shape_to[1]
            else:
                shape_from = shape_from[0]
                shape_to = shape_to[0]

            # get the transformation matrix
            from_points = np.array(shape_from, dtype=np.float64)
            to_points = np.array(shape_to, dtype=np.float64)

            matrix = cv2.estimateAffine2D(from_points, to_points)[0]
            output = cv2.warpAffine(
                img_from,
                matrix,
                (img_to.shape[1], img_to.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # 处理mask_from和mask_to
            if mask_from is not None and i < mask_from.shape[0]:
                # 使用提供的mask_from
                mask_from_tensor = mask_from[i]
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
                        interpolation=cv2.INTER_LINEAR,
                    )
            else:
                # 计算mask_from
                mask_from_np = mask_from_landmarks(img_from, shape_from)

            if mask_to is not None and i < mask_to.shape[0]:
                # 使用提供的mask_to
                mask_to_tensor = mask_to[i]
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
                        interpolation=cv2.INTER_LINEAR,
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
            mask_to = torch.from_numpy(mask_to_np).unsqueeze(0).unsqueeze(-1).float()
            output_mask = torch.min(output_mask, mask_to)

            output = image_to_tensor(output).unsqueeze(0)
            img_to = image_to_tensor(img_to).unsqueeze(0)
            
            # 处理mask维度：[B,H,W,1] -> [B,H,W] 用于mask_process
            output_mask_2d = output_mask.squeeze(-1)   
                     
            # 使用mask_process处理
            processed_mask = mask_process(output_mask_2d, mask_params, unqueeze=False)

            # 恢复维度：[B,H,W] -> [B,H,W,1]
            output_mask = processed_mask.unsqueeze(-1)
           

            padding = 0

            _, y, x, _ = torch.where(mask_to)
            x1 = max(0, x.min().item() - padding)
            y1 = max(0, y.min().item() - padding)
            x2 = min(img_to.shape[2], x.max().item() + padding)
            y2 = min(img_to.shape[1], y.max().item() + padding)
            cm_ref = img_to[:, y1:y2, x1:x2, :]

            _, y, x, _ = torch.where(output_mask)
            x1 = max(0, x.min().item() - padding)
            y1 = max(0, y.min().item() - padding)
            x2 = min(output.shape[2], x.max().item() + padding)
            y2 = min(output.shape[1], y.max().item() + padding)
            cm_image = output[:, y1:y2, x1:x2, :]

            if is_mathcolor:               
                normalized = cm.transfer(
                    src=Normalizer(cm_image[0].numpy()).type_norm(),
                    ref=Normalizer(cm_ref[0].numpy()).type_norm(),
                    method="mkl",
                )
                normalized = torch.from_numpy(normalized).unsqueeze(0)
            else:
                normalized = cm_image

            factor = 0.8

            output[:, y1 : y1 + cm_image.shape[1], x1 : x1 + cm_image.shape[2], :] = (
                factor * normalized + (1 - factor) * cm_image
            )

            if include_background:
                # 包含源图像的其他部分
                # 我们直接使用image_from[i]的tensor版本，并确保它与output尺寸匹配
                img_from = image_from[i].unsqueeze(0)
                
                # 确保img_from与output尺寸匹配
                if img_from.shape[1:3] != output.shape[1:3]:
                    # 在tensor上使用F.interpolate进行调整大小
                    img_from = torch.nn.functional.interpolate(
                        img_from.permute(0, 3, 1, 2),  # [B,C,H,W]格式
                        size=(output.shape[1], output.shape[2]),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)  # 返回到[B,H,W,C]格式
                
                output_image = output * output_mask + img_from * (1 - output_mask)
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