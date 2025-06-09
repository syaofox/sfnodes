import cv2
import numpy as np
import os
import torch
import torchvision.transforms.v2 as T
import folder_paths

from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer

from comfy.utils import ProgressBar
from PIL import Image, ImageDraw, ImageFont, ImageColor

from .utils.image_convert import tensor_to_image, image_to_tensor
from .utils.insightface_utils import InsightFace
from .utils.mask_utils import expand_mask, blur_mask, fill_holes
from insightface.app import FaceAnalysis


_CATEGORY = "sfnodes/face_analysis"

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")


def mask_from_landmarks(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.float64)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, points, color=(1,))
    return mask


class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["antelopev2", "buffalo_l"],
                    {"tooltip": "选择要加载的模型"},
                ),
                "provider": (
                    ["CPU", "CUDA", "ROCM", "CoreML"],
                    {"tooltip": "选择要使用的提供者"},
                ),
            },
        }

    RETURN_TYPES = (
        "FACEANALYSIS",
        "ANALYSIS_MODELS",
    )
    RETURN_NAMES = (
        "insightface",
        "analysis_models",
    )
    FUNCTION = "load_insight_face"
    CATEGORY = _CATEGORY

    def load_insight_face(self, model_name, provider):
        model = FaceAnalysis(
            name=model_name,
            root=INSIGHTFACE_DIR,
            providers=[
                provider + "ExecutionProvider",
            ],
        )  # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        out = InsightFace(model)

        return (
            model,
            out,
        )


class FaceEmbedDistance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "reference": ("IMAGE", {"tooltip": "参考图像"}),
                "image": ("IMAGE", {"tooltip": "待分析图像"}),
                "similarity_metric": (
                    ["L2_norm", "cosine", "euclidean"],
                    {"tooltip": "相似度度量方式"},
                ),
                "filter_thresh": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "min": 0.001,
                        "max": 100.0,
                        "step": 0.001,
                        "tooltip": "过滤阈值",
                    },
                ),
                "filter_best": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "过滤最佳匹配数量",
                    },
                ),
                "generate_image_overlay": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "是否生成图像叠加"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "distance")
    FUNCTION = "analize"
    CATEGORY = _CATEGORY

    def analize(
        self,
        analysis_models,
        reference,
        image,
        similarity_metric,
        filter_thresh,
        filter_best,
        generate_image_overlay=True,
    ):
        if generate_image_overlay:
            font = ImageFont.truetype(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "Inconsolata.otf"
                ),
                32,
            )
            background_color = ImageColor.getrgb("#000000AA")
            txt_height = font.getmask("Q").getbbox()[3] + font.getmetrics()[1]

        if filter_thresh == 0.0:
            filter_thresh = analysis_models.thresholds[similarity_metric]

        # you can send multiple reference images in which case the embeddings are averaged
        ref = []
        for i in reference:
            ref_emb = analysis_models.get_embeds(
                np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert("RGB"))
            )
            if ref_emb is not None:
                ref.append(torch.from_numpy(ref_emb))

        if ref == []:
            raise Exception("No face detected in reference image")

        ref = torch.stack(ref)
        ref = np.array(torch.mean(ref, dim=0))

        out = []
        out_dist = []

        for i in image:
            img = np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert("RGB"))

            img = analysis_models.get_embeds(img)

            if img is None:  # No face detected
                dist = 100.0
                norm_dist = 0
            else:
                if np.array_equal(ref, img):  # Same face
                    dist = 0.0
                    norm_dist = 0.0
                else:
                    if similarity_metric == "L2_norm":
                        # dist = euclidean_distance(ref, img, True)
                        ref = ref / np.linalg.norm(ref)
                        img = img / np.linalg.norm(img)
                        dist = np.float64(np.linalg.norm(ref - img))
                    elif similarity_metric == "cosine":
                        dist = np.float64(
                            1
                            - np.dot(ref, img)
                            / (np.linalg.norm(ref) * np.linalg.norm(img))
                        )
                        # dist = cos_distance(ref, img)
                    else:
                        # dist = euclidean_distance(ref, img)
                        dist = np.float64(np.linalg.norm(ref - img))

                    norm_dist = min(
                        1.0, 1 / analysis_models.thresholds[similarity_metric] * dist
                    )

            if dist <= filter_thresh:
                print(
                    f"\033[96mFace Analysis: value: {dist}, normalized: {norm_dist}\033[0m"
                )

                if generate_image_overlay:
                    tmp = T.ToPILImage()(i.permute(2, 0, 1)).convert("RGBA")
                    txt = Image.new(
                        "RGBA", (image.shape[2], txt_height), color=background_color
                    )
                    draw = ImageDraw.Draw(txt)
                    draw.text(
                        (0, 0),
                        f"VALUE: {round(dist, 3)} | DIST: {round(norm_dist, 3)}",
                        font=font,
                        fill=(255, 255, 255, 255),
                    )
                    composite = Image.new("RGBA", tmp.size)
                    composite.paste(txt, (0, tmp.height - txt.height))
                    composite = Image.alpha_composite(tmp, composite)
                    out.append(T.ToTensor()(composite).permute(1, 2, 0))
                else:
                    out.append(i)

                out_dist.append(dist)

        if not out:
            raise Exception("No image matches the filter criteria.")

        out = torch.stack(out)

        # filter out the best matches
        if filter_best > 0:
            filter_best = min(filter_best, len(out))
            out_dist, idx = torch.topk(
                torch.tensor(out_dist), filter_best, largest=False
            )
            out = out[idx]
            out_dist = out_dist.cpu().numpy().tolist()

        if out.shape[3] > 3:
            out = out[:, :, :, :3]

        return (
            out,
            out_dist,
        )


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

            # 使用修改后的expand_mask函数，它会保持输入输出维度一致
            grow_count = (
                int(grow_percent * max(output_mask.shape[1], output_mask.shape[2]))
                + grow
            )
            if grow_count != 0:
                output_mask = expand_mask(
                    output_mask.squeeze(-1), grow_count, grow_tapered
                )
            else:
                output_mask = output_mask.squeeze(-1)

            if fill:
                output_mask = fill_holes(output_mask)

            if blur > 0:
                output_mask = blur_mask(output_mask, blur)

            # 添加通道维度用于乘法运算
            output_mask = output_mask.unsqueeze(-1)

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
