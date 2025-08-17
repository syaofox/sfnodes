import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import folder_paths

from PIL import Image, ImageDraw, ImageFont, ImageColor
from .utils.insightface_utils import InsightFace
from insightface.app import FaceAnalysis
from .utils.image_convert import image_to_tensor, tensor_to_image
from .utils.mask_utils import blur_mask, fill_holes, invert_mask, expand_mask, mask_process

_CATEGORY = "sfnodes/face_analysis"

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

def mask_from_landmarks(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.float64)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, points, color=1)

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



class FaceSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "area": (["face", "main_features", "eyes", "left_eye", "right_eye", "nose", "mouth", "face+forehead (if available)"], ),
                
            },
            "optional": {
                "mask_params": ("MASKPARAMS",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "image", "seg_mask", "seg_image", "x", "y", "width", "height")
    FUNCTION = "segment"
    CATEGORY = _CATEGORY

    def segment(self, analysis_models, image, area, mask_params=None):
        steps = image.shape[0]

        if steps > 1:
            pbar = ProgressBar(steps)

        out_mask = []
        out_image = []
        out_seg_mask = []
        out_seg_image = []
        out_x = []
        out_y = []
        out_w = []
        out_h = []

        for img in image:       
            face = tensor_to_image(img)

            if face is None:
                print(f"\033[96mNo face detected at frame {len(out_image)}\033[0m")
                img = torch.zeros_like(img)
                mask = img.clone()[:,:,:1]
                out_mask.append(mask)
                out_image.append(img)
                out_seg_mask.append(mask[:8,:8,:])
                out_seg_image.append(img[:8,:8,:])
                out_x.append(0)
                out_y.append(0)
                continue

            landmarks = analysis_models.get_landmarks(face, extended_landmarks=("forehead" in area))

            if landmarks is None:
                print(f"\033[96mNo landmarks detected at frame {len(out_image)}\033[0m")
                img = torch.zeros_like(img)
                mask = img.clone()[:,:,:1]
                out_mask.append(mask)
                out_image.append(img)
                out_seg_mask.append(mask[:8,:8,:])
                out_seg_image.append(img[:8,:8,:])
                out_x.append(0)
                out_y.append(0)
                continue

            if area == "face":
                landmarks = landmarks[-2]
            elif area == "eyes":
                landmarks = landmarks[2]
            elif area == "left_eye":
                landmarks = landmarks[3]
            elif area == "right_eye":
                landmarks = landmarks[4]
            elif area == "nose":
                landmarks = landmarks[5]
            elif area == "mouth":
                landmarks = landmarks[6]
            elif area == "main_features":
                landmarks = landmarks[1]
            elif "forehead" in area:
                landmarks = landmarks[-1]

            #mask = np.zeros(face.shape[:2], dtype=np.float64)
            #points = cv2.convexHull(landmarks)
            #cv2.fillConvexPoly(mask, points, color=1)

            mask = mask_from_landmarks(face, landmarks)
            mask = image_to_tensor(mask).unsqueeze(0).squeeze(-1).clamp(0, 1).to(device=img.device)

            _, y, x = torch.where(mask)
            x1, x2 = x.min().item(), x.max().item()
            y1, y2 = y.min().item(), y.max().item()
            smooth = int(min(max((x2 - x1), (y2 - y1)) * 0.2, 99))

            if smooth > 1:
                if smooth % 2 == 0:
                    smooth+= 1
                mask = T.functional.gaussian_blur(mask.bool().unsqueeze(1), smooth).squeeze(1).float()
            
            mask = mask_process(mask, mask_params)

            # extract segment from image
            y, x, _ = torch.where(mask)
            x1, x2 = x.min().item(), x.max().item()
            y1, y2 = y.min().item(), y.max().item()
            segment_mask = mask[y1:y2, x1:x2, :]
            segment_image = img[y1:y2, x1:x2, :]
            
            img = img * mask.repeat(1, 1, 3)

            out_mask.append(mask)
            out_image.append(img)
            out_seg_mask.append(segment_mask)
            out_seg_image.append(segment_image)
            out_x.append(x1)
            out_y.append(y1)

            if steps > 1:
                pbar.update(1)
        
        out_mask = torch.stack(out_mask).squeeze(-1)
        out_image = torch.stack(out_image)

        # find the max size of out_seg_image
        max_w = max([img.shape[1] for img in out_seg_image])
        max_h = max([img.shape[0] for img in out_seg_image])
        pad_left = [(max_w - img.shape[1])//2 for img in out_seg_image]
        pad_right = [max_w - img.shape[1] - pad_left[i] for i, img in enumerate(out_seg_image)]
        pad_top = [(max_h - img.shape[0])//2 for img in out_seg_image]
        pad_bottom = [max_h - img.shape[0] - pad_top[i] for i, img in enumerate(out_seg_image)]
        out_seg_image = [F.pad(img.unsqueeze(0).permute([0,3,1,2]), (pad_left[i], pad_right[i], pad_top[i], pad_bottom[i])) for i, img in enumerate(out_seg_image)]
        out_seg_mask = [F.pad(mask.unsqueeze(0).permute([0,3,1,2]), (pad_left[i], pad_right[i], pad_top[i], pad_bottom[i])) for i, mask in enumerate(out_seg_mask)]

        out_seg_image = torch.cat(out_seg_image).permute([0,2,3,1])
        out_seg_mask = torch.cat(out_seg_mask).squeeze(1)

        if len(out_x) == 1:
            out_x = out_x[0]
            out_y = out_y[0]

        out_w = out_seg_image.shape[2]
        out_h = out_seg_image.shape[1]

        return (out_mask, out_image, out_seg_mask, out_seg_image, out_x, out_y, out_w, out_h)