import math
import cv2
import numpy as np
import os
import torch
import zipfile
from pathlib import Path
import torchvision.transforms.v2 as T
import folder_paths

from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer

from .utils.downloader import download_model
from comfy.utils import ProgressBar
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter
from comfy.utils import  common_upscale
from .utils.image_convert import np2tensor, pil2mask, pil2tensor,  tensor2np, tensor2pil, tensor_to_image, image_to_tensor
from .utils.insightface_utils import InsightFace
from .utils.mask_utils import expand_mask, blur_mask, fill_holes

_CATEGORY = 'sfnodes/face_analysis'

def mask_from_landmarks(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.float64)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, points, color=(1,))
    return mask


class AlignImageByFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'analysis_models': ('ANALYSIS_MODELS',),
                'image_from': ('IMAGE',),
                'expand': ('BOOLEAN', {'default': True, 'tooltip': '是否扩展图像，如果为True，则扩展图像以包含整个人脸'}),
                'simple_angle': ('BOOLEAN', {'default': False, 'tooltip': '是否简化角度，如果为True，则只考虑90度、180度、270度、360度'}),
            },
            'optional': {
                'image_to': ('IMAGE',),
            },
        }

    RETURN_TYPES = ('IMAGE', 'FLOAT', 'FLOAT')
    RETURN_NAMES = ('aligned_image', 'rotation_angle', 'inverse_rotation_angle')
    FUNCTION = 'align'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据图像中的人脸进行旋转对齐'

    def align(self, analysis_models, image_from, expand=True, simple_angle=False, image_to=None):
        source_image = tensor2np(image_from[0])

        def find_nearest_angle(angle):
            angles = [-360, -270, -180, -90, 0, 90, 180, 270, 360]
            normalized_angle = angle % 360
            return min(angles, key=lambda x: min(abs(x - normalized_angle), abs(x - normalized_angle - 360), abs(x - normalized_angle + 360)))

        def calculate_angle(shape):
            left_eye, right_eye = shape[:2]
            return float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

        def detect_face(img, flip=False):
            if flip:
                img = Image.fromarray(img).rotate(180, expand=expand, resample=Image.Resampling.BICUBIC)
                img = np.array(img)
            face_shape = analysis_models.get_keypoints(img)
            return face_shape, img

        # 尝试检测人脸，如果失败则翻转图像再次尝试
        face_shape, processed_image = detect_face(source_image)
        if face_shape is None:
            face_shape, processed_image = detect_face(source_image, flip=True)
            is_flipped = True
            if face_shape is None:
                raise Exception('无法在图像中检测到人脸。')
        else:
            is_flipped = False

        rotation_angle = calculate_angle(face_shape)
        if simple_angle:
            rotation_angle = find_nearest_angle(rotation_angle)

        # 如果提供了目标图像，计算相对旋转角度
        if image_to is not None:
            target_shape = analysis_models.get_keypoints(tensor2np(image_to[0]))
            if target_shape is not None:
                print(f'目标图像人脸关键点: {target_shape}')
                rotation_angle -= calculate_angle(target_shape)

        original_image = tensor2np(image_from[0]) if not is_flipped else processed_image

        rows, cols = original_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)

        if expand:
            # 计算新的边界以确保整个图像都包含在内
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_cols = int((rows * sin) + (cols * cos))
            new_rows = int((rows * cos) + (cols * sin))
            M[0, 2] += (new_cols / 2) - cols / 2
            M[1, 2] += (new_rows / 2) - rows / 2
            new_size = (new_cols, new_rows)
        else:
            new_size = (cols, rows)

        aligned_image = cv2.warpAffine(original_image, M, new_size, flags=cv2.INTER_LINEAR)

        # 转换为张量

        aligned_image_tensor = np2tensor(aligned_image).unsqueeze(0)

        if is_flipped:
            rotation_angle += 180

        return (aligned_image_tensor, rotation_angle, 360 - rotation_angle)


class FaceCutout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'analysis_models': ('ANALYSIS_MODELS',),
                'image': ('IMAGE',),
                'padding': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1, 'tooltip': '设置图像的填充像素数'}),
                'padding_percent': ('FLOAT', {'default': 0.1, 'min': 0.0, 'max': 2.0, 'step': 0.01, 'tooltip': '设置图像的填充百分比'}),
                'rescale_mode': (['sdxl', 'sd15', 'sdxl+', 'sd15+', 'none', 'custom'], {'default': 'sdxl', 'tooltip': '选择缩放模式，sdxl: 缩放到1024x1024像素; sd15: 缩放到512x512像素; sdxl+: 缩放到1024x1280像素; sd15+: 缩放到512x768像素; none: 不缩放; custom: 使用自定义的像素数'}),
                'custom_megapixels': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 16.0, 'step': 0.01, 'tooltip': '设置自定义的像素数，如果选择custom，则使用自定义的像素数'}),
                'margin': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1, 'tooltip': '设置贴回去图像的边距像素数'}),
                'margin_percent': ('FLOAT', {'default': 0.10, 'min': 0.0, 'max': 2.0, 'step': 0.05, 'tooltip': '设置贴回去图像的边距百分比'}),
                'blur_radius': ('INT', {'default': 10, 'min': 0, 'max': 4096, 'step': 1, 'tooltip': '设置贴回去图像的模糊半径'}),
                'is_square': ('BOOLEAN', {'default': False, 'tooltip': '是否将图像裁剪为正方形'}),
            },
        }

    RETURN_TYPES = ('BOUNDINGINFOS', 'IMAGES', 'INT',)
    RETURN_NAMES = ('bounding_infos', 'crop_images','face_count',)
    OUTPUT_IS_LIST = (True, True, False,)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '切下图像中所有人脸并进行缩放，返回所有人脸信息'

    def execute(self, analysis_models, image, padding, padding_percent, rescale_mode, custom_megapixels, margin, margin_percent, blur_radius, is_square=False):
        target_size = self._get_target_size(rescale_mode, custom_megapixels)

        img = image[0]

        pil_image = tensor2pil(img)

        faces, x_coords, y_coords, widths, heights = analysis_models.get_bbox(pil_image, padding, padding_percent)

        face_count = len(faces)
        if face_count == 0:
            raise Exception('未在图像中检测到人脸。')

        bounding_infos = []
        crop_images = []

        for i, face in enumerate(faces):
            scale_factor = 1
            
            # 如果需要正方形，调整宽度和高度
            if is_square:
                # 计算正方形边长，取宽高的最大值
                square_size = max(widths[i], heights[i])
                # 计算新的x和y坐标，使人脸居中
                center_x = x_coords[i] + widths[i] // 2
                center_y = y_coords[i] + heights[i] // 2
                new_x = center_x - square_size // 2
                new_y = center_y - square_size // 2
                
                # 确保新坐标不超出图像边界
                new_x = max(0, new_x)
                new_y = max(0, new_y)
                if new_x + square_size > pil_image.width:
                    new_x = pil_image.width - square_size
                if new_y + square_size > pil_image.height:
                    new_y = pil_image.height - square_size
                
                # 更新坐标和尺寸
                x_coords[i] = new_x
                y_coords[i] = new_y
                widths[i] = square_size
                heights[i] = square_size
                
                # 重新裁剪人脸区域为正方形
                face_np = tensor2np(image[0])
                face_crop = face_np[new_y:new_y+square_size, new_x:new_x+square_size]
                face = np2tensor(face_crop).unsqueeze(0)

            if target_size > 0:
                scale_factor = math.sqrt(target_size / (widths[i] * heights[i]))
                new_width = round(widths[i] * scale_factor)
                new_height = round(heights[i] * scale_factor)
                scaled_face = self._rescale_image(face, new_width, new_height)
            else:
                scaled_face = face

            # 为每个人脸创建单独的 BOUNDINGINFO 结构
            bounding_info = {
                'x': x_coords[i],
                'y': y_coords[i],
                'width': widths[i],
                'height': heights[i],
                'scale_factor': scale_factor,
                'margin': margin,
                'margin_percent': margin_percent,
                'blur_radius': blur_radius,
            }
            crop_images.append(scaled_face)
            
            bounding_infos.append(bounding_info)

        return (bounding_infos, crop_images, face_count,)

    @staticmethod
    def _get_target_size(rescale_mode, custom_megapixels):
        if rescale_mode == 'custom':
            return int(custom_megapixels * 1024 * 1024)
        size_map = {'sd15': 512 * 512, 'sd15+': 512 * 768, 'sdxl': 1024 * 1024, 'sdxl+': 1024 * 1280, 'none': -1}
        return size_map.get(rescale_mode, -1)

    @staticmethod
    def _rescale_image(image, width, height):
        samples = image.movedim(-1, 1)
        resized = common_upscale(samples, width, height, 'lanczos', 'disabled')
        return resized.movedim(1, -1)


class FacePaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'bounding_info': ('BOUNDINGINFO',),  
                'source_image': ('IMAGE',),
                'distination_image': ('IMAGE',),               
            },
        }

    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('image', 'mask')
    FUNCTION = 'paste'
    CATEGORY = _CATEGORY
    DESCRIPTION = '将bounding_info中的人脸图像贴回原图'

    @staticmethod
    def create_soft_edge_mask(size, margin, blur_radius):
        mask = Image.new('L', size, 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(((0, 0), size), outline='black', width=margin)
        return mask.filter(ImageFilter.GaussianBlur(blur_radius))

    def paste(self, bounding_info,source_image,  distination_image):

        # 从bounding_info中获取人脸图像和位置信息
        x = bounding_info['x']
        y = bounding_info['y']
        width = bounding_info['width']
        height = bounding_info['height']
        margin = bounding_info['margin']
        margin_percent = bounding_info['margin_percent']
        blur_radius = bounding_info['blur_radius']

        
        destination = tensor2pil(distination_image[0])
        source = tensor2pil(source_image[0])

        # 如果源图像尺寸与目标区域尺寸不匹配，进行调整
        if source.width != width or source.height != height:
            source = source.resize((width, height), resample=Image.Resampling.LANCZOS)

        ref_size = max(source.width, source.height)
        margin_border = int(ref_size * margin_percent) + margin

        mask = self.create_soft_edge_mask(source.size, margin_border, blur_radius)

        position = (x, y)
        destination.paste(source, position, mask)

        return pil2tensor(destination), pil2mask(mask)


class ExtractBoundingBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'bounding_infos': ('BOUNDINGINFOS',),
                'crop_images': ('IMAGES',),
                'index': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1, 'tooltip': '选择要解析的人脸索引'}),
            },
        }

    RETURN_TYPES = ('INT', 'INT', 'INT', 'INT', 'IMAGE', 'BOUNDINGINFO')
    RETURN_NAMES = ('x', 'y', 'width', 'height', 'crop_image', 'bounding_info')
    INPUT_IS_LIST = (True, False,)
    FUNCTION = 'extract'
    CATEGORY = _CATEGORY
    DESCRIPTION = '从边界框信息中提取指定索引的人脸坐标、尺寸和图像'

    def extract(self, bounding_infos, crop_images, index=0):
        # 确保bounding_infos是列表
        if not isinstance(bounding_infos, list):
            bounding_infos = [bounding_infos]

        # 确保index是整数
        if isinstance(index, list):
            if len(index) > 0:
                index = index[0]  # 如果index是列表，取第一个元素
            else:
                index = 0
                
        # 确保index在有效范围内
        if len(bounding_infos) == 0:
            raise Exception("边界框信息列表为空")
            
        if index >= len(bounding_infos):
            print(f"警告：索引 {index} 超出了bounding_infos的范围 {len(bounding_infos)}，使用默认索引0")
            index = 0

        bounding_info = bounding_infos[index]         
        
        # 确保bounding_info是字典类型
        if not isinstance(bounding_info, dict):
            raise Exception(f"边界框信息不是预期的字典格式: {type(bounding_info)}")
        
        # 从bounding_info中提取信息
        x = bounding_info.get('x', 0)
        y = bounding_info.get('y', 0)
        width = bounding_info.get('width', 0)
        height = bounding_info.get('height', 0)       
        crop_image = crop_images[index]
        return (x, y, width, height, crop_image, bounding_info)




class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(cls):
        libraries = []
        libraries.append("insightface")
        libraries.append("auraface")

        return {"required": {
            "library": (libraries, ),
            "provider": (["CPU", "CUDA", "DirectML", "OpenVINO", "ROCM", "CoreML"], ),
        }}

    RETURN_TYPES = ("ANALYSIS_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = _CATEGORY

    def load_models(self, library, provider):
        out = {}

        if library == "insightface":
            out = InsightFace(provider=provider)
        elif library == "auraface":
            # 判断模型是否存在Path(folder_paths.models_dir) /insightface/models/auraface/ 不存在就下载
            if not os.path.exists(Path(folder_paths.models_dir) / "insightface" / "models" / "auraface" ):
                # 下载模型压缩包并解压 https://huggingface.co/Syaofox/sfnodes/resolve/main/AuraFace-v1.zip
                download_model(
                    "https://huggingface.co/Syaofox/sfnodes/resolve/main/AuraFace-v1.zip",
                    Path(folder_paths.models_dir) / "insightface" / "models" ,
                    "AuraFace-v1.zip"
                )
                # 解压模型压缩包
                with zipfile.ZipFile(Path(folder_paths.models_dir) / "insightface" / "models" / "AuraFace-v1.zip", 'r') as zip_ref:
                    zip_ref.extractall(Path(folder_paths.models_dir) / "insightface" / "models" / "auraface")
            out = InsightFace(provider=provider, name="auraface")
        else:
            raise Exception(f"未知的库: {library}")

        return (out, )




class FaceEmbedDistance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "reference": ("IMAGE", {"tooltip": "参考图像"}),
                "image": ("IMAGE", {"tooltip": "待分析图像"}),
                "similarity_metric": (["L2_norm", "cosine", "euclidean"], {"tooltip": "相似度度量方式"}),
                "filter_thresh": ("FLOAT", { "default": 100.0, "min": 0.001, "max": 100.0, "step": 0.001, "tooltip": "过滤阈值" }),
                "filter_best": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "过滤最佳匹配数量" }),
                "generate_image_overlay": ("BOOLEAN", { "default": True, "tooltip": "是否生成图像叠加" }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "distance")
    FUNCTION = "analize"
    CATEGORY = _CATEGORY

    def analize(self, analysis_models, reference, image, similarity_metric, filter_thresh, filter_best, generate_image_overlay=True):
        if generate_image_overlay:
            font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Inconsolata.otf"), 32)
            background_color = ImageColor.getrgb("#000000AA")
            txt_height = font.getmask("Q").getbbox()[3] + font.getmetrics()[1]

        if filter_thresh == 0.0:
            filter_thresh = analysis_models.thresholds[similarity_metric]

        # you can send multiple reference images in which case the embeddings are averaged
        ref = []
        for i in reference:
            ref_emb = analysis_models.get_embeds(np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')))
            if ref_emb is not None:
                ref.append(torch.from_numpy(ref_emb))
        
        if ref == []:
            raise Exception('No face detected in reference image')

        ref = torch.stack(ref)
        ref = np.array(torch.mean(ref, dim=0))

        out = []
        out_dist = []
        
        for i in image:
            img = np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB'))

            img = analysis_models.get_embeds(img)

            if img is None: # No face detected
                dist = 100.0
                norm_dist = 0
            else:
                if np.array_equal(ref, img): # Same face
                    dist = 0.0
                    norm_dist = 0.0
                else:
                    if similarity_metric == "L2_norm":
                        #dist = euclidean_distance(ref, img, True)
                        ref = ref / np.linalg.norm(ref)
                        img = img / np.linalg.norm(img)
                        dist = np.float64(np.linalg.norm(ref - img))
                    elif similarity_metric == "cosine":
                        dist = np.float64(1 - np.dot(ref, img) / (np.linalg.norm(ref) * np.linalg.norm(img)))
                        #dist = cos_distance(ref, img)
                    else:
                        #dist = euclidean_distance(ref, img)
                        dist = np.float64(np.linalg.norm(ref - img))
                    
                    norm_dist = min(1.0, 1 / analysis_models.thresholds[similarity_metric] * dist)
           
            if dist <= filter_thresh:
                print(f"\033[96mFace Analysis: value: {dist}, normalized: {norm_dist}\033[0m")

                if generate_image_overlay:
                    tmp = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGBA')
                    txt = Image.new('RGBA', (image.shape[2], txt_height), color=background_color)
                    draw = ImageDraw.Draw(txt)
                    draw.text((0, 0), f"VALUE: {round(dist, 3)} | DIST: {round(norm_dist, 3)}", font=font, fill=(255, 255, 255, 255))
                    composite = Image.new('RGBA', tmp.size)
                    composite.paste(txt, (0, tmp.height - txt.height))
                    composite = Image.alpha_composite(tmp, composite)
                    out.append(T.ToTensor()(composite).permute(1, 2, 0))
                else:
                    out.append(i)

                out_dist.append(dist)

        if not out:
            raise Exception('No image matches the filter criteria.')
    
        out = torch.stack(out)

        # filter out the best matches
        if filter_best > 0:
            filter_best = min(filter_best, len(out))
            out_dist, idx = torch.topk(torch.tensor(out_dist), filter_best, largest=False)
            out = out[idx]
            out_dist = out_dist.cpu().numpy().tolist()
        
        if out.shape[3] > 3:
            out = out[:, :, :, :3]

        return(out, out_dist,)
    




class FaceWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image_from": ("IMAGE", ),
                "image_to": ("IMAGE", ),
                "keypoints": (["main features", "full face", "full face+forehead (if available)"], ),
                'grow': ('INT', {'default': 0, 'min': -4096, 'max': 4096, 'step': 1, 'tooltip': '设置生长值，范围为-4096到4096，步长为1'}),
                'grow_percent': (
                    'FLOAT',
                    {'default': 0.00, 'min': -2.0, 'max': 2.0, 'step': 0.01, 'tooltip': '设置生长百分比，范围为-2.0到2.0，步长为0.01'},
                ),
                'grow_tapered': ('BOOLEAN', {'default': False, 'tooltip': '是否使用锥形角'}),
                'blur': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1, 'tooltip': '设置模糊值，范围为0到4096，步长为1'}),
                'fill': ('BOOLEAN', {'default': False, 'tooltip': '是否填充孔洞'}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "warp"
    CATEGORY = _CATEGORY

    def warp(self, analysis_models, image_from, image_to, keypoints, grow, grow_percent, grow_tapered, blur, fill):

        

        if image_from.shape[0] < image_to.shape[0]:
            image_from = torch.cat([image_from, image_from[-1:].repeat((image_to.shape[0]-image_from.shape[0], 1, 1, 1))], dim=0)
        elif image_from.shape[0] > image_to.shape[0]:
            image_from = image_from[:image_to.shape[0]]

        steps = image_from.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        cm = ColorMatcher()

        result_image = []
        result_mask = []

        for i in range(steps):
            img_from = tensor_to_image(image_from[i])
            img_to = tensor_to_image(image_to[i])

            shape_from = analysis_models.get_landmarks(img_from, extended_landmarks=("forehead" in keypoints))
            shape_to = analysis_models.get_landmarks(img_to, extended_landmarks=("forehead" in keypoints))

            if shape_from is None or shape_to is None:
                print(f"\033[96mNo landmarks detected at frame {i}\033[0m")
                img = image_to[i].unsqueeze(0)
                mask = torch.zeros_like(img)[:,:,:1]
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
            output = cv2.warpAffine(img_from, matrix, (img_to.shape[1], img_to.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            mask_from = mask_from_landmarks(img_from, shape_from)
            mask_to = mask_from_landmarks(img_to, shape_to)
            output_mask = cv2.warpAffine(mask_from, matrix, (img_to.shape[1], img_to.shape[0]))

            output_mask = torch.from_numpy(output_mask).unsqueeze(0).unsqueeze(-1).float()
            mask_to = torch.from_numpy(mask_to).unsqueeze(0).unsqueeze(-1).float()
            output_mask = torch.min(output_mask, mask_to)

            output = image_to_tensor(output).unsqueeze(0)
            img_to = image_to_tensor(img_to).unsqueeze(0)
            
            # 使用修改后的expand_mask函数，它会保持输入输出维度一致
            grow_count = int(grow_percent * max(output_mask.shape[1], output_mask.shape[2])) + grow            
            if grow_count != 0:
                output_mask = expand_mask(output_mask.squeeze(-1), grow_count, grow_tapered)
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

            normalized = cm.transfer(src=Normalizer(cm_image[0].numpy()).type_norm() , ref=Normalizer(cm_ref[0].numpy()).type_norm(), method='mkl')
            normalized = torch.from_numpy(normalized).unsqueeze(0)

            factor = 0.8

            output[:, y1:y1+cm_image.shape[1], x1:x1+cm_image.shape[2], :] = factor * normalized + (1 - factor) * cm_image

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