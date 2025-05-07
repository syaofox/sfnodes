import math
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFilter
from comfy.utils import  common_upscale
from .utils.image_convert import np2tensor, pil2mask, pil2tensor,  tensor2np, tensor2pil
from .utils.insightface_utils import InsightFace


_CATEGORY = 'sfnodes/face_analysis'



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
                'face_index': ('INT', {'default': -1, 'min': -1, 'max': 4096, 'step': 1, 'tooltip': '选择要裁剪的人脸索引，-1表示自动选择最大人脸'}),
                'rescale_mode': (['sdxl', 'sd15', 'sdxl+', 'sd15+', 'none', 'custom'], {'default': 'sdxl', 'tooltip': '选择缩放模式，sdxl: 缩放到1024x1024像素; sd15: 缩放到512x512像素; sdxl+: 缩放到1024x1280像素; sd15+: 缩放到512x768像素; none: 不缩放; custom: 使用自定义的像素数'}),
                'custom_megapixels': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 16.0, 'step': 0.01, 'tooltip': '设置自定义的像素数，如果选择custom，则使用自定义的像素数'}),
            },
        }

    RETURN_TYPES = ('IMAGE', 'BOUNDINGINFO')
    RETURN_NAMES = ('cutout_image', 'bounding_info')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '切下人脸并进行缩放'

    def execute(self, analysis_models, image, padding, padding_percent, rescale_mode, custom_megapixels, face_index=-1):
        target_size = self._get_target_size(rescale_mode, custom_megapixels)

        img = image[0]

        pil_image = tensor2pil(img)

        faces, x_coords, y_coords, widths, heights = analysis_models.get_bbox(pil_image, padding, padding_percent)

        face_count = len(faces)
        if face_count == 0:
            raise Exception('未在图像中检测到人脸。')

        if face_index == -1:
            face_index = 0

        face_index = min(face_index, face_count - 1)

        face = faces[face_index]
        x = x_coords[face_index]
        y = y_coords[face_index]
        w = widths[face_index]
        h = heights[face_index]

        scale_factor = 1

        if target_size > 0:
            scale_factor = math.sqrt(target_size / (w * h))
            new_width = round(w * scale_factor)
            new_height = round(h * scale_factor)
            face = self._rescale_image(face, new_width, new_height)

        bounding_info = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'scale_factor': scale_factor,
        }

        return (face, bounding_info)

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
                'destination': ('IMAGE',),
                'source': ('IMAGE',),
                'bounding_info': ('BOUNDINGINFO',),
                'margin': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1, 'tooltip': '设置图像的边距像素数'}),
                'margin_percent': ('FLOAT', {'default': 0.10, 'min': 0.0, 'max': 2.0, 'step': 0.05, 'tooltip': '设置图像的边距百分比'}),
                'blur_radius': ('INT', {'default': 10, 'min': 0, 'max': 4096, 'step': 1}),
            },
        }

    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('image', 'mask')
    FUNCTION = 'paste'
    CATEGORY = _CATEGORY
    DESCRIPTION = '将人脸图像贴回原图'

    @staticmethod
    def create_soft_edge_mask(size, margin, blur_radius):
        mask = Image.new('L', size, 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(((0, 0), size), outline='black', width=margin)
        return mask.filter(ImageFilter.GaussianBlur(blur_radius))

    def paste(self, destination, source, bounding_info, margin, margin_percent, blur_radius):
        if not bounding_info:
            return destination, None

        destination = tensor2pil(destination[0])
        source = tensor2pil(source[0])

        if bounding_info.get('scale_factor', 1) != 1:
            new_size = (bounding_info['width'], bounding_info['height'])
            source = source.resize(new_size, resample=Image.Resampling.LANCZOS)

        ref_size = max(source.width, source.height)
        margin_border = int(ref_size * margin_percent) + margin

        mask = self.create_soft_edge_mask(source.size, margin_border, blur_radius)

        position = (bounding_info['x'], bounding_info['y'])
        destination.paste(source, position, mask)

        return pil2tensor(destination), pil2mask(mask)


class ExtractBoundingBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'bounding_info': ('BOUNDINGINFO',),
            },
        }

    RETURN_TYPES = ('INT', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('x', 'y', 'width', 'height')
    FUNCTION = 'extract'
    CATEGORY = _CATEGORY
    DESCRIPTION = '从边界框信息中提取坐标和尺寸'

    def extract(self, bounding_info):
        return (bounding_info['x'], bounding_info['y'], bounding_info['width'], bounding_info['height'])




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
    CATEGORY = "FaceAnalysis"

    def load_models(self, library, provider):
        out = {}

        if library == "insightface":
            out = InsightFace(provider=provider)
        elif library == "auraface":
            out = InsightFace(provider=provider, name="auraface")
        else:
            raise Exception(f"未知的库: {library}")

        return (out, )
