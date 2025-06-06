import cv2
import numpy as np
from PIL import Image
import torch

from .utils.image_convert import np2tensor, np2mask,  tensor2np
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

    RETURN_TYPES = ('ROTATE_INFO',)
    RETURN_NAMES = ('rotate_info',)
    FUNCTION = 'align'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据图像中的人脸进行旋转对齐'

    def align(self, analysis_models, image_from, expand=True, simple_angle=False, image_to=None):
        rotate_info = {}
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

        # 创建一个全白图像作为mask的基础
        white_image = np.ones_like(original_image) * 255

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
        
        # 旋转白色图像以创建mask
        aligned_white = cv2.warpAffine(white_image, M, new_size, flags=cv2.INTER_LINEAR)
        
        # 将旋转后的白色图像转换为mask
        # 将非255的部分（旋转后产生的黑边）设为1，其余设为0
        mask = np.zeros(aligned_white.shape[:2], dtype=np.float32)
        mask[aligned_white[:,:,0] < 255] = 1.0
        
        # 转换为ComfyUI格式的mask
        mask_tensor = np2mask(mask)

        # 转换为张量
        aligned_image_tensor = np2tensor(aligned_image).unsqueeze(0)

        if is_flipped:
            rotation_angle += 180

        rotate_info['aligned_image'] = aligned_image_tensor
        rotate_info['aligned_mask'] = mask_tensor
        rotate_info['rotation_angle'] = rotation_angle
        rotate_info['inverse_rotation_angle'] = 360 - rotation_angle

        return (rotate_info,)


class ExtractRotateInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'rotate_info': ('ROTATE_INFO',),
            },
        }
        
    RETURN_TYPES = ('IMAGE', 'MASK', 'FLOAT', 'FLOAT')
    RETURN_NAMES = ('aligned_image', 'aligned_mask', 'rotation_angle', 'inverse_rotation_angle')
    FUNCTION = 'extract'
    CATEGORY = _CATEGORY
    DESCRIPTION = '从旋转信息中提取旋转后的图像和mask'

    def extract(self, rotate_info):
        return (rotate_info['aligned_image'], rotate_info['aligned_mask'], rotate_info['rotation_angle'], rotate_info['inverse_rotation_angle'])