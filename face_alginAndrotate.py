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
                'threshold': (
                    'INT',
                    {'default': 10, 'min': 0, 'max': 14096, 'step': 1, 'tooltip': '黑色边框阈值，范围为0到14096，步长为1'},
                ),
                'resize': ('BOOLEAN', {'default': False, 'tooltip': '是否调整旋转还原后的图像为原始大小'}),
            },
            'optional': {
                'image_to': ('IMAGE',),
            },
        }

    RETURN_TYPES = ('IMAGE', 'ROTATION_INFO')
    RETURN_NAMES = ('aligned_image', 'rotation_info')
    FUNCTION = 'align'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据图像中的人脸进行旋转对齐'

    def align(self, analysis_models, image_from, expand=True, threshold=10, simple_angle=False, image_to=None, resize=False):
        source_image = tensor2np(image_from[0])
        original_width, original_height = source_image.shape[:2]

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

        rotation_info = {
            'rotation_angle': rotation_angle,
            'inverse_rotation_angle': 360 - rotation_angle,
            'mask': mask_tensor,
            'expand': expand,
            'threshold': threshold,
            'original_width': original_width,
            'original_height': original_height,
            'resize': resize,
        }

        return (aligned_image_tensor, rotation_info)


class RestoreRotatedImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'rotation_info': ('ROTATION_INFO',),
            }
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('restored_image',)
    FUNCTION = 'restore'
    CATEGORY = _CATEGORY
    DESCRIPTION = '将旋转后的图像恢复到原始方向和大小，去除黑边'

    def restore(self, image, rotation_info):
        image_np = tensor2np(image[0])
  

        height, width = image_np.shape[:2]
        center = (width / 2, height / 2)

        # 根据mask，填充image mask部分黑色
        mask = rotation_info['mask']
        
        # 检查mask是否为None
        if mask is None:
            print("警告: mask为None，跳过mask应用")
        else:
            # 处理mask的维度，确保与图像匹配
            # 如果mask有多余的批次和通道维度，需要去除这些维度
            if len(mask.shape) == 4:  # [batch, channel, height, width]
                mask = mask.squeeze(0).squeeze(0)  # 去除批次和通道维度
            elif len(mask.shape) == 3:  # [batch, height, width] 或 [channel, height, width]
                mask = mask.squeeze(0)  # 去除第一个维度
                
            # 再次检查维度是否匹配
            if mask.shape[0] != height or mask.shape[1] != width:
                print(f"警告: 处理后mask尺寸 {mask.shape} 与图像尺寸 ({height}, {width}) 仍不匹配，跳过mask应用")
            else:
                # 创建布尔掩码
                bool_mask = (mask > 0.0)
                
                # 扩展mask维度以匹配图像通道
                bool_mask_expanded = bool_mask.unsqueeze(-1).expand(height, width, 3)
                
                # 将mask区域填充为黑色
                image_np[bool_mask_expanded] = 0

        if rotation_info['expand']:
            # 计算新图像的尺寸
            rot_mat = cv2.getRotationMatrix2D(center, rotation_info['inverse_rotation_angle'], 1.0)
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            new_width = int(height * abs_sin + width * abs_cos)
            new_height = int(height * abs_cos + width * abs_sin)

            # 调整旋转矩阵
            rot_mat[0, 2] += (new_width / 2) - center[0]
            rot_mat[1, 2] += (new_height / 2) - center[1]

            # 执行旋转
            rotated_image = cv2.warpAffine(image_np, rot_mat, (new_width, new_height), flags=cv2.INTER_CUBIC)
        else:
            # 不扩展图像尺寸的旋转
            rot_mat = cv2.getRotationMatrix2D(center, rotation_info['inverse_rotation_angle'], 1.0)
            rotated_image = cv2.warpAffine(image_np, rot_mat, (width, height), flags=cv2.INTER_CUBIC)

        # 转换回tensor格式
        rotated_tensor = np2tensor(rotated_image).unsqueeze(0)

       

        img = tensor2np(rotated_tensor[0])
        img = Image.fromarray(img)
        gray_image = img.convert('L')

        binary_image = gray_image.point(lambda x: 255 if x > rotation_info['threshold'] else 0)
        bbox = binary_image.getbbox()

        if bbox:
            cropped_image = img.crop(bbox)
        else:
            cropped_image = img

        if rotation_info['resize']:
            cropped_image = cropped_image.resize(( rotation_info['original_height'], rotation_info['original_width']), Image.Resampling.LANCZOS)

        cropped_image = np2tensor(cropped_image).unsqueeze(0)
        return (cropped_image, rotation_info)



class ExtractRotationInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'rotation_info': ('ROTATION_INFO',),
            }
        }

    RETURN_TYPES = ('FLOAT', 'FLOAT', 'MASK', 'INT', 'INT')
    RETURN_NAMES = ('rotation_angle', 'inverse_rotation_angle', 'mask', 'original_width', 'original_height')
    FUNCTION = 'extract'
    CATEGORY = _CATEGORY
    DESCRIPTION = '提取旋转信息'

    def extract(self, rotation_info):
        return (rotation_info['rotation_angle'], rotation_info['inverse_rotation_angle'], rotation_info['mask'], rotation_info['original_width'], rotation_info['original_height'])