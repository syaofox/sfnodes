import cv2
import numpy as np
from PIL import Image
from .utils.image_convert import (
    mask2tensor,
    np2tensor,
    np2mask,
    tensor2mask,
    tensor2np,
    rescale_image,
)

_CATEGORY = "sfnodes/face_analysis"


class AlignImageByFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_from": ("IMAGE",),
                "simple_angle": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "是否简化角度，如果为True，则只考虑90度、180度、270度、360度",
                    },
                ),
                "expand": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "是否扩展图像，如果为True，则扩展图像以包含整个人脸",
                    },
                ),
                "angle": (
                    "INT",
                    {
                        "default": 0,
                        "min": -360,
                        "max": 360,
                        "step": 1,
                        "tooltip": "旋转角度，范围为-360到360，负数表示顺时针旋转,正数表示逆时针旋转",
                    },
                ),
                "threshold": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 14096,
                        "step": 1,
                        "tooltip": "黑色边框阈值，范围为0到14096，步长为1",
                    },
                ),
                "resize": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否调整旋转还原后的图像为原始大小"},
                ),
                "rotate_method": (
                    ["INTER_CUBIC", "INTER_LINEAR", "INTER_AREA"],
                    {
                        "default": "INTER_CUBIC",
                        "tooltip": "旋转方法,INTER_CUBIC为双三次插值,INTER_LINEAR为双线性插值,INTER_AREA为区域插值",
                    },
                ),
            },
            "optional": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "image_to": ("IMAGE",),
                "face_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "指定要使用的人脸索引，从0开始",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "ROTATION_INFO")
    RETURN_NAMES = ("aligned_image", "rotation_info")
    FUNCTION = "align"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据图像中的人脸进行旋转对齐"

    def align(
        self,
        expand=True,
        angle=0,
        threshold=10,
        simple_angle=False,
        image_to=None,
        resize=False,
        rotate_method="INTER_CUBIC",
        analysis_models=None,
        image_from=None,
        face_index=0,  # 新增参数
    ):
        source_image = tensor2np(image_from[0])
        original_width, original_height = source_image.shape[:2]

        def find_nearest_angle(angle):
            angles = [-360, -270, -180, -90, 0, 90, 180, 270, 360]
            normalized_angle = angle % 360
            return min(
                angles,
                key=lambda x: min(
                    abs(x - normalized_angle),
                    abs(x - normalized_angle - 360),
                    abs(x - normalized_angle + 360),
                ),
            )

        def calculate_angle(shape):
            left_eye, right_eye = shape[:2]
            return float(
                np.degrees(
                    np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])
                )
            )

        def detect_face(img, flip=False):
            if flip:
                img = Image.fromarray(img).rotate(
                    180, expand=expand, resample=Image.Resampling.BICUBIC
                )
                img = np.array(img)
            # 修改调用，传递face_index参数
            face_shape = (
                analysis_models.get_keypoints(img, face_index=face_index)
                if analysis_models
                else None
            )
            return face_shape, img

        is_flipped = False

        if analysis_models is None:
            rotation_angle = angle
        else:
            # 尝试检测人脸，如果失败则翻转图像再次尝试
            face_shape, processed_image = detect_face(source_image)
            if face_shape is None:
                face_shape, processed_image = detect_face(source_image, flip=True)
                is_flipped = True
                if face_shape is None:
                    raise Exception("无法在图像中检测到人脸。")

            rotation_angle = calculate_angle(face_shape)
            if simple_angle:
                rotation_angle = find_nearest_angle(rotation_angle)

        # 如果提供了目标图像，计算相对旋转角度
        if image_to is not None:
            # 修改调用，传递face_index参数
            target_shape = (
                analysis_models.get_keypoints(
                    tensor2np(image_to[0]), face_index=face_index
                )
                if analysis_models
                else None
            )
            if target_shape is not None:
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

        if rotate_method == "INTER_CUBIC":
            flags = cv2.INTER_CUBIC
        elif rotate_method == "INTER_LINEAR":
            flags = cv2.INTER_LINEAR
        elif rotate_method == "INTER_AREA":
            flags = cv2.INTER_AREA

        aligned_image = cv2.warpAffine(original_image, M, new_size, flags=flags)

        # 旋转白色图像以创建mask
        aligned_white = cv2.warpAffine(white_image, M, new_size, flags=flags)

        # 将旋转后的白色图像转换为mask
        # 将非255的部分（旋转后产生的黑边）设为1，其余设为0
        mask = np.zeros(aligned_white.shape[:2], dtype=np.float32)
        mask[aligned_white[:, :, 0] < 255] = 1.0

        # 转换为ComfyUI格式的mask
        mask_tensor = np2mask(mask)

        # 转换为张量
        aligned_image_tensor = np2tensor(aligned_image).unsqueeze(0)

        if is_flipped:
            rotation_angle += 180

        rotation_info = {
            "rotation_angle": rotation_angle,
            "inverse_rotation_angle": 360 - rotation_angle,
            "mask": mask_tensor,
            "expand": expand,
            "threshold": threshold,
            "original_width": original_width,
            "original_height": original_height,
            "resize": resize,
        }

        return (aligned_image_tensor, rotation_info)


class RestoreRotatedImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rotation_info": ("ROTATION_INFO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将旋转后的图像恢复到原始方向和大小，去除黑边"

    def restore(self, image, rotation_info):
        # 根据mask，填充image mask部分黑色
        mask = rotation_info["mask"]

        if mask is not None:
            image_width = image.shape[2]
            image_height = image.shape[1]

            mask_image = mask2tensor(mask)
            mask_resize = rescale_image(mask_image, image_width, image_height)
            mask = tensor2mask(mask_resize)

            image_np = tensor2np(image[0])
            height, width = image_np.shape[:2]
            center = (width / 2, height / 2)

            # 处理mask的维度，确保与图像匹配
            # 如果mask有多余的批次和通道维度，需要去除这些维度
            if len(mask.shape) == 4:  # [batch, channel, height, width]
                mask = mask.squeeze(0).squeeze(0)  # 去除批次和通道维度
            elif (
                len(mask.shape) == 3
            ):  # [batch, height, width] 或 [channel, height, width]
                mask = mask.squeeze(0)  # 去除第一个维度

            # 创建布尔掩码
            bool_mask = mask > 0.0

            # 扩展mask维度以匹配图像通道
            bool_mask_expanded = bool_mask.unsqueeze(-1).expand(height, width, 3)

            # 将mask区域填充为黑色
            image_np[bool_mask_expanded] = 0

        else:
            image_np = tensor2np(image[0])

        if rotation_info["expand"]:
            # 计算新图像的尺寸
            rot_mat = cv2.getRotationMatrix2D(
                center, rotation_info["inverse_rotation_angle"], 1.0
            )
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            new_width = int(height * abs_sin + width * abs_cos)
            new_height = int(height * abs_cos + width * abs_sin)

            # 调整旋转矩阵
            rot_mat[0, 2] += (new_width / 2) - center[0]
            rot_mat[1, 2] += (new_height / 2) - center[1]

            # 执行旋转
            rotated_image = cv2.warpAffine(
                image_np, rot_mat, (new_width, new_height), flags=cv2.INTER_CUBIC
            )
        else:
            # 不扩展图像尺寸的旋转
            rot_mat = cv2.getRotationMatrix2D(
                center, rotation_info["inverse_rotation_angle"], 1.0
            )
            rotated_image = cv2.warpAffine(
                image_np, rot_mat, (width, height), flags=cv2.INTER_CUBIC
            )

        # 转换回tensor格式
        rotated_tensor = np2tensor(rotated_image).unsqueeze(0)

        img = tensor2np(rotated_tensor[0])
        img = Image.fromarray(img)
        gray_image = img.convert("L")

        binary_image = gray_image.point(
            lambda x: 255 if x > rotation_info["threshold"] else 0
        )
        bbox = binary_image.getbbox()

        if bbox:
            cropped_image = img.crop(bbox)
        else:
            cropped_image = img

        if rotation_info["resize"]:
            cropped_image = cropped_image.resize(
                (rotation_info["original_height"], rotation_info["original_width"]),
                Image.Resampling.LANCZOS,
            )

        cropped_image = np2tensor(cropped_image).unsqueeze(0)
        return (cropped_image, rotation_info)


class ExtractRotationInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rotation_info": ("ROTATION_INFO",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "MASK", "INT", "INT")
    RETURN_NAMES = (
        "rotation_angle",
        "inverse_rotation_angle",
        "mask",
        "original_width",
        "original_height",
    )
    FUNCTION = "extract"
    CATEGORY = _CATEGORY
    DESCRIPTION = "提取旋转信息"

    def extract(self, rotation_info):
        return (
            rotation_info["rotation_angle"],
            rotation_info["inverse_rotation_angle"],
            rotation_info["mask"],
            rotation_info["original_width"],
            rotation_info["original_height"],
        )
