import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .liveportrait.utils.cropper import CropperMediaPipe
from .utils.image_convert import pil2tensor
from .utils.model_manager import ModelManager

_CATEGORY = "sfnodes/face_analysis"

# 模型配置
LANDMARK_MODELS = {
    "landmark": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/landmark.onnx",
        "filename": "landmark.onnx",
        "description": "landmark.onnx",
    },
    "landmark_model": {
        "url": "https://huggingface.co/Syaofox/sfnodes/resolve/main/landmark_model.pth",
        "filename": "landmark_model.pth",
        "description": "landmark_model.pth",
    },
}


class FaceLandmarkExtractor(CropperMediaPipe):
    def extract_face_landmarks(self, img_rgb, face_index):
        landmark_info = {}
        face_result = self.lmk_extractor(img_rgb)
        if face_result is None:
            raise Exception("未在图像中检测到人脸。")
        face_landmarks = face_result[face_index]
        lmks = [
            [
                face_landmarks[index].x * img_rgb.shape[1],
                face_landmarks[index].y * img_rgb.shape[0],
            ]
            for index in range(len(face_landmarks))
        ]
        recon_ret = self.landmark_runner.run(img_rgb, np.array(lmks))
        landmark_info["landmarks"] = recon_ret["pts"]
        return landmark_info


class FaceMorph:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "源图像"}),
                "target_image": ("IMAGE", {"tooltip": "目标图像"}),
                "landmark_type": (
                    ["ALL", "OUTLINE"],
                    {"tooltip": "选择要使用的面部标志类型"},
                ),
                "align_type": (
                    ["Width", "Height", "Landmarks", "JawLine"],
                    {"tooltip": "选择对齐类型"},
                ),
                "onnx_device": (
                    ["CPU", "CUDA", "ROCM", "CoreML", "torch_gpu"],
                    {"tooltip": "选择推理设备", "default": "CPU"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_image",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据输入的源图像和目标图像，进行人脸液化变形"

    def __init__(self):
        self.landmark_extractor = None
        self.current_config = None
        self.model_manager = ModelManager(LANDMARK_MODELS)
        self.model_manager.get_model_path("landmark", sub_dir="face_morph")
        self.model_manager.get_model_path("landmark_model", sub_dir="face_morph")

    def landmark203_to_68(self, source):
        out = []
        jaw_indices = [108, 126, 144]
        for start in jaw_indices:
            out.append(source[start])
            out.extend(
                [
                    (source[start + i] * 3 + source[start + i + 1]) / 4
                    if i % 4 == 2
                    else (source[start + i] + source[start + i + 1]) / 2
                    if i % 4 == 0
                    else (source[start + i] + source[start + i + 1] * 3) / 4
                    if i % 4 == 2
                    else source[start + i]
                    for i in range(2, 17, 2)
                ]
            )

        eyebrow_indices = [(145, 162), (165, 182)]
        for start, end in eyebrow_indices:
            out.append(source[start])
            out.extend(
                [(source[start + i] + source[end - i]) / 2 for i in range(3, 10, 3)]
            )
            out.append(source[start + 10])

        nose_indices = [199, 200, 201, 189, 190, 202, 191, 192]
        out.append(source[199])
        out.append((source[199] + source[200]) / 2)
        out.extend([source[i] for i in nose_indices[1:]])

        eye_indices = [(0, 21), (24, 45)]
        for start, end in eye_indices:
            out.extend([source[i] for i in range(start, end, 4)])

        lip_indices = [
            48,
            51,
            54,
            57,
            60,
            63,
            66,
            69,
            72,
            75,
            78,
            81,
            84,
            87,
            90,
            93,
            96,
            99,
            102,
            105,
        ]
        out.extend([source[i] for i in lip_indices])

        return out

    def initialize_landmark_extractor(self, onnx_device):
        extractor_init_config = {"keep_model_loaded": True, "onnx_device": onnx_device}
        if (
            self.landmark_extractor is None
            or self.current_config != extractor_init_config
        ):
            self.current_config = extractor_init_config
            self.landmark_extractor = FaceLandmarkExtractor(**extractor_init_config)

    def process_image(self, image):
        image_np = (image.contiguous() * 255).byte().numpy()
        if self.landmark_extractor is None:
            raise ValueError("self.landmark_extractor 未初始化")
        landmark_info = self.landmark_extractor.extract_face_landmarks(image_np[0], 0)
        landmarks = self.landmark203_to_68(landmark_info["landmarks"])
        return image_np[0], np.array(landmarks[:65])

    def calculate_facial_features(self, landmarks):
        return {
            "left_eye": np.mean(landmarks[36:42], axis=0),
            "right_eye": np.mean(landmarks[42:48], axis=0),
            "jaw": landmarks[0:17],
            "center_of_jaw": np.mean(landmarks[0:17], axis=0),
        }

    def create_grid_points(self, width, height):
        x = np.linspace(0, width, 16)
        y = np.linspace(0, height, 16)
        xx, yy = np.meshgrid(x, y)
        src_points = np.column_stack((xx.ravel(), yy.ravel()))
        mask = (
            (src_points[:, 0] <= width / 8)
            | (src_points[:, 0] >= 7 * width / 8)
            | (src_points[:, 1] >= 7 * height / 8)
            | (src_points[:, 1] <= height / 8)
        )
        return src_points[mask]

    def calculate_ratios(self, landmarks):
        min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
        ratio = (max_x - min_x) / (max_y - min_y)
        middle_point = [(max_x + min_x) / 2, (max_y + min_y) / 2]
        return ratio, middle_point

    def align_width_height(
        self,
        landmarks1,
        landmarks2,
        jaw1,
        jaw2,
        src_points,
        dst_points,
        features1,
        features2,
        landmark_type,
        align_type,
    ):
        target_points = landmarks1.copy() if landmark_type == "ALL" else jaw1.copy()
        dst_points = np.append(
            dst_points, landmarks1 if landmark_type == "ALL" else jaw1, axis=0
        )

        ratio1, middle_point = self.calculate_ratios(landmarks1)
        ratio2, _ = self.calculate_ratios(landmarks2)

        if align_type == "Width":
            target_points[:, 1] = (
                target_points[:, 1] - middle_point[1]
            ) * ratio1 / ratio2 + middle_point[1]
        else:  # Height
            target_points[:, 0] = (
                target_points[:, 0] - middle_point[0]
            ) * ratio2 / ratio1 + middle_point[0]

        return np.append(src_points, target_points, axis=0), dst_points

    def align_landmarks(
        self,
        landmarks1,
        landmarks2,
        jaw1,
        jaw2,
        src_points,
        dst_points,
        features1,
        features2,
        landmark_type,
        _,
    ):
        if landmark_type == "ALL":
            middle_of_eyes1 = (features1["left_eye"] + features1["right_eye"]) / 2
            middle_of_eyes2 = (features2["left_eye"] + features2["right_eye"]) / 2
            factor = np.linalg.norm(
                features1["left_eye"] - features1["right_eye"]
            ) / np.linalg.norm(features2["left_eye"] - features2["right_eye"])
            target_points = (landmarks2 - middle_of_eyes2) * factor + middle_of_eyes1
            target_points[0:17] = (
                landmarks2[0:17] - features2["center_of_jaw"]
            ) * factor + features1["center_of_jaw"]
            dst_points = np.append(dst_points, landmarks1, axis=0)
        else:
            target_points = (jaw2 - features2["center_of_jaw"]) + features1[
                "center_of_jaw"
            ]
            dst_points = np.append(dst_points, jaw1, axis=0)

        return np.append(src_points, target_points, axis=0), dst_points

    def align_jaw_line(
        self,
        landmarks1,
        landmarks2,
        jaw1,
        jaw2,
        src_points,
        dst_points,
        features1,
        features2,
        landmark_type,
        _,
    ):
        factor = np.linalg.norm(jaw1[0] - jaw1[-1]) / np.linalg.norm(jaw2[0] - jaw2[-1])
        if landmark_type == "ALL":
            target_points = (landmarks2 - jaw2[0]) * factor + jaw1[0]
            dst_points = np.append(dst_points, landmarks1, axis=0)
        else:
            target_points = (jaw2 - jaw2[0]) * factor + jaw1[0]
            dst_points = np.append(dst_points, jaw1, axis=0)
        return np.append(src_points, target_points, axis=0), dst_points

    def warp_image(self, image, src_points, dst_points):
        height, width = image.shape[:2]
        src_points[:, [0, 1]] = src_points[:, [1, 0]]
        dst_points[:, [0, 1]] = dst_points[:, [1, 0]]

        interp = LinearNDInterpolator(src_points, dst_points)

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        map_coords = (
            interp(np.column_stack((yy.ravel(), xx.ravel())))
            .reshape(height, width, 2)
            .astype(np.float32)
        )

        return cv2.remap(
            image, map_coords[:, :, 1], map_coords[:, :, 0], cv2.INTER_LINEAR
        )

    def execute(
        self, source_image, target_image, landmark_type, align_type, onnx_device
    ):
        self.initialize_landmark_extractor(onnx_device)
        image1, landmarks1 = self.process_image(source_image)
        _, landmarks2 = self.process_image(target_image)

        features1 = self.calculate_facial_features(landmarks1)
        features2 = self.calculate_facial_features(landmarks2)

        src_points = self.create_grid_points(*image1.shape[:2][::-1])
        dst_points = src_points.copy()

        align_funcs = {
            "Width": self.align_width_height,
            "Height": self.align_width_height,
            "Landmarks": self.align_landmarks,
            "JawLine": self.align_jaw_line,
        }

        align_func = align_funcs.get(align_type)
        if align_func:
            src_points, dst_points = align_func(
                landmarks1,
                landmarks2,
                features1["jaw"],
                features2["jaw"],
                src_points,
                dst_points,
                features1,
                features2,
                landmark_type,
                align_type,
            )

        warped_image = self.warp_image(image1, src_points, dst_points)
        return (pil2tensor(warped_image),)


class FaceReshape:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像"}),
                "width_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "脸型宽度缩放比例"}),
                "height_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "脸型高度缩放比例"}),
                "preserve_features": (["是", "否"], {"default": "否", "tooltip": "是否保持眼睛和嘴巴等关键特征不变形"}),
                "onnx_device": (
                    ["CPU", "CUDA", "ROCM", "CoreML", "torch_gpu"],
                    {"tooltip": "选择推理设备", "default": "CPU"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reshaped_image",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "通过调整参数直接对脸型进行宽度和高度变形，无需参考图像"

    def __init__(self):
        self.face_morph = FaceMorph()
        self.landmark_extractor = None
        self.current_config = None
        self.model_manager = ModelManager(LANDMARK_MODELS)
        self.model_manager.get_model_path("landmark", sub_dir="face_morph")
        self.model_manager.get_model_path("landmark_model", sub_dir="face_morph")

    def initialize_landmark_extractor(self, onnx_device):
        extractor_init_config = {"keep_model_loaded": True, "onnx_device": onnx_device}
        if (
            self.landmark_extractor is None
            or self.current_config != extractor_init_config
        ):
            self.current_config = extractor_init_config
            self.landmark_extractor = FaceLandmarkExtractor(**extractor_init_config)
    
    def process_image(self, image):
        image_np = (image.contiguous() * 255).byte().numpy()
        if self.landmark_extractor is None:
            raise ValueError("self.landmark_extractor 未初始化")
        landmark_info = self.landmark_extractor.extract_face_landmarks(image_np[0], 0)
        landmarks = self.face_morph.landmark203_to_68(landmark_info["landmarks"])
        return image_np[0], np.array(landmarks[:65])

    def execute(self, image, width_scale, height_scale, preserve_features, onnx_device):
        self.initialize_landmark_extractor(onnx_device)
        img_np, landmarks = self.process_image(image)
        
        # 计算面部特征
        features = self.face_morph.calculate_facial_features(landmarks)
        
        # 创建网格点
        height, width = img_np.shape[:2]
        src_points = self.face_morph.create_grid_points(width, height)
        dst_points = src_points.copy()
        
        # 获取关键特征点位置
        chin_point = landmarks[8]  # 下巴底部点
        forehead_point = np.mean(landmarks[17:27], axis=0)  # 额头区域的平均点
        
        # 计算脸部垂直中点（不是简单平均，而是在额头和下巴之间）
        face_vertical_center = (forehead_point + chin_point) / 2
        
        # 计算面部水平中点（使用眼睛中点）
        face_horizontal_center = (features["left_eye"] + features["right_eye"]) / 2
        
        # 创建目标landmarks用于变形
        target_landmarks = landmarks.copy()
        
        # 应用宽度和高度变形
        for i in range(len(target_landmarks)):
            point = target_landmarks[i]
            
            # 处理宽度变形 - 相对于垂直中线
            x_offset = point[0] - face_horizontal_center[0]
            # 处理高度变形 - 区分上下部分
            is_upper_face = point[1] < face_vertical_center[1]
            
            # 计算点到垂直中心的距离
            y_offset = point[1] - face_vertical_center[1]
            
            # 应用变形
            if preserve_features == "是":
                # 如果保持关键特征，只变形下巴轮廓点
                is_jaw_or_cheek = i < 17  # 下巴轮廓点
                
                if is_jaw_or_cheek:
                    # 宽度变形
                    target_landmarks[i, 0] = face_horizontal_center[0] + x_offset * width_scale
                    
                    # 高度变形 - 根据点是在脸的上部还是下部来决定方向
                    if is_upper_face:
                        # 上部向上移动或压缩
                        target_landmarks[i, 1] = face_vertical_center[1] - abs(y_offset) * height_scale
                    else:
                        # 下部向下移动或压缩
                        target_landmarks[i, 1] = face_vertical_center[1] + abs(y_offset) * height_scale
            else:
                # 全部特征点都进行变形
                # 宽度变形
                target_landmarks[i, 0] = face_horizontal_center[0] + x_offset * width_scale
                
                # 高度变形 - 根据点是在脸的上部还是下部来决定方向
                if is_upper_face:
                    # 上部向上移动或压缩
                    target_landmarks[i, 1] = face_vertical_center[1] - abs(y_offset) * height_scale
                else:
                    # 下部向下移动或压缩
                    target_landmarks[i, 1] = face_vertical_center[1] + abs(y_offset) * height_scale
        
        # 合并变形点
        src_points = np.append(src_points, landmarks, axis=0)
        dst_points = np.append(dst_points, target_landmarks, axis=0)
        
        # 应用变形
        warped_image = self.face_morph.warp_image(img_np, src_points, dst_points)
        
        return (pil2tensor(warped_image),)
