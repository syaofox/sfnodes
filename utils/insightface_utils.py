import torch
import torchvision.transforms.v2 as T
import os
import folder_paths
import numpy as np

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

THRESHOLDS = {  # from DeepFace
    "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "L2_norm": 1.17},
    "Facenet": {"cosine": 0.40, "euclidean": 10, "L2_norm": 0.80},
    "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "L2_norm": 1.04},
    "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "L2_norm": 1.13},
    "Dlib": {"cosine": 0.07, "euclidean": 0.6, "L2_norm": 0.4},
    "SFace": {"cosine": 0.593, "euclidean": 10.734, "L2_norm": 1.055},
    "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "L2_norm": 0.55},
    "DeepFace": {"cosine": 0.23, "euclidean": 64, "L2_norm": 0.64},
    "DeepID": {"cosine": 0.015, "euclidean": 45, "L2_norm": 0.17},
    "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "L2_norm": 1.10},
}


class InsightFace:
    def __init__(self, face_analysis):
        self.face_analysis = face_analysis
        self.thresholds = THRESHOLDS["ArcFace"]

    def get_face(self, image):
        # 使用最大的检测尺寸来检测尽可能多的人脸
        self.face_analysis.det_model.input_size = (640, 640)
        faces = self.face_analysis.get(image)

        # 如果未检测到人脸，尝试使用较小的尺寸
        if len(faces) == 0:
            for size in [(size, size) for size in range(576, 256, -64)]:
                self.face_analysis.det_model.input_size = size
                faces = self.face_analysis.get(image)
                if len(faces) > 0:
                    break

        if len(faces) > 0:
            # 按人脸面积从大到小排序
            return sorted(
                faces,
                key=lambda x: (x["bbox"][2] - x["bbox"][0])
                * (x["bbox"][3] - x["bbox"][1]),
                reverse=True,
            )
        return None

    def get_embeds(self, image, face_index=0):
        face = self.get_face(image)

        if face is None:
            return None

        face_index = min(face_index, len(face) - 1)

        face_embedding = face[face_index].normed_embedding
        return face_embedding


    def get_bbox(
        self, image, padding=0, padding_percent=0
    ) -> tuple[list[torch.Tensor], list[int], list[int], list[int], list[int]]:
        faces = self.get_face(np.array(image))
        img = []
        x = []
        y = []
        w = []
        h = []
        if faces is not None:
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                width = x2 - x1
                height = y2 - y1

                x1 = int(max(0, x1 - int(width * padding_percent) - padding))
                y1 = int(max(0, y1 - int(height * padding_percent) - padding))
                x2 = int(min(image.width, x2 + int(width * padding_percent) + padding))
                y2 = int(
                    min(image.height, y2 + int(height * padding_percent) + padding)
                )
                crop = image.crop((x1, y1, x2, y2))
                img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
                x.append(x1)
                y.append(y1)
                w.append(x2 - x1)
                h.append(y2 - y1)
        return (img, x, y, w, h)

    def get_keypoints(self, image, face_index=0):
        face = self.get_face(image)
        if face is None:
            return None

        face_index = min(face_index, len(face) - 1)

        shape = face[face_index]["kps"]
        right_eye = shape[0]
        left_eye = shape[1]
        nose = shape[2]
        left_mouth = shape[3]
        right_mouth = shape[4]

        return [left_eye, right_eye, nose, left_mouth, right_mouth]


    def get_landmarks(self, image, extended_landmarks=False, face_index=0):
        face = self.get_face(image)

        if face is None:
            return None

        face_index = min(face_index, len(face) - 1)



        shape = face[face_index]["landmark_2d_106"]
        landmarks = np.round(shape).astype(np.int64)

        main_features = landmarks[33:]
        left_eye = landmarks[87:97]
        right_eye = landmarks[33:43]
        eyes = landmarks[[*range(33, 43), *range(87, 97)]]
        nose = landmarks[72:87]
        mouth = landmarks[52:72]
        left_brow = landmarks[97:106]
        right_brow = landmarks[43:52]
        outline = landmarks[[*range(33), *range(48, 51), *range(102, 105)]]
        outline_forehead = outline

        return [
            landmarks,
            main_features,
            eyes,
            left_eye,
            right_eye,
            nose,
            mouth,
            left_brow,
            right_brow,
            outline,
            outline_forehead,
        ]


    def get_single_bbox(
        self, image, padding=0, padding_percent=0,face_index=0
    ) -> tuple[torch.Tensor, int, int, int, int]:
        """
        获取指定索引的人脸bbox
        
        Args:
            image: 输入图像
            face_index: 人脸索引，默认为0（最大的人脸）
            padding: 边框填充像素
            padding_percent: 边框填充百分比
            
        Returns:
            tuple: (裁剪图像, x坐标, y坐标, 宽度, 高度)
        """
        faces = self.get_face(np.array(image))
        if faces is None:
            return (None, 0, 0, 0, 0)

        face_index = min(face_index, len(faces) - 1)

        face = faces[face_index]
        x1, y1, x2, y2 = face["bbox"]
        width = x2 - x1
        height = y2 - y1

        x1 = int(max(0, x1 - int(width * padding_percent) - padding))
        y1 = int(max(0, y1 - int(height * padding_percent) - padding))
        x2 = int(min(image.width, x2 + int(width * padding_percent) + padding))
        y2 = int(
            min(image.height, y2 + int(height * padding_percent) + padding)
        )
        crop = image.crop((x1, y1, x2, y2))
        img_tensor = T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0)

        return (img_tensor, x1, y1, x2 - x1, y2 - y1)

