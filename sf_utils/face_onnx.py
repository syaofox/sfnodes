# -*- coding: utf-8 -*-
"""人脸 ONNX 推理核心（无 insightface 依赖）。

SCRFD 人脸检测 / 2d106 关键点 / ArcFace 512 维特征：预处理、输出解码与
对齐语义对齐 insightface（MIT License, https://github.com/deepinsight/insightface ），
其中 5 点相似变换改用 numpy（Umeyama）实现以替代其 skimage 依赖，
解码结果可与 insightface 逐值比对（见 experience/nodes-image.md §104）。

仅依赖 numpy / cv2 / onnxruntime（无 torch / folder_paths / onnx / skimage）。
"""

import os

import cv2
import numpy as np
import onnxruntime

# ArcFace 5 点对齐模板（112x112）
ARC_FACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# 模型目录内候选文件名（按优先级探测）
DETECTION_MODEL_FILES = ("det_10g.onnx", "scrfd_10g_bnkps.onnx")
LANDMARK_MODEL_FILES = ("2d106det.onnx",)
RECOGNITION_MODEL_FILES = ("w600k_r50.onnx", "glintr100.onnx")

# 检测模型输入归一化（SCRFD）
DET_INPUT_MEAN = 127.5
DET_INPUT_STD = 128.0
# 2d106det 为 MXNet 转换模型，Sub/Mul 已烘焙进计算图（mean/std=0/1）
LMK_INPUT_MEAN = 0.0
LMK_INPUT_STD = 1.0
# ArcFace 识别模型输入归一化
REC_INPUT_MEAN = 127.5
REC_INPUT_STD = 127.5


def distance2bbox(points, distance):
    """anchor 中心 + 四边距离 → (x1, y1, x2, y2)"""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    """anchor 中心 + 偏移 → 关键点 (N, 2*K)"""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets, thresh):
    """按得分降序的贪心 NMS（与 insightface 同式：面积为 +1 版本）"""
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def anchor_centers(height, width, stride, num_anchors=1):
    """SCRFD 特征图 anchor 中心（与 insightface np.mgrid 版本同序）"""
    centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    centers = (centers * stride).reshape((-1, 2))
    if num_anchors > 1:
        centers = np.stack([centers] * num_anchors, axis=1).reshape((-1, 2))
    return centers


def _umeyama(src, dst):
    """Umeyama 1991 相似变换估计（与 skimage SimilarityTransform.estimate 等价）"""
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    a = dst_demean.T @ src_demean / num
    d = np.ones(dim)
    if np.linalg.det(a) < 0:
        d[dim - 1] = -1
    t = np.eye(dim + 1)
    u, s, vt = np.linalg.svd(a)
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        raise ValueError("无法估计相似变换：源点退化")
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            t[:dim, :dim] = u @ vt
        else:
            saved = d[dim - 1]
            d[dim - 1] = -1
            t[:dim, :dim] = u @ np.diag(d) @ vt
            d[dim - 1] = saved
    else:
        t[:dim, :dim] = u @ np.diag(d) @ vt
    scale = 1.0 / src_demean.var(axis=0).sum() * (s @ d)
    t[:dim, dim] = dst_mean - scale * (t[:dim, :dim] @ src_mean.T)
    t[:dim, :dim] *= scale
    return t[:dim, :]


def estimate_norm(lmk, image_size=112):
    """5 点 → ArcFace 对齐矩阵（替代 insightface 的 skimage 实现）"""
    lmk = np.asarray(lmk, dtype=np.float64)
    if lmk.shape != (5, 2):
        raise ValueError(f"estimate_norm 需要 5x2 关键点，收到 {lmk.shape}")
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = ARC_FACE_DST.astype(np.float64) * ratio
    dst[:, 0] += diff_x
    return _umeyama(lmk, dst).astype(np.float32)


def norm_crop(img, landmark, image_size=112):
    """按 5 点相似变换裁剪对齐（norm_crop 等价实现）"""
    m = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, m, (image_size, image_size), borderValue=0.0)


def bbox_crop(img, center, output_size, scale, rotate=0.0):
    """bbox 中心相似变换裁剪（对齐 insightface face_align.transform，rotate 恒 0）"""
    rot = float(rotate) * np.pi / 180.0
    cos_v, sin_v = np.cos(rot), np.sin(rot)
    m2 = np.array(
        [[scale * cos_v, -scale * sin_v], [scale * sin_v, scale * cos_v]],
        dtype=np.float64,
    )
    t = np.array([output_size / 2.0, output_size / 2.0]) - m2 @ np.asarray(
        center, dtype=np.float64
    )
    m = np.hstack([m2, t.reshape(2, 1)]).astype(np.float32)
    cropped = cv2.warpAffine(img, m, (output_size, output_size), borderValue=0.0)
    return cropped, m


def trans_points(pts, m):
    """仿射矩阵批量变换 2D 点"""
    pts = np.asarray(pts, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    return np.hstack([pts, ones]) @ m.T


class Face(dict):
    """检测结果载体（字典 + 属性双访问，兼容 insightface Face 用法）"""

    def __getattr__(self, name):
        return self.get(name)

    @property
    def embedding_norm(self):
        emb = self.get("embedding")
        if emb is None:
            return None
        return float(np.linalg.norm(emb))

    @property
    def normed_embedding(self):
        emb = self.get("embedding")
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm


class FaceDetectionONNX:
    """SCRFD 人脸检测（det_10g / scrfd_10g_bnkps），输出 (N,5) bbox+score 与 (N,5,2) kps"""

    def __init__(self, model_path=None, providers=None, session=None):
        if session is None:
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(f"人脸检测模型不存在: {model_path}")
            session = onnxruntime.InferenceSession(
                model_path, providers=providers or ["CPUExecutionProvider"]
            )
        self.session = session
        self.center_cache = {}
        self.det_thresh = 0.5
        self.nms_thresh = 0.4
        input_cfg = session.get_inputs()[0]
        self.input_name = input_cfg.name
        fixed = [
            int(v)
            for v in input_cfg.shape[2:4]
            if isinstance(v, (int, np.integer)) and int(v) > 0
        ]
        self.input_size = (fixed[1], fixed[0]) if len(fixed) == 2 else (640, 640)
        outputs = session.get_outputs()
        count = len(outputs)
        self.batched = len(outputs[0].shape) == 3
        if count in (6, 9):
            self.fmc = 3
            self.feat_stride_fpn = (8, 16, 32)
            self.num_anchors = 2
            self.use_kps = count == 9
        elif count in (10, 15):
            self.fmc = 5
            self.feat_stride_fpn = (8, 16, 32, 64, 128)
            self.num_anchors = 1
            self.use_kps = count == 15
        else:
            raise RuntimeError(f"不支持的 SCRFD 模型输出数量: {count}")
        self.output_names = [o.name for o in outputs]

    def _prepare_blob(self, img, input_size):
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / DET_INPUT_STD,
            input_size,
            (DET_INPUT_MEAN, DET_INPUT_MEAN, DET_INPUT_MEAN),
            swapRB=True,
        )
        return np.ascontiguousarray(blob, dtype=np.float32)

    def forward(self, img, threshold):
        """单次前向 + 各 stride 解码为 (scores, bboxes, kpss) 列表"""
        scores_list = []
        bboxes_list = []
        kpss_list = []
        height, width = img.shape[0], img.shape[1]
        blob = self._prepare_blob(img, (width, height))
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height, input_width = blob.shape[2], blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self.feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0] * stride
                kps_preds = net_outs[idx + fmc * 2][0] * stride if self.use_kps else None
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc] * stride
                kps_preds = net_outs[idx + fmc * 2] * stride if self.use_kps else None
            feat_height = input_height // stride
            feat_width = input_width // stride
            key = (feat_height, feat_width, stride)
            centers = self.center_cache.get(key)
            if centers is None:
                centers = anchor_centers(feat_height, feat_width, stride, self.num_anchors)
                if len(self.center_cache) < 100:
                    self.center_cache[key] = centers
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.use_kps:
                kpss = distance2kps(centers, kps_preds).reshape(
                    (kps_preds.shape[0], -1, 2)
                )
                kpss_list.append(kpss[pos_inds])
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, det_size=(640, 640), det_thresh=None):
        """letterbox 缩放检测 → 原图坐标 bbox (N,5) 与 kps (N,5,2)；无检出返回空数组"""
        threshold = self.det_thresh if det_thresh is None else float(det_thresh)
        det_size = (int(det_size[0]), int(det_size[1]))
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(det_size[1]) / det_size[0]
        if im_ratio > model_ratio:
            new_height = det_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = det_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized
        scores_list, bboxes_list, kpss_list = self.forward(det_img, threshold)
        if not scores_list or sum(int(s.size) for s in scores_list) == 0:
            empty_kps = (0, 5, 2) if self.use_kps else None
            return (
                np.empty((0, 5), dtype=np.float32),
                np.empty(empty_kps, dtype=np.float32) if empty_kps else None,
            )
        scores = np.vstack(scores_list).ravel()
        order = np.argsort(-scores, kind="stable")
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale if self.use_kps else None
        pre_det = np.hstack((bboxes, scores.reshape(-1, 1))).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        if kpss is not None:
            kpss = kpss[order, :, :]
        keep = nms(pre_det, self.nms_thresh)
        det = pre_det[keep, :]
        if kpss is not None:
            kpss = kpss[keep, :, :]
        return det, kpss


class FaceLandmarkONNX:
    """2d106det 106 点关键点（bbox 中心对齐，输出原图坐标）"""

    def __init__(self, model_path=None, providers=None, session=None):
        if session is None:
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(f"人脸关键点模型不存在: {model_path}")
            session = onnxruntime.InferenceSession(
                model_path, providers=providers or ["CPUExecutionProvider"]
            )
        self.session = session
        input_cfg = session.get_inputs()[0]
        self.input_name = input_cfg.name
        fixed = [
            int(v)
            for v in input_cfg.shape[2:4]
            if isinstance(v, (int, np.integer)) and int(v) > 0
        ]
        self.input_size = (fixed[1], fixed[0]) if len(fixed) == 2 else (192, 192)
        out_shape = session.get_outputs()[0].shape
        out_dim = (
            int(out_shape[1])
            if len(out_shape) > 1 and isinstance(out_shape[1], (int, np.integer))
            else 212
        )
        if out_dim == 3309:
            raise RuntimeError("不支持 68 点 3D 关键点模型（仅支持 2d106det 的 106 点）")
        self.lmk_num = out_dim // 2
        self.lmk_dim = 2

    def get(self, img, face):
        bbox = face["bbox"]
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        center = ((bbox[2] + bbox[0]) / 2.0, (bbox[3] + bbox[1]) / 2.0)
        scale = self.input_size[0] / (max(w, h) * 1.5)
        aimg, m = bbox_crop(img, center, self.input_size[0], scale)
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / LMK_INPUT_STD,
            (self.input_size[0], self.input_size[1]),
            (LMK_INPUT_MEAN, LMK_INPUT_MEAN, LMK_INPUT_MEAN),
            swapRB=True,
        )
        pred = self.session.run(None, {self.input_name: blob})[0][0]
        pred = np.asarray(pred).reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1 :, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= self.input_size[0] // 2
        invert = cv2.invertAffineTransform(m)
        pred = trans_points(pred, invert)
        face["landmark_2d_106"] = pred
        return pred


class FaceRecognitionONNX:
    """ArcFace 512 维人脸特征（5 点 norm_crop 对齐，返回未归一化 embedding）"""

    def __init__(self, model_path=None, providers=None, session=None):
        if session is None:
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(f"人脸识别模型不存在: {model_path}")
            session = onnxruntime.InferenceSession(
                model_path, providers=providers or ["CPUExecutionProvider"]
            )
        self.session = session
        input_cfg = session.get_inputs()[0]
        self.input_name = input_cfg.name
        fixed = [
            int(v)
            for v in input_cfg.shape[2:4]
            if isinstance(v, (int, np.integer)) and int(v) > 0
        ]
        self.input_size = (fixed[1], fixed[0]) if len(fixed) == 2 else (112, 112)

    def get(self, img, face):
        kps = face["kps"]
        if kps is None:
            raise ValueError("缺少 5 点关键点，无法进行人脸对齐")
        aimg = norm_crop(img, kps, self.input_size[0])
        blob = cv2.dnn.blobFromImages(
            [aimg],
            1.0 / REC_INPUT_STD,
            (self.input_size[0], self.input_size[1]),
            (REC_INPUT_MEAN, REC_INPUT_MEAN, REC_INPUT_MEAN),
            swapRB=True,
        )
        feat = self.session.run(None, {self.input_name: blob})[0]
        feat = np.asarray(feat).reshape(-1)
        face["embedding"] = feat
        return feat


class FaceEngine:
    """人脸模型引擎：按目录探测加载检测/关键点/识别模型（替代 insightface.FaceAnalysis）"""

    def __init__(self, model_dir, providers=None, allowed_modules=None):
        self.model_dir = str(model_dir)
        providers = providers or ["CPUExecutionProvider"]
        if allowed_modules is None:
            allowed = {"detection", "landmark_2d_106", "recognition"}
        else:
            allowed = set(allowed_modules)
        try:
            onnxruntime.set_default_logger_severity(3)
        except Exception:
            pass
        self.detector = None
        self.landmark = None
        self.recognizer = None
        if "detection" in allowed:
            path = self._find(DETECTION_MODEL_FILES)
            if path is not None:
                self.detector = FaceDetectionONNX(path, providers)
        if "landmark_2d_106" in allowed:
            path = self._find(LANDMARK_MODEL_FILES)
            if path is not None:
                self.landmark = FaceLandmarkONNX(path, providers)
        if "recognition" in allowed:
            path = self._find(RECOGNITION_MODEL_FILES)
            if path is not None:
                self.recognizer = FaceRecognitionONNX(path, providers)
        if self.detector is None:
            raise RuntimeError(
                "未找到人脸检测模型（{}），目录: {}".format(
                    " / ".join(DETECTION_MODEL_FILES), self.model_dir
                )
            )

    def _find(self, candidates):
        for name in candidates:
            path = os.path.join(self.model_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def detect(self, img, det_size=(640, 640), det_thresh=0.5):
        return self.detector.detect(img, det_size=det_size, det_thresh=det_thresh)

    def get(
        self,
        img,
        det_size=(640, 640),
        det_thresh=0.5,
        need_landmark=False,
        need_embedding=False,
    ):
        bboxes, kpss = self.detect(img, det_size=det_size, det_thresh=det_thresh)
        faces = []
        for i in range(bboxes.shape[0]):
            face = Face(
                bbox=bboxes[i, 0:4].copy(),
                kps=None if kpss is None else kpss[i].copy(),
                det_score=float(bboxes[i, 4]),
                embedding=None,
                landmark_2d_106=None,
            )
            if need_landmark and self.landmark is not None:
                self.landmark.get(img, face)
            if need_embedding and self.recognizer is not None:
                self.recognizer.get(img, face)
            faces.append(face)
        return faces
