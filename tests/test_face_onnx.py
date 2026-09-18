# sf_utils/face_onnx.py + face_analysis.py（InsightFace 兼容层）测试
# （Node/Python 直接运行：python3 tests/test_face_onnx.py）
# 覆盖：
#   - 纯函数：distance2bbox / distance2kps / nms / anchor_centers /
#     estimate_norm（Umeyama 对齐）/ trans_points / bbox_crop
#   - 模型解码（fake ORT session）：SCRFD detect（含空检出）、2d106 关键点
#     后处理、ArcFace 特征
#   - FaceEngine：allowed_modules 过滤 / need_* 懒推理
#   - InsightFace 兼容层：多尺寸回退、面积排序、get_embeds/get_keypoints
# mock：cv2 / onnxruntime / torch / torchvision / folder_paths
import os
import sys
import tempfile
import types

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── mock cv2 ──
def _blob_from_image(img, scale, size, mean, swapRB=True):
    return np.zeros((1, 3, size[1], size[0]), dtype=np.float32)


def _blob_from_images(imgs, scale, size, mean, swapRB=True):
    return np.zeros((len(imgs), 3, size[1], size[0]), dtype=np.float32)


cv2 = types.ModuleType("cv2")
cv2.dnn = types.ModuleType("cv2.dnn")
cv2.dnn.blobFromImage = _blob_from_image
cv2.dnn.blobFromImages = _blob_from_images
cv2.warpAffine = lambda img, m, dsize, borderValue=0.0: np.zeros(
    (dsize[1], dsize[0], 3), dtype=np.uint8
)
cv2.resize = lambda img, dsize: np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)
cv2.invertAffineTransform = lambda m: np.linalg.inv(
    np.vstack([m, np.array([0.0, 0.0, 1.0])])
)[:2, :]
cv2.BORDER_CONSTANT = 0
sys.modules["cv2"] = cv2
sys.modules["cv2.dnn"] = cv2.dnn


# ── mock onnxruntime（fake session 工厂）──
class _NS:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


def _zero_outs(h, w):
    counts = [(h // stride) * (w // stride) * 2 for stride in (8, 16, 32)]
    outs = [np.zeros((count, 1), dtype=np.float32) for count in counts]
    outs += [np.zeros((count, 4), dtype=np.float32) for count in counts]
    outs += [np.zeros((count, 10), dtype=np.float32) for count in counts]
    return outs


class _DetSession:
    input_name = "input.1"

    def get_inputs(self):
        return [_NS("input.1", [1, 3, "?", "?"])]

    def get_outputs(self):
        specs = [("s8", 1), ("s16", 1), ("s32", 1),
                 ("b8", 4), ("b16", 4), ("b32", 4),
                 ("k8", 10), ("k16", 10), ("k32", 10)]
        return [_NS(n, [0, width]) for n, width in specs]

    def run(self, names, feed):
        blob = list(feed.values())[0]
        _, _, h, w = blob.shape
        outs = _zero_outs(h, w)
        outs[0][0, 0] = 0.9
        outs[3][0] = np.array([-2, -3, 4, 5], dtype=np.float32) / 8.0
        outs[6][0] = np.array(
            [10, 10, 40, 10, 25, 30, 15, 50, 35, 50], dtype=np.float32
        ) / 8.0
        return outs


class _EmptyDetSession(_DetSession):
    def run(self, names, feed):
        blob = list(feed.values())[0]
        _, _, h, w = blob.shape
        return _zero_outs(h, w)


class _LmkSession:
    input_name = "data"

    def get_inputs(self):
        return [_NS("data", [None, 3, 192, 192])]

    def get_outputs(self):
        return [_NS("fc1", [1, 212])]

    def run(self, names, feed):
        return [np.zeros((1, 212), dtype=np.float32)]


class _RecSession:
    input_name = "input.1"

    def get_inputs(self):
        return [_NS("input.1", [None, 3, 112, 112])]

    def get_outputs(self):
        return [_NS("683", [1, 512])]

    def run(self, names, feed):
        return [np.ones((1, 512), dtype=np.float32)]


def _session_for(path):
    base = os.path.basename(str(path))
    if base in ("det_10g.onnx", "scrfd_10g_bnkps.onnx"):
        return _DetSession()
    if base == "2d106det.onnx":
        return _LmkSession()
    if base in ("w600k_r50.onnx", "glintr100.onnx"):
        return _RecSession()
    raise AssertionError(f"unexpected model: {path}")


onnxruntime = types.ModuleType("onnxruntime")
onnxruntime.InferenceSession = lambda path, providers=None: _session_for(path)
onnxruntime.set_default_logger_severity = lambda level: None
sys.modules["onnxruntime"] = onnxruntime

# ── mock torch / torchvision / folder_paths（face_analysis 导入需要）──
torch = types.ModuleType("torch")
torch.Tensor = type("Tensor", (), {})
sys.modules["torch"] = torch

torchvision = types.ModuleType("torchvision")
transforms = types.ModuleType("torchvision.transforms")
v2 = types.ModuleType("torchvision.transforms.v2")
transforms.v2 = v2
torchvision.transforms = transforms
sys.modules["torchvision"] = torchvision
sys.modules["torchvision.transforms"] = transforms
sys.modules["torchvision.transforms.v2"] = v2

fp = types.ModuleType("folder_paths")
fp.models_dir = tempfile.mkdtemp(prefix="sf_face_models_")
sys.modules["folder_paths"] = fp

face_onnx = L.load_node("sf_utils/face_onnx.py")
face_analysis = L.load_node("sf_utils/face_analysis.py")
Face = face_onnx.Face

# ── 纯函数 ──
check(
    "distance2bbox",
    np.allclose(
        face_onnx.distance2bbox(
            np.array([[10, 20]], dtype=np.float32),
            np.array([[1, 2, 3, 4]], dtype=np.float32),
        ),
        [[9, 18, 13, 24]],
    ),
)
check(
    "distance2kps",
    np.allclose(
        face_onnx.distance2kps(
            np.array([[10, 20]], dtype=np.float32),
            np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float32),
        ),
        [[11, 22, 13, 24, 15, 26, 17, 28, 19, 30]],
    ),
)

nms = face_onnx.nms
check(
    "nms 抑制重叠",
    nms(
        np.array([[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8]], dtype=np.float32),
        0.4,
    )
    == [0],
)
check(
    "nms 保留不重叠",
    nms(
        np.array([[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.8]], dtype=np.float32),
        0.4,
    )
    == [0, 1],
)

anchors = face_onnx.anchor_centers(2, 3, 8, 2)
check(
    "anchor_centers 2x3 stride8 双 anchor",
    anchors.shape == (12, 2)
    and np.allclose(anchors[0], [0, 0])
    and np.allclose(anchors[1], [0, 0])
    and np.allclose(anchors[2], [8, 0])
    and np.allclose(anchors[4], [16, 0])
    and np.allclose(anchors[6], [0, 8]),
)

m112 = face_onnx.estimate_norm(face_onnx.ARC_FACE_DST.copy(), 112)
check("estimate_norm 同点对 → 单位阵", np.allclose(m112, [[1, 0, 0], [0, 1, 0]], atol=1e-4))
m128 = face_onnx.estimate_norm(face_onnx.ARC_FACE_DST.copy(), 128)
check(
    "estimate_norm 128 → 平移 8",
    np.allclose(m128, [[1, 0, 8], [0, 1, 0]], atol=1e-4),
)
mapped = face_onnx.trans_points(
    face_onnx.ARC_FACE_DST.copy(), face_onnx.estimate_norm(face_onnx.ARC_FACE_DST, 112)
)
check("estimate_norm 映射回模板", np.allclose(mapped, face_onnx.ARC_FACE_DST, atol=1e-3))

_, bm = face_onnx.bbox_crop(
    np.zeros((200, 200, 3), dtype=np.uint8), (10, 20), 192, 0.5
)
check(
    "bbox_crop 矩阵（中心映射到输出中心）",
    np.allclose(bm, [[0.5, 0, 91], [0, 0.5, 86]], atol=1e-6),
)

# ── SCRFD 解码 ──
det = face_onnx.FaceDetectionONNX(session=_DetSession())
det_result, det_kps = det.detect(np.zeros((640, 640, 3), dtype=np.uint8))
check(
    "SCRFD detect bbox/score",
    det_result.shape == (1, 5)
    and np.allclose(det_result[0, :4], [2, 3, 4, 5])
    and abs(float(det_result[0, 4]) - 0.9) < 1e-6,
)
check(
    "SCRFD detect kps",
    det_kps is not None
    and det_kps.shape == (1, 5, 2)
    and np.allclose(det_kps[0, 0], [10, 10])
    and np.allclose(det_kps[0, 4], [35, 50]),
)

det_empty = face_onnx.FaceDetectionONNX(session=_EmptyDetSession())
empty_det, empty_kps = det_empty.detect(np.zeros((640, 640, 3), dtype=np.uint8))
check(
    "SCRFD 空检出",
    empty_det.shape == (0, 5)
    and empty_kps is not None
    and empty_kps.shape == (0, 5, 2),
)

# ── 2d106 关键点后处理 ──
lmk = face_onnx.FaceLandmarkONNX(session=_LmkSession())
face_lmk = Face(
    bbox=np.array([0, 0, 100, 100], dtype=np.float32),
    kps=None,
    embedding=None,
    landmark_2d_106=None,
)
pts = lmk.get(np.zeros((200, 200, 3), dtype=np.uint8), face_lmk)
check(
    "landmark 形状与逆变换（bbox 中心对齐）",
    pts.shape == (106, 2)
    and np.allclose(pts, 50.0, atol=1e-3)
    and face_lmk["landmark_2d_106"] is pts,
)

# ── ArcFace 特征 ──
rec = face_onnx.FaceRecognitionONNX(session=_RecSession())
face_rec = Face(
    bbox=np.zeros(4, dtype=np.float32),
    kps=face_onnx.ARC_FACE_DST.copy(),
    embedding=None,
    landmark_2d_106=None,
)
feat = rec.get(np.zeros((200, 200, 3), dtype=np.uint8), face_rec)
check("ArcFace 特征形状", feat.shape == (512,))
check(
    "Face.normed_embedding 归一化",
    abs(float(np.linalg.norm(face_rec.normed_embedding)) - 1.0) < 1e-6,
)

# ── FaceEngine ──
model_dir = tempfile.mkdtemp(prefix="sf_face_dir_")
for name in ("det_10g.onnx", "2d106det.onnx", "w600k_r50.onnx"):
    open(os.path.join(model_dir, name), "wb").close()

engine_det = face_onnx.FaceEngine(model_dir, allowed_modules=["detection"])
check(
    "engine 仅加载 detection",
    engine_det.detector is not None
    and engine_det.landmark is None
    and engine_det.recognizer is None,
)

engine = face_onnx.FaceEngine(model_dir)
faces = engine.get(
    np.zeros((640, 640, 3), dtype=np.uint8),
    need_landmark=True,
    need_embedding=True,
)
check(
    "engine.get 完整推理",
    len(faces) == 1
    and faces[0]["landmark_2d_106"] is not None
    and faces[0]["embedding"] is not None,
)

# ── InsightFace 兼容层 ──
InsightFace = face_analysis.InsightFace


class StubEngine:
    def __init__(self, table):
        self.table = table
        self.calls = []
        self.recognizer = None
        self.landmark = None

    def get(self, img, det_size=(640, 640), det_thresh=0.5, **kwargs):
        self.calls.append(tuple(det_size))
        return self.table.get(tuple(det_size), [])


def _mk_face(size):
    return Face(
        bbox=np.array([0, 0, size, size], dtype=np.float32),
        kps=face_onnx.ARC_FACE_DST.copy(),
        embedding=None,
        landmark_2d_106=None,
    )


small = _mk_face(2)
large = _mk_face(10)
stub = StubEngine({(640, 640): [], (576, 576): [small, large]})
wrapper = InsightFace(stub)
result = wrapper.get_face(np.zeros((64, 64, 3), dtype=np.uint8))
check(
    "get_face 多尺寸回退并面积降序",
    result is not None
    and stub.calls == [(640, 640), (576, 576)]
    and result[0]["bbox"][2] == 10,
)

stub_none = StubEngine({})
wrapper_none = InsightFace(stub_none)
check("get_face 无检出返回 None", wrapper_none.get_face(np.zeros((8, 8, 3), dtype=np.uint8)) is None)


class StubRecognizer:
    def __init__(self):
        self.count = 0

    def get(self, img, face):
        self.count += 1
        face["embedding"] = np.ones(4, dtype=np.float32)


stub2 = StubEngine({(640, 640): [small]})
stub2.recognizer = StubRecognizer()
wrapper2 = InsightFace(stub2)
emb = wrapper2.get_embeds(np.zeros((8, 8, 3), dtype=np.uint8))
check(
    "get_embeds 触发识别并归一化",
    stub2.recognizer.count == 1
    and emb is not None
    and abs(float(np.linalg.norm(emb)) - 1.0) < 1e-6,
)

kps_face = Face(
    bbox=np.zeros(4, dtype=np.float32),
    kps=np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float32),
    embedding=None,
    landmark_2d_106=None,
)
stub3 = StubEngine({(640, 640): [kps_face]})
wrapper3 = InsightFace(stub3)
kp = wrapper3.get_keypoints(np.zeros((8, 8, 3), dtype=np.uint8))
check(
    "get_keypoints 左右眼顺序（left=shape[1]）",
    np.allclose(kp[0], [2, 2]) and np.allclose(kp[1], [1, 1]),
)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
