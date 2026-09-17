# brush_mask_tools 后端测试（python3 tests/test_brush_mask_tools.py）
# 覆盖（全部 mock，不碰真实模型/torch）：
#   - person_mask 纯核心：normalize_parts / build_mask / segment_mask（fake mediapipe）
#   - 人物部位路由：busy 409 / 缺源 400 / 成功回笔触 + parts 归一 + refine 二次分割
#   - YOLO：模型清单扫描（多目录）/ 白名单防穿越 / task 自检 / 类别过滤 / imgsz 白名单 /
#     输入 dtype 归一（float 0..1 → uint8 0..255 BGR；ultralytics 按 uint8 处理）/
#     _boxes_to_mask（rect+ellipse）/
#     _polygons_to_mask / run_yolo（bbox 与 segm，conf 钳制，labels）/
#     路由 busy 409（不加载模型）/ 无效模型 400 / 成功回笔触 + detected
#   - 导入遮罩：缩放到源图尺寸 + 追踪回笔触
#   - unload_all：清三层缓存
# mock：torch/aiohttp/folder_paths（含 models_dir）/cv2（行为桩）/mediapipe/ultralytics
import importlib.util
import os
import sys
import tempfile
import types

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── mock torch（numpy 代理）──
torch = types.ModuleType("torch")
torch.float32 = np.float32
torch.Tensor = type("Tensor", (), {})
torch.zeros = lambda shape, **k: np.zeros(shape, dtype=np.float32)
torch.ones = lambda shape, **k: np.ones(shape, dtype=np.float32)
torch.from_numpy = lambda a: np.asarray(a, dtype=np.float32)
torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: setattr(torch.cuda, "_emptied", True))
torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
torch.nn.functional.interpolate = lambda *a, **k: None
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional

# ── mock aiohttp ──
aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.ModuleType("aiohttp.web")
aiohttp.web.json_response = lambda *a, **k: types.SimpleNamespace(status=200, body=a)
aiohttp.web.Response = types.SimpleNamespace
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = aiohttp.web

# ── mock folder_paths（含 models_dir：YOLO 权重目录）──
tmp_in = tempfile.mkdtemp(prefix="sf_bmt_input_")
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: tmp_in
fp.get_temp_directory = lambda: os.path.join(tmp_in, "temp")
fp.models_dir = os.path.join(tmp_in, "models")
sys.modules["folder_paths"] = fp

# ── mock cv2（行为桩：findContours 返回注入方框）──
cv2 = types.ModuleType("cv2")
cv2.RETR_EXTERNAL = 0
cv2.CHAIN_APPROX_SIMPLE = 1
cv2.boxes = None


def _shoelace(pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0)


def _fake_find_contours(bw, mode, method):
    if not np.asarray(bw).any():
        return [], None
    return list(cv2.boxes or []), None


cv2.findContours = _fake_find_contours
cv2.contourArea = lambda cnt: _shoelace(np.asarray(cnt).reshape(-1, 2))
cv2.approxPolyDP = lambda cnt, eps, closed: np.asarray(cnt).reshape(-1, 1, 2)
sys.modules["cv2"] = cv2

# ── mock mediapipe（fake segmenter：按 mp_state.masks 生成 confidence_masks）──
mp = types.ModuleType("mediapipe")
mp_state = {"masks": {}}


class _MPImage:
    def __init__(self, image_format=None, data=None):
        self.data = data


class _MPImageFormat:
    SRGB = "srgb"
    SRGBA = "srgra"


class _MPBaseOptions:
    def __init__(self, **kw):
        pass


class _MPOptions:
    def __init__(self, **kw):
        pass


class _MPRunningMode:
    IMAGE = "image"


class _FakeCM:
    def __init__(self, arr):
        self._a = arr

    def numpy_view(self):
        return self._a


class _FakeResult:
    def __init__(self, shape):
        names = ("background", "hair", "body", "face", "clothes")
        self.confidence_masks = [
            _FakeCM(np.full(shape, float(mp_state["masks"].get(n, 0.0)), dtype=np.float32))
            for n in names
        ]


class _FakeSegmenter:
    calls = 0

    @classmethod
    def create_from_options(cls, options):
        class _Ctx:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def segment(self_, mp_image):
                _FakeSegmenter.calls += 1
                return _FakeResult(np.asarray(mp_image.data).shape[:2])

        return _Ctx()


mp.Image = _MPImage
mp.ImageFormat = _MPImageFormat
mp.tasks = types.SimpleNamespace(
    BaseOptions=_MPBaseOptions,
    vision=types.SimpleNamespace(
        ImageSegmenter=_FakeSegmenter,
        ImageSegmenterOptions=_MPOptions,
        RunningMode=_MPRunningMode,  # 真实 API 名（曾误用 VisionRunningMode → 真机报错）
    ),
)
sys.modules["mediapipe"] = mp

# ── mock ultralytics（fake YOLO：按 yolo_state.result 返回）──
yolo_state = {"result": None}


class _FakeTensor:
    def __init__(self, v):
        self._v = np.asarray(v)

    def tolist(self):
        return self._v.tolist()

    def item(self):
        return self._v.item()


class _FakeBoxes:
    def __init__(self, xyxy, cls=None):
        self.xyxy = [np.asarray(b) for b in xyxy]
        self.cls = None if cls is None else [np.asarray(c) for c in cls]

    def __len__(self):
        return len(self.xyxy)


class _FakeMasks:
    def __init__(self, xy):
        self.xy = [np.asarray(p) for p in xy]


class _FakeYOLOResult:
    def __init__(self, boxes=None, masks=None, names=None):
        self.boxes = boxes
        self.masks = masks
        self.names = names or {}


class _FakeYOLO:
    last = {}
    predict_calls = 0
    task = "detect"
    names = {}

    def __init__(self, path):
        self.path = path
        self.task = _FakeYOLO.task
        self.names = dict(_FakeYOLO.names)

    def predict(self, source=None, conf=0.25, verbose=False, imgsz=640, classes=None):
        _FakeYOLO.predict_calls += 1
        _FakeYOLO.last = {"source": np.asarray(source), "conf": conf,
                          "imgsz": imgsz, "classes": classes}
        return [yolo_state["result"]]


ultra = types.ModuleType("ultralytics")
ultra.YOLO = _FakeYOLO
sys.modules["ultralytics"] = ultra

# ── 注册 sfnodes 包结构 ──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.image"); pkg3.__path__ = [os.path.join(root, "nodes", "image")]; sys.modules["sfnodes.nodes.image"] = pkg3
pkg_u = types.ModuleType("sfnodes.sf_utils"); pkg_u.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg_u


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("sfnodes.nodes.image.crop", os.path.join(root, "nodes", "image", "crop.py"))
_load("sfnodes.nodes.image.brush_mask_sam", os.path.join(root, "nodes", "image", "brush_mask_sam.py"))
tools = _load("sfnodes.nodes.image.brush_mask_tools", os.path.join(root, "nodes", "image", "brush_mask_tools.py"))
pm = __import__("sfnodes.sf_utils.person_mask", fromlist=["person_mask"])

# ── person_mask 纯核心 ──
check("normalize_parts 过滤非法", pm.normalize_parts(["face", "bogus", "body"]) == ["face", "body"])
check("normalize_parts 空回退 face", pm.normalize_parts([]) == ["face"] and pm.normalize_parts(None) == ["face"])
check("normalize_parts 逗号串", pm.normalize_parts("hair, body") == ["hair", "body"])

# 顺序：background/hair/body/face/clothes → hair=1.0、face=0.2（阈值 0.5 只留 hair）
fake_result = types.SimpleNamespace(confidence_masks=[_FakeCM(np.full((4, 5), 0.0)), _FakeCM(np.full((4, 5), 1.0)),
                                                     _FakeCM(np.full((4, 5), 0.0)), _FakeCM(np.full((4, 5), 0.2)),
                                                     _FakeCM(np.full((4, 5), 0.0))])
m = pm.build_mask(fake_result, (4, 5), ["hair", "face"], 0.5)
check("build_mask 部位并集阈值", m[0, 0] == 255 and np.all(m == 255))

fake_result2 = types.SimpleNamespace(confidence_masks=[_FakeCM(np.zeros((4, 5), np.float32)) for _ in range(5)])
m2 = pm.build_mask(fake_result2, (4, 5), ["hair"], 0.5)
check("build_mask 全低置信度全黑", np.all(m2 == 0))


class _FakeManager:
    def __init__(self, path):
        self.path = path

    def get_model_path(self, name, sub_dir=None):
        assert sub_dir == "person_mask"
        with open(self.path, "wb") as f:
            f.write(b"TFLITE")
        return self.path


_buf_path = os.path.join(tmp_in, "fake.tflite")
check("load_model_buffer 注入 manager", pm.load_model_buffer(_FakeManager(_buf_path)) == b"TFLITE")

from PIL import Image as _PILImage

mp_state["masks"] = {"face": 1.0, "body": 0.6}
seg = pm.segment_mask(_PILImage.new("RGB", (5, 4)), ["face"], 0.5, False, b"X")
check("segment_mask face 并集", seg.shape == (4, 5) and seg[0, 0] == 1.0)
seg2 = pm.segment_mask(_PILImage.new("RGB", (5, 4)), ["body"], 0.7, False, b"X")
check("segment_mask 阈值过滤", seg2[0, 0] == 0.0)
_FakeSegmenter.calls = 0
pm.segment_mask(_PILImage.new("RGB", (5, 4)), ["face"], 0.5, True, b"X")
check("refine 二次分割", _FakeSegmenter.calls >= 2)

# ── YOLO 清单 / 白名单 ──
for kind, names in (("bbox", ["b.pt", "a.pt", "note.txt"]), ("segm", ["s.pt"])):
    d = os.path.join(fp.models_dir, "ultralytics", kind)
    os.makedirs(d, exist_ok=True)
    for n in names:
        open(os.path.join(d, n), "wb").write(b"\x00")
lst = tools.list_yolo_models()
check("list_yolo_models 排序过滤", lst["bbox"] == ["a.pt", "b.pt"] and lst["segm"] == ["s.pt"])
check("resolve 合法", tools.resolve_yolo_model("bbox", "a.pt") is not None)
check("resolve 防穿越", tools.resolve_yolo_model("bbox", "../a.pt") is None
      and tools.resolve_yolo_model("bbox", "a.pt/../../x.pt") is None)
check("resolve 非清单", tools.resolve_yolo_model("bbox", "zz.pt") is None
      and tools.resolve_yolo_model("nope", "a.pt") is None
      and tools.resolve_yolo_model("bbox", "a.txt") is None)

# ── 掩码换算（纯 numpy）──
mb = tools._boxes_to_mask([[1, 1, 3, 3], [6, 0, 9, 4]], 10, 5, "rect")
check("_boxes_to_mask rect", mb[1, 1] == 1.0 and mb[1, 6] == 1.0 and mb[0, 0] == 0.0)
me = tools._boxes_to_mask([[0, 0, 10, 10]], 10, 10, "ellipse")
check("_boxes_to_mask ellipse 中心实边缘空", me[5, 5] == 1.0 and me[0, 0] == 0.0 and me[0, 5] == 1.0)
mpoly = tools._polygons_to_mask([[[1, 1], [8, 1], [8, 4], [1, 4]]], 10, 5)
check("_polygons_to_mask 填充", mpoly[2, 4] == 1.0 and mpoly[0, 0] == 0.0)

# ── run_yolo bbox / segm ──
np_rgb = np.zeros((5, 10, 3), dtype=np.float32)
yolo_state["result"] = _FakeYOLOResult(boxes=_FakeBoxes([[1, 1, 4, 4], [5, 0, 9, 3]], cls=[0, 1]),
                                       names={0: "nipples", 1: "body"})
mask, detected, labels = tools.run_yolo(np_rgb, "bbox", tools.resolve_yolo_model("bbox", "a.pt"), 0.4, "rect")
check("run_yolo bbox 掩码+计数", detected == 2 and mask[1, 1] == 1.0 and mask[2, 5] == 1.0 and mask[4, 5] == 0.0)
check("run_yolo labels", labels == ["body", "nipples"])
check("run_yolo conf 透传", abs(_FakeYOLO.last["conf"] - 0.4) < 1e-9)
check("run_yolo source 转 BGR（通道反转）", _FakeYOLO.last["source"].shape == (5, 10, 3))
tools.run_yolo(np_rgb, "bbox", tools.resolve_yolo_model("bbox", "a.pt"), 99, "rect")
check("run_yolo conf 钳制 1.0", _FakeYOLO.last["conf"] == 1.0)

yolo_state["result"] = _FakeYOLOResult(masks=_FakeMasks([[[1, 1], [8, 1], [8, 4], [1, 4]]]))
mask, detected, labels = tools.run_yolo(np_rgb, "segm", tools.resolve_yolo_model("segm", "s.pt"), 0.25, "rect")
check("run_yolo segm 多边形并集", detected == 1 and mask[2, 4] == 1.0 and mask[0, 0] == 0.0)

# ── 源图（磁盘）──
_crop_dir = os.path.join(tmp_in, "sfnodes_crop")
os.makedirs(_crop_dir, exist_ok=True)
from PIL import Image

Image.new("RGB", (10, 5), (200, 100, 50)).save(os.path.join(_crop_dir, "src_tools.png"), "PNG")
SRC = "sfnodes_crop/src_tools.png"

# ── 人物部位路由 ──
tools._person_cache["buffer"] = b"FAKE"
code, payload = tools._handle_person_parts({"src_path": SRC, "parts": ["face", "bogus"]}, busy=True)
check("person busy 409", code == 409 and payload.get("busy") is True)
code, payload = tools._handle_person_parts({"src_path": ""}, busy=False)
check("person 缺源 400", code == 400)
mp_state["masks"] = {"face": 1.0}
cv2.boxes = [np.array([[[1, 1]], [[8, 1]], [[8, 4]], [[1, 4]]])]
code, payload = tools._handle_person_parts({"src_path": SRC, "parts": ["face", "bogus"], "confidence": 0.5}, busy=False)
check("person 成功回笔触+parts 归一", code == 200 and payload["parts"] == ["face"]
      and payload["count"] == 1 and payload["strokes"][0]["mode"] == "fill")
check("person 覆盖率", payload["coverage"] == 1.0 and (payload["width"], payload["height"]) == (10, 5))

# ── 状态汇总 / person 落盘状态（提示文案用；不触发下载）──
check("person_status 未落盘但已驻留", tools.person_status() == {"model_found": False, "loaded": True})
_pdir = os.path.join(fp.models_dir, "sfnodes", "person_mask")
os.makedirs(_pdir, exist_ok=True)
open(os.path.join(_pdir, "selfie_multiclass_256x256.tflite"), "wb").write(b"\x00")
check("person_status 已落盘", tools.person_status()["model_found"] is True)
code, payload = tools._handle_ai_status()
check("ai_status 汇总键", code == 200 and set(("busy", "sam", "person", "yolo", "yolo_models")) <= set(payload))
check("ai_status person/yolo 内容", payload["person"]["model_found"] is True
      and isinstance(payload["yolo"]["loaded"], list) and "bbox" in payload["yolo_models"])

# ── YOLO 路由 ──
code, payload = tools._handle_yolo({"src_path": SRC, "kind": "bbox", "model": "a.pt"}, busy=True)
check("yolo busy 409", code == 409)
calls_before = _FakeYOLO.predict_calls
code, payload = tools._handle_yolo({"src_path": SRC, "kind": "bbox", "model": "../a.pt"}, busy=False)
check("yolo 无效模型 400", code == 400 and _FakeYOLO.predict_calls == calls_before)
cv2.boxes = [np.array([[[0, 0]], [[9, 0]], [[9, 4]], [[0, 4]]])]
yolo_state["result"] = _FakeYOLOResult(boxes=_FakeBoxes([[1, 1, 4, 4]], cls=[0]), names={0: "nipples"})
code, payload = tools._handle_yolo({"src_path": SRC, "kind": "bbox", "model": "a.pt", "conf": 0.3}, busy=False)
check("yolo 成功回笔触+detected", code == 200 and payload["detected"] == 1
      and payload["labels"] == ["nipples"] and payload["count"] == 1)

# ── 扫描目录扩展：models/yolo（RMBG legacy）+ 首命中 ──
_yolo_dir = os.path.join(fp.models_dir, "yolo")
os.makedirs(_yolo_dir, exist_ok=True)
open(os.path.join(_yolo_dir, "person.pt"), "wb").write(b"\x00")
open(os.path.join(_yolo_dir, "a.pt"), "wb").write(b"\x00")   # 与 ultralytics/bbox 同名 → bbox 优先
lst2 = tools.list_yolo_models()
check("多目录合并（yolo → bbox 清单）", "person.pt" in lst2["bbox"] and "a.pt" in lst2["bbox"])
check("多目录去重首命中", tools.resolve_yolo_model("bbox", "a.pt").startswith(os.path.join(fp.models_dir, "ultralytics", "bbox"))
      and tools.resolve_yolo_model("bbox", "person.pt").startswith(_yolo_dir))

# ── task 自检（纯函数 + 路由）──
class _TaskModel:
    def __init__(self, task):
        self.task = task


check("task：bbox+segment → 放行+警告", tools._check_yolo_task(_TaskModel("segment"), "bbox")[0] is True
      and "分割模型" in tools._check_yolo_task(_TaskModel("segment"), "bbox")[1])
check("task：segm+detect → 拦截", tools._check_yolo_task(_TaskModel("detect"), "segm")[0] is False)
check("task：segm+segment → 放行", tools._check_yolo_task(_TaskModel("segment"), "segm") == (True, None))
check("task：未知 task 不拦截", tools._check_yolo_task(_TaskModel(None), "segm") == (True, None))

# 注意：_load_yolo 按路径缓存 → 用例需先清缓存再改 fake 的 task/names
tools._yolo_cache.clear()
_FakeYOLO.task = "detect"
code, payload = tools._handle_yolo({"src_path": SRC, "kind": "segm", "model": "s.pt"}, busy=False)
check("detect 权重选 segm → 400", code == 400 and "不能用于分割" in payload.get("error", "")
      and payload.get("task") == "detect")
tools._yolo_cache.clear()
_FakeYOLO.task = "segment"
yolo_state["result"] = _FakeYOLOResult(boxes=_FakeBoxes([[1, 1, 4, 4]], cls=[0]), names={0: "nipples"})
code, payload = tools._handle_yolo({"src_path": SRC, "kind": "bbox", "model": "a.pt"}, busy=False)
check("segment 权重选 bbox → 放行+warning", code == 200 and "warning" in payload
      and payload.get("task") == "segment")
tools._yolo_cache.clear()
_FakeYOLO.task = "detect"

# ── 类别过滤 / imgsz ──
tools._yolo_cache.clear()
_FakeYOLO.names = {0: "nipples", 1: "pussy", 2: "anus", 3: "watermark"}
yolo_state["result"] = _FakeYOLOResult(boxes=_FakeBoxes([[1, 1, 4, 4]], cls=[1]), names=_FakeYOLO.names)
tools.run_yolo(np_rgb, "bbox", tools.resolve_yolo_model("bbox", "a.pt"), 0.3, "rect",
               classes=["pussy", "不存在的类"], imgsz=960)
check("类别名 → id 过滤（未知名丢弃）", _FakeYOLO.last["classes"] == [1])
check("imgsz 透传", _FakeYOLO.last["imgsz"] == 960)
tools.run_yolo(np_rgb, "bbox", tools.resolve_yolo_model("bbox", "a.pt"), 0.3, "rect", classes=[2, 0], imgsz=1234)
check("类别 id 直传排序", _FakeYOLO.last["classes"] == [0, 2])
check("imgsz 白名单回退 640", _FakeYOLO.last["imgsz"] == 640)
tools.run_yolo(np_rgb, "bbox", tools.resolve_yolo_model("bbox", "a.pt"), 0.3, "rect", classes=[], imgsz=640)
check("空类别不过滤（不传 classes）", _FakeYOLO.last["classes"] is None)
check("_normalize_classes 混合/去重", tools._normalize_classes(["anus", 0, "anus", "x"], _FakeYOLO.names) == [0, 2])

# ── 输入 dtype 归一（真机漏检修复：ultralytics numpy 输入按 uint8 0..255 处理，float 0..1 会被二次 /255）──
arr_rgb = np.zeros((2, 2, 3), dtype=np.float32)
arr_rgb[..., 0] = 1.0   # R
arr_rgb[..., 2] = 0.5   # B
tools.run_yolo(arr_rgb, "bbox", tools.resolve_yolo_model("bbox", "a.pt"), 0.3, "rect", imgsz=640)
src = _FakeYOLO.last["source"]
check("float 输入转 uint8", src.dtype == np.uint8)
check("float 0..1 → 0..255 且 RGB→BGR", src[0, 0, 0] == 128 and src[0, 0, 2] == 255)
arr_u8 = np.zeros((2, 2, 3), dtype=np.uint8)
arr_u8[..., 1] = 200    # G
tools.run_yolo(arr_u8, "bbox", tools.resolve_yolo_model("bbox", "a.pt"), 0.3, "rect", imgsz=640)
check("uint8 原样透传（BGR 位序）", _FakeYOLO.last["source"].dtype == np.uint8
      and _FakeYOLO.last["source"][0, 0, 1] == 200)

# ── 类别清单路由 ──
code, payload = tools._handle_yolo_classes("bbox", "a.pt")
check("yolo_classes 成功", code == 200 and payload["names"] == {"0": "nipples", "1": "pussy", "2": "anus", "3": "watermark"}
      and payload["task"] == "detect")
code, payload = tools._handle_yolo_classes("bbox", "../a.pt")
check("yolo_classes 无效 400", code == 400)

# ── 导入遮罩（缩放到源图尺寸）──
mask_pil = Image.new("L", (20, 10), 0)
for x in range(5, 15):
    for y in range(2, 8):
        mask_pil.putpixel((x, y), 255)
mask_pil.save(os.path.join(_crop_dir, "mask_in.png"), "PNG")
cv2.boxes = [np.array([[[0, 0]], [[9, 0]], [[9, 4]], [[0, 4]]])]  # 面积 ≥ min_area(16)
code, payload = tools._handle_import_mask({"src_path": "sfnodes_crop/mask_in.png", "src_w": 10, "src_h": 5})
check("import 缩放到源图尺寸", code == 200 and (payload["width"], payload["height"]) == (10, 5))
check("import 回笔触", payload["count"] == 1 and payload["strokes"][0]["mode"] == "fill")
code, payload = tools._handle_import_mask({"src_path": "sfnodes_crop/missing.png"})
check("import 缺文件 400", code == 400)

# ── unload_all ──
from sfnodes.nodes.image import brush_mask_sam as sam_mod
sam_mod._cache["x"] = ("m", "c")
tools._yolo_cache["y"] = object()
res = tools.unload_all()
check("unload_all 清三层", res["sam"] is True and res["person"] is True and res["yolo"] >= 1)
check("unload_all 后缓存空", not sam_mod._cache and not tools._yolo_cache
      and tools._person_cache["buffer"] is None)

# ── 结果 ──
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
