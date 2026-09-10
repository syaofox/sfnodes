# brush_mask_sam 后端测试（python3 tests/test_brush_mask_sam.py）
# 覆盖（全部 mock，不碰真实 SAM/torch；cv2 用行为桩——findContours 返回
# 已知方框、面积用鞋带公式、approxPolyDP 恒等，测的是我方管线而非 cv2 本体，
# 真实几何正确性由 cv2 保证 + 用户真机验证）：
#   - sam_status 三态（core 缺席 / 模型缺失 / 就绪）
#   - _load_stack 常驻缓存（comfy.sd 只调一次）与缺模型报错
#   - run_sam_mask 委托链（空 prompt→"object"、阈值钳制、refine=2/union 参数）
#   - unload_sam 清缓存 + empty_cache
#   - mask_to_fill_strokes（单轮廓/空/面积过滤/数量上限/点序）
#   - rasterize fill 分支（PIL 真实像素断言，无需 cv2）
#   - _handle_sam（无源 400 / 成功回 strokes+coverage / 推理异常 500）
#   - brush_mask.execute：fill 笔触成块 + 旧 sam_mask_path 被忽略（向后兼容）
import importlib.util
import json
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


# ── mock torch（numpy 代理 + FakeTensor 链式调用）──
class _FakeTensor:
    def __init__(self, a):
        self._a = np.asarray(a, dtype=np.float32)

    def detach(self): return self

    def to(self, *a, **k): return self

    def cpu(self): return self

    def numpy(self): return self._a

    def __getitem__(self, idx):
        res = self._a[idx]
        return _FakeTensor(res) if isinstance(res, np.ndarray) else float(res)

    @property
    def shape(self): return self._a.shape


torch = types.ModuleType("torch")
torch.float32 = np.float32
torch.Tensor = type("Tensor", (), {})
torch.zeros = lambda shape, **k: np.zeros(shape, dtype=np.float32)
torch.ones = lambda shape, **k: np.ones(shape, dtype=np.float32)
torch.from_numpy = lambda a: np.asarray(a, dtype=np.float32)
torch.clamp = lambda a, lo, hi: np.clip(np.asarray(a), lo, hi)
torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: setattr(torch.cuda, "_emptied", True))
torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
torch.nn.functional.interpolate = lambda *a, **k: None
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional

# ── mock cv2（行为桩：方框轮廓 + 鞋带面积 + approx 恒等）──
cv2 = types.ModuleType("cv2")
cv2.RETR_EXTERNAL = 0
cv2.CHAIN_APPROX_SIMPLE = 1


def _shoelace(pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0)


cv2.boxes = None  # 测试按需注入轮廓列表（每个为 Nx1x2 数组）


def _fake_find_contours(bw, mode, method):
    if not np.asarray(bw).any():
        return [], None
    return list(cv2.boxes or []), None


cv2.findContours = _fake_find_contours
cv2.contourArea = lambda cnt: _shoelace(np.asarray(cnt).reshape(-1, 2))
cv2.approxPolyDP = lambda cnt, eps, closed: np.asarray(cnt).reshape(-1, 1, 2)
sys.modules["cv2"] = cv2

# ── mock aiohttp ──
aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.ModuleType("aiohttp.web")
aiohttp.web.json_response = lambda *a, **k: types.SimpleNamespace(status=200, body=a)
aiohttp.web.Response = types.SimpleNamespace
aiohttp.web.Request = types.SimpleNamespace
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = aiohttp.web

# ── mock folder_paths ──
tmp_in = tempfile.mkdtemp(prefix="sf_sam_input_")
ckpt_name = "sam3.1_multiplex_fp16.safetensors"
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: tmp_in
fp.get_temp_directory = lambda: os.path.join(tmp_in, "temp")
fp.get_folder_paths = lambda kind: [os.path.join(tmp_in, "embeddings")]
fp.get_full_path = lambda kind, name: os.path.join(tmp_in, kind, name) if name == ckpt_name else None

def _full_or_raise(kind, name):
    p = fp.get_full_path(kind, name)
    if not p:
        raise FileNotFoundError(name)
    return p

fp.get_full_path_or_raise = _full_or_raise
sys.modules["folder_paths"] = fp

# ── mock comfy.sd / comfy.model_management / comfy.utils ──
comfy = types.ModuleType("comfy")
comfy_sd = types.ModuleType("comfy.sd")
comfy_sd.calls = []


def _fake_load_ckpt(ckpt_path, **kwargs):
    comfy_sd.calls.append((ckpt_path, kwargs))
    return ("FAKE_MODEL", "FAKE_CLIP", None)


comfy_sd.load_checkpoint_guess_config = _fake_load_ckpt
comfy.sd = comfy_sd
comfy_mm = types.ModuleType("comfy.model_management")
comfy_mm.get_torch_device = lambda: types.SimpleNamespace(type="cuda")
comfy_mm.get_autocast_device = lambda d: "cuda"
comfy_mm.is_device_mps = lambda d: False
comfy.model_management = comfy_mm
comfy_utils = types.ModuleType("comfy.utils")
comfy_utils.PROGRESS_BAR_HOOK = "ORIG_HOOK"
comfy.utils = comfy_utils
sys.modules["comfy"] = comfy
sys.modules["comfy.sd"] = comfy_sd
sys.modules["comfy.model_management"] = comfy_mm
sys.modules["comfy.utils"] = comfy_utils

# ── mock nodes.CLIPTextEncode ──
nodes_mod = types.ModuleType("nodes")


class _FakeCLIPTextEncode:
    last = {}

    def encode(self, clip, text):
        _FakeCLIPTextEncode.last = {"clip": clip, "text": text}
        return ([["EMB", {}]],)


nodes_mod.CLIPTextEncode = _FakeCLIPTextEncode
sys.modules["nodes"] = nodes_mod

# ── mock comfy_extras.nodes_sam3.SAM3_Detect ──
pkg_ex = types.ModuleType("comfy_extras"); pkg_ex.__path__ = []
sys.modules["comfy_extras"] = pkg_ex
sam3_mod = types.ModuleType("comfy_extras.nodes_sam3")


class _FakeSAM3Detect:
    last = {}
    mask = None  # 测试按需注入 (H, W) float32，None 则用中央块

    @classmethod
    def execute(cls, model, image, conditioning=None, threshold=0.5,
                refine_iterations=2, individual_masks=False):
        _FakeSAM3Detect.last = {
            "model": model, "threshold": threshold,
            "refine_iterations": refine_iterations,
            "individual_masks": individual_masks,
            "cond": conditioning, "image_shape": tuple(np.asarray(image).shape),
            "hook_during_run": sys.modules["comfy.utils"].PROGRESS_BAR_HOOK,
        }
        h, w = np.asarray(image).shape[1:3]
        if _FakeSAM3Detect.mask is not None:
            m = np.asarray(_FakeSAM3Detect.mask, dtype=np.float32)
        else:
            m = np.zeros((h, w), dtype=np.float32)
            m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1.0

        class _Out:
            def __getitem__(self, i):
                # 真实契约：out[0] 为 [B,H,W]，调用方再取 [0] 得单帧
                return (_FakeTensor(m[np.newaxis, ...]), None)[i]

        return _Out()


sam3_mod.SAM3_Detect = _FakeSAM3Detect
sys.modules["comfy_extras.nodes_sam3"] = sam3_mod

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
sam = _load("sfnodes.nodes.image.brush_mask_sam", os.path.join(root, "nodes", "image", "brush_mask_sam.py"))
pure = _load("sfnodes.sf_utils.brush_mask", os.path.join(root, "sf_utils", "brush_mask.py"))
bm = _load("sfnodes.nodes.image.brush_mask", os.path.join(root, "nodes", "image", "brush_mask.py"))

# ── sam_status ──
os.makedirs(os.path.join(tmp_in, "checkpoints"), exist_ok=True)
open(os.path.join(tmp_in, "checkpoints", ckpt_name), "wb").write(b"\x00" * 64)
st = sam.sam_status()
check("status 就绪", st["available"] is True and st["model_found"] is True and st["loaded"] is False)

os.remove(os.path.join(tmp_in, "checkpoints", ckpt_name))
st = sam.sam_status()
check("status 模型缺失", st["available"] is False and "checkpoints" in st["reason"])
open(os.path.join(tmp_in, "checkpoints", ckpt_name), "wb").write(b"\x00" * 64)

saved = sys.modules.pop("comfy_extras.nodes_sam3")
st = sam.sam_status()
check("status core 缺席", st["available"] is False and "SAM3_Detect" in st["reason"])
sys.modules["comfy_extras.nodes_sam3"] = saved

# ── _load_stack 缓存 ──
sam._cache.clear()
comfy_sd.calls.clear()
m1 = sam._load_stack()
m2 = sam._load_stack()
check("stack 返回 MODEL+CLIP", m1[0] == "FAKE_MODEL" and m1[1] == "FAKE_CLIP")
check("stack 常驻缓存（只拆一次）", len(comfy_sd.calls) == 1 and m1 is m2)

# ── run_sam_mask 委托链 ──
_FakeSAM3Detect.mask = None
img = np.zeros((1, 8, 6, 3), dtype=np.float32)
mask = sam.run_sam_mask(img, "", 0.5)
check("空 prompt 回退 object", _FakeCLIPTextEncode.last["text"] == "object")
check("阈值/refine/union 参数", _FakeSAM3Detect.last["threshold"] == 0.5
      and _FakeSAM3Detect.last["refine_iterations"] == 2
      and _FakeSAM3Detect.last["individual_masks"] is False)
check("返回中央块遮罩", mask.shape == (8, 6) and mask[4, 3] == 1.0 and mask[0, 0] == 0.0)
check("推理期 hook 置空", _FakeSAM3Detect.last["hook_during_run"] is None)
check("hook 事后还原", sys.modules["comfy.utils"].PROGRESS_BAR_HOOK == "ORIG_HOOK")

# ── mask_to_fill_strokes（cv2 行为桩）──
box = np.array([[[1, 1]], [[4, 1]], [[4, 3]], [[1, 3]]])  # 3×2 矩形，面积 6
cv2.boxes = [box]
strokes = pure.mask_to_fill_strokes(np.ones((8, 6), dtype=np.float32), min_area=1)
check("单轮廓一笔", len(strokes) == 1 and strokes[0]["mode"] == "fill")
check("顶点顺序透传", strokes[0]["points"] == [[1, 1], [4, 1], [4, 3], [1, 3]])
check("默认面积过滤", pure.mask_to_fill_strokes(np.ones((8, 6), dtype=np.float32)) == [])
check("空掩码无笔", pure.mask_to_fill_strokes(np.zeros((8, 6), dtype=np.float32)) == [])
big = np.array([[[0, 0]], [[5, 0]], [[5, 5]], [[0, 5]]])  # 面积 25
cv2.boxes = [box, big]
strokes = pure.mask_to_fill_strokes(np.ones((8, 6), dtype=np.float32), min_area=1, max_contours=1)
check("数量上限取最大", len(strokes) == 1 and strokes[0]["points"][2] == [5, 5])
strokes = pure.mask_to_fill_strokes(np.ones((8, 6), dtype=np.float32), min_area=1, max_contours=64)
check("面积降序", len(strokes) == 2 and strokes[0]["points"][2] == [5, 5])

# ── rasterize fill 分支（PIL 真实像素）──
m = pure.rasterize_strokes([{"mode": "fill", "size": 0, "points": [[1, 1], [4, 1], [4, 3], [1, 3]]}], 6, 5)
check("fill 块内白", m[2, 2] == 1.0)
check("fill 块外黑", m[0, 0] == 0.0 and m[4, 5] == 0.0)
m = pure.rasterize_strokes([
    {"mode": "fill", "size": 0, "points": [[0, 0], [5, 0], [5, 4], [0, 4]]},
    {"mode": "erase", "size": 3, "points": [(2, 2)]},
], 6, 5)
check("erase 可擦 fill", m[2, 2] == 0.0 and m[0, 0] == 1.0)

# ── parse_state_strokes 允许 fill ──
ps = pure.parse_state_strokes({"src_w": 10, "src_h": 10, "brush_size": 80,
                               "strokes": [{"mode": "fill", "size": 0, "points": [[1, 1], [5, 1], [5, 5], [1, 5]]}]})
check("state 保留 fill", len(ps) == 1 and ps[0]["mode"] == "fill")

# ── unload_sam ──
check("unload 释放过", sam.unload_sam() is True)
check("unload 后缓存空", len(sam._cache) == 0)
check("unload 空缓存返回 False", sam.unload_sam() is False)

# ── _handle_sam 回 strokes（无文件链路）──
code, payload = sam._handle_sam({"src_path": ""})
check("无源 400", code == 400 and "error" in payload)

from PIL import Image
_crop_dir = os.path.join(tmp_in, "sfnodes_crop")
os.makedirs(_crop_dir, exist_ok=True)
Image.new("RGB", (6, 5), (10, 20, 30)).save(os.path.join(_crop_dir, "crop_src_samx.png"), "PNG")
cv2.boxes = [box]
_FakeSAM3Detect.mask = np.ones((5, 6), dtype=np.float32)  # 面积 30 ≥ 默认 min_area
cv2.boxes = [np.array([[[0, 0]], [[5, 0]], [[5, 4]], [[0, 4]]])]  # 桩轮廓面积 20 ≥ 16
code, payload = sam._handle_sam({"src_path": "sfnodes_crop/crop_src_samx.png", "prompt": "person", "threshold": 0.5})
check("成功 200", code == 200)
check("回 fill 笔触", payload.get("count") == 1 and payload["strokes"][0]["mode"] == "fill"
      and payload["strokes"][0]["points"] == [[0, 0], [5, 0], [5, 4], [0, 4]])
check("回尺寸", (payload.get("width"), payload.get("height")) == (6, 5))
check("覆盖率回传", payload.get("coverage") == 1.0)
check("不落盘（无 brush_sam 文件）",
      not [f for f in os.listdir(_crop_dir) if f.startswith("brush_sam_")])

_FakeSAM3Detect.mask = np.zeros((5, 6), dtype=np.float32)
code, payload = sam._handle_sam({"src_path": "sfnodes_crop/crop_src_samx.png"})
check("空结果 strokes 为空", code == 200 and payload.get("strokes") == [] and payload.get("count") == 0)
_FakeSAM3Detect.mask = None

orig_run = sam.run_sam_mask
sam.run_sam_mask = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    code, payload = sam._handle_sam({"src_path": "sfnodes_crop/crop_src_samx.png"})
    check("推理异常 500", code == 500 and "boom" in payload.get("error", ""))
finally:
    sam.run_sam_mask = orig_run

# ── brush_mask.execute：fill 成块 + 旧 sam_mask_path 被忽略 ──
node = bm.SFImageBrushMask()
state = json.dumps({
    "src_path": "sfnodes_crop/crop_src_samx.png",
    "src_w": 6, "src_h": 5, "brush_size": 80,
    "strokes": [{"mode": "fill", "size": 0, "points": [[1, 1], [4, 1], [4, 3], [1, 3]]}],
    "sam_mask_path": "sfnodes_crop/stale.png",  # 旧版残留，必须被忽略
})
arr = np.asarray(node.execute(SFBrushMaskJson=state)[1])[0]
check("fill 块内白", arr[2, 2] == 1.0)
check("fill 块外黑", arr[0, 0] == 0.0 and arr[4, 5] == 0.0)

# ── 结果 ──
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
