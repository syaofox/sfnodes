# SFImageResizePlus 后端测试（Python 直接运行：python tests/test_image_resize_plus.py）
# 覆盖：
#   - resize_engine.total_pixels_to_wh 纯函数（1024² MP 约定、宽高比保持、
#     0/负/垃圾输入回退 None、round 取整、1px 下限）
#   - 节点模块（mock torch/torchvision/comfy.utils/cv2/scipy/kornia）：
#     INPUT_TYPES 新增 size_mode/total_pixels 默认值
#   - execute：size_mode="total pixels" 忽略 width/height 按源宽高比缩放，
#     divisible_by 仍生效；size_mode="width & height" 行为不变
#   - ImageScalerByPixels 收敛到 total_pixels_to_wh 后结果不变
import importlib.util
import os
import sys
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

# ── 1. 纯函数（无 mock，直接从源文件加载避免重复 mock）──
def load_resize_engine():
    spec = importlib.util.spec_from_file_location(
        "sf_utils_resize_engine_solo", os.path.join(root, "sf_utils", "resize_engine.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m

re_mod = load_resize_engine()
tp2wh = re_mod.total_pixels_to_wh

MP = 1024 * 1024

r = tp2wh(1024, 1024, 1.0)
check("纯函数 1:1 1MP 直通", r == (1024, 1024))

r = tp2wh(2000, 1000, 1.0)
check("纯函数 2:1 宽高比保持", r == (1448, 724))

r = tp2wh(512, 512, 1.0)
check("纯函数 512² 1MP 放大 2x -> 1024²", r == (1024, 1024))

r = tp2wh(2000, 1000, 2.0)
expect_w, expect_h = round(2000 * (2 * MP / 2_000_000) ** 0.5), round(1000 * (2 * MP / 2_000_000) ** 0.5)
check("纯函数 2MP 面积翻倍", r == (expect_w, expect_h) and r[0] * r[1] > MP)

check("纯函数 0 MP -> None", tp2wh(1024, 1024, 0) is None)
check("纯函数 负 MP -> None", tp2wh(1024, 1024, -1.0) is None)
check("纯函数 0 宽 -> None", tp2wh(0, 100, 1.0) is None)
check("纯函数 垃圾输入 -> None", tp2wh("x", None, "y") is None)
check("纯函数 极小值不小于 1px", tp2wh(1, 1, 0.01)[0] >= 1)

# ── 2. 节点模块（mock 重量依赖）──
class FakeTensor:
    """最小张量替身：numpy 后端，支持 shape/permute/getitem/clamp 透传。"""
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    @property
    def shape(self):
        return self._arr.shape

    def permute(self, *order):
        return FakeTensor(np.transpose(self._arr, order))

    def movedim(self, src, dst):
        nd = self._arr.ndim
        order = list(range(nd))
        dim = order.pop(src % nd)
        order.insert(dst % nd, dim)
        return FakeTensor(np.transpose(self._arr, order))

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def to(self, dtype):
        return self

    def __getitem__(self, idx):
        return FakeTensor(self._arr[idx])


def np_from_bhwc(arr):
    return FakeTensor(np.asarray(arr, dtype=np.float32))


torch = types.ModuleType("torch")
torch.float32 = np.float32
torch.zeros = lambda *shape, **k: FakeTensor(np.zeros(shape, dtype=np.float32))
torch.full = lambda shape, value, **k: FakeTensor(np.full(shape, value, dtype=np.float32))
torch.clamp = lambda t, lo, hi: t

torch_nn = types.ModuleType("torch.nn")
F = types.ModuleType("torch.nn.functional")

def fake_interpolate(x, size=None, mode=None, **k):
    b, c = x._arr.shape[0], x._arr.shape[1]
    return FakeTensor(np.zeros((b, c, size[0], size[1]), dtype=np.float32))

F.interpolate = fake_interpolate
F.pad = lambda x, pad, **k: x
sys.modules.update({
    "torch": torch,
    "torch.nn": torch_nn,
    "torch.nn.functional": F,
})

tv = types.ModuleType("torchvision")
tv.__path__ = []
tvt = types.ModuleType("torchvision.transforms")
tvt.__path__ = []
tvt.Compose = lambda *a, **k: None
tvtv2 = types.ModuleType("torchvision.transforms.v2")
tvtv2.Compose = lambda *a, **k: None
sys.modules.update({
    "torchvision": tv, "torchvision.transforms": tvt, "torchvision.transforms.v2": tvtv2,
})

for missing in ["cv2", "kornia"]:
    sys.modules[missing] = types.ModuleType(missing)
scipy = types.ModuleType("scipy")
scipy.__path__ = []
scipy_nd = types.ModuleType("scipy.ndimage")
scipy_nd.binary_closing = lambda *a, **k: None
scipy_nd.binary_fill_holes = lambda *a, **k: None
sys.modules.update({"scipy": scipy, "scipy.ndimage": scipy_nd})

comfy = types.ModuleType("comfy")
comfy.__path__ = []
cu = types.ModuleType("comfy.utils")
def fake_common_upscale(samples, width, height, upscale_method, crop):
    b, c = samples._arr.shape[0], samples._arr.shape[1]
    return FakeTensor(np.zeros((b, c, height, width), dtype=np.float32))

cu.common_upscale = fake_common_upscale
cu.lanczos = lambda t, w, h: fake_interpolate(t, size=(h, w))
comfy.utils = cu
sys.modules.update({"comfy": comfy, "comfy.utils": cu})

nodes_mod = types.ModuleType("nodes")
nodes_mod.MAX_RESOLUTION = 16384
nodes_mod.LoadImage = type("LoadImage", (), {})
sys.modules["nodes"] = nodes_mod
sys.modules["folder_paths"] = types.ModuleType("folder_paths")

for name, path in [
    ("sfnodes", root),
    ("sfnodes.nodes", os.path.join(root, "nodes")),
    ("sfnodes.nodes.image", os.path.join(root, "nodes", "image")),
    ("sfnodes.sf_utils", os.path.join(root, "sf_utils")),
]:
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    sys.modules[name] = pkg

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.scale", os.path.join(root, "nodes", "image", "scale.py")
)
scale = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scale
spec.loader.exec_module(scale)

# ── 2.1 INPUT_TYPES ──
req = scale.ImageResizePlus.INPUT_TYPES()["required"]
check("INPUT_TYPES size_mode 存在", "size_mode" in req)
check("INPUT_TYPES size_mode 默认 width & height", req["size_mode"][1]["default"] == "width & height")
check("INPUT_TYPES size_mode 两选项", req["size_mode"][0] == ["width & height", "total pixels"])
check("INPUT_TYPES total_pixels 存在", "total_pixels" in req)
check("INPUT_TYPES total_pixels 默认 1.00", req["total_pixels"][1]["default"] == 1.00)
check("INPUT_TYPES total_pixels 范围 0.01-16", req["total_pixels"][1]["min"] == 0.01 and req["total_pixels"][1]["max"] == 16.0)
check("INPUT_TYPES width/height 默认 1024", req["width"][1]["default"] == 1024 and req["height"][1]["default"] == 1024)
order = list(req.keys())
check("INPUT_TYPES 顺序: size_mode 置顶（image 之后）", order.index("size_mode") == 1)
check("INPUT_TYPES 顺序: width/height 紧随 size_mode",
      order.index("width") == 2 and order.index("height") == 3)
check("INPUT_TYPES 顺序: total_pixels 紧随 height", order.index("total_pixels") == 4)

# ── 2.2 execute：total pixels 模式 ──
img = np_from_bhwc(np.zeros((1, 1000, 2000, 3)))  # [B,H,W,C] 2000x1000

out, _, w, h = scale.ImageResizePlus.execute(
    scale.ImageResizePlus(), img, 0, 0,
    method="stretch", condition="always", divisible_by=1,
    size_mode="total pixels", total_pixels=1.0,
)
check("total pixels 1MP -> 1448x724（忽略 width/height=0）", (w, h) == (1448, 724))

out, _, w, h = scale.ImageResizePlus.execute(
    scale.ImageResizePlus(), img, 9999, 9999,
    method="stretch", condition="always", divisible_by=1,
    size_mode="total pixels", total_pixels=1.0,
)
check("total pixels 忽略显式 width/height", (w, h) == (1448, 724))

out, _, w, h = scale.ImageResizePlus.execute(
    scale.ImageResizePlus(), img, 0, 0,
    method="stretch", condition="always", divisible_by=8,
    size_mode="total pixels", total_pixels=1.0,
)
check("total pixels divisible_by=8 -> 1448x720（后置裁切）", (w, h) == (1448, 720))

# total_pixels <= 0 -> 回退 width/height 直通（0 = 自动档）
out, _, w, h = scale.ImageResizePlus.execute(
    scale.ImageResizePlus(), img, 0, 0,
    method="stretch", condition="always", divisible_by=1,
    size_mode="total pixels", total_pixels=0,
)
check("total pixels 0 MP 回退自动档", (w, h) == (2000, 1000))

# ── 2.3 execute：width & height 模式行为不变 ──
out, _, w, h = scale.ImageResizePlus.execute(
    scale.ImageResizePlus(), img, 1024, 512,
    method="stretch", condition="always", divisible_by=1,
    size_mode="width & height", total_pixels=1.0,
)
check("width & height 模式直用显式宽高", (w, h) == (1024, 512))

out, _, w, h = scale.ImageResizePlus.execute(
    scale.ImageResizePlus(), img, 0, 0,
    method="stretch", condition="always", divisible_by=1,
    size_mode="width & height", total_pixels=1.0,
)
check("width & height 0/0 自动档直通", (w, h) == (2000, 1000))

# ── 2.4 ImageScalerByPixels 收敛后结果不变（prepare_result 返回 dict）──
img2 = np_from_bhwc(np.zeros((1, 500, 1000, 3)))  # 1000x500
res = scale.ImageScalerByPixels.execute(
    scale.ImageScalerByPixels(), img2, "lanczos", 1.0,
    limit=False, divisible_by=1, mask=None,
)
w, h = res["result"][2], res["result"][3]
# 1 MP = 1048576; scale = sqrt(1048576/500000) = 1.44815...
check("ImageScalerByPixels 1MP -> 1448x724", (w, h) == (1448, 724))

# ── 3. 注册字典一致性 ──
sys.path.insert(0, root)
with open(os.path.join(root, "__init__.py"), encoding="utf-8") as f:
    init_src = f.read()
check("__init__ 含 SFImageResizePlus", '"SFImageResizePlus"' in init_src)

if failures:
    print(f"\n{len(failures)} FAILED")
    sys.exit(1)
print("\nALL PASS")
