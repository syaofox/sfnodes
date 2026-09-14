# SFMaskToTrackData 后端逻辑测试（Node/Python 直接运行：python tests/test_mask_to_track_data.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（masks required MASK）、根 __init__.py 注册键一致
#   - mask_to_track_data 纯函数（numpy 版 pack 桩，真实位打包）：
#     3D [T,H,W]→[T,1,H,W//8]、2D [H,W]→单帧、W 补到 8 的倍数、
#     多对象维报错、空帧→packed None、None 输入报错
#   - execute 集成（mock torch + comfy.ldm.sam3.tracker）
import ast
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


# ── numpy 版 pack/unpack（严格复刻 comfy.ldm.sam3.tracker 位序）──
def numpy_pack(masks):
    a = (np.asarray(masks) > 0).astype(np.uint8)
    *lead, h, w = a.shape
    a = a.reshape(*lead, h, w // 8, 8)
    shifts = np.arange(8, dtype=np.uint8)
    return (a * (1 << shifts)).sum(-1).astype(np.uint8)


def numpy_unpack(packed):
    p = np.asarray(packed)
    bits = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    return ((p[..., None] & bits) != 0).reshape(*p.shape[:-1], p.shape[-1] * 8)


# ── FakeArr：numpy 代理（支持 dim()/unsqueeze/shape）──
class FakeArr:
    device = "cpu"

    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    def dim(self):
        return self.data.ndim

    def unsqueeze(self, dim):
        return FakeArr(np.expand_dims(self.data, dim))

    def numpy(self):
        return self.data


def _stub_pack(masks):
    m = masks.data if isinstance(masks, FakeArr) else masks
    return FakeArr(numpy_pack(m))


class _FakeFunctional:
    @staticmethod
    def pad(t, pad):
        left, right = pad
        a = t.data if isinstance(t, FakeArr) else np.asarray(t)
        pads = [(0, 0)] * (a.ndim - 1) + [(left, right)]
        return FakeArr(np.pad(a, pads, mode="constant", constant_values=0))


fake_torch = types.ModuleType("torch")
fake_torch.bool = np.bool_
fake_nn = types.ModuleType("torch.nn")
fake_nn.functional = _FakeFunctional()
fake_torch.nn = fake_nn
sys.modules["torch"] = fake_torch

for name in ("comfy", "comfy.ldm", "comfy.ldm.sam3"):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
tracker = types.ModuleType("comfy.ldm.sam3.tracker")
tracker.pack_masks = _stub_pack
sys.modules["comfy.ldm.sam3.tracker"] = tracker


def load(modpath, modname):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = load(os.path.join(root, "nodes", "image", "mask_to_track_data.py"), "sfnodes.nodes.image.mask_to_track_data")
SFMaskToTrackData = mod.SFMaskToTrackData
mask_to_track_data = mod.mask_to_track_data


def convert(masks):
    return mask_to_track_data(masks, pack_masks=_stub_pack, torch=fake_torch)


def unpack_out(td):
    return numpy_unpack(td["packed_masks"].numpy())


# ── 1. 结构 ──
check("CATEGORY", SFMaskToTrackData.CATEGORY == "sfnodes/image")
check("FUNCTION", SFMaskToTrackData.FUNCTION == "execute")
check("RETURN_TYPES", SFMaskToTrackData.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("RETURN_NAMES", SFMaskToTrackData.RETURN_NAMES == ("track_data",))
check("DESCRIPTION 存在", isinstance(getattr(SFMaskToTrackData, "DESCRIPTION", None), str)
      and SFMaskToTrackData.DESCRIPTION.strip() != "")

schema = SFMaskToTrackData.INPUT_TYPES()
check("masks required MASK", schema["required"]["masks"][0] == "MASK")

with open(os.path.join(root, "__init__.py"), encoding="utf-8") as f:
    init_src = f.read()
tree = ast.parse(init_src)
mapping_keys = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
                if isinstance(node.value, ast.Dict):
                    mapping_keys[t.id] = {ast.literal_eval(k) for k in node.value.keys if k is not None}
check("__init__ 注册 SFMaskToTrackData 双字典一致",
      "SFMaskToTrackData" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFMaskToTrackData" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. mask_to_track_data 纯函数 ──
# 3D [T,H,W]：帧0 左半，帧1 右侧
m0 = np.zeros((2, 4, 8), dtype=np.float32)
m0[0, :, :4] = 1.0
m0[1, :, 4:] = 1.0
out = convert(FakeArr(m0))
check("3D packed shape", out["packed_masks"].shape == (2, 1, 4, 1))
check("orig_size", out["orig_size"] == (4, 8))
check("n_frames", out["n_frames"] == 2)
check("scores", out["scores"] == [1.0])
check("3D 内容往返", np.array_equal(unpack_out(out)[:, 0], m0 > 0))

# 2D [H,W] → 单帧
m1 = np.zeros((3, 8), dtype=np.float32)
m1[1, :] = 1.0
out1 = convert(FakeArr(m1))
check("2D packed shape", out1["packed_masks"].shape == (1, 1, 3, 1))
check("2D n_frames", out1["n_frames"] == 1)
check("2D 内容往返", np.array_equal(unpack_out(out1)[0, 0], m1 > 0))

# W 非 8 倍数 → 补零
m2 = np.ones((1, 3, 5), dtype=np.float32)
out2 = convert(FakeArr(m2))
check("补零 packed shape", out2["packed_masks"].shape == (1, 1, 3, 1))
u2 = unpack_out(out2)[0, 0]
check("补零 orig_size", out2["orig_size"] == (3, 5))
check("补零内容前5列", np.array_equal(u2[:, :5], m2[0] > 0))
check("补零内容后3列全False", not u2[:, 5:].any())

# 多对象维报错
try:
    convert(FakeArr(np.zeros((2, 2, 4, 8), dtype=np.float32)))
    check("多对象报错", False)
except ValueError:
    check("多对象报错", True)

# 空帧 → packed None
out3 = convert(FakeArr(np.zeros((0, 4, 8), dtype=np.float32)))
check("空帧→packed None", out3["packed_masks"] is None and out3["scores"] == [])

# None 报错
try:
    convert(None)
    check("None 报错", False)
except ValueError:
    check("None 报错", True)

# ── 3. execute 集成 ──
node = SFMaskToTrackData()
res = node.execute(FakeArr(m0))
check("execute 返回单元素 tuple", isinstance(res, tuple) and len(res) == 1)
check("execute 输出可解包", np.array_equal(unpack_out(res[0])[:, 0], m0 > 0))

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
