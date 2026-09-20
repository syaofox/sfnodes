# SFTrackDataToMask 后端逻辑测试（Node/Python 直接运行：python tests/test_track_data_to_mask.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（track_data/frame_index required、object_indices optional）、
#     根 __init__.py 注册键一致
#   - parse_object_indices（空=全部、子集、非法/越界忽略、空白容忍）
#   - frame_mask_from_track_data（numpy 版 pack/unpack 桩，真实位打包）：
#     指定帧内容、负索引取尾、多对象并集、对象子集、非法索引→全零、
#     越界报错、packed None→全零、orig_size 缺失报错、尺寸还原参数（bilinear）、
#     不改输入 track_data
#   - execute 集成（mock torch + comfy.ldm.sam3.tracker）
import ast
import importlib
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

    def float(self):
        return FakeArr(self.data.astype(np.float32))

    def numpy(self):
        return self.data

    def __getitem__(self, key):
        return FakeArr(self.data[key])

    def __or__(self, other):
        return FakeArr(self.data | (other.data if isinstance(other, FakeArr) else other))


def _arr(x):
    return x.data if isinstance(x, FakeArr) else np.asarray(x)


def _stub_unpack(packed):
    return FakeArr(numpy_unpack(_arr(packed)))


interp_calls = []


def _stub_interpolate(t, size, mode="bilinear", align_corners=False):
    interp_calls.append({"size": tuple(size), "mode": mode, "align_corners": align_corners})
    a = _arr(t)
    _, _, h, w = a.shape
    th, tw = size
    ys = np.clip((np.arange(th) * h / th).astype(int), 0, h - 1)
    xs = np.clip((np.arange(tw) * w / tw).astype(int), 0, w - 1)
    return FakeArr(a[:, :, ys][:, :, :, xs])


fake_torch = types.ModuleType("torch")
fake_torch.float32 = np.float32
fake_torch.bool = np.bool_
fake_torch.zeros = lambda shape, dtype=None, device=None: FakeArr(
    np.zeros(shape, dtype=np.float32 if dtype is None else dtype))
fake_torch.nn = types.SimpleNamespace(
    functional=types.SimpleNamespace(interpolate=_stub_interpolate))
sys.modules["torch"] = fake_torch

for name in ("comfy", "comfy.ldm", "comfy.ldm.sam3"):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
tracker = types.ModuleType("comfy.ldm.sam3.tracker")
tracker.pack_masks = lambda masks: FakeArr(numpy_pack(_arr(masks)))
tracker.unpack_masks = _stub_unpack
sys.modules["comfy.ldm.sam3.tracker"] = tracker

# 注册 sfnodes 包结构使节点相对导入（sf_utils.track_data_ops）解析
for _pkg, _rel in [("sfnodes", "."), ("sfnodes.nodes", "nodes"),
                   ("sfnodes.nodes.image", "nodes/image"), ("sfnodes.sf_utils", "sf_utils")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [os.path.join(root, _rel)]
    sys.modules[_pkg] = _m


def load(modpath, modname):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


track_ops = importlib.import_module("sfnodes.sf_utils.track_data_ops")
parse_object_indices = track_ops.parse_object_indices
frame_mask_from_track_data = track_ops.frame_mask_from_track_data

mod = load(os.path.join(root, "nodes", "image", "track_data_to_mask.py"),
           "sfnodes.nodes.image.track_data_to_mask")
SFTrackDataToMask = mod.SFTrackDataToMask


def make_td(orig_size=(4, 8), n_frames=3):
    m = np.zeros((3, 2, 4, 8), dtype=bool)
    m[0, 0, :, :4] = True     # 帧0 对象0 左半
    m[1, 0, :, 4:] = True     # 帧1 对象0 右半
    m[2, 0, :2, :] = True     # 帧2 对象0 上半
    m[1, 1, 2:, :] = True     # 帧1 对象1 下半
    return {"packed_masks": FakeArr(numpy_pack(m)), "n_frames": n_frames,
            "scores": [0.9, 0.8], "orig_size": orig_size}, m


def extract(td, frame_index=0, object_indices=""):
    return frame_mask_from_track_data(
        td, frame_index, object_indices,
        unpack_masks=_stub_unpack, torch=fake_torch, interpolate=_stub_interpolate)


# ── 1. 结构 ──
check("CATEGORY", SFTrackDataToMask.CATEGORY == "sfnodes/image")
check("FUNCTION", SFTrackDataToMask.FUNCTION == "execute")
check("RETURN_TYPES", SFTrackDataToMask.RETURN_TYPES == ("MASK",))
check("RETURN_NAMES", SFTrackDataToMask.RETURN_NAMES == ("mask",))
check("DESCRIPTION 存在", isinstance(getattr(SFTrackDataToMask, "DESCRIPTION", None), str)
      and SFTrackDataToMask.DESCRIPTION.strip() != "")

schema = SFTrackDataToMask.INPUT_TYPES()
check("track_data required SAM3_TRACK_DATA", schema["required"]["track_data"][0] == "SAM3_TRACK_DATA")
check("frame_index required INT", schema["required"]["frame_index"][0] == "INT"
      and schema["required"]["frame_index"][1]["min"] < 0)
check("object_indices optional STRING", schema["optional"]["object_indices"][0] == "STRING")

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
check("__init__ 注册 SFTrackDataToMask 双字典一致",
      "SFTrackDataToMask" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFTrackDataToMask" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. parse_object_indices ──
check("索引空=全部", parse_object_indices("", 3) == [0, 1, 2])
check("索引 None=全部", parse_object_indices(None, 2) == [0, 1])
check("索引子集+空白", parse_object_indices(" 1 , 2 ", 4) == [1, 2])
check("索引非法/越界忽略", parse_object_indices("x,1,9,-1", 3) == [1])
check("索引全非法=空", parse_object_indices("x,9", 3) == [])
check("索引 0 对象=空", parse_object_indices("0", 0) == [])

# ── 3. frame_mask_from_track_data ──
td, m = make_td()

out0 = extract(td, 0)
check("帧0 形状 [1,H,W]", out0.shape == (1, 4, 8))
check("帧0 内容=对象0", np.array_equal(out0.numpy()[0] > 0, m[0, 0]))

out1 = extract(td, 1)
check("帧1 全对象并集", np.array_equal(out1.numpy()[0] > 0, m[1, 0] | m[1, 1]))

out_neg = extract(td, -1)
check("负索引取尾帧", np.array_equal(out_neg.numpy()[0] > 0, m[2, 0]))

out_sub = extract(td, 1, "1")
check("对象子集", np.array_equal(out_sub.numpy()[0] > 0, m[1, 1]))

out_bad = extract(td, 1, "5,x")
check("非法索引→全零", out_bad.shape == (1, 4, 8) and not out_bad.numpy().any())

for bad_index in (3, -4):
    try:
        extract(td, bad_index)
        check(f"越界帧 {bad_index} 报错", False)
    except ValueError:
        check(f"越界帧 {bad_index} 报错", True)

empty = {"packed_masks": None, "n_frames": 3, "scores": [], "orig_size": (4, 8)}
out_empty = extract(empty, -1)
check("packed None→全零", out_empty.shape == (1, 4, 8) and not out_empty.numpy().any())

no_orig = {"packed_masks": td["packed_masks"], "n_frames": 3, "scores": [], "orig_size": (0, 0)}
try:
    extract(no_orig, 0)
    check("orig_size 非法报错", False)
except ValueError:
    check("orig_size 非法报错", True)

# 尺寸还原：orig_size 大于工作网格 → 按 orig_size 双线性
interp_calls.clear()
scaled, m2 = make_td(orig_size=(16, 32))
out_scaled = extract(scaled, 0)
check("尺寸还原形状", out_scaled.shape == (1, 16, 32))
check("interpolate 参数", interp_calls[-1] == {"size": (16, 32), "mode": "bilinear", "align_corners": False})
check("不改输入 n_frames/scores",
      td["n_frames"] == 3 and td["scores"] == [0.9, 0.8] and td["orig_size"] == (4, 8))

# ── 4. execute 集成 ──
node = SFTrackDataToMask()
res = node.execute(td, 1)
check("execute 返回单元素 tuple", isinstance(res, tuple) and len(res) == 1)
check("execute 输出帧1 并集", np.array_equal(res[0].numpy()[0] > 0, m[1, 0] | m[1, 1]))

try:
    node.execute("not a track data")
    check("execute 非法输入报错", False)
except ValueError:
    check("execute 非法输入报错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
