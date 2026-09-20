# SFInvertTrackData 后端逻辑测试（Node/Python 直接运行：python tests/test_invert_track.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（track_data required SAM3_TRACK_DATA / object_indices optional）、
#     根 __init__.py 注册键一致
#   - invert_track_data 纯函数（numpy 版 pack/unpack 桩，真实位打包）：
#     单对象取反、多对象并集取反、object_indices 子集、空帧→全选、
#     非法索引→空、packed None→按 orig_size 全帧、orig_size/n_frames 保留
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


# ── FakeArr：numpy 代理（支持 any(dim=)/~/索引/unsqueeze）──
class FakeArr:
    device = "cpu"

    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    def any(self, dim=None):
        return FakeArr(self.data.any() if dim is None else self.data.any(axis=dim))

    def __invert__(self):
        return FakeArr(np.logical_not(self.data))

    def __getitem__(self, key):
        return FakeArr(self.data[key])

    def unsqueeze(self, dim):
        return FakeArr(np.expand_dims(self.data, dim))

    def numpy(self):
        return self.data


def _stub_unpack(packed):
    p = packed.data if isinstance(packed, FakeArr) else np.asarray(packed)
    return FakeArr(numpy_unpack(p))


def _stub_pack(masks):
    m = masks.data if isinstance(masks, FakeArr) else masks
    return FakeArr(numpy_pack(m))


fake_torch = types.ModuleType("torch")
fake_torch.bool = np.bool_
fake_torch.ones = lambda shape, dtype=None: FakeArr(np.ones(shape, dtype=bool))
sys.modules["torch"] = fake_torch

for name in ("comfy", "comfy.ldm", "comfy.ldm.sam3"):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
tracker = types.ModuleType("comfy.ldm.sam3.tracker")
tracker.pack_masks = _stub_pack
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


mod = load(os.path.join(root, "nodes", "image", "invert_track.py"), "sfnodes.nodes.image.invert_track")
SFInvertTrackData = mod.SFInvertTrackData
invert_track_data = mod.invert_track_data


def invert(td, indices="", torch=fake_torch):
    return invert_track_data(td, object_indices=indices, unpack_masks=_stub_unpack,
                             pack_masks=_stub_pack, torch=torch)


def unpack_out(td):
    return numpy_unpack(td["packed_masks"].numpy())


# ── 1. 结构 ──
check("CATEGORY", SFInvertTrackData.CATEGORY == "sfnodes/image")
check("FUNCTION", SFInvertTrackData.FUNCTION == "execute")
check("RETURN_TYPES", SFInvertTrackData.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("RETURN_NAMES", SFInvertTrackData.RETURN_NAMES == ("track_data",))
check("DESCRIPTION 存在", isinstance(getattr(SFInvertTrackData, "DESCRIPTION", None), str)
      and SFInvertTrackData.DESCRIPTION.strip() != "")

schema = SFInvertTrackData.INPUT_TYPES()
check("track_data required SAM3_TRACK_DATA", schema["required"]["track_data"][0] == "SAM3_TRACK_DATA")
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
check("__init__ 注册 SFInvertTrackData 双字典一致",
      "SFInvertTrackData" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFInvertTrackData" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. invert_track_data 纯函数 ──
# 单对象：帧0 左半/帧1 空；orig 4x8
m0 = np.zeros((2, 1, 4, 8), dtype=bool)
m0[0, 0, :, :4] = True
packed = FakeArr(numpy_pack(m0))
td = {"packed_masks": packed, "n_frames": 2, "orig_size": (4, 8), "scores": [0.9]}

out = invert(td)
inv = unpack_out(out)
check("单对象取反帧0", np.array_equal(inv[0, 0], ~m0[0, 0]))
check("空帧取反→全选", inv[1, 0].all())
check("输出单身份 shape", out["packed_masks"].shape == (2, 1, 4, 1))
check("scores 重置为 [1.0]", out["scores"] == [1.0])
check("orig_size 保留", out["orig_size"] == (4, 8))
check("n_frames 保留", out["n_frames"] == 2)
check("原 track_data 未被改写", td["scores"] == [0.9])

# 多对象并集取反
m1 = np.zeros((1, 2, 4, 8), dtype=bool)
m1[0, 0, :, :2] = True
m1[0, 1, :, 6:] = True
td1 = {"packed_masks": FakeArr(numpy_pack(m1)), "n_frames": 1, "orig_size": (4, 8)}
inv1 = unpack_out(invert(td1))
expected = ~(m1[0, 0] | m1[0, 1])
check("多对象并集取反", np.array_equal(inv1[0, 0], expected))

# object_indices 子集：只反转对象1
inv2 = unpack_out(invert(td1, indices="1"))
check("object_indices 子集取反", np.array_equal(inv2[0, 0], ~m1[0, 1]))

# 非法索引 → 空身份
out3 = invert(td1, indices="9")
check("非法索引→packed None", out3["packed_masks"] is None and out3["scores"] == [])

# packed None → 按 orig_size 全帧（W 补到 8 的倍数）
td4 = {"packed_masks": None, "n_frames": 2, "orig_size": (3, 5)}
out4 = invert(td4)
full = unpack_out(out4)
check("packed None→全帧 shape", out4["packed_masks"].shape == (2, 1, 3, 1))
check("packed None→全帧内容", full.all() and out4["scores"] == [1.0])

# packed None 且无 n_frames/orig_size → 仍空
out5 = invert({"packed_masks": None, "n_frames": None, "orig_size": None})
check("packed None 无尺寸→空", out5["packed_masks"] is None)

# ── 3. execute 集成 ──
node = SFInvertTrackData()
res = node.execute(td)
check("execute 返回单元素 tuple", isinstance(res, tuple) and len(res) == 1)
check("execute 输出可解包", np.array_equal(unpack_out(res[0])[0, 0], ~m0[0, 0]))

try:
    node.execute({"not": "track"})
    check("execute 非法输入抛错", False)
except ValueError:
    check("execute 非法输入抛错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
