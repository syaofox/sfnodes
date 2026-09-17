# SFTrackDataSubtract / SFTrackDataAdd / SFTrackDataSlice / sf_utils.track_data_ops
# 后端逻辑测试（Node/Python 直接运行：python tests/test_track_data_ops.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（track_data required / exclude_1..20 · add_1..20 多类型 /
#     slice 的 start·length）、根 __init__.py 注册键一致
#   - track_data_ops 纯函数（numpy 版 pack/unpack 桩，真实位打包）：
#     pad_track_data_front 前补空帧 / 已足够长 / packed None；
#     subtract_from_track_data 逐对象相减、多路并集、MASK 与 TRACK_DATA 两种排除、
#     尺寸不一致 resize、帧数不一致报错、空排除直通、packed None 直通；
#     add_to_track_data 逐帧并集塌单身份、多路并集、MASK 与 TRACK_DATA 两种叠加、
#     resize、帧数不一致报错、空基础+叠加、空基础无叠加直通；
#     slice_track_data 区间/负 start/到尾/截断/越界空/保留字段/packed None
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


class FakeTensor:
    def __init__(self, data, dtype=None, device="cpu"):
        self.data = np.asarray(data)
        self.dtype = self.data.dtype if dtype is None else dtype
        self.device = device

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def dim(self):
        return self.data.ndim

    def numpy(self):
        return self.data

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, dim))

    def contiguous(self):
        return FakeTensor(np.ascontiguousarray(self.data), dtype=self.dtype, device=self.device)

    def any(self, dim=None):
        return FakeTensor(self.data.any() if dim is None else self.data.any(axis=dim))

    def float(self):
        return FakeTensor(self.data.astype(np.float32))

    def __invert__(self):
        return FakeTensor(np.logical_not(self.data))

    def __and__(self, other):
        return FakeTensor(self.data & _arr(other))

    def __or__(self, other):
        return FakeTensor(self.data | _arr(other))

    def __getitem__(self, key):
        return FakeTensor(self.data[key])

    def __gt__(self, other):
        return FakeTensor(self.data > other)


def _arr(x):
    return x.data if isinstance(x, FakeTensor) else np.asarray(x)


def _stub_pack(masks):
    return FakeTensor(numpy_pack(_arr(masks)))


def _stub_unpack(packed):
    return FakeTensor(numpy_unpack(_arr(packed)))


def fake_interpolate(t, size, mode="nearest"):
    a = _arr(t)
    tD, c, h, w = a.shape
    th, tw = size
    ys = np.clip((np.arange(th) * h / th).astype(int), 0, h - 1)
    xs = np.clip((np.arange(tw) * w / tw).astype(int), 0, w - 1)
    out = a[:, :, ys][:, :, :, xs]
    return FakeTensor(out)


def fake_pad(t, pads, mode="constant", value=0):
    a = _arr(t)
    left, right = pads
    return FakeTensor(np.pad(a, [(0, 0)] * (a.ndim - 1) + [(left, right)], mode="constant"))


fake_torch = types.SimpleNamespace(
    bool=np.bool_,
    zeros=lambda shape, dtype=None, device=None: FakeTensor(
        np.zeros(shape, dtype=np.uint8 if dtype is None else dtype), device=device or "cpu"),
    cat=lambda tensors, dim=0: FakeTensor(np.concatenate([_arr(t) for t in tensors], axis=dim)),
    nn=types.SimpleNamespace(functional=types.SimpleNamespace(interpolate=fake_interpolate, pad=fake_pad)),
)
sys.modules["torch"] = fake_torch

for name in ("comfy", "comfy.ldm", "comfy.ldm.sam3"):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
tracker = types.ModuleType("comfy.ldm.sam3.tracker")
tracker.pack_masks = _stub_pack
tracker.unpack_masks = _stub_unpack
sys.modules["comfy.ldm.sam3.tracker"] = tracker

# 注册 sfnodes 包结构使节点相对导入解析
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


ops = load(os.path.join(root, "sf_utils", "track_data_ops.py"), "sfnodes.sf_utils.track_data_ops")
mod = load(os.path.join(root, "nodes", "image", "track_data_subtract.py"),
           "sfnodes.nodes.image.track_data_subtract")
SFTrackDataSubtract = mod.SFTrackDataSubtract
add_mod = load(os.path.join(root, "nodes", "image", "track_data_add.py"),
               "sfnodes.nodes.image.track_data_add")
SFTrackDataAdd = add_mod.SFTrackDataAdd
merge_mod = load(os.path.join(root, "nodes", "image", "track_data_merge.py"),
                 "sfnodes.nodes.image.track_data_merge")
SFTrackDataMerge = merge_mod.SFTrackDataMerge
slice_mod = load(os.path.join(root, "nodes", "image", "track_data_slice.py"),
                 "sfnodes.nodes.image.track_data_slice")
SFTrackDataSlice = slice_mod.SFTrackDataSlice


def parse_init_keys():
    with open(os.path.join(root, "__init__.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    keys = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
                    if isinstance(node.value, ast.Dict):
                        keys[t.id] = {ast.literal_eval(k) for k in node.value.keys if k is not None}
    return keys


# ── 1. 结构 ──
check("CATEGORY", SFTrackDataSubtract.CATEGORY == "sfnodes/image")
check("FUNCTION", SFTrackDataSubtract.FUNCTION == "execute")
check("RETURN_TYPES", SFTrackDataSubtract.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("RETURN_NAMES", SFTrackDataSubtract.RETURN_NAMES == ("track_data",))
check("DESCRIPTION 存在", isinstance(getattr(SFTrackDataSubtract, "DESCRIPTION", None), str)
      and SFTrackDataSubtract.DESCRIPTION.strip() != "")

schema = SFTrackDataSubtract.INPUT_TYPES()
check("track_data required SAM3_TRACK_DATA", schema["required"]["track_data"][0] == "SAM3_TRACK_DATA")
check("exclude 多类型 20 路", all(schema["optional"][f"exclude_{i}"][0] == "MASK,SAM3_TRACK_DATA"
                                for i in range(1, 21)))

check("Add CATEGORY", SFTrackDataAdd.CATEGORY == "sfnodes/image")
check("Add FUNCTION", SFTrackDataAdd.FUNCTION == "execute")
check("Add RETURN_TYPES", SFTrackDataAdd.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("Add RETURN_NAMES", SFTrackDataAdd.RETURN_NAMES == ("track_data",))
check("Add DESCRIPTION 存在", isinstance(getattr(SFTrackDataAdd, "DESCRIPTION", None), str)
      and SFTrackDataAdd.DESCRIPTION.strip() != "")
add_schema = SFTrackDataAdd.INPUT_TYPES()
check("Add track_data required SAM3_TRACK_DATA", add_schema["required"]["track_data"][0] == "SAM3_TRACK_DATA")
check("add 多类型 20 路", all(add_schema["optional"][f"add_{i}"][0] == "MASK,SAM3_TRACK_DATA"
                             for i in range(1, 21)))

check("Merge CATEGORY", SFTrackDataMerge.CATEGORY == "sfnodes/image")
check("Merge FUNCTION", SFTrackDataMerge.FUNCTION == "execute")
check("Merge RETURN_TYPES", SFTrackDataMerge.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("Merge RETURN_NAMES", SFTrackDataMerge.RETURN_NAMES == ("track_data",))
check("Merge DESCRIPTION 存在", isinstance(getattr(SFTrackDataMerge, "DESCRIPTION", None), str)
      and SFTrackDataMerge.DESCRIPTION.strip() != "")
merge_schema = SFTrackDataMerge.INPUT_TYPES()
check("Merge track_data required SAM3_TRACK_DATA",
      merge_schema["required"]["track_data"][0] == "SAM3_TRACK_DATA")
check("Merge track 多类型 20 路", all(merge_schema["optional"][f"track_{i}"][0] == "MASK,SAM3_TRACK_DATA"
                                  for i in range(1, 21)))
check("Merge hidden SlotModes STRING", merge_schema["hidden"]["SlotModes"][0] == "STRING")
check("Merge _parse_modes 容错", merge_mod._parse_modes('{"track_1": "add"}') == {"track_1": "add"}
      and merge_mod._parse_modes("garbage") == {} and merge_mod._parse_modes('{"track_2": "x"}') == {"track_2": "sub"})

check("Slice CATEGORY", SFTrackDataSlice.CATEGORY == "sfnodes/image")
check("Slice FUNCTION", SFTrackDataSlice.FUNCTION == "execute")
check("Slice RETURN_TYPES", SFTrackDataSlice.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("Slice RETURN_NAMES", SFTrackDataSlice.RETURN_NAMES == ("track_data",))
check("Slice DESCRIPTION 存在", isinstance(getattr(SFTrackDataSlice, "DESCRIPTION", None), str)
      and SFTrackDataSlice.DESCRIPTION.strip() != "")
slice_schema = SFTrackDataSlice.INPUT_TYPES()
check("Slice track_data required SAM3_TRACK_DATA",
      slice_schema["required"]["track_data"][0] == "SAM3_TRACK_DATA")
check("Slice start INT 支持负值", slice_schema["required"]["start"][0] == "INT"
      and slice_schema["required"]["start"][1].get("min", 0) < 0)
check("Slice length INT min 0", slice_schema["required"]["length"][0] == "INT"
      and slice_schema["required"]["length"][1].get("min") == 0)

init_keys = parse_init_keys()
for _node in ("SFTrackDataSubtract", "SFTrackDataAdd", "SFTrackDataMerge", "SFTrackDataSlice"):
    check(f"__init__ 注册 {_node} 双字典一致",
          _node in init_keys.get("NODE_CLASS_MAPPINGS", set())
          and _node in init_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      init_keys.get("NODE_CLASS_MAPPINGS", set()) == init_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))


# ── 2. pad_track_data_front ──
base_masks = np.zeros((2, 1, 4, 8), dtype=bool)
base_masks[0, 0, :, :3] = True
td = {"packed_masks": FakeTensor(numpy_pack(base_masks)), "n_frames": 2, "orig_size": (4, 8), "scores": [0.5]}

padded = ops.pad_track_data_front(td, 5, torch=fake_torch)
check("补帧 n_frames", padded["n_frames"] == 5)
check("补帧 packed shape", padded["packed_masks"].shape == (5, 1, 4, 1))
u = numpy_unpack(_arr(padded["packed_masks"]))
check("补帧前 3 帧全空", not u[:3].any())
check("补帧保留原内容", np.array_equal(u[3:], base_masks))
check("补帧不改原输入", td["n_frames"] == 2)

check("已足够长原样返回", ops.pad_track_data_front(td, 1, torch=fake_torch)["n_frames"] == 2)
td_none = {"packed_masks": None, "n_frames": 2, "orig_size": (4, 8)}
check("packed None 仅补 n_frames",
      ops.pad_track_data_front(td_none, 6, torch=fake_torch)["n_frames"] == 6)


# ── 3. subtract_from_track_data ──
# 基础：2 帧，1 对象；帧0 左半，帧1 全屏
base = np.zeros((2, 1, 4, 8), dtype=bool)
base[0, 0, :, :4] = True
base[1, 0, :, :] = True
td_base = {"packed_masks": FakeTensor(numpy_pack(base)), "n_frames": 2, "orig_size": (4, 8), "scores": [0.9]}

# MASK 排除：帧0 吃掉左侧 2 列，帧1 吃掉右侧 4 列
ex = np.zeros((2, 4, 8), dtype=bool)
ex[0, :, :2] = True
ex[1, :, 4:] = True
out = ops.subtract_from_track_data(td_base, [FakeTensor(ex)], pack_masks=_stub_pack,
                                   unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
ub = numpy_unpack(_arr(out["packed_masks"]))
expected0 = base[0, 0] & ~ex[0]
expected1 = base[1, 0] & ~ex[1]
check("MASK 排除帧0", np.array_equal(ub[0, 0], expected0))
check("MASK 排除帧1", np.array_equal(ub[1, 0], expected1))
check("保持对象数", out["packed_masks"].shape[1] == 1)
check("orig_size 保留", out["orig_size"] == (4, 8))
check("scores 保留", out["scores"] == [0.9])
check("不改原输入", td_base["scores"] == [0.9])

# 多路并集
ex2 = np.zeros((2, 4, 8), dtype=bool)
ex2[0, :, 2:4] = True
out2 = ops.subtract_from_track_data(td_base, [FakeTensor(ex), FakeTensor(ex2)], pack_masks=_stub_pack,
                                    unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
u2 = numpy_unpack(_arr(out2["packed_masks"]))
check("多路并集帧0", np.array_equal(u2[0, 0], base[0, 0] & ~(ex[0] | ex2[0])))
check("多路并集帧1（ex2 不影响）", np.array_equal(u2[1, 0], base[1, 0] & ~ex[1]))

# 多对象：各自减去并集，互不影响
multi = np.zeros((1, 2, 4, 8), dtype=bool)
multi[0, 0, :, :4] = True
multi[0, 1, :, 4:] = True
td_multi = {"packed_masks": FakeTensor(numpy_pack(multi)), "n_frames": 1, "orig_size": (4, 8)}
exm = np.zeros((1, 4, 8), dtype=bool)
exm[0, :, :2] = True
out_m = ops.subtract_from_track_data(td_multi, [FakeTensor(exm)], pack_masks=_stub_pack,
                                     unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
um = numpy_unpack(_arr(out_m["packed_masks"]))
check("多对象对象数保留", um.shape[1] == 2)
check("多对象对象0被减", np.array_equal(um[0, 0], multi[0, 0] & ~exm[0]))
check("多对象对象1不重叠", np.array_equal(um[0, 1], multi[0, 1]))

# SAM3_TRACK_DATA 作为排除输入
ex_td = {"packed_masks": FakeTensor(numpy_pack(ex[:, None])), "n_frames": 2, "orig_size": (4, 8)}
out_td = ops.subtract_from_track_data(td_base, [ex_td], pack_masks=_stub_pack,
                                      unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
check("TRACK_DATA 排除等价", np.array_equal(numpy_unpack(_arr(out_td["packed_masks"])), ub))

# resize：基础 4x8，排除 4x16（右半为 True）→ 缩到 4x8 后右 4 列 True
ex_big = np.zeros((2, 4, 16), dtype=bool)
ex_big[:, :, 8:] = True
out_rz = ops.subtract_from_track_data(td_base, [FakeTensor(ex_big)], pack_masks=_stub_pack,
                                      unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
check("resize 路径不崩", out_rz["packed_masks"] is not None)

# 4D 排除通道维必须为 1
try:
    ops.subtract_from_track_data(td_base, [np.zeros((2, 2, 4, 8), dtype=bool)],
                                 pack_masks=_stub_pack, unpack_masks=_stub_unpack,
                                 torch=fake_torch, interpolate=fake_interpolate)
    check("4D 多通道报错", False)
except ValueError:
    check("4D 多通道报错", True)

# 帧数不一致报错
try:
    ops.subtract_from_track_data(td_base, [np.zeros((3, 4, 8), dtype=bool)],
                                 pack_masks=_stub_pack, unpack_masks=_stub_unpack,
                                 torch=fake_torch, interpolate=fake_interpolate)
    check("帧数不一致报错", False)
except ValueError:
    check("帧数不一致报错", True)

# 空排除 / packed None 直通
check("空排除直通", ops.subtract_from_track_data(td_base, [None, None], pack_masks=_stub_pack,
                                               unpack_masks=_stub_unpack, torch=fake_torch,
                                               interpolate=fake_interpolate)["packed_masks"].shape == (2, 1, 4, 1))
check("packed None 直通", ops.subtract_from_track_data({"packed_masks": None, "n_frames": 2},
                                                     [ex], pack_masks=_stub_pack, unpack_masks=_stub_unpack,
                                                     torch=fake_torch, interpolate=fake_interpolate)["packed_masks"] is None)


# ── 3b. add_to_track_data ──
add_a = np.zeros((2, 4, 8), dtype=bool)
add_a[0, :, 2:6] = True
add_a[1, :, 4:] = True
out_add = ops.add_to_track_data(td_base, [FakeTensor(add_a)], pack_masks=_stub_pack,
                                unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
ua = numpy_unpack(_arr(out_add["packed_masks"]))
check("并集单身份", out_add["packed_masks"].shape[1] == 1)
check("并集帧0", np.array_equal(ua[0, 0], base[0, 0] | add_a[0]))
check("并集帧1", np.array_equal(ua[1, 0], base[1, 0] | add_a[1]))
check("并集 orig_size 保留", out_add["orig_size"] == (4, 8))
check("并集 n_frames 保留", out_add["n_frames"] == 2)
check("并集 scores=[1.0]", out_add["scores"] == [1.0])
check("并集不改原输入", td_base["scores"] == [0.9])

# 多路并集
add_b = np.zeros((2, 4, 8), dtype=bool)
add_b[0, :, :1] = True
out_add2 = ops.add_to_track_data(td_base, [FakeTensor(add_a), FakeTensor(add_b)], pack_masks=_stub_pack,
                                 unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
ua2 = numpy_unpack(_arr(out_add2["packed_masks"]))
check("多路并集帧0", np.array_equal(ua2[0, 0], base[0, 0] | add_a[0] | add_b[0]))

# 多对象基础塌成单身份
add_m1 = add_a[:1]
out_multi = ops.add_to_track_data(td_multi, [FakeTensor(add_m1)], pack_masks=_stub_pack,
                                  unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
um2 = numpy_unpack(_arr(out_multi["packed_masks"]))
check("多对象塌单身份", out_multi["packed_masks"].shape[1] == 1)
check("多对象并集内容", np.array_equal(um2[0, 0], multi[0, 0] | multi[0, 1] | add_m1[0]))

# TRACK_DATA 作为叠加输入
add_td = {"packed_masks": FakeTensor(numpy_pack(add_a[:, None])), "n_frames": 2, "orig_size": (4, 8)}
out_add_td = ops.add_to_track_data(td_base, [add_td], pack_masks=_stub_pack,
                                   unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
check("TRACK_DATA 叠加等价", np.array_equal(numpy_unpack(_arr(out_add_td["packed_masks"])), ua))

# resize：叠加 4x16（右半 True）→ 缩到 4x8 后右 4 列 True
add_big = np.zeros((2, 4, 16), dtype=bool)
add_big[:, :, 8:] = True
out_add_rz = ops.add_to_track_data(td_base, [FakeTensor(add_big)], pack_masks=_stub_pack,
                                   unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
check("并集 resize 路径不崩", out_add_rz["packed_masks"] is not None)

# 帧数不一致报错
try:
    ops.add_to_track_data(td_base, [np.zeros((3, 4, 8), dtype=bool)], pack_masks=_stub_pack,
                          unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
    check("并集帧数不一致报错", False)
except ValueError:
    check("并集帧数不一致报错", True)

# 空基础 + 叠加（orig_size 已知）→ 输出即叠加内容
td_none = {"packed_masks": None, "n_frames": 2, "orig_size": (4, 8)}
out_none = ops.add_to_track_data(td_none, [FakeTensor(add_a)], pack_masks=_stub_pack,
                                 unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
un = numpy_unpack(_arr(out_none["packed_masks"]))
check("空基础并集单身份", out_none["packed_masks"].shape[1] == 1)
check("空基础并集内容", np.array_equal(un, add_a[:, None]))

# 空基础 + 宽度非 8 倍数 MASK（orig_size 缺失，走补宽路径）
td_none2 = {"packed_masks": None, "n_frames": 2}
add_w6 = np.zeros((2, 4, 6), dtype=bool)
add_w6[:, :, :3] = True
out_w6 = ops.add_to_track_data(td_none2, [FakeTensor(add_w6)], pack_masks=_stub_pack,
                               unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
uw6 = numpy_unpack(_arr(out_w6["packed_masks"]))
check("空基础补宽到 8", uw6.shape[-1] == 8)
check("空基础补宽内容保留", np.array_equal(uw6[:, 0, :, :6], add_w6))

# 空基础无叠加直通
check("空基础无叠加直通", ops.add_to_track_data({"packed_masks": None, "n_frames": 2},
                                          [], pack_masks=_stub_pack, unpack_masks=_stub_unpack,
                                          torch=fake_torch, interpolate=fake_interpolate)["packed_masks"] is None)


# track_data 工作分辨率（方形 packed）≠ 真实宽高（orig_size）：add 必须保留 orig_size
sq = np.zeros((2, 1, 8, 8), dtype=bool)
sq[0, 0, :4, :] = True
td_sq = {"packed_masks": FakeTensor(numpy_pack(sq)), "n_frames": 2, "orig_size": (4, 16), "scores": [0.7]}
add_sq = np.zeros((2, 8, 8), dtype=bool)
add_sq[1, :, :] = True
out_sq = ops.add_to_track_data(td_sq, [FakeTensor(add_sq)], pack_masks=_stub_pack,
                               unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
check("add 保留非方形 orig_size（方形工作网格）", out_sq["orig_size"] == (4, 16))
check("add 输出仍为工作网格 packed", out_sq["packed_masks"].shape == (2, 1, 8, 1))
check("add 保留 n_frames", out_sq["n_frames"] == 2)
check("add 保留 orig_size 不改原输入", td_sq["orig_size"] == (4, 16))


# ── 3c. merge_track_data（先减后加组合）──
# 仅相减：保留对象数、scores
out_sub_only = ops.merge_track_data(td_base, [FakeTensor(ex)], [], pack_masks=_stub_pack,
                                    unpack_masks=_stub_unpack, torch=fake_torch,
                                    interpolate=fake_interpolate)
check("merge 仅相减保留对象数", out_sub_only["packed_masks"].shape[1] == 1)
check("merge 仅相减内容", np.array_equal(numpy_unpack(_arr(out_sub_only["packed_masks"]))[0, 0], expected0))
check("merge 仅相减保留 scores", out_sub_only["scores"] == [0.9])

# 先减后加：塌单身份，减掉的区域不会因 add 复活（先减后加）
out_mix = ops.merge_track_data(td_base, [FakeTensor(ex)], [FakeTensor(add_a)], pack_masks=_stub_pack,
                               unpack_masks=_stub_unpack, torch=fake_torch, interpolate=fake_interpolate)
umix = numpy_unpack(_arr(out_mix["packed_masks"]))
expected_mix = (base[0, 0] & ~ex[0]) | add_a[0]
check("merge 先减后加塌单身份", out_mix["packed_masks"].shape[1] == 1)
check("merge 先减后加内容", np.array_equal(umix[0, 0], expected_mix))
check("merge 先减后加 scores=[1.0]", out_mix["scores"] == [1.0])

# 仅叠加（无相减）：等价 add_to_track_data
out_add_only = ops.merge_track_data(td_base, [], [FakeTensor(add_a)], pack_masks=_stub_pack,
                                    unpack_masks=_stub_unpack, torch=fake_torch,
                                    interpolate=fake_interpolate)
check("merge 仅叠加等价 add", np.array_equal(numpy_unpack(_arr(out_add_only["packed_masks"])), ua))

# 空基础 + 叠加
out_empty = ops.merge_track_data({"packed_masks": None, "n_frames": 2, "orig_size": (4, 8)},
                                 [], [FakeTensor(add_a)], pack_masks=_stub_pack,
                                 unpack_masks=_stub_unpack, torch=fake_torch,
                                 interpolate=fake_interpolate)
check("merge 空基础叠加单身份", out_empty["packed_masks"].shape[1] == 1)
check("merge 空基础叠加内容", np.array_equal(numpy_unpack(_arr(out_empty["packed_masks"])), add_a[:, None]))

# 全空直通
check("merge 全空直通", ops.merge_track_data(td_base, [], [None], pack_masks=_stub_pack,
                                         unpack_masks=_stub_unpack, torch=fake_torch,
                                         interpolate=fake_interpolate)["scores"] == [0.9])

# 方形工作网格 + 非方形 orig_size：先减后加后仍保留真实宽高
ex_sq = np.zeros((2, 8, 8), dtype=bool)
ex_sq[0, :, :2] = True
out_ms = ops.merge_track_data(td_sq, [FakeTensor(ex_sq)], [FakeTensor(add_sq)],
                              pack_masks=_stub_pack, unpack_masks=_stub_unpack,
                              torch=fake_torch, interpolate=fake_interpolate)
check("merge 减+加保留非方形 orig_size", out_ms["orig_size"] == (4, 16))
check("merge 减+加塌单身份", out_ms["packed_masks"].shape[1] == 1)


# ── 3d. slice_track_data（时间维切片）──
sl = ops.slice_track_data(td_base, 1, 1)
check("slice 帧数", sl["n_frames"] == 1)
check("slice packed 首维", sl["packed_masks"].shape == (1, 1, 4, 1))
check("slice 内容等于第 1 帧", np.array_equal(numpy_unpack(_arr(sl["packed_masks"])), base[1:2]))
check("slice 保留 orig_size", sl["orig_size"] == (4, 8))
check("slice 保留 scores", sl["scores"] == [0.9])
check("slice 不改原输入", td_base["n_frames"] == 2 and td_base["packed_masks"].shape[0] == 2)

sl_neg = ops.slice_track_data(td_base, -1, 1)
check("slice 负 start 取尾", np.array_equal(numpy_unpack(_arr(sl_neg["packed_masks"])), base[1:2]))

sl_tail = ops.slice_track_data(td_base, -1, 0)
check("slice 负 start + length=0 到结尾", sl_tail["n_frames"] == 1)

sl_all = ops.slice_track_data(td_base, 0, 0)
check("slice length=0 全量", sl_all["n_frames"] == 2
      and np.array_equal(numpy_unpack(_arr(sl_all["packed_masks"])), base))

sl_clip = ops.slice_track_data(td_base, 1, 99)
check("slice 尾部截断", sl_clip["n_frames"] == 1)

sl_empty = ops.slice_track_data(td_base, 5, 2)
check("slice start 越界为空", sl_empty["n_frames"] == 0 and sl_empty["packed_masks"].shape[0] == 0)

sl_under = ops.slice_track_data(td_base, -99, 1)
check("slice 负值越界从头", np.array_equal(numpy_unpack(_arr(sl_under["packed_masks"])), base[:1]))

sl_none = ops.slice_track_data({"packed_masks": None, "n_frames": 5, "orig_size": (4, 8)}, 1, 2)
check("slice packed None 只调 n_frames", sl_none["packed_masks"] is None and sl_none["n_frames"] == 2
      and sl_none["orig_size"] == (4, 8))

sl_multi = ops.slice_track_data(td_multi, 0, 1)
check("slice 保留对象数", sl_multi["packed_masks"].shape[1] == 2)


# ── 3e. concat_track_data_segments（分段重锚拼接）──
seg_a = np.zeros((2, 1, 4, 8), dtype=bool)
seg_a[0, 0, :, :4] = True
seg_a[1, 0, :, :] = True
td_a = {"packed_masks": FakeTensor(numpy_pack(seg_a)), "n_frames": 2, "orig_size": (4, 8), "scores": [0.9]}
# 2 对象段：并集塌单身份
seg_b = np.zeros((2, 2, 4, 8), dtype=bool)
seg_b[0, 0, :, :2] = True
seg_b[0, 1, :, 4:6] = True
seg_b[1, 0, :, :] = True
td_b = {"packed_masks": FakeTensor(numpy_pack(seg_b)), "n_frames": 2, "orig_size": (4, 8), "scores": [1.0, 0.8]}

cc = ops.concat_track_data_segments([(0, td_a), (3, td_b)], 5, torch=fake_torch)
ucc = numpy_unpack(_arr(cc["packed_masks"]))
check("concat 全长/单身份", cc["packed_masks"].shape == (5, 1, 4, 1))
check("concat n_frames", cc["n_frames"] == 5)
check("concat scores=[1.0]", cc["scores"] == [1.0])
check("concat orig_size 取首段", cc["orig_size"] == (4, 8))
check("concat 段0帧0", np.array_equal(ucc[0, 0], seg_a[0, 0]))
check("concat 段0帧1", np.array_equal(ucc[1, 0], seg_a[1, 0]))
check("concat 段间空隙补零", not ucc[2].any())
check("concat 段1多对象并集", np.array_equal(ucc[3, 0], seg_b[0, 0] | seg_b[0, 1]))
check("concat 段1末帧", np.array_equal(ucc[4, 0], seg_b[1, 0]))

# 多对象段在前 + 空隙补零（补零帧对象维须为 1，回归：曾用 ref.shape[1:] 导致 2 通道）
cc_multi_first = ops.concat_track_data_segments([(1, td_b), (4, td_a)], 6, torch=fake_torch)
umf = numpy_unpack(_arr(cc_multi_first["packed_masks"]))
check("concat 多对象段在前补零单身份", cc_multi_first["packed_masks"].shape == (6, 1, 4, 1)
      and not umf[0].any() and np.array_equal(umf[1, 0], seg_b[0, 0] | seg_b[0, 1])
      and not umf[3].any() and np.array_equal(umf[4, 0], seg_a[0, 0]))

# 首锚 >0：前补空帧
cc_front = ops.concat_track_data_segments([(2, td_a)], 4, torch=fake_torch)
uf = numpy_unpack(_arr(cc_front["packed_masks"]))
check("concat 首锚前补空帧", cc_front["packed_masks"].shape == (4, 1, 4, 1)
      and not uf[:2].any() and np.array_equal(uf[2:], seg_a))

# 空段（packed None）跳过，其帧区间补零
td_empty = {"packed_masks": None, "n_frames": 1, "orig_size": (4, 8)}
cc_empty = ops.concat_track_data_segments([(0, td_a), (2, td_empty), (3, td_b)], 5, torch=fake_torch)
ue = numpy_unpack(_arr(cc_empty["packed_masks"]))
check("concat 空段跳过补零", not ue[2].any() and np.array_equal(ue[3, 0], seg_b[0, 0] | seg_b[0, 1]))

# 单段即全长（不走 cat）
cc_single = ops.concat_track_data_segments([(0, td_a)], 2, torch=fake_torch)
check("concat 单段直通", np.array_equal(numpy_unpack(_arr(cc_single["packed_masks"])), seg_a))

# 全空：packed None 仅保留 n_frames/orig_size
cc_none = ops.concat_track_data_segments([(0, td_empty)], 3, torch=fake_torch)
check("concat 全空 packed None", cc_none["packed_masks"] is None and cc_none["n_frames"] == 3
      and cc_none["orig_size"] == (4, 8) and cc_none["scores"] == [])

# 重叠 / 越界 / 网格不一致报错
for _name, _segs in [
    ("concat 重叠报错", [(0, td_a), (1, td_b)]),
    ("concat 越界报错", [(4, td_a)]),
]:
    try:
        ops.concat_track_data_segments(_segs, 5, torch=fake_torch)
        check(_name, False)
    except ValueError:
        check(_name, True)
try:
    td_grid = {"packed_masks": FakeTensor(numpy_pack(np.zeros((1, 1, 8, 8), dtype=bool))),
               "n_frames": 1, "orig_size": (8, 8)}
    ops.concat_track_data_segments([(0, td_a), (2, td_grid)], 5, torch=fake_torch)
    check("concat 网格不一致报错", False)
except ValueError:
    check("concat 网格不一致报错", True)


# ── 4. execute 集成 ──
node = SFTrackDataSubtract()
res = node.execute(td_base, exclude_1=FakeTensor(ex))
check("execute 返回单元素 tuple", isinstance(res, tuple) and len(res) == 1)
check("execute 输出可解包", np.array_equal(numpy_unpack(_arr(res[0]["packed_masks"]))[0, 0], expected0))
check("execute 未接排除直通",
      numpy_unpack(_arr(node.execute(td_base)[0]["packed_masks"])).shape == (2, 1, 4, 8))

try:
    node.execute({"not": "track"})
    check("execute 非法输入抛错", False)
except ValueError:
    check("execute 非法输入抛错", True)

add_node = SFTrackDataAdd()
res_add = add_node.execute(td_base, add_1=FakeTensor(add_a))
check("Add execute 返回单元素 tuple", isinstance(res_add, tuple) and len(res_add) == 1)
check("Add execute 输出单身份", res_add[0]["packed_masks"].shape[1] == 1)
check("Add execute 并集内容",
      np.array_equal(numpy_unpack(_arr(res_add[0]["packed_masks"]))[0, 0], base[0, 0] | add_a[0]))
check("Add execute 未接叠加直通",
      numpy_unpack(_arr(add_node.execute(td_base)[0]["packed_masks"])).shape == (2, 1, 4, 8))

try:
    add_node.execute({"not": "track"})
    check("Add execute 非法输入抛错", False)
except ValueError:
    check("Add execute 非法输入抛错", True)

merge_node = SFTrackDataMerge()
# 默认模式 = sub（track_1 未在 SlotModes 中出现）
res_default = merge_node.execute(td_base, track_1=FakeTensor(ex))
check("Merge 默认 sub 保留对象数", res_default[0]["packed_masks"].shape[1] == 1)
check("Merge 默认 sub 内容",
      np.array_equal(numpy_unpack(_arr(res_default[0]["packed_masks"]))[0, 0], expected0))
# 显式 add + sub 混合
res_mix = merge_node.execute(td_base, '{"track_1": "sub", "track_2": "add"}',
                             track_1=FakeTensor(ex), track_2=FakeTensor(add_a))
check("Merge execute 混合塌单身份", res_mix[0]["packed_masks"].shape[1] == 1)
check("Merge execute 混合内容",
      np.array_equal(numpy_unpack(_arr(res_mix[0]["packed_masks"]))[0, 0], expected_mix))
check("Merge execute 返回单元素", isinstance(res_mix, tuple) and len(res_mix) == 1)
# execute 层：方形工作网格 + 非方形 orig_size 经 + 后仍保留真实宽高
res_sq = merge_node.execute(td_sq, '{"track_2": "add"}',
                            track_1=FakeTensor(ex_sq), track_2=FakeTensor(add_sq))
check("Merge execute 保留非方形 orig_size", res_sq[0]["orig_size"] == (4, 16))
try:
    merge_node.execute({"not": "track"})
    check("Merge execute 非法输入抛错", False)
except ValueError:
    check("Merge execute 非法输入抛错", True)

slice_node = SFTrackDataSlice()
res_slice = slice_node.execute(td_base, 1, 1)
check("Slice execute 返回单元素", isinstance(res_slice, tuple) and len(res_slice) == 1)
check("Slice execute 内容",
      np.array_equal(numpy_unpack(_arr(res_slice[0]["packed_masks"])), base[1:2]))
check("Slice execute 默认参数全量", slice_node.execute(td_base)[0]["n_frames"] == 2)
try:
    slice_node.execute({"not": "track"})
    check("Slice execute 非法输入抛错", False)
except ValueError:
    check("Slice execute 非法输入抛错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
