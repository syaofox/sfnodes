# SFTrackDataSubtract / sf_utils.track_data_ops 后端逻辑测试
# （Node/Python 直接运行：python tests/test_track_data_ops.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（track_data required / exclude_1..4 多类型）、根 __init__.py 注册键一致
#   - track_data_ops 纯函数（numpy 版 pack/unpack 桩，真实位打包）：
#     pad_track_data_front 前补空帧 / 已足够长 / packed None；
#     subtract_from_track_data 逐对象相减、多路并集、MASK 与 TRACK_DATA 两种排除、
#     尺寸不一致 resize、帧数不一致报错、空排除直通、packed None 直通
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


fake_torch = types.SimpleNamespace(
    zeros=lambda shape, dtype=None, device=None: FakeTensor(
        np.zeros(shape, dtype=np.uint8 if dtype is None else dtype), device=device or "cpu"),
    cat=lambda tensors, dim=0: FakeTensor(np.concatenate([_arr(t) for t in tensors], axis=dim)),
    nn=types.SimpleNamespace(functional=types.SimpleNamespace(interpolate=fake_interpolate)),
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
check("exclude 多类型 4 路", all(schema["optional"][f"exclude_{i}"][0] == "MASK,SAM3_TRACK_DATA"
                                for i in range(1, 5)))

init_keys = parse_init_keys()
check("__init__ 注册 SFTrackDataSubtract 双字典一致",
      "SFTrackDataSubtract" in init_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFTrackDataSubtract" in init_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
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

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
