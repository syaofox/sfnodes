# SFSAM3PointTrack 后端逻辑测试（Node/Python 直接运行：python tests/test_sam3_point_track.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES、根 __init__.py 注册键一致
#   - execute：点提示 → SAM3_Detect 锚帧检测 → SAM3_VideoTrack 传播；
#     anchor_frame=0 不补帧、anchor_frame>0 前补空帧回全长；
#     initial_mask 直通跳过点检测；点/遮罩都缺时报错；anchor 越界报错；非 IMAGE 报错
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

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, dim))

    def numpy(self):
        return self.data

    def __getitem__(self, key):
        return FakeTensor(self.data[key])


fake_torch = types.SimpleNamespace(
    zeros=lambda shape, dtype=None, device=None: FakeTensor(
        np.zeros(shape, dtype=np.uint8 if dtype is None else dtype), device=device or "cpu"),
    cat=lambda tensors, dim=0: FakeTensor(np.concatenate([t.data for t in tensors], axis=dim)),
)
sys.modules["torch"] = fake_torch


# ── fake 核心 SAM3 节点 ──
class FakeOut:
    def __init__(self, *result):
        self.result = result


calls = {}


class FakeSAM3Detect:
    @classmethod
    def execute(cls, **kwargs):
        calls["detect"] = kwargs
        mask = np.zeros((1, 4, 8), dtype=bool)
        mask[0, :, :4] = True
        return FakeOut(FakeTensor(mask))


class FakeSAM3VideoTrack:
    @classmethod
    def execute(cls, **kwargs):
        calls["track"] = kwargs
        n = kwargs["images"].shape[0]
        masks = np.zeros((n, 1, 4, 8), dtype=bool)
        masks[:, 0, :, :2] = True
        return FakeOut({"packed_masks": FakeTensor(numpy_pack(masks)), "n_frames": n,
                        "orig_size": (4, 8), "scores": [1.0]})


ce = types.ModuleType("comfy_extras")
ce.__path__ = []
sys.modules["comfy_extras"] = ce
ce_sam3 = types.ModuleType("comfy_extras.nodes_sam3")
ce_sam3.SAM3_Detect = FakeSAM3Detect
ce_sam3.SAM3_VideoTrack = FakeSAM3VideoTrack
sys.modules["comfy_extras.nodes_sam3"] = ce_sam3

# 注册 sfnodes 包结构使节点相对导入解析
for _pkg, _rel in [("sfnodes", "."), ("sfnodes.nodes", "nodes"),
                   ("sfnodes.nodes.video", "nodes/video"), ("sfnodes.sf_utils", "sf_utils")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [os.path.join(root, _rel)]
    sys.modules[_pkg] = _m


def load(modpath, modname):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = load(os.path.join(root, "nodes", "video", "sam3_point_track.py"), "sfnodes.nodes.video.sam3_point_track")
SFSAM3PointTrack = mod.SFSAM3PointTrack

# ── 1. 结构 ──
check("CATEGORY", SFSAM3PointTrack.CATEGORY == "sfnodes/video")
check("FUNCTION", SFSAM3PointTrack.FUNCTION == "execute")
check("RETURN_TYPES", SFSAM3PointTrack.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("RETURN_NAMES", SFSAM3PointTrack.RETURN_NAMES == ("track_data",))
check("DESCRIPTION 存在", isinstance(getattr(SFSAM3PointTrack, "DESCRIPTION", None), str)
      and SFSAM3PointTrack.DESCRIPTION.strip() != "")

schema = SFSAM3PointTrack.INPUT_TYPES()
check("images required IMAGE", schema["required"]["images"][0] == "IMAGE")
check("model required MODEL", schema["required"]["model"][0] == "MODEL")
check("positive_coords optional forceInput",
      schema["optional"]["positive_coords"][0] == "STRING"
      and schema["optional"]["positive_coords"][1].get("forceInput") is True)

with open(os.path.join(root, "__init__.py"), encoding="utf-8") as f:
    tree = ast.parse(f.read())
keys = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
                if isinstance(node.value, ast.Dict):
                    keys[t.id] = {ast.literal_eval(k) for k in node.value.keys if k is not None}
check("__init__ 注册 SFSAM3PointTrack 双字典一致",
      "SFSAM3PointTrack" in keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFSAM3PointTrack" in keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      keys.get("NODE_CLASS_MAPPINGS", set()) == keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))


# ── 2. execute ──
node = SFSAM3PointTrack()
images = FakeTensor(np.zeros((5, 4, 8, 3), dtype=np.float32))

calls.clear()
res = node.execute(images, model="M", anchor_frame=0,
                   positive_coords='[{"x":1,"y":1}]', negative_coords=None)
check("返回单元素 tuple", isinstance(res, tuple) and len(res) == 1)
check("anchor=0 调用 detect", "detect" in calls)
check("detect 收到锚帧（1 帧）", calls["detect"]["image"].shape == (1, 4, 8, 3))
check("detect 收到点提示", calls["detect"]["positive_coords"] == '[{"x":1,"y":1}]')
check("track 收到全序列", calls["track"]["images"].shape == (5, 4, 8, 3))
check("track 无文本 conditioning", calls["track"]["conditioning"] is None)
check("anchor=0 长度不变", res[0]["n_frames"] == 5 and res[0]["packed_masks"].shape[0] == 5)

calls.clear()
res2 = node.execute(images, model="M", anchor_frame=2,
                    positive_coords='[{"x":1,"y":1}]')
check("anchor>0 detect 收到第 2 帧", calls["detect"]["image"].shape == (1, 4, 8, 3))
check("anchor>0 track 收到切片", calls["track"]["images"].shape == (3, 4, 8, 3))
check("anchor>0 前补空帧回全长", res2[0]["n_frames"] == 5 and res2[0]["packed_masks"].shape[0] == 5)
u = numpy_unpack(res2[0]["packed_masks"].numpy())
check("anchor>0 前 2 帧空", not u[:2].any())
check("anchor>0 后段来自追踪", u[2:].any())

# initial_mask 直通
calls.clear()
seed = FakeTensor(np.ones((1, 4, 8), dtype=bool))
res3 = node.execute(images, model="M", anchor_frame=0, initial_mask=seed)
check("initial_mask 跳过 detect", "detect" not in calls)
check("initial_mask 透传给 track", calls["track"]["initial_mask"] is seed)

# 2D initial_mask 升维
calls.clear()
seed2d = FakeTensor(np.ones((4, 8), dtype=bool))
node.execute(images, model="M", anchor_frame=0, initial_mask=seed2d)
check("2D initial_mask 升 3D", calls["track"]["initial_mask"].shape == (1, 4, 8))

# 既无点也无遮罩
try:
    node.execute(images, model="M", anchor_frame=0)
    check("缺提示报错", False)
except ValueError:
    check("缺提示报错", True)

# 空点提示（"[]"）也应报错
try:
    node.execute(images, model="M", anchor_frame=0, positive_coords="[]")
    check("空点提示报错", False)
except ValueError:
    check("空点提示报错", True)

# anchor 越界
try:
    node.execute(images, model="M", anchor_frame=5, positive_coords='[{"x":1,"y":1}]')
    check("anchor 越界报错", False)
except ValueError:
    check("anchor 越界报错", True)

# 非 IMAGE
try:
    node.execute(FakeTensor(np.zeros((4, 8), dtype=np.float32)), model="M",
                 anchor_frame=0, positive_coords="[]")
    check("非 IMAGE 报错", False)
except ValueError:
    check("非 IMAGE 报错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
