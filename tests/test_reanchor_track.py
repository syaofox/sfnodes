# SFSAM3ReanchorTrack 后端逻辑测试（Node/Python 直接运行：python tests/test_reanchor_track.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（images/model/anchor_frames required，clip/prompts/conditioning/
#     initial_mask/detection_threshold/max_objects/detect_interval optional）、
#     根 __init__.py 注册键一致
#   - execute：锚帧解析（逗号分隔/排序去重/空串回退 [0]/越界与非法报错）；
#     每段切片调用 SAM3_VideoTrack；prompts 逐行编码（缺 clip 报错）；
#     空行回退 conditioning；initial_mask 仅首锚（2D 升维）；
#     无提示/条件报错；空段输出补零；输出全长单身份 track_data
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

    def any(self):
        return bool(self.data.any())

    def numpy(self):
        return self.data

    def __or__(self, other):
        return FakeTensor(self.data | (other.data if isinstance(other, FakeTensor) else other))

    def __getitem__(self, key):
        return FakeTensor(self.data[key])


fake_torch = types.SimpleNamespace(
    bool=np.bool_,
    zeros=lambda shape, dtype=None, device=None: FakeTensor(
        np.zeros(shape, dtype=np.uint8 if dtype is None else dtype), device=device or "cpu"),
    cat=lambda tensors, dim=0: FakeTensor(np.concatenate([t.data for t in tensors], axis=dim)),
)
sys.modules["torch"] = fake_torch


# ── fake 核心 SAM3 节点 / CLIP ──
class FakeOut:
    def __init__(self, *result):
        self.result = result


calls = {"track": [], "tokens": []}
empty_segments = set()


class FakeSAM3VideoTrack:
    @classmethod
    def execute(cls, **kwargs):
        calls["track"].append(kwargs)
        index = len(calls["track"]) - 1
        n = kwargs["images"].shape[0]
        if index in empty_segments:
            return FakeOut({"packed_masks": None, "n_frames": n, "orig_size": (4, 8), "scores": []})
        masks = np.zeros((n, 2, 4, 8), dtype=bool)
        masks[:, 0, :, :2] = True
        masks[:, 1, :, 4:6] = True
        return FakeOut({"packed_masks": FakeTensor(numpy_pack(masks)), "n_frames": n,
                        "orig_size": (4, 8), "scores": [1.0, 0.5]})


class FakeClip:
    def tokenize(self, text):
        calls["tokens"].append(text)
        return {"fake": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [("COND:" + tokens["fake"], {})]


ce = types.ModuleType("comfy_extras")
ce.__path__ = []
sys.modules["comfy_extras"] = ce
ce_sam3 = types.ModuleType("comfy_extras.nodes_sam3")
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


mod = load(os.path.join(root, "nodes", "video", "sam3_reanchor_track.py"),
           "sfnodes.nodes.video.sam3_reanchor_track")
SFSAM3ReanchorTrack = mod.SFSAM3ReanchorTrack

# ── 1. 结构 ──
check("CATEGORY", SFSAM3ReanchorTrack.CATEGORY == "sfnodes/video")
check("FUNCTION", SFSAM3ReanchorTrack.FUNCTION == "execute")
check("RETURN_TYPES", SFSAM3ReanchorTrack.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("RETURN_NAMES", SFSAM3ReanchorTrack.RETURN_NAMES == ("track_data",))
check("DESCRIPTION 存在", isinstance(getattr(SFSAM3ReanchorTrack, "DESCRIPTION", None), str)
      and SFSAM3ReanchorTrack.DESCRIPTION.strip() != "")

schema = SFSAM3ReanchorTrack.INPUT_TYPES()
check("images required IMAGE", schema["required"]["images"][0] == "IMAGE")
check("model required MODEL", schema["required"]["model"][0] == "MODEL")
check("anchor_frames STRING default 0", schema["required"]["anchor_frames"][0] == "STRING"
      and schema["required"]["anchor_frames"][1].get("default") == "0")
check("prompts optional multiline", schema["optional"]["prompts"][0] == "STRING"
      and schema["optional"]["prompts"][1].get("multiline") is True)
check("initial_mask optional MASK", schema["optional"]["initial_mask"][0] == "MASK")
check("max_objects default 1", schema["optional"]["max_objects"][0] == "INT"
      and schema["optional"]["max_objects"][1].get("default") == 1)
check("无 initial_mask_always（已由批语义取代）", "initial_mask_always" not in schema["optional"])

with open(os.path.join(root, "__init__.py"), encoding="utf-8") as f:
    tree = ast.parse(f.read())
keys = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
                if isinstance(node.value, ast.Dict):
                    keys[t.id] = {ast.literal_eval(k) for k in node.value.keys if k is not None}
check("__init__ 注册 SFSAM3ReanchorTrack 双字典一致",
      "SFSAM3ReanchorTrack" in keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFSAM3ReanchorTrack" in keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      keys.get("NODE_CLASS_MAPPINGS", set()) == keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))


# ── 2. 锚帧/提示词解析 ──
check("解析：空串回退 [0]", mod._parse_anchors("", 5) == [0] and mod._parse_anchors(None, 5) == [0])
check("解析：逗号/空格/分号 + 排序去重", mod._parse_anchors("5, 2;5\n3", 8) == [2, 3, 5])
check("解析：提示词逐行保留空行", mod._parse_prompts("person\n\nwoman") == ["person", "", "woman"])
check("解析：空提示词", mod._parse_prompts("") == [] and mod._parse_prompts(None) == [])


# ── 3. execute ──
node = SFSAM3ReanchorTrack()
images = FakeTensor(np.zeros((8, 4, 8, 3), dtype=np.float32))

calls["track"].clear()
calls["tokens"].clear()
empty_segments.clear()
res = node.execute(images, model="M", anchor_frames="2,5", clip=FakeClip(), prompts="person\nwoman")
check("返回单元素 tuple", isinstance(res, tuple) and len(res) == 1)
check("两段各调一次 track", len(calls["track"]) == 2)
check("段0 切片 [2:5]", calls["track"][0]["images"].shape == (3, 4, 8, 3))
check("段1 切片 [5:8]", calls["track"][1]["images"].shape == (3, 4, 8, 3))
check("逐行编码提示词", calls["tokens"] == ["person", "woman"])
check("段0 conditioning=person", calls["track"][0]["conditioning"] == [("COND:person", {})])
check("段1 conditioning=woman", calls["track"][1]["conditioning"] == [("COND:woman", {})])
check("无 initial_mask 时传 None", calls["track"][0]["initial_mask"] is None)
check("检测参数透传", calls["track"][0]["detection_threshold"] == 0.5
      and calls["track"][0]["max_objects"] == 1 and calls["track"][0]["detect_interval"] == 1)

out = res[0]
u = numpy_unpack(out["packed_masks"].numpy())
check("输出全长单身份", out["n_frames"] == 8 and out["packed_masks"].shape == (8, 1, 4, 1))
check("scores=[1.0]", out["scores"] == [1.0])
check("orig_size 继承", out["orig_size"] == (4, 8))
check("首锚前补空帧", not u[:2].any())
check("段内多对象并集", u[2, 0, :, :2].all() and u[2, 0, :, 4:6].all())
check("段1 内容", u[7, 0, :, :2].all() and u[7, 0, :, 4:6].all())

# 锚帧排序 + initial_mask 仅首锚 + 2D 升维
calls["track"].clear()
seed2d = FakeTensor(np.ones((4, 8), dtype=bool))
res2 = node.execute(images, model="M", anchor_frames="5,2,5", initial_mask=seed2d,
                    conditioning="COND_OBJ")
check("锚帧排序去重后两段", len(calls["track"]) == 2
      and calls["track"][0]["images"].shape == (3, 4, 8, 3)
      and calls["track"][1]["images"].shape == (3, 4, 8, 3))
check("2D 遮罩升 3D", calls["track"][0]["initial_mask"].shape == (1, 4, 8))
check("initial_mask 仅首锚", calls["track"][1]["initial_mask"] is None)
check("非首锚回退 conditioning", calls["track"][1]["conditioning"] == "COND_OBJ")
check("排序后输出全长", res2[0]["n_frames"] == 8)

# 单锚仅遮罩无提示也可跑
calls["track"].clear()
res_seed_only = node.execute(images, model="M", anchor_frames="0", initial_mask=seed2d)
check("单锚仅遮罩可跑", res_seed_only[0]["n_frames"] == 8
      and calls["track"][0]["conditioning"] is None)

# initial_mask 批语义：第 i 张按顺序对应第 i 个锚段
calls["track"].clear()
calls["tokens"].clear()
batch3 = FakeTensor(np.zeros((3, 4, 8), dtype=bool))
batch3.data[0, :, :2] = True
batch3.data[1, :, 2:4] = True
batch3.data[2, :, 4:6] = True
res_batch = node.execute(images, model="M", anchor_frames="1,3,6", clip=FakeClip(),
                         prompts="a\nb\nc", initial_mask=batch3)
check("批：三段各取对应一张", len(calls["track"]) == 3
      and all(calls["track"][i]["initial_mask"].shape == (1, 4, 8) for i in range(3)))
check("批：第 i 段种子内容对应第 i 张",
      np.array_equal(calls["track"][0]["initial_mask"].numpy(), batch3.numpy()[0:1])
      and np.array_equal(calls["track"][1]["initial_mask"].numpy(), batch3.numpy()[1:2])
      and np.array_equal(calls["track"][2]["initial_mask"].numpy(), batch3.numpy()[2:3]))
check("批：提示词仍逐行编码", calls["tokens"] == ["a", "b", "c"])
check("批：输出全长", res_batch[0]["n_frames"] == 8)

# 遮罩不足：剩余段回退 conditioning
calls["track"].clear()
res_few = node.execute(images, model="M", anchor_frames="0,4,6", initial_mask=batch3[:2],
                       conditioning="COND_OBJ")
check("批：不足的段回退文本", calls["track"][0]["initial_mask"] is not None
      and calls["track"][1]["initial_mask"] is not None
      and calls["track"][2]["initial_mask"] is None
      and calls["track"][2]["conditioning"] == "COND_OBJ")
check("批：不足仍输出全长", res_few[0]["n_frames"] == 8)

# 遮罩多余：丢弃且不影响对应段
calls["track"].clear()
res_extra = node.execute(images, model="M", anchor_frames="0,4", initial_mask=batch3,
                         conditioning="COND_OBJ")
check("批：多余丢弃且前两段取到", len(calls["track"]) == 2
      and calls["track"][0]["initial_mask"] is not None
      and calls["track"][1]["initial_mask"] is not None
      and res_extra[0]["n_frames"] == 8)

# 全零占位：该段视为未提供（回退文本），其余段照常
calls["track"].clear()
batch_zero = FakeTensor(np.zeros((3, 4, 8), dtype=bool))
batch_zero.data[0, :, :2] = True
batch_zero.data[2, :, 4:6] = True
node.execute(images, model="M", anchor_frames="0,4,6", initial_mask=batch_zero,
             conditioning="COND_OBJ")
check("批：全零段回退文本", calls["track"][0]["initial_mask"] is not None
      and calls["track"][1]["initial_mask"] is None
      and calls["track"][2]["initial_mask"] is not None)

# 4D 遮罩报错
try:
    node.execute(images, model="M", anchor_frames="0",
                 initial_mask=FakeTensor(np.zeros((2, 1, 4, 8), dtype=bool)), conditioning="C")
    check("4D 遮罩报错", False)
except ValueError:
    check("4D 遮罩报错", True)

# 空行回退 conditioning
calls["track"].clear()
calls["tokens"].clear()
res3 = node.execute(images, model="M", anchor_frames="0,4", clip=FakeClip(),
                    prompts="person\n", conditioning="COND_OBJ")
check("首行编码", calls["track"][0]["conditioning"] == [("COND:person", {})])
check("空行回退 conditioning", calls["track"][1]["conditioning"] == "COND_OBJ")
check("行数不足仍全长", res3[0]["n_frames"] == 8)

# 空段（第二段无检测）→ 该段补零
calls["track"].clear()
empty_segments.add(1)
res4 = node.execute(images, model="M", anchor_frames="0,4", conditioning="COND_OBJ")
u4 = numpy_unpack(res4[0]["packed_masks"].numpy())
check("空段区间补零", not u4[4:].any() and u4[:4].any())
empty_segments.clear()

# 错误路径
for _name, _kwargs in [
    ("提示词缺 clip 报错", dict(anchor_frames="0", prompts="person")),
    ("无提示/条件报错", dict(anchor_frames="0")),
    ("锚帧越界报错", dict(anchor_frames="9", conditioning="C")),
    ("非法锚帧报错", dict(anchor_frames="abc", conditioning="C")),
]:
    try:
        node.execute(images, model="M", **_kwargs)
        check(_name, False)
    except ValueError:
        check(_name, True)

try:
    node.execute(FakeTensor(np.zeros((4, 8), dtype=np.float32)), model="M",
                 anchor_frames="0", conditioning="C")
    check("非 IMAGE 报错", False)
except ValueError:
    check("非 IMAGE 报错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
