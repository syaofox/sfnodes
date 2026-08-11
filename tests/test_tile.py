# SFImageTile / SFImageUntile 后端逻辑测试（Node/Python 直接运行：
# python tests/test_tile.py）
# 覆盖：
#   - 纯逻辑 sf_utils.tiling（无 mock）：resolve_overlap（比例+像素、钳制、
#     单行/列清零）、tile_rects（行优先、重叠偏移、边界钳制）、tile_plan（整除、
#     不可整除丢弃）
#   - 结构：两个类、CATEGORY、RETURN_TYPES/NAMES、SF_TILE_INFO 线型、FUNCTION、
#     DESCRIPTION、INPUT_TYPES、根 __init__.py 注册键一致
#   - execute 集成（FakeTensor numpy 代理）：无重叠完全还原、带重叠完全还原、
#     单行/单列、不可整除截断、多帧输入、帧数不足报错
#   - tile_info 契约：info 字段齐全、tile 输出名 tiles、非法/缺失 info 报错
#   - 缩放自动恢复：2x 放大、非等比放大（PIL 双线性 mock F.interpolate）
# mock：torch / torch.nn.functional / nodes（numpy/PIL 本机真实可用）
import ast
import importlib.util
import os
import sys
import types

import numpy as np
from PIL import Image

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── FakeTensor：numpy 代理，够跑 tile/untile 的数值路径 ──
class FakeTensor:
    device = "cpu"

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, axis=dim))

    def repeat(self, *shape):
        return FakeTensor(np.tile(self.data, tuple(shape)))

    def permute(self, *dims):
        return FakeTensor(self.data.transpose(*dims))

    def contiguous(self):
        return self

    def __getattr__(self, name):
        attr = getattr(self.data, name)
        if callable(attr):
            def wrapped(*a, **k):
                r = attr(*a, **k)
                return FakeTensor(r) if isinstance(r, np.ndarray) else r
            return wrapped
        return attr

    def __getitem__(self, k):
        r = self.data[k]
        return FakeTensor(r) if isinstance(r, np.ndarray) else r

    def __setitem__(self, k, v):
        self.data[k] = v.data if isinstance(v, FakeTensor) else v

    def __add__(self, o):
        return FakeTensor(self.data + (o.data if isinstance(o, FakeTensor) else o))

    def __sub__(self, o):
        return FakeTensor(self.data - (o.data if isinstance(o, FakeTensor) else o))

    def __mul__(self, o):
        return FakeTensor(self.data * (o.data if isinstance(o, FakeTensor) else o))

    def __rsub__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) - self.data)

    def __radd__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) + self.data)

# ── mock torch ──
torch = types.ModuleType("torch")
torch.Tensor = FakeTensor
torch.cat = lambda seq, dim=0: FakeTensor(np.concatenate(
    [s.data if isinstance(s, FakeTensor) else s for s in seq], axis=dim))


def _tshape(s):
    return s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s


torch.zeros = lambda *s, **k: FakeTensor(np.zeros(
    tuple(int(x) for x in _tshape(s)), dtype=np.float32))
torch.ones = lambda *s, **k: FakeTensor(np.ones(
    tuple(int(x) for x in _tshape(s)), dtype=np.float32))
torch.linspace = lambda a, b, n, **k: FakeTensor(np.linspace(a, b, n))
sys.modules["torch"] = torch

# F.interpolate：PIL 双线性（本机无 torch）
def _np_bilinear(arr, size):
    b, c, h, w = arr.shape
    th, tw = size
    out = np.zeros((b, c, th, tw), dtype=arr.dtype)
    for bi in range(b):
        for ci in range(c):
            im = Image.fromarray(arr[bi, ci])
            out[bi, ci] = np.asarray(im.resize((tw, th), Image.BILINEAR), dtype=arr.dtype)
    return out

torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
torch.nn.functional.interpolate = lambda x, size=None, mode=None, align_corners=None: FakeTensor(
    _np_bilinear(x.data if isinstance(x, FakeTensor) else x, size))
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional


def F_interpolate_up(x, size):
    """模拟中间处理对 tiles 的缩放（与 untile 内部恢复共用同一 mock interpolate）。"""
    x = x.permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()

# ── mock nodes（MAX_RESOLUTION）──
nodes = types.ModuleType("nodes")
nodes.MAX_RESOLUTION = 16384
sys.modules["nodes"] = nodes

# ── 注册 sfnodes 包结构，使节点的相对导入可解析 ──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.image"); pkg3.__path__ = [os.path.join(root, "nodes", "image")]; sys.modules["sfnodes.nodes.image"] = pkg3
pkg4 = types.ModuleType("sfnodes.sf_utils"); pkg4.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg4


def load(modpath, modname):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

# ── 1. 纯逻辑 sf_utils.tiling（无 mock，直接 import）──
from sf_utils.tiling import resolve_overlap, tile_rects, tile_plan  # noqa: E402

oh, ow = resolve_overlap(10, 20, 2, 2, 0.1, 2, 1)
check("resolve_overlap 比例+像素相加", oh == 2 and ow == 4)

oh, ow = resolve_overlap(10, 20, 2, 2, 0.0, 0, 0)
check("resolve_overlap 零重叠", oh == 0 and ow == 0)

oh, ow = resolve_overlap(10, 20, 2, 2, 0.5, 100, 100)
check("resolve_overlap 钳制到块一半", oh == 5 and ow == 10)

oh, ow = resolve_overlap(10, 20, 1, 2, 0.2, 0, 4)
check("单行垂直重叠清零", oh == 0 and ow == 4)

oh, ow = resolve_overlap(10, 20, 2, 1, 0.2, 4, 0)
check("单列水平重叠清零", oh == 2 and ow == 0)

r = tile_rects(2, 2, 4, 4, 1, 1, 8, 8)
check("tile_rects 行优先 4 块", len(r) == 4)
check("tile_rects 块0 原点", r[0] == (0, 0, 5, 5))
check("tile_rects 块1 左偏", r[1] == (0, 3, 5, 8))
check("tile_rects 块2 上偏", r[2] == (3, 0, 8, 5))
check("tile_rects 块3 双向偏", r[3] == (3, 3, 8, 8))

r = tile_rects(2, 1, 4, 4, 1, 0, 7, 4)
check("tile_rects 末尾越界钳制", r[1] == (2, 0, 7, 4))

p = tile_plan(100, 80, 2, 2)
check("tile_plan 整除", p["tile_h"] == 50 and p["tile_w"] == 40
      and p["tile_height"] == 50 and p["tile_width"] == 40)
check("tile_plan rects 顺序", [r[0] for r in p["rects"]] == [0, 0, 50, 50])

p = tile_plan(10, 10, 3, 3)
check("tile_plan 不可整除丢弃（3→9）", p["tile_h"] == 3 and p["tile_w"] == 3
      and p["rects"][-1] == (6, 6, 9, 9))

p = tile_plan(8, 8, 2, 2, overlap=0.1, overlap_x=1, overlap_y=1)
check("tile_plan 带重叠块尺寸", p["tile_height"] == 5 and p["tile_width"] == 5
      and p["overlap_h"] == 1 and p["overlap_w"] == 1)

# ── 2. 节点结构 ──
mod = load(os.path.join(root, "nodes", "image", "tile.py"), "sfnodes.nodes.image.tile")

check("SFImageTile 已加载", hasattr(mod, "SFImageTile"))
check("SFImageUntile 已加载", hasattr(mod, "SFImageUntile"))
check("SFImageTileInfo 已加载", hasattr(mod, "SFImageTileInfo"))
check("SF_TILE_INFO 常量", mod.SF_TILE_INFO == "SF_TILE_INFO")
check("CATEGORY", mod.SFImageTile.CATEGORY == "sfnodes/image"
      and mod.SFImageUntile.CATEGORY == "sfnodes/image"
      and mod.SFImageTileInfo.CATEGORY == "sfnodes/image")
check("Tile DESCRIPTION", isinstance(mod.SFImageTile.DESCRIPTION, str)
      and len(mod.SFImageTile.DESCRIPTION) > 0)
check("Untile DESCRIPTION", isinstance(mod.SFImageUntile.DESCRIPTION, str)
      and len(mod.SFImageUntile.DESCRIPTION) > 0)
check("TileInfo DESCRIPTION", isinstance(mod.SFImageTileInfo.DESCRIPTION, str)
      and len(mod.SFImageTileInfo.DESCRIPTION) > 0)
check("Tile RETURN_TYPES", mod.SFImageTile.RETURN_TYPES == ("IMAGE", "SF_TILE_INFO"))
check("Tile RETURN_NAMES", mod.SFImageTile.RETURN_NAMES == ("tiles", "tile_info"))
check("Tile OUTPUT_IS_LIST（tiles 槽列表声明）", mod.SFImageTile.OUTPUT_IS_LIST == (True, False))
check("Untile RETURN_TYPES", mod.SFImageUntile.RETURN_TYPES == ("IMAGE",))
check("Tile FUNCTION", mod.SFImageTile.FUNCTION == "tile")
check("Untile FUNCTION", mod.SFImageUntile.FUNCTION == "untile")
check("TileInfo FUNCTION", mod.SFImageTileInfo.FUNCTION == "parse")
check("TileInfo RETURN_NAMES 含 block_count",
      mod.SFImageTileInfo.RETURN_NAMES == ("rows", "cols", "tile_w", "tile_h",
      "overlap_x", "overlap_y", "out_w", "out_h", "orig_w", "orig_h", "block_count",
      "full_tile_w", "full_tile_h"))
check("TileInfo RETURN_TYPES 全 INT",
      len(mod.SFImageTileInfo.RETURN_TYPES) == 13
      and all(t == "INT" for t in mod.SFImageTileInfo.RETURN_TYPES))
check("TileInfo OUTPUT_TOOLTIPS 13 项且非空",
      len(mod.SFImageTileInfo.OUTPUT_TOOLTIPS) == 13
      and all(len(x) > 0 for x in mod.SFImageTileInfo.OUTPUT_TOOLTIPS))
check("Tile OUTPUT_TOOLTIPS 2 项且非空",
      len(mod.SFImageTile.OUTPUT_TOOLTIPS) == 2
      and all(len(x) > 0 for x in mod.SFImageTile.OUTPUT_TOOLTIPS))
check("Untile OUTPUT_TOOLTIPS 1 项且非空",
      len(mod.SFImageUntile.OUTPUT_TOOLTIPS) == 1
      and len(mod.SFImageUntile.OUTPUT_TOOLTIPS[0]) > 0)

it = mod.SFImageTile.INPUT_TYPES()
check("Tile required 含 image", it["required"]["image"][0] == "IMAGE")
check("Tile rows/cols 默认 2", it["required"]["rows"][1]["default"] == 2
      and it["required"]["cols"][1]["default"] == 2)
check("Tile overlap 0~0.5", it["required"]["overlap"][1]["max"] == 0.5)
check("Tile overlap_x 上限 MAX_RESOLUTION//2",
      it["required"]["overlap_x"][1]["max"] == nodes.MAX_RESOLUTION // 2)
for k in ("image", "rows", "cols", "overlap", "overlap_x", "overlap_y", "as_list"):
    check(f"Tile 输入 {k} 有 tooltip", "tooltip" in it["required"][k][1])
check("Tile as_list 默认 False", it["required"]["as_list"][1]["default"] is False)
itu = mod.SFImageUntile.INPUT_TYPES()
check("Untile required 键集", set(itu["required"].keys()) == {"tiles", "tile_info", "resize_to_original"})
check("Untile tiles 输入线型 IMAGE", itu["required"]["tiles"][0] == "IMAGE")
check("Untile tile_info 输入线型 SF_TILE_INFO", itu["required"]["tile_info"][0] == "SF_TILE_INFO")
check("Untile resize_to_original 默认 True", itu["required"]["resize_to_original"][1]["default"] is True)
check("Untile 输入 resize_to_original 有 tooltip", "tooltip" in itu["required"]["resize_to_original"][1])
check("Untile 无 optional 输入", "optional" not in itu or len(itu.get("optional", {})) == 0)
check("Untile INPUT_IS_LIST", mod.SFImageUntile.INPUT_IS_LIST is True)
check("Untile _first 解包", mod.SFImageUntile._first([7]) == 7
      and mod.SFImageUntile._first((None,)) is None
      and mod.SFImageUntile._first(9) == 9)

# 根 __init__.py 注册键一致（AST 解析两个字典）
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
for key in ("SFImageTile", "SFImageUntile", "SFImageTileInfo"):
    check(f"__init__ 注册 {key} 双字典一致",
          key in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
          and key in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 3. execute 集成 ──

def make_img(h, w, frames=1):
    """逐像素渐变图：值 = (帧, y, x) 编码，可精确断言还原。"""
    arr = np.zeros((frames, h, w, 3), dtype=np.float32)
    for b in range(frames):
        for y in range(h):
            for x in range(w):
                arr[b, y, x, 0] = b
                arr[b, y, x, 1] = y
                arr[b, y, x, 2] = x
    return FakeTensor(arr)

# 3.0 tile_info 字段契约
img = make_img(10, 12)
t = mod.SFImageTile().tile(img, 2, 3, 0.1, 1, 2)
info = t[1]
check("info 字段齐全", set(info.keys()) == {"rows", "cols", "tile_w", "tile_h",
      "overlap_w", "overlap_h", "out_w", "out_h", "orig_w", "orig_h"})
check("info 网格与画布", (info["rows"], info["cols"]) == (2, 3)
      and (info["out_w"], info["out_h"]) == (12, 10))
check("info 原始大小", (info["orig_w"], info["orig_h"]) == (12, 10))
check("info 净块尺寸", (info["tile_w"], info["tile_h"]) == (4, 5))
check("info 实际重叠", (info["overlap_w"], info["overlap_h"]) == (1, 2))
check("info 全部为 int", all(isinstance(v, int) for v in info.values()))

# 3.0b SFImageTileInfo 解析
img = make_img(10, 12)
t = mod.SFImageTile().tile(img, 2, 3, 0.1, 1, 2)
p = mod.SFImageTileInfo().parse(t[1])
check("TileInfo 输出与 info 一致",
      p == (2, 3, 4, 5, 1, 2, 12, 10, 12, 10, 6, 5, 7))
check("TileInfo 分块数 = rows×cols", p[10] == 2 * 3)
check("TileInfo 完整块尺寸 = 净块 + 重叠", p[11] == 4 + 1 and p[12] == 5 + 2)
try:
    mod.SFImageTileInfo().parse(None)
    check("TileInfo info 为 None 抛 ValueError", False)
except ValueError:
    check("TileInfo info 为 None 抛 ValueError", True)
try:
    mod.SFImageTileInfo().parse({"rows": 2})
    check("TileInfo info 缺字段抛 ValueError", False)
except ValueError:
    check("TileInfo info 缺字段抛 ValueError", True)

# 3.1 无重叠完全还原（默认 as_list=False：t[0] 为单元素列表 [batch]，等效 batch 直连）
img = make_img(4, 4)
t = mod.SFImageTile().tile(img, 2, 2, 0, 0, 0)
check("默认形态 = [batch] 单元素列表", isinstance(t[0], list) and len(t[0]) == 1)
check("无重叠 tile 块数/尺寸", t[0][0].shape[0] == 4 and t[0][0].shape[1:3] == (2, 2))
t_list = mod.SFImageTile().tile(img, 2, 2, 0, 0, 0, as_list=True)
check("as_list 形态长度 = 块数", len(t_list[0]) == 4)
check("as_list 每项单帧含 batch 维",
      all(x.shape == (1, 2, 2, 3) for x in t_list[0]))
check("as_list 内容与 batch 逐帧一致",
      all(np.array_equal(x.numpy(), t[0][0].numpy()[i:i+1]) for i, x in enumerate(t_list[0])))
u = mod.SFImageUntile().untile(t[0], t[1])
check("无重叠 untile 完全还原（列表形态输入）", np.array_equal(u[0].numpy(), img.numpy()))
u2 = mod.SFImageUntile().untile(t[0][0], t[1])
check("无重叠 untile 完全还原（batch 输入）", np.array_equal(u2[0].numpy(), img.numpy()))

# 3.2 带重叠完全还原（重叠区羽化但内容一致）
img = make_img(8, 8)
t = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1)
check("带重叠 tile 块尺寸含重叠", t[0][0].shape[1:3] == (5, 5))
u = mod.SFImageUntile().untile(t[0][0], t[1])
check("带重叠 untile 完全还原", u[0].shape[1:3] == (8, 8)
      and np.array_equal(u[0].numpy(), img.numpy()))

# 3.3 单行/单列：重叠清零
img = make_img(6, 10)
t = mod.SFImageTile().tile(img, 1, 2, 0.1, 0, 2)
check("单行垂直重叠清零", t[1]["overlap_h"] == 0 and t[0][0].shape[1:3] == (6, 5))
u = mod.SFImageUntile().untile(t[0][0], t[1])
check("单行 untile 还原", np.array_equal(u[0].numpy(), img.numpy()))
t = mod.SFImageTile().tile(img, 2, 1, 0.1, 2, 0)
check("单列水平重叠清零", t[1]["overlap_w"] == 0 and t[0][0].shape[1:3] == (3, 10))
u = mod.SFImageUntile().untile(t[0][0], t[1])
check("单列 untile 还原", np.array_equal(u[0].numpy(), img.numpy()))

# 3.4 不可整除截断：3 行时丢弃末行，untile 还原的是截断后的图
img = make_img(10, 10)
t = mod.SFImageTile().tile(img, 3, 3, 0, 0, 0)
check("不可整除 tile 块尺寸向下取整", t[0][0].shape[1:3] == (3, 3) and t[0][0].shape[0] == 9)
u = mod.SFImageUntile().untile(t[0][0], t[1])
check("不可整除 untile 输出截断尺寸", u[0].shape[1:3] == (9, 9))
check("不可整除 untile 还原截断图", np.array_equal(u[0].numpy(), img.numpy()[:, :9, :9, :]))

# 3.5 多帧输入：2 帧 → 8 块，untile 只取前 rows×cols 帧还原单帧
# （块按"逐块帧内展开"cat，前 4 帧 = 块0帧0/块0帧1/块1帧0/块1帧1——原版行为，保持）
img = make_img(4, 4, frames=2)
t = mod.SFImageTile().tile(img, 2, 2, 0, 0, 0)
check("多帧 tile 块数 = 帧×rows×cols", t[0][0].shape[0] == 8)
u = mod.SFImageUntile().untile(t[0][0], t[1])
check("多帧 untile 输出单帧", u[0].shape[0] == 1)
tn = t[0][0].numpy()
expected = np.concatenate([
    np.concatenate([tn[0], tn[1]], axis=1),
    np.concatenate([tn[2], tn[3]], axis=1),
], axis=0)
check("多帧 untile 用前 rows×cols 帧拼图", np.array_equal(u[0].numpy()[0], expected))

# 3.6 帧数不足报错
img = make_img(4, 4)
t = mod.SFImageTile().tile(img, 2, 2, 0, 0, 0)
tiles = t[0][0][:3]
try:
    mod.SFImageUntile().untile(tiles, t[1])
    check("帧数不足抛 ValueError", False)
except ValueError:
    check("帧数不足抛 ValueError", True)

# 3.6b as_list=True 列表输入：与 batch 输入还原结果一致
img = make_img(4, 4)
t = mod.SFImageTile().tile(img, 2, 2, 0, 0, 0)
t_l = mod.SFImageTile().tile(img, 2, 2, 0, 0, 0, as_list=True)
u_list = mod.SFImageUntile().untile(list(t_l[0]), t[1])  # 逐帧列表
check("as_list 列表输入还原", np.array_equal(u_list[0].numpy(), img.numpy()))
u_wrap = mod.SFImageUntile().untile([t[0][0]], t[1])  # batch 包装形态
check("单元素列表（batch 包装）还原", np.array_equal(u_wrap[0].numpy(), img.numpy()))
u_batch = mod.SFImageUntile().untile(t[0][0], t[1])  # 直传 tensor
check("直传批次还原", np.array_equal(u_batch[0].numpy(), img.numpy()))

# 3.6c 列表内尺寸不一致：逐块 resize 后仍还原（按原始块尺寸统一）
img = make_img(8, 8)
t = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1)
t_l = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1, as_list=True)
mixed = list(t_l[0])
mixed[1] = F_interpolate_up(mixed[1], (10, 10))  # 第 2 块放大到 10x10
u_mix = mod.SFImageUntile().untile(mixed, t[1])
check("混合尺寸列表还原", u_mix[0].shape[1:3] == (8, 8)
      and np.allclose(u_mix[0].numpy(), img.numpy(), atol=1.0))

# 3.6d tiles 未连接抛 ValueError
try:
    mod.SFImageUntile().untile(None, t[1])
    check("tiles 未连接抛 ValueError", False)
except ValueError:
    check("tiles 未连接抛 ValueError", True)

# 3.7 info 缺失/非法报错
img = make_img(4, 4)
t = mod.SFImageTile().tile(img, 2, 2, 0, 0, 0)
try:
    mod.SFImageUntile().untile(t[0][0], None)
    check("info 为 None 抛 ValueError", False)
except ValueError:
    check("info 为 None 抛 ValueError", True)
try:
    mod.SFImageUntile().untile(t[0][0], {"rows": 2})
    check("info 缺字段抛 ValueError", False)
except ValueError:
    check("info 缺字段抛 ValueError", True)

# 3.8 缩放自动恢复：块被 2x 放大后仍还原为原始画布尺寸
img = make_img(6, 8)
t = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1)
check("缩放前块尺寸 4x5", t[0][0].shape[1:3] == (4, 5))
big = F_interpolate_up(t[0][0], (8, 10))  # 2x 放大
u = mod.SFImageUntile().untile(big, t[1])
check("2x 放大后 untile 输出原始画布", u[0].shape[1:3] == (6, 8))
check("2x 放大后内容近似还原",
      np.allclose(u[0].numpy(), img.numpy(), atol=1.0))

# 3.9 缩放自动恢复：非等比缩放（宽 2x 高 1.5x）
img = make_img(6, 8)
t = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1)
wide = F_interpolate_up(t[0][0], (6, 15))  # 高 6、宽 15
u = mod.SFImageUntile().untile(wide, t[1])
check("非等比缩放 untile 输出原始画布", u[0].shape[1:3] == (6, 8))
check("非等比缩放内容近似还原",
      np.allclose(u[0].numpy(), img.numpy(), atol=1.0))

# 3.10 缩放自动恢复：缩小 0.5x 后还原
img = make_img(8, 8)
t = mod.SFImageTile().tile(img, 2, 2, 0.0, 0, 0)
small = F_interpolate_up(t[0][0], (2, 2))
u = mod.SFImageUntile().untile(small, t[1])
check("缩小后 untile 输出原始画布", u[0].shape[1:3] == (8, 8))

# 3.11 resize_to_original=False：不缩放，按当前块尺寸合并
# 3.11a 无重叠：2x 放大块直接拼接出放大画布（内容精确相等）
img = make_img(4, 4)
t = mod.SFImageTile().tile(img, 2, 2, 0, 0, 0)  # 2x2 块
big = F_interpolate_up(t[0][0], (4, 4))  # 2x 放大
u = mod.SFImageUntile().untile(big, t[1], False)
check("关闭缩放输出放大画布", u[0].shape[1:3] == (8, 8))
bn = big.numpy()
expected = np.concatenate([
    np.concatenate([bn[0], bn[1]], axis=1),
    np.concatenate([bn[2], bn[3]], axis=1),
], axis=0)
check("关闭缩放内容 = 放大块直接拼接", np.array_equal(u[0].numpy()[0], expected))
u2 = mod.SFImageUntile().untile(big, t[1], True)
check("开启缩放输出原始画布", u2[0].shape[1:3] == (4, 4))

# 3.11b 带重叠：重叠随块按比例缩放（块 2x → 重叠 2x），画布 = rows×净块
# cell=6x8, ov=2x2, 块尺寸 8x10；网格：块0 [0:8,0:10)、块1 [0:8,6:16)、
# 块2 [4:12,0:10)、块3 [4:12,6:16)
img = make_img(6, 8)
t = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1)  # tile 3x4 + 重叠 1 → 块 4x5
big = F_interpolate_up(t[0][0], (8, 10))  # 2x → 8x10
u = mod.SFImageUntile().untile(big, t[1], False)
check("关闭缩放带重叠输出放大画布", u[0].shape[1:3] == (12, 16))
out = u[0].numpy()[0]
bn = big.numpy()
check("块0 左上纯保留区定位", np.array_equal(out[0:4, 0:6], bn[0][0:4, 0:6]))
check("块2 覆盖块0 底部（重叠区同源覆盖）", np.array_equal(out[6:8, 0:6], bn[2][2:4, 0:6]))
check("顶部羽化带同源（mask=0 保留旧值）", np.array_equal(out[4, 0:6], bn[0][4, 0:6]))
check("顶部羽化带同源（mask=1 新值）", np.array_equal(out[5, 0:6], bn[2][1, 0:6]))
check("左侧羽化带同源（mask=0 保留旧值）", np.array_equal(out[0:3, 6], bn[0][0:3, 6]))
check("左侧羽化带同源（mask=1 新值）", np.array_equal(out[0:3, 7], bn[1][0:3, 1]))
check("块3 覆盖右上块下带（定位正确）", np.array_equal(out[6:8, 8:16], bn[3][2:4, 2:10]))
check("右下块非羽化区定位（最强定位断言）",
      np.array_equal(out[6:12, 8:16], bn[3][2:8, 2:10]))

# 3.11b2 带重叠缩小 0.5x：重叠随比例缩小，画布 = rows×净块
# cur=2x3, scale_h=0.5 → cell_h=round(3×0.5)=2, cell_w=round(4×0.6)=2 → 画布 4x4
img = make_img(6, 8)
t = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1)
small = F_interpolate_up(t[0][0], (2, 3))  # 0.5x → 2x3
u = mod.SFImageUntile().untile(small, t[1], False)
check("关闭缩放带重叠缩小画布自洽", u[0].shape[1:3] == (4, 4))

# 3.11c 关闭缩放 + 混合尺寸列表抛 ValueError
img = make_img(8, 8)
t = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1)
t_l = mod.SFImageTile().tile(img, 2, 2, 0.0, 1, 1, as_list=True)
mixed = list(t_l[0])
mixed[1] = F_interpolate_up(mixed[1], (10, 10))
try:
    mod.SFImageUntile().untile(mixed, t[1], False)
    check("关闭缩放混合尺寸列表抛 ValueError", False)
except ValueError:
    check("关闭缩放混合尺寸列表抛 ValueError", True)

# ── 4. 随机属性：任意 rows/cols/overlap 组合，tile→untile 恒还原截断后的原图
# （羽化重叠带两侧同源，混合后仍等于原值；用 allclose——mask=1/3 等无限小数
#   的浮点混合有 ~1e-7 舍入，非节点错误）
rng = np.random.default_rng(20260811)
prop_fail = 0
for case in range(30):
    h = int(rng.integers(5, 40))
    w = int(rng.integers(5, 40))
    rows = int(rng.integers(1, 5))
    cols = int(rng.integers(1, 5))
    if h // rows == 0 or w // cols == 0:
        continue  # 块尺寸为 0 是无效参数（原版同样会崩）
    overlap = float(rng.choice([0.0, 0.1, 0.25, 0.5]))
    ox = int(rng.integers(0, 4))
    oy = int(rng.integers(0, 4))
    img = make_img(h, w)
    t = mod.SFImageTile().tile(img, rows, cols, overlap, ox, oy)
    u = mod.SFImageUntile().untile(t[0][0], t[1])
    exp_h = (h // rows) * rows
    exp_w = (w // cols) * cols
    ok = (u[0].shape[1:3] == (exp_h, exp_w)
          and np.allclose(u[0].numpy(), img.numpy()[:, :exp_h, :exp_w, :], atol=1e-6))
    if not ok:
        prop_fail += 1
        print(f"  FAIL case {case}: h={h} w={w} rows={rows} cols={cols} "
              f"overlap={overlap} ox={ox} oy={oy}")
check(f"随机属性 30 例全还原（跳过 {30 - (30 - prop_fail) - prop_fail} 例无效参数）", prop_fail == 0)

print()
if failures:
    print(f"共 {len(failures)} 项失败: {failures}")
    sys.exit(1)
print("全部通过")
