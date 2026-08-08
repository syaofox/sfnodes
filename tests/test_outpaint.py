# SFImageOutpaint / SFImageOutpaintStitch 后端逻辑测试（Node/Python 直接运行：
# python tests/test_outpaint.py）
# 覆盖：
#   - 结构：两个类、CATEGORY、RETURN_TYPES/NAMES、隐藏输入 SFOutpaintState、
#     DESCRIPTION、SF_OUTPAINT_INFO 线型、根 __init__.py 注册键一致
#   - 纯函数：_parse_state（类型容错/夹紧/Infinity 防御）、_parse_ratio
#     （inf/nan/非正拒绝）、_pads_for_ratio（三方向 + anchor + round-half-up
#     边界）、_fit_pad（OOM 防御比例分配）、_tensor_to_pils（通道防御）
#   - 数值：outpaint 全流程（ratio/sides/limit 尺寸、填充色、info dict、ui 存档）、
#     stitch（透传兜底、feather 贴回、mask、批次配对、resize 恢复、color_match）
# mock：torch（FakeTensor numpy 代理 + F.interpolate PIL 双线性）/ folder_paths
# （numpy/PIL 本机真实可用）
import ast
import importlib.util
import json
import os
import sys
import tempfile
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

# ── FakeTensor：numpy 代理，够跑 outpaint/stitch 的数值路径 ──
class FakeTensor:
    device = "cpu"

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def numpy(self):
        return self.data

    def __array__(self, dtype=None):
        return self.data.astype(dtype) if dtype else self.data

    def clone(self):
        return FakeTensor(self.data.copy())

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def contiguous(self):
        return self

    def dim(self):
        return self.data.ndim

    def numel(self):
        return self.data.size

    def view(self, *shape):
        return FakeTensor(self.data.reshape(shape[0] if len(shape) == 1 else shape))

    def new_zeros(self, shape):
        return FakeTensor(np.zeros(tuple(shape), dtype=np.float32))

    def repeat(self, *shape):
        return FakeTensor(np.tile(self.data, tuple(shape)))

    def clamp(self, lo=None, hi=None, min=None, max=None):
        a, b = (lo if lo is not None else min), (hi if hi is not None else max)
        return FakeTensor(np.clip(self.data, a, b))

    def to(self, device=None, dtype=None):
        return self

    def permute(self, *dims):
        return FakeTensor(self.data.transpose(*dims))

    def narrow(self, dim, start, length):
        idx = [slice(None)] * self.data.ndim
        idx[dim] = slice(start, start + length)
        return FakeTensor(self.data[tuple(idx)])

    def index_select(self, dim, index):
        return FakeTensor(np.take(self.data, index.data.astype(int), axis=dim))

    def __getattr__(self, name):
        attr = getattr(self.data, name)
        if callable(attr):
            def wrapped(*a, **k):
                # numpy 用 axis/keepdims，torch 用 dim/keepdim
                k = {
                    ("axis" if kk == "dim" else "keepdims" if kk == "keepdim" else kk): vv
                    for kk, vv in k.items()
                }
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

    def __truediv__(self, o):
        return FakeTensor(self.data / (o.data if isinstance(o, FakeTensor) else o))

    def __rsub__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) - self.data)

    def __radd__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) + self.data)

    def __rmul__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) * self.data)

    def __rtruediv__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) / self.data)

    def __neg__(self):
        return FakeTensor(-self.data)

    def __gt__(self, o):
        return self.data > (o.data if isinstance(o, FakeTensor) else o)

    def __ge__(self, o):
        return self.data >= (o.data if isinstance(o, FakeTensor) else o)

    def __lt__(self, o):
        return self.data < (o.data if isinstance(o, FakeTensor) else o)

    def __le__(self, o):
        return self.data <= (o.data if isinstance(o, FakeTensor) else o)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def __eq__(self, o):
        return self.data == (o.data if isinstance(o, FakeTensor) else o)

# ── mock torch ──
torch = types.ModuleType("torch")
torch.Tensor = FakeTensor
torch.from_numpy = lambda a: FakeTensor(a)

def _tshape(s):
    return s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s

torch.zeros = lambda *s, **k: FakeTensor(np.zeros(tuple(int(x) for x in _tshape(s)), dtype=np.float32))
torch.ones = lambda *s, **k: FakeTensor(np.ones(tuple(int(x) for x in _tshape(s)), dtype=np.float32))
torch.full = lambda shape, fill, **k: FakeTensor(np.full(tuple(int(x) for x in shape), fill, dtype=np.float32))
torch.arange = lambda n, **k: FakeTensor(np.arange(n))
torch.float32 = "float32"
torch.minimum = lambda a, b: FakeTensor(np.minimum(
    a.data if isinstance(a, FakeTensor) else a, b.data if isinstance(b, FakeTensor) else b))
torch.stack = lambda seq, dim=0: FakeTensor(np.stack(
    [s.data if isinstance(s, FakeTensor) else s for s in seq], axis=dim))
torch.cat = lambda seq, dim=0: FakeTensor(np.concatenate(
    [s.data if isinstance(s, FakeTensor) else s for s in seq], axis=dim))
torch.cumsum = lambda x, dim: FakeTensor(np.cumsum(x.data, axis=dim))
torch.zeros_like = lambda x: FakeTensor(np.zeros_like(x.data))
torch.index_select = lambda x, dim, index: FakeTensor(
    np.take(x.data, index.data.astype(int), axis=dim))
torch.where = lambda cond, a, b: FakeTensor(np.where(
    cond.data if isinstance(cond, FakeTensor) else cond,
    a.data if isinstance(a, FakeTensor) else a,
    b.data if isinstance(b, FakeTensor) else b))
torch.clamp = lambda x, **k: FakeTensor(np.clip(
    x.data,
    k.get("a_min", k.get("min", -np.inf)),
    k.get("a_max", k.get("max", np.inf))))

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
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional

# ── mock folder_paths ──
tmp_in = tempfile.mkdtemp(prefix="sf_outpaint_input_")
tmp_temp = os.path.join(tmp_in, "temp")
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_input_directory = lambda: tmp_in
folder_paths.get_temp_directory = lambda: tmp_temp
sys.modules["folder_paths"] = folder_paths

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


mod = load(os.path.join(root, "nodes", "image", "outpaint.py"), "sfnodes.nodes.image.outpaint")

# ── 结构断言 ──
check("SFImageOutpaint 已加载", hasattr(mod, "SFImageOutpaint"))
check("SFImageOutpaintStitch 已加载", hasattr(mod, "SFImageOutpaintStitch"))
check("CATEGORY", mod.SFImageOutpaint.CATEGORY == "sfnodes/image"
      and mod.SFImageOutpaintStitch.CATEGORY == "sfnodes/image")
check("Outpaint DESCRIPTION", isinstance(mod.SFImageOutpaint.DESCRIPTION, str)
      and len(mod.SFImageOutpaint.DESCRIPTION) > 0)
check("Stitch DESCRIPTION", isinstance(mod.SFImageOutpaintStitch.DESCRIPTION, str)
      and len(mod.SFImageOutpaintStitch.DESCRIPTION) > 0)
check("SF_OUTPAINT_INFO 常量", mod.SF_OUTPAINT_INFO == "SF_OUTPAINT_INFO")
check("Outpaint RETURN_TYPES",
      mod.SFImageOutpaint.RETURN_TYPES == ("IMAGE", "INT", "INT", "SF_OUTPAINT_INFO"))
check("Stitch RETURN_TYPES", mod.SFImageOutpaintStitch.RETURN_TYPES == ("IMAGE", "MASK"))
check("Outpaint FUNCTION", mod.SFImageOutpaint.FUNCTION == "outpaint")
check("Stitch FUNCTION", mod.SFImageOutpaintStitch.FUNCTION == "stitch")

it = mod.SFImageOutpaint.INPUT_TYPES()
check("Outpaint required 含 image", it["required"]["image"][0] == "IMAGE")
check("Outpaint hidden 声明 SFOutpaintState", it["hidden"]["SFOutpaintState"][0] == "STRING")
its = mod.SFImageOutpaintStitch.INPUT_TYPES()
check("Stitch required 含 image", its["required"]["image"][0] == "IMAGE")
check("Stitch optional 含 outpaint_info 线型", its["optional"]["outpaint_info"][0] == "SF_OUTPAINT_INFO")
check("Stitch feather 默认 64", its["optional"]["feather"][1]["default"] == 64)
check("Stitch color_match 默认 100", its["optional"]["color_match"][1]["default"] == 100)

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
check("__init__ 两个字典含 SFImageOutpaint",
      "SFImageOutpaint" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFImageOutpaint" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 两个字典含 SFImageOutpaintStitch",
      "SFImageOutpaintStitch" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFImageOutpaintStitch" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── _parse_state 容错 ──
ps = mod._parse_state
st = ps(None)
check("_parse_state None -> 默认", st["mode"] == "ratio" and st["color"] == "#808080")
st = ps("not json")
check("_parse_state 坏 JSON -> 默认", st["mode"] == "ratio")
st = ps('{"mode": "bogus"}')
check("_parse_state 非法 mode 回退", st["mode"] == "ratio")
st = ps('{"mode": "sides", "top": 99999, "left": -5}')
check("_parse_state 单边夹紧到 _MAX_PAD", st["top"] == 8192 and st["left"] == 0)
st = ps('{"top": Infinity}')
check("_parse_state Infinity -> 0（OverflowError 防御）", st["top"] == 0)
st = ps('{"limit": 100, "limit2": 0}')
check("_parse_state limit 超上限 -> 0", st["limit"] == 0)
st = ps('{"limit": 2.5}')
check("_parse_state limit 有限值保留", st["limit"] == 2.5)
st = ps('{"color": "#ff0000"}')
check("_parse_state 合法颜色", st["color"] == "#ff0000")
st = ps('{"color": "red"}')
check("_parse_state 非法颜色回退", st["color"] == "#808080")
st = ps('{"color": "#12345"}')
check("_parse_state 长度不足回退", st["color"] == "#808080")
st = ps('{"snap": 16}')
check("_parse_state snap 合法", st["snap"] == 16)
st = ps('{"snap": 7}')
check("_parse_state snap 非法 -> 0", st["snap"] == 0)
st = ps('{"anchor": "bottom", "ratio": "9:16"}')
check("_parse_state anchor/ratio 保留", st["anchor"] == "bottom" and st["ratio"] == "9:16")

# ── _parse_ratio ──
pr = mod._parse_ratio
check("_parse_ratio 正常", pr("3:2") == (3.0, 2.0))
check("_parse_ratio 空格", pr(" 4 : 3 ") == (4.0, 3.0))
check("_parse_ratio inf 拒绝", pr("inf:2") is None)
check("_parse_ratio nan 拒绝", pr("nan:1") is None)
check("_parse_ratio 零宽拒绝", pr("3:0") is None)
check("_parse_ratio 零高拒绝", pr("0:2") is None)
check("_parse_ratio 多冒号拒绝", pr("3:2:5") is None)
check("_parse_ratio 无冒号拒绝", pr("32") is None)
check("_parse_ratio 非字符串拒绝", pr(3) is None)

# ── _pads_for_ratio ──
pfr = mod._pads_for_ratio
# 100x50（宽>高），"4:3"（1.333 < 2）-> 纵向增长 add = round(100/1.3333)-50 = 25
check("ratio 更高 top", pfr(100, 50, "4:3", "top") == (25, 0, 0, 0))
check("ratio 更高 bottom", pfr(100, 50, "4:3", "bottom") == (0, 25, 0, 0))
check("ratio 更高 middle 平分", pfr(100, 50, "4:3", "middle") == (12, 13, 0, 0))
# 100x50，"21:9"（2.333 > 2）-> 横向增长 add = round(50*2.3333)-100 = 17
check("ratio 更宽 left", pfr(100, 50, "21:9", "left") == (0, 0, 17, 0))
check("ratio 更宽 right", pfr(100, 50, "21:9", "right") == (0, 0, 0, 17))
check("ratio 更宽 centre 平分", pfr(100, 50, "21:9", "centre") == (0, 0, 8, 9))
# 跨轴 anchor 别名（存储遗留）
check("ratio 跨轴 top->left 语义", pfr(100, 50, "21:9", "top") == (0, 0, 17, 0))
# 已匹配比例 -> 零 pad
check("ratio 已匹配 -> 零", pfr(100, 100, "1:1", "centre") == (0, 0, 0, 0))
# round-half-up 边界：999 高，"3:2" -> round(999*1.5)=round(1498.5)=1499, add=500
check("ratio round-half-up 边界", pfr(999, 999, "3:2", "right")[3] == 500)
# 非法比例 -> 零
check("ratio 非法比例 -> 零", pfr(100, 50, "oops", "centre") == (0, 0, 0, 0))

# ── _fit_pad（_MAX_DIM = 16384）──
fp = mod._fit_pad
check("_fit_pad 已满足不变", fp(100, 100, 1000) == (100, 100))
check("_fit_pad 按比例收缩", fp(9000, 9000, 1000) == (7692, 7692))
check("_fit_pad 单边收缩", fp(16385, 0, 100) == (16284, 0))
check("_fit_pad 零 pad", fp(0, 0, 100) == (0, 0))
check("_fit_pad 收缩后不超上限", sum(fp(9000, 9000, 1000)) + 1000 <= 16384)

# ── _tensor_to_pils 通道防御 ──
ttp = mod._tensor_to_pils
arr3 = np.full((1, 2, 2, 3), 0.5, dtype=np.float32)
pils = ttp(FakeTensor(arr3))
check("_tensor_to_pils 3 通道", len(pils) == 1 and pils[0].size == (2, 2) and pils[0].mode == "RGB")
pils1 = ttp(FakeTensor(np.full((1, 2, 2, 1), 0.5, dtype=np.float32)))
check("_tensor_to_pils 1 通道 -> RGB", pils1[0].mode == "RGB" and pils1[0].size == (2, 2))
pils2 = ttp(FakeTensor(np.full((1, 2, 2, 2), 0.5, dtype=np.float32)))
check("_tensor_to_pils 2 通道 -> 灰", pils2[0].mode == "RGB")
pils5 = ttp(FakeTensor(np.full((1, 2, 2, 5), 0.5, dtype=np.float32)))
check("_tensor_to_pils 5 通道 -> 前 3 通道", pils5[0].mode == "RGB")

# ── outpaint 全流程 ──
Outpaint = mod.SFImageOutpaint()
# ratio 模式：32x64 源，"2:1"（2 > 0.5）-> 横向增长 add = round(64*2)-32 = 96，anchor right
src_ratio = FakeTensor(np.full((1, 64, 32, 3), 0.5, dtype=np.float32))
r = Outpaint.outpaint(src_ratio, json.dumps({"mode": "ratio", "ratio": "2:1", "anchor": "right"}))
out, w, h, info = r["result"]
check("outpaint ratio 尺寸", (w, h) == (128, 64) and out.data.shape == (1, 64, 128, 3))
check("outpaint ratio info pad", info["left"] == 0 and info["top"] == 0 and info["right"] == 96 and info["bottom"] == 0)
check("outpaint info 原始尺寸", info["orig_w"] == 32 and info["orig_h"] == 64)
check("outpaint info 画布尺寸", info["canvas_w"] == 128 and info["canvas_h"] == 64)
check("outpaint info 携带原始张量", info["original"] is src_ratio)
check("outpaint ui 存档键", "sf_outpaint_base" in r["ui"])
stash = r["ui"]["sf_outpaint_base"][0]
check("outpaint ui 存档文件存在", os.path.isfile(os.path.join(tmp_temp, stash["filename"])))
check("outpaint ui 存档文件名前缀", stash["filename"].startswith("sf_outpaint_base_"))

# sides 模式 + 颜色：top=10 left=5 -> 37x74，角点应为填充色
src_sides = FakeTensor(np.full((1, 64, 32, 3), 0.5, dtype=np.float32))
r = Outpaint.outpaint(src_sides, json.dumps({"mode": "sides", "top": 10, "left": 5,
                                             "bottom": 0, "right": 0, "color": "#ff0000"}))
out, w, h, info = r["result"]
check("outpaint sides 尺寸", (w, h) == (37, 74) and info["top"] == 10 and info["left"] == 5)
check("outpaint sides 填充色角点", np.allclose(out.data[0, 0, 0], [1.0, 0.0, 0.0], atol=1e-3))
# 原图区域（canvas(15,20) -> 原图(10,10)）：PIL 通道经 128/255 量化，容差放宽
check("outpaint sides 原图保持", np.allclose(out.data[0, 20, 15], [0.5, 0.5, 0.5], atol=1e-2))

# limit：pad 128x64 -> max_mp 0.05MP -> factor=sqrt(0.05*1048576/8192)=sqrt(6.4)
# new_w=round(128*2.5298)=324, new_h=round(64*2.5298)=162
r = Outpaint.outpaint(src_ratio, json.dumps({"mode": "ratio", "ratio": "2:1",
                                             "anchor": "right", "limit": 0.05}))
out, w, h, info = r["result"]
check("outpaint limit 缩放", (w, h) == (324, 162))

# snap：limit 关时 pad 过程吸附（snap 16）：52x80 -> floor 48x80
r = Outpaint.outpaint(src_sides, json.dumps({"mode": "sides", "left": 20, "top": 16,
                                             "bottom": 0, "right": 0, "snap": 16}))
out, w, h, info = r["result"]
check("outpaint snap 无 limit 时 pad 吸附", (w, h) == (48, 80))

# ── stitch 透传兜底 ──
Stitch = mod.SFImageOutpaintStitch()
img = FakeTensor(np.full((1, 8, 8, 3), 0.2, dtype=np.float32))
out, mask = Stitch.stitch(img, None)
check("stitch 无 info 透传", out is img and mask.data.shape == (1, 8, 8) and not mask.data.any())
out, mask = Stitch.stitch(img, {"original": "nope"})
check("stitch 非法 info 透传", out is img)

# ── stitch 全流程：原始 4x4 灰(0.5)，画布 8x8(0.2)，feather=1 ──
orig = FakeTensor(np.full((1, 4, 4, 3), 0.5, dtype=np.float32))
info = {"original": orig, "left": 2, "top": 2, "right": 2, "bottom": 2,
        "orig_w": 4, "orig_h": 4, "canvas_w": 8, "canvas_h": 8}
canvas = FakeTensor(np.full((1, 8, 8, 3), 0.2, dtype=np.float32))
out, mask = Stitch.stitch(canvas, info, feather=1, color_match=0)
check("stitch 尺寸", out.data.shape == (1, 8, 8, 3))
check("stitch 原始中心贴回", np.allclose(out.data[0, 4, 4], [0.5, 0.5, 0.5], atol=1e-3))
check("stitch 接缝角点保持画布", np.allclose(out.data[0, 2, 2], [0.2, 0.2, 0.2], atol=1e-3))
check("stitch 原区域外不动", np.allclose(out.data[0, 0, 0], [0.2, 0.2, 0.2], atol=1e-3))
check("stitch mask 尺寸", mask.data.shape == (1, 8, 8))
check("stitch mask 原始内部 0", mask.data[0, 4, 4] == 0.0)
check("stitch mask 接缝角点 1", mask.data[0, 2, 2] == 1.0)
check("stitch mask 区域外 1", mask.data[0, 0, 0] == 1.0)

# resize 恢复：画布给半尺寸 4x4，info 声称画布 8x8 -> 先放大再贴
small = FakeTensor(np.full((1, 4, 4, 3), 0.2, dtype=np.float32))
out, mask = Stitch.stitch(small, info, feather=0, color_match=0)
check("stitch resize 恢复", out.data.shape == (1, 8, 8, 3))
check("stitch resize 后原始中心贴回", np.allclose(out.data[0, 4, 4], [0.5, 0.5, 0.5], atol=1e-3))

# 批次配对：画布 b=2，原始 b=1 -> repeat
canvas2 = FakeTensor(np.full((2, 8, 8, 3), 0.2, dtype=np.float32))
out, mask = Stitch.stitch(canvas2, info, feather=0, color_match=0)
check("stitch 批次配对", out.data.shape == (2, 8, 8, 3) and mask.data.shape == (2, 8, 8))
check("stitch 两帧都贴回", np.allclose(out.data[1, 4, 4], [0.5, 0.5, 0.5], atol=1e-3))

# ── _feather_sides 数值 ──
fs = Stitch._feather_sides
a = fs(10, 10, 4, True, False, False, False)
check("_feather_sides 左沿 0", a[0, 0] == 0.0)
check("_feather_sides 3px 处 0.75", abs(a[0, 3] - 0.75) < 1e-6)
check("_feather_sides 5px 处 1", a[0, 5] == 1.0)
check("_feather_sides 未标记边硬", a[0, 9] == 1.0 and a[9, 5] == 1.0)
a = fs(10, 10, 4, False, False, False, False)
check("_feather_sides 无边标记全 1", a.sum() == 100.0)
a = fs(10, 10, 0, True, False, False, False)
check("_feather_sides feather 0 全 1", a.sum() == 100.0)
a = fs(10, 10, 4, False, True, True, False)
check("_feather_sides 顶/右边淡出", a[0, 5] == 0.0 and a[5, 9] == 0.0 and a[5, 0] == 1.0)

# ── _color_match 数值：均匀场景下整画布应被平移到原始色调 ──
orig = FakeTensor(np.full((1, 4, 4, 3), 0.3, dtype=np.float32))
canvas = FakeTensor(np.full((1, 8, 8, 3), 0.7, dtype=np.float32))
cm = Stitch._color_match(canvas, orig, 2, 2, 2, 2, 1.0)
check("_color_match 均匀色调平移", np.allclose(cm.data, 0.3, atol=1e-4))
cm0 = Stitch._color_match(canvas, orig, 2, 2, 2, 2, 0.0)
check("_color_match 强度 0 原样", np.allclose(cm0.data, 0.7, atol=1e-6))
cm_none = Stitch._color_match(canvas, orig, 0, 0, 0, 0, 1.0)
check("_color_match 无边不动作", cm_none is canvas)

# ── 收尾 ──
print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
