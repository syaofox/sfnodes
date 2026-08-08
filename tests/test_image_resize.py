# SFImageResize 后端逻辑测试（Python 直接运行：python tests/test_image_resize.py）
# 覆盖：
#   - sf_utils.resize_engine._apply_wired_size（wired 尺寸优先级镜像：
#     longest_side > 单轴比例 > 双轴精确盒；0/负值 = 直通；纯函数不修改入参）
#   - 节点模块（mock torch / comfy.model_management）：INPUT_TYPES 结构、
#     _tensor_to_pils 通道防御（1ch/4ch）、alpha -> MASK 反转、mask 尺寸对齐、
#     execute 集成（wired 驱动、mask 输出、UI payload）
import importlib.util
import json
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── 1. _apply_wired_size（纯函数，无 mock）──
from sf_utils.resize_engine import _apply_wired_size, RESIZE_DEFAULTS  # noqa: E402

st = dict(RESIZE_DEFAULTS)

def ws(width=None, height=None, longest_side=None, orig_w=100, orig_h=50, state=None):
    return _apply_wired_size(state or dict(RESIZE_DEFAULTS), width, height, longest_side, orig_w, orig_h)

r = ws()
check("无 wired -> 原 state（引用）", r is st or r == st)
check("无 wired 不改 mode", ws()["mode"] == "off")

r = ws(longest_side=1024)
check("longest_side 1024 -> mode=longest_side", r["mode"] == "longest_side" and r["longest_side"] == 1024)

r = ws(longest_side=0)
check("longest_side 0 -> off 直通", r["mode"] == "off")

r = ws(width=200)
check("只接 width -> scale_factor 2.0", r["mode"] == "scale_factor" and abs(r["scale_factor"] - 2.0) < 1e-9)

r = ws(height=100)
check("只接 height -> scale_factor 2.0", r["mode"] == "scale_factor" and abs(r["scale_factor"] - 2.0) < 1e-9)

r = ws(width=0)
check("只接 width 0 -> off 直通", r["mode"] == "off")

r = ws(width=50, height=50)
check("双接 -> cover 精确盒", r["mode"] == "cover" and r["cover_w"] == 50 and r["cover_h"] == 50)

r = ws(width=50, height=50, state=dict(RESIZE_DEFAULTS, mode="fit_inside"))
check("双接 + fit_inside -> 保持 fit_inside", r["mode"] == "fit_inside" and r["fit_w"] == 50 and r["fit_h"] == 50)

base = dict(RESIZE_DEFAULTS, mode="pad", pad_top=4)
r = _apply_wired_size(base, None, None, 512, 100, 50)
check("原 state 不被修改（纯函数）", base["mode"] == "pad" and r["mode"] == "longest_side")

# ── 2. 节点模块（mock torch / comfy.model_management）──
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


class FakeTensor:
    """最小张量替身：clamp/cpu/numpy 返回真实 numpy，to 透传（测试只断言形状/值）。"""
    def __init__(self, arr):
        self._arr = arr

    def clamp(self, lo, hi):
        return FakeTensor(np.clip(self._arr, lo, hi))

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def to(self, dtype):
        return self

    def __getitem__(self, idx):
        return FakeTensor(self._arr[idx])


torch = types.ModuleType("torch")
torch.float32 = np.float32
torch.from_numpy = lambda a: FakeTensor(a.astype(np.float32))
torch.cat = lambda lst, dim=0: FakeTensor(np.concatenate([x._arr for x in lst], axis=dim))
sys.modules["torch"] = torch

comfy = types.ModuleType("comfy")
sys.modules["comfy"] = comfy
mm = types.ModuleType("comfy.model_management")
mm.intermediate_dtype = lambda: np.float32
sys.modules["comfy.model_management"] = mm

# 注册 sfnodes 包结构，使节点的相对导入（from ...sf_utils.resize_engine import）可解析
_sf_pkg = types.ModuleType("sfnodes")
_sf_pkg.__path__ = [root]
sys.modules["sfnodes"] = _sf_pkg
_sf_nodes_pkg = types.ModuleType("sfnodes.nodes")
_sf_nodes_pkg.__path__ = [os.path.join(root, "nodes")]
sys.modules["sfnodes.nodes"] = _sf_nodes_pkg
_sf_image_pkg = types.ModuleType("sfnodes.nodes.image")
_sf_image_pkg.__path__ = [os.path.join(root, "nodes", "image")]
sys.modules["sfnodes.nodes.image"] = _sf_image_pkg
_sf_utils_pkg = types.ModuleType("sfnodes.sf_utils")
_sf_utils_pkg.__path__ = [os.path.join(root, "sf_utils")]
sys.modules["sfnodes.sf_utils"] = _sf_utils_pkg

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.resize_image",
    os.path.join(root, "nodes", "image", "resize_image.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

check("模块已加载", hasattr(mod, "SFImageResize"))
check("CATEGORY", mod.SFImageResize.CATEGORY == "sfnodes/image")
check("DESCRIPTION 非空", isinstance(mod.SFImageResize.DESCRIPTION, str) and len(mod.SFImageResize.DESCRIPTION) > 0)
check("RETURN_TYPES", mod.SFImageResize.RETURN_TYPES == ("IMAGE", "MASK", "INT", "INT", "INT"))
check("RETURN_NAMES", mod.SFImageResize.RETURN_NAMES == ("image", "mask", "width", "height", "longest_side"))
check("FUNCTION", mod.SFImageResize.FUNCTION == "resize")
check("无 OUTPUT_NODE", getattr(mod.SFImageResize, "OUTPUT_NODE", False) is False)
check("本地注册键", mod.NODE_CLASS_MAPPINGS["SFImageResize"] is mod.SFImageResize)
check("本地显示名", mod.NODE_DISPLAY_NAME_MAPPINGS["SFImageResize"] == "SF Image Resize")

it = mod.SFImageResize.INPUT_TYPES()
check("required 含 image", "image" in it["required"] and it["required"]["image"][0] == "IMAGE")
check("optional mask", it["optional"]["mask"][0] == "MASK")
for n in ("width", "height", "longest_side"):
    check(f"optional {n} INT forceInput", it["optional"][n][0] == "INT" and it["optional"][n][1].get("forceInput") is True)
check("hidden 含 SFImageResizeState", "SFImageResizeState" in it["hidden"])

# ── 3. 张量/遮罩转换（FakeTensor + 真实 numpy/PIL）──

def ft(h, w, channels, fill=128):
    arr = np.full((1, h, w, channels), fill, dtype=np.float32) / 255.0
    return FakeTensor(arr)

pils, alphas = mod._tensor_to_pils(ft(10, 20, 3))
check("3ch -> RGB PIL 无 alpha", len(pils) == 1 and pils[0].mode == "RGB" and pils[0].size == (20, 10) and alphas is None)

pils, alphas = mod._tensor_to_pils(ft(10, 20, 1))
check("1ch 灰度 -> RGB 复制", pils[0].mode == "RGB" and pils[0].getpixel((0, 0)) == (128, 128, 128))

rgba = np.zeros((1, 5, 8, 4), dtype=np.float32)
rgba[..., 0:3] = 0.5
rgba[..., 3] = 0.25  # alpha 64
pils, alphas = mod._tensor_to_pils(FakeTensor(rgba))
check("4ch RGBA -> RGB + alpha", pils[0].mode == "RGB" and alphas is not None and alphas[0].mode == "L")
check("alpha 值保留", np.array(alphas[0])[0, 0] == 64)

mask_pils = mod._alpha_to_mask_pils(alphas, (8, 5))
check("alpha -> mask 反转（1=透明）", np.array(mask_pils[0])[0, 0] == 255 - 64)
mask_pils = mod._alpha_to_mask_pils([None], (8, 5))
check("无 alpha 帧 -> 全 0 mask", np.array(mask_pils[0]).max() == 0)

m64 = np.zeros((1, 64, 64), dtype=np.float32)
m64[0, 0, 0] = 1.0
out = mod._mask_to_pils(FakeTensor(m64), 2, (8, 5))
check("mask 尺寸对齐到图 + 补齐帧数", len(out) == 2 and out[0].size == (8, 5))
m8 = np.zeros((1, 5, 8), dtype=np.float32)
m8[0, 2, 3] = 1.0
out = mod._mask_to_pils(FakeTensor(m8), 1, (8, 5))
check("mask 同尺寸直通保留值", np.array(out[0])[2, 3] == 255)
m_up = np.zeros((1, 64, 64), dtype=np.float32)
m_up[0, 0, 0] = 1.0
out = mod._mask_to_pils(FakeTensor(m_up), 1, (128, 128))
check("mask 放大 NEAREST 保留角点", np.array(out[0])[0, 0] == 255)
out = mod._mask_to_pils(None, 2, (8, 5))
check("None mask -> 全 0", len(out) == 2 and np.array(out[1]).max() == 0)
check("None mask -> 全 0", np.array(out[1]).max() == 0)

# ── 4. execute 集成 ──
img3 = np.full((1, 50, 100, 3), 0.5, dtype=np.float32)  # 100x50
img_rgba = np.zeros((1, 50, 100, 4), dtype=np.float32)
img_rgba[..., 0:3] = 0.5
img_rgba[..., 3] = 0.0  # 全透明

node = mod.SFImageResize()
state_off = json.dumps(dict(mod.DEFAULT_STATE, mode="off"))

res = node.resize(FakeTensor(img3), SFImageResizeState=state_off)
img, msk, w, h, longest = res["result"]
check("off 直通尺寸", (w, h) == (100, 50) and longest == 100)
check("输出 IMAGE 形状", img._arr.shape == (1, 50, 100, 3))
check("输出 MASK 形状（无输入 -> 全 0）", msk._arr.shape == (1, 50, 100) and msk._arr.max() == 0)
check("UI payload 键 sf_image_resize", res["ui"]["sf_image_resize"][0]["in_w"] == 100 and res["ui"]["sf_image_resize"][0]["out_w"] == 100)

res = node.resize(FakeTensor(img3), width=200, SFImageResizeState=state_off)
img, msk, w, h, longest = res["result"]
check("wired width=200 -> 200x100", (w, h) == (200, 100) and longest == 200)

res = node.resize(FakeTensor(img3), longest_side=512, SFImageResizeState=state_off)
_, _, w, h, longest = res["result"]
check("wired longest_side=512 -> 512x256", (w, h) == (512, 256) and longest == 512)

res = node.resize(FakeTensor(img3), width=0, SFImageResizeState=state_off)
_, _, w, h, _ = res["result"]
check("wired width=0 -> off 直通", (w, h) == (100, 50))

# RGBA 图：透明度走 MASK 输出
res = node.resize(FakeTensor(img_rgba), SFImageResizeState=state_off)
_, msk, w, h, _ = res["result"]
check("RGBA 图 -> MASK=1-alpha（全透明 -> 全 1）", msk._arr.shape == (1, 50, 100) and msk._arr.min() == 1.0)

# 显式 mask 优先于图片自带 alpha
mask_in = np.zeros((1, 50, 100), dtype=np.float32)
res = node.resize(FakeTensor(img_rgba), mask=FakeTensor(mask_in), SFImageResizeState=state_off)
_, msk, _, _, _ = res["result"]
check("显式 mask 覆盖 alpha", msk._arr.max() == 0.0)

# pad 模式 mask 边框
state_pad = json.dumps(dict(mod.DEFAULT_STATE, mode="pad", pad_left=8, pad_right=8, pad_top=4, pad_bottom=4))
res = node.resize(FakeTensor(img3), SFImageResizeState=state_pad)
_, msk, w, h, _ = res["result"]
check("pad 尺寸 100x50 -> 116x58", (w, h) == (116, 58))
check("pad 边框 mask=1", msk._arr[0, 0, 0] == 1.0)
check("pad 原区域 mask=0", msk._arr[0, 20, 20] == 0.0)

# ── 5. 注册键一致性（AST）──
import ast
src = open(os.path.join(root, "__init__.py")).read()
tree = ast.parse(src)
classmap = dispmap = None
for anode in ast.walk(tree):
    if isinstance(anode, ast.Assign):
        for t in anode.targets:
            if isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS":
                classmap = {k.value for k in anode.value.keys}
            elif isinstance(t, ast.Name) and t.id == "NODE_DISPLAY_NAME_MAPPINGS":
                dispmap = {k.value for k in anode.value.keys}
check("注册键两字典一致", classmap == dispmap)
check("SFImageResize 已注册", "SFImageResize" in (classmap or set()))
check("显示名映射", dispmap is not None and "SFImageResize" in dispmap)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
