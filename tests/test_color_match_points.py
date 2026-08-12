# SFImageColorMatchByPoints 测试（Python 直接运行：python tests/test_color_match_points.py）
# 覆盖：
#   1. sf_utils/color_match_points.py 纯逻辑（真实 numpy）：
#      三点提取（灰阶/彩色三块图、同亮度兜底）、LUT（端点/恒等/单调/非单调防御）、
#      查表应用、三点闭环（目标三点 → 参考三点）
#   2. 节点 execute 集成（mock torch/comfy，数值路径全真实 numpy）：
#      strength 混合、mask 软混合、mid_percentile 越界 clamp、多帧参考三点平均
#   3. INPUT_TYPES 结构
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


# ── 纯逻辑模块（真实 numpy，先加载供节点模块复用）──
spec_cmp = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.color_match_points",
    os.path.join(root, "sf_utils", "color_match_points.py"),
)
logic = importlib.util.module_from_spec(spec_cmp)
sys.modules["sfnodes.sf_utils.color_match_points"] = logic
spec_cmp.loader.exec_module(logic)

from sf_utils.color_match_points import extract_points, build_lut, apply_lut  # noqa: E402


# ── 1. 纯逻辑：三点提取 ──
img_gray = np.zeros((3, 3, 3), dtype=np.float32)
img_gray[:1] = 0.1
img_gray[1:2] = 0.5
img_gray[2:] = 0.9
dark, mid, light = extract_points(img_gray)
check("extract: 灰阶三块 暗点=0.1", np.allclose(dark, 0.1, atol=1e-6))
check("extract: 灰阶三块 灰点=0.5", np.allclose(mid, 0.5, atol=1e-6))
check("extract: 灰阶三块 亮点=0.9", np.allclose(light, 0.9, atol=1e-6))

img_rgb = np.zeros((3, 3, 3), dtype=np.float32)
img_rgb[:1] = [0.1, 0.2, 0.3]
img_rgb[1:2] = [0.5, 0.4, 0.3]
img_rgb[2:] = [0.8, 0.7, 0.9]
dark, mid, light = extract_points(img_rgb)
check("extract: 彩色三块 暗点=各块 RGB", np.allclose(dark, [0.1, 0.2, 0.3], atol=1e-6))
check("extract: 彩色三块 灰点=各块 RGB", np.allclose(mid, [0.5, 0.4, 0.3], atol=1e-6))
check("extract: 彩色三块 亮点=各块 RGB", np.allclose(light, [0.8, 0.7, 0.9], atol=1e-6))

img_flat = np.full((4, 4, 3), 0.42, dtype=np.float32)
dark, mid, light = extract_points(img_flat)
check("extract: 全图同亮度 三点=该值", np.allclose(dark, 0.42, atol=1e-6) and np.allclose(mid, 0.42, atol=1e-6) and np.allclose(light, 0.42, atol=1e-6))

img_bimodal = np.zeros((2, 4, 3), dtype=np.float32)
img_bimodal[:1] = 0.2
img_bimodal[1:] = 0.8
dark, mid, light = extract_points(img_bimodal)
check("extract: 双极图灰点区间无像素 兜底单像素", mid[0] in (0.2, 0.8) and np.allclose(dark, 0.2) and np.allclose(light, 0.8))

img_nan = img_rgb.copy()
img_nan[0, 0] = np.nan
img_nan[1, 1] = np.inf
dark, mid, light = extract_points(img_nan)
check("extract: NaN/Inf 像素被忽略", np.allclose(dark, [0.1, 0.2, 0.3], atol=1e-6) and np.allclose(mid, [0.5, 0.4, 0.3], atol=1e-6) and np.allclose(light, [0.8, 0.7, 0.9], atol=1e-6))

img_allnan = np.full((2, 2, 3), np.nan, dtype=np.float32)
dark, mid, light = extract_points(img_allnan)
check("extract: 全 NaN 兜底黑色三点", np.allclose(dark, 0) and np.allclose(mid, 0) and np.allclose(light, 0))


# ── 2. 纯逻辑：LUT ──
p_ok = (np.array([0.2, 0.3, 0.4]), np.array([0.5, 0.5, 0.5]), np.array([0.8, 0.7, 0.6]))
lut_id = build_lut(p_ok, p_ok)
xs = np.linspace(0.0, 1.0, 256)
check("LUT: target=ref 恒等", np.allclose(lut_id, xs, atol=1e-6))
check("LUT: 端点 0→0 / 1→1", lut_id[0][0] == 0.0 and lut_id[0][-1] == 1.0)
check("LUT: 形状 [3,256] float32", lut_id.shape == (3, 256) and lut_id.dtype == np.float32)

lut_ok = build_lut(p_ok, (np.array([0.3, 0.3, 0.3]), np.array([0.5, 0.5, 0.5]), np.array([0.7, 0.7, 0.7])))
check("LUT: 单调不减", (np.diff(lut_ok, axis=1) >= -1e-6).all())

p_bad = (np.array([0.6, 0.6, 0.6]), np.array([0.4, 0.4, 0.4]), np.array([0.8, 0.8, 0.8]))
lut_bad = build_lut(p_bad, p_ok)
check("LUT: 非单调三点防御 有限值且端点正确", np.isfinite(lut_bad).all() and lut_bad[0][0] == 0.0 and lut_bad[0][-1] == 1.0)


# ── 3. 纯逻辑：查表 + 三点闭环（单帧、空间三块图）──
target = np.zeros((1, 3, 4, 3), dtype=np.float32)
target[0, :1] = 0.2
target[0, 1:2] = 0.4
target[0, 2:] = 0.6
ref = np.zeros((1, 3, 4, 3), dtype=np.float32)
ref[0, :1] = 0.3
ref[0, 1:2] = 0.5
ref[0, 2:] = 0.7
t_points = extract_points(target[0])
r_points = extract_points(ref[0])
out = apply_lut(target[0], build_lut(t_points, r_points))
check("闭环: 暗块 → 参考暗点", np.allclose(out[:1], 0.3, atol=1e-6))
check("闭环: 中块 → 参考灰点", np.allclose(out[1:2], 0.5, atol=1e-6))
check("闭环: 亮块 → 参考亮点", np.allclose(out[2:], 0.7, atol=1e-6))


# ── mock torch / comfy（数值路径全真实 numpy）──
class ND:
    """torch.from_numpy 的桩：返回 numpy 数据，仅补 .to(device) 接口。"""

    def __init__(self, a):
        self.a = a
        self.shape = a.shape

    def to(self, *k):
        return self.a


torch = types.ModuleType("torch")
torch.from_numpy = lambda a: ND(a)
torch.float32 = "float32"


def fake_cat(ts, dim=0):
    if any(getattr(t, "data", None) is not None for t in ts):
        return np.concatenate(ts, axis=dim)
    return FakeTensor((sum(t.shape[0] for t in ts),) + ts[0].shape[1:])


torch.cat = fake_cat
sys.modules["torch"] = torch

comfy = types.ModuleType("comfy")
comfy.utils = types.ModuleType("comfy.utils")
upscale_calls = []


def fake_upscale(t, w, h, **k):
    upscale_calls.append((w, h))
    arr = t.numpy()  # [B,1,H,W]
    out = np.empty((arr.shape[0], 1, h, w), dtype=arr.dtype)
    for b in range(arr.shape[0]):
        y = (np.arange(h) * arr.shape[2] // h).clip(0, arr.shape[2] - 1)
        x = (np.arange(w) * arr.shape[3] // w).clip(0, arr.shape[3] - 1)
        out[b, 0] = arr[b, 0][np.ix_(y, x)]
    return FakeTensor(data=out)


comfy.utils.common_upscale = fake_upscale
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils

for pkg in ("sfnodes", "sfnodes.sf_utils"):
    m = types.ModuleType(pkg)
    m.__path__ = []
    sys.modules[pkg] = m

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.color_match_points",
    os.path.join(root, "nodes", "image", "color_match_points.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

node = mod.ImageColorMatchByPoints()


class FakeTensor:
    """桩：有 data 时走真实 numpy 运算（数值路径），无 data 时仅推形状（流程路径）。"""

    def __init__(self, shape=None, data=None):
        self.data = data
        self.shape = tuple(int(s) for s in (shape if shape is not None else data.shape))
        self.device = "cpu"
        self.dtype = "float32"

    def float(self):
        return self

    def detach(self):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        if self.data is not None:
            return self.data
        return np.zeros(self.shape, dtype=np.float32)

    def unsqueeze(self, dim):
        if self.data is not None:
            return FakeTensor(data=np.expand_dims(self.data, dim))
        d = dim if dim >= 0 else len(self.shape) + dim + 1
        s = list(self.shape)
        s.insert(d, 1)
        return FakeTensor(s)

    def repeat(self, *s):
        if self.data is not None:
            return FakeTensor(data=np.tile(self.data, tuple(s)))
        return FakeTensor([a * b for a, b in zip(self.shape, s)])

    def __getitem__(self, idx):
        if self.data is not None:
            return FakeTensor(data=self.data[idx])
        if isinstance(idx, tuple):
            new_s = []
            for i, it in enumerate(idx):
                if it is None:
                    new_s.append(1)
                elif isinstance(it, slice):
                    n = self.shape[i]
                    new_s.append(max(0, ((it.stop if it.stop is not None else n) - (it.start or 0) + (it.step or 1) - 1) // (it.step or 1)))
                else:  # int: 丢弃该维
                    continue
            return FakeTensor(tuple(new_s) + tuple(self.shape[len(idx):]))
        if isinstance(idx, slice):
            n = self.shape[0]
            return FakeTensor((max(0, (idx.stop if idx.stop is not None else n) - (idx.start or 0)),) + self.shape[1:])
        return FakeTensor(self.shape[1:])


# ── 4. INPUT_TYPES 结构 ──
it = node.INPUT_TYPES()
check("INPUT_TYPES: 双图输入", "target_image" in it["required"] and "reference_image" in it["required"])
check("INPUT_TYPES: 三百分位+强度", {"dark_percentile", "mid_percentile", "light_percentile", "strength"} <= set(it["required"]))
check("INPUT_TYPES: 分位默认 0.5/50/99.5", it["required"]["dark_percentile"][1]["default"] == 0.5 and it["required"]["mid_percentile"][1]["default"] == 50.0 and it["required"]["light_percentile"][1]["default"] == 99.5)
check("INPUT_TYPES: strength 0-2", it["required"]["strength"][1]["max"] == 2.0)
check("INPUT_TYPES: target_mask 可选", "target_mask" in it["optional"])
check("INPUT_TYPES: 全部输入在 execute 签名中", set(it["required"]) | set(it["optional"]) <= set(
    node.execute.__code__.co_varnames[:node.execute.__code__.co_argcount]))


# ── 5. 节点 execute 集成（数值）──
def run(target, ref, strength=1.0, mask=None, dark_p=5.0, mid_p=50.0, light_p=95.0):
    return node.execute(
        FakeTensor(target.shape, target), FakeTensor(ref.shape, ref),
        dark_p, mid_p, light_p, strength, target_mask=mask,
    )[0]


out = run(target, ref)
check("execute: 输出形状", out.shape == (1, 3, 4, 3))
check("execute: 暗块 → 参考暗点", np.allclose(out[0, :1], 0.3, atol=1e-6))
check("execute: 中块 → 参考灰点", np.allclose(out[0, 1:2], 0.5, atol=1e-6))
check("execute: 亮块 → 参考亮点", np.allclose(out[0, 2:], 0.7, atol=1e-6))

target2 = np.concatenate([target, target], axis=0)
out2 = run(target2, ref)
check("execute: 多帧目标 逐帧独立匹配", np.allclose(out2[0], out[0]) and np.allclose(out2[1], out[0]))

out0 = run(target, ref, strength=0.0)
check("execute: strength=0 原样", np.allclose(out0, target, atol=1e-6))

out_half = run(target, ref, strength=0.5)
check("execute: strength=0.5 半程混合", np.allclose(out_half[0, :1], 0.25, atol=1e-6))

mask_all = FakeTensor((1, 3, 4), np.ones((1, 3, 4), dtype=np.float32))
out_masked = run(target, ref, mask=mask_all)
check("execute: mask 全 1 仍匹配", np.allclose(out_masked[0, :1], 0.3, atol=1e-6))

mask_none = FakeTensor((1, 3, 4), np.zeros((1, 3, 4), dtype=np.float32))
out_nomask = run(target, ref, mask=mask_none)
check("execute: mask 全 0 保持原图", np.allclose(out_nomask, target, atol=1e-6))

out_clamped = run(target, ref, dark_p=49.0, mid_p=99.0, light_p=95.0)
check("execute: mid 越界 clamp 不炸且形状对", out_clamped.shape == (1, 3, 4, 3))

ref2 = np.concatenate([ref, ref], axis=0)
out_multi = run(target, ref2)
check("execute: 多帧参考 三点平均", np.allclose(out_multi[0, :1], 0.3, atol=1e-6))

upscale_calls.clear()
mask_small = FakeTensor((1, 2, 2), np.ones((1, 2, 2), dtype=np.float32))
out_resized = run(target, ref, mask=mask_small)
check("execute: mask 尺寸不匹配走 common_upscale", upscale_calls == [(4, 3)])
check("execute: mask resize 后形状对", out_resized.shape == (1, 3, 4, 3))


# ── 6. 节点 execute 集成（全假流程：多帧/分批形状）──
def run_fake(batch=2, ref_batch=2, mask_batch=1):
    t = FakeTensor((batch, 16, 16, 3))
    r = FakeTensor((ref_batch, 24, 24, 3))
    kw = {}
    if mask_batch:
        kw["target_mask"] = FakeTensor((mask_batch, 16, 16))
    return node.execute(t, r, 5.0, 50.0, 95.0, 1.0, **kw)[0]


outf = run_fake(batch=3, ref_batch=2, mask_batch=1)
check("流程: 多帧+单帧 mask 广播 输出形状", outf.shape == (3, 16, 16, 3))
outf = run_fake(batch=2, ref_batch=4, mask_batch=0)
check("流程: 多帧参考+无 mask 输出形状", outf.shape == (2, 16, 16, 3))

# ── 汇总 ──
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
