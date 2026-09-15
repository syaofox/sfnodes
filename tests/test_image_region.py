# SFReferenceRegionNeutralize 后端模拟测试（python tests/test_image_region.py）
# 覆盖：
#   - sf_utils.image_region.neutralize_region 纯逻辑：strength=0 原样 / blur|mean|fill /
#     区域外不变 / 羽化过渡 / 批量与 mask 广播 / mask 尺寸不符抛错
#   - 节点壳：CATEGORY / RETURN_NAMES / INPUT_TYPES / execute（尺寸一致路径）
# mock：torch / torch.nn.functional（纯逻辑只需存在 torch 模块即可，模糊走 PIL）

import os
import sys
import types
import importlib.util

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


# ── fake torch（inpaint_helpers 顶层 import torch；模糊本身走 PIL）──────────────
fake_torch = types.ModuleType("torch")
fake_torch.float32 = "float32"
sys.modules["torch"] = fake_torch

# ── 纯逻辑 ───────────────────────────────────────────────────────────────────
from sf_utils.image_region import neutralize_region  # noqa: E402


def make_img(h=32, w=32, seed=0, c=3):
    rng = np.random.RandomState(seed)
    return rng.rand(1, h, w, c).astype(np.float32)


def make_mask(h=32, w=32, box=(8, 24, 8, 24)):
    m = np.zeros((1, h, w), dtype=np.float32)
    m[0, box[0]:box[1], box[2]:box[3]] = 1.0
    return m


img = make_img()
mask = make_mask()

# strength=0 → 原样
out = neutralize_region(img, mask, mode="blur", strength=0.0)
check("strength=0 原样", np.array_equal(out, img))

# blur：区域内变化（噪声被抹平）、区域外严格不变
out_blur = neutralize_region(img, mask, mode="blur", strength=1.0, blur_radius=6, feather=0.0)
inside = mask[0] > 0.5
check("blur 区域内变化", not np.allclose(out_blur[0][inside], img[0][inside]))
check("blur 区域内方差下降", out_blur[0][inside].std() < img[0][inside].std())
outside = ~inside
check("blur 区域外不变", np.array_equal(out_blur[0][outside], img[0][outside]))

# mean：区域内等于该区域均值、区域外不变
out_mean = neutralize_region(img, mask, mode="mean", strength=1.0, feather=0.0)
region_mean = img[0][inside].reshape(-1, 3).mean(axis=0)
check("mean 区域内为区域均值", np.allclose(out_mean[0][inside], region_mean, atol=1e-5))
check("mean 区域外不变", np.array_equal(out_mean[0][outside], img[0][outside]))

# fill：区域内等于常量、区域外不变
out_fill = neutralize_region(img, mask, mode="fill", strength=1.0, fill_value=0.25, feather=0.0)
check("fill 区域内为常量", np.allclose(out_fill[0][inside], 0.25))
check("fill 区域外不变", np.array_equal(out_fill[0][outside], img[0][outside]))

# 羽化：边界出现 0/1 之间的过渡值，且硬区中心仍被替换
out_feather = neutralize_region(img, mask, mode="fill", strength=1.0, fill_value=0.0,
                                feather=0.05)
center = np.zeros_like(mask, dtype=bool)
center[0, 14:18, 14:18] = True
check("羽化中心完全替换", np.allclose(out_feather[0][center[0]], 0.0))
band = out_feather[0][8, 8:12]  # 上边界附近
check("羽化边界存在过渡值", np.any((band > 0.02) & (band < 0.98)))

# 批量：每帧独立 mask；mask 批量=1 广播
img2 = np.concatenate([make_img(seed=1), make_img(seed=2)], axis=0)
m2 = np.stack([mask[0], np.zeros((32, 32), np.float32)])
out_b = neutralize_region(img2, m2, mode="fill", strength=1.0, fill_value=0.0, feather=0.0)
check("批量透传形状", out_b.shape == img2.shape)
check("批量第2帧 mask=0 原样", np.array_equal(out_b[1], img2[1]))
img3 = img2[0:1]
out_bc = neutralize_region(img3, mask, mode="fill", strength=1.0, fill_value=0.0, feather=0.0)
check("mask 单帧广播", np.allclose(out_bc[0][inside], 0.0))

# mask 尺寸不符 → 抛错（节点负责缩放）
try:
    neutralize_region(img, make_mask(16, 16), strength=1.0)
    check("mask 尺寸不符抛错", False)
except ValueError:
    check("mask 尺寸不符抛错", True)

# 3D 输入返回 3D
out3 = neutralize_region(img[0], mask[0], mode="fill", strength=1.0, fill_value=0.0)
check("3D 输入 3D 输出", out3.ndim == 3 and out3.shape == img[0].shape)

# ── 节点壳 ───────────────────────────────────────────────────────────────────
class FT:
    def __init__(self, a):
        self._a = np.asarray(a, dtype=np.float32)

    @property
    def shape(self):
        return self._a.shape

    @property
    def ndim(self):
        return self._a.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def unsqueeze(self, i):
        return FT(np.expand_dims(self._a, i))


fake_torch.Tensor = FT
fake_torch.from_numpy = lambda a: FT(a)
fake_nn = types.ModuleType("torch.nn")
fake_nnf = types.ModuleType("torch.nn.functional")
fake_nnf.interpolate = lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调用 interpolate"))
fake_nn.functional = fake_nnf
fake_torch.nn = fake_nn
sys.modules["torch"] = fake_torch
sys.modules["torch.nn"] = fake_nn
sys.modules["torch.nn.functional"] = fake_nnf

for pkg, path in [
    ("sfnodes", [root]),
    ("sfnodes.nodes", [os.path.join(root, "nodes")]),
    ("sfnodes.nodes.image", [os.path.join(root, "nodes", "image")]),
    ("sfnodes.sf_utils", [os.path.join(root, "sf_utils")]),
]:
    m = types.ModuleType(pkg)
    m.__path__ = path
    sys.modules[pkg] = m

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.region_neutralize",
    os.path.join(root, "nodes", "image", "region_neutralize.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

Node = mod.SFReferenceRegionNeutralize
check("节点 CATEGORY", Node.CATEGORY == "sfnodes/image")
check("节点 RETURN_NAMES", Node.RETURN_NAMES == ("image",))
it = Node.INPUT_TYPES()
check("INPUT_TYPES required", set(it["required"]) ==
      {"image", "mask", "mode", "strength", "blur_radius", "feather", "fill_value"})
check("mode 选项", it["required"]["mode"][0] == ["blur", "mean", "fill"])
check("mode 默认 blur", it["required"]["mode"][1]["default"] == "blur")

res = Node().neutralize(FT(img), FT(mask), mode="fill", strength=1.0, fill_value=0.25,
                        feather=0.0)
check("execute 返回 1 路", isinstance(res, tuple) and len(res) == 1)
check("execute 结果区域替换", np.allclose(res[0].numpy()[0][inside], 0.25))

# 尺寸不符的 mask → 走 _fit_mask（此处应触发 interpolate 断言，证明分支被调用）
try:
    Node().neutralize(FT(img), FT(make_mask(16, 16)), mode="fill", strength=1.0)
    check("mask 尺寸不符走缩放分支", False)
except AssertionError:
    check("mask 尺寸不符走缩放分支", True)

# ── 汇总 ─────────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"{len(failures)} failures: {failures}")
    sys.exit(1)
print("All tests passed.")
