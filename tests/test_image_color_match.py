# SFImageColorMatch 增强测试（Python 直接运行：python tests/test_image_color_match.py）
# 覆盖：
#   1. INPUT_TYPES 结构（method 四算法、target_sample_mask、strength 范围）
#   2. numpy 数学对照：MKL 公式 vs mini-nodes 的 apply_mkl（同输入同期望）、
#      除零语义（posinf=0 -> 收敛参考均值）、多帧合并方差、Mean 公式
#   3. execute 集成（mock torch/kornia/comfy）：四方法分支跑通、
#      reference_mask 单帧广播到多帧参考（旧版 assert 会炸）、
#      target_sample_mask 传入目标统计、nan_to_num 除零参数、target_mask 软混合
# mock：torch/kornia/comfy（numpy 本机真实可用）
import importlib.util
import itertools
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


# ── FakeTensor：形状推导 + 全假运算（仅用于流程/结构验证）──
class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(int(s) for s in shape)
        self.device = "cpu"
        self.dtype = torch.float32

    @property
    def T(self):
        return FakeTensor(self.shape[::-1])

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        return FakeTensor([self.shape[d] for d in dims])

    def to(self, *a, **k):
        return self

    def unsqueeze(self, dim):
        s = list(self.shape)
        d = dim if dim >= 0 else len(s) + dim + 1
        s.insert(d, 1)
        return FakeTensor(s)

    def float(self):
        return self

    def bool(self):
        return self

    def expand(self, *s):
        return self

    def reshape(self, *s):
        if len(s) == 1 and isinstance(s[0], (tuple, list)):
            s = s[0]
        out = []
        unknown = None
        for i, x in enumerate(s):
            if x == -1:
                unknown = i
                out.append(1)
            else:
                out.append(int(x))
        if unknown is not None:
            total = 1
            for x in self.shape:
                total *= x
            known = 1
            for x in out:
                known *= x
            out[unknown] = total // known
        return FakeTensor(out)

    def repeat(self, *s):
        return FakeTensor([a * b for a, b in zip(self.shape, s)])

    def mean(self, dim=None, keepdim=False):
        return self._reduce(dim, keepdim)

    def std(self, dim=None, keepdim=False):
        return self._reduce(dim, keepdim)

    def sum(self, dim=None, keepdim=False):
        return self._reduce(dim, keepdim)

    def _reduce(self, dim, keepdim):
        s = list(self.shape)
        if dim is None:
            return FakeTensor([1] if keepdim else [])
        dims = [dim] if isinstance(dim, int) else list(dim)
        for d in sorted(dims, reverse=True):
            if keepdim:
                s[d] = 1
            else:
                del s[d]
        return FakeTensor(s)

    def clamp(self, *a, **k):
        return self

    def nan_to_num(self, *a, **k):
        return self

    def __getitem__(self, idx):
        if isinstance(idx, FakeTensor):
            return FakeTensor((999999,))  # bool 索引：展平为未知 N
        if isinstance(idx, tuple):
            new_s = []
            for i, it in enumerate(idx):
                if it is None:
                    new_s.append(1)
                elif isinstance(it, slice):
                    n = self.shape[i]
                    start, stop, step = it.start, it.stop, it.step
                    if stop is None:
                        stop = n
                    if start is None:
                        start = 0
                    if step is None:
                        step = 1
                    new_s.append(max(0, (stop - start + step - 1) // step))
            while len(new_s) < len(self.shape):
                new_s.append(self.shape[len(new_s)])
            return FakeTensor(new_s)
        if isinstance(idx, slice):
            start, stop = idx.start, idx.stop
            n = self.shape[0]
            return FakeTensor((max(0, (stop if stop is not None else n) - (start or 0)),) + self.shape[1:])
        return self

    def _bshape(self, o):
        if not hasattr(o, "shape"):
            return self.shape
        sa, sb = list(self.shape), list(o.shape)
        out = []
        for x, y in itertools.zip_longest(reversed(sa), reversed(sb), fillvalue=1):
            out.append(x if x >= y else y)
        return tuple(reversed(out))

    def __add__(self, o): return FakeTensor(self._bshape(o))
    def __radd__(self, o): return FakeTensor(self._bshape(o))
    def __sub__(self, o): return FakeTensor(self._bshape(o))
    def __rsub__(self, o): return FakeTensor(self._bshape(o))
    def __mul__(self, o): return FakeTensor(self._bshape(o))
    def __rmul__(self, o): return FakeTensor(self._bshape(o))
    def __truediv__(self, o): return FakeTensor(self._bshape(o))
    def __rtruediv__(self, o): return FakeTensor(self._bshape(o))
    def __pow__(self, o): return FakeTensor(self._bshape(o))
    def __matmul__(self, o): return FakeTensor(self._bshape(o))
    def __rmatmul__(self, o): return FakeTensor(self._bshape(o))
    def __neg__(self): return FakeTensor(self.shape)
    def __gt__(self, o): return FakeTensor(self.shape)
    def __lt__(self, o): return FakeTensor(self.shape)


# ── mock torch ──
torch = types.ModuleType("torch")
torch.float32 = "float32"
nan_to_num_calls = []

torch.split = lambda t, bs, dim=0: ([FakeTensor(t.shape)] if bs == 0 or bs >= t.shape[0] else
                                    [FakeTensor((bs,) + t.shape[1:]), FakeTensor((t.shape[0] - bs,) + t.shape[1:])])
torch.cat = lambda ts, dim=0: FakeTensor((sum(t.shape[0] for t in ts),) + ts[0].shape[1:])
torch.nan_to_num = lambda t, **kw: (nan_to_num_calls.append(kw), t)[1]
torch.sqrt = lambda t, **k: t
torch.clamp = lambda t, **k: t
torch.diag = lambda t: FakeTensor((t.shape[0],) * 2)
torch.eye = lambda n, *a, **k: FakeTensor((n, n))
torch.cov = lambda t: FakeTensor((t.shape[0], t.shape[0]))
torch.linalg = types.ModuleType("torch.linalg")
torch.linalg.eigh = lambda t: (FakeTensor((t.shape[0],)), FakeTensor(t.shape))
torch._modules = {"linalg": torch.linalg}
sys.modules["torch"] = torch

# ── mock cv2（本机无 opencv，仅模块加载需要）──
cv2 = types.ModuleType("cv2")
sys.modules["cv2"] = cv2

# ── mock comfy / kornia / comfy_extras ──
comfy = types.ModuleType("comfy")
comfy.utils = types.ModuleType("comfy.utils")
comfy.utils.common_upscale = lambda t, w, h, **k: FakeTensor((t.shape[0], t.shape[1], h, w))
comfy.model_management = types.ModuleType("comfy.model_management")
comfy.model_management.get_torch_device = lambda: "cuda:0"
comfy.model_management.intermediate_device = lambda: "cpu"
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils
sys.modules["comfy.model_management"] = comfy.model_management

comfy_extras = types.ModuleType("comfy_extras")
comfy_extras.nodes_post_processing = types.ModuleType("comfy_extras.nodes_post_processing")
for cls_name in ("Blend", "Blur", "Quantize"):
    setattr(comfy_extras.nodes_post_processing, cls_name, type(cls_name, (), {}))
sys.modules["comfy_extras"] = comfy_extras
sys.modules["comfy_extras.nodes_post_processing"] = comfy_extras.nodes_post_processing

kornia = types.ModuleType("kornia")
kornia.color = types.ModuleType("kornia.color")
for fn in ("rgb_to_lab", "rgb_to_linear_rgb", "rgb_to_ycbcr", "rgb_to_luv", "rgb_to_yuv", "rgb_to_xyz",
           "lab_to_rgb", "linear_rgb_to_rgb", "ycbcr_to_rgb", "luv_to_rgb", "yuv_to_rgb", "xyz_to_rgb"):
    setattr(kornia.color, fn, lambda t: t)
kornia.filters = types.ModuleType("kornia.filters")
kornia.filters.gaussian_blur2d = lambda t, ks, sig: t
sys.modules["kornia"] = kornia
sys.modules["kornia.color"] = kornia.color
sys.modules["kornia.filters"] = kornia.filters

# ── 包桩（processing.py 的相对导入 ...sf_utils.image_convert）──
for pkg in ("sfnodes", "sfnodes.sf_utils"):
    m = types.ModuleType(pkg)
    m.__path__ = []
    sys.modules[pkg] = m
fake_ic = types.ModuleType("sfnodes.sf_utils.image_convert")
fake_ic.image_posterize = lambda *a, **k: None
sys.modules["sfnodes.sf_utils.image_convert"] = fake_ic

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.processing",
    os.path.join(root, "nodes", "image", "processing.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

node = mod.ImageColorMatch()

# ── 1. INPUT_TYPES 结构 ──
it = node.INPUT_TYPES()
check("INPUT_TYPES: method 四选项", it["required"]["method"][0] == ["Statistical", "Mean", "MKL", "Wavelet"])
check("INPUT_TYPES: method 默认 Statistical", it["required"]["method"][1]["default"] == "Statistical")
check("INPUT_TYPES: target_image/reference_image 命名", "target_image" in it["required"] and "reference_image" in it["required"])
check("INPUT_TYPES: target_sample_mask 可选输入", "target_sample_mask" in it["optional"])
check("INPUT_TYPES: reference_mask/target_mask 保留", "reference_mask" in it["optional"] and "target_mask" in it["optional"])
check("INPUT_TYPES: strength 0-2", it["required"]["strength"][1]["max"] == 2.0)
check("INPUT_TYPES: 全部输入在 execute 签名中", set(it["required"]) | set(it["optional"]) <= set(
    node.execute.__code__.co_varnames[:node.execute.__code__.co_argcount]))


# ── 2. numpy 数学对照 ──
# 2.1 MKL：本项目公式 vs mini-nodes apply_mkl（同输入同期望）
def mkl_mine(t_img, t_pixels, r_pixels):
    mu_t = t_pixels.mean(axis=0)
    mu_r = r_pixels.mean(axis=0)
    cov_t = np.cov(t_pixels, rowvar=False) + np.eye(3) * 1e-6
    cov_r = np.cov(r_pixels, rowvar=False) + np.eye(3) * 1e-6
    evals_t, evecs_t = np.linalg.eigh(cov_t)
    inv_sqrt_t = evecs_t @ np.diag(1.0 / np.sqrt(np.maximum(evals_t, 1e-6))) @ evecs_t.T
    evals_r, evecs_r = np.linalg.eigh(cov_r)
    sqrt_r = evecs_r @ np.diag(np.sqrt(np.maximum(evals_r, 0))) @ evecs_r.T
    transform = sqrt_r @ inv_sqrt_t
    out = (t_img - mu_t) @ transform.T + mu_r
    return np.clip(out, 0, 1)


def mkl_mini(t_img, t_pixels, r_pixels):
    # 从 comfyui-mini-nodes/nodes_color_match.py apply_mkl 复刻
    mu_t = np.mean(t_pixels, axis=0)
    mu_r = np.mean(r_pixels, axis=0)
    t_centered = t_pixels - mu_t
    r_centered = r_pixels - mu_r
    cov_t = np.cov(t_centered, rowvar=False) + np.eye(3) * 1e-6
    cov_r = np.cov(r_centered, rowvar=False) + np.eye(3) * 1e-6
    evals_t, evecs_t = np.linalg.eigh(cov_t)
    inv_sqrt_t = evecs_t @ np.diag(1.0 / np.sqrt(np.maximum(evals_t, 1e-6))) @ evecs_t.T
    evals_r, evecs_r = np.linalg.eigh(cov_r)
    sqrt_r = evecs_r @ np.diag(np.sqrt(np.maximum(evals_r, 0))) @ evecs_r.T
    t = sqrt_r @ inv_sqrt_t
    out = (t_img - mu_t) @ t.T + mu_r
    return np.clip(out, 0, 1)


rng = np.random.default_rng(42)
t_pix = rng.random((500, 3))
r_pix = rng.random((600, 3)) * 0.5 + np.array([0.6, 0.2, 0.9])
t_img = rng.random((8, 8, 3))
out_mine = mkl_mine(t_img, t_pix, r_pix)
out_mini = mkl_mini(t_img, t_pix, r_pix)
check("MKL: 与 mini apply_mkl 输出一致", np.allclose(out_mine, out_mini, atol=1e-6))

# MKL mask 像素提取：HWC 域广播 bool 索引 ≡ mini 的 mask>0.1 采样（逐帧）
img_np = rng.random((2, 3, 10, 10))
mask_np = (rng.random((2, 1, 10, 10)) > 0.4).astype(np.float32)
img_hwc = img_np.transpose(0, 2, 3, 1)  # [B,H,W,C]
mask_b_hwc = np.broadcast_to(mask_np.transpose(0, 2, 3, 1) > 0.1, img_hwc.shape)
pix_mine = img_hwc[mask_b_hwc].reshape(-1, 3)
pix_mini = np.concatenate([img_np[i].transpose(1, 2, 0)[mask_np[i, 0] > 0.1] for i in range(2)], axis=0)
check("MKL: mask 像素提取与 mini 采样等价", pix_mine.shape == pix_mini.shape and np.allclose(pix_mine, pix_mini))

# 2.2 除零语义：mask 区域 std=0 而全图存在偏离像素 -> (x-mean)/0 = inf
#    新行为 posinf=0 -> 收敛参考均值；旧行为 nan_to_num 默认 inf->最大有限值
x = np.array([0.5, 0.6, 0.7])  # mask 区域均值 0.5、std 0；全图有偏离像素
ref_mean, ref_std = 0.7, 0.2
ratio = (x - 0.5) / 0.0  # [nan, inf, inf]
old = np.nan_to_num(ratio)  # 旧行为：nan->0, inf->1.79e308
new = np.nan_to_num(ratio, posinf=0.0, neginf=0.0)  # 新行为：inf -> 0
matched_new = np.clip(new * ref_std + ref_mean, 0, 1)
check("除零: std=0 时收敛参考均值且有限", np.allclose(matched_new, ref_mean) and np.isfinite(matched_new).all())
check("除零: 旧行为确为 inf->大数（证实修复必要性）", np.allclose(old[1], 1.7976931348623157e308))

# 2.3 多帧合并方差 == 全局合并统计（有偏 std，与公式假设一致）
#    execute 语义：所有帧的 E[x²] 直接帧间平均（concat 后平均），非帧对齐相加
a = rng.random((2, 3, 8, 8))
b = rng.random((2, 3, 8, 8))
ma, sa = a.mean(axis=(2, 3), keepdims=True), a.std(axis=(2, 3), keepdims=True, ddof=0)
mb, sb = b.mean(axis=(2, 3), keepdims=True), b.std(axis=(2, 3), keepdims=True, ddof=0)
E_all = np.concatenate([sa ** 2 + ma ** 2, sb ** 2 + mb ** 2], axis=0).mean(axis=0, keepdims=True)
mean_all = np.concatenate([ma, mb], axis=0).mean(axis=0, keepdims=True)
merged_var = E_all - mean_all ** 2
c = np.concatenate([a, b], axis=0)  # 4 帧
mc = c.mean(axis=(2, 3), keepdims=True).mean(axis=0, keepdims=True)  # 全局均值（帧间等权）
sc2 = (c ** 2).mean(axis=(2, 3), keepdims=True).mean(axis=0, keepdims=True)  # 全局 E[x²]
check("合并方差: 多帧聚合 == 全局方差", np.allclose(np.sqrt(merged_var), np.sqrt(sc2 - mc ** 2), atol=1e-6))

# 2.4 Mean 公式
t_mean, r_mean = np.array([0.3, 0.4, 0.5]), np.array([0.6, 0.5, 0.4])
check("Mean: 平移公式", np.allclose(np.array([0.3, 0.4, 0.5]) + (r_mean - t_mean), [0.6, 0.5, 0.4]))


# ── 3. execute 集成（FakeTensor 流程）──
def run(method, batch=2, ref_batch=2, masks=None, color_space="LAB", strength=1.0, batch_size=0):
    nan_to_num_calls.clear()
    target_image = FakeTensor((batch, 16, 16, 3))
    reference_image = FakeTensor((ref_batch, 24, 24, 3))
    kw = dict(masks or {})
    return node.execute(target_image, reference_image, color_space, strength, device="auto",
                        batch_size=batch_size, method=method, **kw)[0]


out = run("Statistical", batch=2, ref_batch=2)
check("Statistical: 输出 shape [B,H,W,C]", out.shape == (2, 16, 16, 3))

out = run("Statistical", batch=2, ref_batch=2, masks={"target_mask": FakeTensor((1, 16, 16))})
check("Statistical+target_mask: 分支内软混合输出 shape", out.shape == (2, 16, 16, 3))

out = run("Mean", batch=2, ref_batch=2)
check("Mean: 输出 shape", out.shape == (2, 16, 16, 3))

out = run("MKL", batch=2, ref_batch=2)
check("MKL: 输出 shape", out.shape == (2, 16, 16, 3))

out = run("Wavelet", batch=2, ref_batch=2)
check("Wavelet: 输出 shape", out.shape == (2, 16, 16, 3))

out = run("Statistical", batch=4, ref_batch=4, batch_size=2)
check("batch_size=2: 分批输出 shape", out.shape == (4, 16, 16, 3))

out = run("MKL", batch=2, ref_batch=1, masks={"reference_mask": FakeTensor((1, 16, 16)), "target_mask": FakeTensor((1, 16, 16)), "target_sample_mask": FakeTensor((1, 16, 16))})
check("mask 全接入: 多帧 image + 单帧 mask 广播不炸（旧版 ref assert 会失败）", out.shape == (2, 16, 16, 3))

# 4.1 除零参数：matched 除法处的 nan_to_num 必须带 posinf/neginf=0
#    （聚合后的 nan_to_num 无参数属设计，仅修复 LUV nan 传播）
run("Statistical", batch=2, ref_batch=2)
check("除零: nan_to_num 传 posinf/neginf=0",
      any(kw.get("posinf") == 0.0 and kw.get("neginf") == 0.0 for kw in nan_to_num_calls))

# 4.2 target_sample_mask 传入目标统计（Statistical 分支）
seen_sample_mask = []


def wrap_compute(orig):
    def wrapped(tensor, mask=None):
        seen_sample_mask.append(mask)
        return orig(tensor, mask)
    return wrapped


node.compute_mean_std = wrap_compute(node.compute_mean_std)
run("Statistical", batch=2, ref_batch=2, masks={"target_sample_mask": FakeTensor((1, 16, 16))})
check("target_sample_mask: 目标统计收到非 None 采样遮罩", any(m is not None for m in seen_sample_mask))

# 4.3 MKL 分支 target_sample_mask 同样生效（走 _mkl_components，不调 compute_mean_std）
seen_mkl_mask = []


def wrap_mkl(orig):
    def wrapped(tensor, mask):
        seen_mkl_mask.append(mask)
        return orig(tensor, mask)
    return wrapped


node._mkl_components = wrap_mkl(node._mkl_components)
run("MKL", batch=2, ref_batch=2, masks={"target_sample_mask": FakeTensor((1, 16, 16))})
check("target_sample_mask: MKL 统计收到非 None 采样遮罩", any(m is not None for m in seen_mkl_mask))
node._mkl_components = mod.ImageColorMatch._mkl_components

# 4.4 Wavelet: ref_low 尺寸对齐（common_upscale 到目标宽高）+ 多帧聚合
calls = []
orig_upscale = comfy.utils.common_upscale
comfy.utils.common_upscale = lambda t, w, h, **k: (calls.append((w, h)), t)[1]
run("Wavelet", batch=2, ref_batch=3)
check("Wavelet: ref_low 缩放到目标尺寸", calls and calls[0] == (16, 16))
comfy.utils.common_upscale = orig_upscale

# 4.5 多帧参考聚合：Statistical 合并方差后 ref 统计为单帧（shape (1,3,1,1)）
seen_ref_shape = []


def wrap_compute_ref(orig):
    def wrapped(tensor, mask=None):
        r = orig(tensor, mask)
        if tensor.shape[0] == 3:
            seen_ref_shape.append((tensor.shape[0],))
        return r
    return wrapped


node.compute_mean_std = wrap_compute_ref(node.compute_mean_std)
node.execute(FakeTensor((2, 16, 16, 3)), FakeTensor((3, 24, 24, 3)), "LAB", 1.0, "auto", 0, "Statistical")
check("多帧 ref: 统计接收全部帧（3 帧）", seen_ref_shape and seen_ref_shape[0] == (3,))
node.compute_mean_std = mod.ImageColorMatch.compute_mean_std

# ── 汇总 ──
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
