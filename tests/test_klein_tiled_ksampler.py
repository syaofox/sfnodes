# SFKleinTiledKSampler 后端逻辑测试（Python 直接运行：python tests/test_klein_tiled_ksampler.py）
# 覆盖：
#   - 结构：类名 / CATEGORY / DESCRIPTION / INPUT_TYPES（键集、默认值、tooltip、
#     sampler/scheduler combo）/ RETURN_TYPES / OUTPUT_TOOLTIPS / FUNCTION
#   - _get_tile_positions：tile 大于图钳制为单块、网格 stride、末尾边界钳制、
#     overlap > tile 时 stride=1 且去重
#   - _sort_tiles_by_content：内容丰富度降序、等分保持原序
#   - _crop_conditioning_refs / _merge_conditioning_refs：reference_latents 裁剪/
#     batch 合并，无 reference_latents 直通、其余键保留、原 ref 不被改动
#   - _scale_conditioning_refs：ref 缩放到目标尺寸
#   - _make_weight_mask：形状 / 中心峰值 1.0 / 角点最小 / 对称
#   - _match_color_stats：strength 0 直通、1 完全对齐（均值+标准差）、0.5 插值、
#     常量通道（std≈0）跳过
#   - _process_tile / _process_tile_pair：噪声 blend、ref 裁剪到 cpu、cond 裁剪/
#     合并、comfy.sample.sample 参数透传、callback 推进内部 pbar、pair batch=2 拆分
#   - sample() 集成：单 tile 直通、多 tile 两两并行（batch=2）+ 奇数尾 tile 单跑、
#     B=2 全单跑、blend 归一化与噪声混合（对照 numpy 参考）、权重归一化（常量生成
#     → 结果处处=常量）、单覆盖像素精确等于参考值、色彩对齐、种子确定性
# mock：torch / torch.nn.functional / comfy.* / latent_preview（numpy 本机真实可用）
import ast
import importlib.util
import os
import sys
import types

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── FakeTensor：numpy 代理（float32，视图语义支持 result[slice] += x）──
class FakeTensor:
    device = "cpu"

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data

    def clone(self):
        return FakeTensor(self.data.copy())

    def to(self, device=None, dtype=None):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, axis=dim))

    def expand(self, *dims):
        # torch expand：-1 表示保持原尺寸
        d = list(dims)
        if len(d) != self.data.ndim:
            d = d[-self.data.ndim:]
        shape = tuple(self.data.shape[i] if x == -1 else x for i, x in enumerate(d))
        return FakeTensor(np.broadcast_to(self.data, shape))

    def clamp(self, min=None, max=None):
        return FakeTensor(np.clip(self.data, a_min=min, a_max=max))

    # torch 语义：std/var 默认 unbiased（ddof=1），与 numpy 默认 ddof=0 不同
    def std(self, dim=None):
        if dim is None:
            return float(np.std(self.data, ddof=1))
        return FakeTensor(np.std(self.data, axis=dim, ddof=1))

    def var(self, dim=None):
        if dim is None:
            return float(np.var(self.data, ddof=1))
        return FakeTensor(np.var(self.data, axis=dim, ddof=1))

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

    def __iadd__(self, o):
        self.data += o.data if isinstance(o, FakeTensor) else o
        return self

    def __add__(self, o):
        return FakeTensor(self.data + (o.data if isinstance(o, FakeTensor) else o))

    def __radd__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) + self.data)

    def __sub__(self, o):
        return FakeTensor(self.data - (o.data if isinstance(o, FakeTensor) else o))

    def __rsub__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) - self.data)

    def __mul__(self, o):
        return FakeTensor(self.data * (o.data if isinstance(o, FakeTensor) else o))

    def __rmul__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) * self.data)

    def __truediv__(self, o):
        return FakeTensor(self.data / (o.data if isinstance(o, FakeTensor) else o))

# ── mock torch ──
torch = types.ModuleType("torch")
torch.Tensor = FakeTensor
torch.float32 = np.float32
torch.float64 = np.float64


def _tshape(s):
    return s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s


torch.zeros = lambda *s, **k: FakeTensor(np.zeros(
    tuple(int(x) for x in _tshape(s)), dtype=np.float32))
torch.arange = lambda *a, **k: FakeTensor(np.arange(*a))
torch.cat = lambda seq, dim=0: FakeTensor(np.concatenate(
    [s.data if isinstance(s, FakeTensor) else s for s in seq], axis=dim))
torch.min = lambda a, b=None: FakeTensor(np.minimum(
    a.data if isinstance(a, FakeTensor) else a,
    b.data if isinstance(b, FakeTensor) else b)) if b is not None else float(np.min(a.data))


class FakeGenerator:
    def __init__(self, device=None):
        self._seed = None

    def manual_seed(self, s):
        self._seed = s


torch.Generator = FakeGenerator


def fake_randn(*s, **k):
    shape = tuple(int(x) for x in _tshape(s))
    gen = k.get("generator")
    seed = gen._seed if gen is not None else 0
    return FakeTensor(np.random.default_rng(seed).standard_normal(shape))


torch.randn = fake_randn
sys.modules["torch"] = torch

torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")


def fake_interpolate(x, size=None, mode=None, align_corners=None):
    a = x.data if isinstance(x, FakeTensor) else x
    if size is None or (a.shape[2] == size[0] and a.shape[3] == size[1]):
        return FakeTensor(a)
    b, c, h, w = a.shape
    th, tw = size
    out = np.empty((b, c, th, tw), dtype=a.dtype)
    for bi in range(b):
        for ci in range(c):
            out[bi, ci] = np.resize(a[bi, ci], (th, tw))
    return FakeTensor(out)


torch.nn.functional.interpolate = fake_interpolate
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional

# ── mock comfy / latent_preview ──
comfy = types.ModuleType("comfy"); comfy.__path__ = []
comfy_samplers = types.ModuleType("comfy.samplers")
comfy_sample = types.ModuleType("comfy.sample")
comfy_mm = types.ModuleType("comfy.model_management")
comfy_utils = types.ModuleType("comfy.utils")
comfy.samplers = comfy_samplers
comfy.sample = comfy_sample
comfy.model_management = comfy_mm
comfy.utils = comfy_utils
sys.modules["comfy"] = comfy
sys.modules["comfy.samplers"] = comfy_samplers
sys.modules["comfy.sample"] = comfy_sample
sys.modules["comfy.model_management"] = comfy_mm
sys.modules["comfy.utils"] = comfy_utils

comfy_samplers.KSampler = types.SimpleNamespace(
    SAMPLERS=("euler", "euler_ancestral"), SCHEDULERS=("normal", "karras"))
comfy_mm.get_torch_device = lambda: "cpu"

PB_INSTANCES = []


class FakeProgressBar:
    def __init__(self, total, node_id=None):
        self.total = total
        self.updates = []
        PB_INSTANCES.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total, preview))


comfy_utils.ProgressBar = FakeProgressBar


class SampleRecorder:
    """记录 comfy.sample.sample 调用；factory(noise) -> 生成结果（默认恒等）。"""

    def __init__(self, factory=None):
        self.calls = []
        self.factory = factory or (lambda noise: noise.clone())

    def __call__(self, m, noise, steps, cfg, sampler_name, scheduler, positive,
                 negative, latent_image, denoise=1.0, disable_noise=False,
                 start_step=None, last_step=None, force_full_denoise=False,
                 noise_mask=None, sigmas=None, callback=None,
                 disable_pbar=False, seed=None):
        self.calls.append({
            "model": m, "noise": noise, "steps": steps, "cfg": cfg,
            "sampler": sampler_name, "scheduler": scheduler,
            "positive": positive, "negative": negative,
            "ref": latent_image, "denoise": denoise, "seed": seed,
            "callback": callback,
        })
        return self.factory(noise)


# comfy.sample.sample 代理：转发到当前 recorder（每个测试场景可换）
_current_rec = None


def _sample_proxy(*a, **k):
    return _current_rec(*a, **k)


comfy_sample.sample = _sample_proxy


latent_preview = types.ModuleType("latent_preview")
latent_preview.get_previewer = lambda device, latent_format: None
sys.modules["latent_preview"] = latent_preview

# ── 加载节点模块 ──
spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.latent.klein_tiled_ksampler",
    os.path.join(root, "nodes", "latent", "klein_tiled_ksampler.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

Node = mod.SFKleinTiledKSampler
node = Node()

# ── 1. 结构 ──
check("CATEGORY", node.CATEGORY == "sfnodes/latent")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("RETURN_TYPES", node.RETURN_TYPES == ("LATENT",) and node.RETURN_NAMES == ("latent",))
check("OUTPUT_TOOLTIPS 非空", len(node.OUTPUT_TOOLTIPS) == 1 and len(node.OUTPUT_TOOLTIPS[0]) > 0)
check("FUNCTION = sample", node.FUNCTION == "sample")

it = node.INPUT_TYPES()
required = it["required"]
check("required 键集",
      set(required.keys()) == {
          "model", "positive", "negative", "latent_image", "latent_blend",
          "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
          "tile_width", "tile_height", "overlap", "blend_strength",
          "color_preserve"})
check("seed 上限 64bit", required["seed"][1]["max"] == 0xffffffffffffffff)
check("tile_width 默认 512/step 8", required["tile_width"][1]["default"] == 512
      and required["tile_width"][1]["step"] == 8)
check("overlap 默认 128", required["overlap"][1]["default"] == 128)
check("blend_strength 默认 0.3", required["blend_strength"][1]["default"] == 0.3)
check("color_preserve 默认 0.1", required["color_preserve"][1]["default"] == 0.1)
check("sampler/scheduler combo", required["sampler_name"][0] == ("euler", "euler_ancestral")
      and required["scheduler"][0] == ("normal", "karras"))
for k in required:
    check(f"输入 {k} 有 tooltip", isinstance(required[k][1].get("tooltip"), str)
          and len(required[k][1]["tooltip"]) > 0)

# 根 __init__.py 注册键一致（AST 解析两个字典）
with open(os.path.join(root, "__init__.py"), encoding="utf-8") as f:
    init_src = f.read()
tree = ast.parse(init_src)
mapping_keys = {}
for astnode in ast.walk(tree):
    if isinstance(astnode, ast.Assign):
        for t in astnode.targets:
            if isinstance(t, ast.Name) and t.id in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
                if isinstance(astnode.value, ast.Dict):
                    mapping_keys[t.id] = {ast.literal_eval(k) for k in astnode.value.keys if k is not None}
check("__init__ 注册 SFKleinTiledKSampler 双字典一致",
      "SFKleinTiledKSampler" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFKleinTiledKSampler" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. _get_tile_positions ──
pos = node._get_tile_positions(8, 8, 16, 16, 4)
check("tile 大于图 → 单块钳制", pos == [(0, 0, 8, 8)])

pos = node._get_tile_positions(16, 16, 8, 8, 0)
check("2x2 网格（stride=tile）", sorted(pos) == [
    (0, 0, 8, 8), (0, 8, 8, 8), (8, 0, 8, 8), (8, 8, 8, 8)])

pos = node._get_tile_positions(100, 100, 64, 64, 16)
ys = sorted({p[0] for p in pos}); xs = sorted({p[1] for p in pos})
# stride=48，y=48 时 y0=min(48,36)=36（先钳制再判 break）→ 网格 {0,36}
check("末尾行/列钳制到边界", ys == [0, 36] and xs == [0, 36]
      and len(pos) == 4)
check("所有 tile 不越界", all(p[0] + p[2] <= 100 and p[1] + p[3] <= 100 for p in pos))

pos = node._get_tile_positions(70, 70, 64, 64, 0)
check("钳制去重（y=64 与 y=128 同钳制到 6）", len(pos) == 4
      and sorted({p[0] for p in pos}) == [0, 6])

pos = node._get_tile_positions(16, 16, 8, 8, 32)
ys = sorted({p[0] for p in pos})
check("overlap>tile → stride=1 全覆盖", ys == list(range(9))
      and len(pos) == 9 * 9)

# ── 3. _sort_tiles_by_content ──
blend_up = FakeTensor(np.zeros((1, 4, 4, 4)))
blend_up.data[0, :, 0:2, 0:2] = np.arange(4).reshape(2, 2)  # 左上象限高方差
positions = [(0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2)]
ordered = node._sort_tiles_by_content(positions, blend_up)
check("内容丰富度降序", ordered[0] == (0, 0, 2, 2))
check("排序后仍是同一集合", sorted(ordered) == sorted(positions))

ordered_eq = node._sort_tiles_by_content(
    [(0, 0, 2, 2), (0, 2, 2, 2)], FakeTensor(np.zeros((1, 4, 4, 4))))
check("等分保持原序", ordered_eq == [(0, 0, 2, 2), (0, 2, 2, 2)])

# ── 4. conditioning refs 裁剪/合并/缩放 ──
ref_full = FakeTensor(np.random.default_rng(1).standard_normal((1, 4, 16, 16)))
emb = FakeTensor(np.zeros((1, 77, 2048)))
cond = [
    [emb, {"reference_latents": [ref_full], "pooled": "P"}],
    [emb, {"cross_attn": 1}],
]
cropped = node._crop_conditioning_refs(cond, 4, 4, 8, 8)
check("裁剪 shape", cropped[0][1]["reference_latents"][0].shape == (1, 4, 8, 8))
check("裁剪后其余键保留", cropped[0][1]["pooled"] == "P")
check("无 reference_latents 直通", cropped[1][1] == {"cross_attn": 1})
check("原 ref 未被改动", ref_full.shape == (1, 4, 16, 16))

cond_a = [[emb, {"reference_latents": [FakeTensor(np.zeros((1, 4, 8, 8)))]}]]
cond_b = [[emb, {"reference_latents": [FakeTensor(np.ones((1, 4, 8, 8)))]}]]
merged = node._merge_conditioning_refs(cond_a, cond_b)
check("合并 batch 维", merged[0][1]["reference_latents"][0].shape == (2, 4, 8, 8))
check("合并内容 a 在上 b 在下",
      merged[0][1]["reference_latents"][0].data[0].sum() == 0.0
      and merged[0][1]["reference_latents"][0].data[1].sum() == 8 * 8 * 4)
merged_noref = node._merge_conditioning_refs(
    [[emb, {}]], [[emb, {"x": 2}]])
check("无 reference_latents 合并直通", merged_noref[0][1] == {})

scaled = node._scale_conditioning_refs(cond, 8, 8)
check("缩放 reference_latents 到目标尺寸",
      scaled[0][1]["reference_latents"][0].shape == (1, 4, 8, 8)
      and scaled[1][1] == {"cross_attn": 1})

# ── 5. _make_weight_mask ──
w = node._make_weight_mask(4, 4, "cpu")
check("权重形状 [1,1,h,w]", w.shape == (1, 1, 4, 4))
check("中心峰值 1.0", w.data.max() == 1.0 and w.data[0, 0, 1, 1] == 1.0)
check("角点最小且对称",
      w.data[0, 0, 0, 0] == w.data[0, 0, -1, -1] == w.data[0, 0, 0, -1] == 0.25
      and w.data[0, 0, 0, 0] < w.data[0, 0, 1, 1])
check("权重全为正", (w.data > 0).all())

# ── 6. _match_color_stats ──
rng = np.random.default_rng(2)
res = FakeTensor(rng.standard_normal((1, 3, 8, 8)))
orig = FakeTensor(rng.standard_normal((1, 3, 8, 8)) * 2.0 + 5.0)

out0 = node._match_color_stats(res, orig, 0.0)
check("strength=0 直通（同一对象）", out0 is res)

out1 = node._match_color_stats(res, orig, 1.0)
ok = True
for c in range(3):
    r = res.data[0, c].reshape(-1)
    o = orig.data[0, c].reshape(-1)
    adj = (r - r.mean()) / r.std(ddof=1) * o.std(ddof=1) + o.mean()
    ok = ok and np.allclose(out1.data[0, c].reshape(-1), adj, atol=1e-5)
check("strength=1 完全对齐（对照 numpy 参考）", ok)

out05 = node._match_color_stats(res, orig, 0.5)
ok = True
for c in range(3):
    r = res.data[0, c].reshape(-1)
    o = orig.data[0, c].reshape(-1)
    adj = (r - r.mean()) / r.std(ddof=1) * o.std(ddof=1) + o.mean()
    expected = r * 0.5 + adj * 0.5
    ok = ok and np.allclose(out05.data[0, c].reshape(-1), expected, atol=1e-5)
check("strength=0.5 线性插值", ok)

const_res = FakeTensor(np.ones((1, 2, 4, 4)) * 3.0)
const_res.data[0, 1] = np.arange(16).reshape(4, 4)
out_const = node._match_color_stats(const_res, orig[:, :2], 1.0)
check("常量通道（std≈0）跳过保持原值",
      (out_const.data[0, 0] == 3.0).all()
      and not np.allclose(out_const.data[0, 1], 3.0))

# ── 7. _process_tile（直接调用） ──
def make_grad(shape):
    return FakeTensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100.0)


m = object()
samples = make_grad((1, 4, 16, 16))
global_noise = FakeTensor(rng.standard_normal((1, 4, 16, 16)))
blend_up = FakeTensor(np.full((1, 4, 16, 16), 0.25, dtype=np.float32))
cond = [[emb, {"reference_latents": [samples.clone()]}]]

rec = SampleRecorder()
_current_rec = rec
PB_INSTANCES.clear()
res = node._process_tile(
    m, cond, cond, samples, global_noise, blend_up,
    4, 4, 8, 8, 1, 0.3, 4, 1.0, "euler", "normal", 1.0, 7, None, "cpu")

call = rec.calls[0]
check("_process_tile 调用 1 次", len(rec.calls) == 1)
expected_noise = global_noise.data[0, :, 4:12, 4:12] * 0.7 + blend_up.data[0, :, 4:12, 4:12] * 0.3
check("噪声按 blend_strength 混合", np.allclose(call["noise"].data[0], expected_noise, atol=1e-5))
check("ref 裁剪", np.allclose(call["ref"].data, samples.data[0, :, 4:12, 4:12], atol=1e-5))
check("positive refs 裁剪", call["positive"][0][1]["reference_latents"][0].shape == (1, 4, 8, 8))
check("参数透传", (call["steps"], call["cfg"], call["sampler"], call["scheduler"],
      call["denoise"], call["seed"]) == (4, 1.0, "euler", "normal", 1.0, 7))
check("callback 可调用", callable(call["callback"]))
check("返回 .to(device) 结果", np.allclose(res.data, call["noise"].data, atol=1e-6))
call["callback"](0, None, None, 4)
check("callback 推进内部 pbar", PB_INSTANCES[-1].updates == [(1, 4, None)])

# ── 8. _process_tile_pair（直接调用） ──
rec2 = SampleRecorder()
_current_rec = rec2
res_a, res_b = node._process_tile_pair(
    m, cond, cond, samples, global_noise, blend_up,
    0, 0, 8, 8, 0, 8, 8, 8, 1, 0.3, 4, 1.0, "euler", "normal", 1.0, 7, None, "cpu")

call2 = rec2.calls[0]
check("pair 调用 1 次且 noise batch=2", len(rec2.calls) == 1
      and call2["noise"].shape == (2, 4, 8, 8))
na = global_noise.data[0, :, 0:8, 0:8] * 0.7 + blend_up.data[0, :, 0:8, 0:8] * 0.3
nb = global_noise.data[0, :, 0:8, 8:16] * 0.7 + blend_up.data[0, :, 0:8, 8:16] * 0.3
check("pair 噪声 a/b 顺序", np.allclose(call2["noise"].data[0], na, atol=1e-5)
      and np.allclose(call2["noise"].data[1], nb, atol=1e-5))
check("pair ref batch=2", call2["ref"].shape == (2, 4, 8, 8))
check("pair positive refs 合并 batch=2",
      call2["positive"][0][1]["reference_latents"][0].shape == (2, 4, 8, 8))
check("pair 结果按 B 拆分", res_a.shape == (1, 4, 8, 8) and res_b.shape == (1, 4, 8, 8)
      and np.allclose(res_a.data, call2["noise"].data[0:1], atol=1e-6)
      and np.allclose(res_b.data, call2["noise"].data[1:2], atol=1e-6))

# ── 9. sample() 集成 ──
fake_model = types.SimpleNamespace(model=types.SimpleNamespace(latent_format="fake"))

def run_sample(B, H, W, tile_w, tile_h, overlap, blend_data, seed=123,
               color_preserve=0.0, factory=None):
    global _current_rec
    rec = SampleRecorder(factory)
    _current_rec = rec
    latent = {"samples": FakeTensor(rng.standard_normal((B, 4, H, W)))}
    blend = {"samples": FakeTensor(np.asarray(blend_data, dtype=np.float32))}
    out = node.sample(
        fake_model, cond, cond, latent, blend,
        seed=seed, steps=4, cfg=1.0, sampler_name="euler", scheduler="normal",
        denoise=1.0, tile_width=tile_w, tile_height=tile_h, overlap=overlap,
        blend_strength=0.3, color_preserve=color_preserve)
    return rec, out[0]["samples"], latent["samples"].data

# 9.1 单 tile：结果 == 混合噪声（对照 numpy 参考），seed 确定性
blend_grad = np.linspace(0.0, 1.0, 4 * 8 * 8, dtype=np.float32).reshape(1, 4, 8, 8)
rec, out, _ = run_sample(1, 8, 8, 512, 512, 0, blend_grad)
check("单 tile 调用 1 次", len(rec.calls) == 1)
global_ref = np.random.default_rng(123).standard_normal((1, 4, 8, 8))
blend_norm = (blend_grad - 0.0) / (1.0 - 0.0) * 2.0 - 1.0
expected = global_ref * 0.7 + blend_norm * 0.3
check("单 tile 结果 == 全局噪声*0.7 + 归一化 blend*0.3",
      np.allclose(out.data, expected, atol=1e-4))
check("单 tile 调用参数（steps/cfg/seed 透传）",
      (rec.calls[0]["steps"], rec.calls[0]["cfg"], rec.calls[0]["seed"]) == (4, 1.0, 123)
      and rec.calls[0]["noise"].shape == (1, 4, 8, 8))
check("单 tile 返回 latent 形状", out.shape == (1, 4, 8, 8))

rec2_, out2, _ = run_sample(1, 8, 8, 512, 512, 0, blend_grad)
check("种子确定性（两次运行结果一致）", np.array_equal(out.data, out2.data))

# 9.2 多 tile：3 块（两两并行 1 次 batch=2 + 尾 tile 单跑）
rec, out, _ = run_sample(1, 16, 16, 64, 128, 0, np.full((1, 4, 16, 16), 0.5))
check("3 tile → 1 次 batch=2 + 1 次单跑", len(rec.calls) == 2)
check("pair 调用 noise batch=2", rec.calls[0]["noise"].shape[0] == 2)
check("单 tile 调用 noise batch=1", rec.calls[1]["noise"].shape[0] == 1)
# tile (0,0,16,8) 与 (0,7,16,8) 尺寸相同 → pair；(0,8,16,8) 单跑
g = np.random.default_rng(123).standard_normal((1, 4, 16, 16))
blend_norm2 = np.full((1, 4, 16, 16), 0.5)
# 列 0-6 仅被 tile(0,0,16,8) 覆盖 → 权重归一化后结果 == 该 tile 的生成值（恒等工厂）
check("单覆盖列精确等于参考", np.allclose(
    out.data[:, :, :, 0:7], g[:, :, :, 0:7] * 0.7 + blend_norm2[:, :, :, 0:7] * 0.3,
    atol=1e-4))
check("多 tile 结果形状", out.shape == (1, 4, 16, 16))

# 9.3 常量生成 → 权重归一化处处=常量
rec, out, _ = run_sample(1, 16, 16, 64, 128, 0, np.full((1, 4, 16, 16), 0.5),
                      factory=lambda noise: FakeTensor(np.ones_like(noise.data)))
check("权重归一化：常量生成 → 结果处处=1.0", np.allclose(out.data, 1.0, atol=1e-5))

# 9.4 B=2 不并行（全单跑，noise batch = B = 2）
rec, out, _ = run_sample(2, 16, 16, 64, 128, 0, np.full((1, 4, 16, 16), 0.5))
check("B=2 全单跑（3 次调用，无 batch=2 并行）", len(rec.calls) == 3
      and all(c["noise"].shape[0] == 2 for c in rec.calls))
check("B=2 blend batch 扩展", rec.calls[0]["noise"].shape == (2, 4, 16, 8)
      and out.shape == (2, 4, 16, 16))

# 9.5 色彩对齐（单 tile，非恒等生成，color_preserve=1.0）
rec, out, latent_ref = run_sample(1, 8, 8, 512, 512, 0, blend_grad, color_preserve=1.0,
                      factory=lambda noise: FakeTensor(noise.data + 2.0))
ok = True
for c in range(4):
    ok = ok and abs(out.data[0, c].mean() - latent_ref[0, c].mean()) < 1e-4
check("color_preserve=1 对齐原图通道均值", ok)

# 9.6 色彩对齐=0 保持生成值
rec, out, _ = run_sample(1, 8, 8, 512, 512, 0, blend_grad, color_preserve=0.0,
                      factory=lambda noise: FakeTensor(noise.data + 2.0))
check("color_preserve=0 不改变结果",
      np.allclose(out.data, np.random.default_rng(123).standard_normal((1, 4, 8, 8)) * 0.7
                  + blend_norm * 0.3 + 2.0, atol=1e-4))

# ── 汇总 ──
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
