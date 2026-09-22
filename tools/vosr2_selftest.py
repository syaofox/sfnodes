"""VOSR2 节点自检（需在 ComfyUI 运行环境内执行；不加载权重、不占 GPU）。

覆盖本机测试无法验证的部分：真实 torch/comfy 导入、节点 schema、LightningDiT 小模型
前向（含动态 RoPE）、Qwen VAE 编解码与分块一致性、色彩对齐、FakeBundle 端到端管线
（分块/整图/批/种子/非方形/视频时序缓存）以及 OOM 自动降级路径。

用法（宿主机工作副本同步到容器后执行，或用 docker cp 拷整个包）：

    docker exec -w /home/comfy/app comfyui-docker python3 \
        /path/in/container/sfnodes/tools/vosr2_selftest.py

可用环境变量：
    COMFYUI_ROOT  ComfyUI 源码根（含 comfy/、folder_paths.py），默认 /home/comfy/app

退出码：0 全部通过；1 存在失败；2 运行环境不满足（缺 torch/comfy）。
"""

import os
import sys
import types

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.environ.get("COMFYUI_ROOT", "/home/comfy/app")

sys.path.insert(0, COMFY_ROOT)

# 包骨架：直接指向工作副本，不执行根 __init__.py（避免连带加载全部节点与依赖）
if "sfnodes" not in sys.modules:
    _pkg = types.ModuleType("sfnodes")
    _pkg.__path__ = [PKG_ROOT]
    sys.modules["sfnodes"] = _pkg
for _name, _rel in (
    ("sfnodes.sf_utils", "sf_utils"),
    ("sfnodes.nodes", "nodes"),
    ("sfnodes.nodes.model", "nodes/model"),
    ("sfnodes.nodes.model.vosr2", "nodes/model/vosr2"),
    ("sfnodes.nodes.image", "nodes/image"),
    ("sfnodes.nodes.video", "nodes/video"),
):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [os.path.join(PKG_ROOT, _rel)]
        sys.modules[_name] = _m

try:
    import torch
    import torch.nn.functional as F

    import comfy  # noqa: F401
except ImportError as exc:
    print(f"运行环境不满足（需要 ComfyUI 运行时 + torch）: {exc}")
    sys.exit(2)

failures = []


def check(name, cond, extra=""):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name} {extra}")


# ── 1. 节点导入与 schema ──
from sfnodes.nodes.model.vosr2_loader import SFVOSR2ModelLoader
from sfnodes.nodes.model.vosr2_settings import SFVOSR2Settings
from sfnodes.nodes.image.vosr2_upscale import SFVOSR2Upscale
from sfnodes.nodes.video.vosr2_video import SFVOSR2Video

for cls, name in [
    (SFVOSR2ModelLoader, "SFVOSR2ModelLoader"),
    (SFVOSR2Settings, "SFVOSR2Settings"),
    (SFVOSR2Upscale, "SFVOSR2Upscale"),
    (SFVOSR2Video, "SFVOSR2Video"),
]:
    it = cls.INPUT_TYPES()
    method = getattr(cls, cls.FUNCTION, None)
    check(f"{name} INPUT_TYPES 可构建", "required" in it and bool(it["required"]))
    check(f"{name} FUNCTION 方法存在", callable(method))
    check(f"{name} CATEGORY/RETURN/DESCRIPTION",
          cls.CATEGORY.startswith("sfnodes/") and cls.RETURN_TYPES and cls.DESCRIPTION)

check("图片节点 settings 可选输入", "settings" in SFVOSR2Upscale.INPUT_TYPES().get("optional", {}))
check("视频节点 settings 可选输入", "settings" in SFVOSR2Video.INPUT_TYPES().get("optional", {}))

# ── 2. LightningDiT 小模型前向（RoPE + cross-attn + unpatchify）──
from sfnodes.nodes.model.vosr2.lightningdit import LightningDiT, vosr_attention_backend

torch.manual_seed(0)
dit = LightningDiT(
    input_size=8, patch_size=2, in_channels=32, out_channels=16, hidden_size=64,
    depth=2, num_heads=4, mlp_ratio=2.0, use_qknorm=True, use_swiglu=True,
    use_rope=True, use_rmsnorm=True, z_dims=32, encdim_ratio=2, num_fused_layers=1,
)
dit.eval()
# comfy.ops.disable_weight_init 不初始化权重（留给 checkpoint 加载），测试需自行填数
with torch.no_grad():
    for _n, _p in dit.named_parameters():
        _p.normal_(0, 0.02)
x = torch.randn(2, 32, 16, 16)
t = torch.full((2,), 1.0)
z = [torch.randn(2, 25, 32)]
with torch.no_grad():
    out = dit.forward_flexible(x, t, torch.zeros(2), z)
    out2 = dit.forward_flexible(x, t, torch.zeros(2), z)
check("DiT 前向输出形状", out.shape == (2, 16, 16, 16), out.shape)
check("DiT 前向确定性", torch.allclose(out, out2))
check("注意力后端可查询", vosr_attention_backend() in ("sage", "sdpa"), vosr_attention_backend())

x2 = torch.randn(1, 32, 24, 24)  # 非训练分辨率 → 动态 RoPE 路径
with torch.no_grad():
    o1 = dit.forward_flexible(x2, torch.full((1,), 1.0), torch.zeros(1), [torch.randn(1, 36, 32)])
check("动态 RoPE 非训练分辨率前向", o1.shape == (1, 16, 24, 24), o1.shape)

# ── 3. Qwen VAE 小模型 encode/decode + 分块一致性 ──
from sfnodes.nodes.model.vosr2.qwenimage_vae2d import AutoencoderKLQwenImage2D
from sfnodes.nodes.model.vosr2 import tiled_vae

vae = AutoencoderKLQwenImage2D(
    base_dim=8, z_dim=4, dim_mult=(1, 2, 4, 4), num_res_blocks=1,
    latents_mean=[0.0, 0.0, 0.0, 0.0], latents_std=[1.0, 1.0, 1.0, 1.0],
).eval()
with torch.no_grad():
    for _n, _p in vae.named_parameters():
        _p.normal_(0, 0.02)
img = torch.rand(1, 3, 64, 64) * 2 - 1
with torch.no_grad():
    lat, mean, std = tiled_vae.encode_latent(vae, img)
    dec = tiled_vae.decode_latent(vae, lat, mean, std)
check("VAE 单次编码形状", lat.shape == (1, 4, 8, 8), lat.shape)
check("VAE 单次解码形状", dec.shape == (1, 3, 64, 64), dec.shape)

with torch.no_grad():
    lat_t, mean_t, std_t = tiled_vae.tiled_encode_latent(vae, img, 32, 8)
    dec_t = tiled_vae.tiled_decode_latent(vae, lat_t, mean_t, std_t, 32, 8)
    lat_d, mean_d, std_d = tiled_vae.encode_dispatch(vae, img, 0, 0)
    dec_d = tiled_vae.decode_dispatch(vae, lat_d, mean_d, std_d, 0, 0)
check("VAE 分块编码形状一致", lat_t.shape == lat.shape, lat_t.shape)
check("VAE 分块解码形状一致", dec_t.shape == dec.shape, dec_t.shape)
check("VAE 分块 dispatch 直通", torch.allclose(lat_d, lat, atol=1e-5))
check("VAE 分块解码与整图量级一致", (dec_t - dec).abs().mean().item() < 0.2)

# ── 4. 色彩对齐 ──
from sfnodes.nodes.model.vosr2.color_fix import apply_color_alignment

tgt = torch.rand(2, 3, 32, 32)
src = torch.rand(2, 3, 32, 32)
for mode in ("none", "adain", "wavelet"):
    r = apply_color_alignment(tgt, src, mode)
    check(f"色彩对齐 {mode}", r.shape == tgt.shape and float(r.min()) >= 0.0 and float(r.max()) <= 1.0)
check("色彩对齐 wavelet downsample",
      apply_color_alignment(tgt, src, "wavelet", 2).shape == tgt.shape)

# ── 5. FakeBundle 端到端管线（分块 / 整图 / 批 / 种子 / 非方形）──
from sfnodes.nodes.model.vosr2.inferencer import DinoTemporalCache, VOSR2Inferencer
from sfnodes.nodes.model.vosr2.settings import VOSR2Settings
from sfnodes.nodes.model.vosr2.sizing import TargetSizeSpec


class _FakePatcher:
    load_device = torch.device("cpu")
    model = types.SimpleNamespace(pos_embed=torch.zeros(1, dtype=torch.float32))


class FakeBundle:
    dit_patcher = _FakePatcher()
    vae_patcher = _FakePatcher()
    dino_patcher = _FakePatcher()

    def encode(self, x, tile_size=0, tile_overlap=0, amp=False):
        z = F.avg_pool2d(x, 8)
        lat = z[:, :1].repeat(1, 4, 1, 1)
        return lat, torch.zeros(1, 4, 1, 1), torch.ones(1, 4, 1, 1)

    def decode(self, lat, mean, std, tile_size=0, tile_overlap=0, amp=False):
        up = F.interpolate(lat[:, :1], scale_factor=8, mode="nearest").repeat(1, 3, 1, 1)
        return up * 2 - 1

    def vision_features(self, crops):
        b = crops.shape[0]
        base = crops.mean(dim=(1, 2, 3), keepdim=True).reshape(b, 1, 1)
        return [base.repeat(1, 36, 32)]

    def dit_velocity(self, inp, t_cur, t_next, venc):
        return torch.tanh(inp[:, : inp.shape[1] // 2])

    def denoise_one_step(self, lq, noise, venc):
        return noise * 0.5

    def prepare_vae_decode(self):
        pass

    def clear_staged(self):
        pass


inf = VOSR2Inferencer(FakeBundle())

images = torch.rand(2, 128, 128, 3)
s_tiled = VOSR2Settings(tile_strategy="tiled", dit_tile_batch=2, dino_batch=2, image_batch=2,
                        vae_encode_amp=False, auto_expand_vae_tile=False, temporal_cache=False)
with torch.no_grad():
    out = inf.upscale(images, 1, 42, settings=s_tiled, tile_size=64, tile_overlap=0,
                      vae_tile_size=64, vae_tile_overlap=0, progress=True)
check("分块管线输出形状", tuple(out.shape) == (2, 128, 128, 3), out.shape)
check("分块管线输出范围", float(out.min()) >= 0.0 and float(out.max()) <= 1.0)

with torch.no_grad():
    out_again = inf.upscale(images, 1, 42, settings=s_tiled, tile_size=64, tile_overlap=0,
                            vae_tile_size=64, vae_tile_overlap=0, progress=False)
check("同种子结果一致", torch.allclose(out, out_again, atol=1e-6))
with torch.no_grad():
    out_seed = inf.upscale(images, 1, 43, settings=s_tiled, tile_size=64, tile_overlap=0,
                           vae_tile_size=64, vae_tile_overlap=0, progress=False)
check("换种子结果变化", not torch.allclose(out, out_seed, atol=1e-6))

s_full = VOSR2Settings(tile_strategy="full_frame", vae_encode_amp=False,
                       auto_expand_vae_tile=False, temporal_cache=False)
with torch.no_grad():
    out_full = inf.upscale(torch.rand(1, 32, 64, 3), 2, 7, settings=s_full, tile_size=0,
                           tile_overlap=0, vae_tile_size=0, vae_tile_overlap=0, progress=False)
check("整图路径非方形放大", tuple(out_full.shape) == (1, 64, 128, 3), out_full.shape)

with torch.no_grad():
    out_mixed = inf.upscale(torch.rand(3, 32, 32, 3), 1, 1, settings=s_full, tile_size=0,
                            tile_overlap=0, vae_tile_size=0, vae_tile_overlap=0, progress=False)
check("批次逐项输出", tuple(out_mixed.shape) == (3, 32, 32, 3), out_mixed.shape)

# ── 6. 时序缓存（真实张量）──
cache = DinoTemporalCache(enabled=True, threshold=0.05, refresh=0)
pix = torch.rand(1, 3, 64, 64)
feat = torch.rand(1, 36, 32)
cache.begin_frame((128, 128, 8, [(0, 0)]))
cache.store((0, 0), pix, feat)
reused = cache.reuse((0, 0), pix, 1, torch.device("cpu"), torch.float32)
check("相同帧命中缓存", reused is not None and reused.shape == (1, 36, 32))
check("差异帧不复用", cache.reuse((0, 0), torch.rand(1, 3, 64, 64), 1,
                                  torch.device("cpu"), torch.float32) is None)
cache_refresh = DinoTemporalCache(enabled=True, threshold=1.0, refresh=4)
cache_refresh.store((0, 0), pix, feat)
check("refresh 周期强制重算",
      cache_refresh.reuse((0, 0), pix, 4, torch.device("cpu"), torch.float32) is None)
check("非刷新帧可命中",
      cache_refresh.reuse((0, 0), pix, 5, torch.device("cpu"), torch.float32) is not None)

# ── 6b. 时序缓存接入管线（整图 / 分块两条路径都要生效）──
cache_full = DinoTemporalCache(enabled=True, threshold=1.0, refresh=0)
same_frames = torch.rand(1, 64, 64, 3).repeat(3, 1, 1, 1)
with torch.no_grad():
    out_cached = inf.upscale(same_frames, 1, 11, settings=s_full, tile_size=0, tile_overlap=0,
                             vae_tile_size=0, vae_tile_overlap=0, progress=False, cache=cache_full)
check("整图路径时序缓存命中", cache_full.hits >= 2 and cache_full.misses >= 1,
      f"hits={cache_full.hits} misses={cache_full.misses}")
check("整图路径缓存输出形状", tuple(out_cached.shape) == (3, 64, 64, 3), out_cached.shape)

s_tiled_seq = VOSR2Settings(tile_strategy="tiled", dit_tile_batch=2, dino_batch=2, image_batch=1,
                            vae_encode_amp=False, auto_expand_vae_tile=False, temporal_cache=False)
cache_tiled = DinoTemporalCache(enabled=True, threshold=1.0, refresh=0)
same_big = torch.rand(1, 128, 128, 3).repeat(2, 1, 1, 1)
with torch.no_grad():
    out_cached_tiled = inf.upscale(same_big, 1, 12, settings=s_tiled_seq, tile_size=64,
                                   tile_overlap=0, vae_tile_size=64, vae_tile_overlap=0,
                                   progress=False, cache=cache_tiled)
check("分块路径时序缓存命中", cache_tiled.hits > 0 and cache_tiled.misses > 0,
      f"hits={cache_tiled.hits} misses={cache_tiled.misses}")
check("分块路径缓存输出形状", tuple(out_cached_tiled.shape) == (2, 128, 128, 3), out_cached_tiled.shape)

# ── 6c. 目标尺寸模式端到端（浮点倍率 / 总像素 / 缩小）──
with torch.no_grad():
    out_15 = inf.upscale(torch.rand(1, 64, 32, 3), TargetSizeSpec(mode="scale", scale=1.5), 21,
                         settings=s_full, tile_size=0, tile_overlap=0, vae_tile_size=0,
                         vae_tile_overlap=0, progress=False)
check("浮点 1.5x 输出尺寸", tuple(out_15.shape) == (1, 96, 48, 3), out_15.shape)

with torch.no_grad():
    out_mp = inf.upscale(torch.rand(1, 64, 64, 3),
                         TargetSizeSpec(mode="total pixels", total_pixels=0.02), 22,
                         settings=s_full, tile_size=0, tile_overlap=0, vae_tile_size=0,
                         vae_tile_overlap=0, progress=False)
# 0.02MP = 20972px，64x64=4096 → 倍率≈2.263 → 145x145（非 16 倍数，走 pad 路径）
check("总像素 0.02MP 输出尺寸", tuple(out_mp.shape) == (1, 145, 145, 3), out_mp.shape)

with torch.no_grad():
    out_down = inf.upscale(torch.rand(1, 64, 64, 3), TargetSizeSpec(mode="scale", scale=0.5), 23,
                           settings=s_full, tile_size=0, tile_overlap=0, vae_tile_size=0,
                           vae_tile_overlap=0, progress=False)
check("缩小 0.5x 输出尺寸", tuple(out_down.shape) == (1, 32, 32, 3), out_down.shape)

# ── 7. 视频节点执行（FakeBundle 注入 + 时序缓存路径）──
class _FakeModel(FakeBundle):
    def set_memory_policy(self, policy):
        self.policy = policy

    def set_torch_compile(self, enabled):
        self.compile = enabled
        return False

    def offload(self):
        self.offloaded = True


import sfnodes.nodes.video.vosr2_video as video_mod

node = video_mod.SFVOSR2Video()
fake_model = _FakeModel()
(settings,) = SFVOSR2Settings().make(
    "speed", "auto", "auto", False, False, False, 2, 2, 1, 2, 0, True, 1.0, 0
)
with torch.no_grad():
    (video_out,) = node.upscale(fake_model, torch.rand(3, 64, 64, 3),
                                "scale", 1.0, 1.0, 1024, 1024,
                                5, "wavelet", 1, 64, 0, 64, 0, False, settings)
check("视频节点输出形状", tuple(video_out.shape) == (3, 64, 64, 3), video_out.shape)
check("视频节点应用设置", fake_model.policy == "auto" and fake_model.compile is False)

# ── 8. OOM 自动降级路径 ──
class OomOnceBundle(FakeBundle):
    def __init__(self):
        self.calls = []

    def dit_velocity(self, inp, t_cur, t_next, venc):
        self.calls.append(int(inp.shape[0]))
        if inp.shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. simulated")
        return super().dit_velocity(inp, t_cur, t_next, venc)


oom_bundle = OomOnceBundle()
s_oom = VOSR2Settings(tile_strategy="tiled", dit_tile_batch=4, dino_batch=4, image_batch=1,
                      vae_encode_amp=False, auto_expand_vae_tile=False, temporal_cache=False)
with torch.no_grad():
    out_oom = VOSR2Inferencer(oom_bundle).upscale(
        torch.rand(1, 128, 128, 3), 1, 3, settings=s_oom, tile_size=64, tile_overlap=0,
        vae_tile_size=64, vae_tile_overlap=0, progress=False,
    )
check("DiT 批量 OOM 降级后成功", tuple(out_oom.shape) == (1, 128, 128, 3), out_oom.shape)
check("降级先大后小重试", 4 in oom_bundle.calls and oom_bundle.calls[-1] == 1, oom_bundle.calls)

seen_sizes = []
decode_bundle = FakeBundle()
_orig_decode = decode_bundle.decode


def _decode_oom(lat, mean, std, tile_size=0, tile_overlap=0, amp=False):
    seen_sizes.append(tile_size)
    if tile_size and tile_size > 512:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. simulated")
    return _orig_decode(lat, mean, std, tile_size, tile_overlap, amp)


decode_bundle.decode = _decode_oom
with torch.no_grad():
    dec_out = VOSR2Inferencer(decode_bundle)._decode_with_fallback(
        torch.randn(1, 4, 64, 64), torch.zeros(1, 4, 1, 1), torch.ones(1, 4, 1, 1),
        1024, 32, False, 512, 512,
    )
check("VAE 瓦片 OOM 降级", seen_sizes == [1024, 512], seen_sizes)
check("VAE 降级输出形状", tuple(dec_out.shape) == (1, 3, 512, 512), dec_out.shape)

print("\n" + ("全部通过" if not failures else f"{len(failures)} 项失败: {failures}"))
sys.exit(1 if failures else 0)
