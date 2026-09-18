# SF SCAIL-2 预处理内存优化测试（python3 tests/test_scail2_mem.py）
# 覆盖：
#   - sf_utils/scail2_mem.py read_options：默认值 / 非法值回退 / 显式注入
#   - 分块纯函数与核心逐行同构实现逐元素一致：render_colored_masks_chunked
#     （chunk 边界 / 黑白色底 / N=0 / packed=None / half）与
#     extract_mask_to_28ch_chunked（4n+1 时间打包顺序）
#   - 包装器门控：Enabled=False 完全原生（含不转 f16）、短输入走原生 + half
#   - WanSCAILToVideo.execute 参考蒙版裁剪（n_mask > n_ref 才裁；kwargs 与
#     位置参数两种调用形态；disabled/n_mask<=n_ref 不裁）
#   - install 幂等与 _MARK 守卫（假模块注入，不碰真实核心）
# mock：numpy 代理张量（分块函数本就依赖注入 torch/interpolate/unpack_masks，
# 不 import 真实 torch）；_to_half 的 `import torch` 用假模块短时注入。
import os
import sys
import types

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from sf_utils import scail2_mem as mem  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def assert_eq(a, b, msg=""):
    check(f"{msg}: {a!r} == {b!r}", a == b)


PALETTE = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
]


# ── numpy 代理张量（只实现分块函数用到的 torch 接口）──
_DTYPES = (np.float16, np.float32, np.uint8, np.bool_)


def _np_dtype(dtype):
    return dtype if dtype in _DTYPES else None


def _raw(value):
    return value.a if isinstance(value, NpT) else value


def _idx(idx):
    if isinstance(idx, tuple):
        return tuple(_idx(item) for item in idx)
    return idx.a if isinstance(idx, NpT) else idx


class NpT:
    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def shape(self):
        return self.a.shape

    @property
    def dtype(self):
        return self.a.dtype

    @property
    def device(self):
        return "cpu"

    def float(self):
        return NpT(self.a.astype(np.float32))

    def to(self, dtype=None, device=None):
        nd = _np_dtype(dtype)
        return NpT(self.a.astype(nd)) if nd is not None else self

    def view(self, *shape):
        return NpT(self.a.reshape(*shape))

    def unsqueeze(self, dim):
        return NpT(np.expand_dims(self.a, dim))

    def movedim(self, source, destination):
        return NpT(np.moveaxis(self.a, source, destination))

    def repeat(self, *reps):
        return NpT(np.tile(self.a, reps))

    def expand_as(self, other):
        return NpT(np.broadcast_to(self.a, other.shape))

    def any(self, dim):
        return NpT(self.a.any(axis=dim))

    def argmax(self, dim):
        return NpT(self.a.argmax(axis=dim))

    def __getitem__(self, idx):
        return NpT(self.a[_idx(idx)])

    def __setitem__(self, idx, value):
        self.a[_idx(idx)] = _raw(value)

    def __gt__(self, other):
        return NpT(self.a > _raw(other))

    def __mul__(self, other):
        return NpT(self.a * _raw(other))

    def __rmul__(self, other):
        return NpT(_raw(other) * self.a)

    def __rsub__(self, other):
        return NpT(_raw(other) - self.a)


torch_stub = types.SimpleNamespace(
    float16=np.float16,
    float32=np.float32,
    uint8=np.uint8,
    empty=lambda *shape, device=None, dtype=None: NpT(np.empty(shape, dtype=_np_dtype(dtype) or np.float32)),
    tensor=lambda seq, device=None, dtype=None: NpT(np.array(seq, dtype=_np_dtype(dtype) or np.float32)),
    where=lambda cond, a, b: NpT(np.where(_raw(cond), _raw(a), _raw(b))),
    cat=lambda items, dim=0: NpT(np.concatenate([_raw(item) for item in items], axis=dim)),
)


def fake_interpolate(x, size=None, mode="nearest", **kwargs):
    """测试用插值：nearest 最近邻 / area 块均值（测试尺寸均整除）。"""
    src = _raw(x)
    if size is None:
        size = kwargs.get("size")
    target_h, target_w = size
    src_h, src_w = src.shape[-2], src.shape[-1]
    if mode == "nearest":
        ys = np.clip(((np.arange(target_h) + 0.5) * src_h / target_h).astype(int), 0, src_h - 1)
        xs = np.clip(((np.arange(target_w) + 0.5) * src_w / target_w).astype(int), 0, src_w - 1)
        out = src[..., ys, :][..., :, xs]
    else:
        factor_y, factor_x = src_h // target_h, src_w // target_w
        out = src.reshape(*src.shape[:-2], target_h, factor_y, target_w, factor_x).mean(axis=(-3, -1))
    return NpT(out)


def fake_unpack_masks(packed):
    """复刻核心 unpack_masks：位打包 [*, H, W//8] uint8 → bool [*, H, W*8]。"""
    a = _raw(packed)
    bits = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    unpacked = (a[..., None] & bits).astype(bool)
    return NpT(unpacked.reshape(a.shape[0], a.shape[1], a.shape[2], -1))


# ── 核心实现逐行同构参考（comfy_extras/nodes_scail.py 的 numpy 版）──
def native_render(track_data, background="black"):
    packed = track_data["packed_masks"]
    H, W = track_data["orig_size"]
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    if packed is None or packed.shape[1] == 0:
        T = track_data.get("n_frames", 1) if packed is None else packed.shape[0]
        out = torch_stub.empty(T, H, W, 3, device="cpu", dtype=np.float32)
        out[..., 0], out[..., 1], out[..., 2] = bg_rgb[0], bg_rgb[1], bg_rgb[2]
        return out
    T, N_obj = packed.shape[0], packed.shape[1]
    colors = torch_stub.tensor([PALETTE[i % len(PALETTE)] for i in range(N_obj)], device="cpu", dtype=np.float32)
    masks_full = fake_unpack_masks(packed.to("cpu")).float()
    Hm, Wm = masks_full.shape[-2], masks_full.shape[-1]
    masks_full = (
        fake_interpolate(masks_full.view(T * N_obj, 1, Hm, Wm), size=(H, W), mode="nearest").view(T, N_obj, H, W)
        > 0.5
    )
    any_mask = masks_full.any(dim=1)
    color_overlay = colors[masks_full.to(torch_stub.uint8).argmax(dim=1)]
    bg_tensor = torch_stub.tensor(bg_rgb, device="cpu", dtype=color_overlay.dtype).view(1, 1, 1, 3)
    return torch_stub.where(any_mask.unsqueeze(-1), color_overlay, bg_tensor.expand_as(color_overlay))


def native_extract(rgb_video):
    T, H, W, _ = rgb_video.shape
    on_thresh = 225.0 / 255.0
    mask = rgb_video.movedim(-1, 1).float()
    R = (mask[:, 0:1] > on_thresh).float()
    G = (mask[:, 1:2] > on_thresh).float()
    B = (mask[:, 2:3] > on_thresh).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary_7ch = torch_stub.cat(
        [
            R * G * B,
            R * nG * nB,
            nR * G * nB,
            nR * nG * B,
            R * G * nB,
            R * nG * B,
            nR * G * B,
        ],
        dim=1,
    )
    H_lat, W_lat = H, W
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    binary_7ch = fake_interpolate(binary_7ch, size=(H_lat, W_lat), mode="area")
    T_latent = (T - 1) // 4 + 1
    padded = torch_stub.cat([binary_7ch[:1].repeat(4, 1, 1, 1), binary_7ch[1:]], dim=0)
    out = padded.view(T_latent, 28, H_lat, W_lat)
    return out.unsqueeze(0)


def np_eq(actual, expected):
    a, e = _raw(actual), _raw(expected)
    return a.shape == e.shape and a.dtype == e.dtype and np.array_equal(a, e)


def make_track_data(rng, T, N, Hm=4, Wm=8, target=(8, 8)):
    packed = NpT(rng.integers(0, 256, size=(T, N, Hm, Wm // 8), dtype=np.uint8))
    return {"packed_masks": packed, "n_frames": T, "orig_size": target}


# ── read_options ──
assert_eq(mem.read_options({}), (True, 32, True), "read_options 默认")
assert_eq(mem.read_options("not-a-dict"), (True, 32, True), "read_options 非 dict")
assert_eq(
    mem.read_options(
        {
            mem.SETTING_ENABLED: False,
            mem.SETTING_CHUNK: 8,
            mem.SETTING_HALF: False,
        }
    ),
    (False, 8, False),
    "read_options 显式值",
)
assert_eq(mem.read_options({mem.SETTING_CHUNK: "abc"})[1], 32, "read_options 非法 chunk 回退")
assert_eq(mem.read_options({mem.SETTING_CHUNK: 0})[1], 32, "read_options 0 回退")
assert_eq(mem.read_options({mem.SETTING_CHUNK: -3})[1], 32, "read_options 负值回退")

# ── render 分块等价性 ──
rng = np.random.default_rng(20260918)
for T in (1, 10):
    for N in (0, 1, 3):
        for background in ("black", "white"):
            track = make_track_data(rng, T, N)
            expected = _raw(native_render(track, background))
            for half in (False, True):
                if half:
                    expected = expected.astype(np.float16)
                for chunk in (1, 2, 3, 4, 5, 7, 10, 11, 100):
                    got = mem.render_colored_masks_chunked(
                        track,
                        background,
                        torch=torch_stub,
                        unpack_masks=fake_unpack_masks,
                        interpolate=fake_interpolate,
                        palette=PALETTE,
                        device="cpu",
                        dtype=np.float32,
                        chunk_frames=chunk,
                        half=half,
                    )
                    check(
                        f"render 等价 T={T} N={N} bg={background} half={half} chunk={chunk}",
                        np_eq(got, expected),
                    )

# packed=None / 空 packed 分支
for track in (
    {"packed_masks": None, "n_frames": 3, "orig_size": (8, 8)},
    {"packed_masks": NpT(np.zeros((3, 0, 4, 1), dtype=np.uint8)), "n_frames": 3, "orig_size": (8, 8)},
):
    for half in (False, True):
        expected = _raw(native_render(track, "black"))
        if half:
            expected = expected.astype(np.float16)
        got = mem.render_colored_masks_chunked(
            track,
            "black",
            torch=torch_stub,
            unpack_masks=fake_unpack_masks,
            interpolate=fake_interpolate,
            palette=PALETTE,
            device="cpu",
            dtype=np.float32,
            chunk_frames=2,
            half=half,
        )
        check(f"render 空 packed half={half}", np_eq(got, expected))

# ── 28ch 分块等价性 ──
for T in (5, 9, 17):
    rgb = NpT(rng.integers(0, 2, size=(T, 16, 8, 3)).astype(np.float32))
    expected = native_extract(rgb)
    for chunk in (1, 2, 4, 8, 32, 64):
        got = mem.extract_mask_to_28ch_chunked(rgb, torch=torch_stub, interpolate=fake_interpolate, chunk_frames=chunk)
        check(f"28ch 等价 T={T} chunk={chunk}", np_eq(got, expected))

# ── 包装器门控（read_options 注入）──
class HalfSpy:
    def __init__(self):
        self.to_calls = 0

    def to(self, dtype):
        self.to_calls += 1
        return ("half", dtype)


_orig_read_options = mem.read_options
_fake_torch = types.ModuleType("torch")
_fake_torch.float16 = "F16"
_prev_torch = sys.modules.get("torch")
_prev_patched = mem._PATCHED
try:
    sys.modules["torch"] = _fake_torch

    def make_render_wrapper(track_data, background="black"):
        spy = HalfSpy()
        orig_calls = []

        def orig(td, bg="black"):
            orig_calls.append((td, bg))
            return spy

        wrapper = mem._wrap_render(PALETTE, orig)
        result = wrapper(track_data, background)
        return result, spy, orig_calls

    mem.read_options = lambda: (False, 32, True)
    spy_track = {"packed_masks": NpT(np.zeros((32, 1, 4, 1), dtype=np.uint8)), "n_frames": 32, "orig_size": (8, 8)}
    result, spy, calls = make_render_wrapper(spy_track)
    check("disabled 完全原生（未转 f16）", result is spy and spy.to_calls == 0 and len(calls) == 1)

    mem.read_options = lambda: (True, 32, True)
    result, spy, calls = make_render_wrapper(spy_track)
    check("enabled+half 短输入转 f16", result == ("half", "F16") and spy.to_calls == 1)

    mem.read_options = lambda: (True, 32, False)
    result, spy, calls = make_render_wrapper(spy_track)
    check("enabled+half=False 不转", result is spy and spy.to_calls == 0)

    mem.read_options = lambda: (True, 32, True)
    result, spy, calls = make_render_wrapper({"packed_masks": None, "n_frames": 4, "orig_size": (8, 8)})
    check("packed=None 短输入仍转 f16", result == ("half", "F16") and spy.to_calls == 1)

    # ── WanSCAILToVideo.execute 参考蒙版裁剪 ──
    class Seq:
        def __init__(self, n):
            self.shape = (n,)

        def __getitem__(self, item):
            if isinstance(item, slice):
                return Seq(len(range(*item.indices(self.shape[0]))))
            raise IndexError(item)

    exec_calls = []

    def fake_execute(cls, positive, negative, vae, width, height, length, batch_size,
                     pose_strength, pose_start, pose_end, video_frame_offset, previous_frame_count,
                     replacement_mode=False, reference_image=None, clip_vision_output=None,
                     pose_video=None, pose_video_mask=None, reference_image_mask=None, previous_frames=None):
        exec_calls.append({"cls": cls, "ref_mask": reference_image_mask, "ref_img": reference_image})
        return ("exec-out",)

    wrapped_execute = mem._wrap_execute(fake_execute)
    mask4, img2 = Seq(4), Seq(2)
    mem.read_options = lambda: (True, 32, False)
    out = wrapped_execute(
        "CLS",
        positive=1, negative=2, vae=3, width=512, height=896, length=81, batch_size=1,
        pose_strength=1.0, pose_start=0.0, pose_end=1.0, video_frame_offset=0, previous_frame_count=5,
        reference_image=img2, reference_image_mask=mask4,
    )
    check("execute 裁剪 n_mask>n_ref", out == ("exec-out",) and exec_calls[-1]["ref_mask"].shape == (2,))
    check("execute 裁剪生成新对象", exec_calls[-1]["ref_mask"] is not mask4)

    mask2 = Seq(2)
    wrapped_execute("CLS", positive=1, negative=2, vae=3, width=512, height=896, length=81, batch_size=1,
                    pose_strength=1.0, pose_start=0.0, pose_end=1.0, video_frame_offset=0, previous_frame_count=5,
                    reference_image=img2, reference_image_mask=mask2)
    check("execute n_mask==n_ref 不裁", exec_calls[-1]["ref_mask"] is mask2)

    mask1 = Seq(1)
    wrapped_execute("CLS", positive=1, negative=2, vae=3, width=512, height=896, length=81, batch_size=1,
                    pose_strength=1.0, pose_start=0.0, pose_end=1.0, video_frame_offset=0, previous_frame_count=5,
                    reference_image=Seq(3), reference_image_mask=mask1)
    check("execute n_mask<n_ref 不裁（原生广播）", exec_calls[-1]["ref_mask"] is mask1)

    mem.read_options = lambda: (False, 32, False)
    wrapped_execute("CLS", positive=1, negative=2, vae=3, width=512, height=896, length=81, batch_size=1,
                    pose_strength=1.0, pose_start=0.0, pose_end=1.0, video_frame_offset=0, previous_frame_count=5,
                    reference_image=img2, reference_image_mask=mask4)
    check("execute disabled 不裁", exec_calls[-1]["ref_mask"] is mask4)

    mem.read_options = lambda: (True, 32, False)
    wrapped_execute("CLS", 1, 2, 3, 512, 896, 81, 1, 1.0, 0.0, 1.0, 0, 5,
                    False, Seq(1), None, None, None, Seq(5), None)
    check("execute 位置参数形态裁剪", exec_calls[-1]["ref_mask"].shape == (1,))

    # ── install 幂等（假模块注入）──
    fake_mod = types.ModuleType("fake_nodes_scail")
    fake_mod.DEFAULT_PALETTE = PALETTE

    def fake_render(track_data, background="black"):
        return "render-native"

    def fake_extract(rgb_video):
        return "extract-native"

    fake_mod._render_colored_masks = fake_render
    fake_mod._extract_mask_to_28ch = fake_extract

    class FakeWanSCAILToVideo:
        @classmethod
        def execute(cls, positive, reference_image=None, reference_image_mask=None):
            return "exec-native"

    fake_mod.WanSCAILToVideo = FakeWanSCAILToVideo

    mem._PATCHED = False
    patched = mem.install(module=fake_mod)
    check("install 补丁 3 项", patched == 3)
    check("render 已标记", getattr(fake_mod._render_colored_masks, mem._MARK, False))
    check("extract 已标记", getattr(fake_mod._extract_mask_to_28ch, mem._MARK, False))
    check("execute 已标记", getattr(
        getattr(fake_mod.WanSCAILToVideo.__dict__["execute"], "__func__", None),
        "_sf_scail2_mem_exec_patched",
        False,
    ))
    check("install 二次返回 0", mem.install(module=fake_mod) == 0)

    mem._PATCHED = False
    check("install 已标记模块再装返回 0", mem.install(module=fake_mod) == 0)

    mem.read_options = lambda: (False, 32, True)
    check(
        "补丁后 disabled render 走原生",
        fake_mod._render_colored_masks({"packed_masks": None}) == "render-native",
    )
finally:
    mem.read_options = _orig_read_options
    mem._PATCHED = _prev_patched
    if _prev_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = _prev_torch

if failures:
    print(f"\n{len(failures)} FAILURES")
    sys.exit(1)
print("\ntest_scail2_mem: all assertions passed")
