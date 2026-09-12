# SFPainterFluxImageEdit 后端模拟测试（Python 直接运行：python tests/test_painter_flux_edit.py）
# 覆盖：
#   - sf_utils.flux_edit 纯逻辑：VL 面积缩放 / imageN 前缀（encode_vision 开关）/
#     reference_latents 注入正负条件 / reference_latents_method / 首图 ref latent 起点 vs 空 latent /
#     遮罩对齐 latent 空间 / batch 对齐（repeat_to_batch_size，不复制 conditioning）/
#     negative_prompt
#   - mock torch + comfy.utils + comfy.model_management + node_helpers（本机无 torch，FakeTensor 走 numpy）
#   - 节点壳：INPUT_TYPES / 动态 imageN 收集 / image1_mask 传递 / 文本-only 路径

import os
import sys
import math
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


# ── FakeTensor / fake torch / fake comfy ──────────────────────────────────────

class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)
        self.shape = self._arr.shape
        self.ndim = self._arr.ndim
        self.dtype = self._arr.dtype
        self.device = "cpu"

    def movedim(self, src, dst):
        return FakeTensor(np.moveaxis(self._arr, src, dst))

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self._arr, dim))

    def squeeze(self, dim=None):
        return FakeTensor(np.squeeze(self._arr, axis=dim))

    def dim(self):
        return self.ndim

    def narrow(self, dim, start, length):
        idx = [slice(None)] * self.ndim
        idx[dim] = slice(start, start + length)
        return FakeTensor(self._arr[tuple(idx)])

    def repeat(self, *reps):
        return FakeTensor(np.tile(self._arr, reps))

    def __getitem__(self, idx):
        res = self._arr[idx]
        return FakeTensor(res) if isinstance(res, np.ndarray) else res


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = FakeTensor
fake_torch.zeros = lambda *shape, **kw: FakeTensor(np.zeros(shape, dtype=np.float32))
sys.modules["torch"] = fake_torch

upscale_calls = []


def fake_common_upscale(samples, width, height, upscale_method, crop):
    upscale_calls.append({"in_shape": samples.shape, "w": int(width), "h": int(height),
                          "method": upscale_method, "crop": crop})
    b, c = samples.shape[0], samples.shape[1]
    return FakeTensor(np.zeros((b, c, int(height), int(width)), dtype=np.float32))


def fake_repeat_to_batch_size(tensor, batch_size, dim=0):
    arr = tensor._arr
    n = arr.shape[dim]
    if n == batch_size:
        return tensor
    idx = [slice(None)] * arr.ndim
    if n > batch_size:
        idx[dim] = slice(0, batch_size)
        return FakeTensor(arr[tuple(idx)])
    reps = [1] * arr.ndim
    reps[dim] = math.ceil(batch_size / n)
    tiled = np.tile(arr, reps)
    idx[dim] = slice(0, batch_size)
    return FakeTensor(tiled[tuple(idx)])


fake_comfy = types.ModuleType("comfy")
fake_comfy_utils = types.ModuleType("comfy.utils")
fake_comfy_utils.common_upscale = fake_common_upscale
fake_comfy_utils.repeat_to_batch_size = fake_repeat_to_batch_size
fake_comfy.utils = fake_comfy_utils
fake_comfy_mm = types.ModuleType("comfy.model_management")
fake_comfy_mm.get_torch_device = lambda: "cpu"
fake_comfy.model_management = fake_comfy_mm
sys.modules["comfy"] = fake_comfy
sys.modules["comfy.utils"] = fake_comfy_utils
sys.modules["comfy.model_management"] = fake_comfy_mm

fake_node_helpers = types.ModuleType("node_helpers")


def fake_conditioning_set_values(conditioning, values, append=False):
    out = []
    for c in conditioning:
        d = dict(c[1]) if len(c) > 1 and isinstance(c[1], dict) else {}
        for k, v in values.items():
            if append and k in d and isinstance(d[k], list):
                d[k] = d[k] + list(v)
            else:
                d[k] = v
        out.append([c[0], d])
    return out


fake_node_helpers.conditioning_set_values = fake_conditioning_set_values
sys.modules["node_helpers"] = fake_node_helpers

# ── 纯逻辑导入 ────────────────────────────────────────────────────────────────

from sf_utils.flux_edit import encode_painter_flux  # noqa: E402


def make_image(h, w, b=1, c=3):
    return FakeTensor(np.zeros((b, h, w, c), dtype=np.float32))


def make_mask(h, w, b=1):
    return FakeTensor(np.zeros((b, h, w), dtype=np.float32))


class FakeVae:
    """encode 返回 Flux2 形状 latent（128ch / 16x），值以调用序号填充便于区分。"""

    def __init__(self):
        self.encoded = []

    def encode(self, img):
        self.encoded.append(img.shape)
        b, h, w = img.shape[0], img.shape[1], img.shape[2]
        return FakeTensor(np.full((b, 128, h // 16, w // 16), len(self.encoded), dtype=np.float32))


class FakeClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, images=None):
        self.calls.append({"prompt": prompt, "images": [i.shape for i in (images or [])]})
        return ["TOKENS"]

    def encode_from_tokens_scheduled(self, tokens):
        return [["COND", {}]]


# ── 1. 单图 + 遮罩 + VL（encode_vision=True）──────────────────────────────────

vae = FakeVae()
clip = FakeClip()
upscale_calls.clear()
positive, negative, latent = encode_painter_flux(
    clip, vae, " make it snow", [make_image(512, 512)],
    mask=make_mask(512, 512), width=1024, height=1024, batch_size=1, encode_vision=True,
)

check("vae 编码两次（参考图 + 空 latent）", len(vae.encoded) == 2)
check("参考图拉伸出图尺寸", vae.encoded[0][1] == 1024 and vae.encoded[0][2] == 1024)
check("VL 面积≈384²", any(abs(c["w"] * c["h"] - 384 * 384) < 384 for c in upscale_calls))
check("VL 用 area + center", any(c["method"] == "area" and c["crop"] == "center" for c in upscale_calls))
check("positive 含 reference_latents", "reference_latents" in positive[0][1])
check("negative 含 reference_latents", "reference_latents" in negative[0][1])
check("reference_latents 数量=1", len(positive[0][1]["reference_latents"]) == 1)
check("latent.samples = 首图 ref latent", latent["samples"]._arr.flat[0] == 1.0)
check("noise_mask 对齐 latent 空间（16x）", latent["noise_mask"].shape == (1, 64, 64))
check("noise_mask 非 //8", latent["noise_mask"].shape[1:] != (128, 128))
check("full_prompt 前缀", clip.calls[0]["prompt"].startswith(
    "image1: <|vision_start|><|image_pad|><|vision_end|> "))
check("negative tokenize 空串", clip.calls[1]["prompt"] == "" and clip.calls[1]["images"] == [])

# ── 2. encode_vision 默认关 + negative_prompt + method + init ────────────────

vae2 = FakeVae()
clip2 = FakeClip()
upscale_calls.clear()
pos2, neg2, lat2 = encode_painter_flux(
    clip2, vae2, "edit", [make_image(256, 256)], negative_prompt="bad stuff",
    width=512, height=512, reference_latents_method="uxo",
)
check("encode_vision 默认关：无 VL 缩放", not any(
    c["method"] == "area" and abs(c["w"] * c["h"] - 384 * 384) < 384 for c in upscale_calls))
check("encode_vision 默认关：无 vision 前缀", clip2.calls[0]["prompt"] == "edit")
check("encode_vision 默认关：images 为空", clip2.calls[0]["images"] == [])
check("negative_prompt 生效", clip2.calls[1]["prompt"] == "bad stuff")
check("reference_latents_method 注入正条件", pos2[0][1].get("reference_latents_method") == "uxo")
check("reference_latents_method 注入负条件", neg2[0][1].get("reference_latents_method") == "uxo")
check("默认以首图 ref latent 为起点", lat2["samples"]._arr.flat[0] == 1.0)

vae3 = FakeVae()
pos3, neg3, lat3 = encode_painter_flux(
    FakeClip(), vae3, "x", [make_image(256, 256)],
    width=512, height=512, use_reference_latent_as_init=False,
)
check("关起点开关时用空 latent", lat3["samples"]._arr.flat[0] == 2.0)
check("无 method 时不注入", "reference_latents_method" not in pos3[0][1])

# ── 3. 多图（编号前缀 + 数量，encode_vision=True）────────────────────────────

vae4 = FakeVae()
clip4 = FakeClip()
pos4, neg4, lat4 = encode_painter_flux(
    clip4, vae4, " edit", [make_image(320, 256), make_image(100, 100), make_image(64, 64)],
    width=512, height=512, batch_size=1, encode_vision=True,
)
check("三图 vae 编码 4 次（3 参考 + 空）", len(vae4.encoded) == 4)
check("三图 reference_latents=3", len(pos4[0][1]["reference_latents"]) == 3)
check("多图编号前缀", "image1: <|vision_start|><|image_pad|><|vision_end|>" in clip4.calls[0]["prompt"])
check("多图 image3 前缀", "image3: <|vision_start|><|image_pad|><|vision_end|>" in clip4.calls[0]["prompt"])
check("多图 vl_images=3", len(clip4.calls[0]["images"]) == 3)
check("无 mask 时 latent 无 noise_mask", "noise_mask" not in lat4)

# ── 4. batch_size（不复制 conditioning，latent/mask 对齐 batch）─────────────

vae5 = FakeVae()
clip5 = FakeClip()
pos5, neg5, lat5 = encode_painter_flux(
    clip5, vae5, " x", [make_image(256, 256)],
    mask=make_mask(256, 256), width=512, height=512, batch_size=4,
)
check("batch 不复制 conditioning（positive 仍 1 条）", len(pos5) == 1 and len(neg5) == 1)
check("batch latent 复制 4", lat5["samples"].shape[0] == 4)
check("batch noise_mask 复制 4", lat5["noise_mask"].shape == (4, 32, 32))

# 参考图自带 batch=2、batch_size=1 -> 切片到 1（不放大）
vae6 = FakeVae()
pos6, neg6, lat6 = encode_painter_flux(
    FakeClip(), vae6, "x", [make_image(256, 256, b=2)], width=512, height=512, batch_size=1,
)
check("源 batch>目标时切片", lat6["samples"].shape[0] == 1)

# ── 5. 无图纯文本路径 ────────────────────────────────────────────────────────

vae7 = FakeVae()
clip7 = FakeClip()
pos7, neg7, lat7 = encode_painter_flux(clip7, vae7, "hello", [], width=512, height=512)
check("纯文本 vae 只编码空 latent", len(vae7.encoded) == 1)
check("纯文本无 reference_latents", "reference_latents" not in pos7[0][1])
check("纯文本 latent 尺寸来自空 latent", lat7["samples"].shape == (1, 128, 32, 32))
check("纯文本无 noise_mask", "noise_mask" not in lat7)
check("纯文本 full_prompt", clip7.calls[0]["prompt"] == "hello")

# ── 6. 缺 VAE 抛错 ───────────────────────────────────────────────────────────

raised = False
try:
    encode_painter_flux(FakeClip(), None, "x", [])
except RuntimeError:
    raised = True
check("缺 VAE 抛 RuntimeError", raised)

# ── 7. 节点壳 ────────────────────────────────────────────────────────────────

_sf_pkg = types.ModuleType("sfnodes")
_sf_pkg.__path__ = [root]
sys.modules["sfnodes"] = _sf_pkg
_sf_nodes_pkg = types.ModuleType("sfnodes.nodes")
_sf_nodes_pkg.__path__ = [os.path.join(root, "nodes")]
sys.modules["sfnodes.nodes"] = _sf_nodes_pkg
_sf_nutils_pkg = types.ModuleType("sfnodes.nodes.utils")
_sf_nutils_pkg.__path__ = [os.path.join(root, "nodes", "utils")]
sys.modules["sfnodes.nodes.utils"] = _sf_nutils_pkg
_sf_sutils_pkg = types.ModuleType("sfnodes.sf_utils")
_sf_sutils_pkg.__path__ = [os.path.join(root, "sf_utils")]
sys.modules["sfnodes.sf_utils"] = _sf_sutils_pkg

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.utils.painter_flux_edit",
    os.path.join(root, "nodes", "utils", "painter_flux_edit.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

Node = mod.SFPainterFluxImageEdit
check("节点 CATEGORY", Node.CATEGORY == "sfnodes/model")
check("节点 FUNCTION", Node.FUNCTION == "encode")
check("节点 RETURN_TYPES", Node.RETURN_TYPES == ("CONDITIONING", "CONDITIONING", "LATENT"))
check("节点 RETURN_NAMES", Node.RETURN_NAMES == ("positive", "negative", "latent"))

it = Node.INPUT_TYPES()
check("required 含 clip/prompt/negative_prompt/batch_size/width/height",
      all(k in it["required"] for k in ("clip", "prompt", "negative_prompt", "batch_size", "width", "height")))
check("width step=16", it["required"]["width"][1]["step"] == 16)
check("height step=16", it["required"]["height"][1]["step"] == 16)
check("optional 含 vae/image1_mask/image1",
      all(k in it["optional"] for k in ("vae", "image1_mask", "image1")))
check("optional 含 P2 三参数", all(k in it["optional"] for k in (
    "reference_latents_method", "use_reference_latent_as_init", "encode_vision")))
check("encode_vision 默认 False", it["optional"]["encode_vision"][1]["default"] is False)
check("use_reference_latent_as_init 默认 True",
      it["optional"]["use_reference_latent_as_init"][1]["default"] is True)
check("无 mode 枚举", "mode" not in it["required"])

node = Node()
clip_node = FakeClip()
res = node.encode(
    clip=clip_node, vae=FakeVae(), prompt="x", negative_prompt="neg", batch_size=1,
    width=512, height=512, image1=make_image(256, 256), image3=make_image(128, 128),
)
check("节点收集乱序 imageN（image1+image3）", len(res[0][0][1]["reference_latents"]) == 2)
check("节点 negative_prompt 透传", clip_node.calls[1]["prompt"] == "neg")

res2 = node.encode(clip=FakeClip(), vae=FakeVae(), prompt="solo", batch_size=1, width=512, height=512)
check("节点无图纯文本", "reference_latents" not in res2[0][0][1])

# ── 汇总 ─────────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"{len(failures)} failures: {failures}")
    sys.exit(1)
print("All tests passed.")
