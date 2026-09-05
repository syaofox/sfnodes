# SFQwenEditTextEncode / SFQwenEditOutputExtractor 后端模拟测试（Python 直接运行：python tests/test_qwen_edit.py）
# 覆盖：
#   - sf_utils.qwen_edit 纯逻辑：longest_edge 缩放 / pad 画布 vae_unit 对齐 / pad_info / mask→noise_mask / Picture 编号
#   - mock torch + comfy.utils（本机无 torch，FakeTensor 走 numpy）
#   - 节点壳：INPUT_TYPES / 每图参数收集 / mask 形状不符丢弃 / 无图纯文本路径
#   - Extractor 拆包一致性

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
        self._arr = np.asarray(arr)
        self.shape = self._arr.shape
        self.ndim = self._arr.ndim
        self.dtype = self._arr.dtype
        self.device = "cpu"

    def movedim(self, src, dst):
        return FakeTensor(np.moveaxis(self._arr, src, dst))

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self._arr, dim))

    def repeat(self, *reps):
        return FakeTensor(np.tile(self._arr, reps))

    def squeeze(self, dim=None):
        return FakeTensor(np.squeeze(self._arr, axis=dim))

    def __getitem__(self, idx):
        res = self._arr[idx]
        return FakeTensor(res) if isinstance(res, np.ndarray) else res

    def __setitem__(self, idx, value):
        self._arr[idx] = value._arr if isinstance(value, FakeTensor) else value


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = FakeTensor
def fake_zeros(shape, **kw):
    if isinstance(shape, tuple):
        return FakeTensor(np.zeros(shape, dtype=np.float32))
    return FakeTensor(np.zeros((shape,), dtype=np.float32))


fake_torch.zeros = fake_zeros
fake_torch.zeros_like = lambda t: FakeTensor(np.zeros_like(t._arr, dtype=np.float32))
sys.modules["torch"] = fake_torch

fake_comfy = types.ModuleType("comfy")
fake_comfy_utils = types.ModuleType("comfy.utils")


def fake_common_upscale(samples, width, height, upscale_method, crop):
    # 形状正确的占位缩放（nearest，仅用于形状断言）
    b, c = samples._arr.shape[0], samples._arr.shape[1]
    return FakeTensor(np.zeros((b, c, height, width), dtype=np.float32))


fake_comfy_utils.common_upscale = fake_common_upscale
fake_comfy.utils = fake_comfy_utils
sys.modules["comfy"] = fake_comfy
sys.modules["comfy.utils"] = fake_comfy_utils

fake_node_helpers = types.ModuleType("node_helpers")


def fake_conditioning_set_values(conditioning, values, append=True):
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

from sf_utils.qwen_edit import (  # noqa: E402
    scale_longest_edge,
    pad_info_from,
    mask_matches,
    encode_qwen_edit,
    TEXT_ONLY_LATENT_SHAPE,
)


def make_image(h, w, b=1, c=3):
    return FakeTensor(np.zeros((b, h, w, c), dtype=np.float32))


def make_mask(h, w, b=1):
    return FakeTensor(np.zeros((b, h, w), dtype=np.float32))


class FakeVae:
    def __init__(self):
        self.encoded = []

    def encode(self, img):
        self.encoded.append(img.shape)
        return {"samples": FakeTensor(np.zeros((img.shape[0], 16, img.shape[1] // 8, img.shape[2] // 8)))}


class FakeClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, images=None, llama_template=None):
        self.calls.append({"prompt": prompt, "images": [i.shape for i in (images or [])],
                           "llama_template": llama_template})
        return ["TOKENS"]

    def encode_from_tokens_scheduled(self, tokens):
        return [["COND", {}]]


# ── 1. 纯函数 ────────────────────────────────────────────────────────────────


def _raised(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


check("scale_longest_edge 等比", scale_longest_edge(64, 50, 32) == (32, 25))
check("scale_longest_edge 横图", scale_longest_edge(48, 96, 32) == (16, 32))
check("scale_longest_edge 非法尺寸", _raised(lambda: scale_longest_edge(0, 10, 32)))

check("pad_info_from 数值", pad_info_from(64, 50, 25, 32) == {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 2.0})

check("mask_matches 一致", mask_matches(make_mask(64, 50), make_image(64, 50)))
check("mask_matches 不一致", not mask_matches(make_mask(63, 50), make_image(64, 50)))
check("mask_matches None", not mask_matches(None, make_image(64, 50)))


# ── 2. encode_qwen_edit 双图（主图 pad + mask，副图 center + 无 mask）──────────

vae = FakeVae()
clip = FakeClip()
entries = [
    {"image": make_image(64, 50), "mask": make_mask(64, 50), "ref_longest_edge": 32, "ref_crop": "pad"},
    {"image": make_image(40, 100), "mask": None, "ref_longest_edge": 40, "ref_crop": "center"},
]
cond, latent_out, custom, main_image, noise_mask = encode_qwen_edit(clip, vae, " edit it", entries)

check("custom_output 键集合", set(custom.keys()) == {
    "pad_info", "full_refs_cond", "main_image", "vae_images", "ref_latents",
    "vl_images", "full_prompt", "no_refs_cond", "mask"})
check("ref_latents 两份", len(custom["ref_latents"]) == 2)
check("vae 编码两份", len(vae.encoded) == 2)
check("latent 输出 = 主图 ref latent", latent_out["samples"] is custom["ref_latents"][0])
check("noise_mask 形状=主图画布", noise_mask.shape[1:] == custom["main_image"].shape[1:3])
check("noise_mask 只来自主图", noise_mask.shape[1:] == (32, 32))
check("pad_info 只记主图", custom["pad_info"]["width"] == 7 and custom["pad_info"]["height"] == 0)
check("pad_info scale_by", custom["pad_info"]["scale_by"] == 2.0)
check("full_prompt Picture 编号", custom["full_prompt"] ==
      "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
      "Picture 2: <|vision_start|><|image_pad|><|vision_end|> edit it")
check("vl 缩放面积≈目标", abs(custom["vl_images"][0].shape[1] * custom["vl_images"][0].shape[2] - 384 * 384) < 384)
check("vl 编号连续", [i.shape for i in custom["vl_images"]][0][1:] != [i.shape for i in custom["vl_images"]][1][1:])
check("no_refs_cond 无 reference_latents", "reference_latents" not in custom["no_refs_cond"][0][1])
check("full_refs_cond 有 reference_latents", "reference_latents" in custom["full_refs_cond"][0][1])
check("llama_template 传入", clip.calls[0]["llama_template"] is not None)

# ── 3. 无图纯文本路径 ────────────────────────────────────────────────────────

vae2 = FakeVae()
clip2 = FakeClip()
cond2, latent2, custom2, main2, nm2 = encode_qwen_edit(clip2, vae2, "hello", [])

check("纯文本不编码图片", len(vae2.encoded) == 0)
check("纯文本 latent 占位", latent2["samples"].shape == TEXT_ONLY_LATENT_SHAPE)
check("纯文本无 noise_mask", nm2 is None and latent2.get("noise_mask") is None)
check("纯文本 main_image None", main2 is None)
check("纯文本 full_prompt", custom2["full_prompt"] == "hello")
check("纯文本 vl_images 空", custom2["vl_images"] == [])

# ── 4. 节点壳 ────────────────────────────────────────────────────────────────

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
    "sfnodes.nodes.utils.qwen_edit",
    os.path.join(root, "nodes", "utils", "qwen_edit.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

Node = mod.SFQwenEditTextEncode
check("节点 CATEGORY", Node.CATEGORY == "sfnodes/model")
check("节点 FUNCTION", Node.FUNCTION == "execute")
check("节点 RETURN_TYPES", Node.RETURN_TYPES[0] == "CONDITIONING" and Node.RETURN_TYPES[1] == "LATENT"
      and Node.RETURN_TYPES[3] == "IMAGE" and Node.RETURN_TYPES[4] == "MASK")
check("节点 RETURN_NAMES", Node.RETURN_NAMES == ("conditioning", "latent", "custom_output", "main_image", "mask"))

it = Node.INPUT_TYPES()
check("required 含 clip/vae/prompt", all(k in it["required"] for k in ("clip", "vae", "prompt")))
for i in (1, 2, 3):
    for k in ("image%d" % i, "mask%d" % i, "ref_longest_edge%d" % i, "ref_crop%d" % i):
        check(f"optional 含 {k}", k in it["optional"])
check("optional 共享参数", all(k in it["optional"] for k in ("ref_upscale", "vl_target_size", "vl_crop", "vl_upscale")))
check("ref_crop 选项", it["optional"]["ref_crop1"][0] == ["pad", "center", "disabled"])

# 节点执行：image1 主图(64x50 pad+mask) + image3 副图，image2 缺席（Picture 重编号）
node = Node()
result = node.execute(
    clip=FakeClip(), vae=FakeVae(), prompt="x",
    image1=make_image(64, 50), mask1=make_mask(64, 50),
    image3=make_image(40, 100),
    ref_longest_edge1=32, ref_crop1="pad",
    ref_longest_edge3=40, ref_crop3="center",
)
_c, _l, custom3, _m, _n = result
check("跳过缺席图 Picture 重编号", custom3["full_prompt"].startswith("Picture 1:") and "Picture 2:" in custom3["full_prompt"])
check("副图无 mask 时 noise_mask 仍来自主图", _n is not None and _n.shape[1:] == (32, 32))

# mask 形状不符丢弃：image2 的 mask 尺寸错误
node2 = Node()
_r = node2.execute(clip=FakeClip(), vae=FakeVae(), prompt="x",
                   image1=make_image(64, 50), mask1=make_mask(63, 50))
check("mask 形状不符被忽略", _r[4] is None)

# 无图纯文本
node3 = Node()
_r3 = node3.execute(clip=FakeClip(), vae=FakeVae(), prompt="solo")
check("无图纯文本 latent 占位", _r3[1]["samples"].shape == TEXT_ONLY_LATENT_SHAPE)
check("无图 custom_output main_image None", _r3[3] is None)

# ── 5. Extractor ─────────────────────────────────────────────────────────────

Ext = mod.SFQwenEditOutputExtractor
check("Extractor RETURN_NAMES", Ext.RETURN_NAMES == (
    "pad_info", "full_refs_cond", "main_image", "vae_images", "ref_latents",
    "vl_images", "full_prompt", "no_refs_cond", "mask"))
check("Extractor INPUT_TYPES custom_output", "custom_output" in Ext.INPUT_TYPES()["required"])

ext = Ext()
out = ext.extract(custom)
names = ("pad_info", "full_refs_cond", "main_image", "vae_images", "ref_latents",
         "vl_images", "full_prompt", "no_refs_cond", "mask")
for i, name in enumerate(names):
    check(f"Extractor {name} 一致", out[i] is custom[name])

out2 = ext.extract({})
check("Extractor 缺键返回 None", all(v is None for v in out2))

# ── 汇总 ─────────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"{len(failures)} failures: {failures}")
    sys.exit(1)
print("All tests passed.")
