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

    def __mul__(self, k):
        return FakeTensor(self._arr * k)

    __rmul__ = __mul__


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
    resize_to_target,
    zero_conditioning,
    TEXT_ONLY_LATENT_SHAPE,
    DEFAULT_LLAMA_TEMPLATE,
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
        # 与真实 VAE.encode 一致返回张量（值为 1，便于断言 strength 缩放）
        return FakeTensor(np.ones((img.shape[0], 16, img.shape[1] // 8, img.shape[2] // 8)))


class FakeClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, images=None, llama_template=None):
        self.calls.append({"prompt": prompt, "images": [i.shape for i in (images or [])],
                           "llama_template": llama_template})
        return ["TOKENS"]

    def encode_from_tokens_scheduled(self, tokens):
        return [[FakeTensor(np.ones((1, 4))), {"pooled_output": FakeTensor(np.ones((1, 4)))}]]


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
check("纯文本仍透传非空 llama_template（对齐 EditUtils）",
      clip2.calls[0]["llama_template"] == DEFAULT_LLAMA_TEMPLATE)
clip_txt = FakeClip()
encode_qwen_edit(clip_txt, FakeVae(), "hello", [], llama_template="")
check("空 llama_template 不传（走 tokenizer 默认）", clip_txt.calls[0]["llama_template"] is None)

# ── 3.5 ref_resize_mode / to_ref / to_vl / rope offsets ──────────────────────

from sf_utils.qwen_edit import scale_reference  # noqa: E402

check("scale_reference area 与 longest_edge 不同",
      scale_reference(100, 100, 50, "area") == (200, 200)
      and scale_reference(100, 100, 50, "longest_edge") == (50, 50))

# to_ref=False 只做 VL：ref_latents 不增长；to_vl=False 无 Picture 编号
vae3 = FakeVae()
clip3 = FakeClip()
entries3 = [
    {"image": make_image(32, 32), "mask": None, "ref_longest_edge": 32, "ref_crop": "center"},
    {"image": make_image(32, 32), "mask": None, "ref_longest_edge": 32, "ref_crop": "center",
     "to_ref": False},
    {"image": make_image(32, 32), "mask": None, "ref_longest_edge": 32, "ref_crop": "center",
     "to_ref": False, "to_vl": False},
]
_cond3, _lat3, custom3, _main3, _nm3 = encode_qwen_edit(clip3, vae3, "p", entries3)
check("to_ref=False 不进 ref_latents", len(custom3["ref_latents"]) == 1)
check("to_ref=False 仍进 VL", len(custom3["vl_images"]) == 2)
check("to_vl=False 不编号", "Picture 3:" not in custom3["full_prompt"])

# rope offsets 非零 → conditioning 写 reference_rope_offsets
vae4 = FakeVae()
clip4 = FakeClip()
entries4 = [{"image": make_image(32, 32), "mask": None, "ref_longest_edge": 32,
             "ref_crop": "center", "rope_x_offset": 8, "rope_y_offset": 8}]
_cond4, _lat4, custom4, _main4, _nm4 = encode_qwen_edit(clip4, vae4, "p", entries4)
extra = custom4["full_refs_cond"][0][1]
check("rope offsets 写入 conditioning",
      extra.get("reference_rope_offsets") == [(8, 8)] and "reference_latents" in extra)

# 全零 rope offsets 不写该键（保持旧行为）
vae5 = FakeVae()
entries5 = [{"image": make_image(32, 32), "mask": None, "ref_longest_edge": 32,
             "ref_crop": "center"}]
_cond5, _lat5, custom5, _main5, _nm5 = encode_qwen_edit(FakeClip(), vae5, "p", entries5)
check("零 rope offsets 不写键", "reference_rope_offsets" not in custom5["full_refs_cond"][0][1])

# ── 3.7 Easy 语义：目标尺寸统一 / init_image / strength / zero_conditioning ──

_patch = make_image(100, 50).movedim(-1, 1)  # [B,C,H=100,W=50]
check("resize_to_target crop/stretch/pad 形状", (
    resize_to_target(_patch, 64, 80).shape,
    resize_to_target(_patch, 64, 80, "stretch").shape,
    resize_to_target(_patch, 64, 80, "pad").shape) == ((1, 3, 64, 80),) * 3)
check("resize_to_target 目标下限 32", resize_to_target(_patch, 8, 8, "stretch").shape == (1, 3, 32, 32))

# init_image：初始 latent 只来自它；参考图统一到目标尺寸；mask→noise_mask
vae6 = FakeVae()
clip6 = FakeClip()
entries6 = [
    {"image": make_image(100, 50), "mask": None, "ref_longest_edge": 32, "ref_crop": "center"},
    {"image": make_image(40, 100), "mask": None, "ref_longest_edge": 40, "ref_crop": "center"},
]
_cond6, lat6, custom6, main6, nm6 = encode_qwen_edit(
    clip6, vae6, "p", entries6,
    init_image=make_image(64, 40), init_mask=make_mask(64, 40), ref_target_mode="crop")
check("init 路径 vae 编码 3 次（init+2 ref）", len(vae6.encoded) == 3)
check("init 初始 latent 形状", lat6["samples"].shape == (1, 16, 8, 5))
check("init noise_mask 尺寸=latent_image", nm6.shape == (1, 64, 40))
check("init 初始 latent 独立于 ref", lat6["samples"] is not custom6["ref_latents"][0])
check("目标尺寸统一 ref latent 形状", all(r.shape == (1, 16, 8, 5) for r in custom6["ref_latents"]))
check("ref vae 图像统一到目标", all(v.shape == (1, 64, 40, 3) for v in custom6["vae_images"]))
check("VL 仍用原图（面积 384²）", len(custom6["vl_images"]) == 2 and custom6["vl_images"][0].shape[1:3] != (64, 40))

# ref_strength：只缩放写入 conditioning 的副本，初始 latent 保持未缩放
vae7 = FakeVae()
entries7 = [
    {"image": make_image(64, 64), "mask": None, "ref_longest_edge": 32, "ref_crop": "center", "ref_strength": 0.5},
    {"image": make_image(64, 64), "mask": None, "ref_longest_edge": 32, "ref_crop": "center"},
]
_cond7, lat7, custom7, _m7, _n7 = encode_qwen_edit(FakeClip(), vae7, "p", entries7, ref_target_mode="stretch")
_raw_sum = float(np.ones((1, 16, 8, 8)).sum())
check("strength 缩放 ref latent", float(custom7["ref_latents"][0]._arr.sum()) == 0.5 * _raw_sum)
check("strength=1 的 ref 不缩放", float(custom7["ref_latents"][1]._arr.sum()) == _raw_sum)
check("初始 latent 不缩放", float(lat7["samples"]._arr.sum()) == _raw_sum)
check("strength 缩放产生新张量", custom7["ref_latents"][0] is not lat7["samples"])

# zero_conditioning：cond/pooled 置零、dict 字段保留；pooled_output=None 不炸（Krea2 条件实测）
_zero_in = [[FakeTensor(np.ones((1, 4))),
             {"pooled_output": FakeTensor(np.ones((1, 4))), "reference_latents": ["R"]}]]
_zero_out = zero_conditioning(_zero_in)
check("zero_conditioning cond 置零", float(_zero_out[0][0]._arr.sum()) == 0.0)
check("zero_conditioning pooled 置零", float(_zero_out[0][1]["pooled_output"]._arr.sum()) == 0.0)
check("zero_conditioning 保留 dict 字段", _zero_out[0][1]["reference_latents"] == ["R"])
_zero_none = zero_conditioning([[FakeTensor(np.ones((1, 4))),
                                 {"pooled_output": None, "conditioning_lyrics": None}]])
check("zero_conditioning pooled=None 不炸", _zero_none[0][1]["pooled_output"] is None
      and float(_zero_none[0][0]._arr.sum()) == 0.0)

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

# ── 6. SFEasyKrea2Edit ───────────────────────────────────────────────────────

Easy = mod.SFEasyKrea2Edit
check("Easy CATEGORY", Easy.CATEGORY == "sfnodes/model")
check("Easy FUNCTION", Easy.FUNCTION == "encode")
check("Easy RETURN_NAMES", Easy.RETURN_NAMES == ("positive", "zero_negative", "latent"))
_it_e = Easy.INPUT_TYPES()
check("Easy required", all(k in _it_e["required"] for k in ("clip", "vae", "prompt")))
check("Easy optional 基础", all(k in _it_e["optional"] for k in (
    "image1", "latent_image", "latent_mask", "auto_resize", "vl_size",
    "system_prompt", "reference_latents_method")))
check("Easy hidden 状态", "SFEasyKrea2EditState" in _it_e["hidden"])
check("Easy auto_resize 选项", _it_e["optional"]["auto_resize"][0] == ["crop", "pad", "stretch"])
check("Easy system_prompt 默认=Qwen 编辑模板",
      _it_e["optional"]["system_prompt"][1]["default"].startswith("Describe the key features"))

# 动态槽位：image1 显式 + image2 走 kwargs（前端追加槽）；独立 latent_image + mask + strength
easy = Easy()
clip_e, vae_e = FakeClip(), FakeVae()
pos, neg, lat = easy.encode(
    clip=clip_e, vae=vae_e, prompt="make it real",
    image1=make_image(64, 40),
    image2=make_image(100, 50),
    latent_image=make_image(64, 40),
    latent_mask=make_mask(64, 40),
    SFEasyKrea2EditState='{"strengths": {"2": 0.25}}',
)
check("Easy vae 编码 init+2 ref", len(vae_e.encoded) == 3)
check("Easy 初始 latent 形状", lat["samples"].shape == (1, 16, 8, 5))
check("Easy noise_mask 来自 init", lat["noise_mask"].shape == (1, 64, 40))
_refs = pos[0][1].get("reference_latents")
check("Easy reference_latents 两份", _refs is not None and len(_refs) == 2)
check("Easy 逐图 strength 生效", float(_refs[1]._arr.sum()) == 0.25 * float(_refs[0]._arr.sum()))
check("Easy 写入 reference_latents_method",
      pos[0][1].get("reference_latents_method") == "index_timestep_zero")
check("Easy zero_negative cond 置零", float(neg[0][0]._arr.sum()) == 0.0)
check("Easy zero_negative 保留 reference_latents", neg[0][1].get("reference_latents") is not None)
check("Easy 模板包装（Picture 1/2）", clip_e.calls[0]["prompt"].startswith("Picture 1:") and
      "Picture 2:" in clip_e.calls[0]["prompt"])

# latent_image 缺省：image1 兼作初始 latent 与参考，latent_mask 作用于主图
easy2 = Easy()
clip_e2, vae_e2 = FakeClip(), FakeVae()
_pos2, _neg2, lat2 = easy2.encode(clip=clip_e2, vae=vae_e2, prompt="p",
                                  image1=make_image(64, 40), latent_mask=make_mask(64, 40))
check("Easy 无 latent_image 时 image1 兼作初始 latent", lat2["samples"].shape == (1, 16, 8, 5))
check("Easy 无 latent_image 时 mask 作用于主图", lat2["noise_mask"].shape == (1, 64, 40))
check("Easy 无 latent_image 时单次 vae 编码", len(vae_e2.encoded) == 1)

# method 置空 = 不写
easy3 = Easy()
_pos3, _neg3, _lat3 = easy3.encode(clip=FakeClip(), vae=FakeVae(), prompt="p",
                                   image1=make_image(32, 32), reference_latents_method="")
check("Easy method 空串不写键", "reference_latents_method" not in _pos3[0][1])

# 无图纯文本（占位 latent，与旧路径一致）
easy4 = Easy()
_p4, _n4, lat4 = easy4.encode(clip=FakeClip(), vae=FakeVae(), prompt="solo")
check("Easy 无图占位 latent", lat4["samples"].shape == TEXT_ONLY_LATENT_SHAPE)

# ── 汇总 ─────────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"{len(failures)} failures: {failures}")
    sys.exit(1)
print("All tests passed.")
