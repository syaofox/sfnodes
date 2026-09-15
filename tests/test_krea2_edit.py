# Krea2 参考图编辑节点/纯逻辑后端模拟测试（python tests/test_krea2_edit.py）
# 覆盖：
#   - sf_utils.krea2_edit：get_system_prompt 前缀/后缀/{}清洗/默认；crop_with_pad_info 裁剪数学；
#     make_ref_positions 的 frame 索引 / scale_to_grid 中心对齐 / rope offsets 像素→token
#   - 节点壳：SFKrea2ModelConfig / SFKrea2EditApply（非 Krea2 直通）/ SFCropWithPadInfo /
#     SFKrea2ConfigPreparer / SFKrea2EditTextEncode（6 路输出 + 纯文本 + 非法 model_config）
# mock：FakeTensor(numpy) + fake torch/einops/comfy.ldm.common_dit/comfy.utils/node_helpers

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


# ── FakeTensor（numpy 代理，覆盖 make_ref_positions 用到的算子）─────────────────

class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float64)

    @property
    def data(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def device(self):
        return types.SimpleNamespace(type="cpu")

    @property
    def dtype(self):
        return "float32"

    def squeeze(self, dim=None):
        return FakeTensor(np.squeeze(self._arr, axis=dim))

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return FakeTensor(np.broadcast_to(self._arr, shape))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return FakeTensor(self._arr.reshape(shape))

    def repeat(self, *reps):
        return FakeTensor(np.tile(self._arr, reps))

    def movedim(self, src, dst):
        return FakeTensor(np.moveaxis(self._arr, src, dst))

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self._arr, dim))

    def flatten(self):
        return FakeTensor(self._arr.flatten())

    def __getitem__(self, idx):
        r = self._arr[idx]
        return FakeTensor(r) if isinstance(r, np.ndarray) else float(r)

    def __setitem__(self, idx, val):
        self._arr[idx] = val._arr if isinstance(val, FakeTensor) else val

    def __add__(self, o):
        return FakeTensor(self._arr + (o._arr if isinstance(o, FakeTensor) else o))

    __radd__ = __add__

    def __sub__(self, o):
        return FakeTensor(self._arr - (o._arr if isinstance(o, FakeTensor) else o))

    def __rsub__(self, o):
        return FakeTensor((o._arr if isinstance(o, FakeTensor) else o) - self._arr)

    def __mul__(self, o):
        return FakeTensor(self._arr * (o._arr if isinstance(o, FakeTensor) else o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return FakeTensor(self._arr / (o._arr if isinstance(o, FakeTensor) else o))


# ── fake torch / einops / comfy ───────────────────────────────────────────────

fake_torch = types.ModuleType("torch")
fake_torch.float32 = "float32"
fake_torch.Tensor = FakeTensor


def _zeros(*shape, **kw):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return FakeTensor(np.zeros(shape))


fake_torch.zeros = _zeros
fake_torch.arange = lambda n, **kw: FakeTensor(np.arange(n, dtype=np.float64))
fake_torch.zeros_like = lambda t, **kw: FakeTensor(np.zeros_like(t._arr))
sys.modules["torch"] = fake_torch

fake_einops = types.ModuleType("einops")


def _rearrange(x, pattern, **kw):
    if pattern.startswith("b c (h ph) (w pw) -> b (h w) (c ph pw)"):
        b, c, H, W = x.shape
        ph, pw = kw.get("ph"), kw.get("pw")
        return FakeTensor(np.zeros((b, (H // ph) * (W // pw), c * ph * pw)))
    raise AssertionError(f"unexpected rearrange pattern: {pattern}")


fake_einops.rearrange = _rearrange
sys.modules["einops"] = fake_einops

fake_comfy = types.ModuleType("comfy")
fake_ldm = types.ModuleType("comfy.ldm")
fake_common_dit = types.ModuleType("comfy.ldm.common_dit")
fake_common_dit.pad_to_patch_size = lambda x, size: x
fake_utils = types.ModuleType("comfy.utils")


def _common_upscale(samples, width, height, upscale_method="area", crop="disabled"):
    b, c = samples.shape[0], samples.shape[1]
    return FakeTensor(np.zeros((b, c, height, width)))


fake_utils.common_upscale = _common_upscale
fake_ldm.common_dit = fake_common_dit
fake_comfy.ldm = fake_ldm
fake_comfy.utils = fake_utils
sys.modules["comfy"] = fake_comfy
sys.modules["comfy.ldm"] = fake_ldm
sys.modules["comfy.ldm.common_dit"] = fake_common_dit
sys.modules["comfy.utils"] = fake_utils

fake_node_helpers = types.ModuleType("node_helpers")


def _conditioning_set_values(conditioning, values, append=True):
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


fake_node_helpers.conditioning_set_values = _conditioning_set_values
sys.modules["node_helpers"] = fake_node_helpers

# ── 纯逻辑导入 ────────────────────────────────────────────────────────────────

from sf_utils.krea2_edit import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    get_system_prompt,
    crop_with_pad_info,
    make_ref_positions,
)


# ── 1. get_system_prompt ─────────────────────────────────────────────────────

_prefix = "<|im_start|>system\n"
_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

t = get_system_prompt("")
check("sys: 空指令用内置默认", "Describe the key features" in t)
check("sys: 默认模板结构", t.startswith(_prefix) and t.endswith("<|im_start|>assistant\n"))

t = get_system_prompt(DEFAULT_INSTRUCTION)
check("sys: 自定义指令注入", DEFAULT_INSTRUCTION in t)

t = get_system_prompt(_prefix + "SYS" + _suffix)
check("sys: 前缀/后缀剥离后仅剩内容", t == _prefix + "SYS" + _suffix)

t = get_system_prompt("Do {} it")
check("sys: {} 被清洗", "Do  it" in t)

# ── 2. crop_with_pad_info ────────────────────────────────────────────────────

img = FakeTensor(np.arange(1 * 10 * 12 * 3, dtype=np.float64).reshape(1, 10, 12, 3))
cropped, scale_by = crop_with_pad_info(img, {"x": 0, "y": 0, "width": 2, "height": 1, "scale_by": 0.5})
check("crop: 形状=去 padding", cropped.shape == (1, 9, 10, 3))
check("crop: 数值=左上角内容", np.allclose(cropped.data, img.data[:, :9, :10, :]))
check("crop: scale_by 透传", scale_by == 0.5)

cropped2, _ = crop_with_pad_info(img, {})
check("crop: 空 pad_info 原样", cropped2.shape == img.shape)

# ── 3. make_ref_positions ────────────────────────────────────────────────────

ref = FakeTensor(np.zeros((1, 16, 4, 4)))
tokens, positions, total = make_ref_positions([ref], bs=1, patch_size=2, device="cpu", dtype="float32")
check("pos: total = patch token 数", total == 4 and len(tokens) == 1)
p = positions[0].data[0]  # (rh*rw, 3)
check("pos: frame index = +1", np.allclose(p[:, 0], 1.0))
check("pos: h id 行优先", np.allclose(p[:, 1], [0, 0, 1, 1]))
check("pos: w id 行优先", np.allclose(p[:, 2], [0, 1, 0, 1]))
check("pos: token 形状", tokens[0].shape == (1, 4, 16 * 2 * 2))

_, positions, _ = make_ref_positions([ref], 1, 2, "cpu", "float32", scale_to_grid=(4, 4))
p = positions[0].data[0]
check("pos: scale_to_grid 中心对齐 h", np.allclose(p[:, 1], [0.5, 0.5, 2.5, 2.5]))
check("pos: scale_to_grid 中心对齐 w", np.allclose(p[:, 2], [0.5, 2.5, 0.5, 2.5]))

_, positions, _ = make_ref_positions([ref], 1, 2, "cpu", "float32",
                                     rope_offsets=[(4, 2)])
p = positions[0].data[0]
check("pos: rope offset 生效（像素→token）",
      np.allclose(p[:, 1], [1, 1, 2, 2]) and np.allclose(p[:, 2], [2, 3, 2, 3]))

ref2 = FakeTensor(np.zeros((1, 16, 2, 2)))
_, positions, total2 = make_ref_positions([ref, ref2], 1, 2, "cpu", "float32")
check("pos: 多 ref 第二张 frame=2", np.allclose(positions[1].data[0][:, 0], 2.0) and total2 == 5)

# ── 4. importlib 加载节点模块 ────────────────────────────────────────────────

for pkg, path in [
    ("sfnodes", [root]),
    ("sfnodes.nodes", [os.path.join(root, "nodes")]),
    ("sfnodes.nodes.model", [os.path.join(root, "nodes", "model")]),
    ("sfnodes.nodes.utils", [os.path.join(root, "nodes", "utils")]),
    ("sfnodes.nodes.image", [os.path.join(root, "nodes", "image")]),
    ("sfnodes.sf_utils", [os.path.join(root, "sf_utils")]),
]:
    m = types.ModuleType(pkg)
    m.__path__ = path
    sys.modules[pkg] = m


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, *rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


krea2_edit_mod = _load("sfnodes.nodes.model.krea2_edit", ("nodes", "model", "krea2_edit.py"))
crop_pad_mod = _load("sfnodes.nodes.image.crop_with_pad", ("nodes", "image", "crop_with_pad.py"))
qwen_mod = _load("sfnodes.nodes.utils.qwen_edit", ("nodes", "utils", "qwen_edit.py"))

# ── 5. SFKrea2ModelConfig ────────────────────────────────────────────────────

MC = krea2_edit_mod.SFKrea2ModelConfig
check("modelconfig CATEGORY", MC.CATEGORY == "sfnodes/model")
check("modelconfig RETURN_NAMES", MC.RETURN_NAMES == ("model_config",))
(cfg,) = MC().configure_model("")
check("modelconfig 路由 qwen/vae_unit 8", cfg["model_name"] == "qwen" and cfg["vae_unit"] == 8)
check("modelconfig config_for", cfg["config_for"] == "krea2")
check("modelconfig 默认指令注入模板", DEFAULT_INSTRUCTION in cfg["llama_template"])
(cfg2,) = MC().configure_model("CUSTOM")
check("modelconfig 自定义指令", "CUSTOM" in cfg2["llama_template"])

# ── 6. SFKrea2EditApply（非 Krea2 直通）──────────────────────────────────────

EA = krea2_edit_mod.SFKrea2EditApply
check("editapply RETURN_NAMES", EA.RETURN_NAMES == ("model",))
it = EA.INPUT_TYPES()
check("editapply 输入顺序", list(it["optional"].keys()) ==
      ["mode", "ref_pos_match_target", "ref_kv_cache", "ref_strength", "reset_cache", "debug_log"])
check("editapply ref_strength 默认 1.0", it["optional"]["ref_strength"][1]["default"] == 1.0)

fake_model = types.SimpleNamespace(model=types.SimpleNamespace())
out = EA().apply_patch(fake_model)
check("editapply 非 Krea2 原样返回", out[0] is fake_model)

# ── 7. SFCropWithPadInfo ─────────────────────────────────────────────────────

CP = crop_pad_mod.SFCropWithPadInfo
check("croppad CATEGORY", CP.CATEGORY == "sfnodes/image")
check("croppad RETURN_NAMES", CP.RETURN_NAMES == ("cropped_image", "scale_by"))
r = CP().crop_image(img, {"x": 0, "y": 0, "width": 2, "height": 1, "scale_by": 0.5})
check("croppad 裁剪结果", r[0].shape == (1, 9, 10, 3) and r[1] == 0.5)
r2 = CP().crop_image(img, None)
check("croppad 非法 pad_info 原样返回", r2[0] is img and r2[1] == 1.0)

# ── 8. SFKrea2ConfigPreparer ─────────────────────────────────────────────────

PREP = qwen_mod.SFKrea2ConfigPreparer
check("preparer RETURN_NAMES", PREP.RETURN_NAMES == ("configs", "config"))
img_small = FakeTensor(np.zeros((1, 8, 8, 3)))
out_configs, config = PREP().prepare_config(img_small, None, ref_longest_edge=512)
check("preparer 追加一项", len(out_configs) == 1 and out_configs[0] is config)
check("preparer 默认值", config["to_ref"] is True and config["ref_crop"] == "pad"
      and config["ref_resize_mode"] == "longest_edge" and config["ref_longest_edge"] == 512)
check("preparer 无 mask 时不带 key", "mask" not in config)

prev = [{"image": img_small, "to_ref": True}]
out2, _ = PREP().prepare_config(img_small, prev)
check("preparer deepcopy 不污染原链", len(out2) == 2 and out2[0] is not prev[0] and len(prev) == 1)

mask_ok = FakeTensor(np.zeros((1, 8, 8)))
_, cfg_mask = PREP().prepare_config(img_small, None, mask=mask_ok)
check("preparer 合法 mask 保留", "mask" in cfg_mask)
mask_bad = FakeTensor(np.zeros((1, 4, 4)))
_, cfg_nomask = PREP().prepare_config(img_small, None, mask=mask_bad)
check("preparer 尺寸不符 mask 丢弃", "mask" not in cfg_nomask)

# ── 9. SFKrea2EditTextEncode ─────────────────────────────────────────────────

ENC = qwen_mod.SFKrea2EditTextEncode
check("encode RETURN_NAMES", ENC.RETURN_NAMES ==
      ("conditioning", "latent", "custom_output", "main_image", "mask", "pad_info"))


class FakeVae:
    def encode(self, img):
        return FakeTensor(np.zeros((img.shape[0], 16, img.shape[1] // 8, img.shape[2] // 8)))


class FakeClip:
    def tokenize(self, prompt, images=None, llama_template=None):
        return ["TOKENS"]

    def encode_from_tokens_scheduled(self, tokens):
        return [["COND", {}]]


(mcfg,) = MC().configure_model("")
_, cfg = PREP().prepare_config(FakeTensor(np.zeros((1, 64, 48, 3))), None,
                               ref_longest_edge=32, ref_crop="pad")
res = ENC().execute(clip=FakeClip(), vae=FakeVae(), prompt=" edit", model_config=mcfg,
                    configs=[cfg])
check("encode 6 路输出", len(res) == 6)
check("encode pad_info 输出 = custom_output.pad_info", res[5] is res[2]["pad_info"])
check("encode latent = 主图 ref latent", res[1]["samples"] is res[2]["ref_latents"][0])
check("encode main_image 非空", res[3] is not None)

res_text = ENC().execute(clip=FakeClip(), vae=FakeVae(), prompt="solo", model_config=mcfg,
                         configs=[])
check("encode 纯文本路径", res_text[3] is None and res_text[1]["samples"].shape == (1, 4, 128, 128))

try:
    ENC().execute(clip=FakeClip(), vae=FakeVae(), prompt="x", model_config=None)
    check("encode 未接 model_config 抛错", False)
except ValueError:
    check("encode 未接 model_config 抛错", True)

try:
    ENC().execute(clip=FakeClip(), vae=FakeVae(), prompt="x",
                  model_config={"model_name": "flux2klein"}, configs=[])
    check("encode 非法 model_name 抛错", False)
except ValueError:
    check("encode 非法 model_name 抛错", True)

# ── 汇总 ─────────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"{len(failures)} failures: {failures}")
    sys.exit(1)
print("All tests passed.")
