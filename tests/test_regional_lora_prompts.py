# SFRegionalLoRA 区域提示词链路模拟测试（Python 直接运行：
#   python tests/test_regional_lora_prompts.py）
# 无 torch 环境下 mock torch/safetensors/folder_paths/clip，验证：
#   - 结构：clip 为 optional 输入、hidden SFRegionsJson
#   - apply：逐区域 CLIP 编码 → prompt_tokens/prompt_blocks → info
#     prompt_mode/prompt_tokens/prompt 字段 → attn1 patch 与 wrapper 挂载
#   - session：context 追加区域提示词（cond 行保留、uncond 行置零）
#   - attn patch：返回 [B,1,S,S] bool 掩码；无 block_index/无 extra_options 不返回
#   - prompt-only：无 LoRA 只留提示词的区域、strength=0 保留、LoRA 加载失败保留
#   - 降级：encode 失败提示词忽略但 LoRA 仍在、无 clip/legacy JSON → lora_only
import importlib.util
import json
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


# ── mock torch ─────────────────────────────────────────────────────────────
class FakeTensor:
    def __init__(self, shape=(1, 16, 128, 128), device="cuda", dtype="float32", fill=1.0):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        n = self.shape[1] if len(self.shape) >= 2 else self.shape[0]
        self.rows = [[float(fill)] * n for _ in range(self.shape[0])]

    def to(self, *a, **k):
        return self

    def dim(self):
        return len(self.shape)

    def numel(self):
        return int(np.prod(self.shape))

    def expand(self, *sizes):
        shape = tuple(s if s != -1 else self.shape[i] for i, s in enumerate(sizes))
        out = FakeTensor(shape, self.device, self.dtype)
        out.rows = [list(self.rows[0]) for _ in range(shape[0])]
        return out

    def clone(self):
        out = FakeTensor(self.shape, self.device, self.dtype)
        out.rows = [list(r) for r in self.rows]
        return out

    def __setitem__(self, idx, val):
        if isinstance(idx, int):
            self.rows[idx] = [float(val)] * self.shape[1]
        else:
            raise AssertionError(f"unexpected setitem key {idx!r}")

    def __mul__(self, s):
        return self

    def __rmul__(self, s):
        return self

    def __array__(self, dtype=None):
        return np.zeros(self.shape, dtype=np.float32)


def _cat(tensors, dim=1, **k):
    assert dim == 1, "only dim=1 cat supported in this test"
    b = tensors[0].shape[0]
    seq = sum(t.shape[1] for t in tensors)
    out = FakeTensor((b, seq, tensors[0].shape[2]), tensors[0].device, tensors[0].dtype)
    out.cat_parts = list(tensors)
    return out


class _Finfo:
    min = -3.4e38


torch_mod = types.ModuleType("torch")
torch_mod.bfloat16 = "bfloat16"
torch_mod.float32 = "float32"
torch_mod.long = "long"
torch_mod.bool = "bool"
torch_mod.is_tensor = lambda x: isinstance(x, FakeTensor)
torch_mod.from_numpy = lambda a, **k: FakeTensor(shape=a.shape)
torch_mod.zeros = lambda *a, **k: FakeTensor()
torch_mod.zeros_like = lambda t, **k: FakeTensor(shape=t.shape)
torch_mod.arange = lambda *a, **k: FakeTensor()
torch_mod.cat = _cat
torch_mod.finfo = lambda dt: _Finfo()
torch_nn = types.ModuleType("torch.nn")
torch_nn.Linear = type("Linear", (), {})
torch_mod.nn = torch_nn
sys.modules["torch"] = torch_mod

# ── mock safetensors / folder_paths ────────────────────────────────────────
KREA2_KEYS = ("blocks.0.attn.wq", "blocks.0.attn.wk", "blocks.0.mlp.gate")


def _make_sd(layers=KREA2_KEYS, seed=0):
    sd = {}
    for L in layers:
        sd[f"lora_unet_{L}.lora_down.weight"] = FakeTensor(shape=(4, 8))
        sd[f"lora_unet_{L}.lora_up.weight"] = FakeTensor(shape=(8, 4))
        sd[f"lora_unet_{L}.alpha"] = np.float32(4.0)
    return sd


SD_BY_PATH = {
    "/loras/a.safetensors": _make_sd(seed=1),
    "/loras/b.safetensors": _make_sd(seed=2),
}

safetensors_mod = types.ModuleType("safetensors")
st_mod = types.ModuleType("safetensors.torch")
st_mod.load_file = lambda path: SD_BY_PATH[path]
safetensors_mod.torch = st_mod
sys.modules["safetensors"] = safetensors_mod
sys.modules["safetensors.torch"] = st_mod

fp_mod = types.ModuleType("folder_paths")
fp_mod.get_full_path = lambda cat, name: f"/loras/{name}"
sys.modules["folder_paths"] = fp_mod

# ── 注册 sfnodes 包结构（相对导入）─────────────────────────────────────────
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.model"); pkg3.__path__ = [os.path.join(root, "nodes", "model")]; sys.modules["sfnodes.nodes.model"] = pkg3

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.model.regional_lora",
    os.path.join(root, "nodes", "model", "regional_lora.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

SFRegionalLoRA = mod.SFRegionalLoRA

# ── fake clip / patcher / DiT ──────────────────────────────────────────────
class FakeClip:
    def __init__(self, lengths=None, fail=()):
        self.lengths = lengths or {}
        self.fail = set(fail)
        self.calls = []

    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        text = tokens["text"]
        self.calls.append(text)
        if text in self.fail:
            raise RuntimeError("boom")
        n = self.lengths.get(text, max(1, len(text.split())))
        return [[FakeTensor(shape=(1, n, 8)), {}]]


class FakeHandle:
    def __init__(self, owner):
        self.owner = owner

    def remove(self):
        self.owner.hook = None


class FakeModule:
    def __init__(self):
        self.weight = True
        self.hook = None

    def register_forward_hook(self, fn):
        self.hook = fn
        return FakeHandle(self)


class FakeDiT:
    def __init__(self, names):
        self._mods = [(n, FakeModule()) for n in names]

    def named_modules(self):
        return list(self._mods)

    def parameters(self):
        return []


class FakeModel:
    def __init__(self, names):
        self.diffusion_model = FakeDiT(names)


class FakePatcher:
    def __init__(self, names=KREA2_KEYS):
        self.model = FakeModel(names)
        self.wrapper = None
        self.attn1_patch = None
        self.clone_called = 0

    def clone(self):
        self.clone_called += 1
        return self

    def add_wrapper_with_key(self, enum, key, wrapper):
        self.wrapper = wrapper

    def add_wrapper(self, enum, wrapper):
        self.wrapper = wrapper

    def set_model_attn1_patch(self, patch):
        self.attn1_patch = patch


def regions_json(entries):
    rows = []
    for i, e in enumerate(entries):
        row = {"lora": e.get("lora", "None"), "strength": e.get("strength", 1.0),
               "enable": e.get("enable", True),
               "x": e.get("x", i / max(1, len(entries))), "y": 0.0,
               "w": e.get("w", 1.0 / max(1, len(entries))), "h": 1.0}
        if "prompt" in e:
            row["prompt"] = e["prompt"]
        rows.append(row)
    return json.dumps(rows)


def run_session(patcher, x_shape=(1, 16, 128, 128), ctx_shape=(2, 5, 8), cond=(0, 1)):
    """Run the mounted wrapper once; returns (result, captured args, hooks_ok)."""
    captured = {}

    def executor(*a, **k):
        captured["args"] = a
        captured["kwargs"] = k
        captured["hooks_ok"] = all(m.hook is not None
                                   for _, m in patcher.model.diffusion_model._mods)
        return "ran"

    ctx = FakeTensor(shape=ctx_shape)
    res = patcher.wrapper(executor, FakeTensor(shape=x_shape), None, ctx, None, None,
                          {"cond_or_uncond": list(cond)})
    captured["ctx_in"] = ctx
    return res, captured


node = SFRegionalLoRA()

# ── 结构 ───────────────────────────────────────────────────────────────────
it = node.INPUT_TYPES()
check("structure: clip optional input", "clip" in it.get("optional", {}))
check("structure: hidden SFRegionsJson", "SFRegionsJson" in it["hidden"])
check("structure: RETURN_TYPES", node.RETURN_TYPES == ("MODEL", "IMAGE", "STRING"))

# ── apply：逐区域编码 + info + patch/wrapper 挂载 ──────────────────────────
patcher = FakePatcher()
clip = FakeClip(lengths={"red hair": 2, "blue eyes": 3})
out = node.apply(model=patcher, canvas_width=512, canvas_height=512, clip=clip,
                 SFRegionsJson=regions_json([
                     {"lora": "a.safetensors", "prompt": "red hair"},
                     {"lora": "b.safetensors", "prompt": "blue eyes"}]))
info = json.loads(out[2])
check("apply: clip encodes each region", clip.calls == ["red hair", "blue eyes"])
check("apply: info prompt_mode regional", info["prompt_mode"] == "regional")
check("apply: info prompt text kept",
      info["regions"][0]["prompt"] == "red hair" and info["regions"][1]["prompt"] == "blue eyes")
check("apply: info prompt_tokens", info["prompt_tokens"] == 5)
check("apply: attn1 patch mounted", patcher.attn1_patch is not None)
check("apply: wrapper mounted", patcher.wrapper is not None)

# ── session：context 追加 + cond/uncond 行处理 + hook 注册 ─────────────────
res, captured = run_session(patcher)
check("session: executor result", res == "ran")
check("session: hooks registered during run", captured["hooks_ok"] is True)
check("session: hooks removed after run",
      all(m.hook is None for _, m in patcher.model.diffusion_model._mods))
ctx_out = captured["args"][2]
check("session: context extended by prompt tokens", ctx_out.shape == (2, 5 + 5, 8))
extra = ctx_out.cat_parts[1]
check("session: cond row keeps real prompt tokens", all(v != 0 for v in extra.rows[0]))
check("session: uncond row zeroed", all(v == 0 for v in extra.rows[1]))

# ── attn patch：掩码形状与守卫 ────────────────────────────────────────────
q = FakeTensor(shape=(2, 48, 18, 64))
k = FakeTensor(shape=(2, 12, 18, 64))
res_mask = patcher.attn1_patch(q, k, None, None, None, {"block_index": 0})
check("patch: returns attn_mask", "attn_mask" in res_mask)
check("patch: mask shape [B,1,S,S]", res_mask["attn_mask"].shape == (2, 1, 18, 18))
check("patch: no block_index -> no mask",
      patcher.attn1_patch(q, k, None, None, None, {}) == {})
check("patch: no extra_options -> no mask",
      patcher.attn1_patch(q, k, None, None, None, None) == {})

# ── prompt-only：无 LoRA 只留提示词的区域 ─────────────────────────────────
patcher_p = FakePatcher()
clip_p = FakeClip(lengths={"a castle": 2, "a knight": 2})
out_p = node.apply(model=patcher_p, clip=clip_p, SFRegionsJson=regions_json([
    {"lora": "None", "prompt": "a castle"},
    {"lora": "None", "prompt": "a knight"}]))
info_p = json.loads(out_p[2])
check("prompt-only: two regions kept", info_p["n_regions"] == 2)
check("prompt-only: no LoRA layers",
      all(r["layers_total"] == 0 for r in info_p["regions"]))
check("prompt-only: strength-0 region prompt kept", json.loads(node.apply(
    model=FakePatcher(), clip=clip_p,
    SFRegionsJson=regions_json([
        {"lora": "a.safetensors", "strength": 0.0, "prompt": "a castle"}]))[2]
)["n_regions"] == 1)
check("prompt-only: attn1 patch mounted", patcher_p.attn1_patch is not None)

# ── load-fail：加载失败区域按有无提示词决定保留/跳过 ──────────────────────
out_lf = node.apply(model=FakePatcher(), clip=FakeClip(lengths={"kept": 2}),
                    SFRegionsJson=regions_json([
                        {"lora": "missing.safetensors", "prompt": "kept"},
                        {"lora": "missing2.safetensors"}]))
info_lf = json.loads(out_lf[2])
check("load-fail: prompt-only region survives",
      info_lf["n_regions"] == 1 and info_lf["regions"][0]["prompt"] == "kept")
check("load-fail: plain region skipped", info_lf["regions"][0]["lora"] == "missing.safetensors")

# ── encode-fail：提示词忽略但 LoRA 仍生效、无提示词时不再挂 patch ─────────
patcher_ef = FakePatcher()
clip_ef = FakeClip(fail=("boom",))
out_ef = node.apply(model=patcher_ef, clip=clip_ef, SFRegionsJson=regions_json([
    {"lora": "a.safetensors", "prompt": "boom"}]))
info_ef = json.loads(out_ef[2])
check("encode-fail: lora still armed",
      info_ef["n_regions"] == 1 and info_ef["regions"][0]["layers_matched"] == 3)
check("encode-fail: prompt ignored", info_ef["prompt_mode"] == "lora_only"
      and info_ef["prompt_tokens"] == 0)
check("encode-fail: no attn patch", patcher_ef.attn1_patch is None)

# ── no-clip：纯 LoRA 模式，context 不被改动 ───────────────────────────────
patcher_nc = FakePatcher()
out_nc = node.apply(model=patcher_nc, clip=None, SFRegionsJson=regions_json([
    {"lora": "a.safetensors", "prompt": "ignored"}]))
info_nc = json.loads(out_nc[2])
check("no-clip: lora-only mode", info_nc["prompt_mode"] == "lora_only")
check("no-clip: no attn patch", patcher_nc.attn1_patch is None)
_, captured_nc = run_session(patcher_nc)
check("no-clip: context untouched", captured_nc["args"][2] is captured_nc["ctx_in"])

# ── legacy JSON：无 prompt 字段 → lora_only ───────────────────────────────
patcher_lg = FakePatcher()
out_lg = node.apply(model=patcher_lg, clip=FakeClip(),
                    SFRegionsJson=regions_json([{"lora": "a.safetensors"}]))
check("legacy json: no prompt -> lora_only",
      json.loads(out_lg[2])["prompt_mode"] == "lora_only")
check("legacy json: no attn patch", patcher_lg.attn1_patch is None)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
