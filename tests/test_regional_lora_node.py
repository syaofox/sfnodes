# SFRegionalLoRA 节点链路模拟测试（Python 直接运行：python tests/test_regional_lora_node.py）
# 无 torch 环境下 mock torch/safetensors/folder_paths，验证：
#   - 节点结构：INPUT_TYPES（hidden SFRegionsJson）、RETURN_TYPES、CATEGORY、注册键
#   - apply 全链路：regions 解析 → 矩阵加载 → plan + per-region 匹配诊断 →
#     wrapper 挂载 → preview/info 输出
#   - 诊断回归：region 的 LoRA 键与模型不匹配时 matched=0 且不报错（第二个
#     LoRA 失效可见性的核心）
#   - 异常路径：加载失败 skip、无 active region 直通
#   - session.run：executor 透传、hook 注册/移除
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
    def __init__(self, shape=(1, 16, 128, 128), device="cuda"):
        self.shape = tuple(shape)
        self.device = device

    def to(self, *a, **k):
        return self

    def dim(self):
        return len(self.shape)

    def __mul__(self, s):
        return self

    def __rmul__(self, s):
        return self

    def __array__(self, dtype=None):
        return np.zeros(self.shape, dtype=np.float32)


torch_mod = types.ModuleType("torch")
torch_mod.bfloat16 = "bfloat16"
torch_mod.long = "long"
torch_mod.float32 = "float32"
torch_mod.is_tensor = lambda x: isinstance(x, FakeTensor) or isinstance(x, np.ndarray)
torch_mod.from_numpy = lambda a, **k: FakeTensor(shape=a.shape)
torch_mod.zeros = lambda *a, **k: FakeTensor()
torch_mod.zeros_like = lambda *a, **k: FakeTensor()
torch_mod.arange = lambda *a, **k: FakeTensor()
torch_nn = types.ModuleType("torch.nn")
torch_nn.Linear = type("Linear", (), {})
torch_mod.nn = torch_nn
sys.modules["torch"] = torch_mod

# ── mock safetensors ───────────────────────────────────────────────────────
KREA2_KEYS = ("blocks.0.attn.wq", "blocks.0.attn.wk", "blocks.0.mlp.gate")
FLUX_KEYS = ("blocks.0.attn1.to_q", "blocks.0.attn1.to_k")


def _make_sd(layers, seed=0):
    rng = np.random.default_rng(seed)
    sd = {}
    for L in layers:
        sd[f"lora_unet_{L}.lora_down.weight"] = FakeTensor(shape=(4, 8))
        sd[f"lora_unet_{L}.lora_up.weight"] = FakeTensor(shape=(8, 4))
        sd[f"lora_unet_{L}.alpha"] = np.float32(4.0)
    return sd


SD_BY_PATH = {
    "/loras/a.safetensors": _make_sd(KREA2_KEYS, seed=1),
    "/loras/b.safetensors": _make_sd(KREA2_KEYS, seed=2),
    "/loras/flux.safetensors": _make_sd(FLUX_KEYS, seed=3),
}

safetensors_mod = types.ModuleType("safetensors")
st_mod = types.ModuleType("safetensors.torch")
st_mod.load_file = lambda path: SD_BY_PATH[path]
safetensors_mod.torch = st_mod  # import safetensors.torch 后属性访问路径
sys.modules["safetensors"] = safetensors_mod
sys.modules["safetensors.torch"] = st_mod

# ── mock folder_paths ──────────────────────────────────────────────────────
fp_mod = types.ModuleType("folder_paths")
fp_mod.get_full_path = lambda cat, name: f"/loras/{name}"
sys.modules["folder_paths"] = fp_mod

# ── 注册 sfnodes 包结构（相对导入 from ...sf_utils...）──
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

# ── fake model patcher / DiT ───────────────────────────────────────────────
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
    def __init__(self, names):
        self.model = FakeModel(names)
        self.wrapper = None
        self.clone_called = 0

    def clone(self):
        self.clone_called += 1
        return self

    def add_wrapper_with_key(self, enum, key, wrapper):
        self.wrapper = wrapper

    def add_wrapper(self, enum, wrapper):
        self.wrapper = wrapper


def regions_json(lora0="a.safetensors", lora1="b.safetensors",
                 enable0=True, enable1=True, str0=1.0, str1=1.0):
    return json.dumps([
        {"lora": lora0, "strength": str0, "enable": enable0, "x": 0, "y": 0, "w": 0.5, "h": 1},
        {"lora": lora1, "strength": str1, "enable": enable1, "x": 0.5, "y": 0, "w": 0.5, "h": 1},
    ])


node = SFRegionalLoRA()

# ── 结构 ───────────────────────────────────────────────────────────────────
it = node.INPUT_TYPES()
check("structure: hidden SFRegionsJson", "SFRegionsJson" in it["hidden"])
check("structure: required model/canvas/params",
      set(it["required"]) == {"model", "canvas_width", "canvas_height",
                              "base_strength", "seam_feather", "sparse_threshold"})
check("structure: RETURN_TYPES", node.RETURN_TYPES == ("MODEL", "IMAGE", "STRING"))
check("structure: CATEGORY", node.CATEGORY == "sfnodes/model")
check("structure: DESCRIPTION", bool(node.DESCRIPTION))
check("structure: default regions json parses to 2",
      len(mod.parse_regions(mod.DEFAULT_REGIONS_JSON)) == 2)

# ── apply：双 region 正常路径 ─────────────────────────────────────────────
patcher = FakePatcher([n for _, n in
                       [("blocks.0.attn.wq", None), ("blocks.0.attn.wk", None),
                        ("blocks.0.mlp.gate", None)]])
# 重建带真实 module 的 DiT
patcher.model.diffusion_model = FakeDiT(
    ["blocks.0.attn.wq", "blocks.0.attn.wk", "blocks.0.mlp.gate"])
out = node.apply(model=patcher, canvas_width=512, canvas_height=512,
                 SFRegionsJson=regions_json())
check("apply: returns (patched, preview, info)", len(out) == 3 and out[0] is patcher)
check("apply: preview shape [1,512,512,3]", tuple(out[1].shape) == (1, 512, 512, 3))
info = json.loads(out[2])
check("apply: n_regions=2", info["n_regions"] == 2)
check("apply: both regions matched 3/3",
      info["regions"][0]["layers_matched"] == 3 and info["regions"][1]["layers_matched"] == 3)
check("apply: strengths passed", info["regions"][0]["strength"] == 1.0)
check("apply: clone used", patcher.clone_called == 1)
check("apply: wrapper mounted", patcher.wrapper is not None)

# ── session.run：executor 透传 + hook 注册/移除 ────────────────────────────
diag = patcher.model.diffusion_model._mods
seen = {}
def executor(*a, **k):
    seen["hooks_while_running"] = all(m.hook is not None for _, m in diag)
    return "ran"
res = patcher.wrapper(executor, FakeTensor(), None, None)
check("session: executor result", res == "ran")
check("session: hooks registered during run", seen.get("hooks_while_running") is True)
check("session: hooks removed after run", all(m.hook is None for _, m in diag))

# ── 诊断回归：region 2 的 LoRA 键不匹配（Flux 键）→ matched=0，不报错 ─────
patcher2 = FakePatcher([])
patcher2.model.diffusion_model = FakeDiT(
    ["blocks.0.attn.wq", "blocks.0.attn.wk", "blocks.0.mlp.gate"])
out2 = node.apply(model=patcher2, canvas_width=512, canvas_height=512,
                  SFRegionsJson=regions_json(lora1="flux.safetensors"))
info2 = json.loads(out2[2])
check("diag: region1 matched 3", info2["regions"][0]["layers_matched"] == 3)
check("diag: region2 (flux keys) matched 0",
      info2["regions"][1]["layers_matched"] == 0)
check("diag: file-layer count still reported (2 keys in flux file)",
      info2["regions"][1]["layers_total"] == 2)

# ── 异常路径 ───────────────────────────────────────────────────────────────
# 无 active region → 直通 model
patcher3 = FakePatcher([])
out3 = node.apply(model=patcher3, canvas_width=512, canvas_height=512,
                  SFRegionsJson=regions_json(enable0=False, enable1=False))
check("no-active: model passed through", out3[0] is patcher3)
check("no-active: info n_regions=0", json.loads(out3[2])["n_regions"] == 0)

# 文件加载失败 → 该 region skip，另一 region 照常
patcher4 = FakePatcher([])
patcher4.model.diffusion_model = FakeDiT(
    ["blocks.0.attn.wq", "blocks.0.attn.wk", "blocks.0.mlp.gate"])
out4 = node.apply(model=patcher4, canvas_width=512, canvas_height=512,
                  SFRegionsJson=regions_json(lora0="missing.safetensors"))
info4 = json.loads(out4[2])
check("load-fail: surviving region only", info4["n_regions"] == 1
      and info4["regions"][0]["lora"] == "b.safetensors")

# strength=0 → region 不 active
patcher5 = FakePatcher([])
out5 = node.apply(model=patcher5, canvas_width=512, canvas_height=512,
                  SFRegionsJson=regions_json(str1=0.0))
check("zero-strength: region1 skipped", json.loads(out5[2])["n_regions"] == 1)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
