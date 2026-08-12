# SFPowerLoraLoader ortho_gs 测试（Python 直接运行：python tests/test_power_lora_ortho.py）
# 覆盖：
#   - INPUT_TYPES：merge_method combo（linear 默认 + tooltip）
#   - 加载路径分派：linear 走 LoraLoader().load_lora（顺序回归）；ortho_gs 走
#     lora_ortho_load.ortho_apply（≥2 行才正交化）；key map 失败/单行回落线性
#   - 归一化交互：ortho 用归一化后强度
#   - preset 优先 + ortho 分派；model None 直通；无 clip sc=0
import importlib.util
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── mock comfy / nodes / torch / folder_paths（本机无 ComfyUI 运行时）────────
comfy = types.ModuleType("comfy"); comfy.__path__ = []; sys.modules["comfy"] = comfy
comfy_utils = types.ModuleType("comfy.utils"); sys.modules["comfy.utils"] = comfy_utils
comfy_sd = types.ModuleType("comfy.sd"); sys.modules["comfy.sd"] = comfy_sd
comfy_lora = types.ModuleType("comfy.lora"); sys.modules["comfy.lora"] = comfy_lora
comfy_lora_convert = types.ModuleType("comfy.lora_convert"); sys.modules["comfy.lora_convert"] = comfy_lora_convert
comfy.utils = comfy_utils
comfy.sd = comfy_sd
comfy.lora = comfy_lora
comfy.lora_convert = comfy_lora_convert

torch = types.ModuleType("torch"); sys.modules["torch"] = torch

class MockTensor:
    def __init__(self, name, dim=2, dtype="fp16"):
        self.name = name
        self._dim = dim
        self.dtype = dtype
        self.device = "cpu"
    def dim(self):
        return self._dim
    def float(self):
        return self
    def to(self, dtype):
        self.dtype = dtype
        return self

torch.Tensor = MockTensor

class FakeAdapter:
    def __init__(self, up, down, alpha=1.0, dora=None):
        self.weights = (up, down, alpha, None, dora, None)

LORAS_DIR = tempfile.mkdtemp(prefix="sf_power_ortho_test_")

def fake_get_filename_list(folder):
    if folder == "loras":
        return ["a.safetensors", "b.safetensors", "c.safetensors"]
    return []

def fake_get_full_path(folder, name):
    if folder != "loras":
        return None
    return os.path.join(LORAS_DIR, name.replace("/", os.sep))

folder_paths = types.ModuleType("folder_paths"); sys.modules["folder_paths"] = folder_paths
folder_paths.get_filename_list = fake_get_filename_list
folder_paths.get_full_path = fake_get_full_path
folder_paths.get_user_directory = lambda: tempfile.mkdtemp(prefix="sf_power_ortho_user_")

# ── fake comfy.lora / comfy.lora_convert / comfy.utils / nodes ───────────────
UNET_KEYS = {"lora_unet_k1": "diffusion_model.k1.weight", "lora_unet_k2": "diffusion_model.k2.weight"}
CLIP_KEYS = {"lora_te_k1": "clip_h.k1.weight"}
unet_keys_fail = [False]

def fake_model_lora_keys_unet(model, key_map=None):
    if unet_keys_fail[0]:
        raise RuntimeError("boom")
    return dict(UNET_KEYS)

def fake_model_lora_keys_clip(model, key_map=None):
    return dict(CLIP_KEYS)

comfy_lora.model_lora_keys_unet = fake_model_lora_keys_unet
comfy_lora.model_lora_keys_clip = fake_model_lora_keys_clip

LORA_PATCHES = {}  # basename -> {model_key: patch}
load_lora_calls = []

def fake_load_lora(lora_sd, key_map, log_missing=True):
    load_lora_calls.append(lora_sd["__name__"])
    values = set(key_map.values())
    return {k: p for k, p in lora_sd.get("__patches__", {}).items() if k in values}

comfy_lora.load_lora = fake_load_lora

convert_calls = []

def fake_convert_lora(sd):
    convert_calls.append(sd["__name__"])
    return sd

comfy_lora_convert.convert_lora = fake_convert_lora

def fake_load_torch_file(path, safe_load=True, return_metadata=False):
    name = os.path.basename(path)
    sd = {"__name__": name, "__patches__": LORA_PATCHES.get(name, {})}
    if return_metadata:
        return sd, {"trigger": "meta_" + name}
    return sd

comfy_utils.load_torch_file = fake_load_torch_file

# 官方节点包装（顺序路径）：LoraLoader().load_lora(model, clip, name, sm, sc)
lora_calls = []

class FakeLoraLoader:
    def load_lora(self, model, clip, lora_name, sm, sc):
        lora_calls.append((lora_name, sm, sc))
        return ("m" + str(model), "c" + str(clip) if clip is not None else None)

nodes_mod = types.ModuleType("nodes"); sys.modules["nodes"] = nodes_mod
nodes_mod.LoraLoader = FakeLoraLoader

# ── 注册 sfnodes 包结构（相对导入需要）──────────────────────────────────────
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.model"); pkg3.__path__ = [os.path.join(root, "nodes", "model")]; sys.modules["sfnodes.nodes.model"] = pkg3
pkg4 = types.ModuleType("sfnodes.sf_utils"); pkg4.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg4

def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

load_as("sfnodes.sf_utils.logger", os.path.join(root, "sf_utils", "logger.py"))
load_as("sfnodes.sf_utils.common", os.path.join(root, "sf_utils", "common.py"))
load_as("sfnodes.sf_utils.lora_reader", os.path.join(root, "sf_utils", "lora_reader.py"))
ortho_mod = load_as("sfnodes.sf_utils.lora_ortho", os.path.join(root, "sf_utils", "lora_ortho.py"))
load_as("sfnodes.sf_utils.lora_ortho_load", os.path.join(root, "sf_utils", "lora_ortho_load.py"))

# Power 尾部副作用注册 lora_notes/lora_presets -> 需要 mock aiohttp.web
aiohttp_web = types.ModuleType("aiohttp"); sys.modules["aiohttp"] = aiohttp_web
web_mod = types.ModuleType("aiohttp.web")
web_mod.json_response = lambda *a, **k: None
web_mod.Response = lambda *a, **k: None
web_mod.FileResponse = lambda *a, **k: None
sys.modules["aiohttp.web"] = web_mod

node_mod = load_as("sfnodes.nodes.model.power_lora_loader",
                   os.path.join(root, "nodes", "model", "power_lora_loader.py"))

# ── 节点结构 ─────────────────────────────────────────────────────────────────
cls = node_mod.PowerLoraLoader
check("Power INPUT_TYPES merge_method combo", "merge_method" in cls.INPUT_TYPES()["required"])
check("Power merge_method 默认 linear", cls.INPUT_TYPES()["required"]["merge_method"][1]["default"] == "linear")
check("Power merge_method 选项", cls.INPUT_TYPES()["required"]["merge_method"][0] == ["linear", "ortho_gs"])
check("Power DESCRIPTION 提及 ortho", "Ortho GS" in cls.DESCRIPTION)

# ── 测试环境 ─────────────────────────────────────────────────────────────────
for fn in ("a.safetensors", "b.safetensors", "c.safetensors"):
    with open(os.path.join(LORAS_DIR, fn), "wb") as f:
        f.write(b"x")

gs_calls = []

def fake_gs(downs):
    gs_calls.append(len(downs))
    return [MockTensor("ortho_" + d.name, dtype=d.dtype) for d in downs]

ortho_mod.gram_schmidt_ortho_downs = fake_gs

node = cls()

def reset():
    gs_calls.clear(); lora_calls.clear(); convert_calls.clear(); load_lora_calls.clear()
    return FakeModel(), FakeClip()

def slots(*items):
    return {f"LORA_{i + 1}": dict(item) for i, item in enumerate(items)}

class FakeModel:
    def __init__(self):
        self.model = object()
        self._patches = {}
        self._attachments = {}
    def clone(self):
        return FakeModel()
    def add_patches(self, patches, strength):
        for k, p in patches.items():
            self._patches.setdefault(k, []).append((p, strength))
    def set_attachments(self, key, value):
        self._attachments[key] = value

class FakeClip:
    def __init__(self):
        self.cond_stage_model = object()
        self.patcher = FakeModel()
        self._patches = {}
    def clone(self):
        return FakeClip()
    def add_patches(self, patches, strength):
        for k, p in patches.items():
            self._patches.setdefault(k, []).append((p, strength))

def run(m, c, loras, merge="linear", normalize=False, normalize_weight=1.0, preset=None):
    return node.load_loras(
        normalize, normalize_weight, merge_method=merge,
        model=m, clip=c, preset=preset, **slots(*loras)
    )

# 共享 patches：a/b 重叠 k1+k2（触发 GS），c 只有 k1
LORA_PATCHES["a.safetensors"] = {
    "diffusion_model.k1.weight": FakeAdapter(MockTensor("upA1"), MockTensor("downA1")),
    "diffusion_model.k2.weight": FakeAdapter(MockTensor("upA2"), MockTensor("downA2")),
    "clip_h.k1.weight": FakeAdapter(MockTensor("upA3"), MockTensor("downA3")),
}
LORA_PATCHES["b.safetensors"] = {
    "diffusion_model.k1.weight": FakeAdapter(MockTensor("upB1"), MockTensor("downB1")),
    "diffusion_model.k2.weight": FakeAdapter(MockTensor("upB2"), MockTensor("downB2")),
    "clip_h.k1.weight": FakeAdapter(MockTensor("upB3"), MockTensor("downB3")),
}
LORA_PATCHES["c.safetensors"] = {
    "diffusion_model.k1.weight": FakeAdapter(MockTensor("upC1"), MockTensor("downC1")),
}

# ── 1. linear 默认：走官方 LoraLoader().load_lora ────────────────────────────
m, c = reset()
out_m, out_c = run(m, c, [
    {"on": True, "lora": "a.safetensors", "strength": 0.8, "strengthTwo": 0.8},
    {"on": True, "lora": "b.safetensors", "strength": 1.0},
])
check("linear 默认走 load_lora", lora_calls == [("a.safetensors", 0.8, 0.8), ("b.safetensors", 1.0, 1.0)])
check("linear 不调 GS", gs_calls == [])
check("linear strengthTwo 缺省 = strength", lora_calls[1][2] == 1.0)

# 关/强度 0 行跳过
m, c = reset()
run(m, c, [
    {"on": False, "lora": "a.safetensors", "strength": 0.8},
    {"on": True, "lora": "b.safetensors", "strength": 0.0},
    {"on": True, "lora": "c.safetensors", "strength": 0.5},
])
check("linear 关行/零强度跳过", lora_calls == [("c.safetensors", 0.5, 0.5)])

# 短名（无扩展名）输入：get_lora_by_filename 规范化后传给官方 load_lora
m, c = reset()
run(m, c, [
    {"on": True, "lora": "a", "strength": 0.6},
])
check("linear 短名规范化后应用", lora_calls == [("a.safetensors", 0.6, 0.6)])

# 短名 + ortho：path 用规范化名解析
m, c = reset()
out_m, _ = run(m, c, [
    {"on": True, "lora": "a", "strength": 1.0},
    {"on": True, "lora": "b", "strength": 1.0},
], merge="ortho_gs")
check("ortho 短名正常正交化", gs_calls == [2, 2]
      and [s for _, s in out_m._patches["diffusion_model.k1.weight"]] == [1.0, 1.0])

# ── 2. ortho_gs：双行 -> GS + clone + add（强度按行）─────────────────────────
m, c = reset()
out_m, out_c = run(m, c, [
    {"on": True, "lora": "a.safetensors", "strength": 0.8, "strengthTwo": 0.8},
    {"on": True, "lora": "b.safetensors", "strength": 1.0},
], merge="ortho_gs")
check("ortho 双行每 key 一次 GS(2)", gs_calls == [2, 2])
check("ortho 不走 load_lora", lora_calls == [])
check("ortho 替换后应用", all(p[0].weights[1].name.startswith("ortho_") for p in out_m._patches["diffusion_model.k1.weight"])
      and all(p[0].weights[1].name.startswith("ortho_") for p in out_m._patches["diffusion_model.k2.weight"]))
check("ortho 强度按行", [s for _, s in out_m._patches["diffusion_model.k1.weight"]] == [0.8, 1.0])
check("ortho CLIP 顺序叠加", [s for _, s in out_c._patches["clip_h.k1.weight"]] == [0.8, 1.0])
check("ortho convert 每行一次", convert_calls == ["a.safetensors", "b.safetensors"])
check("ortho metadata attachments", out_m._attachments.get("lora_metadata", {}).get("trigger") == "meta_b.safetensors")

# ── 3. ortho 单行 -> 回落线性 ────────────────────────────────────────────────
m, c = reset()
run(m, c, [
    {"on": True, "lora": "a.safetensors", "strength": 0.5},
], merge="ortho_gs")
check("ortho 单行回落线性", lora_calls == [("a.safetensors", 0.5, 0.5)] and gs_calls == [])

# ── 4. key map 失败 -> 回落线性 ──────────────────────────────────────────────
m, c = reset()
unet_keys_fail[0] = True
run(m, c, [
    {"on": True, "lora": "a.safetensors", "strength": 0.5},
    {"on": True, "lora": "b.safetensors", "strength": 0.7},
], merge="ortho_gs")
unet_keys_fail[0] = False
check("ortho key map 失败回落线性", lora_calls == [("a.safetensors", 0.5, 0.5), ("b.safetensors", 0.7, 0.7)])

# ── 5. 归一化交互：ortho 用归一化后强度 ──────────────────────────────────────
m, c = reset()
out_m, _ = run(m, c, [
    {"on": True, "lora": "a.safetensors", "strength": 1.0, "strengthTwo": 1.0},
    {"on": True, "lora": "b.safetensors", "strength": 1.0},
], merge="ortho_gs", normalize=True, normalize_weight=1.0)
check("ortho 归一化后强度", [s for _, s in out_m._patches["diffusion_model.k1.weight"]] == [0.5, 0.5])

# ── 6. preset 优先 + ortho 分派 ──────────────────────────────────────────────
m, c = reset()
out_m, out_c = run(m, c, [
    {"on": True, "lora": "a.safetensors", "strength": 1.0},
    {"on": True, "lora": "b.safetensors", "strength": 1.0},
], merge="ortho_gs", preset={
    "normalize": False, "normalize_weight": 1.0,
    "loras": [
        {"on": True, "lora": "a.safetensors", "strength": 0.8, "strengthTwo": 0.8},
        {"on": True, "lora": "b.safetensors", "strength": 1.2},
    ],
})
check("ortho preset 优先强度", [s for _, s in out_m._patches["diffusion_model.k1.weight"]] == [0.8, 1.2])
check("ortho preset 单强度时 sc=sm", [s for _, s in out_c._patches["clip_h.k1.weight"]] == [0.8, 1.2])

# ── 7. 全关 -> 直通 ──────────────────────────────────────────────────────────
m, c = reset()
out_m, out_c = run(m, c, [{"on": False, "lora": "a.safetensors", "strength": 0.5}], merge="ortho_gs")
check("全关直通", out_m is m and out_c is c and lora_calls == [])

# ── 8. 无 clip：sc 归 0，仅 model 侧 ─────────────────────────────────────────
m, c = reset()
out_m, out_c = run(m, None, [
    {"on": True, "lora": "a.safetensors", "strength": 1.0, "strengthTwo": 9.0},
    {"on": True, "lora": "b.safetensors", "strength": 1.0, "strengthTwo": 9.0},
], merge="ortho_gs")
check("ortho 无 clip sc=0", out_c is None and len(out_m._patches) == 2 and len(load_lora_calls) == 2)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
