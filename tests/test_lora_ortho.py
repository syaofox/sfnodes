# SFLoraStack ortho_gs 模式测试（Python 直接运行：python tests/test_lora_ortho.py）
# 覆盖：
#   - Gram-Schmidt 数学性质（numpy 参考实现，逐行对应 sf_utils/lora_ortho.py
#     的 torch 版）：首行不变、行两两正交、投影残差在基行空间、子空间覆盖
#     幅度损失、单元素直通、非 2D 直通
#   - extract_up_down / replace_down patch 格式探测（当前 LoRAAdapter.weights
#     与历史 tuple 多格式回退，非 LoRA patch 返回 (None, None)/原样返回）
#   - 节点层 ortho 全链路（mock comfy/folder_paths，monkeypatch GS）：
#     重叠 key 分组正交化、单 key 直通、非 LoRA/conv fallback、CLIP 顺序叠加、
#     强度驱动、触发词、零强度行、key map 失败整体 fallback 顺序、cacheMode 修剪
import importlib.util
import json
import os
import sys
import tempfile
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

# ── mock comfy / torch / folder_paths（本机无 ComfyUI 运行时）────────────────
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
    """轻量张量替身：isinstance(torch.Tensor) 命中 + dim/dtype/device/float/to。"""
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

# 假 LoRA patch（模拟 comfy.weight_adapter.lora.LoRAAdapter，weights 形状
# (up, down, alpha, mid, dora_scale, reshape)，up 是 weights[0]、down 是 [1]）。
class FakeAdapter:
    def __init__(self, up, down, alpha=1.0, dora=None):
        self.weights = (up, down, alpha, None, dora, None)

LORAS_DIR = tempfile.mkdtemp(prefix="sf_lora_ortho_test_")

def fake_get_full_path(folder, name):
    if folder != "loras":
        return None
    return os.path.join(LORAS_DIR, name.replace("/", os.sep))

folder_paths = types.ModuleType("folder_paths"); sys.modules["folder_paths"] = folder_paths
folder_paths.get_full_path = fake_get_full_path

# ── fake comfy.lora / comfy.lora_convert / comfy.utils / comfy.sd ────────────
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
    # 按 key_map 的值（模型权重名）过滤：unet patch 键是 "diffusion_model.x"
    # 形式，clip 键是 "clip_h.x"——模拟官方 load_lora 的 to_load 语义。
    values = set(key_map.values())
    return {k: p for k, p in lora_sd.get("__patches__", {}).items() if k in values}

comfy_lora.load_lora = fake_load_lora

convert_calls = []

def fake_convert_lora(sd):
    convert_calls.append(sd["__name__"])
    return sd

comfy_lora_convert.convert_lora = fake_convert_lora

load_calls = []
apply_calls = []

def fake_load_torch_file(path, safe_load=True, return_metadata=False):
    load_calls.append(path)
    name = os.path.basename(path)
    sd = {"__name__": name, "__patches__": LORA_PATCHES.get(name, {})}
    if return_metadata:
        return sd, {"trigger": "meta_" + name}
    return sd

def fake_load_lora_for_models(model, clip, lora, sm, sc, lora_metadata=None):
    apply_calls.append((sm, sc))
    return ("m" + str(model), "c" + str(clip))

comfy_utils.load_torch_file = fake_load_torch_file
comfy_sd.load_lora_for_models = fake_load_lora_for_models

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
load_as("sfnodes.sf_utils.lora_reader", os.path.join(root, "sf_utils", "lora_reader.py"))
ortho_mod = load_as("sfnodes.sf_utils.lora_ortho", os.path.join(root, "sf_utils", "lora_ortho.py"))

# lora_stack 模块尾部副作用注册 lora_routes -> 需要 mock aiohttp.web
# （路由注册因 server 缺失被 try/except 吞掉，先例 test_lora_reader.py:96）。
aiohttp_web = types.ModuleType("aiohttp"); sys.modules["aiohttp"] = aiohttp_web
web_mod = types.ModuleType("aiohttp.web")
web_mod.json_response = lambda *a, **k: None
web_mod.Response = lambda *a, **k: None
web_mod.FileResponse = lambda *a, **k: None
sys.modules["aiohttp.web"] = web_mod

node_mod = load_as("sfnodes.nodes.model.lora_stack", os.path.join(root, "nodes", "model", "lora_stack.py"))

# ── 1. Gram-Schmidt 数学性质（numpy 参考实现）───────────────────────────────
# 参考实现逐行对应 lora_ortho.gram_schmidt_ortho_downs 的 torch 分支：
#   float64 计算（torch 版用 float()）；SVD 右奇异向量作行空间正交基；
#   阈值 1e-6 相对 + 1e-10 绝对；QR 去线性相关；fro < 1e-10 跳过扩基。
def gs_np(downs):
    result = []
    basis = None
    for d0 in downs:
        d = np.asarray(d0, dtype=np.float64)
        if d.ndim != 2:
            result.append(d0)
            continue
        if basis is not None and basis.shape[0] > 0:
            d_ortho = d - (d @ basis.T) @ basis
        else:
            d_ortho = d.copy()
        result.append(d_ortho)
        if np.linalg.norm(d_ortho, "fro") < 1e-10:
            continue
        try:
            U, S, Vh = np.linalg.svd(d_ortho, full_matrices=False)
            tol = max(S[0] * 1e-6, 1e-10) if S.size > 0 else 1e-10
            keep = S > tol
            if not keep.any():
                continue
            basis_new = Vh[keep]
            if basis is None:
                basis = basis_new
            else:
                combined = np.concatenate([basis, basis_new], axis=0).T
                Q, R = np.linalg.qr(combined, mode="reduced")
                r_diag = np.abs(np.diag(R))
                if r_diag.size > 0 and r_diag.max() > 0:
                    basis = Q[:, r_diag > r_diag.max() * 1e-6].T
                else:
                    basis = Q.T
        except Exception:
            pass
    return result

def rows_orthogonal(A, B, tol=1e-8):
    A = np.asarray(A); B = np.asarray(B)
    for a in A:
        for b in B:
            if abs(np.dot(a, b)) > tol:
                return False
    return True

rng = np.random.default_rng(42)
B1 = rng.standard_normal((2, 5))
B2 = rng.standard_normal((3, 5))
B3 = rng.standard_normal((2, 5))

r1 = gs_np([B1, B2])
check("GS 首行不变", np.allclose(r1[0], B1))
check("GS 行两两正交（2 个）", rows_orthogonal(r1[0], r1[1]))
# 投影残差性质：B2 - B2' 的每行必须落在 B1 的行空间（B1^T 的列空间）。
resid = B2 - r1[1]
coef, *_ = np.linalg.lstsq(r1[0].T, resid.T, rcond=None)
check("GS 投影残差在基行空间", np.linalg.norm(r1[0].T @ coef - resid.T) < 1e-8)

r2 = gs_np([B1, B2, B3])
check("GS 三矩阵两两正交", rows_orthogonal(r2[0], r2[1]) and rows_orthogonal(r2[0], r2[2])
      and rows_orthogonal(r2[1], r2[2]))
check("GS 三矩阵首行不变", np.allclose(r2[0], B1))

# 子空间覆盖：B2 的行全在 B1 行空间内 -> B2' 归零（幅度损失 tradeoff）
covered = B1[0:1] * 0.5 + B1[1:2] * 1.5
r3 = gs_np([B1, covered])
check("GS 覆盖时幅度损失（投影归零）", np.linalg.norm(r3[1], "fro") < 1e-6)

check("GS 单元素原样", np.allclose(gs_np([B1])[0], B1))
one_d = np.zeros(5)
check("GS 非 2D 直通", gs_np([one_d])[0] is one_d)

# ── 2. extract_up_down / replace_down 格式探测 ──────────────────────────────
ex_ = ortho_mod.extract_up_down
up = MockTensor("up"); down = MockTensor("down")

ad = FakeAdapter(up, down, alpha=2.0, dora=MockTensor("dora"))
u, d = ex_(ad)
check("extract LoRAAdapter.weights", u is up and d is down)

short = FakeAdapter.__new__(FakeAdapter)
short.weights = (up,)
check("extract weights 太短 -> None", ex_(short) == (None, None))
check("extract 无 weights 且垃圾 -> None", ex_(None) == (None, None) and ex_("x") == (None, None))

check("extract 字符串标签格式", ex_(("lora", (up, down, 1.0, None))) == (up, down))
check("extract tensor-first 格式", ex_((up, down, 1.0)) == (up, down))
check("extract float 前缀格式", ex_((1.0, (up, down))) == (up, down))
check("extract float+str 格式", ex_((1.0, "lora", (up, down))) == (up, down))
check("extract diff 非 LoRA -> None", ex_(("diff", (MockTensor("w"),))) == (None, None))
check("extract set 非 LoRA -> None", ex_(("set", (MockTensor("w"),))) == (None, None))
check("extract 过短 -> None", ex_(("x",)) == (None, None) and ex_(()) == (None, None))

# replace_down：原对象不动，新对象 down 被替换
rp_ = ortho_mod.replace_down
new_down = MockTensor("new_down")
ad2 = FakeAdapter(up, down, alpha=2.0, dora=MockTensor("dora"))
rp = rp_(ad2, new_down)
check("replace LoRAAdapter 返回新对象", rp is not ad2)
check("replace LoRAAdapter 原对象不变", ad2.weights[1] is down)
check("replace LoRAAdapter down 替换", rp.weights[1] is new_down)
check("replace LoRAAdapter 其余保留", rp.weights[0] is up and rp.weights[2] == 2.0
      and rp.weights[4].name == "dora")
check("replace 字符串标签", rp_(("lora", (up, down, 1.0, None)), new_down)[1][1] is new_down)
check("replace tensor-first", rp_((up, down, 1.0), new_down)[1] is new_down)
check("replace float 前缀", rp_((1.0, (up, down)), new_down)[1][1] is new_down)
check("replace float+str", rp_((1.0, "lora", (up, down)), new_down)[2][1] is new_down)
diff_patch = ("diff", (MockTensor("w"),))
check("replace 无法识别原样返回", rp_("x", new_down) == "x" and rp_(diff_patch, new_down) == diff_patch)

# ── 3. 节点层 ortho 全链路 ───────────────────────────────────────────────────
for fn in ("a.safetensors", "b.safetensors", "c.safetensors"):
    with open(os.path.join(LORAS_DIR, fn), "wb") as f:
        f.write(b"x")

# monkeypatch GS（数学在 §1 由 numpy 参考验证；这里验证调用结构与替换产物）
gs_calls = []

def fake_gs(downs):
    gs_calls.append(len(downs))
    return [MockTensor("ortho_" + d.name, dtype=d.dtype) for d in downs]

ortho_mod.gram_schmidt_ortho_downs = fake_gs

# ── 2.5 build_ortho_replacements 分组/直通/统计（直接单测）───────────────────
gs_calls.clear()
br_ = ortho_mod.build_ortho_replacements
pd0 = (
    {"k1": FakeAdapter(MockTensor("u1"), MockTensor("d1")),
     "k2": FakeAdapter(MockTensor("u2"), MockTensor("d2"))},
    0.8,
)
pd1 = ({"k1": FakeAdapter(MockTensor("u3"), MockTensor("d3"))}, 1.0)
pd2 = ({"k3": FakeAdapter(MockTensor("u4"), MockTensor("d4"))}, 0.6)  # 独占 key
repl, ok_keys, pass_keys = br_([pd0, pd1, pd2])
check("build_ortho 统计 k1 正交化 k2/k3 直通", ok_keys == 1 and pass_keys == 2 and gs_calls == [2])
check("build_ortho 重叠 key 替换", repl[0][0]["k1"].weights[1].name.startswith("ortho_"))
check("build_ortho 单条目 key 原样", repl[0][0]["k2"] is pd0[0]["k2"])
check("build_ortho 重叠行也被替换", repl[1][0]["k1"].weights[1].name.startswith("ortho_"))
check("build_ortho 独立 key 行直通", repl[2][0]["k3"] is pd2[0]["k3"])
check("build_ortho 强度保留", [s for _, s in repl] == [0.8, 1.0, 0.6])

repl2, ok2, pass2 = br_([({}, 0.5)])
check("build_ortho 空 dict 行", repl2 == [({}, 0.5)] and ok2 == 0 and pass2 == 0)

diff_pd0 = ({"k1": ("diff", (MockTensor("w"),))}, 1.0)
diff_pd1 = ({"k1": FakeAdapter(MockTensor("u"), MockTensor("d"))}, 1.0)
repl3, ok3, pass3 = br_([diff_pd0, diff_pd1])
check("build_ortho diff fallback 该 key 直通", ok3 == 0 and pass3 == 1
      and repl3[0][0]["k1"] is diff_pd0[0]["k1"] and repl3[1][0]["k1"] is diff_pd1[0]["k1"])

node = node_mod.SFLoraStack()

def reset():
    gs_calls.clear(); load_calls.clear(); apply_calls.clear(); convert_calls.clear()
    load_lora_calls.clear()
    node._cache = {}; node._last_path = None
    return FakeModel(), FakeClip()

def state(loras, merge="ortho_gs", cache="last"):
    return json.dumps({"loras": loras, "sep": ", ", "cacheMode": cache, "mergeMethod": merge})

class FakeModel:
    def __init__(self):
        self.model = object()  # model_lora_keys_unet(model.model) 用
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

# 3a. 重叠 key：a/b 都 patch k1+k2 -> 每 key 一次 GS（2 个 down），替换后应用
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
m, c = reset()
out_m, out_c, triggers, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 0.8, "sc": 0.8, "triggers": ["alpha"]},
    {"name": "b.safetensors", "on": True, "sm": 1.0, "sc": 1.0, "triggers": ["beta"]},
]))
check("ortho 重叠 key 每 key 一次 GS(2)", gs_calls == [2, 2])
check("ortho 替换后应用（down 为 GS 产物）", all(
    p[0].weights[1].name.startswith("ortho_") for p in out_m._patches["diffusion_model.k1.weight"])
    and all(p[0].weights[1].name.startswith("ortho_") for p in out_m._patches["diffusion_model.k2.weight"]))
check("ortho 强度按行", [s for _, s in out_m._patches["diffusion_model.k1.weight"]] == [0.8, 1.0])
check("ortho up/alpha 不动", out_m._patches["diffusion_model.k1.weight"][0][0].weights[0].name == "upA1"
      and out_m._patches["diffusion_model.k1.weight"][0][0].weights[2] == 1.0)
check("ortho convert_lora 每行一次", convert_calls == ["a.safetensors", "b.safetensors"])
check("ortho lora_metadata attachments", out_m._attachments.get("lora_metadata", {}).get("trigger") == "meta_b.safetensors")
check("ortho CLIP 顺序叠加", [s for _, s in out_c._patches["clip_h.k1.weight"]] == [0.8, 1.0])
check("ortho CLIP metadata", out_c.patcher._attachments.get("lora_metadata", {}).get("trigger") == "meta_b.safetensors")
check("ortho 触发词", triggers == "alpha, beta")
check("ortho 同 key 两次 patch 累积", len(out_m._patches["diffusion_model.k1.weight"]) == 2)

# 3b. 单 key 重叠、另一 key 单条目 -> 单条目直通（原 patch 不替换）
LORA_PATCHES["c.safetensors"] = {
    "diffusion_model.k1.weight": FakeAdapter(MockTensor("upC1"), MockTensor("downC1")),
}
m, c = reset()
out_m, _, _, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1},
    {"name": "c.safetensors", "on": True, "sm": 1, "sc": 1},
]))
check("ortho 混合：仅重叠 key GS(2)", gs_calls == [2])
check("ortho k1 正交化 k2 直通", out_m._patches["diffusion_model.k1.weight"][0][0].weights[1].name.startswith("ortho_")
      and out_m._patches["diffusion_model.k2.weight"][0][0].weights[1].name == "downA2")

# 3c. 非 LoRA patch fallback：a/k1 是 diff -> k1 全部顺序叠加，k2 正常正交化
LORA_PATCHES["a.safetensors"] = {
    "diffusion_model.k1.weight": ("diff", (MockTensor("wA1"),)),
    "diffusion_model.k2.weight": FakeAdapter(MockTensor("upA2"), MockTensor("downA2")),
}
m, c = reset()
out_m, _, _, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1},
    {"name": "b.safetensors", "on": True, "sm": 1, "sc": 1},
]))
k1_entries = out_m._patches["diffusion_model.k1.weight"]
check("ortho 非 LoRA key fallback（diff 原样）", k1_entries[0][0] is LORA_PATCHES["a.safetensors"]["diffusion_model.k1.weight"])
check("ortho 同 key adapter 也直通", k1_entries[1][0].weights[1].name == "downB1")
check("ortho 非 LoRA key 不调 GS", gs_calls == [2])

# 3d. conv down（4D）fallback：k1 整 key 顺序
LORA_PATCHES["a.safetensors"] = {
    "diffusion_model.k1.weight": FakeAdapter(MockTensor("upA1"), MockTensor("downA1c", dim=4)),
}
m, c = reset()
out_m, _, _, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1},
    {"name": "b.safetensors", "on": True, "sm": 1, "sc": 1},
]))
check("ortho conv down fallback", gs_calls == [] and
      out_m._patches["diffusion_model.k1.weight"][0][0].weights[1].name == "downA1c"
      and out_m._patches["diffusion_model.k1.weight"][1][0].weights[1].name == "downB1")

# 3e. 单行：全直通，不调 GS
LORA_PATCHES["a.safetensors"] = {
    "diffusion_model.k1.weight": FakeAdapter(MockTensor("upA1"), MockTensor("downA1")),
    "diffusion_model.k2.weight": FakeAdapter(MockTensor("upA2"), MockTensor("downA2")),
}
m, c = reset()
out_m, _, _, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1},
]))
check("ortho 单行直通不调 GS", gs_calls == [] and
      out_m._patches["diffusion_model.k1.weight"][0][0].weights[1].name == "downA1"
      and out_m._patches["diffusion_model.k2.weight"][0][0].weights[1].name == "downA2")

# 3f. sm=0 行：model 侧跳过、CLIP 侧照常；零强度行计触发词不加载
LORA_PATCHES["a.safetensors"]["clip_h.k1.weight"] = FakeAdapter(MockTensor("upA3"), MockTensor("downA3"))
m, c = reset()
out_m, out_c, triggers, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 0.0, "sc": 1.0, "triggers": ["zero_sm"]},
    {"name": "b.safetensors", "on": True, "sm": 0.0, "sc": 0.0, "triggers": ["zero_both"]},
]))
check("ortho sm=0 不应用 model", out_m._patches == {})
check("ortho sm=0 仍应用 CLIP", [s for _, s in out_c._patches["clip_h.k1.weight"]] == [1.0])
check("ortho 全零强度行计触发词", triggers == "zero_sm, zero_both")
# 只有 a（sc=1，需读盘取 clip patch）读盘；b 全零行不读盘；unet 侧 sm=0 不加载。
check("ortho 全零强度行不读盘", load_calls == [fake_get_full_path("loras", "a.safetensors")])
check("ortho sm=0 不加载 unet patch", load_lora_calls == ["a.safetensors"])

# 3g. key map 失败 -> 整体 fallback 顺序路径
m, c = reset()
unet_keys_fail[0] = True
out_m, out_c, triggers, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 0.5, "sc": 0.5, "triggers": ["fallback"]},
]))
unet_keys_fail[0] = False
check("ortho key map 失败 fallback 顺序", apply_calls == [(0.5, 0.5)])
check("ortho fallback 触发词", triggers == "fallback")
check("ortho fallback 后无 GS", gs_calls == [])

# 3g2. clip key map 失败 -> 同样整体 fallback
clip_keys_fail = [False]

def fake_model_lora_keys_clip_fail(model, key_map=None):
    if clip_keys_fail[0]:
        raise RuntimeError("clip boom")
    return dict(CLIP_KEYS)

comfy_lora.model_lora_keys_clip = fake_model_lora_keys_clip_fail
m, c = reset()
clip_keys_fail[0] = True
out_m, out_c, _, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 0.5, "sc": 0.5},
]))
clip_keys_fail[0] = False
comfy_lora.model_lora_keys_clip = fake_model_lora_keys_clip
check("ortho clip key map 失败 fallback 顺序", apply_calls == [(0.5, 0.5)])

# 3g3. 全部行加载失败（load_lora 抛错）-> 模型原样直通、触发词不计
load_lora_fail = [False]

def fake_load_lora_fail(lora_sd, key_map, log_missing=True):
    if load_lora_fail[0]:
        raise RuntimeError("parse boom")
    return fake_load_lora(lora_sd, key_map, log_missing)

comfy_lora.load_lora = fake_load_lora_fail
m, c = reset()
load_lora_fail[0] = True
out_m, out_c, triggers, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1, "triggers": ["boom"]},
]))
load_lora_fail[0] = False
comfy_lora.load_lora = fake_load_lora
check("ortho 全部行加载失败直通", out_m._patches == {} and out_c._patches == {})
check("ortho 加载失败行不计触发词", triggers == "")

# 3h. 缺失文件行跳过（不进 resolved/不加载）
LORA_PATCHES.pop("c.safetensors", None)
m, c = reset()
out_m, _, triggers, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1, "triggers": ["present"]},
    {"name": "ghost.safetensors", "on": True, "sm": 1, "sc": 1, "triggers": ["ghost"]},
]))
check("ortho 缺失文件跳过", triggers == "present")

# 3i. cacheMode=last 修剪：run 后只留最近路径
m, c = reset()
node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1},
    {"name": "b.safetensors", "on": True, "sm": 1, "sc": 1},
]))
check("ortho cacheMode=last 只留最近", len(node._cache) == 1
      and node._last_path == fake_get_full_path("loras", "b.safetensors")
      and fake_get_full_path("loras", "b.safetensors") in node._cache)

# 3j. cacheMode=all 保留整栈
m, c = reset()
node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 1},
    {"name": "b.safetensors", "on": True, "sm": 1, "sc": 1},
], cache="all"))
check("ortho cacheMode=all 保留整栈", len(node._cache) == 2)

# 3k. sequential 模式回归（mergeMethod=sequential 走官方加载路径）
m, c = reset()
out_m, out_c, triggers, _ = node.apply(m, c, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 0.5, "sc": 0.5, "triggers": ["seq"]},
], merge="sequential"))
check("sequential 回归走 load_lora_for_models", apply_calls == [(0.5, 0.5)])
check("sequential 回归不调 GS", gs_calls == [])
check("sequential 回归触发词", triggers == "seq")

# 3l. clip 未接：sc 归 0，不碰 clip
m, c = reset()
out_m, out_c, _, _ = node.apply(m, None, LoraLoaderState=state([
    {"name": "a.safetensors", "on": True, "sm": 1, "sc": 9},
]))
check("ortho 无 clip sc=0", out_c is None and len(load_lora_calls) == 1)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
