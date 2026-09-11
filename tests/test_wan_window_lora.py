# SFWanWindowLoRA 模拟测试（Python 直接运行：python3 tests/test_wan_window_lora.py）
# mock 官方 lora 链路（comfy.lora/convert/context_windows）+ safetensors +
# folder_paths，无需 torch（patch-swap 架构本身无 torch 依赖），验证：
#   - 灵活 schema：window_99 放行、VALIDATE_INPUTS 接管
#   - preset 行解析：on 开关/strength（模型侧）/strengthTwo 被忽略/坏行丢弃
#   - 位置式：空槽不压紧；preset 只连 window_3 → 初始态 D 键为零占位
#   - 补丁表：slot 内多 LoRA 按行顺序拼接、strength 进元组、base 上游保留
#   - 回调：EVALUATE 按 window_idx 取模换槽（含空槽纯 base、越界轮回）、
#     CLEANUP 复位初始态；无 handler 回退静态 slot0
#   - 异常路径：加载失败/零匹配跳过、全空直通、超上限截断
import importlib.util
import json
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── LoRA 键 fixtures ───────────────────────────────────────────────────────
# RAW_*：文件内 lora 键基名；MODEL_*：映射后的模型键
RAW_A, MODEL_A = "net.A", "diffusion_model.blocks.0.attn.wq.weight"
RAW_B, MODEL_B = "net.B", "diffusion_model.blocks.0.attn.wk.weight"
RAW_C, MODEL_C = "net.C", "diffusion_model.blocks.0.mlp.gate.weight"
RAW_D, MODEL_D = "net.D", "diffusion_model.blocks.1.attn.wq.weight"
RAW_Y, MODEL_Y = "net.Y", "diffusion_model.blocks.9.ghost.weight"  # state 无此键
RAW_X = "net.X"  # key_map 无此键
RAW2MODEL = {RAW_A: MODEL_A, RAW_B: MODEL_B, RAW_C: MODEL_C,
             RAW_D: MODEL_D, RAW_Y: MODEL_Y}
MODEL_STATE_KEYS = [MODEL_A, MODEL_B, MODEL_C, MODEL_D]


def _make_raw(layers):
    sd = {}
    for layer in layers:
        sd[f"{layer}.lora_down.weight"] = ("down", layer)
        sd[f"{layer}.lora_up.weight"] = ("up", layer)
        sd[f"{layer}.alpha"] = 4.0
    return sd


SD_BY_PATH = {
    "/loras/a.safetensors": _make_raw([RAW_A, RAW_B]),
    "/loras/b.safetensors": _make_raw([RAW_B, RAW_C]),
    "/loras/c.safetensors": _make_raw([RAW_D]),
    "/loras/y.safetensors": _make_raw([RAW_Y]),
    "/loras/x.safetensors": _make_raw([RAW_X]),
}

# ── mock 模块 ──────────────────────────────────────────────────────────────
st_mod = types.ModuleType("safetensors.torch")
st_mod.load_file = lambda path: SD_BY_PATH[path]
st_pkg = types.ModuleType("safetensors")
st_pkg.torch = st_mod
sys.modules["safetensors"] = st_pkg
sys.modules["safetensors.torch"] = st_mod

fp_mod = types.ModuleType("folder_paths")
fp_mod.get_full_path = lambda cat, name: f"/loras/{name}"
fp_mod.get_user_directory = lambda: "/tmp/sf_test_user"
sys.modules["folder_paths"] = fp_mod

aiohttp_mod = types.ModuleType("aiohttp")
aiohttp_web = types.ModuleType("aiohttp.web")
aiohttp_web.Response = type("Response", (), {})
aiohttp_web.json_response = lambda *a, **k: None
aiohttp_mod.web = aiohttp_web
sys.modules["aiohttp"] = aiohttp_mod
sys.modules["aiohttp.web"] = aiohttp_web


class _FakeRoutes:
    def get(self, *a, **k):
        return lambda f: f

    def post(self, *a, **k):
        return lambda f: f


server_mod = types.ModuleType("server")
server_mod.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_FakeRoutes()))
sys.modules["server"] = server_mod

comfy_pkg = types.ModuleType("comfy")
lora_mod = types.ModuleType("comfy.lora")
lora_mod.model_lora_keys_unet = lambda model, key_map: dict(RAW2MODEL)


def _fake_load_lora(converted, key_map, log_missing=True):
    out = {}
    for raw, mk in key_map.items():
        if f"{raw}.lora_down.weight" in converted:
            out[mk] = ("adapter", raw)
    return out


lora_mod.load_lora = _fake_load_lora
conv_mod = types.ModuleType("comfy.lora_convert")
conv_mod.convert_lora = lambda sd: sd
cw_mod = types.ModuleType("comfy.context_windows")


class _CB:
    EVALUATE_CONTEXT_WINDOWS = "evaluate_context_windows"
    EXECUTE_CLEANUP = "execute_cleanup"


cw_mod.IndexListCallbacks = _CB
comfy_pkg.lora = lora_mod
comfy_pkg.lora_convert = conv_mod
comfy_pkg.context_windows = cw_mod
sys.modules["comfy"] = comfy_pkg
sys.modules["comfy.lora"] = lora_mod
sys.modules["comfy.lora_convert"] = conv_mod
sys.modules["comfy.context_windows"] = cw_mod

for name, path in [
    ("sfnodes", root),
    ("sfnodes.nodes", os.path.join(root, "nodes")),
    ("sfnodes.nodes.model", os.path.join(root, "nodes", "model")),
]:
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(root, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load("sfnodes.nodes.model.conditioning_combine", "nodes/model/conditioning_combine.py")
_load("sfnodes.nodes.model.lora_preset", "nodes/model/lora_preset.py")
nmod = _load("sfnodes.nodes.model.wan_window_lora", "nodes/model/wan_window_lora.py")

SFWanWindowLoRA = nmod.SFWanWindowLoRA


def preset(*rows):
    return {"loras": list(rows)}


def row(lora, on=True, strength=1.0, strengthTwo=None):
    r = {"lora": lora, "on": on, "strength": strength}
    if strengthTwo is not None:
        r["strengthTwo"] = strengthTwo
    return r


# ── fake patcher/handler ───────────────────────────────────────────────────
class FakeInner:
    def state_dict(self):
        return {k: None for k in MODEL_STATE_KEYS}


class FakeHandler:
    def __init__(self):
        self.callbacks = {}


class FakePatcher:
    def __init__(self, base_patches=None, handler=None):
        self.model = FakeInner()
        self.patches = dict(base_patches or {})
        self.model_options = {"context_handler": handler} if handler is not None else {}
        self.clone_called = 0

    def clone(self):
        self.clone_called += 1
        return self


def fire_evaluate(handler, window_idx):
    cbs = handler.callbacks.get("evaluate_context_windows", {}).get(
        "sf_wan_window_lora", [])
    assert len(cbs) == 1, f"expected 1 evaluate cb, got {len(cbs)}"
    cbs[0](handler, None, None, None, None, {}, window_idx, None)


def fire_cleanup(handler):
    cbs = handler.callbacks.get("execute_cleanup", {}).get(
        "sf_wan_window_lora", [])
    assert len(cbs) == 1, f"expected 1 cleanup cb, got {len(cbs)}"
    cbs[0](handler, None, None, None, {}, {})


node = SFWanWindowLoRA()

# ── 结构 ───────────────────────────────────────────────────────────────────
it = node.INPUT_TYPES()
check("structure: required model", "model" in it["required"])
check("structure: flexible window_99",
      "window_99" in it["optional"] and it["optional"]["window_99"] == ("SF_LORA_PRESET",))
check("structure: VALIDATE_INPUTS", node.VALIDATE_INPUTS(window_99=("SF_LORA_PRESET",)) is True)
check("structure: RETURN_TYPES", node.RETURN_TYPES == ("MODEL", "STRING"))
check("structure: CATEGORY", node.CATEGORY == "sfnodes/model")
check("structure: DESCRIPTION", bool(node.DESCRIPTION))
check("structure: constants", nmod.INITIAL_WINDOW_INPUTS == 2 and nmod.MAX_WINDOW_INPUTS == 10)

# ── preset 行解析 ──────────────────────────────────────────────────────────
_parse = nmod._parse_preset_rows
check("parse: 基本行", _parse(preset(row("a.safetensors", strength=0.5))) == [("a.safetensors", 0.5)])
check("parse: strengthTwo 被忽略",
      _parse(preset(row("a.safetensors", strength=0.5, strengthTwo=7.0))) == [("a.safetensors", 0.5)])
check("parse: on=False 跳过", _parse(preset(row("a.safetensors", on=False))) == [])
check("parse: strength=0 跳过", _parse(preset(row("a.safetensors", strength=0.0))) == [])
check("parse: 坏行丢弃",
      _parse({"loras": [None, "x", {}, {"lora": ""}, {"lora": "None"}, row("b.safetensors")]})
      == [("b.safetensors", 1.0)])
check("parse: 非 dict 永不抛错", _parse(None) == [] and _parse("x") == [] and _parse({}) == [])

# ── apply：3 槽（a 双行 / 空 / c 独占 D）+ 上游 base ────────────────────────
BASE_T = ("base-strength", "base-data", 1.0, None, None)
handler = FakeHandler()
patcher = FakePatcher(base_patches={MODEL_A: [BASE_T]}, handler=handler)
out = node.apply(
    model=patcher,
    window_1=preset(row("a.safetensors", strength=1.0),
                    row("b.safetensors", strength=0.5)),
    window_2=None,
    window_3=preset(row("c.safetensors", strength=2.0)))
check("apply: returns (model, info)", len(out) == 2 and out[0] is patcher)
info = json.loads(out[1])
check("apply: n_slots=3 不压紧", info["n_slots"] == 3)
check("apply: slot0 双 LoRA", len(info["slots"][0]["loras"]) == 2)
check("apply: slot1 empty 标注", info["slots"][1].get("status", "").startswith("empty"))
check("apply: slot2 单 LoRA", len(info["slots"][2]["loras"]) == 1)
check("apply: mode patch-swap", "patch-swap" in info["mode"])
P = patcher.patches
# 初始态 = base + slot0；D 键仅 slot2 独占 → 零占位
check("initial: A = base + a行",
      P[MODEL_A] == [BASE_T, (1.0, ("adapter", RAW_A), 1.0, None, None)])
check("initial: B = a行 + b行（行序拼接，强度 1.0/0.5）",
      [t[0] for t in P[MODEL_B]] == [1.0, 0.5]
      and [t[1] for t in P[MODEL_B]] == [("adapter", RAW_B)] * 2)
check("initial: C = b行", P[MODEL_C] == [(0.5, ("adapter", RAW_C), 1.0, None, None)])
check("initial: D 零占位", P[MODEL_D] == [(0.0, ("adapter", RAW_D), 1.0, None, None)])
check("callbacks: evaluate+cleanup 已注册",
      "sf_wan_window_lora" in handler.callbacks.get("evaluate_context_windows", {})
      and "sf_wan_window_lora" in handler.callbacks.get("execute_cleanup", {}))

# ── EVALUATE：window_idx 取模换槽 ───────────────────────────────────────────
fire_evaluate(handler, 0)
check("eval w0: A = base + a行", patcher.patches[MODEL_A][0] is BASE_T
      and patcher.patches[MODEL_A][1][0] == 1.0)
fire_evaluate(handler, 1)
check("eval w1（空槽）: A 仅 base", patcher.patches[MODEL_A] == [BASE_T])
check("eval w1（空槽）: D 无补丁（纯 base，base 本无 D）", patcher.patches[MODEL_D] == [])
fire_evaluate(handler, 2)
check("eval w2: D = c行 strength 2.0",
      patcher.patches[MODEL_D] == [(2.0, ("adapter", RAW_D), 1.0, None, None)])
check("eval w2: A 退回 base", patcher.patches[MODEL_A] == [BASE_T])
fire_evaluate(handler, 3)
check("eval w3 越界轮回 slot0", patcher.patches[MODEL_B][0][0] == 1.0
      and len(patcher.patches[MODEL_B]) == 2)

# ── CLEANUP 复位 ───────────────────────────────────────────────────────────
fire_evaluate(handler, 2)
fire_cleanup(handler)
check("cleanup: A 复位初始", patcher.patches[MODEL_A][0] is BASE_T
      and len(patcher.patches[MODEL_A]) == 2)
check("cleanup: D 占位回来", patcher.patches[MODEL_D][0][0] == 0.0)

# ── 无 handler 回退静态 slot0 ───────────────────────────────────────────────
patcher_nh = FakePatcher()
out_nh = node.apply(model=patcher_nh, window_1=preset(row("a.safetensors")))
check("no-handler: patches = slot0 初始态",
      patcher_nh.patches[MODEL_A] == [(1.0, ("adapter", RAW_A), 1.0, None, None)])
check("no-handler: n_slots 照记", json.loads(out_nh[1])["n_slots"] == 1)

# ── 诊断：零匹配 warning 不报错 ─────────────────────────────────────────────
patcher_y = FakePatcher(handler=FakeHandler())
out_y = node.apply(model=patcher_y,
                   window_1=preset(row("y.safetensors")),
                   window_2=preset(row("b.safetensors")))
info_y = json.loads(out_y[1])
check("diag: y 零匹配进 failed 原因",
      "0 model keys" in info_y["slots"][0].get("status", ""))
check("diag: b matched 2", info_y["slots"][1]["loras"][0]["model_keys_matched"] == 2)

# ── 异常路径 ───────────────────────────────────────────────────────────────
patcher_m = FakePatcher(handler=FakeHandler())
out_m = node.apply(model=patcher_m,
                   window_1=preset(row("missing.safetensors")),
                   window_2=preset(row("b.safetensors")))
info_m = json.loads(out_m[1])
check("load-fail: 槽1 empty", info_m["slots"][0].get("status", "").startswith("empty"))
check("load-fail: 槽2 正常", len(info_m["slots"][1]["loras"]) == 1)

patcher_x = FakePatcher(handler=FakeHandler())
out_x = node.apply(model=patcher_x, window_1=preset(row("x.safetensors")))
check("unmapped: 文件键全无映射则槽 empty",
      json.loads(out_x[1])["slots"][0].get("status", "").startswith("empty"))

patcher_e = FakePatcher(handler=FakeHandler())
out_e = node.apply(model=patcher_e)
check("no-slots: passthrough", out_e[0] is patcher_e)
check("no-slots: n_slots=0", json.loads(out_e[1])["n_slots"] == 0)

patcher_c = FakePatcher(handler=FakeHandler())
many = {f"window_{i}": preset(row("b.safetensors")) for i in range(1, 12)}
out_c = node.apply(model=patcher_c, **many)
check("cap: 11 槽截断为 10", json.loads(out_c[1])["n_slots"] == 10)

patcher_k = FakePatcher(handler=FakeHandler())
out_k = node.apply(model=patcher_k, window_1=preset(row("b.safetensors")),
                   model_extra=None)
check("kwargs: 非 window_ 键忽略", json.loads(out_k[1])["n_slots"] == 1)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
