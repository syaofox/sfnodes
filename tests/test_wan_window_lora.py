# SFWanWindowLoRA 模拟测试（Python 直接运行：python3 tests/test_wan_window_lora.py）
# mock 官方链路（comfy.lora/convert/context_windows/model_patcher）+
# safetensors + folder_paths，无需 torch（patch-swap 本身无 torch 依赖），验证：
#   - 灵活 schema：window_99 放行、VALIDATE_INPUTS 接管
#   - preset 行解析：on 开关/strength（模型侧）/strengthTwo 被忽略/坏行丢弃
#   - patcher.patches 只读：apply 全程不动上游 base（含烘焙态假设）
#   - 流式函数装配：并集键后装 LowVramPatch（绑定独立槽表）、官方函数保留、
#     旧会话僵尸清除、无 weight_function 属性的模块建表
#   - 换槽：EVALUATE 按 window_idx 取模改表内容、空槽 []、越界轮回；
#     CLEANUP 计数；无 handler 回退官方 add_patches(slot0)
#   - 双重计数 guard：同一键 base 烘焙/流式 + 本槽流式各算一次
#     （official 用精确类 mock：覆盖"官方占位不跳过我方"的共存回归）
#   - GGUF 双通道（§39.11）：量化层不装 weight_function，走张量 patches；
#     base 快照保留/换槽重建/空槽纯 base/跨会话不累积
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
RAW_A, MODEL_A = "net.A", "diffusion_model.blocks.0.attn.wq.weight"
RAW_B, MODEL_B = "net.B", "diffusion_model.blocks.0.attn.wk.weight"
RAW_C, MODEL_C = "net.C", "diffusion_model.blocks.0.mlp.gate.weight"
RAW_D, MODEL_D = "net.D", "diffusion_model.blocks.1.attn.wq.weight"
RAW_Y, MODEL_Y = "net.Y", "diffusion_model.blocks.9.ghost.weight"  # state 无此键
RAW_X = "net.X"  # key_map 无此键
RAW_G, MODEL_G = "net.G", "diffusion_model.blocks.2.attn.wq.weight"  # GGUF 专用键
RAW2MODEL = {RAW_A: MODEL_A, RAW_B: MODEL_B, RAW_C: MODEL_C,
             RAW_D: MODEL_D, RAW_Y: MODEL_Y, RAW_G: MODEL_G}
MODEL_STATE_KEYS = [MODEL_A, MODEL_B, MODEL_C, MODEL_D, MODEL_G]
MOD_BASENAME = {
    MODEL_A: "diffusion_model.blocks.0.attn.wq",
    MODEL_B: "diffusion_model.blocks.0.attn.wk",
    MODEL_C: "diffusion_model.blocks.0.mlp.gate",
    MODEL_D: "diffusion_model.blocks.1.attn.wq",
    MODEL_G: "diffusion_model.blocks.2.attn.wq",
}


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
    "/loras/g.safetensors": _make_raw([RAW_G]),
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


class MockLowVramPatch:
    """模拟官方 LowVramPatch：live 读 patches[key]，返回各行强度序列。"""

    def __init__(self, key, patches, convert_func=None, set_func=None):
        self.key = key
        self.patches = patches
        self.cleared = 0

    def __call__(self, weight):
        return [t[0] for t in self.patches[self.key]]

    def clear_prepared(self):
        self.cleared += 1


class OfficialFn(MockLowVramPatch):
    """上游 base 流式函数 mock 的子类形态（pre_official 预装用；精确类形态
    见 compose 用例的 official_fn，覆盖与上游共存回归）。"""


mp_mod = types.ModuleType("comfy.model_patcher")
mp_mod.LowVramPatch = MockLowVramPatch
mp_mod.get_key_weight = lambda model, key: (None, f"set:{key}", f"conv:{key}")
comfy_pkg.lora = lora_mod
comfy_pkg.lora_convert = conv_mod
comfy_pkg.context_windows = cw_mod
comfy_pkg.model_patcher = mp_mod
sys.modules["comfy"] = comfy_pkg
sys.modules["comfy.lora"] = lora_mod
sys.modules["comfy.lora_convert"] = conv_mod
sys.modules["comfy.context_windows"] = cw_mod
sys.modules["comfy.model_patcher"] = mp_mod

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
class FakeModule:
    def __init__(self, with_fns=True):
        if with_fns:
            self.weight_function = []
            self.bias_function = []


class FakeRoot:
    def __init__(self, names, pre_official=()):
        self._mods = [(MOD_BASENAME[k], FakeModule()) for k in names]
        for k in pre_official:
            mod = dict(self._mods)[MOD_BASENAME[k]]
            mod.weight_function.append(OfficialFn(k, "OFFICIAL_PATCHES"))

    def named_modules(self):
        return list(self._mods)

    def state_dict(self):
        return {k: None for k in MODEL_STATE_KEYS}


class FakeHandler:
    def __init__(self):
        self.callbacks = {}


class FakePatcher:
    def __init__(self, names=None, base_patches=None, handler=None):
        self.model = FakeRoot(names or [])
        self.patches = dict(base_patches or {})
        self.model_options = {"context_handler": handler} if handler is not None else {}
        self.added = []  # add_patches 调用记录（回退路径）

    def clone(self):
        return self

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.added.append((dict(patches), strength_patch, strength_model))
        for k, v in patches.items():
            self.patches.setdefault(k, []).append((strength_patch, v, strength_model, None, None))
        return list(patches)


class FakeGGMLWeight:
    """模拟 ComfyUI-GGUF 的量化张量：补丁只经 `.patches` 生效。"""

    def __init__(self, patches=None):
        self.patches = list(patches or [])


class FakeGGMLModule:
    """模拟 GGMLLayer：is_ggml_quantized() 为真，无 weight_function 消耗。"""

    def __init__(self, patches=None):
        self.weight = FakeGGMLWeight(patches)
        self.bias = None

    def is_ggml_quantized(self, *a, **k):
        return True


class FakeGGMLRoot:
    """MODEL_G 单键 GGUF 根（base 即上游 base 注入的张量条目）。"""

    def __init__(self, base=None):
        self._mods = [(MOD_BASENAME[MODEL_G], FakeGGMLModule(base))]

    def named_modules(self):
        return list(self._mods)

    def state_dict(self):
        return {k: None for k in MODEL_STATE_KEYS}


class FakeGGMLPatcher:
    def __init__(self, handler=None, base=None):
        self.model = FakeGGMLRoot(base)
        self.patches = {}
        self.model_options = {"context_handler": handler} if handler is not None else {}

    def clone(self):
        return self


MODEL_NAMES = [MODEL_A, MODEL_B, MODEL_C, MODEL_D]


def _reader_dispatch(handler, call_type, *args):
    """复刻上游 get_all_callbacks 的真实读取契约（§39.8）：
    以 handler.callbacks 为 transformer_options 形状再取 ["callbacks"]。
    直接形状 [call_type] 在现行 reader 下永远读不到。"""
    cbs = handler.callbacks.get("callbacks", {})
    fns = []
    for lst in cbs.get(call_type, {}).values():
        fns.extend(lst)
    assert len(fns) == 1, f"expected 1 {call_type} cb via reader, got {len(fns)}"
    fns[0](handler, *args)


def fire_evaluate(handler, window_idx):
    _reader_dispatch(handler, "evaluate_context_windows",
                     None, None, None, None, {}, window_idx, None)


def fire_cleanup(handler):
    _reader_dispatch(handler, "execute_cleanup", None, None, None, {}, {})


def live_strengths(patcher, model_key):
    """模拟 forward：执行该层全部 weight_function，返回各函数看到的强度。"""
    mod = dict(patcher.model.named_modules())[MOD_BASENAME[model_key]]
    return [fn(None) for fn in mod.weight_function]


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
check("structure: stream 可用", nmod._HAS_STREAM is True)

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

# ── apply：3 槽 + 上游 base（含 D 独占键）───────────────────────────────────
BASE_T = ("base-strength", "base-data", 1.0, None, None)
handler = FakeHandler()
patcher = FakePatcher(MODEL_NAMES,
                      base_patches={MODEL_A: [BASE_T]},
                      handler=handler)
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
# patcher.patches 只读：上游 base 原样，他槽零写入
check("readonly: patcher.patches 只有 base", patcher.patches == {MODEL_A: [BASE_T]})
check("callbacks: 双形状已注册（直接 + callbacks 分支）",
      "sf_wan_window_lora" in handler.callbacks.get("evaluate_context_windows", {})
      and "sf_wan_window_lora" in handler.callbacks.get("execute_cleanup", {})
      and "sf_wan_window_lora" in handler.callbacks.get("callbacks", {}).get(
          "evaluate_context_windows", {})
      and "sf_wan_window_lora" in handler.callbacks.get("callbacks", {}).get(
          "execute_cleanup", {}))
sess = patcher._sf_window_session
check("session: 诊断可达", sess is not None and sess.n == 3)

# ── 首次换槽：流式函数装配 ─────────────────────────────────────────────────
fire_evaluate(handler, 0)
mods = dict(patcher.model.named_modules())
fns_a = mods[MOD_BASENAME[MODEL_A]].weight_function
fns_d = mods[MOD_BASENAME[MODEL_D]].weight_function
check("stream: A 装了我方函数", any(getattr(f, nmod.FN_TAG, None) for f in fns_a))
check("stream: 我方函数绑定独立槽表",
      all(getattr(f, "patches", None) is sess.tables
          for f in fns_a if getattr(f, nmod.FN_TAG, None)))
check("stream: D 独占键也装上", any(getattr(f, nmod.FN_TAG, None) for f in fns_d))
check("stream: patcher.patches 仍只有 base", patcher.patches == {MODEL_A: [BASE_T]})
# w0 内容：A=[a1.0]（base 不在槽表内！base 走官方表）
check("tables w0: A 仅本槽行", sess.tables[MODEL_A] == [(1.0, ("adapter", RAW_A), 1.0, None, None)])
check("tables w0: B 行序拼接",
      [t[0] for t in sess.tables[MODEL_B]] == [1.0, 0.5])

# ── forward 语义：base（官方表）+ 本槽（独立表）各算一次 ────────────────────
# 模拟：官方 base 列表 + 我方函数 live 读表。
# official_fn 必须用精确类（回归 §39.11：旧代码 `fn.__class__ is LowVramPatch`
# 的 official 跳过分支只对精确类生效，子类 mock 永远测不到它）。
patcher2 = FakePatcher(MODEL_NAMES, base_patches={MODEL_A: [BASE_T]}, handler=FakeHandler())
mods2 = dict(patcher2.model.named_modules())
official_fn = MockLowVramPatch(MODEL_A, patcher2.patches)
mods2[MOD_BASENAME[MODEL_A]].weight_function.append(official_fn)
node.apply(model=patcher2,
           window_1=preset(row("a.safetensors", strength=1.0)),
           window_2=None)
fire_evaluate(patcher2.model_options["context_handler"], 0)
got = live_strengths(patcher2, MODEL_A)
check("compose: 官方函数仍在", official_fn in mods2[MOD_BASENAME[MODEL_A]].weight_function)
check("compose: 我方函数追加（官方占位不跳过）",
      any(getattr(f, nmod.FN_TAG, None) for f in mods2[MOD_BASENAME[MODEL_A]].weight_function))
check("compose: base 与本槽都可见",
      [t[0] for t in patcher2.patches[MODEL_A]] in got and [1.0] in got)
check("compose: 无双重（恰两个函数）", len(got) == 2)

# ── 空槽与轮回 ─────────────────────────────────────────────────────────────
fire_evaluate(handler, 1)
check("eval w1（空槽）: A 表空", sess.tables[MODEL_A] == [])
check("eval w1（空槽）: D 表空", sess.tables[MODEL_D] == [])
fire_evaluate(handler, 2)
check("eval w2: D = c行 strength 2.0",
      sess.tables[MODEL_D] == [(2.0, ("adapter", RAW_D), 1.0, None, None)])
fire_evaluate(handler, 3)
check("eval w3 越界轮回 slot0", [t[0] for t in sess.tables[MODEL_B]] == [1.0, 0.5])

# ── 僵尸清除：旧会话函数不双重应用 ──────────────────────────────────────────
zombie = MockLowVramPatch(MODEL_B, {"other": "dict"})
setattr(zombie, nmod.FN_TAG, True)
mods[MOD_BASENAME[MODEL_B]].weight_function.append(zombie)
sess._stream_ready = False
fire_evaluate(handler, 0)
ours_b = [f for f in mods[MOD_BASENAME[MODEL_B]].weight_function
          if getattr(f, nmod.FN_TAG, None) and getattr(f, "patches", None) is sess.tables]
check("zombie: 旧会话僵尸被清除", zombie not in mods[MOD_BASENAME[MODEL_B]].weight_function)
check("zombie: 本会话函数唯一", len(ours_b) == 1)

# ── CLEANUP 计数（按变计数）────────────────────────────────────────────────
c0 = sess.swap_count
fire_evaluate(handler, 2)
fire_evaluate(handler, 2)  # 连续同槽：第二次不涨
check("recount: 连续同槽只计一次", sess.swap_count == c0 + 1)
before = (sess.swap_count, [t[0] for t in sess.tables[MODEL_B]])
fire_cleanup(handler)
check("cleanup: 不换槽只计数", sess.swap_count == before[0])
check("cleanup: 表内容不动", [t[0] for t in sess.tables[MODEL_B]] == before[1])
check("cleanup: 计数>0（之前换过）", sess.swap_count > 0)

# ── 注册失败回退官方静态 slot0 ───────────────────────────────────────────────
class BrokenHandler:
    def __init__(self):
        self.callbacks = None  # 非 dict → 注册抛错


patcher_broken = FakePatcher(MODEL_NAMES)
patcher_broken.model_options = {"context_handler": BrokenHandler()}
node.apply(model=patcher_broken, window_1=preset(row("a.safetensors", strength=1.0)))
check("register-fail: 回退 add_patches",
      [(sorted(p.keys()), s) for p, s, _ in patcher_broken.added] == [
          (sorted([MODEL_A, MODEL_B]), 1.0)])

# ── 无 handler 回退官方静态 slot0 ───────────────────────────────────────────
patcher_nh = FakePatcher(MODEL_NAMES)
out_nh = node.apply(model=patcher_nh, window_1=preset(row("a.safetensors", strength=1.0),
                                                      row("b.safetensors", strength=0.5)))
check("no-handler: add_patches 逐行调用",
      [(sorted(p.keys()), s) for p, s, _ in patcher_nh.added] == [
          (sorted([MODEL_A, MODEL_B]), 1.0), (sorted([MODEL_B, MODEL_C]), 0.5)])
check("no-handler: patches 官方语义", patcher_nh.patches[MODEL_B][0][0] == 1.0
      and patcher_nh.patches[MODEL_B][1][0] == 0.5)
check("no-handler: n_slots 照记", json.loads(out_nh[1])["n_slots"] == 1)

# ── 诊断：零匹配 warning 不报错 ─────────────────────────────────────────────
patcher_y = FakePatcher(MODEL_NAMES, handler=FakeHandler())
out_y = node.apply(model=patcher_y,
                   window_1=preset(row("y.safetensors")),
                   window_2=preset(row("b.safetensors")))
info_y = json.loads(out_y[1])
check("diag: y 零匹配进 failed 原因",
      "0 model keys" in info_y["slots"][0].get("status", ""))
check("diag: b matched 2", info_y["slots"][1]["loras"][0]["model_keys_matched"] == 2)

# ── 异常路径 ───────────────────────────────────────────────────────────────
patcher_m = FakePatcher(MODEL_NAMES, handler=FakeHandler())
out_m = node.apply(model=patcher_m,
                   window_1=preset(row("missing.safetensors")),
                   window_2=preset(row("b.safetensors")))
info_m = json.loads(out_m[1])
check("load-fail: 槽1 empty", info_m["slots"][0].get("status", "").startswith("empty"))
check("load-fail: 槽2 正常", len(info_m["slots"][1]["loras"]) == 1)

patcher_x = FakePatcher(MODEL_NAMES, handler=FakeHandler())
out_x = node.apply(model=patcher_x, window_1=preset(row("x.safetensors")))
check("unmapped: 文件键全无映射则槽 empty",
      json.loads(out_x[1])["slots"][0].get("status", "").startswith("empty"))

patcher_e = FakePatcher(MODEL_NAMES, handler=FakeHandler())
out_e = node.apply(model=patcher_e)
check("no-slots: passthrough", out_e[0] is patcher_e)
check("no-slots: n_slots=0", json.loads(out_e[1])["n_slots"] == 0)

patcher_c = FakePatcher(MODEL_NAMES, handler=FakeHandler())
many = {f"window_{i}": preset(row("b.safetensors")) for i in range(1, 12)}
out_c = node.apply(model=patcher_c, **many)
check("cap: 11 槽截断为 10", json.loads(out_c[1])["n_slots"] == 10)

patcher_k = FakePatcher(MODEL_NAMES, handler=FakeHandler())
out_k = node.apply(model=patcher_k, window_1=preset(row("b.safetensors")),
                   model_extra=None)
check("kwargs: 非 window_ 键忽略", json.loads(out_k[1])["n_slots"] == 1)

# ── GGUF 双通道（§39.11）：量化层走张量 patches，不装 weight_function ─────────
# 模拟 ComfyUI-GGUF：GGMLLayer.forward 经 get_weight 只读 weight.patches，
# weight_function 永不被调用。base 为上游注入的张量条目，必须原样保留。
GGUF_BASE_T = ("base-strength", "base-data", 1.0, None, None)
handler_g = FakeHandler()
pg = FakeGGMLPatcher(handler=handler_g, base=[([GGUF_BASE_T], MODEL_G)])
out_g = node.apply(model=pg,
                   window_1=preset(row("g.safetensors", strength=2.0)),
                   window_2=None)
sess_g = pg._sf_window_session
gmod = dict(pg.model.named_modules())[MOD_BASENAME[MODEL_G]]
check("gguf: 探测命中", nmod._is_ggml_quantized(gmod) is True)
fire_evaluate(handler_g, 0)
# _gguf 登记发生在首次 EVALUATE 的 _ensure（load 之后），apply 时尚未登记
check("gguf: 量化键登记张量通道", MODEL_G in sess_g._gguf)
check("gguf: 不装 weight_function", not any(
    getattr(f, nmod.FN_TAG, None) for f in getattr(gmod, "weight_function", [])))
expected_g = [(2.0, ("adapter", RAW_G), 1.0, None, None)]
check("gguf: w0 = base + 本槽",
      gmod.weight.patches == [([GGUF_BASE_T], MODEL_G), (expected_g, MODEL_G)])
check("gguf: base 条目原样保留",
      gmod.weight.patches[0] == ([GGUF_BASE_T], MODEL_G))
check("gguf: base 快照已记",
      getattr(gmod.weight, nmod._GGUF_BASE_ATTR, None) == [([GGUF_BASE_T], MODEL_G)])
fire_evaluate(handler_g, 1)  # 空槽
check("gguf: 空槽只剩 base",
      gmod.weight.patches == [([GGUF_BASE_T], MODEL_G)])
fire_evaluate(handler_g, 0)
check("gguf: 回槽不累积",
      gmod.weight.patches == [([GGUF_BASE_T], MODEL_G), (expected_g, MODEL_G)])
# 跨会话：新 apply + 新 handler；marker 常驻旧 param，base 复用不累积
handler_g2 = FakeHandler()
pg.model_options = {"context_handler": handler_g2}
node.apply(model=pg,
           window_1=preset(row("g.safetensors", strength=2.0)),
           window_2=None)
fire_evaluate(handler_g2, 0)
check("gguf: 跨会话不累积",
      gmod.weight.patches == [([GGUF_BASE_T], MODEL_G), (expected_g, MODEL_G)])

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
