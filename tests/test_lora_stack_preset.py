# SFLoraStack preset_export 模拟测试（Python 直接运行：python3 tests/test_lora_stack_preset.py）
# mock folder_paths/comfy/aiohttp/server，验证：
#   - 结构：第 5 输出不 сдви旧 4 输出（序号兼容）
#   - 导出形状：SFLoraPreset 契约 {loras: [{lora,on,strength,strengthTwo}], positive}
#   - 语义：仅实际生效行（关的行剔除，零强度行保留由下游过滤）、preset 输入优先透传
#   - 端到端形状：导出可被窗口槽解析（键集/类型与 _parse_preset_rows 契约一致）
import importlib.util
import json
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


# ── 真临时文件（_build_plan 查 os.path.isfile）──────────────────────────────
tmpd = tempfile.mkdtemp(prefix="sf_stack_preset_")
PATHS = {}
for _n in ("a.safetensors", "b.safetensors", "c.safetensors"):
    _p = os.path.join(tmpd, _n)
    open(_p, "wb").close()
    PATHS[_n] = _p

# ── mock 模块 ──────────────────────────────────────────────────────────────
fp_mod = types.ModuleType("folder_paths")
fp_mod.get_full_path = lambda cat, name: PATHS.get(name)
fp_mod.get_user_directory = lambda: "/tmp/sf_test_user"
sys.modules["folder_paths"] = fp_mod

comfy_pkg = types.ModuleType("comfy")
sd_mod = types.ModuleType("comfy.sd")
CALLS = []


def _fake_load_lora_for_models(model, clip, lora, sm, sc, **kw):
    CALLS.append((sm, sc))
    return model, clip


sd_mod.load_lora_for_models = _fake_load_lora_for_models
utils_mod = types.ModuleType("comfy.utils")
utils_mod.load_torch_file = lambda path, **kw: ({"weights": path}, {"meta": 1})
comfy_pkg.sd = sd_mod
comfy_pkg.utils = utils_mod
sys.modules["comfy"] = comfy_pkg
sys.modules["comfy.sd"] = sd_mod
sys.modules["comfy.utils"] = utils_mod

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

for name, path in [
    ("sfnodes", root),
    ("sfnodes.sf_utils", os.path.join(root, "sf_utils")),
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


_load("sfnodes.nodes.model.lora_preset", "nodes/model/lora_preset.py")
smod = _load("sfnodes.nodes.model.lora_stack", "nodes/model/lora_stack.py")

SFLoraStack = smod.SFLoraStack
node = SFLoraStack()


def state(loras, **kw):
    d = {"loras": loras, "sep": ", "}
    d.update(kw)
    return json.dumps(d)


def row(name, on=True, sm=1.0, sc=None, triggers=None):
    r = {"name": name, "on": on, "sm": sm, "triggers": triggers or []}
    if sc is not None:
        r["sc"] = sc
    return r


MODEL, CLIP = object(), object()

# ── 结构 ───────────────────────────────────────────────────────────────────
check("structure: 5 输出", SFLoraStack.RETURN_TYPES == (
    "MODEL", "CLIP", "STRING", "STRING", "SF_LORA_PRESET"))
check("structure: 旧 4 输出不动",
      SFLoraStack.RETURN_TYPES[:4] == ("MODEL", "CLIP", "STRING", "STRING")
      and SFLoraStack.RETURN_NAMES[:4] == ("MODEL", "CLIP", "triggers", "positive"))
check("structure: 第 5 输出名", SFLoraStack.RETURN_NAMES[4] == "preset_export")
check("structure: tooltips 5 个", len(SFLoraStack.OUTPUT_TOOLTIPS) == 5)
check("structure: DESCRIPTION", bool(SFLoraStack.DESCRIPTION))

# ── 导出语义 ───────────────────────────────────────────────────────────────
out = node.apply(
    model=MODEL, clip=CLIP,
    LoraLoaderState=state([
        row("a.safetensors", on=True, sm=1.0, sc=0.5, triggers=["AAA"]),
        row("b.safetensors", on=False, sm=1.0),
        row("c.safetensors", on=True, sm=0.0),
    ], positive="hello"))
check("apply: 5 元组", len(out) == 5)
check("apply: MODEL/CLIP 透传", out[0] is MODEL and out[1] is CLIP)
check("apply: triggers 仅开着的行", out[2] == "AAA")
check("apply: positive 透传", out[3] == "hello")
exp = out[4]
check("export: 顶层形状", set(exp) == {"loras", "positive"} and exp["positive"] == "hello")
check("export: 关的行剔除", [r["lora"] for r in exp["loras"]] == ["a.safetensors", "c.safetensors"])
check("export: 行形状", all(set(r) == {"lora", "on", "strength", "strengthTwo"} for r in exp["loras"]))
check("export: 强度映射 sm/sc",
      exp["loras"][0] == {"lora": "a.safetensors", "on": True, "strength": 1.0, "strengthTwo": 0.5})
check("export: 零强度行保留（下游过滤）",
      exp["loras"][1] == {"lora": "c.safetensors", "on": True, "strength": 0.0, "strengthTwo": 0.0})
check("export: 触发词不携带", all("triggers" not in r for r in exp["loras"]))

# ── preset 输入优先透传 ────────────────────────────────────────────────────
out2 = node.apply(
    model=MODEL, clip=CLIP,
    preset={"loras": [{"lora": "b.safetensors", "on": True, "strength": 0.7,
                       "strengthTwo": 0.3}], "positive": "pos2"},
    LoraLoaderState=state([row("a.safetensors", on=True, sm=1.0)]))
check("preset-override: 导出走预设行",
      out2[4]["loras"] == [{"lora": "b.safetensors", "on": True,
                            "strength": 0.7, "strengthTwo": 0.3}])
check("preset-override: positive 走预设", out2[4]["positive"] == "pos2")

# ── 空栈 ───────────────────────────────────────────────────────────────────
out3 = node.apply(model=MODEL, LoraLoaderState=state([]))
check("empty: 导出空 preset", out3[4] == {"loras": [], "positive": ""})
check("empty: 前 4 输出正常", out3[:4] == (MODEL, None, "", ""))

# ── 纯配置源（model 悬空）：零文件 IO，导出一致 ─────────────────────────────
READS = []
_orig_load = utils_mod.load_torch_file
utils_mod.load_torch_file = lambda path, **kw: (READS.append(path), ({"w": 1}, None))[1]
try:
    out4 = node.apply(
        model=None, clip=None,
        LoraLoaderState=state([
            row("a.safetensors", on=True, sm=1.0, sc=0.5, triggers=["AAA"]),
            row("b.safetensors", on=False, sm=1.0),
            row("missing.safetensors", on=True, sm=1.0),
        ], positive="hello"))
finally:
    utils_mod.load_torch_file = _orig_load
check("export-only: 零文件读取", READS == [])
check("export-only: MODEL/CLIP 回空", out4[0] is None and out4[1] is None)
check("export-only: triggers 一致", out4[2] == "AAA")
check("export-only: positive 一致", out4[3] == "hello")
check("export-only: 导出与接 MODEL 逐字节一致",
      out4[4] == {"loras": [
          {"lora": "a.safetensors", "on": True, "strength": 1.0, "strengthTwo": 0.5},
      ], "positive": "hello"})
check("export-only: 缺失文件照样过滤", all(
    r["lora"] != "missing.safetensors" for r in out4[4]["loras"]))
check("structure: model 改 optional",
      "model" not in SFLoraStack.INPUT_TYPES()["required"]
      and SFLoraStack.INPUT_TYPES()["optional"]["model"][0] == "MODEL")

# ── 窗口槽契约（端到端形状不断）─────────────────────────────────────────────
# SFWanWindowLoRA._parse_preset_rows 消费 {lora,on,strength}，忽略 strengthTwo/positive
for r in exp["loras"]:
    check(f"contract: 行 {r['lora']} 有 lora/on/strength",
          isinstance(r["lora"], str) and isinstance(r["on"], bool)
          and isinstance(r["strength"], (int, float)))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
