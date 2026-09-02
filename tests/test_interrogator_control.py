import types
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
# mock minimal comfy
comfy = types.ModuleType("comfy")
comfy.utils = types.SimpleNamespace(common_upscale=lambda s, w, h, m, c: s)
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils
sys.modules["torch"] = types.ModuleType("torch")
# mock folder_paths for krea2_presets import
import types as _t
fp = _t.ModuleType("folder_paths")
fp.get_user_directory = lambda: "/tmp"
sys.modules["folder_paths"] = fp
# mock server/aiohttp for route registration
aioh = _t.ModuleType("aiohttp")
aioh.web = _t.SimpleNamespace(json_response=lambda *a, **k: None, Response=lambda *a, **k: None)
sys.modules["aiohttp"] = aioh
sys.modules["aiohttp.web"] = aioh.web
srv = _t.ModuleType("server")
class _R:
    def get(self, p): return lambda fn: fn
    def post(self, p): return lambda fn: fn
    def delete(self, p): return lambda fn: fn
srv.PromptServer = type("PS", (), {"instance": type("I", (), {"routes": _R()})()})
sys.modules["server"] = srv

from nodes.model.krea2 import SFImageInterrogator

fail=0
def check(n,c):
    global fail
    print(f"[{'OK' if c else 'FAIL'}] {n}")
    if not c: fail+=1

it = SFImageInterrogator.INPUT_TYPES()
seed_def = it["required"]["seed"]
check("seed INPUT_TYPES 存在", seed_def is not None)
# seed 是 ("INT", {opts})
check("seed 类型 INT", seed_def[0]=="INT")
opts = seed_def[1]
check("seed 含 control_after_generate True", opts.get("control_after_generate") is True)
check("seed default 0", opts.get("default")==0)
check("seed min 0", opts.get("min")==0)
check("seed max 0xffffffffffffffff", opts.get("max")==0xffffffffffffffff)
# vision/thinking 仍存在且顺序在 seed 之后（widget 追加约束：thinking 在 optional 末尾）
opt_keys = list(it["optional"].keys())
check("optional 含 vision_megapixels", "vision_megapixels" in opt_keys)
check("optional 含 thinking", "thinking" in opt_keys)
check("thinking 在 optional 末尾", opt_keys[-1]=="thinking")
# 验证 control 不影响执行签名（interrogate 仍接受 seed）
import inspect
sig = inspect.signature(SFImageInterrogator.interrogate)
check("interrogate 签名含 seed", "seed" in sig.parameters)

print(f"\n{'ALL PASS' if fail==0 else str(fail)+' FAILURES'}")
sys.exit(1 if fail else 0)
