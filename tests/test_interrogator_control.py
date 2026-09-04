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
# vision/thinking/user_prompt 仍存在；user_prompt 为末尾多行文本框（可连可填），thinking 为倒二
opt_keys = list(it["optional"].keys())
check("optional 含 vision_megapixels", "vision_megapixels" in opt_keys)
check("optional 含 thinking", "thinking" in opt_keys)
check("optional 含 user_prompt", "user_prompt" in opt_keys)
check("user_prompt 在 optional 末尾（文本框化追加）", opt_keys[-1]=="user_prompt")
check("thinking 为倒二（在 user_prompt 之前）", opt_keys[-2]=="thinking")
# image/video/audio 均为可选（对齐原生 Generate Text）
check("image 不在 required", "image" not in it["required"])
check("image 在 optional", "image" in it["optional"])
check("image 类型 IMAGE", it["optional"]["image"][0]=="IMAGE")
check("video 在 optional", "video" in it["optional"])
check("video 类型 IMAGE", it["optional"]["video"][0]=="IMAGE")
check("audio 在 optional", "audio" in it["optional"])
check("audio 类型 AUDIO", it["optional"]["audio"][0]=="AUDIO")
# user_prompt 已改为节点内文本框
up = it["optional"]["user_prompt"]
check("user_prompt 类型 STRING", up[0]=="STRING")
check("user_prompt 非 forceInput", up[1].get("forceInput") is None or up[1].get("forceInput") is not True)
check("user_prompt multiline", up[1].get("multiline") is True)
check("user_prompt default 空串", up[1].get("default")=="")
# 验证 control 不影响执行签名（interrogate 仍接受 seed）
import inspect
sig = inspect.signature(SFImageInterrogator.interrogate)
check("interrogate 签名含 seed", "seed" in sig.parameters)
check("interrogate 签名 image 可选", sig.parameters["image"].default is None)
check("interrogate 签名 video 可选", sig.parameters["video"].default is None)
check("interrogate 签名 audio 可选", sig.parameters["audio"].default is None)
check("interrogate 签名 user_prompt 可选", "user_prompt" in sig.parameters)

print(f"\n{'ALL PASS' if fail==0 else str(fail)+' FAILURES'}")
sys.exit(1 if fail else 0)
