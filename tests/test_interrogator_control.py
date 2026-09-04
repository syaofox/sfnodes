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
# 新分区重排后：required 含文本→采样聚合，optional 含视觉→模板开关（已授权破兼容）
req_keys = list(it["required"].keys())
opt_keys = list(it["optional"].keys())
check("required 含 user_prompt 紧邻 prompt", "user_prompt" in req_keys and req_keys.index("user_prompt") == req_keys.index("prompt")+1)
check("required 含 min_p", "min_p" in req_keys)
check("required 含 presence_penalty", "presence_penalty" in req_keys)
check("required 含 thinking", "thinking" in req_keys)
check("required max_length 上限 8192", it["required"]["max_length"][1].get("max")==8192)
# image/video/audio 均为可选视觉组
check("image 不在 required", "image" not in it["required"])
check("image 在 optional", "image" in it["optional"])
check("image 类型 IMAGE", it["optional"]["image"][0]=="IMAGE")
check("video 在 optional", "video" in it["optional"])
check("video 类型 IMAGE", it["optional"]["video"][0]=="IMAGE")
check("audio 在 optional", "audio" in it["optional"])
check("audio 类型 AUDIO", it["optional"]["audio"][0]=="AUDIO")
check("optional 含 vision_megapixels", "vision_megapixels" in opt_keys)
check("optional 含 use_default_template", "use_default_template" in opt_keys)
check("use_default_template default True", it["optional"]["use_default_template"][1].get("default") is True)
# user_prompt 已前移至 required 紧邻 prompt
up = it["required"]["user_prompt"]
check("user_prompt 类型 STRING", up[0]=="STRING")
check("user_prompt 非 forceInput", up[1].get("forceInput") is None or up[1].get("forceInput") is not True)
check("user_prompt multiline", up[1].get("multiline") is True)
check("user_prompt default 空串", up[1].get("default")=="")
# min_p / presence_penalty 类型与默认
check("min_p 类型 FLOAT", it["required"]["min_p"][0]=="FLOAT")
check("min_p default 0.05", it["required"]["min_p"][1].get("default")==0.05)
check("presence_penalty 类型 FLOAT", it["required"]["presence_penalty"][0]=="FLOAT")
check("presence_penalty default 0.0", it["required"]["presence_penalty"][1].get("default")==0.0)
# 验证 control 不影响执行签名（interrogate 仍接受 seed）
import inspect
sig = inspect.signature(SFImageInterrogator.interrogate)
check("interrogate 签名含 seed", "seed" in sig.parameters)
check("interrogate 签名 image 可选", sig.parameters["image"].default is None)
check("interrogate 签名 video 可选", sig.parameters["video"].default is None)
check("interrogate 签名 audio 可选", sig.parameters["audio"].default is None)
check("interrogate 签名 user_prompt 紧邻 prompt", "user_prompt" in sig.parameters)
check("interrogate 签名 min_p", "min_p" in sig.parameters)
check("interrogate 签名 presence_penalty", "presence_penalty" in sig.parameters)
check("interrogate 签名 use_default_template", "use_default_template" in sig.parameters)

print(f"\n{'ALL PASS' if fail==0 else str(fail)+' FAILURES'}")
sys.exit(1 if fail else 0)
