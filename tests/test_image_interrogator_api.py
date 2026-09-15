# SFImageInterrogatorAPI 后端测试（python tests/test_image_interrogator_api.py）
# 覆盖：INPUT_TYPES/元数据、单帧取帧与 alpha 黑底预乘/越界、请求消息构造
# （预设回退/user_prompt 追加/system 覆盖）、temperature/max_tokens 透传、错误传播。
# 网络调用打桩（chat_completion_sync），不发真实请求。

import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# ── mock 最小 comfy / torch / folder_paths / aiohttp / server（krea2 导入链）──
comfy = types.ModuleType("comfy")
comfy.utils = types.SimpleNamespace(common_upscale=lambda s, w, h, m, c: s)
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils
sys.modules["torch"] = types.ModuleType("torch")

fp = types.ModuleType("folder_paths")
fp.get_user_directory = lambda: "/tmp/sfnodes_test_user"
sys.modules["folder_paths"] = fp

aioh = types.ModuleType("aiohttp")
aioh.web = types.SimpleNamespace(json_response=lambda *a, **k: None, Response=lambda *a, **k: None)
sys.modules["aiohttp"] = aioh
sys.modules["aiohttp.web"] = aioh.web

srv = types.ModuleType("server")


class _R:
    def get(self, p):
        return lambda fn: fn

    def post(self, p):
        return lambda fn: fn

    def delete(self, p):
        return lambda fn: fn


srv.PromptServer = type("PS", (), {"instance": type("I", (), {"routes": _R()})()})
sys.modules["server"] = srv

import numpy as np  # noqa: E402

from nodes.model import image_interrogator_api as mod  # noqa: E402
from nodes.model.image_interrogator_api import SFImageInterrogatorAPI  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except Exception:
        return True


# ── INPUT_TYPES / 元数据 ──
it = SFImageInterrogatorAPI.INPUT_TYPES()
req = it["required"]
check("image 必填 IMAGE", req["image"][0] == "IMAGE")
check("preset 为列表", isinstance(req["preset"][0], list) and "default" in req["preset"][0])
check("prompt 多行", req["prompt"][0] == "STRING" and req["prompt"][1].get("multiline") is True)
check("user_prompt 在 required", "user_prompt" in req)
check("temperature FLOAT", req["temperature"][0] == "FLOAT")
check("max_tokens INT", req["max_tokens"][0] == "INT")
check("vision_megapixels FLOAT", req["vision_megapixels"][0] == "FLOAT")
check("detail 三档", req["detail"][0] == ["auto", "low", "high"] and req["detail"][1].get("default") == "auto")
check("frame_index 默认 0", req["frame_index"][1].get("default") == 0)
check("seed INT + control_after_generate", req["seed"][0] == "INT" and req["seed"][1].get("control_after_generate") is True)
check("send_seed 布尔默认 False", req["send_seed"][0] == "BOOLEAN" and req["send_seed"][1].get("default") is False)
check("system_prompt 可选 forceInput", it["optional"]["system_prompt"][1].get("forceInput") is True)
check("RETURN_TYPES", SFImageInterrogatorAPI.RETURN_TYPES == ("STRING",))
check("RETURN_NAMES", SFImageInterrogatorAPI.RETURN_NAMES == ("text",))
check("FUNCTION", SFImageInterrogatorAPI.FUNCTION == "interrogate")
check("CATEGORY", SFImageInterrogatorAPI.CATEGORY == "sfnodes/model")
check("DESCRIPTION 非空", bool(SFImageInterrogatorAPI.DESCRIPTION))
check("VALIDATE_INPUTS True", SFImageInterrogatorAPI.VALIDATE_INPUTS(preset="whatever") is True)

# ── 取帧 / alpha 预乘 / 越界 ──
node = SFImageInterrogatorAPI()
arr = np.zeros((3, 4, 6, 3), dtype="float32")
arr[0, :, :, :] = 1.0
arr[2, :, :, :] = 0.5
pil0 = SFImageInterrogatorAPI._frame_to_pil(arr, 0)
check("取帧尺寸", pil0.size == (6, 4))
check("取帧像素值", pil0.getpixel((0, 0))[0] == 255)
pil_last = SFImageInterrogatorAPI._frame_to_pil(arr, -1)
check("负索引取末帧", pil_last.getpixel((0, 0))[0] == 128)
check("越界抛错", raises(SFImageInterrogatorAPI._frame_to_pil, arr, 5))
check("负越界抛错", raises(SFImageInterrogatorAPI._frame_to_pil, arr, -4))

rgba = np.zeros((1, 2, 2, 4), dtype="float32")
rgba[0, :, :, :3] = 1.0   # RGB 全白
rgba[0, :, :, 3] = 0.0    # alpha 0 -> 黑底预乘应得黑
check("alpha 黑底预乘", SFImageInterrogatorAPI._frame_to_pil(rgba, 0).getpixel((0, 0))[0] == 0)

# ── interrogate：打桩 chat_completion_sync / get_llm_config ──
captured = {}
real_sync = mod.chat_completion_sync
real_cfg = mod.get_llm_config


def fake_sync(config, messages, **kwargs):
    captured["config"] = config
    captured["messages"] = messages
    captured["kwargs"] = kwargs
    return "A DESCRIBED SCENE"


mod.chat_completion_sync = fake_sync
mod.get_llm_config = lambda: {"provider": "deepseek", "base_url": "u", "model": "m", "api_key": "k"}

try:
    out = node.interrogate(
        image=arr, preset="default", prompt="", user_prompt="keep clothing",
        temperature=0.3, max_tokens=512, seed=5, send_seed=False,
        vision_megapixels=1.0, detail="high", frame_index=0,
    )
    check("输出文本", out == ("A DESCRIBED SCENE",))
    check("config 来自 get_llm_config", captured["config"]["api_key"] == "k")
    check("两条消息", len(captured["messages"]) == 2)
    check("system + user", captured["messages"][0]["role"] == "system" and captured["messages"][1]["role"] == "user")
    check("temperature 透传", captured["kwargs"]["temperature"] == 0.3)
    check("max_tokens 透传", captured["kwargs"]["max_tokens"] == 512)
    check("send_seed=False 不下发 seed", captured["kwargs"]["seed"] is None)
    check("seed 进缓存键", captured["kwargs"]["cache_key_extra"] == (5,))
    user_content = captured["messages"][1]["content"]
    check("content 含文本与图片", user_content[0]["type"] == "text" and user_content[1]["type"] == "image_url")
    instruction = user_content[0]["text"]
    check("prompt 空回退预设", "Generate a detailed paragraph" in instruction)
    check("user_prompt 追加", instruction.endswith("keep clothing"))
    check("detail 透传", user_content[1]["image_url"]["detail"] == "high")
    check("data URL 前缀", user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    # system_prompt 覆盖
    captured.clear()
    node.interrogate(
        image=arr, preset="default", prompt="custom instruction", user_prompt="",
        temperature=0.3, max_tokens=512, seed=9, send_seed=True,
        vision_megapixels=1.0, detail="auto", frame_index=0,
        system_prompt="MY SYSTEM",
    )
    check("system_prompt 覆盖", captured["messages"][0]["content"] == "MY SYSTEM")
    check("prompt 非空覆盖预设", captured["messages"][1]["content"][0]["text"] == "custom instruction")
    check("auto detail 保留字段", captured["messages"][1]["content"][1]["image_url"]["detail"] == "auto")
    check("send_seed=True 下发 seed", captured["kwargs"]["seed"] == 9)

    # 错误传播
    def boom(*a, **k):
        raise RuntimeError("API 挂了")

    mod.chat_completion_sync = boom
    check("API 错误传播", raises(
        node.interrogate,
        image=arr, preset="default", prompt="x", user_prompt="",
        temperature=0.3, max_tokens=512, seed=0, send_seed=False,
        vision_megapixels=1.0, detail="auto", frame_index=0,
    ))
finally:
    mod.chat_completion_sync = real_sync
    mod.get_llm_config = real_cfg

print(f"\nFAILURES: {len(failures)}")
if failures:
    sys.exit(1)
print("test_image_interrogator_api: all assertions passed")
