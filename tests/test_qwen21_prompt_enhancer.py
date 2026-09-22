# SFQwenImage21PromptEnhancer 后端测试（python tests/test_qwen21_prompt_enhancer.py）
# 覆盖：INPUT_TYPES/元数据、输入校验（空 prompt/未知任务/文生图接图/图生图无图/超 8 图/
# 本地缺 clip）、本地官方PE 与本地LLM 分派（官方/通用协议、thinking 预填、图片缩放与
# 视觉占位、官方 profile 采样参数）、API 分派（打桩 chat_completion_sync，文本/多图消息、
# 透传参数）、解析兜底（原文/字段恢复）、空响应与思考链截断报错、模型族软告警、卸载调用。
# 不打桩真实网络；本地生成用 FakeClip 记录调用。
import json
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# ── 最小 mock：comfy.utils（本地图片缩放）+ comfy.model_management（卸载）──
unload_calls = []
comfy = types.ModuleType("comfy")
comfy.utils = types.SimpleNamespace(common_upscale=lambda s, w, h, m, c: s)
comfy.model_management = types.SimpleNamespace(
    unload_all_models=lambda: unload_calls.append("unload"),
    cleanup_models_gc=lambda: unload_calls.append("gc"),
    soft_empty_cache=lambda: unload_calls.append("cache"),
)
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils
sys.modules["comfy.model_management"] = comfy.model_management

# ── 最小 mock：folder_paths（llm_client 设置读取兜底）──
fp = types.ModuleType("folder_paths")
fp.get_user_directory = lambda: "/tmp/sfnodes_test_user"
sys.modules["folder_paths"] = fp

import numpy as np  # noqa: E402

from nodes.text import qwen21_prompt_enhancer as mod  # noqa: E402
from nodes.text.qwen21_prompt_enhancer import SFQwenImage21PromptEnhancer  # noqa: E402
from sf_utils.qwen21_enhance import VISION_BLOCK  # noqa: E402

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


class FakeTensor:
    """最小 IMAGE 张量替身：支持 flatten_to_rgb/_scale_frame 的切片/乘/移维。"""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype="float32")

    @property
    def shape(self):
        return self.arr.shape

    def dim(self):
        return self.arr.ndim

    def unsqueeze(self, d):
        return FakeTensor(np.expand_dims(self.arr, d))

    def __getitem__(self, key):
        return FakeTensor(self.arr[key])

    def __mul__(self, other):
        return FakeTensor(self.arr * (other.arr if isinstance(other, FakeTensor) else other))

    def clamp(self, lo, hi):
        return FakeTensor(np.clip(self.arr, lo, hi))

    def movedim(self, a, b):
        return FakeTensor(np.moveaxis(self.arr, a, b))


class FakeClip:
    def __init__(self, reply, clip_name="qwen35_9b"):
        self.reply = reply
        self.tokenizer = types.SimpleNamespace(clip_name=clip_name)
        self.calls = {}

    def tokenize(self, text, **kwargs):
        self.calls["tokenize"] = {"text": text, **kwargs}
        return {"fake": [[1, 2]]}

    def generate(self, tokens, **kwargs):
        self.calls["generate"] = kwargs
        return [1, 2, 3]

    def decode(self, ids, **kwargs):
        self.calls["decode"] = ids
        return self.reply


class FakeLlama:
    def __init__(self):
        self.calls = []
        self.reply = '{"rewritten_prompt": "llama prompt", "wh_ratio": "3:2"}'

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"choices": [{"message": {"content": self.reply}}], "usage": {"completion_tokens": 42}}


class FakeLlamaStorage:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_config = None
        self.load_calls = []
        self.clean_calls = 0

    def load_model(self, config):
        self.load_calls.append(config)
        self.current_config = dict(config)
        self.llm = FakeLlama()
        self.chat_handler = types.SimpleNamespace(
            clip_model_path=(config.get("mmproj") if config.get("mmproj") not in (None, "None") else None)
        )

    def clean(self, all=False):
        self.clean_calls += 1
        self.llm = None
        self.chat_handler = None
        self.current_config = None


LLAMA_CFG = {
    "model": "GGUF/Qwen3-VL-8B-NSFW-Caption-V4.5.Q4_K_M.gguf",
    "mmproj": "GGUF/Qwen3-VL-8B-NSFW-Caption-V4.5.mmproj-Q8_0.gguf",
    "chat_handler": "Qwen3-VL", "n_ctx": 8192, "vram_limit": -1,
    "image_min_tokens": 0, "image_max_tokens": 0,
}
LLAMA_CFG_NO_VISION = dict(LLAMA_CFG, mmproj="None")

# 软依赖插件替身：按属性指纹（LLAMA_CPP_STORAGE + llama_cpp_instruct_adv）被节点找到
llama_plugin = types.ModuleType("ComfyUI-llama-cpp_vlm_nodes")
llama_plugin.__file__ = "/x/custom_nodes/ComfyUI-llama-cpp_vlm/nodes.py"
llama_plugin.LLAMA_CPP_STORAGE = FakeLlamaStorage()
llama_plugin.llama_cpp_instruct_adv = object()
sys.modules["ComfyUI-llama-cpp_vlm_nodes"] = llama_plugin
STORAGE = llama_plugin.LLAMA_CPP_STORAGE

# 干扰项：插件注册的 torch op 命名空间同名（无 load_model/clean），不得被误认
decoy = types.ModuleType("decoy_torch_ops")
decoy.LLAMA_CPP_STORAGE = types.SimpleNamespace()
decoy.llama_cpp_instruct_adv = object()
sys.modules["decoy_torch_ops"] = decoy


NODE = SFQwenImage21PromptEnhancer()


def base_args(**overrides):
    args = dict(
        mode="本地官方PE", task="文生图", prompt="一只在雨中弹吉他的柯基",
        output_language="英文", max_tokens=8192, temperature=1.0, top_k=20, top_p=0.95,
        min_p=0.0, repetition_penalty=1.0, seed=7, thinking=True, vision_megapixels=1.0,
        detail="auto", unload_after=False,
    )
    args.update(overrides)
    return args


# ── INPUT_TYPES / 元数据 ──
it = SFQwenImage21PromptEnhancer.INPUT_TYPES()
req = it["required"]
check("mode 四选项", req["mode"][0] == ["本地官方PE", "本地LLM", "本地LLaMA", "API"]
      and req["mode"][1]["default"] == "本地官方PE")
check("task 两选项", req["task"][0] == ["文生图", "图生图"] and req["task"][1]["default"] == "文生图")
check("output_language 两选项", req["output_language"][0] == ["英文", "中文"])
check("prompt 多行", req["prompt"][0] == "STRING" and req["prompt"][1].get("multiline") is True)
check("max_tokens 默认 8192/0-32768", req["max_tokens"][0] == "INT"
      and req["max_tokens"][1]["default"] == 8192 and req["max_tokens"][1]["min"] == 0
      and req["max_tokens"][1]["max"] == 32768)
check("官方 profile 采样默认", (req["temperature"][1]["default"], req["top_k"][1]["default"],
      req["top_p"][1]["default"], req["min_p"][1]["default"], req["repetition_penalty"][1]["default"])
      == (1.0, 20, 0.95, 0.0, 1.0))
check("seed 带 control_after_generate", req["seed"][1].get("control_after_generate") is True)
check("thinking 默认 True", req["thinking"][0] == "BOOLEAN" and req["thinking"][1]["default"] is True)
check("vision_megapixels 默认 1MP", req["vision_megapixels"][1]["default"] == 1.0)
check("detail 选项", req["detail"][0] == ["auto", "low", "high"])
check("unload_after 默认 False", req["unload_after"][1]["default"] is False)
opt = it["optional"]
check("optional clip + llama_model + image_1..8 + system_prompt",
      set(opt) == {"clip", "llama_model", "system_prompt"} | {f"image_{i}" for i in range(1, 9)})
check("image 槽全 IMAGE", all(opt[f"image_{i}"][0] == "IMAGE" for i in range(1, 9)))
check("clip 为 CLIP", opt["clip"][0] == "CLIP")
check("llama_model 为 LLAMACPPMODEL", opt["llama_model"][0] == "LLAMACPPMODEL")
check("RETURN_TYPES", SFQwenImage21PromptEnhancer.RETURN_TYPES == ("STRING", "STRING", "STRING", "STRING"))
check("RETURN_NAMES", SFQwenImage21PromptEnhancer.RETURN_NAMES
      == ("enhanced_prompt", "wh_ratio", "ratio_follow", "report_json"))
check("FUNCTION", SFQwenImage21PromptEnhancer.FUNCTION == "enhance")
check("CATEGORY", SFQwenImage21PromptEnhancer.CATEGORY == "sfnodes/text")
check("DESCRIPTION 非空", bool(SFQwenImage21PromptEnhancer.DESCRIPTION))

# ── 输入校验 ──
check("空 prompt 抛错", raises(NODE.enhance, **base_args(prompt="  ")))
check("未知任务抛错", raises(NODE.enhance, **base_args(task="t2v", clip=FakeClip("x"))))
check("文生图接图抛错", raises(NODE.enhance, **base_args(image_1=FakeTensor(np.zeros((1, 2, 2, 3))))))
check("图生图无图抛错", raises(NODE.enhance, **base_args(task="图生图", clip=FakeClip("x"))))
check("超 8 图抛错（批次逐帧）", raises(NODE.enhance, **base_args(
    image_1=FakeTensor(np.zeros((8, 2, 2, 3))), image_2=FakeTensor(np.zeros((1, 2, 2, 3))))))
check("本地缺 clip 抛错", raises(NODE.enhance, **base_args()))

# ── 本地官方PE（文生图）──
clip = FakeClip('{"rewritten_prompt": "A corgi plays guitar in the rain", "wh_ratio": "16:9"}')
out = NODE.enhance(**base_args(clip=clip))
report = json.loads(out[3])
tok = clip.calls["tokenize"]
gen = clip.calls["generate"]
check("t2i 输出提示词", out[0] == "A corgi plays guitar in the rain")
check("t2i 输出 wh_ratio", out[1] == "16:9" and out[2] == "")
check("t2i 报告字段", report["mode"] == "本地官方PE" and report["task"] == "t2i"
      and report["parse_ok"] is True and report["model"] == "qwen35_9b"
      and report["thinking"] is True and report["image_count"] == 0
      and report["generated_tokens"] == 3)
check("官方 t2i 系统提示词", tok["text"].startswith("<|im_start|>system\n# Image Prompt Rewriting Expert"))
check("官方 t2i 契约在提示词内", '"wh_ratio"' in tok["text"])
check("thinking 预填", tok["text"].endswith("<|im_start|>assistant\n<think>\n") and tok["thinking"] is True)
check("t2i 无视觉占位", tok["images"] == [] and VISION_BLOCK not in tok["text"])
check("t2i 默认 max_tokens=8192", (gen["max_length"], gen["presence_penalty"], gen["do_sample"])
      == (8192, 1.5, True))
check("t2i 采样参数透传", (gen["temperature"], gen["top_k"], gen["top_p"], gen["min_p"],
      gen["repetition_penalty"], gen["seed"]) == (1.0, 20, 0.95, 0.0, 1.0, 7))
check("decode 收到生成结果", clip.calls["decode"] == [1, 2, 3])

# ── 本地官方PE（图生图，2 帧 + alpha 剥离）──
imgs = FakeTensor(np.zeros((2, 4, 4, 4)))
clip = FakeClip('{"rewritten_prompt": "Place <image1> into <image2>", "wh_ratio": "", "ratio_follow": "<image1>"}')
out = NODE.enhance(**base_args(task="图生图", clip=clip, image_1=imgs))
report = json.loads(out[3])
tok = clip.calls["tokenize"]
gen = clip.calls["generate"]
check("edit 输出 ratio_follow", out[1] == "" and out[2] == "<image1>")
check("edit 报告 image_count", report["image_count"] == 2 and report["task"] == "edit")
check("官方 edit 系统提示词", tok["text"].startswith("<|im_start|>system\n# Edit Prompt Enhancer"))
check("edit 视觉占位 2 个", tok["text"].count(VISION_BLOCK) == 2)
check("edit 图片缩放为 RGB 单帧", len(tok["images"]) == 2 and tok["images"][0].shape == (1, 4, 4, 3))
check("edit 默认 max_tokens=8192", (gen["max_length"], gen["presence_penalty"]) == (8192, 0.0))

# max_tokens=0 回退官方 profile；非零覆盖
clip = FakeClip('{"rewritten_prompt": "x", "wh_ratio": "1:1"}')
NODE.enhance(**base_args(clip=clip, max_tokens=0))
check("max_tokens=0 用官方 t2i 上限", clip.calls["generate"]["max_length"] == 16256)
clip = FakeClip('{"rewritten_prompt": "x", "wh_ratio": "1:1"}')
NODE.enhance(**base_args(clip=clip, max_tokens=4096))
check("max_tokens 非零覆盖", clip.calls["generate"]["max_length"] == 4096)

# ── 本地LLM 模式 / 模型族告警 / system_prompt 覆盖 / thinking 关闭 ──
clip = FakeClip('{"rewritten_prompt": "generic", "wh_ratio": "1:1"}', clip_name="qwen3vl_8b")
out = NODE.enhance(**base_args(mode="本地LLM", clip=clip))
tok = clip.calls["tokenize"]
check("本地LLM 用通用协议", "You are an expert prompt writer" in tok["text"]
      and "# Image Prompt Rewriting Expert" not in tok["text"])
check("本地LLM 无模型告警", json.loads(out[3])["warning"] == "")
clip = FakeClip('{"rewritten_prompt": "x", "wh_ratio": "1:1"}', clip_name="qwen3vl_8b")
out = NODE.enhance(**base_args(clip=clip))
check("官方PE 非 qwen35 告警", "qwen3vl_8b" in json.loads(out[3])["warning"])
clip = FakeClip('{"rewritten_prompt": "x", "wh_ratio": "1:1"}')
out = NODE.enhance(**base_args(clip=clip, system_prompt="CUSTOM SYS", thinking=False))
tok = clip.calls["tokenize"]
check("system_prompt 覆盖", tok["text"].startswith("<|im_start|>system\nCUSTOM SYS"))
check("thinking 关闭空块", tok["text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
      and tok["thinking"] is False and json.loads(out[3])["thinking"] is False)

# ── API 模式（打桩 chat_completion_sync）──
api_calls = []


def fake_chat(config, messages, **kwargs):
    api_calls.append({"config": config, "messages": messages, "kwargs": kwargs})
    return '{"rewritten_prompt": "api prompt", "wh_ratio": "1:1"}'


real_chat = mod.chat_completion_sync
mod.chat_completion_sync = fake_chat
try:
    out = NODE.enhance(**base_args(mode="API", temperature=0.4, seed=11))
    call = api_calls[-1]
    report = json.loads(out[3])
    check("API 输出", out[0] == "api prompt" and out[1] == "1:1")
    check("API 系统消息 = 通用协议", call["messages"][0]["role"] == "system"
          and "You are an expert prompt writer" in call["messages"][0]["content"])
    check("API 无图 user 纯文本", call["messages"][1]["content"] == "一只在雨中弹吉他的柯基")
    check("API 参数透传", call["kwargs"]["temperature"] == 0.4 and call["kwargs"]["seed"] == 11
          and call["kwargs"]["cache_key_extra"] == (11,))
    check("API 不传 max_tokens", "max_tokens" not in call["kwargs"])
    check("API 报告模型", report["model"] == call["config"]["model"] and report["thinking"] is None
          and report["generated_tokens"] is None)

    out = NODE.enhance(**base_args(
        mode="API", task="图生图", image_1=np.zeros((2, 4, 4, 3), dtype="float32")))
    call = api_calls[-1]
    parts = call["messages"][1]["content"]
    check("API 多图消息部分", isinstance(parts, list) and len(parts) == 3
          and parts[0]["type"] == "text" and all(p["type"] == "image_url" for p in parts[1:]))
    check("API 多图报告 image_count", json.loads(out[3])["image_count"] == 2)
finally:
    mod.chat_completion_sync = real_chat

# ── 本地LLaMA 模式（软依赖：按属性指纹找已加载插件）──
check("指纹查找命中真实插件替身（跳过 torch.ops 干扰项）",
      mod._find_llama_plugin() is llama_plugin)
check("本地LLaMA 缺 llama_model 抛错", raises(NODE.enhance, **base_args(mode="本地LLaMA")))

STORAGE.llm = None
out = NODE.enhance(**base_args(mode="本地LLaMA", llama_model=LLAMA_CFG))
call = STORAGE.llm.calls[-1]
report = json.loads(out[3])
check("LLaMA 首次自动加载模型", bool(STORAGE.load_calls) and STORAGE.load_calls[-1] == LLAMA_CFG)
check("LLaMA 系统消息 = 通用协议", call["messages"][0]["role"] == "system"
      and "You are an expert prompt writer" in call["messages"][0]["content"])
check("LLaMA 无图 user 纯文本", call["messages"][1]["content"] == "一只在雨中弹吉他的柯基")
check("LLaMA 采样参数", (call["kwargs"]["temperature"], call["kwargs"]["top_k"], call["kwargs"]["top_p"],
      call["kwargs"]["min_p"], call["kwargs"]["repeat_penalty"], call["kwargs"]["seed"],
      call["kwargs"]["max_tokens"]) == (1.0, 20, 0.95, 0.0, 1.0, 7, 8192))
check("LLaMA 输出解析", out[0] == "llama prompt" and out[1] == "3:2" and out[2] == "")
check("LLaMA 报告", report["mode"] == "本地LLaMA" and report["model"] == LLAMA_CFG["model"]
      and report["generated_tokens"] == 42 and report["thinking"] is None
      and report["parse_ok"] is True)

STORAGE.llm = None
out = NODE.enhance(**base_args(mode="本地LLaMA", task="图生图", llama_model=LLAMA_CFG,
                               image_1=np.zeros((2, 4, 4, 3), dtype="float32")))
call = STORAGE.llm.calls[-1]
parts = call["messages"][1]["content"]
check("LLaMA 多图 image_url 部分", isinstance(parts, list) and len(parts) == 3
      and parts[0]["type"] == "text" and all(p["type"] == "image_url" for p in parts[1:]))
check("LLaMA 图生图报告 image_count", json.loads(out[3])["image_count"] == 2)
check("LLaMA 无 mmproj 图生图抛错", raises(NODE.enhance, **base_args(
    mode="本地LLaMA", task="图生图", llama_model=LLAMA_CFG_NO_VISION,
    image_1=np.zeros((1, 2, 2, 3), dtype="float32"))))

STORAGE.llm = None
NODE.enhance(**base_args(mode="本地LLaMA", llama_model=LLAMA_CFG, max_tokens=0))
check("LLaMA max_tokens=0 不传限制", "max_tokens" not in STORAGE.llm.calls[-1]["kwargs"])
cfg2 = dict(LLAMA_CFG, n_ctx=4096)
NODE.enhance(**base_args(mode="本地LLaMA", llama_model=cfg2))
check("LLaMA 配置变化重载", STORAGE.load_calls[-1] == cfg2)
STORAGE.llm = None
NODE.enhance(**base_args(mode="本地LLaMA", llama_model=LLAMA_CFG, system_prompt="LLAMA SYS"))
check("LLaMA system_prompt 覆盖", STORAGE.llm.calls[-1]["messages"][0]["content"] == "LLAMA SYS")
STORAGE.clean_calls = 0
NODE.enhance(**base_args(mode="本地LLaMA", llama_model=LLAMA_CFG, unload_after=True))
check("LLaMA unload_after 走 storage.clean", STORAGE.clean_calls == 1)

saved_plugin = sys.modules.pop("ComfyUI-llama-cpp_vlm_nodes")
try:
    check("插件缺失抛错", raises(NODE.enhance, **base_args(mode="本地LLaMA", llama_model=LLAMA_CFG)))
finally:
    sys.modules["ComfyUI-llama-cpp_vlm_nodes"] = saved_plugin

# ── 解析兜底 / 报错 ──
out = NODE.enhance(**base_args(mode="本地LLM", clip=FakeClip("plain enhanced prompt")))
check("非 JSON 原文兜底", out[0] == "plain enhanced prompt" and json.loads(out[3])["parse_ok"] is False
      and "保留模型原文" in json.loads(out[3])["error"])
out = NODE.enhance(**base_args(mode="本地LLM", clip=FakeClip('{"rewritten_prompt": "cut", "wh_ratio": "1:1"')))
check("字段级恢复", out[0] == "cut" and json.loads(out[3])["recovered"] is True)
check("空响应抛错", raises(NODE.enhance, **base_args(clip=FakeClip(""))))
check("思考链无答案抛错", raises(NODE.enhance, **base_args(clip=FakeClip("reasoning</think>"))))
check("官方PE 未闭合思考链抛错", raises(NODE.enhance, **base_args(
    clip=FakeClip("long reasoning without any json"))))
out = NODE.enhance(**base_args(clip=FakeClip("plain text answer"), thinking=False))
check("关闭 thinking 允许原文兜底", out[0] == "plain text answer")

# ── 卸载 ──
unload_calls.clear()
NODE.enhance(**base_args(clip=FakeClip('{"rewritten_prompt": "x", "wh_ratio": "1:1"}')))
check("默认不卸载", unload_calls == [])
NODE.enhance(**base_args(clip=FakeClip('{"rewritten_prompt": "x", "wh_ratio": "1:1"}'), unload_after=True))
check("unload_after 触发卸载", unload_calls == ["unload", "gc", "cache"])

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
