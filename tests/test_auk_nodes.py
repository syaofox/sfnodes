# SF AuK 四节点后端测试（python tests/test_auk_nodes.py）
# 覆盖：V1 schema（类型/槽位/默认值/可选输入）、OpenAI 设置校验与归一化、llama.cpp 适配
# （本地 client 加载模型、cleanup 卸载）、Loader 选项与显存档位校验、Generate 输入校验
# （空指令/时长范围/Flash 固定配方/30s 预算）与音频输出形状、消息与采样参数透传。
# 不发网络请求、不加载真实模型：torch/torchaudio/soundfile/folder_paths/omegaconf/openai
# 全部为最小桩，引擎用 FakeEngine。

import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)


# ── 最小 mock：torch（FakeTensor + cuda）──
class FakeTensor:
    def __init__(self, shape, finite=True):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self._finite = finite

    def __getitem__(self, index):
        if isinstance(index, int):
            return FakeTensor(self.shape[1:], self._finite)
        return self

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim if dim >= 0 else len(shape) + dim + 1, 1)
        return FakeTensor(shape, self._finite)

    def squeeze(self, dim=0):
        if dim < len(self.shape) and self.shape[dim] == 1:
            shape = list(self.shape)
            del shape[dim]
            return FakeTensor(shape, self._finite)
        return self

    def mean(self, dim=0, keepdim=False):
        shape = list(self.shape)
        if keepdim:
            shape[dim] = 1
        else:
            del shape[dim]
        return FakeTensor(shape, self._finite)

    def numpy(self):
        return object()

    def all(self):
        return True


class _Cuda:
    device_count_calls = 0

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def device(index):
        import contextlib

        return contextlib.nullcontext()

    @staticmethod
    def is_bf16_supported():
        return True


torch = types.ModuleType("torch")
torch.cuda = _Cuda()
torch.float32 = "float32"
torch.float16 = "float16"
torch.bfloat16 = "bfloat16"
torch.Tensor = FakeTensor
torch.is_tensor = lambda value: isinstance(value, FakeTensor)
torch.isfinite = lambda value: FakeTensor(getattr(value, "shape", ()), finite=True)
torch.manual_seed = lambda seed: None
sys.modules["torch"] = torch

torchaudio = types.ModuleType("torchaudio")
torchaudio.functional = types.SimpleNamespace(resample=lambda waveform, src, dst: waveform)
torchaudio.transforms = types.SimpleNamespace(Resample=lambda src, dst: (lambda waveform: waveform))
sys.modules["torchaudio"] = torchaudio

comfy = types.ModuleType("comfy")
comfy.model_management = types.SimpleNamespace(
    unload_all_models=lambda: None,
    soft_empty_cache=lambda: None,
)
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = comfy.model_management

sf = types.ModuleType("soundfile")
sf.read = lambda *args, **kwargs: (None, 24000)
sf.write = lambda *args, **kwargs: None
sf.info = lambda *args, **kwargs: types.SimpleNamespace(frames=0, samplerate=24000)
sys.modules["soundfile"] = sf

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = os.path.join(root, "user", "models")
folder_paths.added = []
folder_paths.add_model_folder_path = lambda name, path, is_default=False: folder_paths.added.append((name, path))
folder_paths.get_folder_paths = lambda name: []
sys.modules["folder_paths"] = folder_paths

# omegaconf（auk_paths 用；yaml + 属性访问对象）
import yaml  # noqa: E402


class _Cfg:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        value = self._data[name]
        return _Cfg(value) if isinstance(value, dict) else value

    def get(self, name, default=None):
        value = self._data.get(name, default)
        return _Cfg(value) if isinstance(value, dict) else value


omegaconf = types.ModuleType("omegaconf")
omegaconf.OmegaConf = types.SimpleNamespace(load=lambda path: _Cfg(yaml.safe_load(open(path, encoding="utf-8"))))
sys.modules["omegaconf"] = omegaconf

# openai（_LocalClient 返回对象）
openai_mod = types.ModuleType("openai")
openai_types = types.ModuleType("openai.types")
openai_chat = types.ModuleType("openai.types.chat")


class ChatCompletion:
    @classmethod
    def model_validate(cls, result):
        return result


openai_chat.ChatCompletion = ChatCompletion
sys.modules["openai"] = openai_mod
sys.modules["openai.types"] = openai_types
sys.modules["openai.types.chat"] = openai_chat

# pe 模块桩（_LocalClient.create 仅需要 PromptEnhancerError）
pe_stub = types.ModuleType("nodes.audio.auk.infer.pe")


class PromptEnhancerError(RuntimeError):
    pass


pe_stub.PromptEnhancerError = PromptEnhancerError
sys.modules["nodes.audio.auk.infer.pe"] = pe_stub

# llama-cpp 插件桩（find_llama_plugin 按 sys.modules 属性指纹查找）
storage = types.SimpleNamespace(load_model_calls=[], clean_calls=0)
storage.llm = None
storage.current_config = None


class _FakeLlm:
    def __init__(self):
        self.last_request = None

    def create_chat_completion(self, **request):
        self.last_request = request
        return {"choices": [{"message": {"content": "ok"}}], "model": "fake"}


def _load_model(config):
    storage.llm = _FakeLlm()
    storage.current_config = config
    storage.load_model_calls.append(config)


def _clean():
    storage.clean_calls += 1
    storage.llm = None
    storage.current_config = None


storage.load_model = _load_model
storage.clean = _clean
plugin = types.ModuleType("fake_llama_plugin")
plugin.LLAMA_CPP_STORAGE = storage
plugin.llama_cpp_instruct_adv = object()
sys.modules["fake_llama_plugin"] = plugin

from nodes.audio import auk_config as C  # noqa: E402
from nodes.audio import auk_generate as G  # noqa: E402
from nodes.audio import auk_loader as L  # noqa: E402

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


# ── 注册/槽类型 ──
check("引擎槽类型 SF_AUK_ENGINE", L.AUK_ENGINE == "SF_AUK_ENGINE")
check("设置槽类型 SF_AUK_LLM_CONFIG", C.AUK_LLM_CONFIG == "SF_AUK_LLM_CONFIG")
check("模型目录注册 7 条", len(folder_paths.added) == 7 and all(name in ("auk", "auk_qwen", "auk_vae") for name, _ in folder_paths.added))

# ── SF AuK OpenAI Settings ──
schema = C.SFAuKOpenAISettings.INPUT_TYPES()
check("OpenAI schema 7 项", set(schema["required"]) == {"base_url", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout_sec"})
check("OpenAI 返回类型", C.SFAuKOpenAISettings.RETURN_TYPES == ("SF_AUK_LLM_CONFIG",))
(settings,) = C.SFAuKOpenAISettings().execute("https://api.example.com/v1/chat/completions", "", "my-model")
check("base_url 去掉 /chat/completions", settings.base_url == "https://api.example.com/v1")
check("空 key 占位 not-required", settings.api_key == "not-required")
check("enhancer_kwargs 映射", settings.enhancer_kwargs()["llm_base_url"] == "https://api.example.com/v1"
      and settings.enhancer_kwargs()["llm_top_p"] == 1.0 and settings.enhancer_kwargs()["llm_timeout"] == 120)
check("api_key 不进 repr", "not-required" not in repr(settings))
check("非法 base_url 拒绝", raises(C.SFAuKOpenAISettings().execute, "ftp://x", "k", "m"))
check("base_url 带 query 拒绝", raises(C.SFAuKOpenAISettings().execute, "https://x/v1?a=1", "k", "m"))
check("空模型拒绝", raises(C.SFAuKOpenAISettings().execute, "https://x/v1", "k", " "))
check("温度越界拒绝", raises(C.SFAuKOpenAISettings().execute, "https://x/v1", "k", "m", 3.0))
check("top_p 越界拒绝", raises(C.SFAuKOpenAISettings().execute, "https://x/v1", "k", "m", 0.0, 0.0))
check("max_tokens 越界拒绝", raises(C.SFAuKOpenAISettings().execute, "https://x/v1", "k", "m", 0.0, 1.0, 0))

# ── SF AuK Llama.cpp Adapter ──
schema = C.SFAuKLlamaCppSettings.INPUT_TYPES()
check("Llama schema 输入类型", schema["required"]["llama_model"][0] == "LLAMACPPMODEL")
check("Llama 返回类型", C.SFAuKLlamaCppSettings.RETURN_TYPES == ("SF_AUK_LLM_CONFIG",))
check("Llama 非法模型拒绝", raises(C.SFAuKLlamaCppSettings().execute, None))
check("Llama 空模型名拒绝", raises(C.SFAuKLlamaCppSettings().execute, {"mmproj": "x"}))
(llama_settings,) = C.SFAuKLlamaCppSettings().execute({"model": "m.gguf"}, 0.2, 512, True)
check("Llama 设置返回", llama_settings.temperature == 0.2 and llama_settings.max_tokens == 512 and llama_settings.unload_after_enhance)
check("Llama enhancer_kwargs 本地", llama_settings.enhancer_kwargs()["llm_base_url"] == "local://llama-cpp")
client = llama_settings.enhancer_kwargs()["llm_client"]
result = client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.0, top_p=0.9)
check("本地 client 加载模型并返回", result is not None and storage.load_model_calls[-1] == {"model": "m.gguf"})
check("本地 client 透传 top_p", storage.llm.last_request["top_p"] == 0.9 and storage.llm.last_request["stream"] is False)
llama_settings.cleanup()
check("cleanup 卸载匹配配置", storage.clean_calls == 1 and storage.llm is None)
(no_unload,) = C.SFAuKLlamaCppSettings().execute({"model": "m.gguf"}, 0.0, 16, False)
no_unload.cleanup()
check("unload_after_enhance=False 不卸载", storage.clean_calls == 1)

# ── SF AuK Models Loader ──
schema = L.SFAuKModelsLoader.INPUT_TYPES()
check("Loader schema 六项", set(schema["required"]) == {"model_name", "qwen_name", "memory_mode", "dtype", "device", "sequential_cfg"})
check("Loader 无模型占位", schema["required"]["model_name"][0] == ["No AuK models found"] and schema["required"]["qwen_name"][0] == ["No Qwen2.5-Omni models found"])
check("Loader 档位选项", schema["required"]["memory_mode"][0] == ["low_vram", "balanced", "max_vram"])
check("Loader 无 CUDA 占位", schema["required"]["device"][0] == ["CUDA unavailable"])
check("Loader 非法档位拒绝", raises(L.SFAuKModelsLoader().execute, "x", "y", "turbo", "bf16", "cuda:0"))
check("Loader 非法设备拒绝", raises(L.SFAuKModelsLoader().execute, "x", "y", "low_vram", "fp32", "cuda:0"))


# ── SF AuK Generate / Edit ──
class FakeInference:
    def __init__(self, is_flash=False, sample_rate=24000, downsample=2048):
        self.is_flash = is_flash
        self.target_sample_rate = sample_rate
        self.downsample_rate = downsample
        self.memory_mode = "low_vram"
        self.generate_calls = []

    def generate(self, messages, **kwargs):
        self.generate_calls.append((messages, kwargs))
        return FakeTensor((1, 1000)), self.target_sample_rate


class FakeEngine:
    def __init__(self, **kwargs):
        import threading

        self.inference = FakeInference(**kwargs)
        self.lock = threading.Lock()


schema = G.SFAuKGenerateEdit.INPUT_TYPES()
check("Generate required 八项", set(schema["required"]) == {"engine", "instruction", "generation_seconds", "use_prompt_enhancer", "seed", "nfe_steps", "cfg_strength", "sway_sampling_coef"})
check("Generate optional 两项", set(schema["optional"]) == {"input_audio", "llm_config"})
check("Generate 槽类型", schema["required"]["engine"][0] == "SF_AUK_ENGINE" and schema["optional"]["input_audio"][0] == "AUDIO" and schema["optional"]["llm_config"][0] == "SF_AUK_LLM_CONFIG")
check("Generate 输出", G.SFAuKGenerateEdit.RETURN_TYPES == ("AUDIO", "STRING", "STRING") and G.SFAuKGenerateEdit.RETURN_NAMES == ("generated_audio", "model_instruction", "prompt_enhancer_info"))
check("seed 保持 fixed", schema["required"]["seed"][1]["control_after_generate"] == "fixed")

node = G.SFAuKGenerateEdit()
engine = FakeEngine()
check("空指令拒绝", raises(node.execute, engine, "  ", 1.0, False, 42))
check("时长超范围拒绝", raises(node.execute, engine, "hi", 31.0, False, 42))
check("时长负数拒绝", raises(node.execute, engine, "hi", -1.0, False, 42))
check("关增强且 0 秒拒绝", raises(node.execute, engine, "hi", 0.0, False, 42))
check("Flash 配方校验", raises(node.execute, FakeEngine(is_flash=True), "hi", 1.0, False, 42, 32, 2.0, -1.0))

audio_out, instruction, info = node.execute(engine, " hello ", 1.0, False, 42)
check("返回指令去空白", instruction == "hello")
check("未开启增强信息", info == "Prompt Enhancer: disabled")
check("音频输出形状", audio_out["waveform"].shape == (1, 1, 1000) and audio_out["sample_rate"] == 24000)
check("采样参数透传", engine.inference.generate_calls[-1][1] == {
    "audio": None, "gen_seconds": 1.0, "nfe": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0, "seed": 42,
})
check("消息文本", engine.inference.generate_calls[-1][0][0]["content"] == [{"type": "text", "text": "hello"}])

flash_engine = FakeEngine(is_flash=True)
node.execute(flash_engine, "hi", 2.0, False, 7, 4, 0.0, -1.0)
check("Flash 合法配方可执行", len(flash_engine.inference.generate_calls) == 1)

wav = {"waveform": FakeTensor((1, 2, 44100)), "sample_rate": 44100}
audio_out, _, _ = node.execute(engine, "hi", 1.0, False, 42, input_audio=wav)
check("立体声转单声道并重采样", engine.inference.generate_calls[-1][1]["audio"][1] == 24000)
check("消息携带音频", engine.inference.generate_calls[-1][0][0]["content"][1]["type"] == "audio")
check("input_audio 非法批大小拒绝", raises(node.execute, engine, "hi", 1.0, False, 42, input_audio={"waveform": FakeTensor((2, 1, 100)), "sample_rate": 24000}))
check("input_audio 非法采样率拒绝", raises(node.execute, engine, "hi", 1.0, False, 42, input_audio={"waveform": FakeTensor((1, 1, 100)), "sample_rate": 0}))
check("30s 预算拒绝", raises(node.execute, engine, "hi", 1.0, False, 42, input_audio={"waveform": FakeTensor((1, 1, 30 * 24000)), "sample_rate": 24000}))

if failures:
    print(f"\n{len(failures)} 项失败：")
    for name in failures:
        print("  -", name)
    sys.exit(1)
print("\nOK")
