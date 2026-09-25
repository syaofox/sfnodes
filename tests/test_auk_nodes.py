# SF AuK 四节点后端测试（python tests/test_auk_nodes.py）
# 覆盖：V1 schema（类型/槽位/默认值/可选输入）、OpenAI 设置校验与归一化、llama.cpp 适配
# （本地 client 加载模型、cleanup 卸载）、Loader 选项/显存档位校验/切换模型释放旧引擎与重试、
# Generate 输入校验（空指令/时长范围/Flash 自动锁定/30s 预算）与音频输出形状、消息与采样
# 参数透传、进度条映射与中断检查。
# 不发网络请求、不加载真实模型：torch/torchaudio/soundfile/folder_paths/omegaconf/openai
# 全部为最小桩，引擎用 FakeEngine/FakeAukInfer。

import json
import math
import os
import sys
import tempfile
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

    @property
    def T(self):
        return self

    def __getitem__(self, index):
        if isinstance(index, int):
            return FakeTensor(self.shape[1:], self._finite)
        if (isinstance(index, tuple) and len(index) == 2
                and index[0] is Ellipsis and isinstance(index[1], slice)):
            start, stop, step = index[1].indices(self.shape[-1])
            shape = list(self.shape)
            shape[-1] = len(range(start, stop, step))
            return FakeTensor(shape, self._finite)
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


class _FakeCudaOOM(RuntimeError):
    pass


torch.cuda.OutOfMemoryError = _FakeCudaOOM
torch.float32 = "float32"
torch.float16 = "float16"
torch.bfloat16 = "bfloat16"
torch.Tensor = FakeTensor
torch.is_tensor = lambda value: isinstance(value, FakeTensor)
torch.isfinite = lambda value: FakeTensor(getattr(value, "shape", ()), finite=True)
torch.manual_seed = lambda seed: None
torch.zeros = lambda *shape, **kwargs: FakeTensor(tuple(int(s) for s in shape))


def _fake_cat(tensors, dim=-1):
    total = sum(t.shape[dim] for t in tensors)
    shape = list(tensors[0].shape)
    shape[dim] = total
    return FakeTensor(tuple(shape))


torch.cat = _fake_cat
sys.modules["torch"] = torch

torchaudio = types.ModuleType("torchaudio")
torchaudio.functional = types.SimpleNamespace(resample=lambda waveform, src, dst: waveform)
torchaudio.transforms = types.SimpleNamespace(Resample=lambda src, dst: (lambda waveform: waveform))
sys.modules["torchaudio"] = torchaudio

comfy = types.ModuleType("comfy")


class FakeProgressBar:
    instances = []

    def __init__(self, total):
        self.total = total
        self.updates = []
        FakeProgressBar.instances.append(self)

    def update_absolute(self, value, total=None):
        self.updates.append((value, total))


interrupt_checks = []


comfy.utils = types.SimpleNamespace(ProgressBar=FakeProgressBar)
comfy.model_management = types.SimpleNamespace(
    unload_all_models=lambda: None,
    soft_empty_cache=lambda: None,
    throw_exception_if_processing_interrupted=lambda: interrupt_checks.append(1),
)
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils
sys.modules["comfy.model_management"] = comfy.model_management

# torch.nn（memory_utils 的 import torch.nn.functional；fuse_layers 未被测试调用）
torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional

sf = types.ModuleType("soundfile")
sf_read_calls = []
sf_write_calls = []


def _fake_sf_read(path, *args, **kwargs):
    sf_read_calls.append(path)
    return (None, 24000)


def _fake_sf_write(path, data, samplerate, **kwargs):
    with open(path, "wb") as handle:  # 真实建文件，便于断言临时 WAV 清理
        handle.write(b"wav")
    sf_write_calls.append((str(path), samplerate))


sf.read = _fake_sf_read
sf.write = _fake_sf_write
sf.info = lambda *args, **kwargs: types.SimpleNamespace(frames=0, samplerate=24000)
sys.modules["soundfile"] = sf

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = os.path.join(root, "user", "models")
folder_paths.added = []
folder_paths.paths = {}
folder_paths.add_model_folder_path = lambda name, path, is_default=False: folder_paths.added.append((name, path))
folder_paths.get_folder_paths = lambda name: folder_paths.paths.get(name, [])
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
# 假估计：0.1 秒/字符（便于手算分段）；units = seconds × 默认语速
pe_stub.estimate_speech_seconds = lambda text, language=None: len(str(text)) * 0.1
pe_stub.DEFAULT_SPEECH_RATE = 4.1511
pe_stub.estimate_speech_units = lambda text, language=None: len(str(text)) * 0.1 * 4.1511


class FakeSenseVoice:
    """SenseVoiceSmallASR 替身：记录 language 与音频路径，返回可配置结果。

    texts 为列表时逐次弹出（逐块 ASR 用）；否则用 text。
    """

    calls = []
    paths = []
    texts = None
    text = "识别出的文字"
    error = None

    def __init__(self, *, language="auto", **kwargs):
        self.language = language
        FakeSenseVoice.calls.append(language)

    def transcribe(self, audio_path):
        FakeSenseVoice.paths.append(str(audio_path))
        text = FakeSenseVoice.texts.pop(0) if FakeSenseVoice.texts else FakeSenseVoice.text
        return types.SimpleNamespace(text=text, language="zh", error=FakeSenseVoice.error)


pe_stub.SenseVoiceSmallASR = FakeSenseVoice
sys.modules["nodes.audio.auk.infer.pe"] = pe_stub

# infer_auk 模块桩（Loader 的延迟导入走 sys.modules，不加载 transformers）


class FakeInsufficientVRAMError(ValueError):
    pass


class FakeAukInfer:
    instances = []
    behavior = "retry"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.released = False
        FakeAukInfer.instances.append(self)
        if len(FakeAukInfer.instances) == 1 and FakeAukInfer.behavior == "retry":
            kwargs["vram_retry"]()  # 模拟引擎内部：放置失败 → 回调释放旧引擎 → 重试成功
        elif len(FakeAukInfer.instances) == 1 and FakeAukInfer.behavior == "oom":
            raise torch.cuda.OutOfMemoryError("simulated")

    def release(self):
        self.released = True


infer_auk_stub = types.ModuleType("nodes.audio.auk.infer.infer_auk")
infer_auk_stub.AukInfer = FakeAukInfer
infer_auk_stub.InsufficientVRAMError = FakeInsufficientVRAMError
sys.modules["nodes.audio.auk.infer.infer_auk"] = infer_auk_stub

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
from nodes.audio import auk_long_speech as LS  # noqa: E402
from nodes.audio import auk_transcribe as T  # noqa: E402

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


# ── euler_final 进度回调（按文件路径加载，绕开 model/__init__ 的重依赖链）──
import importlib.util  # noqa: E402

_mu_spec = importlib.util.spec_from_file_location(
    "auk_memory_utils", os.path.join(root, "nodes/audio/auk/model/memory_utils.py")
)
MU = importlib.util.module_from_spec(_mu_spec)
_mu_spec.loader.exec_module(MU)
_steps = []
_value = MU.euler_final(lambda t, x: 0.0, 1.0, [0.0, 0.5, 1.0],
                        progress_cb=lambda done, total: _steps.append((done, total)))
check("euler_final 逐步回调", _steps == [(1, 2), (2, 2)] and _value == 1.0)


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

# ── Loader：切换模型释放旧引擎 + 重试（引擎模块桩）──
tmp_models = tempfile.mkdtemp(prefix="sf_auk_models_")
os.makedirs(os.path.join(tmp_models, "auk", "AuK"))
os.makedirs(os.path.join(tmp_models, "qwen", "Qwen2.5-Omni-3B"))
with open(os.path.join(tmp_models, "auk", "AuK", "config.yaml"), "w", encoding="utf-8") as handle:
    handle.write("model:\n  name: AuK\n")
open(os.path.join(tmp_models, "auk", "AuK", "auk_base.safetensors"), "wb").write(b"")
open(os.path.join(tmp_models, "auk", "AuK", "vae.safetensors"), "wb").write(b"")
with open(os.path.join(tmp_models, "qwen", "Qwen2.5-Omni-3B", "config.json"), "w", encoding="utf-8") as handle:
    json.dump({"model_type": "qwen2_5_omni"}, handle)
folder_paths.paths = {
    "auk": [os.path.join(tmp_models, "auk")],
    "auk_qwen": [os.path.join(tmp_models, "qwen")],
    "auk_vae": [os.path.join(tmp_models, "auk")],
}
torch.cuda.device_count = lambda: 1

check("Loader IS_CHANGED 恒 NaN", math.isnan(L.SFAuKModelsLoader.IS_CHANGED(model_name="x")))


class _OldInference:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


loader = L.SFAuKModelsLoader()
old_inference = _OldInference()
L._CACHE.clear()
L._CACHE[("old-key",)] = L.AuKEngine(old_inference)
FakeAukInfer.instances = []
FakeAukInfer.behavior = "retry"
(engine,) = loader.execute("AuK/auk_base.safetensors", "Qwen2.5-Omni-3B", "max_vram", "bf16", "cuda:0")
check("切换模型释放旧引擎", old_inference.released and ("old-key",) not in L._CACHE)
check("新引擎入缓存", len(L._CACHE) == 1 and list(L._CACHE.values())[0] is engine)
check("vram_retry 由引擎回调触发", len(FakeAukInfer.instances) == 1 and engine.inference is FakeAukInfer.instances[0])

(engine2,) = loader.execute("AuK/auk_base.safetensors", "Qwen2.5-Omni-3B", "max_vram", "bf16", "cuda:0")
check("同参数命中缓存不重建", engine2 is engine and len(FakeAukInfer.instances) == 1)

L._CACHE.clear()
old_inference2 = _OldInference()
L._CACHE[("old-key-2",)] = L.AuKEngine(old_inference2)
FakeAukInfer.instances = []
FakeAukInfer.behavior = "oom"
(engine3,) = loader.execute("AuK/auk_base.safetensors", "Qwen2.5-Omni-3B", "low_vram", "bf16", "cuda:0")
check("放置 OOM 释放旧引擎后重试", old_inference2.released and len(FakeAukInfer.instances) == 2
      and engine3.inference is FakeAukInfer.instances[1])
L._CACHE.clear()
check("_release_others 无其他引擎返回 0", L._release_others(("none",)) == 0)


# ── SF AuK Audio Transcribe ──
schema = T.SFAuKAudioTranscribe.INPUT_TYPES()
check("Transcribe schema 两项", set(schema["required"]) == {"audio", "language"})
check("Transcribe 输入类型", schema["required"]["audio"][0] == "AUDIO" and schema["required"]["language"][0][0] == "自动")
check("Transcribe 默认自动", schema["required"]["language"][1]["default"] == "自动")
check("Transcribe 输出", T.SFAuKAudioTranscribe.RETURN_TYPES == ("STRING",) and T.SFAuKAudioTranscribe.RETURN_NAMES == ("text",))

transcribe_node = T.SFAuKAudioTranscribe()
FakeSenseVoice.calls.clear()
FakeSenseVoice.paths.clear()
sf_write_calls.clear()
(text,) = transcribe_node.execute({"waveform": FakeTensor((1, 1, 16000)), "sample_rate": 16000}, "中文")
check("Transcribe 返回文字", text == "识别出的文字")
check("Transcribe 语言映射", FakeSenseVoice.calls[-1] == "zh")
check("Transcribe 写临时 wav 并传入", bool(sf_write_calls) and FakeSenseVoice.paths[-1] == sf_write_calls[-1][0]
      and FakeSenseVoice.paths[-1].endswith(".wav"))
check("Transcribe 临时文件已清理", not os.path.exists(sf_write_calls[-1][0]))

transcribe_node.execute({"waveform": FakeTensor((1, 1, 16000)), "sample_rate": 16000}, "自动")
check("Transcribe 自动语言映射", FakeSenseVoice.calls[-1] == "auto")
check("Transcribe 未知语言拒绝", raises(transcribe_node.execute, {"waveform": FakeTensor((1, 1, 16000)), "sample_rate": 16000}, "法语"))
check("Transcribe 批大小 1 校验", raises(transcribe_node.execute, {"waveform": FakeTensor((2, 1, 16000)), "sample_rate": 16000}, "自动"))

FakeSenseVoice.error = "boom"
check("Transcribe 识别失败报错", raises(transcribe_node.execute, {"waveform": FakeTensor((1, 1, 16000)), "sample_rate": 16000}, "自动"))
check("Transcribe 失败也清理临时文件", not os.path.exists(FakeSenseVoice.paths[-1]))
FakeSenseVoice.error = None
FakeSenseVoice.text = "   "
check("Transcribe 空文本报错", raises(transcribe_node.execute, {"waveform": FakeTensor((1, 1, 16000)), "sample_rate": 16000}, "自动"))
FakeSenseVoice.text = "识别出的文字"


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
        callback = kwargs.get("progress_cb")
        if callback is not None:
            callback("encode", 1, 1)
            callback("sample", 3, 4)
            callback("decode", 1, 1)
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
flash_engine = FakeEngine(is_flash=True)
node.execute(flash_engine, "hi", 1.0, False, 42, 32, 2.0, -1.0)
flash_call = flash_engine.inference.generate_calls[-1][1]
check("Flash 自动锁定配方", (flash_call["nfe"], flash_call["cfg_strength"], flash_call["sway_sampling_coef"]) == (4, 0.0, -1.0))
check("Flash 忽略 widget 步数（进度条按 4 步）", FakeProgressBar.instances[-1].total == 7)

audio_out, instruction, info = node.execute(engine, " hello ", 1.0, False, 42)
check("返回指令去空白", instruction == "hello")
check("未开启增强信息", info == "Prompt Enhancer: disabled")
check("音频输出形状", audio_out["waveform"].shape == (1, 1, 1000) and audio_out["sample_rate"] == 24000)
check("采样参数透传", {k: v for k, v in engine.inference.generate_calls[-1][1].items() if k != "progress_cb"} == {
    "audio": None, "gen_seconds": 1.0, "nfe": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0, "seed": 42,
})
check("消息文本", engine.inference.generate_calls[-1][0][0]["content"] == [{"type": "text", "text": "hello"}])
bar = FakeProgressBar.instances[-1]
check("进度条总格数（32 步无 PE）", bar.total == 35)
check("进度条阶段映射", bar.updates[-3:] == [(2, 35), (26, 35), (35, 35)])
check("进度回调触发中断检查", len(interrupt_checks) >= 3)

flash_ok = FakeEngine(is_flash=True)
node.execute(flash_ok, "hi", 2.0, False, 7, 4, 0.0, -1.0)
check("Flash 已符合配方可直接执行", len(flash_ok.inference.generate_calls) == 1)

wav = {"waveform": FakeTensor((1, 2, 44100)), "sample_rate": 44100}
audio_out, _, _ = node.execute(engine, "hi", 1.0, False, 42, input_audio=wav)
check("立体声转单声道并重采样", engine.inference.generate_calls[-1][1]["audio"][1] == 24000)
check("消息携带音频", engine.inference.generate_calls[-1][0][0]["content"][1]["type"] == "audio")
check("input_audio 非法批大小拒绝", raises(node.execute, engine, "hi", 1.0, False, 42, input_audio={"waveform": FakeTensor((2, 1, 100)), "sample_rate": 24000}))
check("input_audio 非法采样率拒绝", raises(node.execute, engine, "hi", 1.0, False, 42, input_audio={"waveform": FakeTensor((1, 1, 100)), "sample_rate": 0}))
check("30s 预算拒绝", raises(node.execute, engine, "hi", 1.0, False, 42, input_audio={"waveform": FakeTensor((1, 1, 30 * 24000)), "sample_rate": 24000}))

# ── SF AuK Long Speech ──
schema = LS.SFAuKLongSpeech.INPUT_TYPES()
required_keys = {"engine", "text", "mode", "max_chunk_seconds", "speech_rate", "ref_tail_seconds",
                 "reference_seconds", "pause_seconds", "continuity", "seed", "nfe_steps", "cfg_strength",
                 "sway_sampling_coef", "trim_trailing_silence"}
check("LongSpeech required 十四项", set(schema["required"]) == required_keys)
check("LongSpeech 语速默认 4.15", schema["required"]["speech_rate"][1]["default"] == 4.15
      and schema["required"]["speech_rate"][1]["min"] == 3.0 and schema["required"]["speech_rate"][1]["max"] == 6.0)
check("LongSpeech optional 九项", set(schema["optional"]) == {"input_audio", "voice_description",
      "instruction", "duration_mode", "speed_multiplier",
      "edit_operation", "edit_target", "edit_new", "edit_anchor"})
check("LongSpeech 输出", LS.SFAuKLongSpeech.RETURN_TYPES == ("AUDIO", "STRING")
      and LS.SFAuKLongSpeech.RETURN_NAMES == ("audio", "report"))
check("LongSpeech 模式/连续性选项", schema["required"]["mode"][0] == ["参考音色 TTS", "声音描述 TTS", "长音频处理（编辑/增强）", "语音内容编辑（替换/增添/删除）"]
      and schema["required"]["continuity"][0] == ["滚动参考", "同一参考"]
      and schema["optional"]["duration_mode"][0] == ["等长", "变速"])

long_node = LS.SFAuKLongSpeech()
check("LongSpeech 空文本拒绝", raises(long_node.execute, FakeEngine(), "  ", "参考音色 TTS"))
check("LongSpeech 参考模式缺音频拒绝", raises(long_node.execute, FakeEngine(), "你好。", "参考音色 TTS"))
check("LongSpeech 描述模式缺描述拒绝", raises(long_node.execute, FakeEngine(), "你好。", "声音描述 TTS"))
check("LongSpeech 未知模式拒绝", raises(long_node.execute, FakeEngine(), "你好。", "未知"))
check("LongSpeech 未知连续性拒绝", raises(long_node.execute, FakeEngine(), "你好。", "声音描述 TTS",
      voice_description="温柔女声", continuity="未知"))

real_fade = LS._fade_edges
LS._fade_edges = lambda waveform, sample_rate, fade_ms=5.0, fade_in=True, fade_out=True: waveform
try:
    long_engine = FakeEngine()
    ref = {"waveform": FakeTensor((1, 1, 24000 * 6)), "sample_rate": 24000}
    audio_out, report = long_node.execute(
        long_engine, "第一句。第二句。第三句。第四句。", "参考音色 TTS",
        max_chunk_seconds=0.5, ref_tail_seconds=2.0, reference_seconds=4.0, pause_seconds=0.1,
        trim_trailing_silence=False, input_audio=ref,
    )
    parsed = json.loads(report)
    calls = long_engine.inference.generate_calls
    check("LongSpeech 逐段生成", parsed["chunks"] == 4 and len(calls) == 4)
    check("LongSpeech 段种子递增", [call[1]["seed"] for call in calls] == [42, 43, 44, 45])
    check("LongSpeech 首段用输入参考并裁剪", parsed["segments"][0]["reference"] == "input_audio"
          and parsed["segments"][0]["reference_seconds"] == 4.0)
    check("LongSpeech 后续段滚动参考", all(seg["reference"] == "rolling" for seg in parsed["segments"][1:]))
    check("LongSpeech 逐段消息模板", "第一句。" in calls[0][0][0]["content"][0]["text"]
          and calls[0][0][0]["content"][0]["text"].startswith("Say the following with the same voice"))
    check("LongSpeech 进度条总格", FakeProgressBar.instances[-1].total == 4 * (32 + 3))
    check("LongSpeech 拼接长度", audio_out["waveform"].shape == (1, 1, 4 * 1000 + 3 * 2400)
          and audio_out["sample_rate"] == 24000)
    check("LongSpeech 报告总时长", abs(parsed["total_seconds"] - (4 * 1000 + 3 * 2400) / 24000) < 0.01)

    fixed_engine = FakeEngine()
    long_node.execute(fixed_engine, "第一句。第二句。", "参考音色 TTS", max_chunk_seconds=0.5,
                      reference_seconds=4.0, continuity="同一参考", trim_trailing_silence=False, input_audio=ref)
    fixed_calls = fixed_engine.inference.generate_calls
    check("LongSpeech 同一参考复用", len(fixed_calls) == 2
          and fixed_calls[1][1]["audio"][0] is fixed_calls[0][1]["audio"][0])

    desc_engine = FakeEngine()
    _, report2 = long_node.execute(
        desc_engine, "你好。世界。", "声音描述 TTS", voice_description="温柔的年轻女声",
        max_chunk_seconds=0.5, trim_trailing_silence=False)
    desc_calls = desc_engine.inference.generate_calls
    parsed2 = json.loads(report2)
    check("LongSpeech 描述模式首段无参考", desc_calls[0][1]["audio"] is None
          and parsed2["segments"][0]["reference"] == "none")
    check("LongSpeech 描述模板", "温柔的年轻女声" in desc_calls[0][0][0]["content"][0]["text"])
    check("LongSpeech 描述模式次段滚动参考", desc_calls[1][1]["audio"] is not None
          and parsed2["segments"][1]["reference"] == "rolling")

    rate_engine = FakeEngine()
    _, rate_report = long_node.execute(
        rate_engine, "第一句。第二句。", "参考音色 TTS", max_chunk_seconds=5.0,
        speech_rate=5.0, reference_seconds=4.0, trim_trailing_silence=False, input_audio=ref)
    rated = json.loads(rate_report)
    seg = rated["segments"][0]
    check("LongSpeech 语速写入报告", rated["speech_rate"] == 5.0)
    check("LongSpeech 语速影响估计", abs(seg["estimated_seconds"] - seg["raw_estimated_seconds"] * (4.1511 / 5.0)) < 0.1)
    check("LongSpeech 目标 = 估计 + 头寸", abs(seg["applied_seconds"] - (seg["estimated_seconds"] + 0.15)) < 0.02)

    default_engine = FakeEngine()
    _, default_report = long_node.execute(
        default_engine, "第一句。第二句。", "参考音色 TTS", max_chunk_seconds=5.0,
        reference_seconds=4.0, trim_trailing_silence=False, input_audio=ref)
    default_seg = json.loads(default_report)["segments"][0]
    check("LongSpeech 默认语速=原始估计", abs(default_seg["estimated_seconds"] - default_seg["raw_estimated_seconds"]) < 0.02)
    check("LongSpeech 语速越大目标越短", seg["applied_seconds"] < default_seg["applied_seconds"])

    flash_long = FakeEngine(is_flash=True)
    long_node.execute(flash_long, "你好。", "参考音色 TTS", max_chunk_seconds=1.0,
                      trim_trailing_silence=False, input_audio=ref)
    flash_kwargs = flash_long.inference.generate_calls[-1][1]
    check("LongSpeech Flash 自动锁定",
          (flash_kwargs["nfe"], flash_kwargs["cfg_strength"], flash_kwargs["sway_sampling_coef"]) == (4, 0.0, -1.0))

    trim_calls = []
    real_trim = LS._trim_trailing_silence
    LS._trim_trailing_silence = lambda waveform, sample_rate: trim_calls.append(sample_rate) or waveform
    long_node.execute(FakeEngine(), "你好。", "参考音色 TTS", max_chunk_seconds=1.0,
                      trim_trailing_silence=True, input_audio=ref)
    check("LongSpeech 裁尾静音接线", trim_calls == [24000])
    LS._trim_trailing_silence = real_trim
finally:
    LS._fade_edges = real_fade


# ── SF AuK Long Speech：长音频处理模式 ──
check("Process 单块上限：等长", abs(LS._process_chunk_limit("等长", 1.0, 24.0) - 14.7) < 0.01)
check("Process 单块上限：变速 2x", abs(LS._process_chunk_limit("变速", 2.0, 24.0) - 19.7) < 0.01)
check("Process 单块上限：变速 0.5x", abs(LS._process_chunk_limit("变速", 0.5, 24.0) - 9.7) < 0.01)
check("Process 单块上限：用户上限优先", LS._process_chunk_limit("等长", 1.0, 8.0) == 8.0)

process_node = LS.SFAuKLongSpeech()
check("Process 缺音频拒绝", raises(process_node.execute, FakeEngine(), "", "长音频处理（编辑/增强）",
      instruction="请只去除背景噪声，保留其他内容，输出等长结果。"))
check("Process 缺指令拒绝", raises(process_node.execute, FakeEngine(), "", "长音频处理（编辑/增强）",
      input_audio={"waveform": FakeTensor((1, 1, 24000)), "sample_rate": 24000}))
check("Process 未知时长模式拒绝", raises(process_node.execute, FakeEngine(), "", "长音频处理（编辑/增强）",
      input_audio={"waveform": FakeTensor((1, 1, 24000)), "sample_rate": 24000},
      instruction="x", duration_mode="未知"))

real_split = LS._split_source_chunks
real_fade2 = LS._fade_edges
real_trim2 = LS._trim_trailing_silence
trim_seen = []


def _fake_split(waveform, sample_rate, max_seconds):
    step = 24000 * 10
    return [(waveform[..., i:i + step], i, i + step) for i in range(0, waveform.shape[-1], step)]


LS._split_source_chunks = _fake_split
LS._fade_edges = lambda waveform, sample_rate, fade_ms=5.0, fade_in=True, fade_out=True: waveform
LS._trim_trailing_silence = lambda waveform, sample_rate: trim_seen.append(1) or waveform
try:
    long_source = {"waveform": FakeTensor((1, 1, 24000 * 30)), "sample_rate": 24000}
    process_engine = FakeEngine()
    audio_p, report_p = process_node.execute(
        process_engine, "", "长音频处理（编辑/增强）", max_chunk_seconds=15.0,
        input_audio=long_source, instruction="请只去除背景噪声，保留其他内容，输出等长结果。",
        duration_mode="等长", trim_trailing_silence=True)
    parsed_p = json.loads(report_p)
    p_calls = process_engine.inference.generate_calls
    check("Process 按静音点分块（桩 3 块）", parsed_p["chunks"] == 3 and len(p_calls) == 3)
    check("Process 指令套用到每块", all(call[0][0]["content"][0]["text"] == "请只去除背景噪声，保留其他内容，输出等长结果。" for call in p_calls))
    check("Process 等长目标时长", [round(call[1]["gen_seconds"], 2) for call in p_calls] == [10.0, 10.0, 10.0])
    check("Process 每块以自身为源", all(call[1]["audio"] is not None for call in p_calls))
    check("Process 忽略裁尾静音", trim_seen == [])
    check("Process 分段区间报告", [(s["start_seconds"], s["end_seconds"]) for s in parsed_p["segments"]]
          == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)])
    check("Process 拼接不插静音", audio_p["waveform"].shape == (1, 1, 3000))
    check("Process 进度条总格", FakeProgressBar.instances[-1].total == 3 * (32 + 3))

    speed_engine = FakeEngine()
    _, report_s = process_node.execute(
        speed_engine, "", "长音频处理（编辑/增强）", max_chunk_seconds=15.0,
        input_audio=long_source, instruction="将语速调整为2倍。",
        duration_mode="变速", speed_multiplier=2.0)
    parsed_s = json.loads(report_s)
    check("Process 变速目标时长", [round(call[1]["gen_seconds"], 2) for call in speed_engine.inference.generate_calls]
          == [5.0, 5.0, 5.0])
    check("Process 变速报告", parsed_s["duration_mode"] == "变速" and parsed_s["speed_multiplier"] == 2.0)
finally:
    LS._split_source_chunks = real_split
    LS._fade_edges = real_fade2
    LS._trim_trailing_silence = real_trim2


# ── SF AuK Long Speech：语音内容编辑模式 ──
plan = LS._edit_plan("替换", "旧词", "新词", "")
check("Edit 替换指令/定位/增删", plan == ("把‘旧词’改成‘新词’", "旧词", "新词", "旧词"))
check("Edit 前插指令/定位", LS._edit_plan("前插", "", "嗯", "你好") == ("在‘你好’前面加上‘嗯’", "你好", "嗯", ""))
check("Edit 后插指令", LS._edit_plan("后插", "", "谢谢", "再见")[0] == "在‘再见’后面加上‘谢谢’")
check("Edit 删除指令", LS._edit_plan("删除", "嗯", "", "") == ("删掉‘嗯’", "嗯", "", "嗯"))
check("Edit 锚点前删除指令", LS._edit_plan("锚点前删除", "嗯", "", "你好")[0] == "删掉‘你好’前面的‘嗯’")
check("Edit 锚点后删除指令", LS._edit_plan("锚点后删除", "嗯", "", "你好")[0] == "删掉‘你好’后面的‘嗯’")
check("Edit 替换缺字段拒绝", raises(LS._edit_plan, "替换", "", "新词", ""))
check("Edit 前插缺锚点拒绝", raises(LS._edit_plan, "前插", "", "嗯", ""))
check("Edit 删除缺原词拒绝", raises(LS._edit_plan, "删除", "", "", ""))
check("Edit 未知操作拒绝", raises(LS._edit_plan, "未知", "a", "b", "c"))
check("Edit 匹配归一", LS._normalize_match_text("你好，世界！") == "你好世界"
      and LS._normalize_match_text("Hello, World.") == "helloworld")

edit_node = LS.SFAuKLongSpeech()
check("Edit 缺音频拒绝", raises(edit_node.execute, FakeEngine(), "", "语音内容编辑（替换/增添/删除）",
      edit_target="旧词", edit_new="新词"))
check("Edit 缺参数拒绝", raises(edit_node.execute, FakeEngine(), "", "语音内容编辑（替换/增添/删除）",
      input_audio={"waveform": FakeTensor((1, 1, 24000)), "sample_rate": 24000}))

real_split3 = LS._split_source_chunks
real_fade3 = LS._fade_edges
LS._split_source_chunks = _fake_split
LS._fade_edges = lambda waveform, sample_rate, fade_ms=5.0, fade_in=True, fade_out=True: waveform
try:
    edit_source = {"waveform": FakeTensor((1, 1, 24000 * 30)), "sample_rate": 24000}
    edit_engine = FakeEngine()
    FakeSenseVoice.texts = ["第一句内容", "这里是要替换的旧词", "第三句内容"]
    audio_e, report_e = edit_node.execute(
        edit_engine, "", "语音内容编辑（替换/增添/删除）", max_chunk_seconds=15.0,
        input_audio=edit_source, edit_operation="替换", edit_target="旧词", edit_new="崭新的词语")
    FakeSenseVoice.texts = None
    parsed_e = json.loads(report_e)
    e_calls = edit_engine.inference.generate_calls
    check("Edit 只改命中块", parsed_e["edited_chunks"] == [2] and len(e_calls) == 1)
    check("Edit 指令自动生成", e_calls[0][0][0]["content"][0]["text"] == "把‘旧词’改成‘崭新的词语’")
    check("Edit 命中块以自身为源", e_calls[0][1]["audio"] is not None)
    check("Edit 内容缩放目标时长", abs(parsed_e["segments"][1]["applied_seconds"] - 13.33) < 0.05)
    check("Edit 未命中块直通", parsed_e["segments"][0]["matched"] is False
          and parsed_e["segments"][2]["matched"] is False)
    check("Edit 拼接保留直通块全长", audio_e["waveform"].shape == (1, 1, 240000 * 2 + 1000))
    check("Edit 进度条含 ASR 阶段", FakeProgressBar.instances[-1].total == 3 * (32 + 4))
    check("Edit 报告含 ASR 文本", parsed_e["segments"][1]["asr"] == "这里是要替换的旧词")

    # 多块命中：全部编辑
    multi_engine = FakeEngine()
    FakeSenseVoice.texts = ["这里有旧词", "这里也有旧词", "没有"]
    _, report_m = edit_node.execute(
        multi_engine, "", "语音内容编辑（替换/增添/删除）", max_chunk_seconds=15.0,
        input_audio=edit_source, edit_operation="删除", edit_target="旧词")
    FakeSenseVoice.texts = None
    check("Edit 多块命中全部编辑", json.loads(report_m)["edited_chunks"] == [1, 2]
          and len(multi_engine.inference.generate_calls) == 2)

    # 未命中报错（带 ASR 预览）
    FakeSenseVoice.texts = ["甲", "乙", "丙"]
    check("Edit 未命中报错", raises(edit_node.execute, FakeEngine(), "", "语音内容编辑（替换/增添/删除）",
          max_chunk_seconds=15.0, input_audio=edit_source, edit_operation="删除", edit_target="不存在的词"))
    FakeSenseVoice.texts = None
finally:
    LS._split_source_chunks = real_split3
    LS._fade_edges = real_fade3
    FakeSenseVoice.texts = None


if failures:
    print(f"\n{len(failures)} 项失败：")
    for name in failures:
        print("  -", name)
    sys.exit(1)
print("\nOK")
