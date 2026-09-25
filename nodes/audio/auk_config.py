"""SF AuK 提示词增强设置节点（复刻 ComfyUI-AuK_Doc 的 AuK OpenAI Settings / Llama.cpp Adapter，MIT）。

上游：DocWorkBox/ComfyUI-AuK_Doc（基于 Tencent-Hunyuan/AuK，MIT，见 auk/LICENSE）。
V3→V1 适配；llama.cpp 插件发现改用 sf_utils/llama_cpp.py 公共实现（原 __globals__
探测法与插件注册表等价，公共实现已被 SFQwenImage21PromptEnhancer 使用）。

两个节点只输出设置对象，自身不发送请求；联网发生在 SFAuKGenerateEdit 的提示词增强阶段。
"""

import math
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from urllib.parse import urlsplit

# 顶层包导入时 `...` 正常；测试以 `nodes.audio.auk_config` 顶层导入时 `...` 越界，
# 回退绝对导入（image_interrogator_api.py 同款可移植性兜底）。
try:
    from ...sf_utils.llama_cpp import find_llama_plugin
except Exception:  # pragma: no cover - 测试/移植性兜底
    from sf_utils.llama_cpp import find_llama_plugin  # type: ignore

_CATEGORY = "sfnodes/audio"
AUK_LLM_CONFIG = "SF_AUK_LLM_CONFIG"

_LOCK = threading.RLock()


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    timeout_sec: int

    def enhancer_kwargs(self):
        return dict(llm_api_key=self.api_key, llm_base_url=self.base_url,
                    llm_model=self.model, llm_temperature=self.temperature,
                    llm_top_p=self.top_p, llm_max_tokens=self.max_tokens,
                    llm_timeout=self.timeout_sec)


class SFAuKOpenAISettings:
    DESCRIPTION = (
        "AuK 提示词增强的 OpenAI 兼容 Chat Completions 设置（本节点自身不发送请求）；"
        "输出连接 SF AuK Generate / Edit 的 llm_config 后开启增强。base_url 通常以 /v1 结尾，"
        "本地免认证服务 Key 可留空；Key 会随工作流保存，分享前请清空"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "https://api.openai.com/v1",
                    "tooltip": "API 基地址，通常以 /v1 结尾；末尾的 /chat/completions 会自动去掉",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API Key；本地免认证服务可留空。Key 会随工作流保存，分享前请清空",
                }),
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "服务方提供的模型 ID",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "采样温度（官方提示词增强生产值 0）",
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "advanced": True,
                }),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 1, "max": 131072, "advanced": True,
                }),
                "timeout_sec": ("INT", {
                    "default": 120, "min": 1, "max": 3600, "advanced": True,
                }),
            },
        }

    RETURN_TYPES = (AUK_LLM_CONFIG,)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, base_url, api_key, model, temperature=0.0, top_p=1.0,
                max_tokens=4096, timeout_sec=120):
        base_url = base_url.strip().rstrip('/')
        if base_url.endswith('/chat/completions'):
            base_url = base_url[:-len('/chat/completions')]
        url = urlsplit(base_url)
        if url.scheme not in {'http', 'https'} or not url.netloc or url.query or url.fragment or url.username or url.password:
            raise ValueError('base_url must be an HTTP(S) API base URL without credentials, query or fragment.')
        model = model.strip()
        if not model:
            raise ValueError('Fill in the model ID supplied by your LLM service.')
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be between 0 and 2.')
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError('top_p must be greater than 0 and at most 1.')
        if not 1 <= max_tokens <= 131072 or not 1 <= timeout_sec <= 3600:
            raise ValueError('max_tokens or timeout_sec is outside the supported range.')
        key = api_key.strip()
        return (LLMSettings(base_url, key or 'not-required', model,
                            temperature, top_p, max_tokens, timeout_sec),)


def _resolve_storage():
    """llama-cpp_vlm 插件的 LLAMA_CPP_STORAGE（复用公共插件发现）。"""
    plugin = find_llama_plugin()
    storage = getattr(plugin, 'LLAMA_CPP_STORAGE', None) if plugin is not None else None
    if storage is None or not all(hasattr(storage, name) for name in
                                  ('load_model', 'clean', 'llm', 'current_config')):
        raise ValueError('需要安装并启用 llama-cpp_vllm 的 Llama-cpp Model Loader；当前未找到兼容加载器。')
    return storage


class _LocalClient:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, messages, max_tokens, temperature, **kwargs):
        from openai.types.chat import ChatCompletion

        from .auk.infer.pe import PromptEnhancerError
        try:
            with _LOCK:
                storage = _resolve_storage()
                if storage.llm is None or storage.current_config != self.config:
                    storage.load_model(deepcopy(self.config))
                if hasattr(storage, 'ensure_embedding_mode'):
                    storage.ensure_embedding_mode(False)
                request = dict(messages=deepcopy(messages), max_tokens=max_tokens,
                               temperature=temperature, stream=False)
                if 'top_p' in kwargs:
                    request['top_p'] = kwargs['top_p']
                result = storage.llm.create_chat_completion(**request)
                return ChatCompletion.model_validate(result)
        except Exception as exc:
            raise PromptEnhancerError(f'本地 llama.cpp 调用失败: {type(exc).__name__}: {exc}') from exc


@dataclass(frozen=True)
class LlamaSettings:
    model_config: dict
    temperature: float
    max_tokens: int
    unload_after_enhance: bool

    def enhancer_kwargs(self):
        return dict(llm_client=_LocalClient(self.model_config), llm_model='llama-cpp',
                    llm_base_url='local://llama-cpp', llm_temperature=self.temperature,
                    llm_max_tokens=self.max_tokens)

    def cleanup(self):
        if self.unload_after_enhance:
            with _LOCK:
                storage = _resolve_storage()
                if storage.current_config == self.model_config:
                    storage.clean()


class SFAuKLlamaCppSettings:
    DESCRIPTION = (
        "把 ComfyUI-llama-cpp_vlm 的 Llama-cpp Model Loader 接入 AuK 提示词增强"
        "（无需 API 地址或 Key，模型需支持指令与 JSON 输出）；输出连接 "
        "SF AuK Generate / Edit 的 llm_config。其他同名加载器不保证兼容"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL", {
                    "tooltip": "llama-cpp_vlm 的 Llama-cpp Model Loader 输出",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 1, "max": 131072,
                }),
                "unload_after_enhance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "增强结束后释放共享的 llama.cpp 模型（含失败路径），下次需要时重载",
                }),
            },
        }

    RETURN_TYPES = (AUK_LLM_CONFIG,)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, llama_model, temperature=0.0, max_tokens=4096, unload_after_enhance=True):
        if not isinstance(llama_model, dict) or not llama_model.get('model'):
            raise ValueError('请连接 llama-cpp_vllm 的 Llama-cpp Model Loader 输出。')
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be between 0 and 2.')
        if not 1 <= max_tokens <= 131072:
            raise ValueError('max_tokens must be between 1 and 131072.')
        return (LlamaSettings(deepcopy(llama_model), temperature,
                              max_tokens, unload_after_enhance),)
