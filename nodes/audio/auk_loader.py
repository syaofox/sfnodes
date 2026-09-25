"""SFAuKModelsLoader：AuK 模型加载器（复刻 ComfyUI-AuK_Doc AuK Models Loader，MIT）。

上游：DocWorkBox/ComfyUI-AuK_Doc（基于 Tencent-Hunyuan/AuK，MIT，见 auk/LICENSE）。
V3→V1 适配；引擎在 execute 内延迟导入（节点注册/启动阶段不加载 transformers）。
输出 SF_AUK_ENGINE 连接 SFAuKGenerateEdit；同一组参数命中弱引用缓存，不重复加载。

模型目录沿用上游约定（folder_paths 幂等追加，与 AuK_Doc 共存不冲突）：
  auk: models/auk、models/diffusion_models/auk
  auk_qwen: models/text_encoders、models/LLM
  auk_vae: models/auk/vae、models/vae/auk、models/vae
"""

import threading
import weakref
from pathlib import Path

import folder_paths
import torch

# 顶层包导入时 `...` 正常；测试以 `nodes.audio.auk_loader` 顶层导入时 `...` 越界，
# 回退绝对导入（image_interrogator_api.py 同款可移植性兜底）。
try:
    from ...sf_utils.logger import get_logger
except Exception:  # pragma: no cover - 测试/移植性兜底
    from sf_utils.logger import get_logger  # type: ignore

from .auk_paths import (
    checkpoint_choices,
    config_model_name,
    qwen_choices,
    resolve_choice,
    resolve_config,
    resolve_qwen_config,
    resolve_vae,
)

logger = get_logger(__name__)

_CATEGORY = "sfnodes/audio"
AUK_ENGINE = "SF_AUK_ENGINE"
MEMORY_MODES = ['low_vram', 'balanced', 'max_vram']

folder_paths.add_model_folder_path('auk', str(Path(folder_paths.models_dir) / 'auk'))
folder_paths.add_model_folder_path('auk', str(Path(folder_paths.models_dir) / 'diffusion_models' / 'auk'))
folder_paths.add_model_folder_path('auk_qwen', str(Path(folder_paths.models_dir) / 'text_encoders'))
folder_paths.add_model_folder_path('auk_qwen', str(Path(folder_paths.models_dir) / 'LLM'))
folder_paths.add_model_folder_path('auk_vae', str(Path(folder_paths.models_dir) / 'auk' / 'vae'))
folder_paths.add_model_folder_path('auk_vae', str(Path(folder_paths.models_dir) / 'vae' / 'auk'))
folder_paths.add_model_folder_path('auk_vae', str(Path(folder_paths.models_dir) / 'vae'))

_CACHE = weakref.WeakValueDictionary()
_LOCK = threading.Lock()


class AuKEngine:
    def __init__(self, inference):
        self.inference = inference
        self.lock = threading.Lock()


class SFAuKModelsLoader:
    DESCRIPTION = (
        "从 models/auk 选择 AuK Base/Flash（含 ComfyUI 格式 fp32/bf16/int8 repack，"
        "config.yaml 与 VAE 自动解析）与 Qwen2.5-Omni-3B 编码器；显存档位 low_vram/balanced "
        "在 CPU/GPU 间搬运模型，max_vram 让 DiT+VAE 常驻显存、int8 Qwen 以量化权重常驻"
        "（逐层反量化），宿主内存占用最低。输出 engine 连接 SF AuK Generate / Edit"
    )

    @classmethod
    def INPUT_TYPES(cls):
        valid_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['CUDA unavailable']
        return {
            "required": {
                "model_name": (checkpoint_choices(folder_paths.get_folder_paths('auk')) or ['No AuK models found'], {
                    "tooltip": "models/auk 下的 Base/Flash 权重。ComfyUI 格式 repack（文件名或 metadata 含变体信息）"
                               "会自动解析内置/外部 config.yaml 与 VAE",
                }),
                "qwen_name": (qwen_choices(folder_paths.get_folder_paths('auk_qwen')) or ['No Qwen2.5-Omni models found'], {
                    "tooltip": "Qwen2.5-Omni-3B 完整模型目录（含 config/tokenizer）或 int8 单文件（同目录需有 config.json）",
                }),
                "memory_mode": (MEMORY_MODES, {
                    "default": "low_vram",
                    "tooltip": "low_vram：Qwen + VAE 在 CPU、半精度 DiT 在 GPU；balanced：Qwen 与 DiT 轮流上 GPU、"
                               "VAE 在 CPU（需更多显存，8G 不推荐）；max_vram：DiT + VAE 常驻 GPU、int8 Qwen 以量化"
                               "权重常驻（逐层反量化），宿主内存最低，需 int8 Qwen 与足够空闲显存",
                }),
                "dtype": (["bf16", "fp16"], {
                    "default": "bf16",
                    "tooltip": "推理精度；GPU 不支持 bf16 时选 fp16",
                }),
                "device": (valid_devices, {
                    "tooltip": "推理所用 CUDA 设备（本加载器要求 CUDA）",
                }),
                "sequential_cfg": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "顺序执行 CFG 两分支：降激活显存、耗时更长；Flash 的 CFG 固定为 0，不受此开关影响",
                }),
            },
        }

    RETURN_TYPES = (AUK_ENGINE,)
    RETURN_NAMES = ("engine",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, model_name, qwen_name, memory_mode, dtype, device, sequential_cfg=True):
        if memory_mode not in set(MEMORY_MODES) or dtype not in {'bf16', 'fp16'}:
            raise ValueError('Select a supported memory profile and precision.')
        valid_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        if device not in valid_devices:
            raise ValueError('This loader requires a CUDA GPU. Use the original loader for CPU inference.')
        with torch.cuda.device(device):
            if dtype == 'bf16' and not torch.cuda.is_bf16_supported():
                raise ValueError('This GPU does not support bf16; select fp16.')
        checkpoint = resolve_choice(folder_paths.get_folder_paths('auk'), model_name)
        config = resolve_config(checkpoint, folder_paths.get_folder_paths('auk'))
        if config_model_name(config) not in {'AuK', 'AuK-Flash'}:
            raise ValueError('The selected configuration is not AuK or AuK-Flash.')
        vae = resolve_vae(checkpoint, config, folder_paths.get_folder_paths('auk_vae'))
        qwen = resolve_choice(folder_paths.get_folder_paths('auk_qwen'), qwen_name)
        qwen_config_dir = resolve_qwen_config(qwen, folder_paths.get_folder_paths('auk_qwen'))
        key = (str(checkpoint), str(config), str(vae), str(qwen), str(qwen_config_dir), memory_mode, dtype, device, sequential_cfg)
        with _LOCK:
            engine = _CACHE.get(key)
            if engine is None:
                from .auk.infer.infer_auk import AukInfer

                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache()
                engine = AuKEngine(AukInfer(config_path=str(config), ckpt_path=str(checkpoint),
                                          vae_path=str(vae), qwen_path=str(qwen),
                                          qwen_config_dir=str(qwen_config_dir), device=device, dtype=dtype,
                                          memory_mode=memory_mode, sequential_cfg=sequential_cfg))
                _CACHE[key] = engine
                logger.info(f'Loaded AuK engine: {checkpoint.name} / {qwen.name} / {memory_mode} / {dtype} / {device}')
        return (engine,)
