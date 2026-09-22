"""VOSR2 模型加载器节点。

bundle = VOSR 2.0 one-step 1.4B（LightningDiT + 匹配 Qwen-Image 2D VAE + DINOv2-L），
首次使用自动从 HF CSWRY/VOSR 下载缺失文件到 models/sfnodes/vosr2/VOSR2/；
已存在 TE-Speed-VOSR2 / ComfyUI-VOSR2 的 models/vosr2/VOSR2/ 时直接复用。

模型实例按 (bundle, dtype) 缓存在类级 LRU（避免每次执行重建 7GB 权重）。
"""

from collections import OrderedDict

import comfy.model_management

from ...sf_utils.logger import get_logger
from .vosr2.loader import KNOWN_MODEL, VOSR2LoadError, load_vosr2, model_options

logger = get_logger(__name__)

_CATEGORY = "sfnodes/model"

# 类级缓存：LRU 单槽，切换模型自动卸载旧 bundle
_MAX_CACHED_MODELS = 1
_BUNDLE_CACHE = OrderedDict()


def _get_bundle(model_name, dtype, auto_download):
    key = (model_name, dtype)
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        _BUNDLE_CACHE.move_to_end(key)
        return cached

    # LRU：超容量先把最久未用的 bundle 换出显存（保留对象，工作流中其它节点仍可用）
    while len(_BUNDLE_CACHE) >= _MAX_CACHED_MODELS:
        _, old = _BUNDLE_CACHE.popitem(last=False)
        try:
            old.offload()
        except Exception:
            pass
    comfy.model_management.soft_empty_cache()

    logger.info(f"加载 VOSR2 bundle: {model_name} (dtype={dtype})")
    bundle = load_vosr2(model_name, dtype=dtype, auto_download=auto_download)
    _BUNDLE_CACHE[key] = bundle
    return bundle


class SFVOSR2ModelLoader:
    """VOSR2 模型加载器。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    model_options(),
                    {
                        "default": KNOWN_MODEL,
                        "tooltip": (
                            "VOSR 2.0 bundle（DiT + 匹配 VAE + DINOv2-L 三件套，约 7GB）。"
                            "缺失时首次执行自动从 HF CSWRY/VOSR 下载；"
                            "已装 TE-Speed-VOSR2/ComfyUI-VOSR2 的 models/vosr2/ 布局会被复用"
                        ),
                    },
                ),
                "dtype": (
                    ["default", "fp16", "bf16"],
                    {
                        "default": "default",
                        "tooltip": "DiT 与 DINOv2 的计算精度；VAE 恒为 fp32（Qwen VAE 要求）",
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "缺失文件时自动下载（关闭则只做校验并报错）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VOSR2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = _CATEGORY
    DESCRIPTION = "加载 VOSR 2.0 超分模型（首次使用自动下载到 models/sfnodes/vosr2/）"

    def load(self, model, dtype, auto_download):
        try:
            bundle = _get_bundle(model, dtype, auto_download)
        except VOSR2LoadError as exc:
            raise RuntimeError(f"VOSR2 模型加载失败: {exc}") from exc
        logger.info(
            f"VOSR2 就绪: {model} / {dtype} / DiT 注意力后端 {bundle.attention_backend()}"
        )
        return (bundle,)
