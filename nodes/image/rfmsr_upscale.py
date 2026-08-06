"""RFMSR 超分辨率节点。

RFMSR（Residual Flow Matching for Image Super-Resolution，arxiv:2607.12753）是
SD2.1 VAE 潜空间的流匹配超分模型（28 层 LightningDiT，约 1.9GB）。首次使用时会自动
下载权重到 {folder_paths.models_dir}/sfnodes/rfmsr/ckpts/（与官方 HF 仓库
frozen2001/RFMSR 的 ckpts/ 目录布局一致）：

  - 模型权重 rfmsr_os / rfmsr / rfmsr_consistency（各约 1.9GB，按 combo 选择懒下载）
  - SD2.1 VAE（diffusers 格式，约 335MB）
  - DINOv2 特征提取器（torch_cache：权重约 330MB + 官方源码，离线加载不访问网络）

推理流程与原版 infer_rfmsr.py 一致：BICUBIC 放大 → VAE 编码 → 反向流积分（可选
tiled 分块）→ VAE 解码 → 色彩校正。模型实例缓存在类级 _MODEL_CACHE，避免 ComfyUI
每次执行重复加载 2GB+ 权重。
"""

import gc
import os
from collections import OrderedDict
from pathlib import Path

import torch

import folder_paths
import comfy.model_management

from ...sf_utils.image_convert import tensor2pil, pil2tensor
from ...sf_utils.logger import get_logger
from ..model.rfmsr.inferencer import RFMSRInferencer

logger = get_logger(__name__)

# 官方 HF 权重仓库
RFMSR_REPO = "frozen2001/RFMSR"

# 模型存放根目录：models/sfnodes/rfmsr/（ckpts/ 子目录与官方 HF 仓库布局一致）。
# 注意：huggingface_hub 的 hf_hub_download/snapshot_download 在 local_dir 模式下
# 会保留仓库内相对路径（文件落盘为 <local_dir>/<filename>），因此所有下载统一以
# RFMSR_DIR 为 local_dir、filename 带 "ckpts/" 前缀，落盘位置即官方布局。
RFMSR_DIR = Path(folder_paths.models_dir) / "sfnodes" / "rfmsr"
RFMSR_CKPTS_DIR = RFMSR_DIR / "ckpts"

# combo 可选模型（文件名相对仓库根）
RFMSR_MODELS = {
    "rfmsr_os": {
        "file": "ckpts/rfmsr_os.safetensors",
        "desc": "单步模型（快，推荐）",
    },
    "rfmsr": {
        "file": "ckpts/rfmsr.safetensors",
        "desc": "多步流匹配模型（质量最高，慢）",
    },
    "rfmsr_consistency": {
        "file": "ckpts/rfmsr_consistency.safetensors",
        "desc": "一致性蒸馏单步模型",
    },
}

# SD2.1 VAE（diffusers 格式）
VAE_FILES = ("config.json", "diffusion_pytorch_model.safetensors")

# DINOv2 vitb14 权重（torch.hub 本地加载时从 <hub_dir>/checkpoints/ 直接读取）
DINOV2_CHECKPOINT = "ckpts/torch_cache/checkpoints/dinov2_vitb14_pretrain.pth"

# 类级模型缓存：LRU，最多保留 _MAX_CACHED_MODELS 套（(model_name, device) → RFMSRInferencer）。
# 超出容量时弹出最久未用的并 unload，保证任何时刻最多驻留一套 ~2.9GB，切换模型不累积。
# 刻意不干预 ComfyUI 自身模型管理（不主动卸载/腾挪其模型），混跑冲突由用户自行安排。
_MAX_CACHED_MODELS = 1
_MODEL_CACHE = OrderedDict()

_CATEGORY = "sfnodes/image"


def unload_rfmsr_models():
    """卸载全部已缓存的 RFMSR 模型并清空显存（供手动清理节点调用）。"""
    while _MODEL_CACHE:
        _, inferencer = _MODEL_CACHE.popitem(last=False)
        try:
            inferencer.unload()
        except Exception:
            pass
    comfy.model_management.soft_empty_cache()
    gc.collect()
    logger.info("RFMSR 模型已全部卸载")


def _hf_download(filename, local_dir, legacy_dir=None):
    """下载 HF 仓库单文件到本地目录（兼容新旧 huggingface_hub 签名）。

    local_dir 语义为 <local_dir>/<filename>（filename 为仓库内完整路径）。
    legacy_dir 提供早期版本下载时拼接出的错误落盘根目录：若该位置已有完整文件
    （旧版本已下载的 1.9GB 权重等），直接迁移到正确位置，避免重复下载。
    """
    if legacy_dir is not None:
        legacy = Path(legacy_dir) / filename
        target = Path(local_dir) / filename
        if legacy.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(legacy, target)
            logger.info(f"迁移旧版下载布局权重: {legacy} → {target}")

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=RFMSR_REPO, filename=filename, local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    except TypeError:  # huggingface_hub>=1.0 移除 local_dir_use_symlinks 参数
        return hf_hub_download(repo_id=RFMSR_REPO, filename=filename, local_dir=str(local_dir))


def _hf_snapshot(allow_patterns, local_dir):
    """下载 HF 仓库部分文件（兼容新旧签名）。"""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=RFMSR_REPO, allow_patterns=allow_patterns,
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
            local_dir=str(local_dir), local_dir_use_symlinks=False,
        )
    except TypeError:
        return snapshot_download(
            repo_id=RFMSR_REPO, allow_patterns=allow_patterns,
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
            local_dir=str(local_dir),
        )


def _ensure_rfmsr_weight(model_name):
    """按需下载所选 RFMSR 权重，返回本地路径。"""
    info = RFMSR_MODELS[model_name]
    path = _hf_download(
        info["file"], RFMSR_DIR, legacy_dir=RFMSR_CKPTS_DIR
    )
    logger.info(f"RFMSR 权重就绪: {path}")
    return path


def _ensure_vae():
    """下载 SD2.1 VAE（diffusers 目录结构），返回目录。"""
    for f in VAE_FILES:
        _hf_download(
            f"ckpts/stable-diffusion-2-1-base/vae/{f}",
            RFMSR_DIR,
            legacy_dir=RFMSR_CKPTS_DIR / "stable-diffusion-2-1-base",
        )
    vae_dir = RFMSR_CKPTS_DIR / "stable-diffusion-2-1-base"
    logger.info(f"SD2.1 VAE 就绪: {vae_dir}")
    return vae_dir


def _ensure_dinov2():
    """下载 DINOv2 权重与官方源码到本地 torch_cache，返回 torch.hub 目录。"""
    torch_cache = RFMSR_CKPTS_DIR / "torch_cache"
    _hf_download(
        DINOV2_CHECKPOINT, RFMSR_DIR,
        legacy_dir=RFMSR_CKPTS_DIR / "torch_cache",
    )
    # snapshot 保留仓库内相对路径（ckpts/... 前缀）
    _hf_snapshot(
        ["ckpts/torch_cache/facebookresearch_dinov2_main/**"],
        RFMSR_DIR,
    )
    logger.info(f"DINOv2 torch_cache 就绪: {torch_cache}")
    return torch_cache


def _get_inferencer(model_name):
    """获取（或加载并缓存）模型推理器实例。LRU 单槽：切换模型自动卸载旧模型。"""
    device = comfy.model_management.get_torch_device()
    key = (model_name, str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        _MODEL_CACHE.move_to_end(key)
        return cached

    # LRU：超容量先卸载最久未用的模型
    while len(_MODEL_CACHE) >= _MAX_CACHED_MODELS:
        _, old = _MODEL_CACHE.popitem(last=False)
        try:
            old.unload()
        except Exception:
            pass

    comfy.model_management.soft_empty_cache()
    logger.info(
        f"首次使用 RFMSR 模型 {model_name}，检查/下载权重"
        f"（模型 1.9GB + VAE 335MB + DINOv2 330MB，后续不再下载）..."
    )
    rfmsr_path = _ensure_rfmsr_weight(model_name)
    vae_dir = _ensure_vae()
    torch_cache_dir = _ensure_dinov2()

    inferencer = RFMSRInferencer()
    inferencer.load(
        rfmsr_path=rfmsr_path,
        vae_path=str(vae_dir),
        torch_cache_dir=str(torch_cache_dir),
        device=device,
    )
    _MODEL_CACHE[key] = inferencer
    return inferencer


class SFRFMSRUpscale:
    """RFMSR 高清化节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像，支持 batch 逐张处理"}),
                "model": (
                    list(RFMSR_MODELS.keys()),
                    {
                        "default": "rfmsr_os",
                        "tooltip": (
                            "模型权重（首次使用自动下载，每个约 1.9GB）："
                            "rfmsr_os = 单步快速（推荐）；rfmsr = 多步最高质量"
                            "（steps 建议 8~15）；rfmsr_consistency = 一致性单步"
                        ),
                    },
                ),
                "scale": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.5,
                     "tooltip": "放大倍数，输出尺寸 = 输入 × scale"},
                ),
                "steps": (
                    "INT",
                    {"default": 1, "min": 1, "max": 50,
                     "tooltip": "反向积分步数：单步模型用 1；多步模型（rfmsr）建议 8~15"},
                ),
                "flow_sigma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "流噪声标准差（官方默认 1.0）"},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**64, "tooltip": "随机种子，控制初始噪声"},
                ),
                "chopping": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "大图分块推理，显存友好；非方形图自动启用"},
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 1024, "step": 128,
                     "tooltip": "分块像素尺寸"},
                ),
                "tile_stride": (
                    "INT",
                    {"default": 256, "min": 64, "max": 1024, "step": 64,
                     "tooltip": "分块滑动步长"},
                ),
                "color_correction": (
                    ["wavelet", "adain", "ycbcr", "none"],
                    {"default": "wavelet",
                     "tooltip": "输出色彩校正：wavelet = 小波低频替换（推荐）；"
                                "adain = 全局统计匹配；ycbcr = 色度替换；none = 不校正"},
                ),
                "force_offload": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "推理完成后立即卸载模型，释放显存（下次执行会重新加载）"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "使用 RFMSR（残差流匹配图像超分）将图片高清化放大；"
        "模型自动下载到 models/sfnodes/rfmsr/ckpts/"
    )

    def upscale(self, image, model, scale, steps, flow_sigma, seed,
                chopping, tile_size, tile_stride, color_correction, force_offload):
        inferencer = _get_inferencer(model)
        outputs = []
        for i in range(image.shape[0]):
            src = tensor2pil(image[i])
            result = inferencer.infer(
                src, scale=scale, steps=steps, flow_sigma=flow_sigma,
                seed=seed, chopping=chopping, tile_size=tile_size,
                tile_stride=tile_stride, color_correction=color_correction,
            )
            outputs.append(pil2tensor(result))
        # 参考 llama-cpp Instruct 的 force_offload 做法：执行完成后立即卸载模型，
        # 释放显存；下次执行时由 _get_inferencer 自动重新加载。
        if force_offload:
            unload_rfmsr_models()
        return (torch.cat(outputs, dim=0),)
