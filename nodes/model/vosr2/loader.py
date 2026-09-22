"""VOSR 2.0 bundle 的发现、严格校验、流式加载与运行时封装。

移植自 ylchen333/ComfyUI-VOSR2 的 loader.py（Apache-2.0，严格校验/流式加载/
ModelPatcher 三件套），sfnodes 改动：

- 目录约定：主目录 `models/sfnodes/vosr2/<bundle>/`，同时兼容 TE-Speed-VOSR2 /
  ComfyUI-VOSR2 的 `models/vosr2/<bundle>/`（已装旧包的用户零重复下载）。
- 首次使用时自动从 `CSWRY/VOSR` 下载缺失组件（可用 auto_download 关闭）。
- 运行时封装 `VOSR2ModelBundle` 增加 TE 同款加速项：memory_policy（auto /
  resident / staged，解码前释放 DiT/DINO 驻留）、torch.compile 包装（失败回退
  eager）、RoPE/pos-embed 缓存清理。

bundle 是不可分割的三件套（DiT + 匹配的 Qwen 2D VAE + DINOv2-L），布局：

    <bundle>/args.json
    <bundle>/checkpoints/ema_model.safetensors      (或 clean_weights/、bundle 根)
    <bundle>/Qwen-Image-vae-2d/{config.json,diffusion_pytorch_model.safetensors}
    <bundle>/dinov2_vitl14.safetensors              (或 dinov2_vitl14_pretrain.pth)
"""

import gc
import importlib.util
import json
import logging
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

from .dinov2 import build_dinov2_vitl14
from .lightningdit import LightningDiT, vosr_attention_backend
from .qwenimage_vae2d import AutoencoderKLQwenImage2D

logger = logging.getLogger(__name__)

# 主目录（sfnodes 约定）与旧包目录（TE-Speed-VOSR2 / ComfyUI-VOSR2）
BUNDLE_ROOT = Path(folder_paths.models_dir) / "sfnodes" / "vosr2"
LEGACY_BUNDLE_ROOT = Path(folder_paths.models_dir) / "vosr2"

HF_REPO_ID = "CSWRY/VOSR"
KNOWN_MODEL = "VOSR2"
_VAE_SUBDIR = "Qwen-Image-vae-2d"
_VAE_FILES = ("config.json", "diffusion_pytorch_model.safetensors")
_VISION_FILENAME = "dinov2_vitl14.safetensors"
_VISION_LEGACY_FILENAME = "dinov2_vitl14_pretrain.pth"

_DIT_HF_FILES = ("VOSR2/args.json", "VOSR2/checkpoints/ema_model.safetensors")
_DINOV2_HF_FILE = "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth"

# VOSR 2.0 (one-step 1.4B) 架构固定字段；不匹配直接报错而不是半加载
REQUIRED_ARGS = {
    "ae_type": "qwen",
    "dim": 1536,
    "depth": 36,
    "num_heads": 24,
    "patch_size": 2,
    "enc_type": "dinov2l",
    "enc_dim": 1024,
    "layer_dinov2b_list": [17],
    "auxiliary_time_cond": False,
    "distill_type": "onestep",
}

DIT_ARG_KEYS = ("mlp_ratio", "use_qknorm", "use_swiglu", "use_rope", "use_rmsnorm",
                "encdim_ratio", "resolution")

_TRAINING_ONLY_KEY_PATTERNS = (
    re.compile(r"^n_averaged$"),
    re.compile(r"^step_count$"),
    re.compile(r"^decay$"),
)
_STRIPPABLE_PREFIXES = ("module.", "_orig_mod.", "ema_model.")

# staged 策略下 VAE 解码的显存预留（1.4B DiT 释放后仍可能不够，预留 512MiB）
VAE_DECODE_RESERVE = 512 * 1024 * 1024


class VOSR2LoadError(RuntimeError):
    """缺失文件 / 配置不兼容 / state_dict 不匹配时抛出。"""


# ---------------------------------------------------------------- 目录发现

def _safe_child_dir(root: Path, name: str) -> Path:
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise VOSR2LoadError(f"Invalid VOSR2 bundle name: {name!r}")
    candidate = (root / name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise VOSR2LoadError(f"Invalid VOSR2 bundle name: {name!r}")
    return candidate


def _extra_model_roots():
    """从 folder_paths 已注册类别推导额外模型根目录（models-ext 等）。

    extra_model_paths.yaml 按类别声明（checkpoints/loras/...），没有 vosr2 类别，
    所以取所有已注册类别路径的父目录去重（内置 models/ 也会在其中）；内部字典不可用
    时退回 checkpoints 类别。下载目标仍是内置 models/sfnodes/vosr2（is_default 语义）。
    """
    paths = []
    try:
        for entry in folder_paths.folder_names_and_paths.values():
            candidate = entry[0] if isinstance(entry, (list, tuple)) else entry
            if isinstance(candidate, (list, tuple, set)):
                paths.extend(candidate)
            elif candidate is not None:
                paths.append(candidate)
    except Exception:
        paths = []
    if not paths:
        try:
            paths = folder_paths.get_folder_paths("checkpoints")
        except Exception:
            paths = []

    roots = []
    for p in paths:
        try:
            parent = Path(p).parent
        except Exception:
            continue
        if parent not in roots:
            roots.append(parent)
    return roots


def _bundle_roots():
    """候选 bundle 根目录：内置主目录 → 内置旧包目录 → 各额外模型根（sfnodes/vosr2、vosr2）。"""
    roots = [BUNDLE_ROOT, LEGACY_BUNDLE_ROOT]
    for model_root in _extra_model_roots():
        for candidate in (model_root / "sfnodes" / "vosr2", model_root / "vosr2"):
            if candidate not in roots:
                roots.append(candidate)
    return tuple(roots)


def list_model_bundles():
    """所有候选根目录下含 args.json 的 bundle 名（去重排序）。"""
    names = set()
    for root in _bundle_roots():
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if p.is_dir() and (p / "args.json").is_file():
                names.add(p.name)
    return sorted(names)


def model_options():
    """combo 选项：磁盘上已存在的 bundle + 恒定提供 KNOWN_MODEL（首次运行下载）。"""
    found = list_model_bundles()
    return found if KNOWN_MODEL in found else [KNOWN_MODEL, *found]


def resolve_bundle_dir(model_name: str):
    """返回 (bundle_dir, root)：优先 sfnodes 主目录，其次旧包目录，最后主目录（待下载）。"""
    for root in _bundle_roots():
        candidate = _safe_child_dir(root, model_name)
        if candidate.is_dir():
            return candidate, root
    return _safe_child_dir(BUNDLE_ROOT, model_name), BUNDLE_ROOT


# ---------------------------------------------------------------- 下载

def _hf_download(filename: str, root: Path):
    """下载 HF 仓库单文件到 root（保留仓库内相对路径），兼容新旧 huggingface_hub 签名。"""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=HF_REPO_ID, filename=filename, local_dir=str(root),
            local_dir_use_symlinks=False,
        )
    except TypeError:  # huggingface_hub>=1.0 移除 local_dir_use_symlinks
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=str(root))


def _convert_dinov2_pth_to_safetensors(src_pth: Path, dest: Path) -> None:
    state = comfy.utils.load_torch_file(str(src_pth), safe_load=True)
    if not isinstance(state, dict):
        raise VOSR2LoadError(
            f"Unexpected DINOv2 checkpoint format at {src_pth}: expected a flat state dict."
        )
    tensors = {k: v.contiguous() for k, v in state.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise VOSR2LoadError(f"DINOv2 checkpoint at {src_pth} contained no tensors.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    save_file(tensors, str(tmp))
    tmp.replace(dest)


def _find_dit_weight(bundle_dir: Path):
    for candidate in (
        bundle_dir / "clean_weights" / "ema_model.safetensors",
        bundle_dir / "checkpoints" / "ema_model.safetensors",
        bundle_dir / "ema_model.safetensors",
    ):
        if candidate.is_file():
            return candidate
    return None


def _find_dino_weight(bundle_dir: Path):
    """返回 (路径, 是否需转换为 safetensors)。"""
    safetensors_path = bundle_dir / _VISION_FILENAME
    if safetensors_path.is_file():
        return safetensors_path, False
    legacy = bundle_dir / _VISION_LEGACY_FILENAME
    if legacy.is_file():
        return legacy, True
    return None, False


def ensure_bundle_files(model_name: str, auto_download: bool = True) -> None:
    """补齐 bundle 缺失文件（仅 KNOWN_MODEL 支持自动下载；已存在则完全不联网）。"""
    bundle_dir, root = resolve_bundle_dir(model_name)
    vae_dir = bundle_dir / _VAE_SUBDIR

    dit_missing = _find_dit_weight(bundle_dir) is None or not (bundle_dir / "args.json").is_file()
    vae_missing = not all((vae_dir / name).is_file() for name in _VAE_FILES)
    dino_path, _ = _find_dino_weight(bundle_dir)
    vision_missing = dino_path is None

    if not (dit_missing or vae_missing or vision_missing):
        return
    if not auto_download:
        raise VOSR2LoadError(
            f"VOSR2 bundle {bundle_dir} 缺少文件（DiT/VAE/DINOv2），且 auto_download 已关闭。"
        )
    if model_name != KNOWN_MODEL:
        raise VOSR2LoadError(
            f"VOSR2 bundle not found: {model_name!r}（只有 {KNOWN_MODEL} 支持自动下载）"
        )

    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise VOSR2LoadError(
            "VOSR2 模型文件缺失且 huggingface_hub 不可用，无法自动下载。"
            "请手动放置权重，见节点说明的目录结构。"
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    if dit_missing:
        logger.info("[VOSR2] downloading DiT bundle from %s ...", HF_REPO_ID)
        for f in _DIT_HF_FILES:
            _hf_download(f, root)
    if vae_missing:
        logger.info("[VOSR2] downloading Qwen-Image 2D VAE from %s ...", HF_REPO_ID)
        vae_dir.mkdir(parents=True, exist_ok=True)
        for name in _VAE_FILES:
            _hf_download(f"{_VAE_SUBDIR}/{name}", root)
    if vision_missing:
        logger.info("[VOSR2] downloading + converting DINOv2-L encoder from %s ...", HF_REPO_ID)
        src = _hf_download(_DINOV2_HF_FILE, root)
        _convert_dinov2_pth_to_safetensors(Path(src), bundle_dir / _VISION_FILENAME)


# ---------------------------------------------------------------- 校验 / 加载

def _load_args_json(bundle_dir: Path) -> dict:
    args_path = bundle_dir / "args.json"
    if not args_path.is_file():
        raise VOSR2LoadError(f"VOSR2 model bundle at {bundle_dir} is missing args.json.")
    with open(args_path, "r") as f:
        args = json.load(f)

    for key, expected in REQUIRED_ARGS.items():
        actual = args.get(key)
        if actual != expected:
            raise VOSR2LoadError(
                f"VOSR2 model bundle at {bundle_dir} has an incompatible config: "
                f"expected {key}={expected!r}, got {actual!r}. 仅支持 VOSR 2.0 "
                f"one-step 1.4B checkpoint。"
            )
    missing = [k for k in DIT_ARG_KEYS if k not in args]
    if missing:
        raise VOSR2LoadError(
            f"VOSR2 model bundle at {bundle_dir}'s args.json is missing required field(s): {missing}."
        )
    return args


def _strip_key(key: str):
    stripped = key
    for prefix in _STRIPPABLE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if any(pattern.match(stripped) for pattern in _TRAINING_ONLY_KEY_PATTERNS):
        return None
    return stripped


def _load_state_dict_lean(module: torch.nn.Module, path: Path, dtype: torch.dtype) -> None:
    """流式读 safetensors 逐张转 dtype 后一次性 load_state_dict（assign=True）。

    必须走 nn.Module.load_state_dict：comfy.ops 层支持 aimdo 惰性初始化，构造时
    .weight/.bias 为 None，只有 load_state_dict 触发的 _load_from_state_dict 才会
    真正建 Parameter；手写遍历会静默丢权重。逐张读取（而非 load_file）避免
    checkpoint 以原始 dtype 再驻留一份，峰值 ≈ 最终 dtype 的模型。
    """
    sd: dict = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            stripped = _strip_key(key)
            if stripped is None:
                continue
            sd[stripped] = f.get_tensor(key).to(dtype)

    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise VOSR2LoadError(
            f"VOSR2 checkpoint at {path} does not match the expected {type(module).__name__} "
            f"architecture (missing={missing}, unexpected={unexpected})."
        )
    del sd
    gc.collect()


def _resolve_dtype(dtype: str, device) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "default":
        return comfy.model_management.unet_dtype(device=device)
    raise VOSR2LoadError(f"Unknown dtype option: {dtype!r}")


# ---------------------------------------------------------------- 运行时封装

class VOSR2ModelBundle:
    """DiT / VAE / DINOv2 三件套 + 显存与编译策略。

    只暴露窄接口（vision_features / encode / decode / denoise_one_step），
    调用方无需关心内部组件能力。
    """

    def __init__(self, dit_patcher, vae_patcher, dino_patcher, args: dict):
        self.dit_patcher = dit_patcher
        self.vae_patcher = vae_patcher
        self.dino_patcher = dino_patcher
        self.args = args
        self.vision_layer_index = args["layer_dinov2b_list"][0]
        self.dinov2_size = args.get("dinov2_size", 448)
        self.memory_policy = "auto"
        self._compiled_flex = None
        self._compile_requested = False
        self._compile_failed = False

    # ---- 加载 / 显存 ----

    def _load(self, patcher):
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)
        return patcher.model, patcher.load_device

    def set_memory_policy(self, policy: str):
        self.memory_policy = policy if policy in ("auto", "resident", "staged") else "auto"
        if self.memory_policy == "resident":
            # 常驻：立即把 DiT/DINO 载入（已载入时 load_models_gpu 直接命中）
            self.pin_resident()

    def _transformer_bytes(self) -> int:
        total = 0
        for patcher in (self.dit_patcher, self.dino_patcher):
            try:
                total += comfy.model_management.module_size(patcher.model)
            except Exception:
                pass
        return total

    def pin_resident(self):
        """resident 策略：DiT + DINO 一次性常驻（显存不够时 ComfyUI 仍会自行换出）。"""
        for patcher in (self.dit_patcher, self.dino_patcher):
            self._load(patcher)

    def release_transformers(self):
        """staged 策略：VAE 解码前释放 DiT/DINO 驻留，降低峰值显存。"""
        device = self.dit_patcher.load_device
        try:
            free_before = comfy.model_management.get_free_memory(device)
        except Exception:
            free_before = 0
        needed = self._transformer_bytes() + VAE_DECODE_RESERVE
        comfy.model_management.free_memory(needed, device)
        try:
            free_after = comfy.model_management.get_free_memory(device)
        except Exception:
            free_after = free_before
        logger.info(
            "[VOSR2] releasing DiT/DINO residency for VAE decode "
            "(free %.2f GiB, reserve %.2f GiB)",
            max(free_after - free_before, 0) / (1024 ** 3),
            VAE_DECODE_RESERVE / (1024 ** 3),
        )

    def prepare_vae_decode(self):
        """按策略在 VAE 解码前做显存准备。"""
        if self.memory_policy == "staged":
            self.release_transformers()

    def clear_staged(self):
        """显存紧张时的兜底：清空缓存分配器。"""
        comfy.model_management.soft_empty_cache()

    # ---- torch.compile ----

    def set_torch_compile(self, enabled: bool) -> bool:
        """安装/卸载 DiT 的 torch.compile 包装（幂等；运行时失败后不再自动重试）。"""
        enabled = bool(enabled)
        if not enabled:
            self._compile_requested = False
            self._compiled_flex = None
            self._compile_failed = False
            return False
        if self._compile_failed:
            return False
        if self._compile_requested and self._compiled_flex is not None:
            return True
        self._compile_requested = True
        self._compiled_flex = None
        if importlib.util.find_spec("triton") is None:
            logger.warning("[VOSR2] torch_compile requested but triton is not installed; skipping.")
            self._compile_failed = True
            return False
        try:
            self._setup_compile_cache_env()
            self._compiled_flex = torch.compile(self.dit_patcher.model.forward_flexible, dynamic=False)
            logger.info("[VOSR2] DiT torch.compile wrapper installed; first call will compile.")
            return True
        except Exception as exc:
            logger.warning("[VOSR2] torch.compile failed: %s", exc)
            self._compiled_flex = None
            self._compile_failed = True
            return False

    @staticmethod
    def _setup_compile_cache_env():
        try:
            cache_root = Path(folder_paths.get_temp_directory()) / "sfnodes_vosr2_compile"
            cache_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "inductor"))
            os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
        except Exception:
            pass

    @property
    def compile_active(self):
        return self._compiled_flex is not None

    # ---- 推理窄接口 ----

    def vision_features(self, lq_bchw_01: torch.Tensor) -> list:
        """lq_bchw_01: (B, 3, H, W) ∈ [0, 1] → [ (B, N, enc_dim) ]（固定取中间层）。"""
        model, device = self._load(self.dino_patcher)
        x = torch.nn.functional.interpolate(
            lq_bchw_01, size=(self.dinov2_size, self.dinov2_size), mode="bicubic"
        ).clamp(0.0, 1.0)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x.to(device) - mean) / std
        feats = model.forward_intermediate_layer(x.to(model.pos_embed.dtype), self.vision_layer_index)
        return [feats]

    def encode(self, x_bchw_neg1_1: torch.Tensor, tile_size: int = 0, tile_overlap: int = 0,
               amp: bool = False):
        from . import tiled_vae
        model, device = self._load(self.vae_patcher)
        with _vae_autocast(amp, device):
            return tiled_vae.encode_dispatch(model, x_bchw_neg1_1.to(device), tile_size, tile_overlap)

    def decode(self, latent: torch.Tensor, latents_mean, latents_std,
               tile_size: int = 0, tile_overlap: int = 0, amp: bool = False) -> torch.Tensor:
        from . import tiled_vae
        model, device = self._load(self.vae_patcher)
        with _vae_autocast(amp, device):
            return tiled_vae.decode_dispatch(
                model, latent.to(device), latents_mean, latents_std, tile_size, tile_overlap
            )

    def dit_velocity(self, inp: torch.Tensor, t_cur: float, t_next: float, venc_fea: list) -> torch.Tensor:
        """单次 DiT 前向：`inp = cat([lq_latent, z], dim=1)` 的流速度。"""
        model, device = self._load(self.dit_patcher)
        compute_dtype = model.t_embedder.mlp[0].weight.dtype
        inp = inp.to(device=device, dtype=compute_dtype)
        venc_fea = [f.to(device=device, dtype=compute_dtype) for f in venc_fea]
        b = inp.shape[0]
        t_cur_t = torch.full((b,), t_cur, device=device, dtype=compute_dtype)
        t_next_t = torch.full((b,), t_next, device=device, dtype=compute_dtype)
        if self._compiled_flex is not None:
            try:
                return self._compiled_flex(inp, t_cur_t, t_next_t, venc_fea)
            except Exception as exc:
                if not self._compile_failed:
                    logger.warning("[VOSR2] DiT torch.compile runtime failure; falling back to eager: %s", exc)
                self._compile_failed = True
                self._compiled_flex = None
        return model.forward_flexible(inp, t_cur_t, t_next_t, venc_fea)

    def denoise_one_step(self, lq_latent: torch.Tensor, noise: torch.Tensor, venc_fea: list) -> torch.Tensor:
        """一步 Euler flow-matching（t=1 → t=0）。"""
        device = self.dit_patcher.load_device
        z = noise.to(device)
        u = self.dit_velocity(torch.cat([lq_latent.to(device), z], dim=1), 1.0, 0.0, venc_fea)
        return z - u

    def attention_backend(self):
        return vosr_attention_backend()

    # ---- 释放 ----

    def offload(self):
        """把三件套换出显存但保留对象（下次使用自动重载）——force_offload / LRU 淘汰用。"""
        device = self.dit_patcher.load_device
        needed = self._transformer_bytes() + VAE_DECODE_RESERVE
        comfy.model_management.free_memory(needed, device)
        comfy.model_management.soft_empty_cache()
        gc.collect()


def _vae_autocast(enabled: bool, device):
    """VAE 编码/解码的 bf16 autocast（Qwen VAE 权重恒为 fp32；仅 CUDA 生效）。"""
    if not enabled:
        return _NullContext()
    try:
        if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    except Exception:
        pass
    return _NullContext()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------- 入口

def load_vosr2(model_name: str, dtype: str = "default", auto_download: bool = True,
               memory_policy: str = "auto", torch_compile: bool = False) -> VOSR2ModelBundle:
    ensure_bundle_files(model_name, auto_download=auto_download)

    bundle_dir, _root = resolve_bundle_dir(model_name)
    if not bundle_dir.is_dir():
        raise VOSR2LoadError(f"VOSR2 bundle not found: {model_name!r}")
    args = _load_args_json(bundle_dir)
    dit_weight_path = _find_dit_weight(bundle_dir)
    if dit_weight_path is None:
        raise VOSR2LoadError(
            f"No ema_model.safetensors found under {bundle_dir} "
            f"(looked in clean_weights/, checkpoints/, and the bundle root)."
        )

    vae_dir = bundle_dir / _VAE_SUBDIR
    if not (vae_dir / "config.json").is_file():
        raise VOSR2LoadError(
            f"VOSR2 bundle {bundle_dir} is missing its Qwen-Image 2D VAE "
            f"({vae_dir}/config.json). VAE 潜空间与 DiT 绑定，不可替换。"
        )

    vision_path, needs_convert = _find_dino_weight(bundle_dir)
    if vision_path is None:
        raise VOSR2LoadError(f"VOSR2 bundle {bundle_dir} is missing its DINOv2-L encoder.")
    if needs_convert:
        converted = bundle_dir / _VISION_FILENAME
        logger.info("[VOSR2] converting %s → %s", vision_path.name, converted.name)
        _convert_dinov2_pth_to_safetensors(vision_path, converted)
        vision_path = converted

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    compute_dtype = _resolve_dtype(dtype, load_device)

    # 节点 execute 跑在 torch.inference_mode() 下，此时构造模块会让 comfy.ops 的
    # 惰性层不注册 Parameter（named_parameters/load_state_dict 只见部分权重）。
    # 构造 + 加载必须在 inference_mode(False) 内完成；ModelPatcher 交付在外层。
    with torch.inference_mode(False):
        base_channels = 16
        dit = LightningDiT(
            input_size=args["resolution"] // 8,
            patch_size=args["patch_size"],
            in_channels=2 * base_channels,
            out_channels=base_channels,
            hidden_size=args["dim"],
            depth=args["depth"],
            num_heads=args["num_heads"],
            mlp_ratio=args["mlp_ratio"],
            z_dims=args["enc_dim"],
            encdim_ratio=args["encdim_ratio"],
            auxiliary_time_cond=args["auxiliary_time_cond"],
            use_qknorm=args["use_qknorm"],
            use_swiglu=args["use_swiglu"],
            use_rope=args["use_rope"],
            use_rmsnorm=args["use_rmsnorm"],
            num_fused_layers=len(args["layer_dinov2b_list"]),
        )
        _load_state_dict_lean(dit, dit_weight_path, compute_dtype)
        dit = dit.eval().to(compute_dtype)
        for p in dit.parameters():
            p.requires_grad_(False)

        vae = AutoencoderKLQwenImage2D.from_pretrained(str(vae_dir))
        vae = vae.eval().float()  # Qwen VAE 恒 fp32
        for p in vae.parameters():
            p.requires_grad_(False)

        vision_encoder = build_dinov2_vitl14()
        _load_state_dict_lean(vision_encoder, vision_path, compute_dtype)
        vision_encoder = vision_encoder.eval().to(compute_dtype)
        for p in vision_encoder.parameters():
            p.requires_grad_(False)

    dit_patcher = comfy.model_patcher.ModelPatcher(dit, load_device=load_device, offload_device=offload_device)
    vae_patcher = comfy.model_patcher.ModelPatcher(vae, load_device=load_device, offload_device=offload_device)
    dino_patcher = comfy.model_patcher.ModelPatcher(vision_encoder, load_device=load_device, offload_device=offload_device)

    bundle = VOSR2ModelBundle(dit_patcher, vae_patcher, dino_patcher, args)
    bundle.set_memory_policy(memory_policy)
    if torch_compile:
        bundle.set_torch_compile(True)
    return bundle
