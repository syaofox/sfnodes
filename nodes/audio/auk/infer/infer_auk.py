from __future__ import annotations

import json
import logging
import math
import os

import torch
import torchaudio
from omegaconf import OmegaConf
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration

from ..model import CFMEdit, Flux2Edit
from ..model.vae import load_vae_model
from ..model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())


_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _dequantize_int8(qdata: torch.Tensor, scale: torch.Tensor, conf: dict, target_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a ComfyUI-format int8_tensorwise weight (optionally convrot) to target_dtype."""
    quant_format = str(conf.get("format", "int8_tensorwise"))
    if quant_format != "int8_tensorwise":
        raise ValueError(
            f"Unsupported checkpoint quantization format: {quant_format}. "
            "Only int8_tensorwise AuK repacks are supported by this loader."
        )
    try:
        from comfy_kitchen.tensor import TensorWiseINT8Layout
    except ImportError as error:
        raise RuntimeError(
            "Loading int8 AuK checkpoints requires comfy_kitchen (a ComfyUI build with cu130 support)."
        ) from error
    params = TensorWiseINT8Layout.Params(
        scale=scale.to(torch.float32),
        orig_dtype=target_dtype,
        orig_shape=tuple(qdata.shape),
        is_weight=True,
        convrot=bool(conf.get("convrot", False)),
        convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
    )
    try:
        return TensorWiseINT8Layout.dequantize(qdata, params)
    except RuntimeError:
        # Fall back to an fp32 dequantize when the kernel cannot emit the requested dtype.
        params = TensorWiseINT8Layout.Params(
            scale=scale.to(torch.float32),
            orig_dtype=torch.float32,
            orig_shape=tuple(qdata.shape),
            is_weight=True,
            convrot=bool(conf.get("convrot", False)),
            convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
        )
        return TensorWiseINT8Layout.dequantize(qdata, params).to(target_dtype)


def _iter_safetensors_weights(path: str, target_dtype: torch.dtype):
    """Stream (name, tensor) from a safetensors file, dequantizing int8_tensorwise weights."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        quantized = {key for key in keys if handle.get_slice(key).get_dtype() == "I8"}
        aux = {key + "_scale" for key in quantized} | {key for key in keys if key.endswith(".comfy_quant")}
        quant_conf = {}
        for key in keys:
            if key.endswith(".comfy_quant"):
                raw = handle.get_tensor(key).numpy().tobytes().decode("utf-8")
                quant_conf[key[: -len(".comfy_quant")]] = json.loads(raw)
        for key in keys:
            if key in aux:
                continue
            tensor = handle.get_tensor(key)
            if key in quantized:
                conf = quant_conf.get(key.rsplit(".", 1)[0], {})
                tensor = _dequantize_int8(tensor, handle.get_tensor(key + "_scale"), conf, target_dtype)
            yield key, tensor


def _model_nbytes(module: torch.nn.Module) -> int:
    total = 0
    for tensor in module.parameters():
        total += tensor.numel() * tensor.element_size()
    for tensor in module.buffers():
        total += tensor.numel() * tensor.element_size()
    return total


def _int8_dequant_pre_hook(module, args):
    """Materialize a lazily-quantized linear weight right before its forward."""
    conf = getattr(module, "_int8_quant_conf", {})
    module.weight.data = _dequantize_int8(module.weight_q, module.weight_scale, conf, module.weight.dtype)


def _int8_release_post_hook(module, args, output):
    """Release the materialized weight again; the int8 payload stays resident."""
    module.weight.data = torch.empty(0, dtype=module.weight.dtype, device=module.weight.device)


class AukInfer:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        *,
        device: str | None = None,
        dtype: str = "bf16",
        qwen_path: str | None = None,
        qwen_config_dir: str | None = None,
        vae_path: str | None = None,
        cpu_offload: bool = False,
        memory_mode: str = "standard",
        sequential_cfg: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)
        if memory_mode not in {"standard", "balanced", "low_vram", "max_vram"}:
            raise ValueError(f"Unknown memory mode: {memory_mode}")
        self.memory_mode = memory_mode
        self.gpu_resident = memory_mode == "max_vram"
        low_memory = memory_mode != "standard"
        if low_memory and (not self.device.startswith("cuda") or dtype == "fp32"):
            raise ValueError("Low-memory profiles require CUDA and bf16/fp16.")
        cpu_offload = cpu_offload or memory_mode in {"balanced", "low_vram"}
        self.cpu_offload = cpu_offload
        self.vae_device = self.device if (not low_memory or self.gpu_resident) else "cpu"
        if cpu_offload and (not self.device.startswith("cuda") or not torch.cuda.is_available()):
            raise ValueError("cpu_offload requires a CUDA device.")
        load_device = "cpu" if (cpu_offload or self.gpu_resident) else self.device

        config = OmegaConf.load(config_path)
        if qwen_path:
            config.model.text_encoder.text_encoder_path = qwen_path
        # the VAE ships next to the checkpoint as vae.safetensors; an explicit path wins
        ckpt_dir_vae = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "vae.safetensors")
        if vae_path:
            config.model.vae.vae_model_path = vae_path
        elif os.path.isfile(ckpt_dir_vae):
            config.model.vae.vae_model_path = ckpt_dir_vae

        self.config = config
        # AuK-Flash is a distilled release that only works under a fixed (t_grid, cfg); detect it
        self.is_flash = config.model.get("name", "") == "AuK-Flash"
        if self.is_flash:
            logger.info("Detected AuK-Flash release — locking sampling to the 4-step / CFG-off recipe.")

        vae_config = config.model.vae
        self.target_sample_rate = vae_config.target_sample_rate
        self.downsample_rate = vae_config.downsample_rate
        self.latent_dim = vae_config.latent_dim

        # --- text encoder (Qwen2.5-Omni) ---
        text_encoder_config = config.model.text_encoder

        qwen_source = str(text_encoder_config.text_encoder_path)
        if qwen_source.endswith(".safetensors") and os.path.isfile(qwen_source):
            qwen_config_dir = qwen_config_dir or os.path.dirname(os.path.abspath(qwen_source))
            logger.info(f"Loading Qwen text encoder weights from {qwen_source} (config from {qwen_config_dir}) ...")
            thinker = self._build_qwen_from_file(qwen_source, qwen_config_dir, lazy_int8=self.gpu_resident)
            processor_dir = qwen_config_dir
        else:
            logger.info(f"Loading Qwen text encoder from {qwen_source} ...")
            thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                qwen_source,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            # keep the full multimodal Thinker (text + ref_audio); drop the unused vision tower
            if thinker.visual is not None:
                del thinker.visual
                thinker.visual = None
            processor_dir = qwen_source
        text_encoder = thinker
        text_processor = Qwen2_5OmniProcessor.from_pretrained(processor_dir, local_files_only=True)

        # --- VAE model ---
        logger.info(f"Loading VAE from {vae_config.vae_model_path} ...")
        model_init_kwargs = OmegaConf.to_container(vae_config.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
        vae_model_config = BigVGANFlowVAEConfig.from_dict(model_init_kwargs)
        vae_model = load_vae_model(
            vae_name=vae_config.vae_name,
            vae_cfg=vae_model_config,
            vae_ckpt=vae_config.vae_model_path,
            map_location="cpu",
        )
        # Standard mode keeps VAE on the inference device; enhanced profiles
        # keep it on CPU for both encode and decode, without moving weight_norm modules.
        vae_model = vae_model.to(self.vae_device).eval()
        vae_model.requires_grad_(False)
        self.vae_model = vae_model

        # build CFMEdit (VAE-latent); Flux2Edit is the only supported backbone
        model_arc = OmegaConf.to_container(config.model.arch, resolve=True)
        model_arc["attn_backend"] = "torch"  # inference does not depend on flash_attn
        schedule_config = OmegaConf.to_container(config.model.get("schedule", OmegaConf.create({})), resolve=True)

        logger.info("Building CFMEdit model ...")
        model = CFMEdit(
            transformer=Flux2Edit(
                **model_arc,
                latent_dim=self.latent_dim,
            ),
            text_encoder=text_encoder,
            text_processor=text_processor,
            num_channels=self.latent_dim,
            **schedule_config,
        )
        if low_memory:
            # Cast the freshly built transformer to the runtime dtype before weights arrive so
            # the fp32 checkpoint streams straight into half precision. The previous order
            # (fp32 skeleton + full fp32 state dict) peaked near 30 GiB host RAM and forced
            # heavy swapping on 16 GiB machines.
            model.transformer.to(dtype=self.dtype)
        else:
            model = model.to(torch.float32)

        # --- load EMA weights (strip "ema_model." prefix; text_encoder.* comes from Qwen snapshot) ---
        self._load_ema_weights(model, ckpt_path)
        self.model = model.to(load_device)
        self.model.eval()
        self.model.low_memory = low_memory
        self.model.text_encoder_on_cpu = memory_mode == "low_vram"
        self.model.sequential_cfg = sequential_cfg
        if low_memory:
            if memory_mode == "balanced":
                self.model.text_encoder.to(dtype=self.dtype)
            # The frozen Qwen encoder keeps its bf16 checkpoint precision in low-memory
            # profiles: upcasting to fp32 would add ~9 GiB host RAM for the 3B encoder.
            # PyTorch supports the CPU bf16 operators the encoder needs.

        if cpu_offload:
            from accelerate import cpu_offload_with_hook
            from accelerate.utils import set_module_tensor_to_device

            # Keep only the small layer-fusion parameters on GPU, not the child models.
            for name, _ in self.model.named_parameters(recurse=False):
                set_module_tensor_to_device(self.model, name, self.device)
            if memory_mode == "low_vram":
                _, transformer_hook = cpu_offload_with_hook(self.model.transformer, self.device)
                self._offload_hooks = (transformer_hook,)
            else:
                _, text_hook = cpu_offload_with_hook(self.model.text_encoder, self.device)
                _, transformer_hook = cpu_offload_with_hook(self.model.transformer, self.device, prev_module_hook=text_hook)
                self._offload_hooks = (text_hook, transformer_hook)
        elif self.gpu_resident:
            self._place_gpu_resident(model)

    def _place_gpu_resident(self, model: CFMEdit):
        """max_vram: pin DiT + VAE on GPU and keep as much of the encoder resident as fits."""
        free_bytes, _ = torch.cuda.mem_get_info(torch.device(self.device))
        reserve = int(1.5 * 2 ** 30)  # activations + allocator headroom
        vae_bytes = _model_nbytes(self.vae_model)  # the VAE is already resident here
        dit_bytes = _model_nbytes(model.transformer)
        text_bytes = _model_nbytes(model.text_encoder)
        if dit_bytes + reserve > free_bytes:
            raise ValueError(
                f"max_vram needs about {(dit_bytes + reserve) / 2 ** 30:.1f} GiB free VRAM for the "
                f"DiT but only {free_bytes / 2 ** 30:.1f} GiB is available. Use balanced or low_vram."
            )
        # The small layer-fusion parameters live on the top-level module.
        for _, param in model.named_parameters(recurse=False):
            param.data = param.data.to(self.device)
        model.transformer.to(self.device)
        self.vae_model.to(self.device)
        if dit_bytes + text_bytes + reserve <= free_bytes:
            model.text_encoder.to(self.device)
            model.text_encoder_on_cpu = False
            logger.info(
                "max_vram: DiT, VAE and Qwen encoder resident on %s (%.1f GiB weights).",
                self.device, (vae_bytes + dit_bytes + text_bytes) / 2 ** 30,
            )
        else:
            model.text_encoder_on_cpu = True
            logger.info(
                "max_vram: DiT + VAE resident on %s (%.1f GiB); Qwen encoder stays on CPU "
                "(%.1f GiB does not fit).",
                self.device, (vae_bytes + dit_bytes) / 2 ** 30, text_bytes / 2 ** 30,
            )

    def _copy_weights(self, model: torch.nn.Module, weights):
        """Copy a streamed (name, tensor) sequence into the model; returns (missing, unexpected)."""
        model_keys = list(model.state_dict().keys())
        key_set = set(model_keys)
        seen = set()
        unexpected = []
        for name, tensor in weights:
            if name not in key_set:
                unexpected.append(name)
                continue
            model.load_state_dict({name: tensor}, strict=False)
            seen.add(name)
        missing = [k for k in model_keys if k not in seen]
        return missing, unexpected

    def _load_ema_weights(self, model: CFMEdit, ckpt_path: str):
        logger.info(f"Loading model checkpoint from {ckpt_path} ...")

        if ckpt_path.endswith(".safetensors"):
            # clean weights-only export (fp32/bf16 or ComfyUI int8 repack): stream one tensor
            # at a time so host memory stays near the model size.
            target_dtype = next(model.transformer.parameters()).dtype
            missing, unexpected = self._copy_weights(
                model, _iter_safetensors_weights(ckpt_path, target_dtype)
            )
        else:
            # training checkpoint: pull the EMA weights and strip the "ema_model." prefix.
            # mmap keeps the file-backed payload out of anonymous memory; popitem() drops
            # each source tensor as soon as it has been copied.
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
            except (TypeError, RuntimeError):
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ema = checkpoint["ema_model_state_dict"]

            def iter_ema_weights():
                while ema:
                    name, tensor = ema.popitem()
                    if name in ("initted", "step"):
                        continue
                    yield name.replace("ema_model.", ""), tensor

            missing, unexpected = self._copy_weights(model, iter_ema_weights())

        n_missing_te = sum(1 for k in missing if k.startswith("text_encoder."))
        n_missing_other = len(missing) - n_missing_te
        logger.info(
            f"Loaded EMA weights | missing={len(missing)} (text_encoder.*={n_missing_te}, other={n_missing_other}) "
            f"| unexpected={len(unexpected)}"
        )
        if n_missing_other:
            logger.warning(
                "Some non-text-encoder weights are missing; check config/arch matches the checkpoint. "
                f"Examples: {[k for k in missing if not k.startswith('text_encoder.')][:10]}"
            )
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected[:10]}")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _build_qwen_from_file(self, weights_path: str, config_dir: str, lazy_int8: bool = False):
        """Build the Qwen2.5-Omni Thinker from config and load single-file (quantized) weights.

        The meta-device build avoids instantiating 3B random weights; only the checkpoint
        payload is materialized. The unused vision tower is dropped before materialization
        because the ComfyUI repacks do not include its weights. With lazy_int8 the int8
        weights stay quantized and are dequantized per layer on the compute device, which
        keeps the encoder small enough to stay VRAM-resident.
        """
        from accelerate import init_empty_weights

        try:
            model_config = Qwen2_5OmniThinkerConfig.from_pretrained(config_dir, local_files_only=True)
        except OSError as error:
            raise ValueError(
                f"Qwen config/tokenizer directory is incomplete: {config_dir}. "
                "Point the loader at a full Qwen2.5-Omni-3B folder (config and tokenizer files)."
            ) from error
        with init_empty_weights(include_buffers=False):
            model = Qwen2_5OmniThinkerForConditionalGeneration(model_config)
        if getattr(model, "visual", None) is not None:
            del model.visual
            model.visual = None
        # Materialize meta parameters at bf16 and keep the small computed buffers (RoPE etc.).
        model._apply(
            lambda tensor: (
                torch.empty_like(tensor, device="cpu", dtype=torch.bfloat16)
                if tensor.is_floating_point()
                else torch.empty_like(tensor, device="cpu")
            )
            if tensor.is_meta
            else tensor
        )
        if lazy_int8:
            missing, unexpected = self._load_qwen_lazy_int8(model, weights_path)
        else:
            missing, unexpected = self._copy_weights(
                model, _iter_safetensors_weights(weights_path, torch.bfloat16)
            )
        logger.info(
            f"Loaded Qwen weights | missing={len(missing)} | unexpected={len(unexpected)}"
            + (" | lazy_int8=True" if lazy_int8 else "")
        )
        if missing:
            logger.warning(f"Missing Qwen weights (examples): {missing[:10]}")
        if unexpected:
            logger.warning(f"Unexpected Qwen weights (examples): {unexpected[:10]}")
        return model

    def _load_qwen_lazy_int8(self, model: torch.nn.Module, weights_path: str):
        """Load a ComfyUI int8 Qwen repack without materializing dequantized weights.

        Quantized linear weights are registered as non-persistent buffers on their owning
        module; forward hooks dequantize just before use and release right after, so the
        resident footprint stays at the int8 payload size.
        """
        from safetensors import safe_open

        model_keys = set(model.state_dict().keys())
        seen = set()
        unexpected = []
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            quantized = {key for key in keys if handle.get_slice(key).get_dtype() == "I8"}
            aux = {key + "_scale" for key in quantized} | {key for key in keys if key.endswith(".comfy_quant")}
            quant_conf = {}
            for key in keys:
                if key.endswith(".comfy_quant"):
                    raw = handle.get_tensor(key).numpy().tobytes().decode("utf-8")
                    quant_conf[key[: -len(".comfy_quant")]] = json.loads(raw)
            for key in keys:
                if key in aux:
                    continue
                if key not in model_keys:
                    unexpected.append(key)
                    continue
                tensor = handle.get_tensor(key)
                if key in quantized:
                    module_name = key[: -len(".weight")]
                    try:
                        module = model.get_submodule(module_name)
                    except AttributeError:
                        unexpected.append(key)
                        continue
                    module.weight.data = torch.empty(0, dtype=torch.bfloat16)
                    module.register_buffer("weight_q", tensor, persistent=False)
                    module.register_buffer("weight_scale", handle.get_tensor(key + "_scale"), persistent=False)
                    module._int8_quant_conf = quant_conf.get(module_name, {})
                    module.register_forward_pre_hook(_int8_dequant_pre_hook)
                    module.register_forward_hook(_int8_release_post_hook)
                else:
                    model.load_state_dict({key: tensor}, strict=False)
                seen.add(key)
        missing = [k for k in model_keys if k not in seen]
        return missing, unexpected

    # ------------------------------------------------------------------ helpers

    def _load_audio(self, source: str | tuple[torch.Tensor, int]) -> tuple[torch.Tensor, float]:
        if isinstance(source, str):
            audio, sr = torchaudio.load(source)
        else:
            audio, sr = source
            audio = audio.detach().to(device="cpu", dtype=torch.float32)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            if audio.ndim != 2:
                raise ValueError(f"Audio tensor must have shape [channels, samples], got {tuple(audio.shape)}.")
            if not isinstance(sr, int) or sr <= 0:
                raise ValueError(f"Audio sample rate must be a positive integer, got {sr!r}.")
            if audio.shape[-1] == 0:
                raise ValueError("Audio tensor is empty.")
            if not torch.isfinite(audio).all():
                raise ValueError("Audio tensor contains NaN or Inf.")
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        ref_rms = torch.sqrt(torch.mean(torch.square(audio)))
        if sr != self.target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
        return audio, float(ref_rms)

    @torch.inference_mode()
    def _run(
        self,
        ref_audio: torch.Tensor,  # [1, T] on cpu
        ref_rms: float | None,  # None => no reference audio, skip output RMS restore
        messages: list,  # single-sample chat messages (list of turns)
        gen_latent_len: int,
        *,
        nfe: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        t_grid: list[float] | None,
        seed: int | None,
    ) -> torch.Tensor:
        if ref_rms is None:
            ref_latent_lens_t = torch.zeros(1, dtype=torch.long, device=self.device)
            total_latent_lens_t = torch.tensor([gen_latent_len], dtype=torch.long, device=self.device)
            ref_latents = torch.zeros(1, 0, self.latent_dim, device=self.device, dtype=torch.float32)
        else:
            ref_audio = ref_audio.to(self.vae_device).unsqueeze(0)  # [1, 1, T]
            ref_latent_len = ref_audio.shape[-1] // self.downsample_rate
            total_latent_len = ref_latent_len + gen_latent_len

            ref_latent_lens_t = torch.tensor([ref_latent_len], dtype=torch.long, device=self.device)
            total_latent_lens_t = torch.tensor([total_latent_len], dtype=torch.long, device=self.device)
            audio_lens_t = ref_latent_lens_t * self.downsample_rate

            # --- online VAE encode + normalize ---
            ref_latents, enc_latent_lens = self.vae_model.encoding_and_normalization(
                ref_audio,
                sample_lengths=audio_lens_t.to(self.vae_device),
            )
            ref_latents = ref_latents.to(self.device)
            ref_latent_lens_t = torch.minimum(ref_latent_lens_t, enc_latent_lens.to(ref_latent_lens_t.device))

        # --- CFM sample in latent space ---
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.startswith("cuda")):
            cond_inputs = self.model.build_cond_inputs([messages], self.model.text_processor)
            generated, _ = self.model.sample(
                cond=ref_latents,
                text=cond_inputs,
                duration=total_latent_lens_t,
                lens=ref_latent_lens_t,
                steps=nfe,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                t_grid=t_grid,
                no_ref_audio=False,
                seed=seed,
            )  # [1, T_total, D]

        gen = generated[0]
        rl = ref_latent_lens_t[0].item()
        tl = total_latent_lens_t[0].item()
        gen_latent = gen[rl:tl, :].unsqueeze(0)  # [1, T_new, D]
        if gen_latent.shape[1] == 0:
            raise RuntimeError("Empty generated latent (target duration collapsed to 0).")
        if torch.isnan(gen_latent).any() or torch.isinf(gen_latent).any():
            raise RuntimeError("Generated latent contains NaN/Inf.")

        if self.cpu_offload:
            self._offload_hooks[-1].offload()
        gen_latent = gen_latent.to(device=self.vae_device, dtype=torch.float32)
        gen_latent = self.vae_model.denormalize(gen_latent)
        gen_latent = gen_latent.permute(0, 2, 1)  # [1, D, T_new]

        gen_audio = self.vae_model.inference_from_latents(gen_latent).cpu()
        if gen_audio.ndim == 3:
            gen_audio = gen_audio.squeeze(0)  # [1, T_wav]
        if torch.isnan(gen_audio).any() or torch.isinf(gen_audio).any():
            raise RuntimeError("Generated audio contains NaN/Inf.")

        return gen_audio.to(torch.float32)

    # ------------------------------------------------------------------ public API

    def generate(
        self,
        messages: list,  # caller-composed ChatML turns (must carry a user audio item)
        *,
        audio: str | tuple[torch.Tensor, int] | None = None,
        gen_seconds: float | None = None,
        nfe: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        t_grid: list[float] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        wav_path = audio or extract_audio_path(messages, required=False)
        if wav_path is not None:
            ref_audio, ref_rms = self._load_audio(wav_path)
        else:
            # no reference audio (text-only instruct TTS): empty reference, ref_rms=None
            ref_audio = torch.zeros(1, 0)
            ref_rms = None
            for m in messages:
                if m.get("role") != "user":
                    continue
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text" and not c["text"].endswith("|<no_prompt_audio>|"):
                        c["text"] = c["text"] + "|<no_prompt_audio>|"
        ref_latent_len = ref_audio.shape[-1] // self.downsample_rate  # 0 when no reference

        if gen_seconds is not None:
            gen_latent_len = max(1, int(math.ceil(gen_seconds * self.target_sample_rate / self.downsample_rate)))
        else:
            # default: regenerate a segment as long as the source clip
            gen_latent_len = max(1, ref_latent_len)

        # AuK-Flash: ignore any caller-supplied sampling knobs and pin the distilled recipe —
        # 4-step time_grid (from the checkpoint metadata), CFG off. The DMD student bakes in its
        # own guidance, so re-adding CFG blows up the amplitude (clips hard). Base AuK is unrestricted.
        if self.is_flash:
            nfe = 4
            cfg_strength = 0.0
            sway_sampling_coef = None
            t_grid = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]

        if self.memory_mode != "standard":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            audio_out = self._run(
                ref_audio,
                ref_rms,
                messages,
                gen_latent_len,
                nfe=nfe,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                t_grid=t_grid,
                seed=seed,
            )
        finally:
            if self.cpu_offload:
                self.model.transformer.clear_cache()
                for hook in self._offload_hooks:
                    hook.offload()
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
        if self.memory_mode != "standard":
            logger.info("AuK %s peak PyTorch VRAM: %.2f GiB allocated / %.2f GiB reserved",
                        self.memory_mode, torch.cuda.max_memory_allocated(self.device) / 2**30,
                        torch.cuda.max_memory_reserved(self.device) / 2**30)
        return audio_out, self.target_sample_rate


def extract_audio_path(messages: list, *, required: bool = True) -> str | None:
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "audio":
                path = c.get("audio") or c.get("audio_url")
                if path:
                    return path
    if required:
        raise ValueError("generate() needs a user audio item (type=audio) in messages to VAE-encode.")
    return None


