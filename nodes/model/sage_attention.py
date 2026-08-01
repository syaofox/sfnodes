"""SageAttention 注意力补丁节点（SFPatchSageAttention）。

为 Krea2 / Krea2 Turbo 提供守卫式 SageAttention 加速：只对主扩散注意力路径
（head 数匹配、CUDA、fp16/bf16、无 mask、4D BHLD 输入）启用 SageAttention，
其余调用（如文本融合注意力、fp8 Sage 模式、非 CUDA/非 4D 输入）回退到
ComfyUI 原生 attention，避免 Triton/编译器错误、非法张量形状、NaN 或黑图。

GPL-3.0 代码来源：
  * 节点与守卫逻辑：https://github.com/SurrealByDesign/comfyui-krea2-sageattention-guard
    （对 ComfyUI-KJNodes nodes/model_optimization_nodes.py 的补丁，作者 SurrealByDesign, 2026）
  * get_sage_func 与节点结构：https://github.com/kijai/ComfyUI-KJNodes
"""

import logging

import torch

from comfy.ldm.modules.attention import wrap_attn

from ...sf_utils.logger import get_logger

logger = get_logger(__name__)

sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]

_CATEGORY = "sfnodes/model"


def get_sage_func(sage_attention, allow_compile=False):
    logger.info("Using sage attention mode: %s", sage_attention)
    from sageattention import sageattn
    if sage_attention == "auto":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    elif "sageattn3" in sage_attention:
        from sageattn3 import sageattn3_blackwell
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out

    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head),
                (q, k, v),
            )
            tensor_layout = "NHD"
        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = (
                    out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                )
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out
    return attention_sage


_krea2_sage_logged_reasons = set()
_krea2_sage_validated_shapes = set()


def _krea2_log_once(reason, message, level=logging.INFO):
    if reason in _krea2_sage_logged_reasons:
        return
    _krea2_sage_logged_reasons.add(reason)
    logger.log(level, message)


def _attention_arg(args, kwargs, index, name, default=None):
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _is_krea2_model(model):
    base_model = getattr(model, "model", model)
    try:
        import comfy.model_base
        if isinstance(base_model, comfy.model_base.Krea2):
            return True
    except Exception:
        pass

    model_config = getattr(base_model, "model_config", None)
    unet_config = getattr(model_config, "unet_config", {}) or {}
    if unet_config.get("image_model") == "krea2":
        return True

    return base_model.__class__.__name__ == "Krea2"


def _krea2_main_attention_heads(model):
    base_model = getattr(model, "model", model)
    model_config = getattr(base_model, "model_config", None)
    unet_config = getattr(model_config, "unet_config", {}) or {}
    return unet_config.get("heads", 48)


def _krea2_sage_mode_supported(sage_attention):
    # Krea2 is bf16/fp16-oriented. The fp8 Sage modes are fast, but too risky
    # here because failures tend to surface as black images or non-finite output.
    return "fp8" not in sage_attention


def _run_krea2_sage_dry_run(new_attention, sage_attention, allowed_heads):
    if not torch.cuda.is_available():
        logger.warning("Krea2 SageAttention dry-run skipped: CUDA is not available.")
        return False

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda")
    q = torch.randn((1, allowed_heads, 128, 128), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    try:
        out = new_attention.__wrapped__(q, k, v, allowed_heads, skip_reshape=True, transformer_options={})
    except Exception as exc:
        logger.warning("Krea2 SageAttention dry-run failed for %s: %s", sage_attention, exc)
        return False

    finite = torch.isfinite(out).all().item()
    # skip_reshape=True 时 attention_sage 走 HND 布局，输出 reshape 为 (b, tokens, heads*dim)
    expected_shape = (q.shape[0], q.shape[2], q.shape[1] * q.shape[3])
    if out.shape != expected_shape or out.dtype != dtype or not finite:
        logger.warning(
            "Krea2 SageAttention dry-run failed validation: "
            "shape=%s, expected_shape=%s, dtype=%s, finite=%s",
            tuple(out.shape), expected_shape, out.dtype, finite,
        )
        return False

    logger.info("Krea2 SageAttention dry-run passed for %s.", sage_attention)
    return True


def _make_krea2_sage_attention_override(new_attention, sage_attention, model, dry_run=False):
    allowed_heads = _krea2_main_attention_heads(model)
    logger.info(
        "Krea2 detected; using guarded SageAttention override. "
        "Only diffusion attention with heads=%s, CUDA, fp16/bf16, no mask will be patched.",
        allowed_heads,
    )

    if dry_run:
        _run_krea2_sage_dry_run(new_attention, sage_attention, allowed_heads)

    def attention_override_sage(func, *args, **kwargs):
        q = _attention_arg(args, kwargs, 0, "q")
        k = _attention_arg(args, kwargs, 1, "k")
        v = _attention_arg(args, kwargs, 2, "v")
        heads = _attention_arg(args, kwargs, 3, "heads")
        mask = _attention_arg(args, kwargs, 4, "mask")
        if mask is None:
            mask = kwargs.get("mask", None)
        skip_reshape = _attention_arg(args, kwargs, 6, "skip_reshape", False)
        skip_output_reshape = _attention_arg(args, kwargs, 7, "skip_output_reshape", False)

        def fallback(reason, message, level=logging.INFO):
            _krea2_log_once(reason, message, level)
            return func(*args, **kwargs)

        if not _krea2_sage_mode_supported(sage_attention):
            return fallback(
                f"{sage_attention}:unsupported-mode",
                f"Krea2: skipping SageAttention mode {sage_attention}; fp8 Sage modes are not allowlisted for Krea2.",
                logging.WARNING,
            )

        if not all(torch.is_tensor(t) for t in (q, k, v)):
            return fallback("not-tensors", "Krea2: skipping SageAttention because q/k/v are not tensors.")
        if not skip_reshape or q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            return fallback("shape", "Krea2: skipping SageAttention for non-BHLD attention input.")
        if q.device.type != "cuda":
            return fallback("device", "Krea2: skipping SageAttention because attention is not on CUDA.")
        if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
            return fallback(
                f"dtype:{q.dtype}:{k.dtype}:{v.dtype}",
                f"Krea2: skipping SageAttention for dtype q={q.dtype}, k={k.dtype}, v={v.dtype}.",
            )
        if mask is not None:
            return fallback("mask", "Krea2: skipping SageAttention because an attention mask is present.")
        if heads != allowed_heads or q.shape[1] != allowed_heads:
            return fallback(
                f"heads:{heads}:{q.shape[1]}",
                f"Krea2: skipping SageAttention for heads={heads}, q_heads={q.shape[1]}; likely text-fusion attention.",
            )
        if k.shape[1] != q.shape[1] or v.shape[1] != q.shape[1]:
            return fallback("gqa", "Krea2: skipping SageAttention because q/k/v head counts differ.")

        try:
            out = new_attention.__wrapped__(*args, **kwargs)
        except Exception as exc:
            return fallback(
                f"exception:{type(exc).__name__}",
                f"Krea2: SageAttention failed ({exc}); falling back to original attention.",
                logging.WARNING,
            )

        shape_key = (tuple(q.shape), q.dtype, sage_attention)
        if shape_key not in _krea2_sage_validated_shapes:
            finite = torch.isfinite(out).all().item()
            if skip_output_reshape:
                expected_shape = q.shape
            else:
                # 未跳过输出 reshape 时，attention_sage 将 HND 输出 reshape 为 (b, tokens, heads*dim)
                expected_shape = (q.shape[0], q.shape[2], q.shape[1] * q.shape[3])
            if out.shape != expected_shape or not finite:
                return fallback(
                    f"invalid:{shape_key}",
                    f"Krea2: SageAttention produced invalid output shape={tuple(out.shape)}, expected_shape={expected_shape}, finite={finite}; falling back.",
                    logging.WARNING,
                )
            _krea2_sage_validated_shapes.add(shape_key)
            logger.info("Krea2: SageAttention validated for shape=%s, dtype=%s.", tuple(q.shape), q.dtype)

        return out

    return attention_override_sage


def make_sage_attention_override(sage_attention, allow_compile=False, model=None, dry_run=False):
    new_attention = get_sage_func(sage_attention, allow_compile=allow_compile)

    if model is not None and _is_krea2_model(model):
        return _make_krea2_sage_attention_override(new_attention, sage_attention, model, dry_run=dry_run)

    def attention_override_sage(func, *args, **kwargs):
        return new_attention.__wrapped__(*args, **kwargs)

    return attention_override_sage


class SFPatchSageAttention:
    """全局将注意力替换为 SageAttention 的补丁节点（复刻 KJNodes 的 Patch Sage Attention KJ
    + Krea2 守卫补丁）。Krea2 模型启用守卫式路径，其余模型保持原有全局替换行为。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sage_attention": (sageattn_modes, {
                    "default": False,
                    "tooltip": "要使用的 SageAttention 模式。'disabled' = 恢复原生注意力。"
                               "注意：本节点不使用模型补丁系统，恢复默认注意力需要再次运行本节点"
                               "并选择 disabled",
                }),
            },
            "optional": {
                "allow_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "允许对 SageAttention 函数使用 torch.compile，需要 sageattn 2.2.0 或更高版本",
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用 Krea2 守卫前先在 CUDA 上执行一次小规模 SageAttention 验证"
                               "（仅对 Krea2 模型生效）",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("全局将注意力替换为 SageAttention 加速（实验性）。Krea2 模型启用守卫式路径："
                   "仅主扩散注意力（head 数匹配、CUDA、fp16/bf16、无 mask）使用 SageAttention，"
                   "其余回退原生注意力。恢复默认注意力需再次运行本节点并选择 disabled")

    def patch(self, model, sage_attention, allow_compile=False, dry_run=False):
        if sage_attention == "disabled":
            return (model,)

        model_clone = model.clone()
        # attention override
        model_clone.model_options["transformer_options"]["optimized_attention_override"] = make_sage_attention_override(
            sage_attention,
            allow_compile=allow_compile,
            model=model_clone,
            dry_run=dry_run,
        )

        return (model_clone,)
