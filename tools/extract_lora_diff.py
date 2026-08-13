#!/usr/bin/env python3
"""Extract a LoRA from the weight difference of two same-architecture diffusion models.

    python extract_lora_diff.py \
        --base      /path/to/base.safetensors \
        --finetuned /path/to/finetuned.safetensors \
        --output    out_lora.safetensors --rank 64

How it works
------------
For every quantized linear layer shared by both files (matched by exact key name):
    1. dequantize both to fp32 (int8 + per-row weight_scale; convrot un-rotation
       applied via comfy_kitchen when present, so the delta lives in the same
       weight space ComfyUI patches into for a LoRA-carrying layer)
    2. delta = W_finetuned - W_base
    3. randomized SVD truncated to rank r:  U, S, V = svd_lowrank(delta)
       up   = U * sqrt(S)   [out, r]
       down = sqrt(S) * V^T [r, in]
       alpha = r  ->  alpha/rank = 1, so LoRA strength 1.0 reproduces delta exactly
    4. saved as ComfyUI-native LoRA keys in the SD3.5/Flux kohya convention:
       lora_unet_<key with . -> _>.lora_up.weight / .lora_down.weight / .alpha

Caveats (by design)
-------------------
* **INT8-ONLY**: requires quantized linear layers (weight + weight_scale pairs).
  Plain fp16/bf16 checkpoints, fp8 quantized files, and conv (4D) weights are
  NOT supported — they error out or silently produce wrong results.
* int8 quantization noise of both files is baked into the delta; the SVD rank
  truncation itself filters the high-rank noise floor.
* Extracting a *turbo* finetune against the raw base captures the distillation
  delta, not a pure style delta. For style-only, diff against the turbo base
  instead.
* The script prints a per-layer relative-delta report; a delta below ~1% of the
  weight norm means the extracted signal is weak (mostly quant noise).

Runtime deps: torch, safetensors (comfy_kitchen optional, used for convrot un-rotation).
"""

import argparse
import logging
import sys

import torch
import safetensors.torch

log = logging.getLogger("extract_lora_diff")

try:
    from comfy_kitchen import tensor as ck_tensor
    _HAVE_CK = True
except ImportError:
    _HAVE_CK = False


def collect_layers(path):
    """Return {base_key: (weight_key, scale_key)} for quantized linear layers."""
    layers = {}
    with safetensors.safe_open(path, framework="pt") as sf:
        for k in sf.keys():
            if k == "__metadata__":
                continue
            if not k.endswith(".weight"):
                continue
            if k.endswith(".weight_scale"):
                continue
            base = k[: -len(".weight")]
            scale_key = base + ".weight_scale"
            if scale_key in sf.keys():
                layers[base] = (k, scale_key)
    return layers


def dequantize(path, weight_key, scale_key, device):
    """Dequantize an int8+scale weight pair to fp32 on `device`."""
    with safetensors.safe_open(path, framework="pt") as sf:
        w = sf.get_tensor(weight_key)
        scale = sf.get_tensor(scale_key)
    w = w.to(device)
    scale = scale.to(device)
    conf = _ck_config(path).get(weight_key[: -len(".weight")], {})
    if w.dtype == torch.int8 and _HAVE_CK and conf.get("convrot"):
        params = ck_tensor.TensorWiseINT8Layout.Params(
            scale=scale,
            orig_dtype=torch.float32,
            orig_shape=tuple(w.shape),
            is_weight=True,
            convrot=True,
            convrot_groupsize=conf.get("convrot_groupsize", 256),
        )
        return ck_tensor.TensorWiseINT8Layout.dequantize(w, params)
    if w.dtype == torch.int8:
        return w.to(torch.float32) * scale.to(torch.float32)
    return w.to(torch.float32)


_ck_config_cache = {}


def _ck_config(path):
    """Parse per-layer quant configs from comfy_quant markers or file metadata."""
    if path in _ck_config_cache:
        return _ck_config_cache[path]
    cfg = {}
    with safetensors.safe_open(path, framework="pt") as sf:
        meta = sf.metadata() or {}
        if "_quantization_metadata" in meta:
            import json
            try:
                qmeta = json.loads(meta["_quantization_metadata"])
                layers = qmeta.get("layers", {})
            except Exception:
                layers = {}
        else:
            layers = {}
        for k in sf.keys():
            if k.endswith(".comfy_quant"):
                base = k[: -len(".comfy_quant")]
                import json
                try:
                    cfg[base] = json.loads(bytes(sf.get_tensor(k).tolist()).decode())
                except Exception:
                    pass
        for base, conf in layers.items():
            if base not in cfg and isinstance(conf, dict):
                cfg[base] = conf
    _ck_config_cache[path] = cfg
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base model safetensors")
    ap.add_argument("--finetuned", required=True, help="finetuned model safetensors")
    ap.add_argument("--output", required=True, help="output LoRA safetensors path")
    ap.add_argument("--rank", type=int, default=64, help="SVD truncation rank (default 64)")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"], help="LoRA storage dtype")
    ap.add_argument("--min-rel-delta", type=float, default=0.005,
                    help="warn (not fail) if mean relative delta drops below this")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s rank=%d storage=%s", args.device, args.rank, args.dtype)
    if _HAVE_CK:
        log.info("comfy_kitchen available: convrot un-rotation enabled")
    else:
        log.warning("comfy_kitchen NOT found: convrot layers dequantized WITHOUT un-rotation")

    layers_b = collect_layers(args.base)
    layers_f = collect_layers(args.finetuned)
    common = sorted(set(layers_b) & set(layers_f))
    only_b = sorted(set(layers_b) - set(layers_f))
    only_f = sorted(set(layers_f) - set(layers_b))
    log.info("base layers=%d finetuned layers=%d common=%d", len(layers_b), len(layers_f), len(common))
    if only_b:
        log.info("only in base (%d): %s", len(only_b), only_b[:8])
    if only_f:
        log.info("only in finetuned (%d): %s", len(only_f), only_f[:8])
    if not common:
        log.error("no common layers found — cannot extract")
        sys.exit(1)

    store_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    out_sd = {}
    rel_deltas = []
    for i, base_key in enumerate(common):
        wkey, skey = layers_b[base_key]
        fwkey, fskey = layers_f[base_key]
        w_base = dequantize(args.base, wkey, skey, args.device)
        w_fine = dequantize(args.finetuned, fwkey, fskey, args.device)
        if w_base.shape != w_fine.shape:
            log.warning("shape mismatch %s: base=%s fine=%s — skipping", base_key, tuple(w_base.shape), tuple(w_fine.shape))
            del w_base, w_fine
            continue

        delta = (w_fine - w_base).to(torch.float32)
        w_base_norm = w_base.norm().item()
        del w_base, w_fine
        out_dim, in_dim = delta.shape
        rank = min(args.rank, out_dim, in_dim)

        if w_base_norm > 0:
            rel_deltas.append(delta.norm().item() / w_base_norm)

        U, S, V = torch.svd_lowrank(delta, q=rank)
        S = S.clamp_min(0.0)
        up = U * S.sqrt().unsqueeze(0)          # [out, r]
        down = S.sqrt().unsqueeze(-1) * V.transpose(0, 1)  # [r, in]
        del U, S, V, delta

        lora_base = "lora_unet_" + base_key.replace(".", "_")
        out_sd[f"{lora_base}.lora_up.weight"] = up.contiguous().to(store_dtype).to("cpu")
        out_sd[f"{lora_base}.lora_down.weight"] = down.contiguous().to(store_dtype).to("cpu")
        out_sd[f"{lora_base}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
        del up, down

        if (i + 1) % 28 == 0 or i == len(common) - 1:
            log.info("progress %d/%d (%s)", i + 1, len(common), base_key)

    if not out_sd:
        log.error("nothing extracted")
        sys.exit(1)

    mean_delta = sum(rel_deltas) / len(rel_deltas) if rel_deltas else 0.0
    log.info("extracted %d layers; mean relative delta ||dW||/||W_base|| = %.4f",
             len(rel_deltas), mean_delta)
    if mean_delta < args.min_rel_delta:
        log.warning("mean delta norm %.6f < %.4f — signal may be mostly int8 quant noise", mean_delta, args.min_rel_delta)

    safetensors.torch.save_file(out_sd, args.output)
    size_mb = sum(v.numel() * v.element_size() for v in out_sd.values()) / (1024 * 1024)
    log.info("saved %s (%.1f MB, %d tensors)", args.output, size_mb, len(out_sd))


if __name__ == "__main__":
    main()
