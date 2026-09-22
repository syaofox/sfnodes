"""Regional prompt runtime for SFRegionalLoRA (Krea2).

Pure planning lives in ``sf_utils/regional_engine.py`` (region JSON parsing,
token grid mask math, region-prompt attention mask); this module owns the one
torch/comfy-touching helper: encoding a region prompt with the Krea2 CLIP
(same tokenizer template the global prompt uses).

The node appends the returned contexts to the live model context at sampling
time (positive/cond rows keep them, uncond rows get them zeroed), so no
conditioning re-wiring is needed downstream.
"""

import torch


def encode_region_prompt(clip, text: str):
    """Encode one region prompt and return its Krea2 context tensor.

    Returns None when the tokenize/encode path yields no usable 3D tensor.
    Raises on encoder errors so the caller can report the failing region."""
    tokens = clip.tokenize(text)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if not conditioning:
        return None
    ctx = conditioning[0][0]
    if not torch.is_tensor(ctx) or ctx.dim() != 3:
        return None
    return ctx
