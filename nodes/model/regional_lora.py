"""SF Regional LoRA — multi-region character LoRA + prompt injection for Krea2.

Draw N boxes on the node, assign one LoRA and/or one prompt per box; each
LoRA's activation delta is injected (via forward hooks, never weight merging)
only into the image tokens whose mask lands inside its own box, and each
region's prompt tokens are appended to the text context and attention-masked
so only that region's image tokens can see them. Outside the box the LoRA
effect is exactly zero and the prompt is invisible — a region's identity
never bleeds into another region's. Works on fp8/quantized Krea2 checkpoints
(only activations are read/written).

Architecture (see sf_utils/regional_engine.py for the pure logic):
  - LoRA matrices: kohya (lora_down/up) + diffusers (lora_A/B), alpha/rank
    scale; keys normalized bidirectionally against model module names
    (Krea2: blocks.N.attn.wq/wk/wv/gate/wo, blocks.N.mlp.gate/up/down).
  - Token grid derived from the LIVE latent at first model call (VAE f8 +
    patch2 -> latent//2), sequence layout [text | image] (Krea2 concats
    context before img) so image token i sits at text_len + i.
  - Sparse engine: only tokens whose mask > sparse_threshold pay for the
    LoRA matmul; per-(region, seq) token indices are cached.
  - Region prompts: optional CLIP input; each region's text is encoded once
    (sf_utils/regional_prompt.py) and its embeddings appended to the live
    model context (cond rows keep them, uncond rows are zeroed). An
    attn1_patch (build_prompt_attn_mask) restricts every image token's text
    attention to its own region's columns; base text stays visible to all.
  - Per-region diagnostics: each region logs "matched m/M layers" — a region
    whose LoRA keys don't map onto the model (wrong architecture/format)
    reports 0 layers instead of silently doing nothing.

Outputs: the patched MODEL (feed KSampler), a rainbow mask preview, and an
info JSON with per-region match counts / prompt token counts.
"""

import json

import numpy as np
import torch
import safetensors.torch

import folder_paths

from ...sf_utils.logger import get_logger
from ...sf_utils.regional_engine import (
    normalize_key,
    parse_lora_sd,
    lora_scale,
    parse_regions,
    default_regions_json,
    collect_model_sigs,
    plan_layer_map,
    token_grid,
    rect_token_mask,
    normalize_overlap,
    active_token_indices,
    build_prompt_attn_mask,
    render_preview,
)
from ...sf_utils.regional_prompt import encode_region_prompt

try:
    import comfy.patcher_extension as _pext
    _WRAPPER_ENUM = _pext.WrappersMP.DIFFUSION_MODEL
except Exception:
    _pext = None
    _WRAPPER_ENUM = "diffusion_model"

_CATEGORY = "sfnodes/model"
WRAPPER_KEY = "sf_regional_lora"
_COMPUTE_DTYPE = torch.bfloat16
DEFAULT_REGIONS_JSON = default_regions_json(2)

logger = get_logger(__name__)


def _resolve_lora_path(name: str) -> str:
    try:
        p = folder_paths.get_full_path("loras", name)
        if p:
            return p
    except Exception:
        pass
    return name


def _iter_named_linears(module):
    """All modules a LoRA may target: Linear (incl. fp8-wrapped operations
    variants) or anything else carrying a weight attribute."""
    for name, sub in module.named_modules():
        if isinstance(sub, torch.nn.Linear) or hasattr(sub, "weight"):
            yield name, sub


def _diffusion_model_of(patcher):
    m = patcher.model
    return getattr(m, "diffusion_model", m)


def _materialize_delta_fn(entry, dev, cdt):
    """Compute fn(x_sel) -> delta for one LoRA entry, weights pre-moved once.
    kohya: down [rank, in], up [out, rank]; delta = (x @ down.T) @ up.T."""
    down_d = entry["down"].to(dev, cdt)
    up_d = entry["up"].to(dev, cdt) * entry["scale"]

    def fn(x_sel):
        return (x_sel @ down_d.t()) @ up_d.t()
    return fn


# ============================================================================
# the session: N regions, sparse hook, per-region diagnostics
# ============================================================================
class _RegionSession:
    def __init__(self, patcher, regions, boxes, seam_feather, sparse_threshold,
                 plan, prompt_tokens=None, prompt_blocks=None):
        self.patcher = patcher
        self.active = regions            # list of {'name','lora','mats',...}
        self.boxes = boxes               # list of normalized (x0,y0,x1,y1)
        self.seam_feather = float(seam_feather)
        self.sparse_threshold = max(0.0, float(sparse_threshold))
        self.plan = plan                 # {sig: set(region_idx)}
        self.prompt_tokens = prompt_tokens    # [1, total_tokens, dim] or None
        self.prompt_blocks = prompt_blocks or []  # [(region_idx, start, end)]
        self.n_img = 0
        self._layer_map = None           # name -> (module, {region_idx: fn})
        self._prepared = False
        self._masks = None               # list[np.ndarray] [n_img]
        self._masks_d = None             # list[torch.Tensor] device/dtype-ready
        self._active_cache = {}
        self._dev = None
        self._base_text_len = None       # live text prefix before the prompts
        self._text_len = None            # base + appended region prompt tokens
        self._cond_rows = None           # per batch row: True = cond (positive)
        self._mask = None                # cached bool attention mask
        self._mask_key = None
        self._extra = None               # cached broadcast prompt embeddings
        self._extra_key = None
        self._width_warned = False       # one-shot non-Krea2 context warning

    def prompt_active(self):
        return self.prompt_tokens is not None

    def _build_layer_map(self, dm, dev, cdt):
        sig_to_region_fns = {}
        for sig, region_idxs in self.plan.items():
            d = {}
            for ri in region_idxs:
                entry = self.active[ri]["mats"][sig]
                d[ri] = _materialize_delta_fn(entry, dev, cdt)
            sig_to_region_fns[sig] = d

        layer_map = {}
        for name, mod in _iter_named_linears(dm):
            sig = normalize_key(name)
            if sig in sig_to_region_fns:
                layer_map[name] = (mod, sig_to_region_fns[sig])
        return layer_map

    def _resolve_grid(self, x):
        """Rows/cols of image tokens from the live latent (Krea2: f8 + patch2
        -> (H//2, W//2)). Falls back to a 1024x1024 grid only if x is not a
        latent tensor (never happens in practice)."""
        if torch.is_tensor(x) and x.dim() >= 4:
            H, W = int(x.shape[-2]), int(x.shape[-1])
            rows, cols = H // 2, W // 2
            if rows > 0 and cols > 0:
                return rows, cols
        return token_grid(128, 128)

    def _prepare(self, dev, x):
        self._dev = dev
        self._layer_map = self._build_layer_map(self._diffusion_model(), dev, _COMPUTE_DTYPE)
        rows, cols = self._resolve_grid(x)
        self.n_img = rows * cols
        # 重叠区域按比例归一化：非重叠 token 不变（羽化保留），重叠 token
        # 各 region 按 mask 占比分配（总和 ≤ 1），避免双满幅叠加过强
        self._masks = normalize_overlap(
            [rect_token_mask(rows, cols, b, self.seam_feather)
             for b in self.boxes])
        self._masks_d = [torch.from_numpy(m.astype(np.float32)).to(dev, _COMPUTE_DTYPE)
                         for m in self._masks]
        self._active_cache = {}
        self._prepared = True
        logger.info("prepared | grid=%dx%d n_img=%d regions=%d sparse_threshold=%.3f",
                    rows, cols, self.n_img, len(self.active), self.sparse_threshold)

    def _diffusion_model(self):
        return _diffusion_model_of(self.patcher)

    def _context_layout(self, ctx, transformer_options):
        """Record the text prefix length and append region prompt tokens.

        Prompt tokens are appended to every batch row (the tensor must stay
        rectangular); rows that are not the positive pass (cond_or_uncond
        entry != 0) get them zeroed so the negative pass never sees region
        text. Returns the (possibly augmented) context."""
        if not torch.is_tensor(ctx) or ctx.dim() < 3:
            self._base_text_len = None
            self._text_len = None
            self._cond_rows = None
            self._mask = None
            return ctx
        self._base_text_len = int(ctx.shape[1])
        self._text_len = self._base_text_len
        if self.prompt_tokens is None:
            self._cond_rows = None
            return ctx
        if int(ctx.shape[-1]) != int(self.prompt_tokens.shape[-1]):
            # foreign architecture (region prompts were encoded for Krea2
            # text width) -- appending would corrupt the context
            if not self._width_warned:
                self._width_warned = True
                logger.warning("context width %d != region prompt width %d -- "
                               "region prompts skipped for this model.",
                               int(ctx.shape[-1]), int(self.prompt_tokens.shape[-1]))
            self._cond_rows = None
            return ctx
        batch = int(ctx.shape[0])
        cond = (transformer_options.get("cond_or_uncond")
                if isinstance(transformer_options, dict) else None)
        if isinstance(cond, (list, tuple)) and len(cond) == batch:
            rows = [int(c) == 0 for c in cond]
        else:
            rows = [True] * batch
        key = (str(ctx.device), str(ctx.dtype), batch, tuple(rows))
        extra = self._extra if (self._extra is not None
                                and self._extra_key == key) else None
        if extra is None:
            extra = self.prompt_tokens.to(ctx.device, ctx.dtype).expand(
                batch, -1, -1).clone()
            for b, is_cond in enumerate(rows):
                if not is_cond:
                    extra[b] = 0
            self._extra = extra
            self._extra_key = key
        self._cond_rows = rows
        self._text_len = self._base_text_len + int(self.prompt_tokens.shape[1])
        return torch.cat((ctx, extra), dim=1)

    def _get_attn_mask(self, batch, seq):
        """Cached bool [B, 1, S, S] mask restricting image tokens to their own
        region's prompt columns; None when prompts/masks are unavailable."""
        rows = self._cond_rows or [True] * batch
        if len(rows) != batch:
            rows = [True] * batch
        key = (batch, int(seq), tuple(rows), self._base_text_len,
               self._text_len, self._dev)
        if self._mask is not None and self._mask_key == key:
            return self._mask
        if self._masks is None or self._base_text_len is None or self.n_img <= 0:
            return None
        blocks = [(ri, self._base_text_len + int(s), self._base_text_len + int(e))
                  for (ri, s, e) in self.prompt_blocks]
        arr = build_prompt_attn_mask(rows, seq, self._text_len, self.n_img,
                                     self._masks, blocks, self.sparse_threshold)
        self._mask = torch.from_numpy(arr).to(self._dev or "cpu")
        self._mask_key = key
        return self._mask

    def attn_mask_patch(self, q, k, v, pe, attn_mask, extra_options):
        """attn1_patch: return the region-prompt attention mask. Empty dict
        (no patch) outside the main Krea2 blocks or without region prompts."""
        if not self.prompt_active():
            return {}
        if not isinstance(extra_options, dict) or extra_options.get("block_index") is None:
            return {}
        if not (torch.is_tensor(q) and torch.is_tensor(k)):
            return {}
        mask = self._get_attn_mask(int(q.shape[0]), int(k.shape[-2]))
        if mask is None:
            return {}
        if attn_mask is None:
            return {"attn_mask": mask}
        try:
            if attn_mask.dtype == torch.bool:
                return {"attn_mask": attn_mask & mask}
            # additive float bias: blocked positions get the dtype's min
            bias = torch.zeros_like(attn_mask).masked_fill(
                ~mask, torch.finfo(attn_mask.dtype).min)
            return {"attn_mask": attn_mask + bias}
        except Exception:
            return {"attn_mask": mask}

    def _active_tokens(self, region_idx, seq):
        key = (region_idx, int(seq), self._text_len)
        cached = self._active_cache.get(key)
        if cached is not None:
            return cached
        idx_np, weight_np = active_token_indices(
            self._masks[region_idx], self.sparse_threshold, seq, self.n_img,
            self._text_len)
        idx = torch.from_numpy(idx_np).to(self._dev)
        weight = torch.from_numpy(weight_np).to(self._dev, _COMPUTE_DTYPE)
        self._active_cache[key] = (idx, weight)
        return idx, weight

    def _make_hook(self, region_fns):
        # region_fns: {region_idx: compute_fn}
        def hook(module, inp, out):
            if not torch.is_tensor(out) or out.dim() < 2:
                return out
            x = inp[0]
            if not torch.is_tensor(x) or x.dim() < 2:
                return out
            seq = x.shape[-2]
            xf = x.to(_COMPUTE_DTYPE)
            res = None
            for region_idx, fn in region_fns.items():
                idx, weight = self._active_tokens(region_idx, seq)
                if idx.numel() == 0:
                    continue
                x_sel = torch.index_select(xf, dim=-2, index=idx)
                delta = fn(x_sel)
                delta = delta * weight.view(*([1] * (delta.dim() - 2)), -1, 1)
                if res is None:
                    res = torch.zeros_like(out, dtype=_COMPUTE_DTYPE)
                res.index_add_(dim=-2, index=idx, source=delta)
            if res is None:
                return out
            return out + res.to(out.dtype)
        return hook

    def run(self, executor, *args, **kwargs):
        dm = self._diffusion_model()
        if not self._prepared:
            if args and torch.is_tensor(args[0]):
                dev = args[0].device
            else:
                first = next(dm.parameters(), None)
                dev = first.device if first is not None else "cpu"
            self._prepare(dev, args[0] if args else None)
        transformer_options = kwargs.get("transformer_options")
        if not isinstance(transformer_options, dict):
            for a in args:
                if isinstance(a, dict) and "cond_or_uncond" in a:
                    transformer_options = a
                    break
        args = list(args)
        if len(args) > 2 and (torch.is_tensor(args[2]) or args[2] is None):
            args[2] = self._context_layout(args[2], transformer_options or {})
        elif "context" in kwargs:
            kwargs["context"] = self._context_layout(kwargs["context"],
                                                     transformer_options or {})
        if not self._layer_map:
            return executor(*args, **kwargs)
        handles = []
        try:
            for name, (mod, region_fns) in self._layer_map.items():
                handles.append(mod.register_forward_hook(self._make_hook(region_fns)))
            return executor(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()


# ============================================================================
# the node
# ============================================================================
class SFRegionalLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                    "tooltip": "Preview/mask 输出尺寸。实际 LoRA 掩码网格由 KSampler 的 latent 决定。"}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}),
                "base_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "全局强度系数，乘以每个区域的 strength。"}),
                "seam_feather": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "区域边缘羽化宽度（相对网格比例）。0=硬边界。"}),
                "sparse_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.2, "step": 0.005,
                    "tooltip": "低于此掩码值的 token 跳过 LoRA 计算。0=最安全/最慢。"}),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "接与主提示词相同的 Krea2 CLIP，用于逐区域编码提示词。"
                               "悬空时区域提示词被忽略（纯 LoRA 模式）。",
                }),
            },
            "hidden": {
                "SFRegionsJson": ("STRING", {"default": DEFAULT_REGIONS_JSON}),
            },
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "mask_preview", "info")
    FUNCTION = "apply"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("SF Regional LoRA：多区域角色 LoRA + 区域提示词注入（Krea2）。在节点画布上为每个"
                   "区域画框，分配 LoRA 和/或提示词：每个 LoRA 的激活增量只注入自己框内的图像 token，"
                   "每段区域提示词的文本 token 也只对框内图像可见——区域外效果精确为零，多角色互动"
                   "文生图。支持 kohya/diffusers 格式，fp8 量化模型安全。")

    def apply(self, model, canvas_width=1024, canvas_height=1024, base_strength=1.0,
              seam_feather=0.08, sparse_threshold=0.01, clip=None,
              SFRegionsJson=DEFAULT_REGIONS_JSON):
        regions = parse_regions(SFRegionsJson)
        active = []
        for r in regions:
            lora_on = (r["lora"] not in ("None", "")
                       and (r["strength"] * float(base_strength)) != 0.0)
            prompt_on = bool(str(r.get("prompt") or "").strip())
            if r["enable"] and (lora_on or prompt_on):
                r = dict(r)
                r["_lora_on"] = lora_on
                active.append(r)

        if not active:
            logger.warning("no active regions; passing model through unchanged.")
            blank = torch.zeros((1, 64, 64, 3))
            info = json.dumps({
                "n_regions": 0,
                "prompt_mode": "lora_only",
                "prompt_tokens": 0,
                "note": "no active regions (check enable / lora / prompt / strength)",
                "regions": [],
            }, indent=2, ensure_ascii=False)
            return (model, blank, info)

        # -- load LoRA matrices per active region (per-region failure =
        #    warning + skip / keep for prompt, never aborts the workflow) -----
        file_cache = {}
        prepared = []
        for r in active:
            mats = {}
            if r["_lora_on"]:
                path = _resolve_lora_path(r["lora"])
                if path not in file_cache:
                    try:
                        sd = safetensors.torch.load_file(path)
                        file_cache[path] = parse_lora_sd(sd)
                    except Exception as e:
                        logger.warning("could not load LoRA '%s' (%s).", r["lora"], e)
                        file_cache[path] = None
                mats = file_cache[path]
                if not mats:
                    if str(r["prompt"] or "").strip():
                        logger.warning("region '%s' kept for its prompt only "
                                       "(LoRA '%s' unavailable).", r["name"], r["lora"])
                        mats = {}
                    else:
                        logger.warning("region '%s' skipped (LoRA '%s' unavailable).",
                                       r["name"], r["lora"])
                        continue
            s = r["strength"] * float(base_strength)
            mats_scaled = {sig: {**d, "scale": lora_scale(d) * s}
                           for sig, d in mats.items()}
            prepared.append({"name": r["name"], "lora": r["lora"],
                             "prompt": r["prompt"],
                             "strength": r["strength"], "mats": mats_scaled,
                             "box": r["box"]})

        if not prepared:
            logger.warning("all region LoRAs failed to load; passing model through unchanged.")
            blank = torch.zeros((1, 64, 64, 3))
            info = json.dumps({
                "n_regions": 0,
                "prompt_mode": "lora_only",
                "prompt_tokens": 0,
                "note": "all region LoRAs failed to load (see console log)",
                "regions": [],
            }, indent=2, ensure_ascii=False)
            return (model, blank, info)

        boxes = [p["box"] for p in prepared]

        # -- region prompt encoding (per-region failure = prompt ignored) ------
        prompt_tokens = None
        prompt_blocks = []
        prompt_mode = "lora_only"
        prompted = [p for p in prepared if str(p["prompt"] or "").strip()]
        if clip is not None and prompted:
            parts = []
            cur = 0
            for i, p in enumerate(prepared):
                text = str(p["prompt"] or "").strip()
                if not text:
                    continue
                try:
                    ctx = encode_region_prompt(clip, text)
                except Exception as e:
                    logger.warning("region %d '%s': could not encode prompt (%s) "
                                   "-- prompt ignored for this region.", i, p["name"], e)
                    continue
                if ctx is None:
                    logger.warning("region %d '%s': prompt encoding produced no usable "
                                   "context -- prompt ignored for this region.", i, p["name"])
                    continue
                length = int(ctx.shape[1])
                parts.append(ctx)
                prompt_blocks.append((i, cur, cur + length))
                cur += length
            if parts:
                prompt_tokens = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                prompt_mode = "regional"
                logger.info("region prompts armed: %d region(s), %d text token(s)",
                            len(parts), cur)
        elif clip is None and prompted:
            logger.warning("region prompts are set but no CLIP is connected "
                           "-- prompts ignored (LoRA-only mode).")

        # -- layer planning + per-region diagnostics ---------------------------
        patched = model.clone()
        dm = _diffusion_model_of(patched)
        model_sigs = collect_model_sigs(dm.named_modules())
        plan, per_matched = plan_layer_map([p["mats"] for p in prepared], model_sigs)
        for i, p in enumerate(prepared):
            total = len(p["mats"])
            if total == 0:
                logger.info("region %d '%s': prompt-only (no LoRA).", i, p["name"])
                continue
            logger.info("region %d '%s' (%s): matched %d/%d layers",
                        i, p["name"], p["lora"], per_matched[i], total)
            if per_matched[i] == 0:
                logger.warning("region %d '%s': 0 layers matched the model -- this "
                               "LoRA will NOT take effect (wrong architecture or "
                               "key format for the loaded model).", i, p["name"])

        # 掩码挂载能力前置检查：没有 attn1 patch 通路就不追加提示词，
        # 否则区域提示词会全局可见（比不做隔离更糟）
        if prompt_tokens is not None and not hasattr(patched, "set_model_attn1_patch"):
            logger.warning("this ComfyUI build lacks set_model_attn1_patch "
                           "-- region prompts disabled.")
            prompt_tokens = None
            prompt_blocks = []
            prompt_mode = "lora_only"

        session = _RegionSession(patched, prepared, boxes, seam_feather,
                                 sparse_threshold, plan,
                                 prompt_tokens=prompt_tokens,
                                 prompt_blocks=prompt_blocks)

        def wrapper(executor, *args, **kwargs):
            return session.run(executor, *args, **kwargs)

        if hasattr(patched, "add_wrapper_with_key"):
            patched.add_wrapper_with_key(_WRAPPER_ENUM, WRAPPER_KEY, wrapper)
        elif hasattr(patched, "add_wrapper"):
            patched.add_wrapper(_WRAPPER_ENUM, wrapper)
        else:
            raise RuntimeError("This ComfyUI build lacks model wrapper support. Update ComfyUI.")

        if prompt_tokens is not None:
            patched.set_model_attn1_patch(session.attn_mask_patch)

        # -- rainbow mask preview + info --------------------------------------
        preview = render_preview(boxes, int(canvas_width), int(canvas_height))
        preview_t = torch.from_numpy(preview)

        info = json.dumps({
            "n_regions": len(prepared),
            "prompt_mode": prompt_mode,
            "prompt_tokens": int(prompt_tokens.shape[1]) if prompt_tokens is not None else 0,
            "grid": "derived from live latent at first model call (canvas size only affects preview)",
            "regions": [
                {"name": p["name"], "lora": p["lora"], "prompt": p["prompt"],
                 "strength": p["strength"], "enable": True,
                 "box": [round(v, 4) for v in p["box"]],
                 "layers_matched": per_matched[i],
                 "layers_total": len(p["mats"])}
                for i, p in enumerate(prepared)
            ],
        }, indent=2, ensure_ascii=False)

        logger.info("armed %d region(s)%s.", len(prepared),
                    " + region prompts" if prompt_tokens is not None else "")
        return (patched, preview_t, info)
