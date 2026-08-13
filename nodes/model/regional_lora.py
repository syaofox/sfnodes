"""SF Regional LoRA — multi-region character LoRA injection for Krea2.

Draw N boxes on the node, assign one LoRA per box; each LoRA's activation
delta is injected (via forward hooks, never weight merging) only into the
image tokens whose mask lands inside its own box. Outside the box the effect
is exactly zero — a region's identity never bleeds into another region's.
Works on fp8/quantized Krea2 checkpoints (only activations are read/written).

Architecture (see sf_utils/regional_engine.py for the pure logic):
  - LoRA matrices: kohya (lora_down/up) + diffusers (lora_A/B), alpha/rank
    scale; keys normalized bidirectionally against model module names
    (Krea2: blocks.N.attn.wq/wk/wv/gate/wo, blocks.N.mlp.gate/up/down).
  - Token grid derived from the LIVE latent at first model call (VAE f8 +
    patch2 -> latent//2), sequence layout [text | image] (Krea2 concats
    context before img) so image tokens occupy the tail: offset = seq - n_img.
  - Sparse engine: only tokens whose mask > sparse_threshold pay for the
    LoRA matmul; per-(region, seq) token indices are cached.
  - Per-region diagnostics: each region logs "matched m/M layers" — a region
    whose LoRA keys don't map onto the model (wrong architecture/format)
    reports 0 layers instead of silently doing nothing.

Outputs: the patched MODEL (feed KSampler), a rainbow mask preview, and an
info JSON with per-region match counts.
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
    render_preview,
)

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
                 plan):
        self.patcher = patcher
        self.active = regions            # list of {'name','lora','mats',...}
        self.boxes = boxes               # list of normalized (x0,y0,x1,y1)
        self.seam_feather = float(seam_feather)
        self.sparse_threshold = max(0.0, float(sparse_threshold))
        self.plan = plan                 # {sig: set(region_idx)}
        self.n_img = 0
        self._layer_map = None           # name -> (module, {region_idx: fn})
        self._prepared = False
        self._masks = None               # list[np.ndarray] [n_img]
        self._masks_d = None             # list[torch.Tensor] device/dtype-ready
        self._active_cache = {}
        self._dev = None

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

    def _active_tokens(self, region_idx, seq):
        key = (region_idx, int(seq))
        cached = self._active_cache.get(key)
        if cached is not None:
            return cached
        idx_np, weight_np = active_token_indices(
            self._masks[region_idx], self.sparse_threshold, seq, self.n_img)
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
            "optional": {},
            "hidden": {
                "SFRegionsJson": ("STRING", {"default": DEFAULT_REGIONS_JSON}),
            },
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "mask_preview", "info")
    FUNCTION = "apply"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("SF Regional LoRA：多区域角色 LoRA 注入（Krea2）。在节点画布上为每个区域画框并分配 "
                   "一个 LoRA，每个 LoRA 的激活增量只注入自己框内的图像 token——区域外效果精确为零，"
                   "多角色（LoRA）互动文生图。支持 kohya/diffusers 格式，fp8 量化模型安全。")

    def apply(self, model, canvas_width=1024, canvas_height=1024, base_strength=1.0,
              seam_feather=0.08, sparse_threshold=0.01, SFRegionsJson=DEFAULT_REGIONS_JSON):
        regions = parse_regions(SFRegionsJson)
        active = [r for r in regions
                  if r["enable"] and r["lora"] not in ("None", "")
                  and (r["strength"] * base_strength) != 0.0]

        if not active:
            logger.warning("no active regions; passing model through unchanged.")
            blank = torch.zeros((1, 64, 64, 3))
            info = json.dumps({
                "n_regions": 0,
                "note": "no active regions (check enable / lora / strength)",
                "regions": [],
            }, indent=2, ensure_ascii=False)
            return (model, blank, info)

        # -- load LoRA matrices per active region (per-region failure =
        #    warning + skip, never aborts the workflow) ----------------------
        file_cache = {}
        prepared = []
        for r in active:
            path = _resolve_lora_path(r["lora"])
            if path not in file_cache:
                try:
                    sd = safetensors.torch.load_file(path)
                except Exception as e:
                    logger.warning("could not load LoRA '%s' (%s) -- region '%s' skipped.",
                                   r["lora"], e, r["name"])
                    continue
                file_cache[path] = parse_lora_sd(sd)
            mats = file_cache[path]
            if not mats:
                logger.warning("'%s' contains no recognized LoRA/LoKr-style keys "
                               "-- region '%s' skipped.", r["lora"], r["name"])
                continue
            s = r["strength"] * float(base_strength)
            mats_scaled = {sig: {**d, "scale": lora_scale(d) * s}
                           for sig, d in mats.items()}
            prepared.append({"name": r["name"], "lora": r["lora"],
                             "strength": r["strength"], "mats": mats_scaled,
                             "box": r["box"]})

        if not prepared:
            logger.warning("all region LoRAs failed to load; passing model through unchanged.")
            blank = torch.zeros((1, 64, 64, 3))
            info = json.dumps({
                "n_regions": 0,
                "note": "all region LoRAs failed to load (see console log)",
                "regions": [],
            }, indent=2, ensure_ascii=False)
            return (model, blank, info)

        boxes = [p["box"] for p in prepared]

        # -- layer planning + per-region match diagnostics --------------------
        patched = model.clone()
        dm = _diffusion_model_of(patched)
        model_sigs = collect_model_sigs(dm.named_modules())
        plan, per_matched = plan_layer_map([p["mats"] for p in prepared], model_sigs)
        for i, p in enumerate(prepared):
            total = len(p["mats"])
            logger.info("region %d '%s' (%s): matched %d/%d layers",
                        i, p["name"], p["lora"], per_matched[i], total)
            if per_matched[i] == 0:
                logger.warning("region %d '%s': 0 layers matched the model -- this "
                               "LoRA will NOT take effect (wrong architecture or "
                               "key format for the loaded model).", i, p["name"])

        session = _RegionSession(patched, prepared, boxes, seam_feather,
                                 sparse_threshold, plan)

        def wrapper(executor, *args, **kwargs):
            return session.run(executor, *args, **kwargs)

        if hasattr(patched, "add_wrapper_with_key"):
            patched.add_wrapper_with_key(_WRAPPER_ENUM, WRAPPER_KEY, wrapper)
        elif hasattr(patched, "add_wrapper"):
            patched.add_wrapper(_WRAPPER_ENUM, wrapper)
        else:
            raise RuntimeError("This ComfyUI build lacks model wrapper support. Update ComfyUI.")

        # -- rainbow mask preview + info --------------------------------------
        preview = render_preview(boxes, int(canvas_width), int(canvas_height))
        preview_t = torch.from_numpy(preview)

        info = json.dumps({
            "n_regions": len(prepared),
            "grid": "derived from live latent at first model call (canvas size only affects preview)",
            "regions": [
                {"name": p["name"], "lora": p["lora"], "strength": p["strength"],
                 "enable": True,
                 "box": [round(v, 4) for v in p["box"]],
                 "layers_matched": per_matched[i],
                 "layers_total": len(p["mats"])}
                for i, p in enumerate(prepared)
            ],
        }, indent=2, ensure_ascii=False)

        logger.info("armed %d region(s).", len(prepared))
        return (patched, preview_t, info)
