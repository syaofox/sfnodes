"""Regional LoRA engine — pure logic for the SFRegionalLoRA node.

Single source of truth for: LoRA key normalization / matrix parsing, region
JSON parsing, model-layer planning (with per-region match diagnostics), token
grid + rectangular mask math, sparse token selection, and the rainbow mask
preview. Framework-agnostic (numpy only, no ComfyUI/torch dependency) so it
can be unit-tested directly; the node converts results to torch once per
session (masks/indices are built once per apply, CPU->GPU copies are
negligible).

The runtime injection itself (forward hooks, x @ down.T @ up.T, index_add_)
lives in nodes/model/regional_lora.py — see that file for the engine.
"""

import json
import math
import re

import numpy as np

# ---------------------------------------------------------------------------
# key normalization — bidirectional: applied to both LoRA keys and model
# module names so they can be matched after stripping prefixes/symbols.
# ---------------------------------------------------------------------------
_PREFIXES = (
    "lora_unet_", "lora_te_", "lora_", "diffusion_model.",
    "diffusion_model_", "transformer.", "model.diffusion_model.",
    "model.", "base_model.",
)
_SD_SCRIPT_ORG_PREFIX = "kohya_xx/"


def normalize_key(s: str) -> str:
    """Normalize a LoRA key base or model module name to a comparable sig.

    Strips common prefixes and removes '.'/'_' (also handles the sd-scripts
    'kohya_xx/' org prefix used by some newer trainers)."""
    s = str(s).lower()
    if s.startswith(_SD_SCRIPT_ORG_PREFIX):
        s = s[len(_SD_SCRIPT_ORG_PREFIX):]
    for pre in _PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
    return s.replace(".", "").replace("_", "")


# ---------------------------------------------------------------------------
# LoRA matrix parsing — classifies keys, never copies tensor data.
# Accepts any mapping of key -> tensor-like object (torch.Tensor at runtime,
# numpy arrays in tests); only shapes and alpha scalars are read.
# ---------------------------------------------------------------------------
_DOWN_RE = re.compile(r"(.*?)\.(lora_down|lora_A)\.weight$")
_UP_RE = re.compile(r"(.*?)\.(lora_up|lora_B)\.weight$")


def _scalar_first(t):
    """First element of a tensor-like as a Python float (numpy / torch,
    CPU or GPU — .item() handles all; np.asarray fallback for bare arrays)."""
    try:
        return float(t.flatten()[0].item())
    except Exception:
        return float(np.asarray(t).reshape(-1)[0])


def parse_lora_sd(sd) -> dict:
    """Classify a LoRA state dict into {sig: {"down", "up", "alpha", "rank"}}.

    Supports kohya (lora_down/up) and diffusers (lora_A/B) key formats.
    Entries missing either factor are skipped. alpha falls back to rank when
    absent. The sig is normalize_key(base), so identical layers in different
    formats collide by design (last one wins, both are the same layer)."""
    groups = {}
    alphas = {}
    for k, v in sd.items():
        key = str(k)
        if key.startswith(_SD_SCRIPT_ORG_PREFIX):
            key = key[len(_SD_SCRIPT_ORG_PREFIX):]
        if key.endswith(".alpha") or key.endswith("alpha"):
            base = re.sub(r"\.?alpha$", "", key)
            try:
                alphas[base] = _scalar_first(v)
            except Exception:
                pass
            continue
        m = _DOWN_RE.match(key)
        if m:
            groups.setdefault(m.group(1), {})["down"] = v
            continue
        m = _UP_RE.match(key)
        if m:
            groups.setdefault(m.group(1), {})["up"] = v
            continue

    out = {}
    for base, mats in groups.items():
        if not base:
            continue
        down, up = mats.get("down"), mats.get("up")
        if down is None or up is None:
            continue
        try:
            shape = down.shape if hasattr(down, "shape") else np.asarray(down).shape
            rank = int(shape[0])
        except Exception:
            continue
        alpha = alphas.get(base, float(rank))
        out[normalize_key(base)] = {
            "down": down, "up": up, "alpha": float(alpha), "rank": rank,
        }
    return out


def lora_scale(entry: dict) -> float:
    """alpha/rank — the base multiplier later scaled by region strength."""
    rank = max(1, int(entry.get("rank", 0) or 1))
    return float(entry.get("alpha", float(rank))) / float(rank)


# ---------------------------------------------------------------------------
# region JSON parsing — rows of {lora, strength, enable, x, y, w, h}
# ---------------------------------------------------------------------------
DEFAULT_STRENGTH = 1.0


def default_regions_json(n=2) -> str:
    """Equal left->right columns with no LoRA assigned."""
    rows = []
    for i in range(max(1, n)):
        rows.append({
            "lora": "None", "strength": DEFAULT_STRENGTH, "enable": True,
            "x": round(i / n, 6), "y": 0.0, "w": round(1.0 / n, 6), "h": 1.0,
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "none", "")
    return bool(v)


def parse_regions(regions_json: str) -> list:
    """Parse the hidden regions_json into a list of region dicts.

    Each region: {name, lora, strength, enable, box:(x0,y0,x1,y1)}. Missing /
    malformed x/y/w/h falls back to an equal-width column; boxes are clamped
    to [0,1] and inverted boxes are flipped. Any error returns [] (the node
    then passes the model through unchanged)."""
    try:
        raw = json.loads(regions_json or "[]")
    except Exception:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            continue
        try:
            strength = float(r.get("strength", DEFAULT_STRENGTH))
        except Exception:
            strength = DEFAULT_STRENGTH
        lora = str(r.get("lora", "None") or "None")
        enable = _as_bool(r.get("enable", True))
        box = _box_from_region(r, i, len(raw))
        if box is None:
            continue
        out.append({
            "name": str(r.get("name") or f"region_{i}"),
            "lora": lora,
            "strength": strength,
            "enable": enable,
            "box": box,
        })
    return out


def _box_from_region(r, i, n):
    try:
        has_box = all(k in r for k in ("x", "y", "w", "h"))
        if has_box:
            bx, by, bw, bh = (float(r["x"]), float(r["y"]),
                              float(r["w"]), float(r["h"]))
        else:
            bx, by, bx1, by1 = _default_box(i, n)
            bw, bh = bx1 - bx, by1 - by
    except Exception:
        return None
    if not (math.isfinite(bx) and math.isfinite(by)
            and math.isfinite(bw) and math.isfinite(bh)):
        return None
    if bw < 0 or bh < 0:
        return None
    if bw <= 0 or bh <= 0:
        # degenerate box -> fall back to the equal-width column
        bx, by, bx1, by1 = _default_box(i, n)
        bw, bh = bx1 - bx, by1 - by
    x0, y0 = max(0.0, min(1.0, bx)), max(0.0, min(1.0, by))
    x1, y1 = max(x0, min(1.0, bx + bw)), max(y0, min(1.0, by + bh))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _default_box(i, n):
    n = max(1, n)
    return (i / n, 0.0, (i + 1) / n, 1.0)


# ---------------------------------------------------------------------------
# layer planning — the diagnostic core (per-region match counts)
# ---------------------------------------------------------------------------
def collect_model_sigs(modules_iter, weight_filter=None) -> set:
    """Normalize the module names of an iterable of (name, module) pairs into
    a sig set. Only modules with a weight attribute are considered (Linear /
    fp8-wrapped Linear / anything LoRA keys may target)."""
    filt = weight_filter or (lambda mod: hasattr(mod, "weight"))
    sigs = set()
    for name, mod in modules_iter:
        if filt(mod):
            sigs.add(normalize_key(name))
    return sigs


def plan_layer_map(regions_mats, model_sigs) -> tuple:
    """Plan which sigs need hooks for which regions.

    regions_mats: list (one per active region) of {sig: entry} dicts.
    model_sigs: set of normalized model module sigs.

    Returns (plan, per_region_matched):
      plan: {sig: set(region_idx)} — one forward hook per sig, iterating its
            region fns (mirror of the runtime layer map).
      per_region_matched: list of matched-layer counts, one per region —
            REGION i: m/M layers is the key diagnostic for "a region's LoRA
            never takes effect" (its sigs matched 0 model layers)."""
    plan = {}
    per_region_matched = []
    for i, mats in enumerate(regions_mats):
        matched = 0
        for sig in mats.keys():
            if sig in model_sigs:
                plan.setdefault(sig, set()).add(i)
                matched += 1
        per_region_matched.append(matched)
    return plan, per_region_matched


# ---------------------------------------------------------------------------
# token grid + rectangular mask math (Krea2: VAE f8 + patch2 -> latent//2)
# ---------------------------------------------------------------------------
def token_grid(H: int, W: int):
    """(rows, cols) of image tokens for a latent of height H, width W."""
    return max(1, int(H) // 2), max(1, int(W) // 2)


def _sigmoid(x):
    # float32 exp overflows around x=+-88; clipping to +-80 keeps sigmoid
    # exactly 0/1 at the tails in float32 without overflow warnings.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def normalize_overlap(masks):
    """Scale overlapping region masks so their per-token sum never exceeds 1.

    Tokens covered by a single region (sum <= 1) are untouched — feathered
    edges keep their shape. In overlap zones (sum > 1) every mask is divided
    by the sum, so two fully-overlapping regions split the injection 50/50
    instead of stacking two full-strength deltas (which reads as an
    over-amplified local LoRA stack). Non-overlapping boxes are bit-for-bit
    unaffected, so this is safe to always apply."""
    if not masks:
        return masks
    total = np.sum(masks, axis=0)
    norm = np.maximum(total, 1.0)
    return [m / norm for m in masks]


def rect_token_mask(rows: int, cols: int, box, feather: float) -> np.ndarray:
    """Feathered rectangular mask over the token grid, row-major flattened.

    box: (x0, y0, x1, y1) normalized 0-1.

    Feather semantics: the transition band (mask ~0.05..0.95) spans about
    `feather * cols` tokens (similarly for rows). sigmoid needs ~3 units of
    its argument to go 0.05 -> 0.95, so the scale is feather*cols/6 — the
    naive feather*cols makes the tail reach ~23 columns outside the box at
    feather=0.08 on a 64-wide grid, i.e. left/right region masks overlap
    ~85% and regional isolation silently degrades into full-image mixing
    (observed in the field: two character LoRAs both bleeding everywhere)."""
    x0, y0, x1, y1 = box
    c0, c1 = x0 * cols, x1 * cols
    r0, r1 = y0 * rows, y1 * rows
    fc = max(1e-3, float(feather) * cols / 6.0)
    fr = max(1e-3, float(feather) * rows / 6.0)
    cc = np.arange(cols, dtype=np.float32)[None, :]
    rr = np.arange(rows, dtype=np.float32)[:, None]
    in_x = _sigmoid((cc - c0) / fc) * _sigmoid((c1 - cc) / fc)
    in_y = _sigmoid((rr - r0) / fr) * _sigmoid((r1 - rr) / fr)
    return (in_y * in_x).reshape(-1).clip(0.0, 1.0)


def active_token_indices(mask: np.ndarray, threshold: float, seq: int,
                         n_img: int):
    """Sparse token selection for one region.

    mask: [n_img] token mask. Sequence layout is [text | image] (Krea2
    concatenates context before img), so image tokens occupy the tail:
    idx = keep + (seq - n_img). Falls back to the whole sequence with the
    mean weight when the mask doesn't line up with the sequence.

    Returns (idx, weight) numpy arrays of dtype int64 / float32."""
    if n_img <= 0 or n_img > seq:
        idx = np.arange(seq, dtype=np.int64)
        weight = np.full((seq,), float(mask.mean()),
                         dtype=np.float32) if mask.size else np.zeros(
                             (seq,), dtype=np.float32)
        return idx, weight
    keep = np.nonzero(np.abs(mask) > threshold)[0]
    return (keep + (seq - n_img)).astype(np.int64), mask[keep].astype(np.float32)


# ---------------------------------------------------------------------------
# rainbow mask preview
# ---------------------------------------------------------------------------
def _hsv_to_rgb(h, s, v):
    h = h % 360.0
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    if h < 60:
        r_, g_, b_ = c, x, 0.0
    elif h < 120:
        r_, g_, b_ = x, c, 0.0
    elif h < 180:
        r_, g_, b_ = 0.0, c, x
    elif h < 240:
        r_, g_, b_ = 0.0, x, c
    elif h < 300:
        r_, g_, b_ = x, 0.0, c
    else:
        r_, g_, b_ = c, 0.0, x
    return (r_ + m, g_ + m, b_ + m)


def render_preview(boxes, w: int, h: int) -> np.ndarray:
    """[1, h, w, 3] float32 rainbow preview: one color per box, 0.5 opacity
    over black, overlapping boxes take the max (later boxes win visually)."""
    preview = np.zeros((1, int(h), int(w), 3), dtype=np.float32)
    n = max(1, len(boxes))
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        color = _hsv_to_rgb((i / n) * 360.0, 0.75, 0.9)
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = max(px0 + 1, int(x1 * w)), max(py0 + 1, int(y1 * h))
        px1, py1 = min(int(w), px1), min(int(h), py1)
        if px1 <= px0 or py1 <= py0:
            continue
        fill = np.array([c * 0.5 for c in color], dtype=np.float32)
        preview[0, py0:py1, px0:px1] = np.maximum(
            preview[0, py0:py1, px0:px1], fill)
    return preview
