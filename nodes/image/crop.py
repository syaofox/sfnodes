"""SF Image Crop / SF Image Uncrop — ported from comfyui-pixaroma
node_crop.py + node_uncrop.py (PixaromaCrop / PixaromaUncrop).

Crop: visual crop via the on-node panel + fullscreen editor (web/sf_crop.js);
source comes from a wired IMAGE, drag-drop, Ctrl+V paste or the editor's
Load Image button. Outputs the crop plus an "SF_CROP_INFO" wire carrying the
full original + crop rect, so Uncrop can paste an edited crop back.

Uncrop: paste an edited crop back onto the original at the exact spot,
optionally feathering the seam. Pure Python (no JS).

Disk state lives in input/sfnodes_crop/ (route-guarded: dataURL uploads +
safe_join on every read).
"""

import base64
import io
import json
import os
import re
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aiohttp import web

import folder_paths

from ...sf_utils.common import AnyType

_CATEGORY = "sfnodes/image"

any_type = AnyType("*")

# Custom wire type carrying everything Uncrop needs: the full original image
# plus the crop's top-left (x, y) and size (w, h). Plain string so the two
# classes stay decoupled (no cross-file import chain).
SF_CROP_INFO = "SF_CROP_INFO"

# Disk state root (inside ComfyUI's input/), isolated from other packs.
_CROP_SUBDIR = "sfnodes_crop"


def _crop_dir() -> str:
    d = os.path.join(folder_paths.get_input_directory(), _CROP_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _safe_join(rel: str) -> str:
    """Resolve a saved relative path inside input/sfnodes_crop/, returning an
    absolute path or None if it escapes the directory or doesn't exist.

    Rejects absolute / drive-qualified / UNC values lexically BEFORE any
    filesystem resolve (a UNC path would open an SMB connection just by being
    resolved), then realpath + startswith containment."""
    if not rel or not isinstance(rel, str):
        return None
    q = rel.strip().strip('"').strip("'")
    if not q:
        return None
    if q.replace("/", "\\").startswith("\\\\"):
        return None
    try:
        if os.path.splitdrive(q)[0]:
            return None
        if os.path.isabs(q):
            return None
    except (ValueError, TypeError):
        return None
    root = os.path.realpath(_crop_dir())
    try:
        full = os.path.realpath(os.path.join(root, q))
    except (OSError, ValueError, TypeError):
        return None
    if full == root or not full.startswith(root + os.sep):
        return None
    if not os.path.exists(full):
        return None
    return full


def _sanitize_id(raw, fallback: str) -> str:
    """Strip every character that is not a word char / dash, so a crafted
    project_id can never smuggle a path separator."""
    s = str(raw or "")
    s = re.sub(r"[^A-Za-z0-9_-]", "", s)
    return s[:64] or fallback


def _decode_image(b64: str):
    """Decode a dataURL (or bare base64) into a PIL Image, or None."""
    if not isinstance(b64, str) or not b64:
        return None
    try:
        payload = b64.split(",", 1)[-1] if "," in b64 else b64
        raw = base64.b64decode(payload)
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception:
        return None


# ── 节点类 ────────────────────────────────────────────────────────────────


def _crop_meta_from_widget(crop_data) -> dict:
    """Resolve the CropWidget input (whatever shape the frontend sent) into a
    meta dict. Accepts:
      - dict {"crop_json": "<json>"}            (DOM widget serializeValue 形状)
      - dict 直接含 crop_w 的 meta
      - str  "<json>"                           (部分前端把 widget 值序列化为字符串)
      - str  '{"crop_json": "<json>"}'          (字符串再套一层)
    Returns {} on any parse failure (crop degrades to passthrough)."""
    if crop_data is None:
        return {}
    raw = crop_data
    if isinstance(raw, dict):
        if "crop_json" in raw and isinstance(raw["crop_json"], str):
            raw = raw["crop_json"]
        else:
            return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if isinstance(parsed, dict) and "crop_json" in parsed and isinstance(parsed["crop_json"], str):
        try:
            parsed = json.loads(parsed["crop_json"])
        except Exception:
            return {}
    return parsed if isinstance(parsed, dict) else {}


class _CropOptionalInputs(dict):
    """Any-type optional inputs that ALSO declare concrete IMAGE / MASK
    inputs, so the node's registered schema lists them and the node appears
    when you drag from an IMAGE output and search for a compatible node."""

    def __init__(self, type):
        super().__init__()
        self.type = type
        self["image"] = ("IMAGE", {
            "tooltip": "接入上游 IMAGE 进行裁剪（LoadImage、VAE Decode 等任意来源）。也可拖放图片文件到节点上或 Ctrl+V 粘贴——那些方式会直接加载图片并断开此连线。",
        })
        self["mask"] = ("MASK", {
            "tooltip": "可选。接入 MASK（如 LoadImage 的 MASK 输出）以相同的矩形同时裁剪透明通道，结果从 mask 输出。不接时 mask 输出为与裁剪同尺寸的全不透明遮罩（除非加载的文件自带透明度，会取用其透明通道）。",
        })

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return (self.type,)


class SFImageCrop:
    DESCRIPTION = (
        "可视化裁剪任意图片，无需手输像素坐标。三种提供源的方式：接入上游 "
        "IMAGE（LoadImage、VAE Decode、ControlNet 输出等任意来源）、拖放图片文件到"
        "节点上、或 Ctrl+V 从剪贴板粘贴。拖放与粘贴会自动断开上游连线，以手动加载"
        "的图片为准。\n\n"
        "节点面板提供 宽/高/X/Y/比例/对齐 字段，数字输入支持算式（如 1024+512、"
        "512*2）。选择非 Free 的对齐后，修改 W/H 会自动重算 X/Y（如选中居中裁剪时"
        "把 W 改成 512 会自动居中对齐）。全屏编辑器支持拖拽裁剪矩形与手柄，提供常用"
        "比例（1:1、16:9、9:16 等），编辑器内还可加载图片（同样断开上游连线）。\n\n"
        "输出裁剪后的 IMAGE、匹配的 MASK（接入 MASK 则按同一矩形裁透明通道）、宽高，"
        "以及 crop_info 连线——把它接入 SF Image Uncrop 即可把编辑后的裁剪区域贴回"
        "原图的精确位置（裁剪→修复/放大→贴回工作流）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _CropOptionalInputs(any_type),
            # 隐藏状态输入：必须在 Python 侧声明，否则前端 validatePrompt 会把
            # 不在节点 schema 中的输入（SFCropJson/CropWidget）从 prompt 剥离，
            # 后端收到空 crop 数据 → 透传原图（实测 kwargs_keys=['image','mask']）。
            # 前端隐藏 STRING widget（同名）的值经标准 widget 通道收集。
            "hidden": {
                "SFCropJson": ("STRING", {"default": "{}"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", SF_CROP_INFO, "INT", "INT")
    RETURN_NAMES = ("image", "mask", "crop_info", "width", "height")
    OUTPUT_TOOLTIPS = (
        "裁剪后的图片",
        "与图片同一矩形裁出的遮罩。来自接入的 MASK 输入（或拖放/粘贴文件自身的透明度）；无透明信息时为与裁剪同尺寸的全不透明遮罩",
        "供 SF Image Uncrop 使用的裁剪信息——携带原图与裁剪位置，编辑后的裁剪区域可精确贴回。接入 SF Image Uncrop",
        "裁剪宽度（像素）",
        "裁剪高度（像素）",
    )
    FUNCTION = "load_crop"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when crop metadata changes. Upstream IMAGE
        changes are already detected by ComfyUI's input-hash mechanism; the
        disk-composite fallback additionally keys on the file mtime."""
        # SFCropJson（隐藏 STRING widget，标准 widget 值通道）优先——Vue 前端
        # 对 DOM widget（CropWidget）的值收集在某些版本/渲染器下不可靠；
        # 回退 CropWidget 兼容旧 prompt。
        crop_data = kwargs.get("SFCropJson")
        if crop_data is None:
            crop_data = kwargs.get("CropWidget")
        if not crop_data:
            return ""
        try:
            meta = _crop_meta_from_widget(crop_data)
            rect_key = f"{meta.get('crop_x','')}-{meta.get('crop_y','')}-{meta.get('crop_w','')}-{meta.get('crop_h','')}"
            if kwargs.get("image") is not None:
                return rect_key
            composite_path = meta.get("composite_path", "")
            if composite_path:
                full_path = _safe_join(composite_path)
                if full_path and os.path.exists(full_path):
                    return f"{os.path.getmtime(full_path)}:{rect_key}"
        except Exception:
            pass
        return str(crop_data)

    def _save_source_temp(self, tensor):
        """Save the *input* tensor (full uncropped, batch slot 0) to ComfyUI's
        temp/ as a UUID-named PNG so the JS editor + mini-preview can fetch
        it via /view?type=temp. Best-effort."""
        try:
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4 or tensor.shape[0] == 0:
                return None
            arr = tensor[0].clamp(0.0, 1.0).cpu().numpy()
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
            img = Image.fromarray(arr)
            temp_dir = folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            fname = f"sf_crop_src_{uuid.uuid4().hex}.png"
            img.save(os.path.join(temp_dir, fname), "PNG")
            return fname
        except Exception as e:
            print(f"[SFImageCrop] temp source save failed: {e}")
            return None

    def load_crop(self, **kwargs):
        empty_image = torch.ones((1, 1024, 1024, 3), dtype=torch.float32)

        # SFCropJson（隐藏 STRING widget，标准 widget 值通道）优先——Vue 前端
        # 对 DOM widget（CropWidget）的值收集在某些版本/渲染器下不可靠；
        # 回退 CropWidget 兼容旧 prompt。
        crop_data = kwargs.get("SFCropJson")
        if crop_data is None:
            crop_data = kwargs.get("CropWidget")
        upstream = kwargs.get("image")
        upstream_mask = kwargs.get("mask")

        # No widget AND no upstream → return empty
        if not crop_data and upstream is None and upstream_mask is None:
            return (empty_image, self._default_mask(1024, 1024),
                    self._identity_crop_info(empty_image), 1024, 1024)

        # Parse crop metadata (may be empty if user just wired upstream and
        # never opened the editor)
        meta = {}
        if crop_data:
            meta = _crop_meta_from_widget(crop_data)

        # Capture the *input* tensor URL for the JS editor + mini-preview.
        ui_payload = None
        if isinstance(upstream, torch.Tensor):
            src_fname = self._save_source_temp(upstream)
            if src_fname:
                ui_payload = {"sf_crop_source": [
                    {"filename": src_fname, "subfolder": "", "type": "temp"}
                ]}

        # Apply the crop: IMAGE + MASK are cut with the SAME absolute-pixel
        # rect so transparency lines up with the cropped image.
        if isinstance(upstream, torch.Tensor):
            try:
                img_t, out_w, out_h = self._crop_tensor(upstream, meta)
                mask_t = self._crop_mask(upstream_mask, meta, out_w, out_h)
                full_mask = upstream_mask if isinstance(upstream_mask, torch.Tensor) else None
                crop_info = self._make_crop_info(upstream, meta, full_mask)
            except Exception as e:
                print(f"[SFImageCrop] upstream crop error: {e}")
                img_t, mask_t, out_w, out_h, crop_info = self._load_disk_composite(meta, empty_image, upstream_mask)
        else:
            img_t, mask_t, out_w, out_h, crop_info = self._load_disk_composite(meta, empty_image, upstream_mask)

        result = (img_t, mask_t, crop_info, out_w, out_h)
        if ui_payload:
            return {"ui": ui_payload, "result": result}
        return result

    # ─────────────────────────────────────────────────────────────────────

    def _rect_from_meta(self, meta, w, h):
        """Resolve the saved crop rect to absolute, clamped pixel bounds
        (x0, y0, x1, y1) for a w×h surface, or None when the crop should be a
        pass-through. Coordinates are ABSOLUTE pixels (no proportional rescale
        from original_w/original_h); out-of-bounds coords are clamped."""
        if not meta or meta.get("crop_w") in (None, 0):
            return None
        crop_x = float(meta.get("crop_x", 0))
        crop_y = float(meta.get("crop_y", 0))
        crop_w = float(meta.get("crop_w", w))
        crop_h = float(meta.get("crop_h", h))
        x0 = max(0, int(round(crop_x)))
        y0 = max(0, int(round(crop_y)))
        x1 = min(int(w), int(round(crop_x + crop_w)))
        y1 = min(int(h), int(round(crop_y + crop_h)))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _crop_tensor(self, tensor, meta):
        """Crop an upstream IMAGE tensor [B,H,W,C] using the saved rect.
        Empty meta → pass through unmodified."""
        if tensor.dim() != 4 or tensor.shape[0] == 0:
            if tensor.dim() >= 3:
                return (tensor, int(tensor.shape[-2]), int(tensor.shape[-3]))
            return (tensor, 0, 0)

        b, h, w, c = tensor.shape
        rect = self._rect_from_meta(meta, w, h)
        if rect is None:
            return (tensor, int(w), int(h))

        x0, y0, x1, y1 = rect
        cropped = tensor[:, y0:y1, x0:x1, :].contiguous()
        return (cropped, int(x1 - x0), int(y1 - y0))

    def _default_mask(self, w, h):
        """Fully-opaque mask (zeros) sized w×h — 0 means 'keep / opaque'."""
        return torch.zeros((1, max(1, int(h)), max(1, int(w))), dtype=torch.float32)

    def _crop_mask(self, mask, meta, fallback_w, fallback_h):
        """Crop a MASK tensor [B,H,W] (also tolerates [H,W]) with the SAME
        rect as the image; rect clamped to the mask's own dimensions."""
        if not isinstance(mask, torch.Tensor):
            return self._default_mask(fallback_w, fallback_h)
        m = mask
        if m.dim() == 2:
            m = m[None, ...]
        if m.dim() != 3:
            return self._default_mask(fallback_w, fallback_h)
        mh, mw = int(m.shape[-2]), int(m.shape[-1])
        rect = self._rect_from_meta(meta, mw, mh)
        if rect is None:
            return m.contiguous()
        x0, y0, x1, y1 = rect
        return m[:, y0:y1, x0:x1].contiguous()

    def _make_crop_info(self, original, meta, full_mask=None):
        """Bundle what Uncrop needs: the FULL original image and (optionally)
        the FULL original mask, plus the crop's top-left (x, y) and size."""
        if not isinstance(original, torch.Tensor) or original.dim() != 4:
            return None
        H, W = int(original.shape[1]), int(original.shape[2])
        rect = self._rect_from_meta(meta, W, H)
        if rect is None:
            x0, y0, cw, ch = 0, 0, W, H
        else:
            x0, y0, x1, y1 = rect
            cw, ch = x1 - x0, y1 - y0
        info = {"image": original, "x": x0, "y": y0, "w": cw, "h": ch,
                "orig_w": W, "orig_h": H}
        if isinstance(full_mask, torch.Tensor):
            info["mask"] = full_mask
        return info

    def _identity_crop_info(self, image_t, full_mask=None):
        """Crop info for an already-cropped image (editor composite or the
        empty fallback): paste-back becomes a whole-image replace at (0, 0)."""
        if not isinstance(image_t, torch.Tensor) or image_t.dim() != 4:
            return None
        H, W = int(image_t.shape[1]), int(image_t.shape[2])
        info = {"image": image_t, "x": 0, "y": 0, "w": W, "h": H,
                "orig_w": W, "orig_h": H}
        if isinstance(full_mask, torch.Tensor):
            info["mask"] = full_mask
        return info

    def _full_mask_from_pil(self, pil):
        """The FULL (uncropped) mask from a loaded file's alpha channel
        (mask = 1 - alpha), or None when the file has no transparency."""
        try:
            if "A" in pil.getbands():
                alpha = np.array(pil.convert("RGBA").split()[-1]).astype(np.float32) / 255.0
                return torch.from_numpy(1.0 - alpha)[None,]
        except Exception:
            pass
        return None

    def _load_disk_composite(self, meta, empty_image, upstream_mask=None):
        """Load a saved image from input/sfnodes_crop/. Two paths:

        1. src_path: the FULL uncropped source (paste / drag-drop / the
           editor's Load Image). Load it and apply crop_x/y/w/h on the Python
           side AND build a real crop_info carrying the full original — which
           Uncrop needs for paste-back. PREFERRED.
        2. composite_path: the editor-saved pre-cropped PNG. Used ONLY when no
           usable source is on disk. Its crop_info is identity (paste-back
           can't reconstruct the full original from it).
        """
        doc_w = int(meta.get("doc_w", 1024))
        doc_h = int(meta.get("doc_h", 1024))

        composite_path = meta.get("composite_path", "")
        src_path = meta.get("src_path", "")

        # Prefer the FULL source so crop_info carries the original for paste-back.
        if src_path and _safe_join(src_path):
            return self._load_src_and_crop(src_path, meta, doc_w, doc_h, empty_image, upstream_mask)

        if composite_path and _safe_join(composite_path):
            return self._load_image_from_disk(
                composite_path, doc_w, doc_h, empty_image, meta, upstream_mask, already_cropped=True)

        # Nothing on disk → return a blank doc-sized image + matching mask
        arr = np.ones((doc_h, doc_w, 3), dtype=np.float32)
        blank = torch.from_numpy(arr)[None,]
        mask_t = self._crop_mask(upstream_mask, meta, doc_w, doc_h)
        full_mask = upstream_mask if isinstance(upstream_mask, torch.Tensor) else None
        return (blank, mask_t, doc_w, doc_h, self._identity_crop_info(blank, full_mask))

    def _derive_disk_mask(self, pil, meta, upstream_mask, out_w, out_h, already_cropped):
        """Work out the MASK for a disk-loaded image. Priority:
        1. A wired MASK input (cropped with the saved rect).
        2. The file's own alpha channel if it has one (mask = 1 - alpha).
        3. A fully-opaque default mask sized to the crop.
        """
        if isinstance(upstream_mask, torch.Tensor):
            return self._crop_mask(upstream_mask, meta, out_w, out_h)
        try:
            if "A" in pil.getbands():
                alpha = np.array(pil.convert("RGBA").split()[-1]).astype(np.float32) / 255.0
                m = torch.from_numpy(1.0 - alpha)[None,]  # [1,H,W], 1 = transparent
                if already_cropped:
                    return m.contiguous()
                return self._crop_mask(m, meta, out_w, out_h)
        except Exception as e:
            print(f"[SFImageCrop] alpha mask derive failed: {e}")
        return self._default_mask(out_w, out_h)

    def _load_image_from_disk(self, rel_path, doc_w, doc_h, empty_image,
                              meta=None, upstream_mask=None, already_cropped=True):
        full_path = _safe_join(rel_path)
        if not full_path:
            return (empty_image, self._default_mask(doc_w, doc_h), doc_w, doc_h,
                    self._identity_crop_info(empty_image))
        try:
            pil = Image.open(full_path)
            img = pil.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            t = torch.from_numpy(arr)[None,]
            # Report the ACTUAL loaded dimensions (they can disagree with the
            # stale doc_w/doc_h if the composite was modified externally).
            ow, oh = int(t.shape[2]), int(t.shape[1])
            mask_t = self._derive_disk_mask(pil, meta or {}, upstream_mask, ow, oh, already_cropped)
            return (t, mask_t, ow, oh, self._identity_crop_info(t, mask_t))
        except Exception as e:
            print(f"[SFImageCrop] Load error: {e}")
            return (empty_image, self._default_mask(1024, 1024), 1024, 1024,
                    self._identity_crop_info(empty_image))

    def _load_src_and_crop(self, src_path, meta, doc_w, doc_h, empty_image, upstream_mask=None):
        """Load the uncropped source image and apply crop_x/y/w/h."""
        full_path = _safe_join(src_path)
        if not full_path:
            return (empty_image, self._default_mask(doc_w, doc_h), doc_w, doc_h,
                    self._identity_crop_info(empty_image))
        try:
            pil = Image.open(full_path)
            img = pil.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr)[None,]  # [1, H, W, 3] (FULL source)
            img_t, ow, oh = self._crop_tensor(tensor, meta)
            mask_t = self._derive_disk_mask(pil, meta, upstream_mask, ow, oh, already_cropped=False)
            full_mask = upstream_mask if isinstance(upstream_mask, torch.Tensor) else self._full_mask_from_pil(pil)
            return (img_t, mask_t, ow, oh, self._make_crop_info(tensor, meta, full_mask))
        except Exception as e:
            print(f"[SFImageCrop] src load error: {e}")
            return (empty_image, self._default_mask(doc_w, doc_h), doc_w, doc_h,
                    self._identity_crop_info(empty_image))


class SFImageUncrop:
    DESCRIPTION = (
        "把编辑后的裁剪区域贴回原图的精确位置——经典的裁剪→修复/放大→贴回工作流。\n\n"
        "将 SF Image Crop 的 crop_info 输出接入 crop_info，把编辑后的裁剪图"
        "（放大、修复、调色后的任意结果）接入 image。节点会把编辑图自动缩放到"
        "原裁剪区域尺寸并合成到完整原图上，裁剪区域之外保持原样。\n\n"
        "透明度全帧传递：mask 输出是原遮罩（裁剪区域更新为新值），把 Image Crop "
        "的 mask 直连过来即可保持整图透明度（或接入编辑后的区域遮罩只改该区域）。"
        "feather 羽化接缝实现无缝融合。输出重组后的完整图片与全帧遮罩。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        # Slot order is image, mask, crop_info so wires run straight across
        # from Image Crop's outputs.
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "要贴回的编辑后裁剪图（放大/修复/调色等任意结果）。尺寸与原裁剪区域不同时会自动缩放。",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "可选。裁剪区域的遮罩——用于更新全帧 mask 输出的该区域（其余保持原遮罩）。把 Image Crop 的 mask 直连过来可保持整图透明度。会自动缩放到裁剪区域尺寸。它不限制图片贴回，整个区域都会被贴上。",
                }),
                "crop_info": (SF_CROP_INFO, {
                    "tooltip": "接入 SF Image Crop 的 crop_info 输出。携带原图与裁剪位置，编辑后的裁剪图可精确贴回。不接时编辑图直接透传。",
                }),
                "feather": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "将贴回区域的边缘按该像素数向内羽化，使其与原图融合。0 = 硬边。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", SF_CROP_INFO)
    RETURN_NAMES = ("image", "mask", "crop_info")
    OUTPUT_TOOLTIPS = (
        "贴回编辑裁剪图后的完整原图",
        "全帧遮罩：原遮罩的裁剪区域被接入的 mask 更新（未接入则保持不变）。可接入 Join Image with Alpha 等保持透明度",
        "原样透传 crop_info，便于不重拉连线就转发给其他节点",
    )
    FUNCTION = "uncrop"
    CATEGORY = _CATEGORY

    # ─────────────────────────────────────────────────────────────────────

    def _resize_bhwc(self, t, target_w, target_h):
        """Resize an image tensor [B,H,W,3] to [B,target_h,target_w,3]."""
        x = t.permute(0, 3, 1, 2)  # [B,3,H,W]
        x = F.interpolate(x, size=(int(target_h), int(target_w)),
                          mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()

    def _resize_mask(self, m, target_w, target_h):
        """Resize a mask tensor [B,H,W] to [B,target_h,target_w]."""
        x = m[:, None, ...]  # [B,1,H,W]
        x = F.interpolate(x, size=(int(target_h), int(target_w)),
                          mode="bilinear", align_corners=False)
        return x[:, 0, ...].contiguous()

    def _feather_alpha(self, alpha, feather):
        """Feather the alpha INWARD: ramp from 0 at the rectangle edge up to
        the alpha's own value `feather` pixels inward, so a pasted crop fades
        all the way to nothing at its boundary. A box blur would bottom out at
        ~0.5 at the edge (50/50 blend right at the boundary) — the distance
        ramp reaches a true 0 at the edge."""
        k = int(feather)
        if k <= 0:
            return alpha
        ch, cw = int(alpha.shape[-2]), int(alpha.shape[-1])
        ys = torch.arange(ch, dtype=torch.float32).view(ch, 1)
        xs = torch.arange(cw, dtype=torch.float32).view(1, cw)
        dist_y = torch.minimum(ys, (ch - 1) - ys)
        dist_x = torch.minimum(xs, (cw - 1) - xs)
        dist = torch.minimum(dist_y, dist_x)            # px to the nearest edge
        ramp = (dist / float(k)).clamp(0.0, 1.0)        # 0 at edge -> 1 at k px in
        return (alpha * ramp).clamp(0.0, 1.0)

    def _build_alpha(self, mask, cw, ch, feather):
        """Alpha map [ch,cw] in 0..1 for the paste: from the optional mask
        (else all-ones), with the edges feathered by `feather` px."""
        if isinstance(mask, torch.Tensor):
            m = mask
            if m.dim() == 2:
                m = m[None, ...]
            if m.dim() == 3:
                a = self._resize_mask(m[:1], cw, ch)[0]  # [ch,cw]
            else:
                a = torch.ones((ch, cw), dtype=torch.float32)
        else:
            a = torch.ones((ch, cw), dtype=torch.float32)
        a = a.to(torch.float32)
        return self._feather_alpha(a.clamp(0.0, 1.0), feather)

    def _passthrough_mask(self, mask, image):
        """When there's nothing to paste (no crop_info), forward the WIRED
        mask unchanged so transparency survives. Falls back to all-zeros
        (every pixel opaque) sized to the image only when no mask is wired."""
        dev = image.device if isinstance(image, torch.Tensor) else "cpu"
        if isinstance(mask, torch.Tensor):
            m = mask
            if m.dim() == 4:  # squeeze a singleton channel dim
                if m.shape[1] == 1:
                    m = m[:, 0]
                elif m.shape[-1] == 1:
                    m = m[..., 0]
            if m.dim() == 2:
                m = m[None, ...]
            if m.dim() == 3:
                return m.to(dev, torch.float32)
        h = int(image.shape[1]) if isinstance(image, torch.Tensor) and image.dim() == 4 else 1
        w = int(image.shape[2]) if isinstance(image, torch.Tensor) and image.dim() == 4 else 1
        return torch.zeros((1, h, w), dtype=torch.float32, device=dev)

    def uncrop(self, image, crop_info=None, mask=None, feather=0):
        # No crop_info wired -> nothing to paste back, so pass the image AND
        # the mask straight through (the mask must NOT be blanked, or
        # transparency is lost).
        if not isinstance(crop_info, dict) or not isinstance(crop_info.get("image"), torch.Tensor):
            print("[SFImageUncrop] no crop_info wired - passing image + mask through")
            ci_out = crop_info if isinstance(crop_info, dict) else None
            return (image, self._passthrough_mask(mask, image), ci_out)

        base = crop_info["image"]
        if base.dim() != 4:
            return (image, self._passthrough_mask(mask, image), crop_info)

        H, W = int(base.shape[1]), int(base.shape[2])
        x = int(crop_info.get("x", 0))
        y = int(crop_info.get("y", 0))
        cw = int(crop_info.get("w", image.shape[2] if image.dim() == 4 else W))
        ch = int(crop_info.get("h", image.shape[1] if image.dim() == 4 else H))

        # Clamp the paste region to the base image bounds (defensive against a
        # hand-edited / stale crop_info).
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        cw = max(1, min(cw, W - x))
        ch = max(1, min(ch, H - y))

        # Resize the edited crop to exactly fill the original crop region.
        patch = image
        if patch.dim() != 4:
            patch = base.new_zeros((1, ch, cw, base.shape[3]))
        if int(patch.shape[1]) != ch or int(patch.shape[2]) != cw:
            patch = self._resize_bhwc(patch, cw, ch)

        # Match the patch's channels to the base (drop alpha, pad gray if needed).
        if patch.shape[3] != base.shape[3]:
            if patch.shape[3] > base.shape[3]:
                patch = patch[..., :base.shape[3]]
            else:
                pad_c = base.shape[3] - patch.shape[3]
                patch = torch.cat([patch, patch[..., -1:].repeat(1, 1, 1, pad_c)], dim=-1)

        # ---- IMAGE: paste the edited crop into the whole region -----------
        # The mask input is NOT a paste-limiter: the entire crop rectangle is
        # pasted and only `feather` softens the seam.
        seam = self._build_alpha(None, cw, ch, feather)  # [ch,cw] ones, feathered edges (cpu)
        a = seam[None, ..., None].to(base.device, base.dtype)  # [1,ch,cw,1]

        out = base.clone()
        B = int(out.shape[0])

        # Align the patch batch to the base batch.
        if patch.shape[0] != B:
            if patch.shape[0] == 1:
                patch = patch.repeat(B, 1, 1, 1)
            elif B == 1:
                out = out.repeat(patch.shape[0], 1, 1, 1)
                B = patch.shape[0]
            else:
                n = min(B, patch.shape[0])
                out = out[:n]
                patch = patch[:n]
                B = n

        patch = patch.to(out.device, out.dtype)
        region = out[:, y:y + ch, x:x + cw, :]
        out[:, y:y + ch, x:x + cw, :] = patch * a + region * (1.0 - a)

        # ---- MASK: full-frame original mask, region updated if one is wired
        base_mask = crop_info.get("mask")
        if isinstance(base_mask, torch.Tensor):
            bm = base_mask
            if bm.dim() == 4:
                if bm.shape[1] == 1:
                    bm = bm[:, 0]
                elif bm.shape[-1] == 1:
                    bm = bm[..., 0]
            if bm.dim() == 2:
                bm = bm[None, ...]
            if bm.dim() == 3 and int(bm.shape[-2]) == H and int(bm.shape[-1]) == W:
                bm = bm[:1].detach().to("cpu", torch.float32)
            else:
                bm = torch.zeros((1, H, W), dtype=torch.float32)
        else:
            bm = torch.zeros((1, H, W), dtype=torch.float32)

        out_mask = bm.clone()
        if isinstance(mask, torch.Tensor):
            region_mask = self._build_alpha(mask, cw, ch, 0).detach().to("cpu", torch.float32)
            sa = seam.detach().to("cpu", torch.float32)
            cur = out_mask[:, y:y + ch, x:x + cw]
            out_mask[:, y:y + ch, x:x + cw] = region_mask[None, ...] * sa + cur * (1.0 - sa)

        out_mask = out_mask.clamp(0.0, 1.0)
        out_mask = out_mask.to(out.device)
        if out_mask.shape[0] == 1 and out.shape[0] > 1:
            out_mask = out_mask.repeat(out.shape[0], 1, 1, 1)

        return (out.clamp(0.0, 1.0), out_mask, crop_info)


# ── 自定义 API 路由（副作用注册，改动需重启容器） ─────────────────────────


def _register_routes():
    try:
        from server import PromptServer
        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.post("/api/sfnodes/crop/save")
        async def _save_crop_composite(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            project_id = _sanitize_id(data.get("project_id", ""), uuid.uuid4().hex)
            img = _decode_image(data.get("image_merged", ""))
            if img is None:
                return web.json_response({"error": "Invalid image data"}, status=400)
            filename = f"crop_composite_{project_id}.png"
            try:
                img.save(os.path.join(_crop_dir(), filename), "PNG")
            except Exception:
                return web.json_response({"error": "Save failed"}, status=500)
            return web.json_response({"status": "success",
                                      "composite_path": os.path.join(_CROP_SUBDIR, filename).replace("\\", "/")})

        @routes.post("/api/sfnodes/crop/upload_src")
        async def _upload_crop_source(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            project_id = _sanitize_id(data.get("project_id", ""), uuid.uuid4().hex)
            img = _decode_image(data.get("image", ""))
            if img is None:
                return web.json_response({"error": "Invalid image data"}, status=400)
            filename = f"crop_src_{project_id}.png"
            try:
                img.save(os.path.join(_crop_dir(), filename), "PNG")
            except Exception:
                return web.json_response({"error": "Save failed"}, status=500)
            return web.json_response({"status": "success",
                                      "path": os.path.join(_CROP_SUBDIR, filename).replace("\\", "/")})
    except Exception:
        pass


_register_routes()
