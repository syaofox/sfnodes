"""SF Load Image Resize — native LoadImage parity + inline resize.

Ported from comfyui-pixaroma's node_load_image.py (PixaromaLoadImage).

Architecture mirrors the source: hidden input + graphToPrompt injection of
state JSON from node.properties (web/sf_load_image.js keeps the frontend in
lockstep). The resize engine lives in sf_utils/resize_engine.py; its JS
mirror is previewResize in web/sf_load_image_resize.js.
"""

import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers

from ...sf_utils.resize_engine import _resize_frame, RESIZE_DEFAULTS

_CATEGORY = "sfnodes/image"

DEFAULT_STATE = {
    "version": 1,
    **RESIZE_DEFAULTS,
    "pad_color": "#808080",
}


def _parse_state(state_json: str) -> dict:
    """Parse the hidden SFLoadImageResizeState JSON. Falls back to
    DEFAULT_STATE on any parse error (state may be missing or malformed in
    subgraph / partial-prompt cases)."""
    if not state_json:
        return dict(DEFAULT_STATE)
    try:
        parsed = json.loads(state_json)
        merged = dict(DEFAULT_STATE)
        merged.update({k: v for k, v in parsed.items() if k in DEFAULT_STATE})
        return merged
    except Exception:
        print("[SFLoadImageResize] Malformed state JSON, using defaults")
        return dict(DEFAULT_STATE)


def _parse_orig_name(state_json: str) -> str:
    """Read the original (non-clipspace) filename the frontend injects at
    submission time. Parsed separately from _parse_state because that helper
    filters keys down to DEFAULT_STATE and would drop orig_name. Returns ""
    when absent or malformed."""
    if not state_json:
        return ""
    try:
        v = json.loads(state_json).get("orig_name")
        return v if isinstance(v, str) else ""
    except Exception:
        return ""


# ── Node class ───────────────────────────────────────────────────────────────


class SFLoadImageResize:
    DESCRIPTION = (
        "加载图片并支持内联缩放：与原生 LoadImage 相同的上传/拖放/粘贴、多帧、"
        "alpha 通道转遮罩，另加 8 种缩放模式——最大百万像素、最长边、倍率、"
        "适应（Fit inside）、裁剪填充（Crop to fill）、匹配宽高比、像素填充（Pad）。"
        "支持吸附（snap）、重采样方式与防放大开关。\n\n"
        "输出：image、mask、宽度、高度、文件名、原始宽度、原始高度。"
        "多数工作流中可替代 Get Image Size + Image Scale + Image Resize 链。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        # Walk input/ recursively so subfolders are visible in the dropdown.
        # Native ComfyUI's LoadImage uses os.listdir (root only), so files
        # inside e.g. input/Studio1/ never appear. Paths are reported relative
        # to input/, with forward slashes, matching what
        # folder_paths.get_annotated_filepath expects on the read side.
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.isdir(input_dir):
            for root, _dirs, fnames in os.walk(input_dir):
                rel_root = os.path.relpath(root, input_dir)
                for fname in fnames:
                    rel = fname if rel_root == "." else os.path.join(rel_root, fname)
                    files.append(rel.replace("\\", "/"))
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True, "tooltip": "从 ComfyUI input 目录加载的图片。可点上传按钮、拖放文件到节点、从剪贴板粘贴，或从下拉列表选择。"}),
            },
            "hidden": {
                "SFLoadImageResizeState": (
                    "STRING",
                    {"default": json.dumps(DEFAULT_STATE)},
                ),
            },
        }

    CATEGORY = _CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "image", "mask", "width", "height",
        "filename", "original_width", "original_height",
    )
    OUTPUT_TOOLTIPS = (
        "加载的图片（缩放后）",
        "图片的遮罩，来自其 alpha 通道（无 alpha 则为空遮罩）",
        "输出宽度（缩放后）",
        "输出高度（缩放后）",
        "图片文件名",
        "原始图片宽度（缩放前）",
        "原始图片高度（缩放前）",
    )
    FUNCTION = "load_image"

    def load_image(self, image: str, SFLoadImageResizeState: str = ""):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        orig_w = orig_h = None
        final_w = final_h = None

        # Match native LoadImage's tensor dtype so fp16 / bf16 pipelines
        # don't need an extra cast downstream. Fall back to float32 if the
        # comfy import isn't available (unit-test or non-Comfy runtime).
        try:
            import comfy.model_management as _mm
            tensor_dtype = _mm.intermediate_dtype()
        except Exception:
            tensor_dtype = torch.float32

        state = _parse_state(SFLoadImageResizeState)

        # Filename output: when the Mask Editor / Copy-Paste Clipspace swaps
        # in a clipspace copy, the loaded path is an auto-generated
        # "clipspace-mask-NNNN.png". The frontend passes the original
        # (non-clipspace) name in orig_name so we report THAT instead, keeping
        # the Filename output stable across masking. Falls back to the actual
        # loaded file's name for normal picks (and for reloaded saved
        # workflows where the original name was never stored).
        orig_name = _parse_orig_name(SFLoadImageResizeState)
        is_clipspace = "clipspace" in image.replace("\\", "/").lower()
        if is_clipspace and orig_name:
            # Normalize Windows separators so a "Studio1\cat.png" orig_name
            # strips the subfolder on a POSIX server too (combo values use "/",
            # so this is belt-and-braces).
            basename = os.path.splitext(os.path.basename(orig_name.replace("\\", "/")))[0]
        else:
            basename = os.path.splitext(os.path.basename(image_path))[0]

        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
            if frame.mode == "I":
                frame = frame.point(lambda px: px * (1 / 255))
            rgb = frame.convert("RGB")

            if orig_w is None:
                orig_w, orig_h = rgb.size
            if rgb.size != (orig_w, orig_h):
                continue

            # Build the PIL mask (1 - alpha, or zeros if none).
            if "A" in frame.getbands():
                alpha = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                mask_pil = Image.fromarray(
                    ((1.0 - alpha) * 255).astype(np.uint8), mode="L"
                )
            elif frame.mode == "P" and "transparency" in frame.info:
                alpha = np.array(
                    frame.convert("RGBA").getchannel("A")
                ).astype(np.float32) / 255.0
                mask_pil = Image.fromarray(
                    ((1.0 - alpha) * 255).astype(np.uint8), mode="L"
                )
            else:
                mask_pil = Image.new("L", rgb.size, 0)

            # Apply resize (Off-mode passthrough until modes fill in).
            rgb_resized, mask_resized, frame_w, frame_h = _resize_frame(
                rgb, mask_pil, state, orig_w, orig_h,
            )
            final_w, final_h = frame_w, frame_h

            arr = np.array(rgb_resized).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr)[None,].to(dtype=tensor_dtype)
            mask_arr = np.array(mask_resized).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).to(dtype=tensor_dtype)

            output_images.append(tensor)
            output_masks.append(mask_tensor)

            if img.format == "MPO":
                break  # native LoadImage same: only first frame for MPO

        if len(output_images) == 0:
            # Defensive — never happens for valid PIL images but keeps tensor
            # shapes consistent if we ever hit a pathological file.
            zeros = torch.zeros((1, 64, 64, 3), dtype=tensor_dtype)
            zeros_mask = torch.zeros((1, 64, 64), dtype=tensor_dtype)
            return (zeros, zeros_mask, 64, 64, basename, 64, 64)

        if len(output_images) > 1:
            out_img = torch.cat(output_images, dim=0)
            out_mask = torch.cat(output_masks, dim=0)
        else:
            out_img = output_images[0]
            out_mask = output_masks[0]

        # `final_w` / `final_h` are set by the last frame's resize call. All
        # frames must produce the same dims by construction (same input dims +
        # same state), so this is safe.
        if final_w is None:
            final_w, final_h = orig_w, orig_h

        return (out_img, out_mask, final_w, final_h, basename, orig_w, orig_h)

    @classmethod
    def IS_CHANGED(cls, image, SFLoadImageResizeState=""):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        m.update((SFLoadImageResizeState or "").encode("utf-8"))
        return m.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, SFLoadImageResizeState=""):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
