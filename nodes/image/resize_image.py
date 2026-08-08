"""SF Image Resize — mid-workflow image resize (ported from comfyui-pixaroma
node_image_resize.py, PixaromaImageResize).

Resizes a wired image (+ optional mask) using the shared resize engine
(sf_utils/resize_engine.py). Optional wired width/height/longest_side drive the
target-size modes. Returns IMAGE, MASK, WIDTH, HEIGHT, LONGEST_SIDE and ships
an `executed` UI payload with input/output dims so the frontend can show the
result readout.

Frontend: web/sf_image_resize*.js (state lives in node.properties +
graphToPrompt injection, same pattern as SFLoadImageResize).
"""

import json

import numpy as np
import torch
from PIL import Image, ImageOps

from ...sf_utils.resize_engine import (
    _apply_wired_size,
    _resize_frame,
    RESIZE_DEFAULTS,
    parse_resize_state,
)

_CATEGORY = "sfnodes/image"

DEFAULT_STATE = {
    "version": 1,
    **RESIZE_DEFAULTS,
    "pad_color": "#808080",  # gray default for Pad (Load Image keeps black)
}

# Modes that consume an explicit W x H target. Wired width/height feed these.
_WH_MODES = ("fit_inside", "cover")


def _tensor_to_pils(image_t):
    """BHWC float tensor -> (list of RGB PIL images, list of alpha PIL 'L' or None).

    Defensive about channel count: ComfyUI IMAGE is normally 3-channel, but a
    stray 1-channel image (grayscale, or a mask rewired into the image slot) or
    a 4-channel RGBA tensor must not crash the run. 1ch -> replicated to RGB.

    The image STAYS 3-channel, deliberately: ComfyUI's VAEEncode hands
    `pixels` to `vae.encode` with NO channel slicing, so a 4-channel image
    reaching a sampler dies in the first conv. The alpha (if any) leaves
    through the MASK output instead — ComfyUI keeps transparency there.
    """
    arr = (image_t.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    out, alphas = [], []
    for frame in arr:
        alpha = None
        if frame.ndim == 2:                        # (H,W) grayscale
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] >= 3:                 # RGB / RGBA
            if frame.shape[-1] >= 4:
                alpha = Image.fromarray(frame[..., 3], "L")
            frame = frame[..., :3]
        else:                                      # 1- or 2-channel -> grayscale
            frame = np.repeat(frame[..., :1], 3, axis=-1)
        out.append(Image.fromarray(frame, "RGB"))
        alphas.append(alpha)
    return out, (alphas if any(a is not None for a in alphas) else None)


def _alpha_to_mask_pils(alphas, size):
    """The picture's own alpha, as masks in ComfyUI's polarity.

    INVERTED, and that is not a detail: `LoadImage` builds its MASK output as
    `1.0 - alpha` and `JoinImageWithAlpha` reads it back as `1.0 - mask`, so
    the house convention is **1 = transparent**. Emitting raw alpha would look
    plausible and rebuild every picture inside out.
    """
    out = []
    for a in alphas:
        if a is None:
            out.append(Image.new("L", size, 0))    # opaque -> nothing masked
        else:
            out.append(ImageOps.invert(a if a.size == size else a.resize(size, Image.NEAREST)))
    return out


def _mask_to_pils(mask_t, count, size):
    """BHW float tensor -> list of L PIL images, each conformed to `size` (the
    image size) so the mask always matches the image. Blank (zeros) when mask_t
    is None. An incoming mask is frequently the wrong size (ComfyUI's LoadImage
    emits a 64x64 zero mask when the image has no alpha); resize it (NEAREST,
    to keep crisp edges) up front, otherwise the output mask won't match the
    output image."""
    if mask_t is None:
        return [Image.new("L", size, 0) for _ in range(count)]
    arr = (mask_t.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    out = []
    for m in arr:
        pim = Image.fromarray(m, "L")
        if pim.size != size:
            pim = pim.resize(size, Image.NEAREST)
        out.append(pim)
    while len(out) < count:
        out.append(Image.new("L", size, 0))
    return out[:count]


class SFImageResize:
    DESCRIPTION = (
        "工作流中途缩放图片（及遮罩）：Off、最大百万像素、最长边、倍率、适应"
        "（Fit inside）、裁剪填充（Crop to fill）、匹配宽高比、像素填充（Pad，"
        "外绘加边框）。可接线驱动目标尺寸（如来自分辨率节点）：只接 width 或 "
        "height 按比例缩放；两者都接为精确尺寸；或只接一个 longest_side 让较长"
        "边缩放为目标值（竖图横图通用，优先于 width/height）。"
        "输出图片、遮罩、宽度、高度、最长边。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "要缩放的图片。"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "可选遮罩，随图片一起缩放（NEAREST 保持清晰边缘）。Pad 模式下新增边框变为白色（修复区域）。"}),
                "width": ("INT", {"forceInput": True, "tooltip": "可选目标宽度（如来自分辨率节点）。只接 width 或只接 height 时按比例缩放；两者都接为精确尺寸。接线期间对应字段被锁定。"}),
                "height": ("INT", {"forceInput": True, "tooltip": "可选目标高度（如来自分辨率节点）。只接 width 或只接 height 时按比例缩放；两者都接为精确尺寸。接线期间对应字段被锁定。"}),
                "longest_side": ("INT", {"forceInput": True, "tooltip": "可选的较长边目标值。接一个数字（如来自数值节点）后图片按比例缩放到较长边等于该值，竖图横图通用，无需选择宽或高。接线时优先于 width/height。"}),
            },
            "hidden": {
                "SFImageResizeState": (
                    "STRING",
                    {"default": json.dumps(DEFAULT_STATE)},
                ),
            },
        }

    CATEGORY = _CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height", "longest_side")
    OUTPUT_TOOLTIPS = (
        "缩放后的图片。",
        "缩放后的遮罩（Pad 模式下白色 = 填充/修复区域）。mask 输入为空且图片自带透明度时，"
        "该透明度转为遮罩输出（1 = 透明），可连同图片接入 Join Image with Alpha 还原透明图。",
        "输出宽度（像素）。",
        "输出高度（像素）。",
        "输出宽高中的较长者，与朝向无关。",
    )
    FUNCTION = "resize"

    def resize(self, image, mask=None, width=None, height=None, longest_side=None, SFImageResizeState=""):
        state = parse_resize_state(SFImageResizeState, DEFAULT_STATE)

        try:
            import comfy.model_management as _mm
            tensor_dtype = _mm.intermediate_dtype()
        except Exception:
            tensor_dtype = torch.float32

        rgb_frames, alpha_frames = _tensor_to_pils(image)
        orig_w, orig_h = rgb_frames[0].size
        # A picture that carries its own transparency, with nothing wired into
        # `mask`, hands that transparency to the mask output — so it survives
        # the resize instead of being silently thrown away (a background-
        # removed image would come out on solid black with an EMPTY mask, i.e.
        # unrecoverable). A WIRED mask always wins: that is an explicit choice,
        # and second-guessing it would be worse than the bug.
        if mask is None and alpha_frames is not None:
            mask_frames = _alpha_to_mask_pils(alpha_frames, (orig_w, orig_h))
            while len(mask_frames) < len(rgb_frames):
                mask_frames.append(Image.new("L", (orig_w, orig_h), 0))
            mask_frames = mask_frames[:len(rgb_frames)]
        else:
            mask_frames = _mask_to_pils(mask, len(rgb_frames), (orig_w, orig_h))

        state = _apply_wired_size(state, width, height, longest_side, orig_w, orig_h)

        out_imgs, out_masks = [], []
        final_w = final_h = None
        for rgb, m in zip(rgb_frames, mask_frames):
            r_rgb, r_mask, fw, fh = _resize_frame(rgb, m, state, orig_w, orig_h)
            final_w, final_h = fw, fh
            out_imgs.append(
                torch.from_numpy(np.array(r_rgb).astype(np.float32) / 255.0)[None,].to(dtype=tensor_dtype)
            )
            out_masks.append(
                torch.from_numpy(np.array(r_mask).astype(np.float32) / 255.0)[None,].to(dtype=tensor_dtype)
            )

        out_image = torch.cat(out_imgs, dim=0) if len(out_imgs) > 1 else out_imgs[0]
        out_mask = torch.cat(out_masks, dim=0) if len(out_masks) > 1 else out_masks[0]
        if final_w is None:
            final_w, final_h = orig_w, orig_h

        longest = max(final_w, final_h)
        return {
            "ui": {
                "sf_image_resize": [{
                    "in_w": orig_w, "in_h": orig_h,
                    "out_w": final_w, "out_h": final_h,
                }],
            },
            "result": (out_image, out_mask, final_w, final_h, longest),
        }


NODE_CLASS_MAPPINGS = {"SFImageResize": SFImageResize}
NODE_DISPLAY_NAME_MAPPINGS = {"SFImageResize": "SF Image Resize"}
