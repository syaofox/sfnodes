# SF Qwen Edit 编码纯逻辑（复刻自 ComfyUI-EditUtils EditTextEncode_EditUtils 的 qwen 路径）
# 职责：
#   - 参考图 ref 处理：longest_edge 缩放 + pad 画布（vae_unit 对齐）/center/disabled 裁剪 + 主图 mask→noise_mask
#   - VL 视觉通路：目标面积缩放 + 裁剪
#   - 组装 conditioning（reference_latents）/ latent（主图 ref latent + noise_mask）/ custom_output
# 与原版差异（有意为之）：
#   - 裁剪 rope offsets（reference_rope_offsets 无 ComfyUI 核心消费端）
#   - ref_resize_mode 仅保留 longest_edge（原包装默认模式）
#   - 每图独立 ref_longest_edge / ref_crop / mask（替代原版共享参数 + Config 链）
# 依赖 torch / comfy.utils（comfy.utils.common_upscale 仅用 "center"/"disabled" 两种 crop）

import math

import torch
import comfy.utils

VQE_UNIT = 8

DEFAULT_LLAMA_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

TEXT_ONLY_LATENT_SHAPE = (1, 4, 128, 128)


def scale_longest_edge(height, width, ref_longest_edge):
    """longest_edge 模式：最长边缩放到 ref_longest_edge，另一边等比。返回 (scaled_h, scaled_w)。"""
    ori_longest = max(height, width)
    if min(height, width) <= 0 or ref_longest_edge <= 0:
        raise ValueError(f"invalid image size {height}x{width} / ref_longest_edge {ref_longest_edge}")
    scale_by = ori_longest / ref_longest_edge
    return int(round(height / scale_by)), int(round(width / scale_by))


def pad_info_from(orig_w, orig_h, resized_w, resized_h):
    """主图 pad 信息：width/height 为右/下黑边像素，scale_by 为原图→缩放图的尺寸比（3 位小数）。"""
    scale_by = math.sqrt(float(resized_w * resized_h) / float(orig_w * orig_h))
    return {
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0,
        "scale_by": round(1.0 / scale_by, 3),
    }


def process_reference(vae, image, ref_longest_edge, ref_crop, ref_upscale, is_main, mask=None, vae_unit=VQE_UNIT):
    """单张参考图的 ref 处理。

    image: [B,H,W,C] float 张量；mask: [B,H,W] 或 None（仅主图生效）。
    返回 dict：
      vae_image   [B,H',W',C] 编码输入
      ref_latent  vae.encode 输出
      noise_mask  [B,H',W'] 或 None（仅主图 + 有 mask）
      pad_info    dict 或 None（仅主图 + pad 模式）
    """
    samples = image.movedim(-1, 1)  # [B,C,H,W]
    batch, channels = samples.shape[0], samples.shape[1]
    orig_h, orig_w = samples.shape[2], samples.shape[3]

    sample_masks = None
    if mask is not None:
        sample_masks = mask.unsqueeze(1).repeat(1, channels, 1, 1)  # [B,C,H,W]

    scaled_h, scaled_w = scale_longest_edge(orig_h, orig_w, ref_longest_edge)
    noise_mask = None
    pad_info = None

    if ref_crop == "pad":
        crop = "center"
        canvas_w = math.ceil(scaled_w / vae_unit) * vae_unit
        canvas_h = math.ceil(scaled_h / vae_unit) * vae_unit
        canvas = torch.zeros((batch, channels, canvas_h, canvas_w), dtype=samples.dtype, device=samples.device)
        resized = comfy.utils.common_upscale(samples, scaled_w, scaled_h, ref_upscale, crop)
        resized_h, resized_w = resized.shape[2], resized.shape[3]
        canvas[:, :, :resized_h, :resized_w] = resized
        if is_main:
            pad_info = pad_info_from(orig_w, orig_h, resized_w, resized_h)
            pad_info["width"] = canvas_w - resized_w
            pad_info["height"] = canvas_h - resized_h
        if sample_masks is not None and is_main:
            mask_canvas = torch.zeros_like(canvas)
            resized_masks = comfy.utils.common_upscale(sample_masks, scaled_w, scaled_h, ref_upscale, crop)
            mask_canvas[:, :, :resized_h, :resized_w] = resized_masks
            noise_mask = mask_canvas[:, :1, :, :].squeeze(1)
        s = canvas
    else:
        crop = ref_crop
        width = round(scaled_w / vae_unit) * vae_unit
        height = round(scaled_h / vae_unit) * vae_unit
        s = comfy.utils.common_upscale(samples, width, height, ref_upscale, crop)
        if sample_masks is not None and is_main:
            m = comfy.utils.common_upscale(sample_masks, width, height, ref_upscale, crop)
            noise_mask = m[:, :1, :, :].squeeze(1)

    vae_image = s.movedim(1, -1)
    return {
        "vae_image": vae_image,
        "ref_latent": vae.encode(vae_image[:, :, :, :3]),
        "noise_mask": noise_mask,
        "pad_info": pad_info,
    }


def process_vl_image(image, vl_target_size=384, vl_crop="center", vl_upscale="lanczos"):
    """视觉塔输入：面积缩放到 vl_target_size²，支持 center/disabled 裁剪。返回 [B,H',W',C]。"""
    samples = image.movedim(-1, 1)
    total = int(vl_target_size * vl_target_size)
    orig_h, orig_w = samples.shape[2], samples.shape[3]
    scale_by = math.sqrt(total / float(orig_w * orig_h))
    width = round(orig_w * scale_by)
    height = round(orig_h * scale_by)
    s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
    return s.movedim(1, -1)


def encode_qwen_edit(clip, vae, prompt, entries, ref_upscale="lanczos",
                     vl_target_size=384, vl_crop="center", vl_upscale="lanczos",
                     llama_template=DEFAULT_LLAMA_TEMPLATE):
    """主编码入口（qwen 路径）。

    entries: 每张已提供图的配置列表（顺序即 Picture 编号顺序）：
      {"image": [B,H,W,C], "mask": [B,H,W]|None, "ref_longest_edge": int, "ref_crop": str}
    返回 (conditioning, latent_out, custom_output, main_image, noise_mask)。
    """
    pad_info = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 1.0}
    main_index = 0 if entries else -1

    ref_latents = []
    vae_images = []
    vl_images = []
    noise_mask = None
    image_prompt = ""

    for i, entry in enumerate(entries):
        image = entry["image"]
        is_main = i == main_index
        ref = process_reference(
            vae,
            image,
            entry["ref_longest_edge"],
            entry["ref_crop"],
            ref_upscale,
            is_main,
            mask=entry.get("mask"),
        )
        ref_latents.append(ref["ref_latent"])
        vae_images.append(ref["vae_image"])
        if ref["pad_info"] is not None:
            pad_info = ref["pad_info"]
        if ref["noise_mask"] is not None:
            noise_mask = ref["noise_mask"]

        vl_image = process_vl_image(image, vl_target_size, vl_crop, vl_upscale)
        vl_images.append(vl_image)
        image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

    full_prompt = image_prompt + prompt

    if vl_images:
        tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
    else:
        tokens = clip.tokenize(full_prompt, images=[])

    conditioning = clip.encode_from_tokens_scheduled(tokens)

    no_refs_cond = conditioning
    if ref_latents:
        conditioning = _set_reference_latents(conditioning, ref_latents)
        latent_samples = ref_latents[main_index]
    else:
        latent_samples = torch.zeros(TEXT_ONLY_LATENT_SHAPE)

    latent_out = {"samples": latent_samples}
    if noise_mask is not None:
        latent_out["noise_mask"] = noise_mask

    main_image = vae_images[main_index] if vae_images else None

    custom_output = {
        "pad_info": pad_info,
        "full_refs_cond": conditioning,
        "main_image": main_image,
        "vae_images": vae_images,
        "ref_latents": ref_latents,
        "vl_images": vl_images,
        "full_prompt": full_prompt,
        "no_refs_cond": no_refs_cond,
        "mask": noise_mask,
    }

    return conditioning, latent_out, custom_output, main_image, noise_mask


def _set_reference_latents(conditioning, ref_latents):
    """向 conditioning 追加 reference_latents（node_helpers.conditioning_set_values 的宽松封装）。"""
    try:
        import node_helpers
        return node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
    except Exception:
        return conditioning


def mask_matches(mask, image):
    """mask 与 image 的 H/W 是否一致（[B,H,W] vs [B,H,W,C]）。"""
    return mask is not None and mask.shape[1] == image.shape[1] and mask.shape[2] == image.shape[2]
