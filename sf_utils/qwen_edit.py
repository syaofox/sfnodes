# SF Qwen Edit 编码纯逻辑（复刻自 ComfyUI-EditUtils EditTextEncode_EditUtils 的 qwen 路径）
# 职责：
#   - 参考图 ref 处理：longest_edge 缩放 + pad 画布（vae_unit 对齐）/center/disabled 裁剪 + 主图 mask→noise_mask
#   - Easy 语义扩展（SFEasyKrea2Edit 用）：ref_target_mode 把参考图统一到目标尺寸 / init_image
#     独立初始 latent / 每图 ref_strength
#   - VL 视觉通路：目标面积缩放 + 裁剪
#   - 组装 conditioning（reference_latents）/ latent（初始 latent + noise_mask）/ custom_output
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


def scale_reference(height, width, ref_longest_edge, ref_resize_mode="longest_edge"):
    """参考图缩放：longest_edge 最长边对齐 / area 总面积对齐 ref_longest_edge²。

    返回 (scaled_h, scaled_w)。
    """
    if min(height, width) <= 0 or ref_longest_edge <= 0:
        raise ValueError(f"invalid image size {height}x{width} / ref_longest_edge {ref_longest_edge}")
    if ref_resize_mode == "area":
        scale_by = math.sqrt((ref_longest_edge * ref_longest_edge) / float(width * height))
    else:
        scale_by = max(height, width) / float(ref_longest_edge)
    return int(round(height / scale_by)), int(round(width / scale_by))


def scale_longest_edge(height, width, ref_longest_edge):
    """longest_edge 模式：最长边缩放到 ref_longest_edge，另一边等比。返回 (scaled_h, scaled_w)。"""
    return scale_reference(height, width, ref_longest_edge, "longest_edge")


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


def resize_to_target(samples, target_h, target_w, mode="crop", method="bicubic"):
    """把 (B,C,H,W) 图像缩放到目标 H×W（Easy_QwenEdit2509 的 auto_resize 语义）。

    crop=缩放覆盖后居中裁剪；pad=完整缩放后居中填充黑边；stretch=强制拉伸。
    目标尺寸强制 ≥32（VAE 3×3 卷积要求）；不做 vae_unit 对齐（调用方负责 floor）。
    """
    target_h = max(32, int(target_h))
    target_w = max(32, int(target_w))
    if mode == "stretch":
        return comfy.utils.common_upscale(samples, target_w, target_h, method, "disabled")

    h, w = samples.shape[2], samples.shape[3]
    if mode == "pad":
        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        scaled = comfy.utils.common_upscale(samples, new_w, new_h, method, "disabled")
        canvas = torch.zeros((samples.shape[0], samples.shape[1], target_h, target_w),
                             dtype=samples.dtype, device=samples.device)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[:, :, y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled
        return canvas

    # crop：覆盖缩放后居中裁剪
    scale = max(target_w / float(w), target_h / float(h))
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    scaled = comfy.utils.common_upscale(samples, new_w, new_h, method, "disabled")
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2
    return scaled[:, :, y_offset:y_offset + target_h, x_offset:x_offset + target_w]


def process_reference(vae, image, ref_longest_edge, ref_crop, ref_upscale, is_main, mask=None,
                      vae_unit=VQE_UNIT, ref_resize_mode="longest_edge",
                      target_size=None, target_mode="crop"):
    """单张参考图的 ref 处理。

    image: [B,H,W,C] float 张量；mask: [B,H,W] 或 None（仅主图生效）。
    ref_resize_mode: longest_edge | area（对齐 EditUtils 的 ref_resize_mode）。
    target_size: (H,W) 给定时忽略 ref_longest_edge/ref_crop/ref_resize_mode，按 target_mode
      （crop/pad/stretch）把参考图统一到目标尺寸，再 floor 到 vae_unit 的倍数（Easy 语义）。
    返回 dict：
      vae_image   [B,H',W',C] 编码输入
      ref_latent  vae.encode 输出
      noise_mask  [B,H',W'] 或 None（仅主图 + 有 mask）
      pad_info    dict 或 None（仅主图 + 旧路径 pad 模式）
    """
    samples = image.movedim(-1, 1)  # [B,C,H,W]
    batch, channels = samples.shape[0], samples.shape[1]
    orig_h, orig_w = samples.shape[2], samples.shape[3]

    sample_masks = None
    if mask is not None:
        sample_masks = mask.unsqueeze(1).repeat(1, channels, 1, 1)  # [B,C,H,W]

    noise_mask = None

    if target_size is not None:
        target_h, target_w = int(target_size[0]), int(target_size[1])
        resized = resize_to_target(samples, target_h, target_w, target_mode, ref_upscale)
        width = max(32, (resized.shape[3] // vae_unit) * vae_unit)
        height = max(32, (resized.shape[2] // vae_unit) * vae_unit)
        s = comfy.utils.common_upscale(resized, width, height, ref_upscale, "disabled")
        if sample_masks is not None and is_main:
            m = resize_to_target(sample_masks, target_h, target_w, target_mode, ref_upscale)
            m = comfy.utils.common_upscale(m, width, height, ref_upscale, "disabled")
            noise_mask = m[:, :1, :, :].squeeze(1)
        vae_image = s.movedim(1, -1)
        return {
            "vae_image": vae_image,
            "ref_latent": vae.encode(vae_image[:, :, :, :3]),
            "noise_mask": noise_mask,
            "pad_info": None,
        }

    pad_info = None

    scaled_h, scaled_w = scale_reference(orig_h, orig_w, ref_longest_edge, ref_resize_mode)

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


def process_init_image(vae, image, mask=None, vae_unit=VQE_UNIT):
    """初始 latent（Easy latent_image 语义）：像素居中裁剪到 vae_unit 倍数后 VAE 编码。

    mask: [B,H,W] 或 None；与 image H/W 一致时按 bilinear 对齐并做同样裁剪，产出 noise_mask。
    返回 (latent, noise_mask, (orig_h, orig_w))；原始尺寸供参考图统一缩放用（Easy 先按
    latent_image 原始尺寸缩放参考图，再 floor 到 vae_unit）。
    """
    samples = image.movedim(-1, 1)  # [B,C,H,W]
    orig_h, orig_w = samples.shape[2], samples.shape[3]
    target_h = max(32, (orig_h // vae_unit) * vae_unit)
    target_w = max(32, (orig_w // vae_unit) * vae_unit)
    y_offset = (orig_h - target_h) // 2
    x_offset = (orig_w - target_w) // 2
    cropped = samples[:, :, y_offset:y_offset + target_h, x_offset:x_offset + target_w]
    latent = vae.encode(cropped.movedim(1, -1)[:, :, :, :3])

    noise_mask = None
    if mask is not None:
        if not mask_matches(mask, image):
            print("process_init_image: mask H/W 与 image 不符，忽略该 mask")
        else:
            m = mask.unsqueeze(1) if mask.ndim == 3 else mask  # [B,1,H,W]
            m = comfy.utils.common_upscale(m, orig_w, orig_h, "bilinear", "disabled")
            if (orig_h, orig_w) != (target_h, target_w):
                m = m[:, :, y_offset:y_offset + target_h, x_offset:x_offset + target_w]
            noise_mask = m[:, :1, :, :].squeeze(1)
    return latent, noise_mask, (orig_h, orig_w)


def process_vl_image(image, vl_target_size=384, vl_crop="center", vl_upscale="lanczos",
                     vl_resize=True):
    """视觉塔输入：面积缩放到 vl_target_size²，支持 center/disabled 裁剪。返回 [B,H',W',C]。

    vl_resize=False 时保持原面积（仅当超过 2048² 才缩小），对齐 EditUtils。
    """
    samples = image.movedim(-1, 1)
    orig_h, orig_w = samples.shape[2], samples.shape[3]
    if vl_resize:
        total = int(vl_target_size * vl_target_size)
    else:
        total = int(orig_w * orig_h)
        if total > 2048 * 2048:
            total = 2048 * 2048
    scale_by = math.sqrt(total / float(orig_w * orig_h))
    width = round(orig_w * scale_by)
    height = round(orig_h * scale_by)
    s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
    return s.movedim(1, -1)


def encode_qwen_edit(clip, vae, prompt, entries, ref_upscale="lanczos",
                     vl_target_size=384, vl_crop="center", vl_upscale="lanczos",
                     llama_template=DEFAULT_LLAMA_TEMPLATE, vae_unit=VQE_UNIT,
                     init_image=None, init_mask=None, ref_target_mode=None):
    """主编码入口（qwen 路径）。

    entries: 每张已提供图的配置列表（顺序即 Picture 编号顺序）：
      {"image": [B,H,W,C], "mask": [B,H,W]|None, "ref_longest_edge": int, "ref_crop": str,
       # 可选（缺省 = 旧行为/函数级共享参数）：to_ref/to_vl/vl_resize/ref_main_image/
       # ref_resize_mode/ref_upscale/vl_target_size/vl_crop/vl_upscale/
       # rope_x_offset/rope_y_offset/ref_strength}
    init_image: 独立初始 latent 图（Easy latent_image 语义）；给定时初始 latent 只来自它，
      参考图按 ref_target_mode 统一缩放到它的尺寸；init_mask → noise_mask。
    ref_target_mode: crop/pad/stretch。init_image 缺省时按首个主参考图的原始尺寸统一
      参考图尺寸（Easy auto_resize 语义）；None = 旧路径不统一。
    ref_strength: 每图参考 latent 缩放（乘在写入 conditioning 的副本上，不影响初始 latent）。
    返回 (conditioning, latent_out, custom_output, main_image, noise_mask)。
    """
    pad_info = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 1.0}

    init_latent = None
    init_size = None
    noise_mask = None
    if init_image is not None:
        init_latent, noise_mask, init_size = process_init_image(vae, init_image, init_mask, vae_unit)

    # 主图选择：首个 to_ref 且 ref_main_image 的项；无则回退首个 to_ref 项。
    main_cfg_index = -1
    for i, entry in enumerate(entries):
        if entry.get("to_ref", True) and entry.get("ref_main_image", i == 0):
            main_cfg_index = i
            break
    if main_cfg_index < 0:
        for i, entry in enumerate(entries):
            if entry.get("to_ref", True):
                main_cfg_index = i
                break
    if main_cfg_index < 0 and entries:
        main_cfg_index = 0

    # 目标尺寸：init_image 原始尺寸优先；否则取主参考图原始尺寸（要求统一时）。
    ref_target_size = init_size
    if ref_target_size is None and ref_target_mode is not None and main_cfg_index >= 0:
        main_ref_image = entries[main_cfg_index]["image"]
        ref_target_size = (main_ref_image.shape[1], main_ref_image.shape[2])

    ref_latents = []
    vae_images = []
    vl_images = []
    rope_offsets = []
    image_prompt = ""
    main_ref_pos = 0
    main_ref_latent = None

    for i, entry in enumerate(entries):
        image = entry["image"]
        to_ref = entry.get("to_ref", True)
        to_vl = entry.get("to_vl", True)
        if not to_ref and not to_vl:
            continue

        mask = entry.get("mask")
        if mask is not None and not mask_matches(mask, image):
            print("encode_qwen_edit: mask H/W 与 image 不符，忽略该 mask")
            mask = None
        is_main = (i == main_cfg_index) and to_ref

        if to_ref:
            ref = process_reference(
                vae,
                image,
                entry.get("ref_longest_edge", 1024),
                entry.get("ref_crop", "center"),
                entry.get("ref_upscale", ref_upscale),
                is_main,
                mask=mask,
                vae_unit=vae_unit,
                ref_resize_mode=entry.get("ref_resize_mode", "longest_edge"),
                target_size=ref_target_size,
                target_mode=ref_target_mode or "crop",
            )
            ref_latent = ref["ref_latent"]
            if is_main:
                main_ref_latent = ref_latent
            strength = float(entry.get("ref_strength", 1.0))
            if strength != 1.0:
                ref_latent = ref_latent * strength
            ref_latents.append(ref_latent)
            vae_images.append(ref["vae_image"])
            rope_offsets.append((entry.get("rope_x_offset", 0), entry.get("rope_y_offset", 0)))
            if ref["pad_info"] is not None:
                pad_info = ref["pad_info"]
            if ref["noise_mask"] is not None and init_image is None:
                noise_mask = ref["noise_mask"]
            if is_main:
                main_ref_pos = len(ref_latents) - 1

        if to_vl:
            vl_image = process_vl_image(
                image,
                entry.get("vl_target_size", vl_target_size),
                entry.get("vl_crop", vl_crop),
                entry.get("vl_upscale", vl_upscale),
                vl_resize=entry.get("vl_resize", True),
            )
            vl_images.append(vl_image)
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

    full_prompt = image_prompt + prompt

    # 与 EditUtils 一致：llama_template 非空时始终传入（即使无图片的纯文本编码），
    # 否则不传（走 tokenizer 默认模板）。
    if llama_template:
        tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
    else:
        tokens = clip.tokenize(full_prompt, images=vl_images)

    conditioning = clip.encode_from_tokens_scheduled(tokens)

    no_refs_cond = conditioning
    if ref_latents:
        if any(x or y for x, y in rope_offsets):
            conditioning = set_conditioning_values(
                conditioning,
                {"reference_latents": ref_latents, "reference_rope_offsets": rope_offsets},
                append=True,
            )
        else:
            conditioning = _set_reference_latents(conditioning, ref_latents)

    # 初始 latent：init_image 优先；否则主参考图的未缩放 latent；纯文本为占位张量。
    if init_latent is not None:
        latent_samples = init_latent
    elif ref_latents:
        latent_samples = main_ref_latent if main_ref_latent is not None else ref_latents[0]
    else:
        latent_samples = torch.zeros(TEXT_ONLY_LATENT_SHAPE)

    latent_out = {"samples": latent_samples}
    if noise_mask is not None:
        latent_out["noise_mask"] = noise_mask

    main_image = vae_images[main_ref_pos] if vae_images else None

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


def set_conditioning_values(conditioning, values, append=False):
    """向 conditioning 写入附加字段（node_helpers.conditioning_set_values 的宽松封装）。

    node_helpers 在部分运行环境不可用时静默降级为原样返回，供 Qwen Edit 与
    Painter Flux Edit 等纯逻辑共用；append=True 时列表字段追加而非覆盖。
    """
    try:
        import node_helpers
        return node_helpers.conditioning_set_values(conditioning, values, append=append)
    except Exception:
        return conditioning


def set_reference_latents(conditioning, ref_latents):
    """向 conditioning 追加 reference_latents（append 语义）。"""
    return set_conditioning_values(conditioning, {"reference_latents": ref_latents}, append=True)


# qwen 内部调用点保留原名（与公共别名等价）。
_set_reference_latents = set_reference_latents


def zero_conditioning(conditioning):
    """零化文本条件（复刻核心 ConditioningZeroOut / Easy zero_out）。

    保留 dict 字段（reference_latents 等），仅 cond 张量与 pooled_output/
    conditioning_lyrics 置零，供 SFEasyKrea2Edit 直接输出负条件。
    """
    out = []
    for t in conditioning:
        d = t[1].copy()
        pooled_output = d.get("pooled_output")
        if pooled_output is not None:
            d["pooled_output"] = torch.zeros_like(pooled_output)
        conditioning_lyrics = d.get("conditioning_lyrics")
        if conditioning_lyrics is not None:
            d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
        out.append([torch.zeros_like(t[0]), d])
    return out


def mask_matches(mask, image):
    """mask 与 image 的 H/W 是否一致（[B,H,W] vs [B,H,W,C]）。"""
    return mask is not None and mask.shape[1] == image.shape[1] and mask.shape[2] == image.shape[2]
