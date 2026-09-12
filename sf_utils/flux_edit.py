# SF Painter Flux Image Edit 编码纯逻辑（复刻 ComfyUI-PainterFluxImageEdit 的 PainterFluxImageEdit）
# 职责：Flux2(Klein) 图编一体编码——把「文本 CLIP 编码 + 每张参考图的 VL 缩放 + VAE 参考编码
#   + reference_latents 注入 + 首图遮罩」组装为 positive / negative / latent 三路。
# 与原版差异（有意为之）：
#   - 去掉 mode 枚举，图片数量由调用方按已连接槽位决定（前端动态槽位）
#   - 不再把 conditioning 列表复制 batch_size 份（ComfyUI 由采样器按 latent batch 自动广播，
#     手动复制会让相同条件被 to_batch 归并成 N² 计算）；latent/noise_mask 用
#     repeat_to_batch_size 对齐 batch（大于切片、小于复制），修复参考图自带 batch 时乘错
#   - 遮罩仍按 latent 空间尺寸生成（原版按 width//8；prepare_mask 本就兜底，这里消除冗余插值）
#   - 新增 negative_prompt / use_reference_latent_as_init / reference_latents_method /
#     encode_vision 开关（后者默认关：当前 ComfyUI 的 Flux2 Klein 视觉通路被忽略）
# 复用 sf_utils/qwen_edit 的 process_vl_image / set_reference_latents / set_conditioning_values
# 依赖 torch / comfy.utils / comfy.model_management / node_helpers（运行时由 ComfyUI 提供）

import torch

import comfy.utils

from .qwen_edit import process_vl_image, set_conditioning_values, set_reference_latents

# VL 视觉塔输入：面积缩放到 384²（与原版一致）。
VL_TARGET_SIZE = 384
VL_UPSCALE = "area"
VL_CROP = "center"

# VAE 参考编码：等比中心裁剪到输出宽高比后缩放到输出尺寸（与原版一致）。
REF_UPSCALE = "lanczos"
REF_CROP = "center"


def _reference_latent(vae, image, width, height):
    """单张参考图编码为 reference latent：中心裁剪到输出宽高比并缩放后 VAE 编码（取 RGB）。"""
    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, width, height, REF_UPSCALE, REF_CROP)
    return vae.encode(resized.movedim(1, -1)[:, :, :, :3])


def _mask_to_noise_mask(mask, latent_height, latent_width):
    """[B,H,W] / [H,W] 遮罩缩放到 latent 空间尺寸，[B,1,H,W] -> [B,h,w]。

    仅接受 2D/3D；其他维度沿用原版语义返回 None（忽略）。
    """
    if mask is None:
        return None
    if mask.dim() == 2:
        m = mask.unsqueeze(0).unsqueeze(1)
    elif mask.dim() == 3:
        m = mask.unsqueeze(1)
    else:
        return None
    return comfy.utils.common_upscale(m, latent_width, latent_height, "area", "center").squeeze(1)


def encode_painter_flux(clip, vae, prompt, images, mask=None, negative_prompt="",
                        width=1024, height=1024, batch_size=1,
                        reference_latents_method="", use_reference_latent_as_init=True,
                        encode_vision=False):
    """Flux2 图编一体编码主入口。

    images: 已提供参考图的张量列表（[B,H,W,C]，顺序即 image1..N）。
    mask:   仅作用于第一张图的遮罩（[B,H,W] 或 [H,W]），可为 None。
    negative_prompt: 负向提示词（空串等价原版恒空）。
    reference_latents_method: 非空时注入（Flux2 默认 index / scale=10）；"" = 用模型默认。
    use_reference_latent_as_init: True（原版）= 有图时以首图 latent 为去噪起点；False = 空 latent。
    encode_vision: True 才计算 VL 缩放并插入 vision token 前缀（当前 Flux2 Klein 视觉通路被忽略）。
    返回 (positive, negative, latent)。
    """
    if vae is None:
        raise RuntimeError("VAE is required. Please connect a VAE loader.")

    images = [img for img in images if img is not None]

    ref_latents = []
    vl_images = []
    image_prompt_prefix = ""

    for i, image in enumerate(images):
        if encode_vision:
            vl_images.append(process_vl_image(image, VL_TARGET_SIZE, VL_CROP, VL_UPSCALE))
            image_prompt_prefix += "image{}: <|vision_start|><|image_pad|><|vision_end|> ".format(i + 1)
        ref_latents.append(_reference_latent(vae, image, width, height))

    full_prompt = image_prompt_prefix + prompt

    tokens = clip.tokenize(full_prompt, images=vl_images)
    positive = clip.encode_from_tokens_scheduled(tokens)

    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt or "", images=[]))

    if ref_latents:
        positive = set_reference_latents(positive, ref_latents)
        negative = set_reference_latents(negative, ref_latents)
    if reference_latents_method:
        positive = set_conditioning_values(positive, {"reference_latents_method": reference_latents_method})
        negative = set_conditioning_values(negative, {"reference_latents_method": reference_latents_method})

    device = comfy.model_management.get_torch_device()
    dummy_pixels = torch.zeros(1, height, width, 3, device=device)
    empty_latent = vae.encode(dummy_pixels)

    latent = {"samples": empty_latent}
    if ref_latents and use_reference_latent_as_init:
        # 有参考图时以首图 latent 作为去噪起点（编辑语义，原版行为）。
        latent["samples"] = ref_latents[0]

    noise_mask = None
    if mask is not None and ref_latents:
        latent_height, latent_width = latent["samples"].shape[-2], latent["samples"].shape[-1]
        noise_mask = _mask_to_noise_mask(mask, latent_height, latent_width)
        if noise_mask is not None:
            latent["noise_mask"] = noise_mask

    # batch 对齐：大于切片、小于复制（不复制 conditioning，采样器自动广播）。
    latent["samples"] = comfy.utils.repeat_to_batch_size(latent["samples"], batch_size)
    if noise_mask is not None:
        latent["noise_mask"] = comfy.utils.repeat_to_batch_size(latent["noise_mask"], batch_size)

    return positive, negative, latent
