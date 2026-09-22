"""高斯融合分块 VAE 编解码（Qwen-Image 2D VAE）— 移植自 VOSR 的 tiled_vae.py，
经 ylchen333/ComfyUI-VOSR2（Apache-2.0）。分块几何统一收敛到 `tiling.py`。

整图 VAE 编解码大图会 OOM（Qwen VAE 内部 fp32），分块把峰值限制在单块激活。
编码用 `latent_dist.mode()`（确定性）而非 `.sample()`：逐块独立采样会引入
块间不相关噪声，在拼缝处产生明显色差。

为什么纯高斯融合就够（不需要 SD/LDM 的 pad+crop VAEHook 机制）：Qwen VAE 用
RMSNorm2D —— 逐像素/逐通道归一化，没有需要跨块统一的统计量（无 GroupNorm），
每块前向独立，重叠区加权融合即可。
"""
import math

import torch
import torch.nn.functional as F

from .tiling import AE_FACTOR, make_tile_grid, vae_tile_params


def _gaussian_weights(tile_h: int, tile_w: int, channels: int, device) -> torch.Tensor:
    """中心峰值二维高斯融合权重 (1, C, tile_h, tile_w)。"""
    var = 0.01
    mid_h, mid_w = (tile_h - 1) / 2, (tile_w - 1) / 2
    y = torch.arange(tile_h, dtype=torch.float32)
    x = torch.arange(tile_w, dtype=torch.float32)
    wy = torch.exp(-((y - mid_h) / tile_h) ** 2 / (2 * var))
    wx = torch.exp(-((x - mid_w) / tile_w) ** 2 / (2 * var))
    w = wy[:, None] * wx[None, :]
    return w.to(device).unsqueeze(0).unsqueeze(0).expand(1, channels, -1, -1)


def pad_reflect_safe(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """右下补边；pad >= 该维尺寸时 reflect 非法（要求 pad < dim），退化为 replicate。

    小图/极端长宽比（如 2:1 补成方形）会触发，上游直接崩；replicate 仅影响贴边像素，
    与「该尺寸本就不该走整图路径」的告警口径一致。
    """
    if pad_h == 0 and pad_w == 0:
        return x
    mode = "reflect" if (pad_h < x.shape[2] and pad_w < x.shape[3]) else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)


def _pad_to_multiple(x: torch.Tensor, multiple: int = AE_FACTOR):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    return pad_reflect_safe(x, pad_h, pad_w), h, w


def encode_latent(vae, x: torch.Tensor):
    """单次确定性 VAE 编码 → 归一化潜变量。返回 (latent, mean, std)。"""
    device = x.device
    latents_mean = torch.tensor(vae.latents_mean, device=device).view(1, -1, 1, 1)
    latents_std = 1.0 / torch.tensor(vae.latents_std, device=device).view(1, -1, 1, 1)
    z = vae.encode(x).latent_dist.mode()
    return (z - latents_mean) * latents_std, latents_mean, latents_std


def decode_latent(vae, sr_latent: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """单次 VAE 解码 → 像素 [-1, 1]。"""
    sr_latent = sr_latent / latents_std + latents_mean
    return vae.decode(sr_latent).sample.clamp(-1, 1)


def tiled_encode_latent(vae, x: torch.Tensor, tile_size: int, tile_overlap: int):
    """输出潜变量形状与整图路径一致：ceil(h/8) x ceil(w/8)。支持任意 B。"""
    pad, orig_h, orig_w = _pad_to_multiple(x, AE_FACTOR)
    lh, lw = pad.shape[2] // AE_FACTOR, pad.shape[3] // AE_FACTOR
    lt_size, lt_overlap = vae_tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        return encode_latent(vae, pad)

    h_pos = make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = make_tile_grid(lw, lt_size, lt_overlap)

    b = pad.shape[0]
    acc = wacc = g = None
    mean = std = None
    for hi in h_pos:
        for wi in w_pos:
            crop = pad[:, :, hi * AE_FACTOR:(hi + lt_size) * AE_FACTOR, wi * AE_FACTOR:(wi + lt_size) * AE_FACTOR]
            z_tile, mean, std = encode_latent(vae, crop)
            if acc is None:
                lc = z_tile.shape[1]
                g = _gaussian_weights(lt_size, lt_size, lc, x.device)
                acc = torch.zeros(b, lc, lh, lw, device=x.device, dtype=z_tile.dtype)
                wacc = torch.zeros_like(acc)
            acc[:, :, hi:hi + lt_size, wi:wi + lt_size] += z_tile * g
            wacc[:, :, hi:hi + lt_size, wi:wi + lt_size] += g

    blended = acc / wacc
    out_lh, out_lw = math.ceil(orig_h / AE_FACTOR), math.ceil(orig_w / AE_FACTOR)
    return blended[:, :, :out_lh, :out_lw], mean, std


def tiled_decode_latent(vae, sr_latent: torch.Tensor, latents_mean, latents_std,
                        tile_size: int, tile_overlap: int) -> torch.Tensor:
    """在像素空间解码（3ch）：每个潜空间块恰好对应 8 倍像素块。支持任意 B。"""
    b, _, lh, lw = sr_latent.shape
    lt_size, lt_overlap = vae_tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        return decode_latent(vae, sr_latent, latents_mean, latents_std)

    h_pos = make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = make_tile_grid(lw, lt_size, lt_overlap)

    out_h, out_w = lh * AE_FACTOR, lw * AE_FACTOR
    g = _gaussian_weights(lt_size * AE_FACTOR, lt_size * AE_FACTOR, 3, sr_latent.device)
    acc = torch.zeros(b, 3, out_h, out_w, device=sr_latent.device, dtype=sr_latent.dtype)
    wacc = torch.zeros_like(acc)
    for hi in h_pos:
        for wi in w_pos:
            he, we = hi + lt_size, wi + lt_size
            pix = decode_latent(vae, sr_latent[:, :, hi:he, wi:we], latents_mean, latents_std)
            acc[:, :, hi * AE_FACTOR:he * AE_FACTOR, wi * AE_FACTOR:we * AE_FACTOR] += pix * g
            wacc[:, :, hi * AE_FACTOR:he * AE_FACTOR, wi * AE_FACTOR:we * AE_FACTOR] += g

    return (acc / wacc).clamp(-1, 1)


def encode_dispatch(vae, x: torch.Tensor, vae_tile_size: int, vae_tile_overlap: int):
    """encode_latent；vae_tile_size > 0 时走分块。"""
    if vae_tile_size and vae_tile_size > 0:
        return tiled_encode_latent(vae, x, vae_tile_size, vae_tile_overlap)
    return encode_latent(vae, x)


def decode_dispatch(vae, sr_latent: torch.Tensor, latents_mean, latents_std,
                    vae_tile_size: int, vae_tile_overlap: int) -> torch.Tensor:
    """decode_latent；vae_tile_size > 0 时走分块。"""
    if vae_tile_size and vae_tile_size > 0:
        return tiled_decode_latent(vae, sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)
    return decode_latent(vae, sr_latent, latents_mean, latents_std)
