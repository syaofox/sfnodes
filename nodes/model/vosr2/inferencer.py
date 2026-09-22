"""VOSR2 推理管线（图片批次 / 视频帧批次共用）。

契约（移植自 cswry/VOSR 的 inference_vosr_onestep.py，经 ylchen333/ComfyUI-VOSR2
的 inference.py，Apache-2.0）：双三次预放大 → pad 16 倍数 → VAE 确定性编码 →
DINOv2-L 条件特征 → 一步流匹配去噪 → VAE 解码 → 色彩对齐 → 裁回精确尺寸。

sfnodes 加速项（对齐 TE-Speed-VOSR2，功能等价实现）：
- 分块批量：DiT 瓦片 / DINO 瓦片按 `dit_tile_batch` / `dino_batch` 拼批前向；
  OOM 自动减半降级（VAE 解码 OOM 自动缩小瓦片）。
- 显存策略：`staged` 在 VAE 解码前释放 DiT/DINO 驻留（bundle.prepare_vae_decode）。
- `torch.compile`：DiT 前向包装（见 loader），失败自动回退 eager。
- 视频 DINO 时序缓存：相邻帧瓦片像素签名差低于阈值即复用上一帧特征（CPU 存储），
  每 `cache_refresh` 帧强制刷新；`auto_expand_vae_tile` 按空闲显存扩张 VAE 瓦片。

未分块路径额外 pad 成方形（forward_flexible 断言方形输入），对输出不可见。
"""

import logging
import math

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

from .color_fix import apply_color_alignment
from .settings import VOSR2Settings, normalize_settings
from .sizing import TargetSizeSpec, coerce_target_size
from .tiled_vae import _gaussian_weights, pad_reflect_safe
from .tiling import (
    PAD_MULTIPLE,
    dit_tile_count,
    dit_tile_params,
    dit_tile_positions,
    pad_to_multiple_amount,
)

logger = logging.getLogger(__name__)

# 分块策略为 tiled 但 tile_size=0 时的兜底瓦片边长（像素）
_DEFAULT_TILE_SIZE = 512
# VAE 瓦片自动扩张候选（像素）
_VAE_TILE_CANDIDATES = (1024, 1536, 2048)
# VAE 解码 OOM 降级的最小瓦片边长
_MIN_VAE_TILE = 256


def _is_oom(exc) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _pad_to_multiple(x: torch.Tensor, multiple: int = PAD_MULTIPLE) -> torch.Tensor:
    _, _, h, w = x.shape
    pad_h = pad_to_multiple_amount(h, multiple)
    pad_w = pad_to_multiple_amount(w, multiple)
    return pad_reflect_safe(x, pad_h, pad_w)


def _pad_to_square(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    side = max(h, w)
    return pad_reflect_safe(x, side - h, side - w)


def _generate_noise(shape, seed: int, device, dtype) -> torch.Tensor:
    """单项噪声；本地 CPU Generator，不污染全局 RNG 状态。"""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def _generate_noise_batch(shape_per_item, seed: int, batch_size: int, device, dtype) -> torch.Tensor:
    return torch.stack(
        [_generate_noise(shape_per_item, int(seed) + i, device, dtype) for i in range(batch_size)], dim=0
    )


def _resize_to_target(images_bhwc01: torch.Tensor, size_hw, device) -> torch.Tensor:
    """双三次预缩放到目标尺寸 (h, w)（任意尺寸，非整数倍亦可）。"""
    x = images_bhwc01.movedim(-1, 1).to(device).float()  # BCHW [0, 1]
    return F.interpolate(x, size=tuple(size_hw), mode="bicubic").clamp(0.0, 1.0)


def _chunks(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class DinoTemporalCache:
    """视频帧间 DINO 特征缓存（TE-Speed-VOSR2 的 temporal_cache 语义）。

    键为瓦片位置 (hi, wi)，跨帧保留「上一帧该瓦片的像素签名 + 特征」。
    当前帧瓦片与上一帧签名差 <= threshold 时复用（特征存 CPU，复用时搬回设备）；
    `refresh` > 0 时每 refresh 帧强制重算；尺寸/瓦片几何变化时整体失效。
    """

    def __init__(self, enabled=True, threshold=0.05, refresh=0):
        self.enabled = bool(enabled)
        self.threshold = float(threshold)
        self.refresh = int(refresh)
        self._feats = {}
        self._signatures = {}
        self._geometry = None
        self.hits = 0
        self.misses = 0

    def reset(self):
        self._feats.clear()
        self._signatures.clear()
        self._geometry = None
        self.hits = 0
        self.misses = 0

    def begin_frame(self, geometry):
        """geometry 变化（尺寸/瓦片）时整体失效。"""
        if self._geometry != geometry:
            self._feats.clear()
            self._signatures.clear()
            self._geometry = geometry

    @staticmethod
    def _signature(pixels_bchw: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(pixels_bchw.detach().float().cpu(), (16, 16))

    def reuse(self, key, pixels_bchw: torch.Tensor, slot: int, device, dtype):
        """返回可复用的特征（已搬到 device/dtype）或 None。"""
        if not self.enabled:
            return None
        if self.refresh > 0 and slot % self.refresh == 0:
            return None
        cached = self._feats.get(key)
        prev_sig = self._signatures.get(key)
        if cached is None or prev_sig is None:
            return None
        sig = self._signature(pixels_bchw)
        if torch.max(torch.abs(sig - prev_sig)).item() > self.threshold:
            return None
        self.hits += 1
        return cached.to(device=device, dtype=dtype)

    def store(self, key, pixels_bchw: torch.Tensor, features: torch.Tensor):
        if not self.enabled:
            return
        self.misses += 1
        self._signatures[key] = self._signature(pixels_bchw)
        self._feats[key] = features.detach().to("cpu")


class VOSR2Inferencer:
    """绑定一个已加载 bundle 的推理器（无状态，可复用）。"""

    def __init__(self, bundle):
        self.bundle = bundle

    # ------------------------------------------------------------ 对外入口

    def upscale(
        self,
        images_bhwc01: torch.Tensor,
        target: TargetSizeSpec,
        seed: int,
        settings: VOSR2Settings = None,
        color_alignment: str = "wavelet",
        color_downsample: int = 1,
        tile_size: int = _DEFAULT_TILE_SIZE,
        tile_overlap: int = 32,
        vae_tile_size: int = 1024,
        vae_tile_overlap: int = 32,
        progress: bool = True,
        cache: DinoTemporalCache = None,
        slot_offset: int = 0,
        item_batch: int = 0,
    ) -> torch.Tensor:
        """IMAGE 批次 [B,H,W,C] → 超分结果（逐项 seed+i，逐项色彩对齐）。

        `target` 为目标尺寸规格（`TargetSizeSpec`，也接受数字 = 倍率）：按源形状
        分组逐组解析目标尺寸，同组项共享目标尺寸。
        """
        if images_bhwc01.shape[-1] != 3:
            raise ValueError(f"VOSR2 需要 3 通道 RGB IMAGE，得到 {images_bhwc01.shape[-1]} 通道。")
        if images_bhwc01.shape[0] == 0:
            raise ValueError("VOSR2 输入为空批次（0 张图片/帧）。")
        settings = normalize_settings(settings)
        spec = coerce_target_size(target)
        error = spec.validate()
        if error:
            raise ValueError(f"VOSR2 目标尺寸参数非法: {error}")

        tile_size_eff = self._effective_tile_size(tile_size, settings)
        if tile_size_eff > 0 and tile_overlap >= tile_size_eff:
            raise ValueError(f"tile_overlap ({tile_overlap}) 必须小于 tile_size ({tile_size_eff})。")
        if vae_tile_size > 0 and vae_tile_overlap >= vae_tile_size:
            raise ValueError(
                f"vae_tile_overlap ({vae_tile_overlap}) 必须小于 vae_tile_size ({vae_tile_size})。"
            )

        device = self.bundle.dit_patcher.load_device
        batch_size = item_batch if item_batch and item_batch > 0 else settings.resolved_image_batch()
        self._warn_if_untiled(images_bhwc01, spec, tile_size_eff, vae_tile_size)

        pbar = self._make_progress(images_bhwc01, spec, tile_size_eff, tile_overlap, settings, progress)

        outputs = []
        logged_target = False
        for group_start, group_end in self._shape_groups(images_bhwc01):
            group = images_bhwc01[group_start:group_end]
            src_h, src_w = int(group.shape[1]), int(group.shape[2])
            target_w, target_h, clamped = spec.resolve(src_w, src_h)
            if not logged_target:
                logger.info(
                    "[VOSR2] 目标尺寸: %sx%s → %sx%s (%s%s)",
                    src_w, src_h, target_w, target_h, spec.describe(),
                    "，已钳制到 16~8192" if clamped else "",
                )
                logged_target = True
            resized = _resize_to_target(group, (target_h, target_w), device)
            for chunk_start, chunk in self._item_chunks(resized, batch_size):
                base_seed = seed + group_start + chunk_start
                outputs.append(self._run_chunk(
                    chunk, base_seed, settings, color_alignment, color_downsample,
                    tile_size_eff, tile_overlap, vae_tile_size, vae_tile_overlap, pbar,
                    cache, slot_offset + group_start + chunk_start,
                ))
        # 内部管线为 BCHW，返回 ComfyUI IMAGE 约定的 BHWC
        return torch.cat(outputs, dim=0).movedim(1, -1)

    # ------------------------------------------------------------ 进度

    def _make_progress(self, images, spec, tile_size, tile_overlap, settings, progress):
        if not progress:
            return None
        total = 0
        for start, end in self._shape_groups(images):
            h, w = int(images[start].shape[0]), int(images[start].shape[1])
            target_w, target_h, _ = spec.resolve(w, h)
            padded_h = target_h + pad_to_multiple_amount(target_h)
            padded_w = target_w + pad_to_multiple_amount(target_w)
            tiled = tile_size > 0 and settings.use_tiling(padded_h, padded_w, tile_size)
            per_item = dit_tile_count(padded_h, padded_w, tile_size, tile_overlap) if tiled else 1
            total += (end - start) * per_item
        return comfy.utils.ProgressBar(max(total, 1))

    @staticmethod
    def _shape_groups(images):
        """按 (H, W) 分组，返回连续区间（同形状项才能拼批）。"""
        groups = []
        start = 0
        for i in range(1, images.shape[0] + 1):
            if i == images.shape[0] or tuple(images[i].shape[:2]) != tuple(images[start].shape[:2]):
                groups.append((start, i))
                start = i
        return groups

    @staticmethod
    def _item_chunks(resized, batch_size):
        for i in range(0, resized.shape[0], batch_size):
            yield i, resized[i:i + batch_size]

    # ------------------------------------------------------------ 单批主流程

    def _run_chunk(self, resized01, base_seed, settings, color_alignment, color_downsample,
                   tile_size, tile_overlap, vae_tile_size, vae_tile_overlap, pbar,
                   cache, slot_base) -> torch.Tensor:
        bundle = self.bundle
        b, _, h, w = resized01.shape
        device = bundle.dit_patcher.load_device
        padded01 = _pad_to_multiple(resized01, PAD_MULTIPLE)
        padded_pm1 = padded01 * 2.0 - 1.0
        padded_h, padded_w = padded01.shape[2], padded01.shape[3]

        lq_latent, latents_mean, latents_std = bundle.encode(
            padded_pm1, vae_tile_size, vae_tile_overlap, amp=settings.vae_encode_amp
        )
        _, lc, lh, lw = lq_latent.shape

        tiled = tile_size > 0 and settings.use_tiling(padded_h, padded_w, tile_size)
        if tiled:
            sr_latent = self._denoise_tiled(
                lq_latent, padded01, base_seed, tile_size, tile_overlap, settings, pbar,
                cache, slot_base,
            )
        else:
            sr_latent = self._denoise_full_frame(
                lq_latent, padded01, base_seed, settings, pbar, cache, slot_base,
            )

        # staged 策略：VAE 解码前释放 DiT/DINO 驻留
        bundle.prepare_vae_decode()
        vae_tile_eff = self._effective_vae_tile(vae_tile_size, padded_h, padded_w, settings)
        decoded_pm1 = self._decode_with_fallback(
            sr_latent, latents_mean, latents_std, vae_tile_eff, vae_tile_overlap,
            settings.vae_encode_amp, padded_h, padded_w,
        )
        decoded01 = (decoded_pm1[:, :, :h, :w].clamp(-1.0, 1.0) + 1.0) / 2.0

        # 逐项色彩对齐（避免整批一次性 CPU 分配；参考图留在 CPU）
        reference = resized01.detach().cpu()
        aligned = [
            apply_color_alignment(decoded01[i:i + 1].cpu(), reference[i:i + 1], color_alignment, color_downsample)
            for i in range(b)
        ]
        return torch.cat(aligned, dim=0)

    # ------------------------------------------------------------ 去噪路径

    def _denoise_full_frame(self, lq_latent, padded01, base_seed, settings, pbar,
                            cache=None, slot_base=0) -> torch.Tensor:
        bundle = self.bundle
        b = lq_latent.shape[0]
        _, _, lh, lw = lq_latent.shape
        lq_sq = _pad_to_square(lq_latent)
        venc_fea = [self._collect_frame_features(padded01, settings, cache, slot_base)]
        noise = _generate_noise_batch(lq_sq.shape[1:], base_seed, b, lq_sq.device, lq_sq.dtype)
        sr = bundle.denoise_one_step(lq_sq, noise, venc_fea)[:, :, :lh, :lw]
        self._tick(pbar, b)
        return sr

    def _denoise_tiled(self, lq_latent, padded01, base_seed, tile_size, tile_overlap,
                       settings, pbar, cache, slot_base) -> torch.Tensor:
        bundle = self.bundle
        b, lc, lh, lw = lq_latent.shape
        device = lq_latent.device
        padded_h, padded_w = padded01.shape[2], padded01.shape[3]
        positions = dit_tile_positions(padded_h, padded_w, tile_size, tile_overlap)
        lt_size, _ = dit_tile_params(tile_size, tile_overlap, lh, lw)

        tile_venc = self._collect_tile_features(
            padded01, positions, lt_size, settings, cache, slot_base,
        )
        noise = _generate_noise_batch(lq_latent.shape[1:], base_seed, b, device, lq_latent.dtype)
        z = noise
        # (lc, lt, lt)：逐项累加时不能再带 B 维（in-place 不允许 rhs 广播出更大形状）
        g_weight = _gaussian_weights(lt_size, lt_size, lc, device)[0]

        # 展平 (item, tile) 对：同几何瓦片拼批前向，OOM 自动减半降级
        pairs = [(i, hi, wi) for i in range(b) for (hi, wi) in positions]
        batch = settings.resolved_dit_tile_batch()
        while True:
            try:
                u_acc = torch.zeros_like(lq_latent)
                w_acc = torch.zeros_like(lq_latent)
                for chunk in _chunks(pairs, batch):
                    inp = torch.stack([
                        torch.cat([lq_latent[i, :, hi:hi + lt_size, wi:wi + lt_size],
                                   z[i, :, hi:hi + lt_size, wi:wi + lt_size]], dim=0)
                        for (i, hi, wi) in chunk
                    ], dim=0)
                    feats = torch.stack([
                        tile_venc[(hi, wi)][i] for (i, hi, wi) in chunk
                    ], dim=0)
                    u = bundle.dit_velocity(inp, 1.0, 0.0, [feats])
                    for row, (i, hi, wi) in enumerate(chunk):
                        u_acc[i, :, hi:hi + lt_size, wi:wi + lt_size] += u[row] * g_weight
                        w_acc[i, :, hi:hi + lt_size, wi:wi + lt_size] += g_weight
                    self._tick(pbar, len(chunk))
                return z - u_acc / w_acc
            except Exception as exc:
                if not _is_oom(exc) or batch <= 1:
                    raise
                new_batch = max(1, batch // 2)
                logger.warning(
                    "[VOSR2] DiT tile batch %s ran out of memory, falling back to %s", batch, new_batch
                )
                bundle.clear_staged()
                batch = new_batch

    # ------------------------------------------------------------ DINO 特征

    def _collect_frame_features(self, padded01, settings, cache, slot_base):
        """整图路径的 DINO 特征（可命中时序缓存）。返回 (B, N, D)。"""
        b = padded01.shape[0]
        device = padded01.device
        key = ("full",)
        if cache is not None:
            cache.begin_frame((padded01.shape[2], padded01.shape[3], 0, None))

        feats = [None] * b
        reuse_flags = [False] * b
        if cache is not None:
            for i in range(b):
                reused = cache.reuse(key, padded01[i:i + 1], slot_base + i, device, self._feature_dtype())
                if reused is not None:
                    feats[i] = reused[0]
                    reuse_flags[i] = True

        miss = [i for i in range(b) if not reuse_flags[i]]
        if miss:
            computed = self._vision_features_with_fallback(
                padded01[miss], settings.resolved_dino_batch()
            )
            for row, i in enumerate(miss):
                feats[i] = computed[row]
                if cache is not None:
                    cache.store(key, padded01[i:i + 1], computed[row:row + 1])
        return torch.stack(feats, dim=0)

    def _collect_tile_features(self, padded01, positions, lt_size, settings, cache, slot_base):
        """逐瓦片 DINO 特征（可拼批、可命中时序缓存）。返回 {(hi,wi): (B, N, D)}。"""
        bundle = self.bundle
        b = padded01.shape[0]
        device = padded01.device
        dino_batch = settings.resolved_dino_batch()
        result = {}

        # 全帧共享的缓存帧槽：同批内每项一帧（图片批次无 cache）
        slots = [slot_base + i for i in range(b)]
        if cache is not None:
            cache.begin_frame((padded01.shape[2], padded01.shape[3], lt_size, positions))

        for (hi, wi) in positions:
            ph_s, pw_s = hi * 8, wi * 8
            ph_e = min((hi + lt_size) * 8, padded01.shape[2])
            pw_e = min((wi + lt_size) * 8, padded01.shape[3])
            crop = padded01[:, :, ph_s:ph_e, pw_s:pw_e]
            key = (hi, wi)

            reuse_flags = [False] * b
            feats = [None] * b
            if cache is not None:
                for i in range(b):
                    reused = cache.reuse(key, crop[i:i + 1], slots[i], device, self._feature_dtype())
                    if reused is not None:
                        feats[i] = reused[0]
                        reuse_flags[i] = True

            miss = [i for i in range(b) if not reuse_flags[i]]
            for chunk in _chunks(miss, dino_batch):
                sub = crop[chunk]
                computed = self._vision_features_with_fallback(sub, dino_batch)
                for row, i in enumerate(chunk):
                    feats[i] = computed[row]
                    if cache is not None:
                        cache.store(key, crop[i:i + 1], computed[row:row + 1])
            result[key] = torch.stack(feats, dim=0)

        if cache is not None:
            logger.debug(
                "[VOSR2] DINO temporal cache: hits=%s misses=%s", cache.hits, cache.misses
            )
        return result

    def _feature_dtype(self):
        return self.bundle.dino_patcher.model.pos_embed.dtype

    def _vision_features_with_fallback(self, crops_bchw, batch_size):
        """DINO 前向 + OOM 减半降级。crops_bchw: (n,3,h,w) → (n,N,D)。"""
        while True:
            try:
                out = []
                for chunk in _chunks(list(range(crops_bchw.shape[0])), batch_size):
                    out.append(self.bundle.vision_features(crops_bchw[chunk])[0])
                return torch.cat(out, dim=0)
            except Exception as exc:
                if not _is_oom(exc) or batch_size <= 1:
                    raise
                new_batch = max(1, batch_size // 2)
                logger.warning(
                    "[VOSR2] DINO batch %s ran out of memory, falling back to %s", batch_size, new_batch
                )
                self.bundle.clear_staged()
                batch_size = new_batch

    # ------------------------------------------------------------ VAE 解码降级

    def _decode_with_fallback(self, sr_latent, latents_mean, latents_std, vae_tile_size,
                              vae_tile_overlap, amp, padded_h, padded_w):
        size = vae_tile_size
        while True:
            try:
                return self.bundle.decode(
                    sr_latent, latents_mean, latents_std, size, vae_tile_overlap, amp=amp
                )
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                new_size = self._fallback_vae_tile(size, padded_h, padded_w)
                if new_size is None:
                    raise
                logger.warning(
                    "[VOSR2] VAE tile %s ran out of memory, falling back to %s", size, new_size
                )
                self.bundle.clear_staged()
                size = new_size

    @staticmethod
    def _fallback_vae_tile(size, padded_h, padded_w):
        """VAE 瓦片 OOM 降级：先减半，整图解码（0）则改用 1024 分块。"""
        if size == 0:
            if min(padded_h, padded_w) > _VAE_TILE_CANDIDATES[0]:
                return _VAE_TILE_CANDIDATES[0]
            return None
        new_size = size // 2
        return new_size if new_size >= _MIN_VAE_TILE else None

    # ------------------------------------------------------------ 瓦片尺寸

    @staticmethod
    def _effective_tile_size(tile_size, settings):
        if tile_size and tile_size > 0:
            return tile_size
        if settings.tile_strategy == "tiled":
            return _DEFAULT_TILE_SIZE
        return 0

    @staticmethod
    def _warn_if_untiled(images, spec, tile_size, vae_tile_size):
        """超过原生 512px 未分块 / 整图 VAE 解码 / 缩小 的告警（上游 README 口径）。"""
        src_h, src_w = int(images.shape[1]), int(images.shape[2])
        target_w, target_h, _ = spec.resolve(src_w, src_h)
        if tile_size <= 0 and (target_h > 512 or target_w > 512):
            logger.warning(
                "[VOSR2] 输出 %sx%s 超过原生 512px 且 tile_size=0：质量可能下降，建议 tile_size=512",
                target_w, target_h,
            )
        if vae_tile_size <= 0 and (target_h > 1024 or target_w > 1024):
            logger.warning(
                "[VOSR2] 输出 %sx%s 且 vae_tile_size=0：整图 VAE 解码易 OOM，建议 vae_tile_size=1024",
                target_w, target_h,
            )
        if target_h < src_h or target_w < src_w:
            logger.warning(
                "[VOSR2] 目标 %sx%s 小于源图 %sx%s（缩小属缩放+修复，VOSR2 未按此训练，质量未验证）",
                target_w, target_h, src_w, src_h,
            )
        if max(target_h, target_w) > 4096:
            logger.warning(
                "[VOSR2] 目标 %sx%s 超过 4096px：显存占用与耗时显著上升，建议开启 tile_size/vae_tile_size",
                target_w, target_h,
            )

    def _effective_vae_tile(self, vae_tile_size, padded_h, padded_w, settings):
        """auto_expand_vae_tile：空闲显存允许时把 VAE 瓦片扩到更大（更少块、更快）。"""
        if not settings.auto_expand_vae_tile or vae_tile_size <= 0:
            return vae_tile_size
        try:
            device = self.bundle.vae_patcher.load_device
            free = comfy.model_management.get_free_memory(device)
        except Exception:
            return vae_tile_size
        # 保守预算：像素 × 3ch × 4B × 16 的中间激活
        budget_pixels = free / (3 * 4 * 16)
        max_side = int(math.sqrt(max(budget_pixels, 1.0)))
        limit = max(padded_h, padded_w)
        candidates = [s for s in _VAE_TILE_CANDIDATES if s <= max_side and s <= limit and s > vae_tile_size]
        return max(candidates) if candidates else vae_tile_size

    # ------------------------------------------------------------ 工具

    @staticmethod
    def _tick(pbar, n=1):
        if pbar is None:
            return
        try:
            pbar.update_absolute(min(pbar.current + n, pbar.total))
        except Exception:
            pass
