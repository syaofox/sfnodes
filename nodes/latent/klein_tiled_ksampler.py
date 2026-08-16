"""SFKleinTiledKSampler - 分块 KSampler（复刻自 Comfy-SZ-KleinKSampler 的
SZ_KleinTiledKSampler，适用于 FLUX.2 Klein 模型）。

主要用途：图像放大修复和细节增强。核心功能：
  1. 外部接入 latent_blend 作为全局引导（归一化到 [-1, 1] 后按 blend_strength
     混入每个 tile 的噪声）
  2. 生成空间连续的全局噪声图（整图一张噪声，tile 各自裁剪，跨 tile 噪声连续）
  3. 分块对应采样：相同尺寸的 tile 两两合并成 batch=2 并行采样，加速约 40-50%
  4. overlap 羽化混合写回（feather weight mask，重叠区按距离加权融合）
  5. 按 latent_blend 内容丰富度排序（内容多的 tile 优先处理）
  6. 可选对齐原始图像的色彩统计量（均值+标准差），防止分块采样后饱和度/明度/
     对比度漂移

输入说明：
  - latent_blend：全局引导 latent（如参考图编码），会被缩放到输入 latent 尺寸
    并归一化到 [-1, 1]；tile 的初始噪声 = 全局噪声 * (1 - blend_strength) +
    blend * blend_strength
  - tile_width / tile_height：以像素为单位，内部按 /8 折算到 latent 空间
  - overlap：tile 之间的重叠（像素），重叠区用羽化权重混合，避免接缝
  - blend_strength：latent_blend 引导强度（0 = 纯全局噪声，1 = 完全用 blend）
  - color_preserve：色彩统计量对齐强度（0 = 不对齐，1 = 完全对齐原图统计量）

复刻说明：行为与源插件保持 1:1（含 `_scale_conditioning_refs` 辅助方法——
源插件中当前未被调用，保留以对齐源实现）。
"""

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

_CATEGORY = "sfnodes/latent"


class SFKleinTiledKSampler:
    DESCRIPTION = (
        "分块 KSampler（复刻自 Comfy-SZ-KleinKSampler，适用于 FLUX.2 Klein "
        "模型），用于图像放大修复和细节增强。把整图切成带 overlap 的 tile 逐个"
        "采样，再用羽化权重混合写回，避免接缝；相同尺寸的 tile 两两合并成 "
        "batch=2 并行采样（约 40-50% 加速）。\n\n"
        "核心机制：\n"
        "  1. latent_blend 全局引导：参考图编码接入 latent_blend，归一化到 "
        "[-1, 1] 后按 blend_strength 混入每个 tile 的初始噪声；\n"
        "  2. 空间连续的全局噪声图：整图生成一张噪声，tile 各自裁剪，跨 tile "
        "噪声连续；\n"
        "  3. 按 latent_blend 内容丰富度排序，内容多的 tile 优先处理；\n"
        "  4. color_preserve 对齐原始图像色彩统计量（均值+标准差），防止分块"
        "采样后饱和度/明度/对比度漂移。\n\n"
        "tile_width / tile_height 以像素为单位（内部按 /8 折算到 latent "
        "空间）；overlap 越大接缝越少但 tile 越多越慢。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "用于采样的模型（如 FLUX.2 Klein）。",
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": "正面条件：描述要在图像中呈现的属性。",
                    },
                ),
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": "负面条件：描述要从图像中排除的属性。",
                    },
                ),
                "latent_image": (
                    "LATENT",
                    {
                        "tooltip": "要分块采样的输入 latent。放大修复/细节增强时接入"
                        "VAE 编码的参考图 latent（与 latent_blend 可同源或不同源）。",
                    },
                ),
                "latent_blend": (
                    "LATENT",
                    {
                        "tooltip": "全局引导 latent（如参考图编码）。会被缩放到输入"
                        " latent 尺寸并归一化到 [-1, 1]，再按 blend_strength 混入"
                        "每个 tile 的初始噪声。",
                    },
                ),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "全局噪声种子：整图生成一张空间连续的噪声，tile "
                    "各自裁剪，跨 tile 噪声连续。同一种子结果可复现。",
                }),
                "steps": ("INT", {
                    "default": 4, "min": 1, "max": 100,
                    "tooltip": "每个 tile 的采样步数。",
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "提示词引导强度（CFG）。FLUX 系模型通常用 1.0。",
                }),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "tooltip": "采样算法，影响生成结果的质量、速度与风格。",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "tooltip": "噪声逐步移除（去噪）的调度方式。",
                    },
                ),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "去噪强度。较低的值保留输入 latent 的结构，用于"
                    "图生图/局部重绘。",
                }),
                "tile_width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 8,
                    "tooltip": "tile 宽度（像素），内部按 /8 折算到 latent 空间。",
                }),
                "tile_height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 8,
                    "tooltip": "tile 高度（像素），内部按 /8 折算到 latent 空间。",
                }),
                "overlap": ("INT", {
                    "default": 128, "min": 0, "max": 512, "step": 8,
                    "tooltip": "相邻 tile 的重叠量（像素）。重叠区用羽化权重混合，"
                    "避免接缝。",
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "latent_blend 引导强度：tile 噪声 = 全局噪声 × "
                    "(1 - 强度) + blend × 强度。",
                }),
                "color_preserve": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "色彩统计量对齐强度（均值+标准差对齐原图，防止"
                    "分块采样后饱和度/明度/对比度漂移）。",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = (
        "分块采样并羽化融合后的完整 latent。",
    )
    FUNCTION = "sample"
    CATEGORY = _CATEGORY

    # ──────────────────────────────────────────────────────────────────────

    def _get_tile_positions(self, H, W, tile_h, tile_w, overlap):
        tile_h = min(tile_h, H)
        tile_w = min(tile_w, W)
        if tile_h >= H and tile_w >= W:
            return [(0, 0, H, W)]
        stride_h = max(1, tile_h - overlap)
        stride_w = max(1, tile_w - overlap)
        positions = []
        seen = set()
        y = 0
        while True:
            y0 = min(y, H - tile_h)
            x = 0
            while True:
                x0 = min(x, W - tile_w)
                key = (y0, x0)
                if key not in seen:
                    seen.add(key)
                    positions.append((y0, x0, tile_h, tile_w))
                if x0 + tile_w >= W:
                    break
                x += stride_w
            if y0 + tile_h >= H:
                break
            y += stride_h
        return positions

    def _sort_tiles_by_content(self, tile_positions, blend_up):
        """按 latent_blend 内容丰富度排序，内容多的 tile 优先处理。"""
        scores = []
        for (y0, x0, th, tw) in tile_positions:
            region = blend_up[:, :, y0:y0+th, x0:x0+tw]
            scores.append(float(region.var()))
        order = sorted(range(len(tile_positions)),
                       key=lambda i: scores[i], reverse=True)
        return [tile_positions[i] for i in order]

    def _scale_conditioning_refs(self, conditioning, aH, aW):
        scaled_cond = []
        for cond_pair in conditioning:
            cond_dict = cond_pair[1].copy()
            if "reference_latents" in cond_dict:
                cond_dict["reference_latents"] = [
                    F.interpolate(ref, size=(aH, aW),
                                  mode="bilinear", align_corners=False)
                    for ref in cond_dict["reference_latents"]
                ]
            scaled_cond.append([cond_pair[0], cond_dict])
        return scaled_cond

    def _crop_conditioning_refs(self, conditioning, y0, x0, th, tw):
        """按 tile 位置裁剪 reference_latents。"""
        cropped_cond = []
        for cond_pair in conditioning:
            cond_dict = cond_pair[1].copy()
            if "reference_latents" in cond_dict:
                cond_dict["reference_latents"] = [
                    ref[:, :, y0:y0+th, x0:x0+tw].clone()
                    for ref in cond_dict["reference_latents"]
                ]
            cropped_cond.append([cond_pair[0], cond_dict])
        return cropped_cond

    def _merge_conditioning_refs(self, cond1, cond2):
        """两个 tile 的 reference_latents 在 batch 维度合并。"""
        merged = []
        for cp1, cp2 in zip(cond1, cond2):
            cond_dict = cp1[1].copy()
            if "reference_latents" in cond_dict:
                refs1 = cp1[1]["reference_latents"]
                refs2 = cp2[1]["reference_latents"]
                cond_dict["reference_latents"] = [
                    torch.cat([r1, r2], dim=0)
                    for r1, r2 in zip(refs1, refs2)
                ]
            merged.append([cp1[0], cond_dict])
        return merged

    def _make_weight_mask(self, h, w, device):
        wy = torch.arange(h, dtype=torch.float32, device=device)
        wy = (torch.min(wy, h - 1 - wy) + 1.0)
        wx = torch.arange(w, dtype=torch.float32, device=device)
        wx = (torch.min(wx, w - 1 - wx) + 1.0)
        weight = (wy.unsqueeze(1) * wx.unsqueeze(0))
        weight = weight / weight.max()
        return weight.unsqueeze(0).unsqueeze(0)

    def _make_callback(self, pbar, previewer, total_steps):
        def callback(step, x0, x, total):
            preview_bytes = None
            if previewer:
                try:
                    preview_bytes = previewer.decode_latent_to_preview_image(
                        "JPEG", x0[:1]
                    )
                except Exception:
                    pass
            pbar.update_absolute(step + 1, total_steps, preview_bytes)
        return callback

    def _process_tile(self, m, positive, negative, samples,
                      global_noise, blend_up,
                      y0, x0, th, tw, B,
                      blend_strength,
                      steps, cfg, sampler_name, scheduler, denoise, seed,
                      previewer, device):
        """处理单个 tile。"""
        tile_noise  = global_noise[:, :, y0:y0+th, x0:x0+tw].clone()
        blend_tile  = blend_up    [:, :, y0:y0+th, x0:x0+tw]

        base_weight = max(0.0, 1.0 - blend_strength)
        tile_noise  = tile_noise * base_weight + blend_tile * blend_strength

        tile_ref      = samples[:, :, y0:y0+th, x0:x0+tw].clone().cpu()
        tile_positive = self._crop_conditioning_refs(positive, y0, x0, th, tw)
        tile_negative = self._crop_conditioning_refs(negative, y0, x0, th, tw)

        inner_pbar    = comfy.utils.ProgressBar(steps)
        tile_callback = self._make_callback(inner_pbar, previewer, steps)

        tile_result = comfy.sample.sample(
            m, tile_noise, steps, cfg, sampler_name, scheduler,
            tile_positive, tile_negative, tile_ref,
            denoise=denoise, seed=seed, callback=tile_callback,
        )
        return tile_result.to(device)

    def _process_tile_pair(self, m, positive, negative, samples,
                           global_noise, blend_up,
                           y0a, x0a, tha, twa,
                           y0b, x0b, thb, twb,
                           B,
                           blend_strength,
                           steps, cfg, sampler_name, scheduler, denoise, seed,
                           previewer, device):
        """两个相同尺寸的 tile 合并成 batch=2 一起处理。"""
        def prep_noise(y0, x0, th, tw):
            noise  = global_noise[:, :, y0:y0+th, x0:x0+tw].clone()
            blend  = blend_up    [:, :, y0:y0+th, x0:x0+tw]
            base_weight = max(0.0, 1.0 - blend_strength)
            return noise * base_weight + blend * blend_strength

        noise_a    = prep_noise(y0a, x0a, tha, twa)
        noise_b    = prep_noise(y0b, x0b, thb, twb)
        tile_noise = torch.cat([noise_a, noise_b], dim=0)

        ref_a    = samples[:, :, y0a:y0a+tha, x0a:x0a+twa].clone().cpu()
        ref_b    = samples[:, :, y0b:y0b+thb, x0b:x0b+twb].clone().cpu()
        tile_ref = torch.cat([ref_a, ref_b], dim=0)

        pos_a = self._crop_conditioning_refs(positive, y0a, x0a, tha, twa)
        pos_b = self._crop_conditioning_refs(positive, y0b, x0b, thb, twb)
        neg_a = self._crop_conditioning_refs(negative, y0a, x0a, tha, twa)
        neg_b = self._crop_conditioning_refs(negative, y0b, x0b, thb, twb)
        tile_positive = self._merge_conditioning_refs(pos_a, pos_b)
        tile_negative = self._merge_conditioning_refs(neg_a, neg_b)

        inner_pbar    = comfy.utils.ProgressBar(steps)
        tile_callback = self._make_callback(inner_pbar, previewer, steps)

        pair_result = comfy.sample.sample(
            m, tile_noise, steps, cfg, sampler_name, scheduler,
            tile_positive, tile_negative, tile_ref,
            denoise=denoise, seed=seed, callback=tile_callback,
        ).to(device)

        return pair_result[:B], pair_result[B:]

    def _match_color_stats(self, result, original, strength):
        """
        对齐生成结果和原始图像的色彩统计量（均值+标准差）
        防止分块采样后饱和度/明度/对比度漂移
        strength=1.0 完全对齐，strength=0.0 不对齐
        """
        if strength <= 0.0:
            return result
        matched = result.clone()
        for c in range(result.shape[1]):
            orig_mean = original[:, c].mean()
            orig_std  = original[:, c].std()
            res_mean  = result[:, c].mean()
            res_std   = result[:, c].std()
            if res_std > 1e-8:
                normalized = (result[:, c] - res_mean) / res_std
                adjusted   = normalized * orig_std + orig_mean
                matched[:, c] = result[:, c] * (1.0 - strength) + adjusted * strength
        return matched

    # ──────────────────────────────────────────────────────────────────────

    def sample(self, model, positive, negative, latent_image, latent_blend,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               tile_width, tile_height, overlap,
               blend_strength, color_preserve):

        device  = comfy.model_management.get_torch_device()
        samples = latent_image["samples"].clone().to(device)
        B, C, H, W = samples.shape

        tile_h = max(1, tile_height // 8)
        tile_w = max(1, tile_width  // 8)
        ovlp   = max(1, overlap     // 8)

        previewer = latent_preview.get_previewer(device, model.model.latent_format)

        # ── 处理 latent_blend ────────────────────────────────────────────
        b = latent_blend["samples"].to(device)
        if b.shape[0] != B:
            b = b.expand(B, -1, -1, -1).clone()
        if b.shape[2] != H or b.shape[3] != W:
            b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
        b_min = b.min(); b_max = b.max()
        if b_max - b_min > 1e-8:
            b = (b - b_min) / (b_max - b_min) * 2.0 - 1.0
        blend_up = b

        # ── 全局噪声图 ────────────────────────────────────────────────────
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        global_noise = torch.randn((B, C, H, W), generator=gen).to(device)

        # ── 规划 tile，按内容排序 ─────────────────────────────────────────
        tile_positions = self._get_tile_positions(H, W, tile_h, tile_w, ovlp)
        tile_positions = self._sort_tiles_by_content(tile_positions, blend_up)
        total_tiles    = len(tile_positions)
        print(f"[SFKleinTiledKSampler] total tiles: {total_tiles}")

        # ── 分块对应采样（两两并行） ──────────────────────────────────────
        result     = torch.zeros((B, C, H, W), device=device)
        weight_map = torch.zeros((B, 1, H, W), device=device)
        outer_pbar = comfy.utils.ProgressBar(total_tiles)

        idx = 0
        while idx < total_tiles:
            y0a, x0a, tha, twa = tile_positions[idx]

            if idx + 1 < total_tiles:
                y0b, x0b, thb, twb = tile_positions[idx + 1]
                same_size = (tha == thb and twa == twb)
            else:
                same_size = False

            if same_size and B == 1:
                result_a, result_b = self._process_tile_pair(
                    model, positive, negative, samples,
                    global_noise, blend_up,
                    y0a, x0a, tha, twa,
                    y0b, x0b, thb, twb,
                    B, blend_strength,
                    steps, cfg, sampler_name, scheduler, denoise, seed,
                    previewer, device
                )
                weight_a = self._make_weight_mask(tha, twa, device)
                weight_b = self._make_weight_mask(thb, twb, device)
                result    [:, :, y0a:y0a+tha, x0a:x0a+twa] += result_a * weight_a
                weight_map[:, :, y0a:y0a+tha, x0a:x0a+twa] += weight_a
                result    [:, :, y0b:y0b+thb, x0b:x0b+twb] += result_b * weight_b
                weight_map[:, :, y0b:y0b+thb, x0b:x0b+twb] += weight_b
                outer_pbar.update_absolute(idx + 2, total_tiles, None)
                idx += 2
            else:
                tile_result = self._process_tile(
                    model, positive, negative, samples,
                    global_noise, blend_up,
                    y0a, x0a, tha, twa, B,
                    blend_strength,
                    steps, cfg, sampler_name, scheduler, denoise, seed,
                    previewer, device
                )
                weight = self._make_weight_mask(tha, twa, device)
                result    [:, :, y0a:y0a+tha, x0a:x0a+twa] += tile_result * weight
                weight_map[:, :, y0a:y0a+tha, x0a:x0a+twa] += weight
                outer_pbar.update_absolute(idx + 1, total_tiles, None)
                idx += 1

        result = result / weight_map.clamp(min=1e-8)

        # ── 色彩统计对齐（防止饱和度漂移）────────────────────────────────
        if color_preserve > 0.0:
            result = self._match_color_stats(result, samples, color_preserve)

        return ({"samples": result.cpu()},)
