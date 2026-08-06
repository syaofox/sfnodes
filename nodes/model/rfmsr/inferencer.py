"""RFMSR 推理器 — Residual Flow Matching 反向积分（SD2.1 VAE）

移植自官方 infer_rfmsr.py（去掉 CLI/yaml 依赖，改为直接传路径/参数）：

Flow path: x_t = z_hr + t*(z_lr - z_hr) + t*sigma*epsilon
Reverse integration: t=1 (LR+noise) → t=0 (HR)
"""

import gc
import math
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from safetensors.torch import load_file as safe_load

from .rfmsr import create_rfmsr
from .dinov2_encoder import create_dinov2_encoder
from .color_fix import apply_color_fix
from ....sf_utils.logger import get_logger

logger = get_logger(__name__)


class RFMSRInferencer:

    def __init__(self):
        pass

    # ---- Model loading ----

    def load(self, rfmsr_path, vae_path=None, torch_cache_dir=None, device="cuda"):
        """Load SD2.1 VAE + RFMSR + DINOv2."""
        from diffusers import AutoencoderKL

        self._device = str(device)

        print(f"Loading SD2.1 VAE from {vae_path} -> {device}...")
        self.ae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
        self.ae = self.ae.to(device).eval()
        self.ae.requires_grad_(False)
        # 官方脚本只对 RFMSR 反向流分块，VAE 编解码仍是整图（fp32），大图在小显存
        # 卡上会 OOM。开启 diffusers 官方 tiling 让 encode/decode 自动分块处理。
        self.ae.enable_tiling()
        print(f"  VAE scaling_factor: {self.ae.config.scaling_factor}")

        print(f"Loading RFMSR from {rfmsr_path} ...")
        self.rfmsr = create_rfmsr()
        sd = safe_load(rfmsr_path)
        # VOSR checkpoint: raw DiT params (no prefix) → RFMSR expects "dit." prefix
        sd.pop("ema_scale", None)
        sd = {"dit." + k if not k.startswith("dit.") else k: v for k, v in sd.items()}
        missing, unexpected = self.rfmsr.load_state_dict(sd, strict=False)
        self.rfmsr = self.rfmsr.to(device, dtype=torch.float32)
        self.rfmsr.eval()
        self.rfmsr.dit.use_checkpoint = False
        n = sum(p.numel() for p in self.rfmsr.parameters()) / 1e6
        print(f"  Params: {n:.2f}M")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        print("Loading DINOv2 encoder ...")
        self.venc = create_dinov2_encoder(torch_cache_dir=torch_cache_dir, device=device)
        print("  DINOv2: loaded")

        print("All models loaded.")

    def unload(self):
        """释放全部模型引用与显存（配合类级 LRU 缓存：切换模型/手动清理时调用）。"""
        for attr in ("rfmsr", "ae", "venc"):
            setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ---- VAE encode/decode ----

    def vae_encode(self, img: torch.Tensor) -> torch.Tensor:
        """img [-1,1] → SD2.1 latent [B,4,H,W] (scaled)."""
        return self.ae.encode(img.float()).latent_dist.sample() * self.ae.config.scaling_factor

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """SD2.1 latent [B,4,H,W] (scaled) → pixel [0,1]."""
        latent = latent / self.ae.config.scaling_factor
        img = self.ae.decode(latent).sample
        return torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)

    # ---- Tile helpers ----

    @staticmethod
    def _make_tile_grid(length: int, tile: int, stride: int) -> list[tuple[int, int]]:
        """Return (start, end) tile positions covering the entire dimension."""
        if length <= tile:
            return [(0, length)]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] + tile < length:
            positions.append(length - tile)
        return [(p, p + tile) for p in sorted(set(positions))]

    @staticmethod
    def _gaussian_weights(tile_h: int, tile_w: int, channels: int, device: torch.device) -> torch.Tensor:
        """2D Gaussian blend weights, OpenCV adaptive sigma."""
        def _kernel_1d(ksize):
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
            if ksize % 2 == 0:
                kernel = cv2.getGaussianKernel(ksize=ksize + 1, sigma=sigma, ktype=cv2.CV_64F)
                kernel = kernel[1:, ]
            else:
                kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_64F)
            return kernel

        kernel_h = _kernel_1d(tile_h)       # (H, 1)
        kernel_w = _kernel_1d(tile_w)       # (W, 1)
        w = np.matmul(kernel_h, kernel_w.T) # (H, W)
        w = torch.from_numpy(w).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return w.to(device).expand(1, channels, -1, -1)

    # ---- Reverse integration (full image) ----

    @torch.no_grad()
    def reverse_flow(self, z_lr: torch.Tensor, steps: int = 28,
                     flow_sigma: float = 1.0, seed: int = 42,
                     lr_pixel=None) -> torch.Tensor:
        """RFMSR reverse flow integration: t=1 → t=0."""
        device = z_lr.device
        B, C, H, W = z_lr.shape
        use_amp = self._device.startswith("cuda")

        # DINOv2: full image once
        venc_fea = None
        if self.venc is not None and lr_pixel is not None:
            venc_fea = self.venc(lr_pixel.float())

        # Timesteps: 1.0 → 0.0
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # Initial state: LR latent + noise
        generator = torch.Generator(device=device).manual_seed(seed)
        x = z_lr + flow_sigma * torch.randn(B, C, H, W, generator=generator, device=device)

        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext():
            for t_curr, t_prev in tqdm(step_pairs, desc="RFMSR",
                                       total=len(step_pairs), leave=False):
                t_batch = torch.full((B,), t_curr, device=device)
                dt = t_prev - t_curr

                v = self.rfmsr(x, t_batch, z_lr, venc_fea=venc_fea).float()
                x = x + dt * v

        return x

    # ---- Reverse integration (tiled, VOSR-style per-step blending) ----

    @torch.no_grad()
    def reverse_flow_tiled(self, z_lr: torch.Tensor, steps: int = 28,
                           flow_sigma: float = 1.0, seed: int = 42,
                           lt_size: int = 64, lt_stride: int = 32,
                           lr_pixel=None) -> torch.Tensor:
        """Per-step tiled velocity prediction with Gaussian-weighted blending back to full latent."""
        device = z_lr.device
        B, C, H, W = z_lr.shape
        use_amp = self._device.startswith("cuda")
        AE_FACTOR = 8

        # Tile grid
        h_tiles = self._make_tile_grid(H, lt_size, lt_stride)
        w_tiles = self._make_tile_grid(W, lt_size, lt_stride)

        # Per-tile DINOv2: pre-compute
        tile_venc = {}
        use_venc = self.venc is not None and lr_pixel is not None
        if use_venc:
            with torch.no_grad():
                for hs, he in h_tiles:
                    for ws, we in w_tiles:
                        ph_s, pw_s = hs * AE_FACTOR, ws * AE_FACTOR
                        ph_e = min(he * AE_FACTOR, lr_pixel.shape[2])
                        pw_e = min(we * AE_FACTOR, lr_pixel.shape[3])
                        lq_crop = lr_pixel[:, :, ph_s:ph_e, pw_s:pw_e]
                        tile_venc[(hs, ws)] = self.venc(lq_crop)

        # Timesteps
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # Initial state: full-image noise (shared across tiles)
        generator = torch.Generator(device=device).manual_seed(seed)
        x = z_lr + flow_sigma * torch.randn(B, C, H, W, generator=generator, device=device)

        # Gaussian blend weights
        g_weight = self._gaussian_weights(lt_size, lt_size, C, device)

        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        for t_curr, t_prev in tqdm(step_pairs, desc="Tiled RFMSR",
                                   total=len(step_pairs), leave=False):
            t_batch = torch.full((B,), t_curr, device=device)
            dt = t_prev - t_curr

            v_acc = torch.zeros(B, C, H, W, device=device)
            w_acc = torch.zeros(B, C, H, W, device=device)

            for hs, he in h_tiles:
                for ws, we in w_tiles:
                    x_tile = x[:, :, hs:he, ws:we]
                    z_lr_tile = z_lr[:, :, hs:he, ws:we]
                    tile_fea = tile_venc.get((hs, ws), None) if use_venc else None

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext():
                        v_tile = self.rfmsr(
                            x_tile, t_batch, z_lr_tile, venc_fea=tile_fea
                        ).float()

                    v_acc[:, :, hs:he, ws:we] += v_tile * g_weight
                    w_acc[:, :, hs:he, ws:we] += g_weight

            v_total = v_acc / w_acc.clamp(min=1e-8)
            x = x + dt * v_total

        return x

    # ---- Main entry ----

    @torch.no_grad()
    def infer(self, src: Image.Image, scale: float = 4.0, steps: int = 1,
              flow_sigma: float = 1.0, seed: int = 42,
              chopping: bool = True, tile_size: int = 512,
              tile_stride: int = 256,
              color_correction: str = "wavelet") -> Image.Image:
        """Input PIL image → RFMSR super-resolution → output PIL image.

        Args:
            src:  输入 RGB 图像（PIL）。
            scale: 放大倍数（输出 = 输入 × scale）。
            steps: 反向积分步数（one-step 模型用 1，多步模型建议 8~15）。
            flow_sigma: 噪声标准差。
            seed: 随机种子（初始噪声）。
            chopping: 大图分块推理（tiled）。
            tile_size / tile_stride: 像素空间分块尺寸/步长。
            color_correction: 'adain' | 'wavelet' | 'ycbcr' | 'none'。
        """
        _chopping = chopping
        _tile_size = tile_size
        _tile_stride = tile_stride

        AE_FACTOR = 8
        PATCH_SIZE = 2
        MOD_PIXEL = 16

        use_amp = self._device.startswith("cuda")
        dtype = torch.bfloat16 if use_amp else torch.float32

        # ---- Resize LR ----
        exact_w = int(src.size[0] * scale)
        exact_h = int(src.size[1] * scale)
        target = src.resize((exact_w, exact_h), Image.BICUBIC)
        logger.info(
            f"RFMSR 推理: 输入 {src.size[0]}x{src.size[1]} × {scale} → {exact_w}x{exact_h}, "
            f"steps={steps}, tiled={'自动/启用' if chopping else '关闭'}"
        )

        im_np = np.array(target).astype(np.float32) / 255.0
        im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0)
        im_cond = im_cond.to(dtype=dtype, device=self._device)
        ori_h, ori_w = im_cond.shape[-2:]

        # ---- Align to multiple of 16 ----
        h, w = im_cond.shape[-2:]
        pad_h = (math.ceil(h / MOD_PIXEL) * MOD_PIXEL) - h
        pad_w = (math.ceil(w / MOD_PIXEL) * MOD_PIXEL) - w
        if pad_h > 0 or pad_w > 0:
            im_cond = F.pad(im_cond, (0, pad_w, 0, pad_h), mode="reflect")

        # ---- VAE global encode ----
        image_tensor = im_cond * 2.0 - 1.0
        z_lr = self.vae_encode(image_tensor)
        lh, lw = z_lr.shape[2], z_lr.shape[3]

        # ---- LR pixel space (for DINOv2) ----
        lr_pixel = None
        if self.venc is not None:
            lr_pixel = im_cond.float()

        # ---- Tile params ----
        lt_size = max((_tile_size // AE_FACTOR // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
        lt_stride = max((_tile_stride // AE_FACTOR // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
        lt_size = min(lt_size, min(lh, lw))
        lt_stride = min(lt_stride, lt_size)

        # 非方形图强制走 tiled 路径（forward_flexible 断言方形输入，tiled 的 tile 恒为方形）
        use_tiling = (_chopping and (lh > lt_size or lw > lt_size)) or (lh != lw)

        if not use_tiling:
            z_hr = self.reverse_flow(z_lr, steps=steps, flow_sigma=flow_sigma,
                                     seed=seed, lr_pixel=lr_pixel)
        else:
            z_hr = self.reverse_flow_tiled(
                z_lr, steps=steps, flow_sigma=flow_sigma, seed=seed,
                lt_size=lt_size, lt_stride=lt_stride,
                lr_pixel=lr_pixel,
            )

        # ---- VAE global decode ----
        res_sr = self.vae_decode(z_hr)
        res_sr = res_sr[:, :, 0:ori_h, 0:ori_w]

        img = torch.clamp(res_sr, 0.0, 1.0)[0]
        decoded = 255.0 * np.moveaxis(img.cpu().float().numpy(), 0, 2)
        decoded = decoded.astype(np.uint8)
        sr_image = Image.fromarray(decoded)

        # ---- Color correction ----
        if color_correction != "none":
            sr_image = apply_color_fix(sr_image, target, method=color_correction)

        return sr_image
