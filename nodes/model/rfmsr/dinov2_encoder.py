"""
DINOv2 Encoder — frozen, extracts semantic features from LR images for DiT Cross-Attention.

权重与源码由节点负责下载到本地 torch_cache 目录（与官方 HF 仓库 ckpts/torch_cache
布局一致），加载时走 torch.hub source='local'，全程不访问网络。

Usage:
    encoder = create_dinov2_encoder(torch_cache_dir, device="cuda")
    features = encoder(lr_tensor)  # lr: [B,3,H,W] float [0,1] → list[[B,N,enc_dim]]
"""

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DINOV2_HUB_NAMES = {
    "dinov2b": "dinov2_vitb14",
    "dinov2l": "dinov2_vitl14",
    "dinov2g": "dinov2_vitg14",
}


class Dinov2Encoder(nn.Module):
    """Frozen DINOv2 feature extractor, outputting features from specified intermediate layers."""

    def __init__(
        self,
        enc_type: str = "dinov2b",
        dinov2_size: int = 448,
        layer_indices: list[int] | None = None,
        device: str = "cuda",
        torch_cache_dir: str | None = None,
    ):
        super().__init__()

        self.dinov2_size = dinov2_size
        self.layer_indices = layer_indices or [8]

        hub_name = DINOV2_HUB_NAMES.get(enc_type)
        if hub_name is None:
            raise ValueError(
                f"Unknown DINOv2 type: {enc_type}, "
                f"expected one of {list(DINOV2_HUB_NAMES)}"
            )

        # 使用本地 torch_cache（官方布局：<torch_cache>/facebookresearch_dinov2_main + checkpoints/）。
        # torch.hub 的 get_dir() 只读 TORCH_HOME 环境变量（set_dir 只是写入该变量），
        # 且 source='local' 时会把 repo 名当相对路径拼接到 hub_dir 下——因此必须传
        # 本地源码目录的绝对路径，并在加载期间临时把 TORCH_HOME 指向 torch_cache，
        # 使 hubconf 内部的 load_state_dict_from_url 命中本地 checkpoints 权重、不联网。
        # 加载完成后恢复环境变量，避免影响进程内其他 torch.hub 用户。
        old_home = os.environ.get("TORCH_HOME")
        old_dir = torch.hub.get_dir()
        if torch_cache_dir:
            os.makedirs(torch_cache_dir, exist_ok=True)
            torch.hub.set_dir(torch_cache_dir)
            os.environ["TORCH_HOME"] = torch_cache_dir
        try:
            repo_dir = os.path.join(torch_cache_dir, "facebookresearch_dinov2_main")
            encoder = torch.hub.load(
                repo_dir, hub_name, source="local", verbose=False
            )
        finally:
            torch.hub.set_dir(old_dir)
            if old_home is not None:
                os.environ["TORCH_HOME"] = old_home
            else:
                os.environ.pop("TORCH_HOME", None)

        # Remove classification head, replace with Identity
        del encoder.head
        encoder.head = torch.nn.Identity()

        # Inject forward_with_features method (identical to VOSR)
        def forward_with_features(self, x, masks=None):
            features = {}
            layer_indices = list(range(len(self.blocks)))
            if isinstance(x, list):
                return self.forward_features_list(x, masks)
            x = self.prepare_tokens_with_masks(x, masks)
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in layer_indices:
                    features[f"layer_{i}"] = x[:, 1:]  # Remove CLS token
            x_norm = self.norm(x)
            return features, x_norm[:, 1:]

        encoder.forward_with_features = types.MethodType(forward_with_features, encoder)

        self.encoder = encoder.to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def preprocess(self, lr: torch.Tensor) -> torch.Tensor:
        """
        lr: [B, 3, H, W] float [0, 1]
        → resize → clamp → ImageNet normalization
        """
        x = F.interpolate(lr, size=self.dinov2_size, mode="bicubic", align_corners=False)
        x = x.clamp(0, 1)
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        return x

    @torch.no_grad()
    def forward(self, lr: torch.Tensor) -> list[torch.Tensor]:
        """
        lr: [B, 3, H, W] float [0, 1]
        → list of [B, N_patches, enc_dim]  each tensor corresponds to one specified layer in layer_indices
        """
        x = self.preprocess(lr)

        features, x_norm = self.encoder.forward_with_features(x)
        z = [v for k, v in features.items() if k.startswith("layer_")]
        z[-1] = x_norm
        z = [z[i] for i in self.layer_indices]

        return z


def create_dinov2_encoder(torch_cache_dir: str | None = None, device: str = "cuda") -> Dinov2Encoder:
    """Create the DINOv2 encoder with official configs/rfmsr.yaml defaults (dinov2b)."""
    return Dinov2Encoder(
        enc_type="dinov2b",
        dinov2_size=448,
        layer_indices=[8],
        device=device,
        torch_cache_dir=torch_cache_dir,
    )
