# Vendored from VOSR's models/swiglu_ffn.py via ylchen333/ComfyUI-VOSR2
# (Apache-2.0; upstream: DINOv2, Apache-2.0), trimmed to the pure-PyTorch
# SwiGLUFFN. xFormers-backed variant is dropped -- VOSR2 always uses this one.
import torch.nn.functional as F
from torch import Tensor, nn

import comfy.ops

# comfy.ops.disable_weight_init 与 torch.nn 同 state_dict 键/同 forward，
# 仅跳过随机初始化开销，并让权重可被 ComfyUI 显存/精度机制管理。
ops = comfy.ops.disable_weight_init


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True) -> None:
        super().__init__()
        self.w12 = ops.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = ops.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)
