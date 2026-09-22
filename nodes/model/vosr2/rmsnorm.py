# Vendored from VOSR's models/rmsnorm.py via ylchen333/ComfyUI-VOSR2 (Apache-2.0;
# upstream: Meta Llama 2 RMSNorm), trimmed to the single class LightningDiT uses.
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
