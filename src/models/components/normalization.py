import torch
from torch import nn

class ChannelNormLayer(nn.Module):


    def __init__(self, norm_rate=0.25, eps=1e-08):
        super().__init__()
        self.norm_rate = norm_rate
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=3, keepdim=True)
        std = x.std(dim=3, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)
        return normalized_x * self.norm_rate
