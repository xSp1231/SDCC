import math
import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    Efficient Channel Attention (CVPR 2020)
    https://arxiv.org/abs/1910.03151

    Replaces the FC-based channel attention in SENet with a lightweight
    1-D convolution over the channel dimension, avoiding dimensionality
    reduction and thus preserving full inter-channel information.

    For c=2048: kernel_size is auto-computed as 7, giving only 7 parameters.
    Input / output shape: [N, C, H, W]  (unchanged)
    """

    def __init__(self, channels: int, b: int = 1, gamma: int = 2):
        print("cgcl")
        super(ECA, self).__init__()
        # Adaptively determine kernel size from channel count
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        y = self.avg_pool(x)                              # [N, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)               # [N, 1, C]
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # [N, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)
