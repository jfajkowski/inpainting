import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, batchnorm=True,
                 downsample=False, upsample=False, activation=torch.relu):
        super().__init__()
        stride = 2 if downsample else 1
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=((kernel_size - 1) * dilation) // 2, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
