import torch
import torch.nn as nn


class BaselineDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)


class BaselineUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding = nn.Sequential(
            BaselineDown(4, 64),
            BaselineDown(64, 128),
            BaselineDown(128, 256),
            BaselineDown(256, 512),
            BaselineDown(512, 512),
            BaselineDown(512, 512),
            BaselineDown(512, 512),
        )
        self.decoding = nn.Sequential(
            BaselineUp(512, 512),
            BaselineUp(512, 512),
            BaselineUp(512, 512),
            BaselineUp(512, 256),
            BaselineUp(256, 128),
            BaselineUp(128, 64),
            BaselineUp(64, 3),
        )
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.encoding(x)
        x = self.decoding(x)
        return torch.tanh(self.out(x))
