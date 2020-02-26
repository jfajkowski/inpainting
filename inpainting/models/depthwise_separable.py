import torch
import torch.nn as nn


class DepthwiseSeparableDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=2, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class DepthwiseSeparableUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1,
                               stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class DepthwiseSeparableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding = nn.Sequential(
            DepthwiseSeparableDown(4, 64),
            DepthwiseSeparableDown(64, 128),
            DepthwiseSeparableDown(128, 256),
            DepthwiseSeparableDown(256, 512),
            DepthwiseSeparableDown(512, 512),
            DepthwiseSeparableDown(512, 512),
            DepthwiseSeparableDown(512, 512),
        )
        self.decoding = nn.Sequential(
            DepthwiseSeparableUp(512, 512),
            DepthwiseSeparableUp(512, 512),
            DepthwiseSeparableUp(512, 512),
            DepthwiseSeparableUp(512, 256),
            DepthwiseSeparableUp(256, 128),
            DepthwiseSeparableUp(128, 64),
            DepthwiseSeparableUp(64, 3),
        )
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.encoding(x)
        x = self.decoding(x)
        return torch.tanh(self.out(x))
