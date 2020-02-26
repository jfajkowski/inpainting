import torch
import torch.nn as nn


class GatedDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedDown, self).__init__()
        self.gate_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        gate_x = self.gate_layers(x)
        feature_x = self.feature_layers(x)
        return gate_x * feature_x


class GatedUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedUp, self).__init__()
        self.gate_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.feature_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        gate_x = self.gate_layers(x)
        feature_x = self.feature_layers(x)
        return gate_x * feature_x


class GatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding = nn.Sequential(
            GatedDown(4, 64),
            GatedDown(64, 128),
            GatedDown(128, 256),
            GatedDown(256, 512),
            GatedDown(512, 512),
            GatedDown(512, 512),
            GatedDown(512, 512),
        )
        self.decoding = nn.Sequential(
            GatedUp(512, 512),
            GatedUp(512, 512),
            GatedUp(512, 512),
            GatedUp(512, 256),
            GatedUp(256, 128),
            GatedUp(128, 64),
            GatedUp(64, 3),
        )
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.encoding(x)
        x = self.decoding(x)
        return torch.tanh(self.out(x))
