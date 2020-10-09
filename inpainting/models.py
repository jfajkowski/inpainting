import torch

from inpainting.layers import PartialConvLayer, GatedConvLayer, ConvLayer


class Dilated(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            ConvLayer(3, 64, 5),
            ConvLayer(64, 128, 3, downsample=True),
            ConvLayer(128, 128, 3),
            ConvLayer(128, 256, 3, downsample=True),
            ConvLayer(256, 256, 3),
            ConvLayer(256, 256, 3),
            ConvLayer(256, 256, 3, dilation=2),
            ConvLayer(256, 256, 3, dilation=4),
            ConvLayer(256, 256, 3, dilation=8),
            ConvLayer(256, 256, 3, dilation=16),
            ConvLayer(256, 256, 3),
            ConvLayer(256, 256, 3),
            ConvLayer(256, 128, 3, upsample=True),
            ConvLayer(128, 128, 3),
            ConvLayer(128, 64, 3, upsample=True),
            ConvLayer(64, 32, 3),
            ConvLayer(32, 2, 3, batchnorm=False, activation=torch.nn.functional.tanh)
        ])

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class UNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample_layers = torch.nn.ModuleList([
            ConvLayer(3, 64, 7, downsample=True),
            ConvLayer(64, 128, 5, downsample=True),
            ConvLayer(128, 256, 5, downsample=True),
            ConvLayer(256, 512, 3, downsample=True),
            ConvLayer(512, 512, 3, downsample=True),
            ConvLayer(512, 512, 3, downsample=True),
            ConvLayer(512, 512, 3, downsample=True)
        ])
        self.upsample_layers = torch.nn.ModuleList([
            ConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(512 + 256, 256, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(256 + 128, 128, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(128 + 64, 64, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(64 + 3, 2, 3, upsample=True, batchnorm=False, activation=torch.nn.functional.tanh),
        ])

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        skip = []
        for layer in self.downsample_layers:
            skip.append(x)
            x = layer(x)
        for layer in self.upsample_layers:
            x = layer(x, skip.pop())
        return x


class GatedConvUNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample_layers = torch.nn.ModuleList([
            GatedConvLayer(3, 64, 7, downsample=True),
            GatedConvLayer(64, 128, 5, downsample=True),
            GatedConvLayer(128, 256, 5, downsample=True),
            GatedConvLayer(256, 512, 3, downsample=True),
            GatedConvLayer(512, 512, 3, downsample=True),
            GatedConvLayer(512, 512, 3, downsample=True),
            GatedConvLayer(512, 512, 3, downsample=True)
        ])
        self.upsample_layers = torch.nn.ModuleList([
            GatedConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            GatedConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            GatedConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            GatedConvLayer(512 + 256, 256, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            GatedConvLayer(256 + 128, 128, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            GatedConvLayer(128 + 64, 64, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            GatedConvLayer(64 + 3, 2, 3, upsample=True, batchnorm=False, activation=torch.nn.functional.tanh),
        ])

    def forward(self, x, mask):
        mask = 1 - mask
        x = x * mask
        x = torch.cat([x, mask], dim=1)
        skip = []
        for layer in self.downsample_layers:
            skip.append(x)
            x = layer(x)
        for layer in self.upsample_layers:
            x = layer(x, skip.pop())
        return x


class PartialConvUNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample_layers = torch.nn.ModuleList([
            PartialConvLayer(2, 64, 7, downsample=True),
            PartialConvLayer(64, 128, 5, downsample=True),
            PartialConvLayer(128, 256, 5, downsample=True),
            PartialConvLayer(256, 512, 3, downsample=True),
            PartialConvLayer(512, 512, 3, downsample=True),
            PartialConvLayer(512, 512, 3, downsample=True),
            PartialConvLayer(512, 512, 3, downsample=True)
        ])
        self.upsample_layers = torch.nn.ModuleList([
            PartialConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            PartialConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            PartialConvLayer(512 + 512, 512, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            PartialConvLayer(512 + 256, 256, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            PartialConvLayer(256 + 128, 128, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            PartialConvLayer(128 + 64, 64, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            PartialConvLayer(64 + 2, 2, 3, upsample=True, batchnorm=False, activation=torch.nn.functional.tanh),
        ])

    def forward(self, x, mask):
        mask = 1 - mask
        mask = torch.cat([mask, mask], dim=1)
        skip_x, skip_mask = [], []
        for layer in self.downsample_layers:
            skip_x.append(x)
            skip_mask.append(mask)
            x, mask = layer(x, mask)
        for layer in self.upsample_layers:
            x, mask = layer(x, mask, skip_x.pop(), skip_mask.pop())
        return x
