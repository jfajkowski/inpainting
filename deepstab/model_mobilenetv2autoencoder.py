from torch import nn


class MobilenetV2Autoencoder(nn.Module):
    def __init__(self):
        super(MobilenetV2Autoencoder, self).__init__()
        self.model = nn.Sequential(
            conv_bn(3, 16, 2),
            InvertedResidual(16, 32, 2, 1),
            InvertedResidual(32, 64, 2, 1),
            InvertedResidual(64, 128, 2, 1),
            InvertedResidual(128, 256, 2, 6),
            InvertedResidual(256, 512, 2, 6),
            nn.Upsample(scale_factor=2),
            InvertedResidual(512, 256, 1, 6),
            nn.Upsample(scale_factor=2),
            InvertedResidual(256, 128, 1, 6),
            nn.Upsample(scale_factor=2),
            InvertedResidual(128, 64, 1, 1),
            nn.Upsample(scale_factor=2),
            InvertedResidual(64, 32, 1, 1),
            nn.Upsample(scale_factor=2),
            InvertedResidual(32, 16, 1, 1),
            nn.Upsample(scale_factor=2),
            conv_bn(16, 3, 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image_in, mask_in):
        return self.model(image_in), mask_in


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
