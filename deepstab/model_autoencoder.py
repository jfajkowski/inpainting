import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingConvolutionAutoencoder(nn.Module):
    def __init__(self):
        super(GatingConvolutionAutoencoder, self).__init__()

        # self.model = nn.Sequential(
        #     GatingConvolution(4, 64, 7, stride=2, padding=3),
        #     nn.BatchNorm2d(64),
        #     GatingConvolution(64, 128, 5, stride=2, padding=2),
        #     nn.BatchNorm2d(128),
        #     GatingConvolution(128, 256, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     GatingConvolution(256, 256, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     GatingConvolution(256, 256, 3, stride=1, padding=2, dilation=2),
        #     nn.BatchNorm2d(256),
        #     GatingConvolution(256, 256, 3, stride=1, padding=4, dilation=4),
        #     nn.BatchNorm2d(256),
        #     GatingConvolution(256, 256, 3, stride=1, padding=8, dilation=8),
        #     nn.BatchNorm2d(256),
        #     nn.Upsample(scale_factor=2),
        #     GatingConvolution(256, 256, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.Upsample(scale_factor=2),
        #     GatingConvolution(256, 128, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.Upsample(scale_factor=2),
        #     GatingConvolution(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.Upsample(scale_factor=2),
        #     GatingConvolution(64, 3, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(3),
        #     GatingConvolution(3, 3, 1, stride=1, padding=0)
        # )

        self.model = nn.Sequential(
            GatingConvolution(4, 64, 7, mode='down', bn=False),
            GatingConvolution(64, 128, 5, mode='down', bn=True),
            GatingConvolution(128, 256, 3, mode='down', bn=True),
            GatingConvolution(256, 256, 3, mode='down', bn=True),
            GatingConvolution(256, 256, 3, mode='keep', bn=True),
            GatingConvolution(256, 256, 3, mode='keep', bn=True),
            GatingConvolution(256, 256, 3, mode='keep', bn=True),
            GatingConvolution(256, 256, 3, mode='up', bn=True),
            GatingConvolution(256, 128, 5, mode='up', bn=True),
            GatingConvolution(128, 64, 7, mode='up', bn=True),
            GatingConvolution(64, 3, 3, mode='up', bn=False),
            nn.Conv2d(3, 3, 1)
        )

    def forward(self, image_in, mask_in):
        x = torch.cat([image_in, mask_in], 1)
        return torch.tanh(self.model(x))


# class GatingConvolution(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
#         super(GatingConvolution, self).__init__()
#         self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                                      dilation=dilation, padding_mode='same')
#         self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                                       dilation=dilation, padding_mode='same')
#
#     def forward(self, image_in):
#         gating_x = self.gating_conv(image_in)
#         feature_x = self.feature_conv(image_in)
#         gating_x = torch.sigmoid(gating_x)
#         feature_x = torch.relu(feature_x)
#         return gating_x * feature_x
#

class GatingConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mode=None, bn=False):
        super(GatingConvolution, self).__init__()

        self.stride = 1 if mode in ['up', 'keep'] else 2
        self.mode = mode

        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=self.stride,
                                     padding=kernel_size // 2, padding_mode='same')
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=self.stride,
                                      padding=kernel_size // 2, padding_mode='same')
        self.bn = bn
        if bn:
            self.gating_bn = nn.BatchNorm2d(out_channels)
            self.feature_bn = nn.BatchNorm2d(out_channels)
        else:
            self.gating_bn = None
            self.feature_bn = None

    def forward(self, image_in1, image_in2=None):
        if self.mode == 'up':
            image_in1 = F.interpolate(image_in1, scale_factor=2)
        if image_in2:
            image_in = torch.cat([image_in1, image_in2], dim=1)
        else:
            image_in = image_in1
        gating_x = self.gating_conv(image_in)
        feature_x = self.feature_conv(image_in)
        if self.bn:
            gating_x = self.gating_bn(gating_x)
            feature_x = self.feature_bn(feature_x)
        gating_x = torch.sigmoid(gating_x)
        feature_x = torch.relu(feature_x)
        return gating_x * feature_x
