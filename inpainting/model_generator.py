import torch
import torch.nn.functional as F
from torch import nn


class GatingConvolutionUNet(nn.Module):

    def __init__(self):
        super(GatingConvolutionUNet, self).__init__()

        self.e_conv1 = GatingConvolution(4, 64, 7, mode='down', bn=False)
        self.e_conv2 = GatingConvolution(64, 128, 5, mode='down', bn=True)
        self.e_conv3 = GatingConvolution(128, 256, 5, mode='down', bn=True)
        self.e_conv4 = GatingConvolution(256, 512, 3, mode='down', bn=True)
        self.e_conv5 = GatingConvolution(512, 512, 3, mode='down', bn=True)
        self.e_conv6 = GatingConvolution(512, 512, 3, mode='down', bn=True)
        self.e_conv7 = GatingConvolution(512, 512, 3, mode='down', bn=True)

        self.d_conv7 = GatingConvolution(1024, 512, 3, mode='up', bn=True)
        self.d_conv6 = GatingConvolution(1024, 512, 3, mode='up', bn=True)
        self.d_conv5 = GatingConvolution(1024, 512, 3, mode='up', bn=True)
        self.d_conv4 = GatingConvolution(768, 256, 3, mode='up', bn=True)
        self.d_conv3 = GatingConvolution(384, 128, 3, mode='up', bn=True)
        self.d_conv2 = GatingConvolution(192, 64, 3, mode='up', bn=True)
        self.d_conv1 = GatingConvolution(67, 3, 3, mode='up', bn=False)

        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, image_in, mask_in):
        x = torch.cat([image_in, mask_in], 1)
        e_image_out1 = self.e_conv1(x)
        e_image_out2 = self.e_conv2(e_image_out1)
        e_image_out3 = self.e_conv3(e_image_out2)
        e_image_out4 = self.e_conv4(e_image_out3)
        e_image_out5 = self.e_conv5(e_image_out4)
        e_image_out6 = self.e_conv6(e_image_out5)
        e_image_out7 = self.e_conv7(e_image_out6)

        d_image_out7 = self.d_conv7(e_image_out7, e_image_out6)
        d_image_out6 = self.d_conv6(d_image_out7, e_image_out5)
        d_image_out5 = self.d_conv5(d_image_out6, e_image_out4)
        d_image_out4 = self.d_conv4(d_image_out5, e_image_out3)
        d_image_out3 = self.d_conv3(d_image_out4, e_image_out2)
        d_image_out2 = self.d_conv2(d_image_out3, e_image_out1)
        d_image_out1 = self.d_conv1(d_image_out2, image_in)

        return torch.tanh(self.out(d_image_out1))


class GatingConvolutionAutoencoder(nn.Module):

    def __init__(self):
        super(GatingConvolutionAutoencoder, self).__init__()

        self.e_conv1 = GatingConvolution(4, 64, 7, mode='down', bn=False)
        self.e_conv2 = GatingConvolution(64, 128, 5, mode='down', bn=True)
        self.e_conv3 = GatingConvolution(128, 256, 3, mode='down', bn=True)
        self.e_conv4 = GatingConvolution(256, 512, 3, mode='down', bn=True)
        self.e_conv5 = GatingConvolution(512, 512, 3, mode='down', bn=True)
        self.e_conv6 = GatingConvolution(512, 512, 3, mode='down', bn=True)
        self.e_conv7 = GatingConvolution(512, 512, 3, mode='down', bn=True)

        self.d_conv7 = GatingConvolution(512, 512, 3, mode='up', bn=True)
        self.d_conv6 = GatingConvolution(512, 512, 3, mode='up', bn=True)
        self.d_conv5 = GatingConvolution(512, 512, 3, mode='up', bn=True)
        self.d_conv4 = GatingConvolution(512, 256, 3, mode='up', bn=True)
        self.d_conv3 = GatingConvolution(256, 128, 3, mode='up', bn=True)
        self.d_conv2 = GatingConvolution(128, 64, 3, mode='up', bn=True)
        self.d_conv1 = GatingConvolution(64, 3, 3, mode='up', bn=False)

        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, image_in, mask_in):
        x = torch.cat([image_in, mask_in], 1)
        e_image_out1 = self.e_conv1(x)
        e_image_out2 = self.e_conv2(e_image_out1)
        e_image_out3 = self.e_conv3(e_image_out2)
        e_image_out4 = self.e_conv4(e_image_out3)
        e_image_out5 = self.e_conv5(e_image_out4)
        e_image_out6 = self.e_conv6(e_image_out5)
        e_image_out7 = self.e_conv7(e_image_out6)

        d_image_out7 = self.d_conv7(e_image_out7)
        d_image_out6 = self.d_conv6(d_image_out7)
        d_image_out5 = self.d_conv5(d_image_out6)
        d_image_out4 = self.d_conv4(d_image_out5)
        d_image_out3 = self.d_conv3(d_image_out4)
        d_image_out2 = self.d_conv2(d_image_out3)
        d_image_out1 = self.d_conv1(d_image_out2)

        return torch.tanh(self.out(d_image_out1))


class GatingConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, mode, bn=False):
        super(GatingConvolution, self).__init__()

        self.mode = mode

        stride = 1 if mode in ['up', 'keep'] else 2
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                     padding=kernel_size // 2, padding_mode='same')
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
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
            image_in1 = F.interpolate(image_in1, scale_factor=2, mode='bilinear')

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
