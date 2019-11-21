import torch
import torch.nn.functional as F
from torch import nn

from deepstab.partialconv import PartialConv2d


class PartialConvolutionAutoencoder(nn.Module):
    def __init__(self):
        super(PartialConvolutionAutoencoder, self).__init__()

        self.e_conv1 = Down(3, 64, 7, bn=False)
        self.e_conv2 = Down(64, 128, 5)
        self.e_conv3 = Down(128, 256, 5)
        self.e_conv4 = Down(256, 512, 3)
        self.e_conv5 = Down(512, 512, 3)
        self.e_conv6 = Down(512, 512, 3)
        self.e_conv7 = Down(512, 512, 3)
        self.e_conv8 = Down(512, 512, 3)

        self.d_conv8 = Up(512, 512, 3)
        self.d_conv7 = Up(512, 512, 3)
        self.d_conv6 = Up(512, 512, 3)
        self.d_conv5 = Up(512, 512, 3)
        self.d_conv4 = Up(512, 256, 3)
        self.d_conv3 = Up(256, 128, 3)
        self.d_conv2 = Up(128, 64, 3)
        self.d_conv1 = Up(64, 3, 3, bn=False)
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, image_in, mask_in):
        e_image_out1, e_mask_out1 = self.e_conv1(image_in, mask_in)
        e_image_out2, e_mask_out2 = self.e_conv2(e_image_out1, e_mask_out1)
        e_image_out3, e_mask_out3 = self.e_conv3(e_image_out2, e_mask_out2)
        e_image_out4, e_mask_out4 = self.e_conv4(e_image_out3, e_mask_out3)
        e_image_out5, e_mask_out5 = self.e_conv5(e_image_out4, e_mask_out4)
        e_image_out6, e_mask_out6 = self.e_conv6(e_image_out5, e_mask_out5)
        e_image_out7, e_mask_out7 = self.e_conv7(e_image_out6, e_mask_out6)
        e_image_out8, e_mask_out8 = self.e_conv8(e_image_out7, e_mask_out7)

        d_image_out8, d_mask_out8 = self.d_conv8(e_image_out8, e_mask_out8)
        d_image_out7, d_mask_out7 = self.d_conv7(d_image_out8, d_mask_out8)
        d_image_out6, d_mask_out6 = self.d_conv6(d_image_out7, d_mask_out7)
        d_image_out5, d_mask_out5 = self.d_conv5(d_image_out6, d_mask_out6)
        d_image_out4, d_mask_out4 = self.d_conv4(d_image_out5, d_mask_out5)
        d_image_out3, d_mask_out3 = self.d_conv3(d_image_out4, d_mask_out4)
        d_image_out2, d_mask_out2 = self.d_conv2(d_image_out3, d_mask_out3)
        d_image_out1, d_mask_out1 = self.d_conv1(d_image_out2, d_mask_out2)
        image_out, mask_out = torch.sigmoid(self.out(d_image_out1)), d_mask_out1

        return image_out, mask_out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(Down, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size, stride=2, padding=int(kernel_size / 2),
                                  padding_mode='same',
                                  multi_channel=True, return_mask=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, image_in, mask_in):
        image_out, mask_out = self.conv(image_in, mask_in)
        if self.bn:
            image_out = self.bn(image_out)
        image_out = F.relu(image_out)
        return image_out, mask_out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(Up, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size, stride=1, padding=int(kernel_size / 2),
                                  padding_mode='same',
                                  multi_channel=True, return_mask=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, image_in1, mask_in1):
        image_in = F.interpolate(image_in1, scale_factor=2)
        mask_in = F.interpolate(mask_in1, scale_factor=2)
        image_out, mask_out = self.conv(image_in, mask_in)
        if self.bn:
            image_out = self.bn(image_out)
        image_out = F.leaky_relu(image_out, negative_slope=0.2)
        return image_out, mask_out
