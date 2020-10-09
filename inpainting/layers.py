import torch
import torch.nn.functional as F
from torch import nn


class PartialConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(PartialConv, self).__init__(*args, **kwargs)

        self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                             self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                      input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-6)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        return output, self.update_mask


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


class PartialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, batchnorm=True,
                 downsample=False, upsample=False, activation=torch.relu):
        super().__init__()
        stride = 2 if downsample else 1
        self.x_upsample = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else nn.Identity()
        self.mask_upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else nn.Identity()
        self.conv = PartialConv(in_channels, out_channels, kernel_size, stride,
                                padding=((kernel_size - 1) * dilation) // 2, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x, mask, skip_x=None, skip_mask=None):
        x = self.x_upsample(x)
        mask = self.mask_upsample(mask)
        if skip_x is not None and skip_mask is not None:
            x = torch.cat([x, skip_x], dim=1)
            mask = torch.cat([mask, skip_mask], dim=1)
        x, mask = self.conv(x, mask)
        x = self.norm(x)
        x = self.activation(x)
        return x, mask


class GatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, batchnorm=True,
                 downsample=False, upsample=False, activation=torch.relu):
        super().__init__()
        self.feature_layer = ConvLayer(in_channels, out_channels, kernel_size, dilation, batchnorm,
                                       downsample, upsample, activation=activation)
        self.gating_layer = ConvLayer(in_channels, out_channels, kernel_size, dilation, batchnorm,
                                      downsample, upsample, activation=torch.sigmoid)

    def forward(self, x, skip=None):
        feature_x = self.feature_layer(x, skip)
        gating_x = self.gating_layer(x, skip)
        return feature_x * gating_x
