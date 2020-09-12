import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F


class AttentionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adjust = in_channels != out_channels

        if self.adjust:
            self.adjust_x = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            self.adjust_z = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 8, kernel_size=1),
            nn.BatchNorm2d(out_channels // 8)
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 8, kernel_size=1),
            nn.BatchNorm2d(out_channels // 8)
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, z=None, debug=False):
        if self.adjust:
            x = self.adjust_x(x)
            z = self.adjust_z(z)

        if not z:
            z = x

        batch = x.shape[0]
        with autocast(enabled=False):
            x = x.float()
            z = z.float()
            proj_query = self.query_conv(x).view(batch, -1, x.size(2) * x.size(3)).float()  # B X C X W_z * H_z
            proj_key = self.key_conv(z).view(batch, -1, z.size(2) * z.size(3)).float()  # B X C x W_x * H_x
            proj_value = self.value_conv(z).view(batch, -1, z.size(2) * z.size(3)).float()  # B X C X W_x * H_x

            attention = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # B X N_z X N_x
            attention = F.softmax(attention, dim=-1)

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(batch, -1, x.size(2), x.size(3))
            out = self.gamma * out + x

        if debug:
            return out, attention
        else:
            return out


class CorrelationLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adjust = in_channels != out_channels

        if self.adjust:
            self.adjust_x = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            self.adjust_z = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, z=None, debug=False):
        if self.adjust:
            x = self.adjust_x(x)
            z = self.adjust_z(z)

        if not z:
            z = x

        with autocast(enabled=False):
            batch, channel = x.shape[:2]
            proj_x = x.view(1, batch * channel, x.size(2), x.size(3)).float()  # 1 x B * C x W_z * H_z
            proj_z = z.view(batch * channel, 1, z.size(2), z.size(3)).float()  # B * C x 1 x W_x * H_x

            correlation = F.conv2d(proj_x, proj_z, padding=z.size(2) // 2, groups=batch * channel)  # 1 x B * C x W_x x H_x
            correlation = correlation.view(batch, channel, 1, x.size(2) * x.size(3))
            correlation = F.softmax(correlation, dim=3)
            correlation = correlation.view(batch, channel, x.size(2), x.size(3))

            out = self.gamma * correlation + x

        if debug:
            return out, correlation
        else:
            return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm='batch',
                 upsample=False, activation=torch.relu):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=((kernel_size - 1) * dilation) // 2, dilation=dilation)
        if norm == 'spectral':
            self.conv = nn.utils.spectral_norm(self.conv)
        self.norm = nn.BatchNorm2d(out_channels) if norm == 'batch' else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, image_in):
        image_in = self.upsample(image_in)
        feature_x = self.conv(image_in)
        feature_x = self.norm(feature_x)
        feature_x = self.activation(feature_x)
        return feature_x


class DepthwiseSeparableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm='batch',
                 upsample=False, activation=torch.relu):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else nn.Identity()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                        padding=((kernel_size - 1) * dilation) // 2, dilation=dilation,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if norm == 'spectral':
            self.depthwise_conv = nn.utils.spectral_norm(self.depthwise_conv)
            self.pointwise_conv = nn.utils.spectral_norm(self.pointwise_conv)
        self.norm = nn.BatchNorm2d(out_channels) if norm == 'batch' else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, image_in):
        image_in = self.upsample(image_in)
        feature_x = self.depthwise_conv(image_in)
        feature_x = self.pointwise_conv(feature_x)
        feature_x = self.norm(feature_x)
        feature_x = self.activation(feature_x)
        return feature_x


class InvertedResidualConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm='batch',
                 upsample=False, activation=None, expansion=6):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = in_channels == out_channels and stride == 1
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else nn.Identity()
        self.pointwise_conv_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride,
                                        padding=((kernel_size - 1) * dilation) // 2, dilation=dilation,
                                        groups=hidden_channels)
        self.pointwise_conv_2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        if norm == 'spectral':
            self.pointwise_conv_1 = nn.utils.spectral_norm(self.pointwise_conv_1)
            self.depthwise_conv = nn.utils.spectral_norm(self.depthwise_conv)
            self.pointwise_conv_2 = nn.utils.spectral_norm(self.pointwise_conv_2)
        self.pointwise_norm_1 = nn.BatchNorm2d(hidden_channels) if norm == 'batch' else nn.Identity()
        self.depthwise_norm = nn.BatchNorm2d(hidden_channels) if norm == 'batch' else nn.Identity()
        self.pointwise_norm_2 = nn.BatchNorm2d(out_channels) if norm == 'batch' else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, image_in):
        image_in = self.upsample(image_in)
        feature_x = self.pointwise_conv_1(image_in)
        feature_x = self.pointwise_norm_1(feature_x)
        feature_x = F.relu6(feature_x)
        feature_x = self.depthwise_conv(feature_x)
        feature_x = self.depthwise_norm(feature_x)
        feature_x = F.relu(feature_x)
        feature_x = self.pointwise_conv_2(feature_x)
        feature_x = self.pointwise_norm_2(feature_x)
        feature_x = self.activation(feature_x)
        return image_in + feature_x if self.use_residual else feature_x


class GatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm='batch',
                 upsample=False, activation=torch.relu):
        super().__init__()
        self.feature_layer = ConvLayer(in_channels, out_channels, kernel_size,
                                       stride, dilation, norm, upsample, activation=activation)
        self.gating_layer = ConvLayer(in_channels, out_channels, kernel_size,
                                      stride, dilation, norm, upsample, activation=torch.sigmoid)

    def forward(self, joint_in):
        feature_x = self.feature_layer(joint_in)
        gating_x = self.gating_layer(joint_in)
        return feature_x * gating_x


class DepthwiseSeparableGatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm='batch',
                 upsample=False, activation=torch.relu):
        super().__init__()
        self.feature_layer = DepthwiseSeparableConvLayer(in_channels, out_channels, kernel_size,
                                                         stride, dilation, norm, upsample, activation=activation)
        self.gating_layer = DepthwiseSeparableConvLayer(in_channels, out_channels, kernel_size,
                                                        stride, dilation, norm, upsample, activation=torch.sigmoid)

    def forward(self, joint_in):
        feature_x = self.feature_layer(joint_in)
        gating_x = self.gating_layer(joint_in)
        return feature_x * gating_x


class InvertedResidualGatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm='batch',
                 upsample=False, activation=None):
        super().__init__()
        self.feature_layer = InvertedResidualConvLayer(in_channels, out_channels, kernel_size,
                                                       stride, dilation, norm, upsample, activation=activation)
        self.gating_layer = InvertedResidualConvLayer(in_channels, out_channels, kernel_size,
                                                      stride, dilation, norm, upsample, activation=torch.sigmoid)

    def forward(self, joint_in):
        feature_x = self.feature_layer(joint_in)
        gating_x = self.gating_layer(joint_in)
        return feature_x * gating_x
