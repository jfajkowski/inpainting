import torch
import torch.nn as nn
import torch.nn.functional as F


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups,
                                            bias=bias))


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1,
                                        padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1,
                                      padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1,
                                    padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, stride=1,
                                       padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma * attn_g
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mode=None, bn=False):
        super(Block, self).__init__()

        self.stride = 1 if mode in ['up', 'keep'] else 2
        self.mode = mode

        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=self.stride,
                                                     padding=kernel_size // 2, padding_mode='same'))
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, image_in1, image_in2=None):
        if self.mode == 'up':
            image_in1 = F.interpolate(image_in1, scale_factor=2, mode='bilinear')
        if image_in2:
            image_in = torch.cat([image_in1, image_in2], dim=1)
        else:
            image_in = image_in1
        x = self.conv(image_in)
        if self.bn:
            x = self.bn(x)
        x = F.leaky_relu(x)
        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            Block(4, 64, 7, mode='down', bn=False),
            Block(64, 128, 5, mode='down', bn=True),
            Block(128, 256, 3, mode='down', bn=True),
            Block(256, 256, 3, mode='down', bn=True),
        )
        self.attention = SelfAttention(256)
        self.decoder = nn.Sequential(
            Block(256, 256, 3, mode='up', bn=True),
            Block(256, 128, 3, mode='up', bn=True),
            Block(128, 64, 5, mode='up', bn=True),
            Block(64, 3, 7, mode='up', bn=False)
        )
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, image_in, mask_in):
        x = torch.cat([image_in, mask_in], 1)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        x = self.out(x)
        return torch.tanh(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Block(3, 64, 3, mode='down', bn=False),
            Block(64, 128, 3, mode='down', bn=True),
            Block(128, 256, 3, mode='down', bn=True),
            SelfAttention(256),
            Block(256, 256, 3, mode='down', bn=False)
        )
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.out(x)
        return x