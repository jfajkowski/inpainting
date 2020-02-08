import torch
import torch.nn as nn
import torch.nn.functional as F

from inpainting.attention import SelfAttention


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
            image_in1 = F.interpolate(image_in1, scale_factor=2)
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
            # Block(128, 256, 3, mode='down', bn=True),
            # Block(256, 256, 3, mode='down', bn=True),
        )
        self.attention = SelfAttention(128)
        self.decoder = nn.Sequential(
            # Block(256, 256, 3, mode='up', bn=True),
            # Block(256, 128, 3, mode='up', bn=True),
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