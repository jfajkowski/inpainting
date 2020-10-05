import random

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid


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
            ConvLayer(256, 256, 3),
            ConvLayer(256, 256, 3),
            ConvLayer(256, 128, 3, upsample=True),
            ConvLayer(128, 128, 3),
            ConvLayer(128, 64, 3, upsample=True),
            ConvLayer(64, 32, 3),
            ConvLayer(32, 3, 3, batchnorm=False, activation=torch.nn.functional.tanh)
        ])

    def forward(self, x):
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
        ])
        self.upsample_layers = torch.nn.ModuleList([
            ConvLayer(512 + 256, 256, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(256 + 128, 128, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(128 + 64, 64, 3, upsample=True, activation=torch.nn.functional.leaky_relu),
            ConvLayer(64 + 3, 3, 3, upsample=True, batchnorm=False, activation=torch.nn.functional.tanh),
        ])

    def forward(self, x):
        skip = []
        for layer in self.downsample_layers:
            skip.append(x)
            x = layer(x)
        for layer in self.upsample_layers:
            x = layer(x, skip.pop())
        return x


class RandomMaskDataset(Dataset):

    def __init__(self, frame_dataset):
        self.frame_dataset = frame_dataset

    def __getitem__(self, index: int):
        frame, _ = self.frame_dataset[index]
        mask = self._random_mask(frame)
        return frame, mask

    @staticmethod
    def _random_mask(frame):
        _, h, w = frame.shape
        s = random.choice([h, w])
        min_points, max_points = 1, s // 5
        min_thickness, max_thickness = 1, s // 5
        min_angle_dir, max_angle_dir = 0, 2 * np.pi
        min_angle_fold, max_angle_fold = - np.pi / 2, np.pi / 2
        min_length, max_length = 1, s // 5

        mask = np.zeros((h, w), dtype='int')
        points = random.randint(min_points, max_points)
        thickness = random.randint(min_thickness, max_thickness)

        prev_x = random.randint(0, w)
        prev_y = random.randint(0, h)

        angle_dir = random.uniform(min_angle_dir, max_angle_dir)
        for i in range(points):
            angle_fold = random.uniform(min_angle_fold, max_angle_fold)
            angle = angle_dir + angle_fold
            length = random.randint(min_length, max_length)
            x = int(prev_x + length * np.sin(angle))
            y = int(prev_y + length * np.cos(angle))
            mask = cv.line(mask, (prev_x, prev_y), (x, y), color=255, thickness=thickness)
            prev_x = x
            prev_y = y
        return torch.tensor(mask).float().unsqueeze(0) / 255

    def __len__(self) -> int:
        return len(self.frame_dataset)


class Inpainting(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.input = None
        self.output = None
        self.target = None

        self.model = UNet()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        self.input = image * (1 - mask)
        self.output = self(self.input)
        self.target = image
        loss = F.l1_loss(self.output, self.target)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        self.input = image * (1 - mask)
        self.output = self(self.input)
        self.target = image
        loss = F.l1_loss(self.output, self.target)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_image('input', make_grid(self.input), trainer.current_epoch)
        self.logger.experiment.add_image('output', make_grid(self.output), trainer.current_epoch)
        self.logger.experiment.add_image('target', make_grid(self.target), trainer.current_epoch)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


if __name__ == '__main__':
    seed_everything(0)

    train = RandomMaskDataset(CIFAR10('data', train=True, download=True, transform=transforms.ToTensor()))
    train = DataLoader(train, batch_size=32, num_workers=4)
    val = RandomMaskDataset(CIFAR10('data', train=True, download=True, transform=transforms.ToTensor()))
    val = DataLoader(val, batch_size=32, num_workers=4)

    model = Inpainting()
    trainer = Trainer(gpus=1, max_epochs=20)
    trainer.fit(model, train, val)
