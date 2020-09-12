import glob

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import inpainting.transforms as T
from inpainting.layers import ConvLayer, DepthwiseSeparableConvLayer, InvertedResidualConvLayer, GatedConvLayer, \
    DepthwiseSeparableGatedConvLayer
from inpainting.load import ImageDataset, ImageObjectRemovalDataset, MergeDataset
from inpainting.utils import normalize, denormalize


class InpaintingModel(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_image = None
        self.input_mask = None
        self.target = None
        self.g_output = None

        self.generator = nn.Sequential(
            ConvLayer(3, 64, 5),
            DepthwiseSeparableConvLayer(64, 128, 3, stride=2),
            DepthwiseSeparableConvLayer(128, 128, 3),
            DepthwiseSeparableConvLayer(128, 256, 3, stride=2),
            DepthwiseSeparableConvLayer(256, 256, 3),
            DepthwiseSeparableConvLayer(256, 256, 3),
            DepthwiseSeparableConvLayer(256, 256, 3, dilation=2),
            DepthwiseSeparableConvLayer(256, 256, 3, dilation=4),
            DepthwiseSeparableConvLayer(256, 256, 3, dilation=8),
            DepthwiseSeparableConvLayer(256, 256, 3, dilation=16),
            DepthwiseSeparableConvLayer(256, 256, 3),
            DepthwiseSeparableConvLayer(256, 256, 3),
            DepthwiseSeparableConvLayer(256, 128, 3, upsample=True),
            DepthwiseSeparableConvLayer(128, 128, 3),
            DepthwiseSeparableConvLayer(128, 64, 3, upsample=True),
            DepthwiseSeparableConvLayer(64, 32, 3),
            ConvLayer(32, 3, 3, norm=None, activation=F.tanh),
        )
        self.discriminator = nn.Sequential(
            ConvLayer(3, 64, norm='spectral', activation=F.leaky_relu),
            ConvLayer(64, 128, stride=2, norm='spectral', activation=F.leaky_relu),
            ConvLayer(128, 256, stride=2, norm='spectral', activation=F.leaky_relu),
            ConvLayer(256, 256, stride=2, norm='spectral', activation=F.leaky_relu),
            ConvLayer(256, 256, stride=2, norm='spectral', activation=F.leaky_relu),
            ConvLayer(256, 256, stride=2, norm='spectral', activation=None)
        )

    def forward(self, image, mask):
        return self.generator(image * (1 - mask))

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, mask, target = batch
        self.input_image = normalize(image)
        self.input_mask = mask
        self.target = normalize(target)
        self.g_output = self(self.input_image, self.input_mask)

        if optimizer_idx == 0:
            d_real_output = self.discriminator(self.target)
            d_fake_output = self.discriminator(self.g_output.detach())

            with autocast(enabled=False):
                d_real_loss = F.relu(1.0 - d_real_output.float()).mean()
                d_fake_loss = F.relu(1.0 + d_fake_output.float()).mean()
            d_loss = d_real_loss + d_fake_loss

            tensorboard_logs = {
                'd/real_loss': d_real_loss,
                'd/fake_loss': d_fake_loss,
                'd/loss': d_loss
            }
            progress_bar_logs = {
                'd_loss': d_loss
            }
            return {'loss': d_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_logs}

        if optimizer_idx == 1:
            g_fake_output = self.discriminator(self.g_output.float())

            g_hole_loss = torch.norm(self.input_mask * (self.target - self.g_output), 1) \
                          / torch.norm(self.input_mask, 1)
            g_valid_loss = torch.norm((1 - self.input_mask) * (self.target - self.g_output), 1) \
                           / torch.norm(1 - self.input_mask, 1)
            with autocast(enabled=False):
                g_adversarial_loss = - g_fake_output.float().mean()
            g_loss = g_hole_loss * 1 + g_valid_loss * 1 + g_adversarial_loss * 1

            tensorboard_logs = {
                'g/hole_loss': g_hole_loss,
                'g/valid_loss': g_valid_loss,
                'g/adversarial_loss': g_adversarial_loss,
                'g/loss': g_loss
            }
            progress_bar_logs = {
                'g_loss': g_loss
            }
            return {'loss': g_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_logs}

    def training_epoch_end(self, outputs):
        self.logger.experiment.add_image('input_image', make_grid(denormalize(self.input_image)), trainer.current_epoch)
        self.logger.experiment.add_image('input_mask', make_grid(self.input_mask), trainer.current_epoch)
        self.logger.experiment.add_image('target', make_grid(denormalize(self.target)), trainer.current_epoch)
        self.logger.experiment.add_image('output', make_grid(denormalize(self.g_output)), trainer.current_epoch)
        return {}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        return (
            {'optimizer': opt_d, 'frequency': 5},
            {'optimizer': opt_g, 'frequency': 1}
        )


if __name__ == '__main__':
    seed_everything(42)

    background_dataset = ImageDataset(
        list(sorted(glob.glob('../data/raw/YouTube-VOS/train/JPEGImages/*'))),
        'image',
        transform=T.Resize((256, 256))
    )
    foreground_dataset = MergeDataset(
        [
            ImageDataset(
                list(sorted(glob.glob('../data/raw/YouTube-VOS/train/JPEGImages/*'))),
                'image'
            ),
            ImageDataset(
                list(sorted(glob.glob('../data/raw/YouTube-VOS/train/Annotations/*'))),
                'annotation'
            )
        ],
        transform=T.Resize((256, 256))
    )

    dataset = ImageObjectRemovalDataset(background_dataset, foreground_dataset,
                                        transform=T.Compose([
                                            T.ToTensor(),
                                        ]))
    loader = DataLoader(dataset, batch_size=16, num_workers=2)

    model = InpaintingModel()
    trainer = pl.Trainer(deterministic=True, gpus=1, precision=16, limit_train_batches=1000, max_epochs=100)
    trainer.fit(model, loader)
