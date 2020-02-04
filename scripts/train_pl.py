from collections import OrderedDict

import opencv_transforms.transforms as transforms
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from deepstab.load import ImageDataset, FileMaskDataset, InpaintingImageDataset
from deepstab.metrics import PSNR, MAE, MSE
from deepstab.model_discriminator import DiscriminatorDCGAN
from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.utils import mask_tensor, normalize


class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        self.image_transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flip(0))
        ])

        self.mask_transforms = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomResizedCrop(256),
            transforms.ToTensor()
        ])

        self.generator = GatingConvolutionUNet()
        self.discriminator = DiscriminatorDCGAN()

        self.reconstruction_criterion = torch.nn.L1Loss()
        self.adversarial_criterion = torch.nn.BCELoss()

        self.metrics = {
            'psnr': PSNR(1),
            'l1': MAE(),
            'l2': MSE()
        }

        self.image_filled = None

    def forward(self, image, mask):
        image = normalize(image)
        image_masked = mask_tensor(image, mask)
        image_filled = self.generator(image_masked, mask)
        return image_filled

    def training_step(self, batch, batch_nb, optimizer_idx):
        image, mask = batch
        real = torch.ones((self.hparams.batch_size, 16384))
        fake = torch.zeros((self.hparams.batch_size, 16384))
        if self.on_gpu:
            real = real.cuda(image.device.index)
            fake = fake.cuda(image.device.index)

        if optimizer_idx == 0:
            self.image_filled = self.forward(image, mask)
            g_reconstruction_loss = self.reconstruction_criterion(self.image_filled, image)
            g_adversarial_loss = self.adversarial_criterion(self.discriminator(self.image_filled), real)
            g_loss = (g_reconstruction_loss + g_adversarial_loss) / 2

            tqdm_dict = {
                'g_reconstruction_loss': g_reconstruction_loss,
                'g_adversarial_loss': g_adversarial_loss,
                'g_loss': g_loss
            }
            for metric_name, metric_function in self.metrics.items():
                tqdm_dict[metric_name] = metric_function(image, self.image_filled)
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            d_real_loss = self.adversarial_criterion(self.discriminator(image), real)
            d_fake_loss = self.adversarial_criterion(self.discriminator(self.image_filled.detach()), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2

            tqdm_dict = {
                'd_real_loss': d_real_loss,
                'd_fake_loss': d_fake_loss,
                'd_loss': d_loss
            }
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_nb):
        image, mask = batch
        real = torch.ones((self.hparams.batch_size, 16384))
        fake = torch.zeros((self.hparams.batch_size, 16384))
        if self.on_gpu:
            real = real.cuda(image.device.index)
            fake = fake.cuda(image.device.index)

        image_filled = self.forward(image, mask)
        g_reconstruction_loss = self.reconstruction_criterion(image_filled, image)
        g_adversarial_loss = self.adversarial_criterion(self.discriminator(image_filled), real)
        g_loss = (g_reconstruction_loss + g_adversarial_loss) / 2

        d_real_loss = self.adversarial_criterion(self.discriminator(image), real)
        d_fake_loss = self.adversarial_criterion(self.discriminator(image_filled.detach()), fake)
        d_loss = (d_real_loss + d_fake_loss) / 2

        output = OrderedDict({
            'val_g_reconstruction_loss': g_reconstruction_loss,
            'val_g_adversarial_loss': g_adversarial_loss,
            'val_g_loss': g_loss,
            'val_d_real_loss': d_real_loss,
            'val_d_fake_loss': d_fake_loss,
            'val_d_loss': d_loss
        })
        for metric_name, metric_function in self.metrics.items():
            output[f'val_{metric_name}'] = metric_function(image, image_filled)
        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        image_dataset = ImageDataset(['../data/raw/image/Places2/data_large'], transform=self.image_transforms)
        mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/train', transform=self.mask_transforms)
        dataset = InpaintingImageDataset(image_dataset, mask_dataset)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size)
        return data_loader

    @pl.data_loader
    def val_dataloader(self):
        image_dataset = ImageDataset(['../data/raw/image/Places2/val_large'], transform=self.image_transforms)
        mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/train', transform=self.mask_transforms)
        dataset = InpaintingImageDataset(image_dataset, mask_dataset)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size)
        return data_loader

    @pl.data_loader
    def test_dataloader(self):
        image_dataset = ImageDataset(['../data/raw/image/Places2/test_large'], transform=self.image_transforms)
        mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/test', transform=self.mask_transforms)
        dataset = InpaintingImageDataset(image_dataset, mask_dataset)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size)
        return data_loader


if __name__ == '__main__':
    from argparse import Namespace

    args = {
        'batch_size': 24,
        'lr': 0.0002,
        'b1': 0.5,
        'b2': 0.999,
        'latent_dim': 100
    }
    hparams = Namespace(**args)
    gan_model = GAN(hparams)

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(gan_model)
