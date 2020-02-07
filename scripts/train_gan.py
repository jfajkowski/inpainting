from collections import OrderedDict

import opencv_transforms.transforms as transforms
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from deepstab.configuration import MyLogger
from deepstab.load import ImageDataset, FileMaskDataset, InpaintingImageDataset
from deepstab.loss import ReconstructionLoss
from deepstab.metrics import PSNR, MAE, MSE
from deepstab.model_discriminator import Discriminator
from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.utils import mask_tensor, denormalize, list_of_dicts_to_dict_of_lists, mean_and_std


class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        self.generator = GatingConvolutionUNet()
        self.discriminator = Discriminator()

        self.reconstruction_criterion = ReconstructionLoss(
            self.hparams.pixel_loss_weight,
            self.hparams.content_loss_weight,
            self.hparams.style_loss_weight,
            self.hparams.tv_loss_weight
        )
        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss()

        self.accuracy_criteria = {
            'psnr': PSNR(1),
            'mae': MAE(),
            'mse': MSE()
        }

        self.image_filled = None
        self.example = None

    def forward(self, image, mask):
        image_masked = mask_tensor(image, mask)
        image_filled = self.generator(image_masked, mask)
        return image_filled, image_masked

    def training_step(self, batch, batch_nb, optimizer_idx):
        image, mask = batch
        real = torch.ones((self.hparams.batch_size, 16384))
        fake = torch.zeros((self.hparams.batch_size, 16384))
        if self.on_gpu:
            real = real.cuda(image.device.index)
            fake = fake.cuda(image.device.index)

        if optimizer_idx == 0:
            self.image_filled, _ = self.forward(image, mask)

            g_reconstruction_loss, g_pixel_loss, g_content_loss, g_style_loss, g_tv_loss = self.reconstruction_criterion(
                self.image_filled, image)
            g_adversarial_loss = self.adversarial_criterion(self.discriminator(self.image_filled), real)
            g_loss = (g_reconstruction_loss + g_adversarial_loss) / 2

            log = OrderedDict({
                'loss/g_loss/reconstruction': g_reconstruction_loss,
                'loss/g_loss/reconstruction/pixel': g_pixel_loss,
                'loss/g_loss/reconstruction/content': g_content_loss,
                'loss/g_loss/reconstruction/style': g_style_loss,
                'loss/g_loss/reconstruction/tv': g_tv_loss,
                'loss/g_loss/adversarial': g_adversarial_loss,
                'loss/g_loss': g_loss,
                **self._prepare_accuracy_log(image, self.image_filled, self.accuracy_criteria)
            })
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': log,
                'log': self._wrap_log(log, 'train')
            })
            return output

        if optimizer_idx == 1:
            d_real_loss = self.adversarial_criterion(self.discriminator(image), real)
            d_fake_loss = self.adversarial_criterion(self.discriminator(self.image_filled.detach()), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2

            log = {
                'loss/d_loss': d_loss
            }
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': log,
                'log': self._wrap_log(log, 'train')
            })
            return output

    def validation_step(self, batch, batch_nb):
        image, mask = batch
        real = torch.ones((self.hparams.batch_size, 16384))
        fake = torch.zeros((self.hparams.batch_size, 16384))
        if self.on_gpu:
            real = real.cuda(image.device.index)
            fake = fake.cuda(image.device.index)

        self.image_filled, _ = self.forward(image, mask)

        g_reconstruction_loss, g_pixel_loss, g_content_loss, g_style_loss, g_tv_loss = self.reconstruction_criterion(
            self.image_filled, image)
        g_adversarial_loss = self.adversarial_criterion(self.discriminator(self.image_filled), real)
        g_loss = (g_reconstruction_loss + g_adversarial_loss) / 2

        d_real_loss = self.adversarial_criterion(self.discriminator(image), real)
        d_fake_loss = self.adversarial_criterion(self.discriminator(self.image_filled.detach()), fake)
        d_loss = (d_real_loss + d_fake_loss) / 2

        output = OrderedDict({
            'loss/g_loss/reconstruction': g_reconstruction_loss,
            'loss/g_loss/reconstruction/pixel': g_pixel_loss,
            'loss/g_loss/reconstruction/content': g_content_loss,
            'loss/g_loss/reconstruction/style': g_style_loss,
            'loss/g_loss/reconstruction/tv': g_tv_loss,
            'loss/g_loss/adversarial': g_adversarial_loss,
            'loss/g_adversarial_loss': g_adversarial_loss,
            'loss/g_loss': g_loss,
            'loss/d_loss': d_loss,
            **self._prepare_accuracy_log(image, self.image_filled, self.accuracy_criteria)
        })
        return output

    def validation_end(self, outputs):
        metrics_lists = list_of_dicts_to_dict_of_lists(outputs)
        metrics_means = {}
        for key, values in metrics_lists.items():
            metrics_means[key] = torch.mean(torch.stack(values))
        output = OrderedDict({
            'val_loss': metrics_means['loss/g_loss'],
            'log': self._wrap_log(metrics_means, 'val')
        })
        return output

    def _prepare_accuracy_log(self, image, image_filled, accuracy_criteria):
        log = OrderedDict()
        image = denormalize(image)
        image_filled = denormalize(image_filled)
        for name, function in accuracy_criteria.items():
            log['accuracy/' + name] = function(image, image_filled)
        return log

    def _wrap_log(self, log, key):
        result = OrderedDict()
        for k, v in log.items():
            result[k] = {key: v}
        return result

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        image_transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flip(0)),
            transforms.Normalize(*mean_and_std())
        ])
        mask_transforms = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomResizedCrop(256),
            transforms.ToTensor()
        ])
        image_dataset = ImageDataset(['../data/raw/image/Places2/data_large'], transform=image_transforms)
        mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/train', transform=mask_transforms)
        dataset = InpaintingImageDataset(image_dataset, mask_dataset)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
        return data_loader

    @pl.data_loader
    def val_dataloader(self):
        image_transforms = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flip(0)),
            transforms.Normalize(*mean_and_std())
        ])
        mask_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        image_dataset = ImageDataset(['../data/raw/image/Places2/val_large'], transform=image_transforms)
        mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/train', transform=mask_transforms)
        dataset = InpaintingImageDataset(image_dataset, mask_dataset)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)
        return data_loader

    def on_after_backward(self):
        if self.trainer.global_step % 1000 == 0:
            for name, weight in self.generator.named_parameters():
                self.logger.experiment.add_histogram(f'g_{name}/weight', weight, self.trainer.global_step)
                self.logger.experiment.add_histogram(f'g_{name}/gradient', weight.grad, self.trainer.global_step)

    def on_epoch_end(self):
        if not self.example:
            self.example = next(iter(self.val_dataloader()[0]))
        image, mask = self.example
        image, mask = image.cuda(), mask.cuda()
        image_filled, image_masked = self.forward(image, mask)
        self.logger.experiment.add_images(f'image', denormalize(image), self.current_epoch)
        self.logger.experiment.add_images(f'mask', mask, self.current_epoch)
        self.logger.experiment.add_images(f'image_masked', denormalize(image_masked), self.current_epoch)
        self.logger.experiment.add_images(f'image_filled', denormalize(image_filled), self.current_epoch)


if __name__ == '__main__':
    from argparse import Namespace

    model_name = 'gan'
    args = {
        'batch_size': 16,
        'lr': 0.0001,
        'b1': 0.5,
        'b2': 0.999,
        'pixel_loss_weight': 0.5,
        'content_loss_weight': 0.1,
        'style_loss_weight': 1000,
        'tv_loss_weight': 0.1
    }
    hparams = Namespace(**args)
    model = GAN(hparams)

    trainer = pl.Trainer(default_save_path=f'../models/{model_name}', gpus=1, use_amp=True,
                         train_percent_check=0.0025,
                         val_percent_check=0.005,
                         test_percent_check=0.005,
                         logger=MyLogger(model_name),
                         early_stop_callback=False)
    trainer.fit(model)
