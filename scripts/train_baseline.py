from argparse import Namespace
from collections import OrderedDict

import numpy as np
import opencv_transforms.transforms as transforms
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from deepstab.configuration import MyLogger
from deepstab.load import ImageDataset, FileMaskDataset, InpaintingImageDataset
from deepstab.loss import ReconstructionLoss
from deepstab.metrics import PSNR, MAE, MSE
from deepstab.model_autoencoder import GatingConvolutionAutoencoder
from deepstab.utils import mask_tensor, denormalize, list_of_dicts_to_dict_of_lists, mean_and_std


class Baseline(pl.LightningModule):

    def __init__(self, hparams):
        super(Baseline, self).__init__()
        self.hparams = hparams

        self.generator = GatingConvolutionAutoencoder()

        self.reconstruction_criterion = ReconstructionLoss(
            self.hparams.pixel_loss_weight,
            self.hparams.content_loss_weight,
            self.hparams.style_loss_weight,
            self.hparams.tv_loss_weight
        )

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

    def training_step(self, batch, batch_nb):
        image, mask = batch

        self.image_filled, _ = self.forward(image, mask)

        g_reconstruction_loss, g_pixel_loss, g_content_loss, g_style_loss, g_tv_loss = self.reconstruction_criterion(
            self.image_filled, image)
        g_loss = g_reconstruction_loss

        log = OrderedDict({
            'loss/g_loss/reconstruction': g_reconstruction_loss,
            'loss/g_loss/reconstruction/pixel': g_pixel_loss,
            'loss/g_loss/reconstruction/content': g_content_loss,
            'loss/g_loss/reconstruction/style': g_style_loss,
            'loss/g_loss/reconstruction/tv': g_tv_loss,
            'loss/g_loss': g_loss,
            **self._prepare_accuracy_log(image, self.image_filled, self.accuracy_criteria)
        })
        output = OrderedDict({
            'loss': g_loss,
            'progress_bar': log,
            'log': self._wrap_log(log, 'train')
        })
        return output

    def validation_step(self, batch, batch_nb):
        image, mask = batch

        self.image_filled, _ = self.forward(image, mask)

        g_reconstruction_loss, g_pixel_loss, g_content_loss, g_style_loss, g_tv_loss = self.reconstruction_criterion(
            self.image_filled, image)
        g_loss = g_reconstruction_loss

        output = OrderedDict({
            'loss/g_loss/reconstruction': g_reconstruction_loss,
            'loss/g_loss/reconstruction/pixel': g_pixel_loss,
            'loss/g_loss/reconstruction/content': g_content_loss,
            'loss/g_loss/reconstruction/style': g_style_loss,
            'loss/g_loss/reconstruction/tv': g_tv_loss,
            'loss/g_loss': g_loss,
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
        return [opt_g], []

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
        image_dataset = ImageDataset(['data/raw/image/Places2/data_large'], transform=image_transforms)
        mask_dataset = FileMaskDataset('data/raw/mask/qd_imd/train', transform=mask_transforms)
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
        image_dataset = ImageDataset(['data/raw/image/Places2/val_large'], transform=image_transforms)
        mask_dataset = FileMaskDataset('data/raw/mask/qd_imd/train', transform=mask_transforms)
        dataset = InpaintingImageDataset(image_dataset, mask_dataset)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)
        return data_loader

    def on_after_backward(self):
        if self.trainer.global_step % 1000 == 0:
            for name, weight in self.generator.named_parameters():
                if torch.isfinite(weight.grad).any():
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
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = 'baseline_bottleneck'
    args = {
        'batch_size': 16,
        'lr': 0.0005,
        'b1': 0.5,
        'b2': 0.999,
        'pixel_loss_weight': 0.5,
        'content_loss_weight': 0.02,
        'style_loss_weight': 1000,
        'tv_loss_weight': 0.0
    }
    hparams = Namespace(**args)
    model = Baseline(hparams)

    trainer = pl.Trainer(default_save_path=f'models', gpus=1, use_amp=True,
                         train_percent_check=0.0025,
                         val_percent_check=0.005,
                         test_percent_check=0.005,
                         logger=MyLogger(model_name),
                         early_stop_callback=False)
    trainer.fit(model)
