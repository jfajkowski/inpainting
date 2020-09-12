import glob

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from inpainting.load import MergeDataset, ImageDataset
from inpainting.utils import mean_and_std


def double_conv(in_channels, out_channels, residual=True):
    if residual:
        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.adaptation = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                self.before_skip = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)
                )
                self.after_skip = nn.Sequential(
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                x = self.adaptation(x)
                x = x + self.before_skip(x)
                return self.after_skip(x)
        return ResidualBlock()
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class SegmentationModel(pl.LightningModule):
    def __init__(self, n_class):
        super().__init__()
        self.save_hyperparameters()

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.downsample(conv1)

        conv2 = self.dconv_down2(x)
        x = self.downsample(conv2)

        conv3 = self.dconv_down3(x)
        x = self.downsample(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

    def training_step(self, batch, batch_nb):
        image, mask = batch
        output = self(image)
        loss = F.binary_cross_entropy_with_logits(output, mask)

        grid = torchvision.utils.make_grid(F.sigmoid(output))
        self.logger.experiment.add_image('generated_images', grid, trainer.current_epoch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':
    input_images_dataset = ImageDataset(
        list(glob.glob(f'../data/raw/DAVIS/JPEGImages/tennis')),
        'image',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(*mean_and_std('imagenet'))
        ])
    )
    input_masks_dataset = ImageDataset(
        list(glob.glob(f'../data/interim/DAVIS/Masks/tennis')),
        'mask',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    )
    dataset = MergeDataset([input_images_dataset, input_masks_dataset])
    loader = DataLoader(dataset, shuffle=True, batch_size=16, num_workers=0)

    model = SegmentationModel(1)
    # model = ClassicUNet.load_from_checkpoint(checkpoint_path='lightning_logs/version_0/checkpoints/epoch=17.ckpt')
    trainer = pl.Trainer(gpus=1, precision=16)
    # trainer = pl.Trainer(resume_from_checkpoint='lightning_logs/version_0/checkpoints/epoch=4.ckpt')
    trainer.fit(model, loader)
    # x, y_true = next(iter(loader))
    # y_pred = model(x)
    # x_grid = torchvision.utils.make_grid(x)
    # y_pred_grid = torchvision.utils.make_grid(y_pred)
    # y_true_grid = torchvision.utils.make_grid(y_true)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(transforms.functional.to_pil_image(x_grid))
    # plt.show()
    # plt.imshow(transforms.functional.to_pil_image(y_true_grid))
    # plt.show()
    # plt.imshow(transforms.functional.to_pil_image(y_pred_grid))
    # plt.show()
