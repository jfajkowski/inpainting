import glob
import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import torchvision
from spatial_correlation_sampler import SpatialCorrelationSampler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import flowiz as fz

from inpainting.load import MergeDataset, ImageDataset, VideoDataset, load_sample
from inpainting.utils import mean_and_std, denormalize


def double_conv(in_channels, out_channels, residual=True):
    if residual:
        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.adaptation = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
                self.before_skip = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)
                )
                self.after_skip = nn.Sequential(
                    nn.BatchNorm2d(out_channels),
                    nn.Tanh()
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
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )

class FlowModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._correlation = SpatialCorrelationSampler(kernel_size=1, patch_size=21, stride=1,
                                                      padding=0, dilation_patch=2)

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.dconv_up4 = double_conv(512 + 441, 2)
        self.dconv_up3 = double_conv(256 + 441 + 2, 2)
        self.dconv_up2 = double_conv(128 + 441 + 2, 2)
        self.dconv_up1 = double_conv(64 + 441 + 2, 2)

    def forward(self, x1, x2):
        conv11, conv21, conv31, x1 = self.encode(x1)
        conv12, conv22, conv32, x2 = self.encode(x2)

        cost_volume = self.correlation(x1, x2)
        x = torch.cat([x1, cost_volume], dim=1)
        x = self.dconv_up4(x)

        upsampled_flow = self.upsample(x)
        x = self.warp_tensor(conv32, upsampled_flow)
        cost_volume = self.correlation(conv31, x)
        x = torch.cat([x, cost_volume, upsampled_flow], dim=1)
        x = self.dconv_up3(x)

        upsampled_flow = self.upsample(x)
        x = self.warp_tensor(conv22, upsampled_flow)
        cost_volume = self.correlation(conv21, x)
        x = torch.cat([x, cost_volume, upsampled_flow], dim=1)
        x = self.dconv_up2(x)

        upsampled_flow = self.upsample(x)
        x = self.warp_tensor(conv12, upsampled_flow)
        cost_volume = self.correlation(conv11, x)
        x = torch.cat([x, cost_volume, upsampled_flow], dim=1)
        x = self.dconv_up1(x)

        return x

    def encode(self, x):
        conv1 = self.dconv_down1(x)
        x = self.downsample(conv1)

        conv2 = self.dconv_down2(x)
        x = self.downsample(conv2)

        conv3 = self.dconv_down3(x)
        x = self.downsample(conv3)

        x = self.dconv_down4(x)
        return conv1, conv2, conv3, x

    def correlation(self, x1, x2):
        tmp = self._correlation(x1, x2)
        s = tmp.size()
        return tmp.reshape(s[0], s[1] * s[2], s[3], s[4])

    def warp_tensor(self, x, flow):
        assert x.size()[-2:] == flow.size()[-2:]
        grid = self.make_grid(x.size())
        grid += 2 * flow
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    @staticmethod
    def make_grid(size, normalized=True):
        _, _, h, w = size
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
        if normalized:
            grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
            grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        return grid

    def training_step(self, batch, batch_nb):
        images = batch

        first = images[0]
        second = images[1]

        forward_flow = self(first, second)
        backward_flow = self(second, first)

        warped_second = self.warp_tensor(images[1], forward_flow)
        warped_first = self.warp_tensor(images[0], backward_flow)

        loss = F.mse_loss(warped_second, first) + \
               F.mse_loss(warped_first, second)

        x = torch.cat([
            denormalize(first, 'imagenet'),
            denormalize(second, 'imagenet'),
            self.flow_to_image_tensor(forward_flow),
            self.flow_to_image_tensor(backward_flow),
            denormalize(warped_second, 'imagenet'),
            denormalize(warped_first, 'imagenet'),
        ], dim=0)
        grid = torchvision.utils.make_grid(x)

        self.logger.experiment.add_image('generated_images', grid, trainer.current_epoch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    @staticmethod
    def flow_to_image_tensor(flow):
        numpy_flow = flow.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        return torch.from_numpy(fz.convert_from_flow(numpy_flow)).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':
    dataset = VideoDataset(
        list(glob.glob(f'../data/raw/DAVIS/JPEGImages/rollerblade')),
        'image',
        sequence_length=2,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(*mean_and_std('imagenet'))
        ])
    )
    loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

    model = FlowModel()
    # model = ClassicUNet.load_from_checkpoint(checkpoint_path='lightning_logs/version_0/checkpoints/epoch=17.ckpt')
    trainer = pl.Trainer(gpus=1, precision=32)
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
