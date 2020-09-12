import glob

import pytorch_lightning as pl
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from inpainting.external.deepflowguidedvideoinpainting.flownet2.submodules import *
from inpainting.load import VideoDataset, VideoObjectRemovalDataset


class InpaintingModel(pl.LightningModule):

    def __init__(self, n=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n = n
        self.images = None
        self.target = None
        self.output = None

        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.inpainting_model = nn.Sequential(
            double_conv(3 * self.n, 64),
            self.downsample,
            double_conv(64, 128),
            self.downsample,
            double_conv(128, 256),
            self.downsample,
            double_conv(256, 512),
            self.downsample,
            SelfAttentionLayer(512),
            self.upsample,
            double_conv(512, 256),
            self.upsample,
            double_conv(256, 128),
            self.upsample,
            double_conv(128, 64),
            self.upsample,
            double_conv(64, 3)
        )

    def forward(self, images, masks):


        return self.inpainting_model()

    def training_step(self, batch, batch_nb):
        self.images, masks, self.target = batch[0], batch[1], batch[2][self.n // 2]
        self.output = self(self.images, masks)

        loss = F.l1_loss(self.output, self.target)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        self.logger.experiment.add_image('images', make_grid(self.images[self.n // 2]), trainer.current_epoch)
        self.logger.experiment.add_image('target', make_grid(self.target), trainer.current_epoch)
        self.logger.experiment.add_image('output', make_grid(self.output), trainer.current_epoch)
        return {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':
    images_dataset = VideoDataset(
        list(sorted(glob.glob(f'../data/raw/DAVIS/JPEGImages/*'))),
        'image',
        sequence_length=3,
        transform=transforms.Resize(size=[256, 256], interpolation=Image.BILINEAR)
    )
    masks_dataset = VideoDataset(
        list(sorted(glob.glob(f'../data/interim/DAVIS/Masks/*'))),
        'mask',
        sequence_length=3,
        transform=transforms.Resize(size=[256, 256], interpolation=Image.NEAREST)
    )

    dataset = VideoObjectRemovalDataset(images_dataset, masks_dataset, transform=transforms.ToTensor())
    loader = DataLoader(dataset, shuffle=True, batch_size=4, num_workers=4)

    model = InpaintingModel()
    trainer = pl.Trainer(gpus=1, precision=32)
    trainer.fit(model, loader)
