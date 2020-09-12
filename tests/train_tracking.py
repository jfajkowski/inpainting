import glob

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.utils import make_grid

import inpainting.transforms as T
from inpainting.external.siammask.resnet import resnet50
from inpainting.layers import AttentionLayer, CorrelationLayer
from inpainting.load import ImageDataset, MergeDataset


class TrackingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.input_search = None
        self.input_exemplar = None
        self.target = None
        self.output = None

        self.backbone = resnet50()
        self.matcher = CorrelationLayer(512, 256)
        self.decoder = FCNHead(256, 1)

    def forward(self, x, z):
        input_shape = x.shape[-2:]
        x_conv1, x_conv2, x_conv3, x_conv4 = self.backbone(x)
        z_conv1, z_conv2, z_conv3, z_conv4 = self.backbone(z)

        out = self.matcher(x_conv4, z_conv4)
        out = self.decoder(out)
        return F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)

    def training_step(self, batch, batch_nb):
        search_image, exemplar_image, search_mask = batch

        self.input_search = search_image
        self.input_exemplar = exemplar_image
        self.target = search_mask
        self.output = self(self.input_search, self.input_exemplar)

        loss = F.binary_cross_entropy_with_logits(self.output, self.target)

        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        self.logger.experiment.add_image('input_search', make_grid(self.input_search), trainer.current_epoch)
        self.logger.experiment.add_image('input_exemplar', make_grid(self.input_exemplar), trainer.current_epoch)
        self.logger.experiment.add_image('target', make_grid(self.target), trainer.current_epoch)
        self.logger.experiment.add_image('output', make_grid(torch.sigmoid(self.output)), trainer.current_epoch)
        return {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    search_image_dataset = ImageDataset(
        sorted(list(glob.glob(f'../data/processed/DAVIS/Tracking/SearchImage'))),
        'image'
    )
    exemplar_image_dataset = ImageDataset(
        sorted(list(glob.glob(f'../data/processed/DAVIS/Tracking/ExemplarImage'))),
        'image'
    )
    search_mask_dataset = ImageDataset(
        list(glob.glob(f'../data/processed/DAVIS/Tracking/SearchMask')),
        'mask'
    )
    dataset = MergeDataset([search_image_dataset, exemplar_image_dataset, search_mask_dataset],
                           transform=T.Compose([
                               T.Resize((256, 256)),
                               T.RandomAffine(30),
                               T.ToTensor()
                           ]))
    loader = DataLoader(dataset, shuffle=True, batch_size=8, num_workers=4)

    model = TrackingModel()
    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.fit(model, loader)
