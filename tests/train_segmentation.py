import glob

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.utils import make_grid

import inpainting.transforms as T
from inpainting.load import MergeDataset, ImageDataset
from inpainting.utils import annotation_to_mask


class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.input = None
        self.target = None
        self.output = None

        self.backbone = list(deeplabv3_resnet50(pretrained=True).children())[0]
        self.head = DeepLabHead(2048, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)['out']
        x = self.head(x)
        return F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

    def training_step(self, batch, batch_nb):
        image, mask = batch
        self.input = image
        self.target = 1 - mask
        self.output = self(image)

        loss = F.binary_cross_entropy_with_logits(self.output, self.target)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        self.logger.experiment.add_image('input', make_grid(self.input), trainer.current_epoch)
        self.logger.experiment.add_image('target', make_grid(self.target), trainer.current_epoch)
        self.logger.experiment.add_image('output', make_grid(torch.sigmoid(self.output)), trainer.current_epoch)
        return {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    input_images_dataset = ImageDataset(
        list(glob.glob(f'../data/raw/DAVIS/JPEGImages/*')),
        'image'
    )
    input_masks_dataset = ImageDataset(
        list(glob.glob(f'../data/raw/DAVIS/Annotations/*')),
        'annotation',
        transform=T.Lambda(annotation_to_mask)
    )
    dataset = MergeDataset([input_images_dataset, input_masks_dataset],
                           transform=T.Compose([
                               T.Resize((256, 256)),
                               T.ToTensor(),
                           ]))
    loader = DataLoader(dataset, shuffle=True, batch_size=16, num_workers=4)

    model = SegmentationModel()
    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.fit(model, loader)
