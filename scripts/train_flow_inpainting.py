import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from inpainting import transforms
from inpainting.load import SequenceDataset, RandomMaskDataset
from inpainting.models import GatedConvUNet
from inpainting.utils import flow_tensor_to_image_tensor, normalize_flow, denormalize_flow, get_paths


class Inpainting(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.input = None
        self.output = None
        self.target = None

        self.model = GatedConvUNet()

    def forward(self, x, mask):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        flow, mask = batch
        flow, mask = normalize_flow(flow[0]), mask[0]
        self.input = flow * (1 - mask)
        self.output = self(self.input, mask)
        self.target = flow
        hole_loss = F.l1_loss(mask * self.output, mask * self.target)
        valid_loss = F.l1_loss((1 - mask) * self.output, (1 - mask) * self.target)
        loss = hole_loss + valid_loss
        return {'loss': loss, 'log': {'loss/train': loss, 'hole_loss/train': hole_loss, 'valid_loss/train': valid_loss}}

    def validation_step(self, batch, batch_idx):
        flow, mask = batch
        flow, mask = normalize_flow(flow[0]), mask[0]
        self.input = flow * (1 - mask)
        self.output = self(self.input, mask)
        self.target = flow
        hole_loss = F.l1_loss(mask * self.output, mask * self.target)
        valid_loss = F.l1_loss((1 - mask) * self.output, (1 - mask) * self.target)
        loss = hole_loss + valid_loss
        return {'loss': loss, 'hole_loss': hole_loss, 'valid_loss': valid_loss}

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_image('input', make_grid(flow_tensor_to_image_tensor(denormalize_flow(self.input))),
                                         trainer.current_epoch)
        self.logger.experiment.add_image('output', make_grid(flow_tensor_to_image_tensor(denormalize_flow(self.output))),
                                         trainer.current_epoch)
        self.logger.experiment.add_image('target', make_grid(flow_tensor_to_image_tensor(denormalize_flow(self.target))),
                                         trainer.current_epoch)
        hole_loss = torch.stack([x['hole_loss'] for x in outputs]).mean()
        valid_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {'val_loss': loss, 'log': {'loss/val': loss, 'hole_loss/val': hole_loss, 'valid_loss/val': valid_loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


if __name__ == '__main__':
    seed_everything(42)

    train = RandomMaskDataset(
        SequenceDataset(
            get_paths('../data/processed/flow_inpainting/YouTube-VOS/train/Flows/*'),
            'flow',
            sequence_length=1
        ),
        transform=transforms.ToTensor()
    )
    train = DataLoader(train, batch_size=8, num_workers=4, shuffle=True)
    val = RandomMaskDataset(
        SequenceDataset(
            get_paths('../data/processed/flow_inpainting/YouTube-VOS/valid/Flows/*'),
            'flow',
            sequence_length=1
        ),
        transform=transforms.ToTensor()
    )
    val = DataLoader(val, batch_size=8, num_workers=4, shuffle=True)

    model = Inpainting()
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=5)
    trainer.fit(model, train, val)
