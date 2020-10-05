import glob

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from inpainting import transforms
from inpainting.inpainting.flowfill import UNet, Dilated
from inpainting.load import SequenceDataset, RandomMaskDataset
from inpainting.utils import normalize_flow, flow_tensor_to_image_tensor


class InpaintingModel(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_flow = None
        self.target_flow = None
        self.output_flow = None

        self.model = Dilated()

    def forward(self, flow, mask):
        return flow * (1 - mask) + self.model(flow, mask) * mask

    def training_step(self, batch, batch_idx):
        flow, mask = batch
        flow, mask = flow[0], mask[0]
        self.input_flow = normalize_flow(flow * (1 - mask))
        self.target_flow = normalize_flow(flow)
        self.output_flow = self(self.input_flow, mask)

        loss = torch.norm(self.target_flow - self.output_flow).mean()

        tensorboard_logs = {
            'loss': loss
        }
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        self.logger.experiment.add_image('input_flow', make_grid(flow_tensor_to_image_tensor(self.input_flow)),
                                         trainer.current_epoch)
        self.logger.experiment.add_image('target_flow', make_grid(flow_tensor_to_image_tensor(self.target_flow)),
                                         trainer.current_epoch)
        self.logger.experiment.add_image('output_flow', make_grid(flow_tensor_to_image_tensor(self.output_flow)),
                                         trainer.current_epoch)
        return {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'loss'}]


if __name__ == '__main__':
    seed_everything(42)

    dataset = RandomMaskDataset(
        SequenceDataset(
            list(sorted(glob.glob('../data/interim/MPI-Sintel-complete/training/flow/*'))),
            'flow',
            sequence_length=1
        ),
        transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=8, num_workers=2, shuffle=True)

    model = InpaintingModel()
    trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=1000, max_epochs=1000)
    trainer.fit(model, loader)
