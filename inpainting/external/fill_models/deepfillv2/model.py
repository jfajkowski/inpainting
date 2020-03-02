from argparse import Namespace

import torch
import torch.nn as nn

from inpainting.external.fill_models.deepfillv2.network import GatedGenerator


class FillModel(nn.Module):
    def __init__(self):
        super().__init__()
        state = torch.load('models/fill_models/deepfillv2/deepfillv2.pth')
        opt = Namespace(**{
            'in_channels': 4,
            'out_channels': 3,
            'latent_channels': 64,
            'pad_type': 'reflect',
            'activation': 'lrelu',
            'norm': 'in',
            'init_type': 'normal',
            'init_gain': 0.02
        })
        self.model = GatedGenerator(opt)
        self.model.load_state_dict(state['generator'])

    def forward(self, x, m):
        _, x2 = self.model(x, m)
        return x * (1 - m) + x2 * m
