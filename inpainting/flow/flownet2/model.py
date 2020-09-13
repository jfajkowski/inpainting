from argparse import Namespace

import torch

from .FlowNet2 import FlowNet2


class FlowNet2Model(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        args = Namespace(
            rgb_max=1.0,
            fp16=False
        )
        self.model = FlowNet2(args)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self, image_1, image_2):
        return self.model(image_1, image_2)
