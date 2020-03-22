from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as f

from inpainting.external.deepflowguidedvideoinpainting.deepfillv1 import DeepFill
from inpainting.external.deepflowguidedvideoinpainting.flownet2 import FlowNet2


class DeepFillV1Model(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = DeepFill.Generator()
        weights = torch.load(path)
        self.model.load_state_dict(weights)

    def forward(self, masked_image, mask):
        mask = 1 - mask
        small_mask = f.interpolate(mask, scale_factor=1 / 8, mode='nearest')
        return self.model(masked_image, mask, small_mask)[1]


class FlowNet2Model(nn.Module):
    def __init__(self, path):
        super().__init__()
        args = Namespace(
            rgb_max=1.0,
            fp16=False
        )
        self.model = FlowNet2(args)
        flownet2_ckpt = torch.load(path)
        self.model.load_state_dict(flownet2_ckpt['state_dict'])

    def forward(self, image_1, image_2):
        return self.model(image_1, image_2)
