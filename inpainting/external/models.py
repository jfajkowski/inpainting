import math
from argparse import Namespace
from os.path import isfile

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

from inpainting.external.liteflownet import Network
from inpainting.external.deepfillv1 import DeepFill
from inpainting.external.flownet2 import FlowNet2
from inpainting.external.siammask.config_helper import load_config
from inpainting.external.siammask.custom import Custom
from inpainting.external.siammask.load_helper import load_pretrain
from inpainting.external.siammask.test import siamese_init, siamese_track


class DeepFillV1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DeepFill.Generator()
        weights = torch.load('models/external/deepfillv1/imagenet_deepfill.pth')
        self.model.load_state_dict(weights)

    def forward(self, masked_image, mask):
        mask = 1 - mask
        small_mask = f.interpolate(mask, scale_factor=1 / 8, mode='nearest')
        return self.model(masked_image, mask, small_mask)[1]


class FlowNet2Model(nn.Module):
    def __init__(self):
        super().__init__()
        args = Namespace(
            rgb_max=1.0,
            fp16=False
        )
        self.model = FlowNet2(args)
        flownet2_ckpt = torch.load('models/external/flownet2/FlowNet2_checkpoint.pth.tar')
        self.model.load_state_dict(flownet2_ckpt['state_dict'])

    def forward(self, image_1, image_2):
        return self.model(image_1, image_2)


class LiteFlowNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Network('models/external/liteflownet/network-default.pytorch')

    def forward(self, image_1, image_2):
        return estimate_flow(self.model, image_1, image_2)


def estimate_flow(model, x_1, x_2):
    assert (x_1.size(2) == x_2.size(2))
    assert (x_1.size(3) == x_2.size(3))

    height = x_1.size(2)
    width = x_1.size(3)

    preprocessed_width = int(math.floor(math.ceil(width / 64.0) * 64.0))
    preprocessed_height = int(math.floor(math.ceil(height / 64.0) * 64.0))

    # Convert to BGR
    x_1 = x_1.flip(1)
    x_2 = x_2.flip(1)

    tensor_preprocessed_first = torch.nn.functional.interpolate(input=x_1,
                                                                size=(preprocessed_height, preprocessed_width),
                                                                mode='bilinear', align_corners=False)
    tensor_preprocessed_second = torch.nn.functional.interpolate(input=x_2,
                                                                 size=(preprocessed_height, preprocessed_width),
                                                                 mode='bilinear', align_corners=False)

    flow = torch.nn.functional.interpolate(
        input=model(tensor_preprocessed_first, tensor_preprocessed_second), size=(height, width),
        mode='bilinear', align_corners=False)

    flow[:, 0, :, :] *= float(width) / float(preprocessed_width)
    flow[:, 1, :, :] *= float(height) / float(preprocessed_height)

    return flow


class SiamMaskModel(nn.Module):
    def __init__(self, init_image, init_roi, device='cuda'):
        super().__init__()
        args = Namespace(
            config='models/external/siammask/config_davis.json',
            resume='models/external/siammask/SiamMask_DAVIS.pth'
        )
        cfg = load_config(args)
        self.model = Custom(anchors=cfg['anchors']).eval()
        if args.resume:
            assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            self.model = load_pretrain(self.model, args.resume)
        self.model.eval().to(device)

        x, y, w, h = init_roi
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(init_image, target_pos, target_sz, self.model, cfg['hp'], device=device)
        self.device = device

    def forward(self, image):
        self.state = siamese_track(self.state, image, mask_enable=True, refine_enable=True, device=self.device)
        location = self.state['ploygon'].flatten()
        mask = self.state['mask'] > self.state['p'].seg_thr
        return mask, location
