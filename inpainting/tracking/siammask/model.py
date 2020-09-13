import numpy as np

from argparse import Namespace

import torch

from inpainting.tracking.siammask.config_helper import load_config
from inpainting.tracking.siammask.custom import Custom
from inpainting.tracking.siammask.load_helper import load_pretrain
from inpainting.tracking.siammask.test import siamese_init, siamese_track
from inpainting.utils import tensor_to_cv_image


class SiamMaskModel(torch.nn.Module):

    def __init__(self, config_path, weights_path):
        super().__init__()
        args = Namespace(
            config=config_path
        )
        self.cfg = load_config(args)
        self.model = Custom(anchors=self.cfg['anchors'])
        load_pretrain(self.model, weights_path)
        self.state = None

    def initialize(self, image, roi):
        device = image.device
        image = tensor_to_cv_image(image)
        (x1, y1), (x2, y2) = roi
        w, h = x2 - x1, y2 - y1
        target_pos = np.array([x1 + w / 2, y1 + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(image, target_pos, target_sz, self.model, self.cfg['hp'], device)

    def forward(self, image):
        device = image.device
        image = tensor_to_cv_image(image)
        self.state = siamese_track(self.state, image, mask_enable=True, refine_enable=True, device=device)
        mask = self.state['mask'] > self.state['p'].seg_thr
        return torch.tensor(mask).unsqueeze(0).float()
