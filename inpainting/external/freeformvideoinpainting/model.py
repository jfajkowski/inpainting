import torch
import torch.nn as nn

from .video_inpainting_model import VideoInpaintingModel


class FreeFormVideoInpaintingModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        checkpoint = torch.load(path)
        self.model = VideoInpaintingModel(**checkpoint['config']['arch']['args'])
        self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self, images, masks):
        return self.model(images, masks)['outputs'].clamp(0, 1)
