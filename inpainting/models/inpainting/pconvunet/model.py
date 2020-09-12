import torch

from inpainting.utils import denormalize, normalize
from .net import PConvUNet


class PConvUNetModel(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = PConvUNet()
        weights = torch.load(path)
        self.model.load_state_dict(weights['model'])

    def forward(self, image, mask):
        masked_image = normalize(denormalize(image), mode='imagenet') * (1 - mask)
        result = self.model(masked_image, 1 - mask.repeat(1, 3, 1, 1))[0]
        return normalize(denormalize(masked_image + result * mask, mode='imagenet'))
