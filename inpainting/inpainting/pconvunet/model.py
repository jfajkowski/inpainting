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
        image = normalize(image, mode='imagenet')
        masked_image = image * (1 - mask)
        result = masked_image + self.model(masked_image, 1 - mask.repeat(1, 3, 1, 1))[0] * mask
        return denormalize(result, mode='imagenet')
