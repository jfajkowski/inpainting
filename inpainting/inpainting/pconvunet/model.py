import torch

from inpainting.utils import denormalize_image, normalize_image
from .net import PConvUNet


class PConvUNetModel(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = PConvUNet()
        weights = torch.load(path)
        self.model.load_state_dict(weights['model'])

    def forward(self, image, mask):
        image = normalize_image(image, mode='imagenet')
        masked_image = image * (1 - mask)
        result = masked_image + self.model(masked_image, 1 - mask.repeat(1, 3, 1, 1))[0] * mask
        return denormalize_image(result, mode='imagenet')
