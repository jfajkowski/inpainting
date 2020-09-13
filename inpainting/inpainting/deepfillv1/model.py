import torch

from inpainting.utils import normalize, denormalize
from .DeepFill import Generator


class DeepFillV1Model(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = Generator()
        weights = torch.load(path)
        self.model.load_state_dict(weights)

    def forward(self, image, mask):
        image = normalize(image)
        masked_image = image * (1 - mask)
        small_mask = torch.nn.functional.interpolate(mask, scale_factor=1 / 8, mode='nearest')
        result = masked_image + self.model(masked_image, mask, small_mask)[1] * mask
        return denormalize(result)
