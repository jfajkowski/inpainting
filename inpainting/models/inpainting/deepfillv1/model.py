import torch


from .DeepFill import Generator


class DeepFillV1Model(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = Generator()
        weights = torch.load(path)
        self.model.load_state_dict(weights)

    def forward(self, image, mask):
        masked_image = image * (1 - mask)
        small_mask = torch.nn.functional.interpolate(mask, scale_factor=1 / 8, mode='nearest')
        return self.model(masked_image, mask, small_mask)[1]
