import torch


from .sa_gan import InpaintSANet


class DeepFillV2Model(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = InpaintSANet()
        weights = torch.load(path)
        self.model.load_state_dict(weights['netG_state_dict'])

    def forward(self, image, mask):
        masked_image = image * (1 - mask)
        return masked_image + self.model(masked_image, mask)[1] * mask
