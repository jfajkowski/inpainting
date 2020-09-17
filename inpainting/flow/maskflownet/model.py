import torch

from inpainting.flow.maskflownet.MaskFlownet import MaskFlownet_S, Upsample


class MaskFlowNetModel(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = MaskFlownet_S(path)
        weights = torch.load(path)
        self.model.load_state_dict(weights)

    def forward(self, image_1, image_2):
        result = self.model(image_1, image_2)[0][-1]
        result = Upsample(result, 4)
        return torch.nn.functional.interpolate(result,
                                               size=[image_1.size(2), image_1.size(3)],
                                               mode='nearest').flip(1)


