import torch

from inpainting.custom.liteflownet.model import LiteFlowNetModel
from inpainting.custom.pwcnet.model import PWCNetModel
from inpainting.external.algorithms import DeepFlowGuidedVideoInpaintingAlgorithm, VideoInpaintingAlgorithm


class MyDeepFlowGuidedVideoInpaintingAlgorithm(DeepFlowGuidedVideoInpaintingAlgorithm):
    def __init__(self, eps=20):
        super().__init__(eps)
        self.flow_model = PWCNetModel().cuda().eval()


class NoopVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def initialize(self):
        pass

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        return current_frame
