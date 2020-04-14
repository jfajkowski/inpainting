import torch

from inpainting.custom.liteflownet.model import LiteFlowNetModel
from inpainting.custom.pwcnet.model import PWCNetModel
from inpainting.external.algorithms import DeepFlowGuidedVideoInpaintingAlgorithm, VideoInpaintingAlgorithm
from inpainting.utils import dilate_tensor


class MyDeepFlowGuidedVideoInpaintingAlgorithm(DeepFlowGuidedVideoInpaintingAlgorithm):
    def __init__(self, eps=20, dilation_size=5, dilation_iterations=5):
        super().__init__(eps)
        self.dilation_size = dilation_size
        self.dilation_iterations = dilation_iterations
        self.flow_model = PWCNetModel().cuda().eval()

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        current_mask = dilate_tensor(current_mask, self.dilation_size, self.dilation_iterations)
        return super().inpaint_online(current_frame, current_mask)


class NoopVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def initialize(self):
        pass

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        return current_frame
