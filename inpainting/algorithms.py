import abc
from argparse import Namespace
from typing import List

from inpainting.flow.flownet2.model import FlowNet2Model
from inpainting.flow.liteflownet.model import LiteFlowNetModel
from inpainting.flow.pwcnet.model import PWCNetModel
from inpainting.inpainting.deepfillv1.model import DeepFillV1Model
from inpainting.inpainting.deepfillv2.model import DeepFillV2Model
from inpainting.inpainting.kernel_inpainting import Inpainter
from inpainting.inpainting.pconvunet.model import PConvUNetModel
from inpainting.inpainting.region_fill import inpaint
from inpainting.tracking.siammask.model import SiamMaskModel
from inpainting.utils import make_grid, warp_tensor, normalize_flow, dilate_mask

import numpy as np
import torch
from inpainting.utils import invert_mask, tensor_to_cv_image
from inpainting.visualize import save_frame


class VideoInpaintingAlgorithm(abc.ABC):

    def initialize(self):
        pass

    def inpaint(self, images: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []
        for image, mask in zip(images, masks):
            result.append(self.inpaint_online(image, mask))
        return result

    @abc.abstractmethod
    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        pass


class SingleFrameVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self, image_inpainting_model='DeepFillV1'):
        if image_inpainting_model == 'RegionFill':
            self.image_inpainting_model = inpaint
        elif image_inpainting_model == 'KernelFill':
            self.image_inpainting_model = Inpainter()
        elif image_inpainting_model == 'DeepFillv1':
            self.image_inpainting_model = DeepFillV1Model(
                'models/inpainting/deepfillv1/imagenet_deepfill.pth').cuda().eval()
        elif image_inpainting_model == 'DeepFillv2':
            self.image_inpainting_model = DeepFillV2Model(
                'models/inpainting/deepfillv2/latest_ckpt.pth.tar').cuda().eval()
        elif image_inpainting_model == 'PConvUNet':
            self.image_inpainting_model = PConvUNetModel('models/inpainting/pconvunet/1000000.pth').cuda().eval()
        else:
            raise ValueError(image_inpainting_model)

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        return self.image_inpainting_model(current_image, current_mask)


class FlowGuidedVideoInpaintingAlgorithm(SingleFrameVideoInpaintingAlgorithm):
    def __init__(self, eps=1, flow_model='FlowNet2', image_inpainting_model='DeepFillv1'):
        super().__init__(image_inpainting_model)

        self.eps = eps

        if flow_model == 'FlowNet2':
            self.flow_model = FlowNet2Model('models/flow/flownet2/FlowNet2_checkpoint.pth.tar').cuda().eval()
        elif flow_model == 'LiteFlowNet':
            self.flow_model = LiteFlowNetModel('models/flow/liteflownet/network-default.pytorch').cuda().eval()
        elif flow_model == 'PWCNet':
            self.flow_model = PWCNetModel('models/flow/pwcnet/network-default.pytorch').cuda().eval()
        else:
            raise ValueError(flow_model)

        self.previous_mask = None
        self.previous_image = None
        self.previous_mask_result = None
        self.previous_image_result = None
        self.previous_available = False

    def initialize(self):
        self.previous_mask = None
        self.previous_image = None
        self.previous_mask_result = None
        self.previous_image_result = None
        self.previous_available = False

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:

        if self.previous_available:
            forward_flow = self.flow_model(self.previous_image, current_image)
            forward_flow_filled = normalize_flow(inpaint(forward_flow, self.previous_mask))

            backward_flow = self.flow_model(current_image, self.previous_image)
            backward_flow_filled = normalize_flow(inpaint(backward_flow, current_mask))

            grid = make_grid(current_image.size(), normalized=False).to(current_image.device)
            backward_grid = warp_tensor(grid, backward_flow_filled)
            forward_grid = warp_tensor(backward_grid, forward_flow_filled)
            flow_propagation_error = forward_grid - grid
            connected_pixels_mask = invert_mask(
                (warp_tensor(self.previous_mask_result, backward_flow_filled) > 0).float()) \
                                    * (torch.norm(flow_propagation_error, 2, dim=1) < self.eps).float().unsqueeze(1)

            current_image_result = current_image * invert_mask(current_mask) \
                                   + warp_tensor(self.previous_image_result,
                                                 backward_flow_filled) * connected_pixels_mask * current_mask
            current_mask_result = current_mask - connected_pixels_mask * current_mask

            # save_frame(current_image, 'debug/a.jpg', frame_type='image')
            # save_frame(current_mask, 'debug/b.png', frame_type='mask')
            # save_frame(self.previous_image, 'debug/c.jpg', frame_type='image')
            # save_frame(self.previous_mask, 'debug/d.png', frame_type='mask')
            # save_frame(forward_flow, 'debug/e.jpg', frame_type='flow')
            # save_frame(forward_flow_filled, 'debug/f.jpg', frame_type='flow')
            # save_frame(backward_flow, 'debug/g.jpg', frame_type='flow')
            # save_frame(backward_flow_filled, 'debug/h.jpg', frame_type='flow')
            # save_frame(connected_pixels_mask, 'debug/i.png', frame_type='mask')
            # save_frame(self.previous_image_result, 'debug/j.jpg', frame_type='image')
            # save_frame(self.previous_mask_result, 'debug/k.png', frame_type='mask')
            # save_frame(current_image_result, 'debug/l.jpg', frame_type='image')
            # save_frame(current_mask_result, 'debug/m.png', frame_type='mask')
        else:
            current_mask_result = current_mask
            current_image_result = current_image

        current_image_result = super().inpaint_online(current_image_result, current_mask_result)
        # save_frame(current_image_result, 'debug/n.jpg', frame_type='image')

        self.previous_mask = current_mask
        self.previous_image = current_image
        self.previous_mask_result = current_mask_result
        self.previous_image_result = current_image_result
        self.previous_available = True

        return current_image_result


class VideoTrackingAlgorithm:
    def __init__(self):
        super().__init__()
        self.tracking_model = SiamMaskModel(
            config_path='models/tracking/siammask/config_davis.json',
            weights_path='models/tracking/siammask/SiamMask_DAVIS.pth'
        ).cuda().eval()

    def initialize(self, image, roi):
        self.tracking_model.initialize(image, roi)

    def track(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []
        for image in images:
            result.append(self.track_online(image))
        return result

    def track_online(self, image: torch.Tensor) -> torch.Tensor:
        return self.tracking_model(image)
