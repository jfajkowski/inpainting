import abc
from argparse import Namespace
from typing import List
import torch

from inpainting.models.inpainting.deepfillv1.DeepFill import DeepFillV1Model
from inpainting.models.flow.liteflownet.model import LiteFlowNetModel
from inpainting.models.inpainting.kernel_inpainting import inpaint
from inpainting.models.utils import make_grid, warp_tensor
from inpainting.utils import dilate_tensor

import numpy as np
import cv2 as cv
import torch
from inpainting.models.tracking.siammask.config_helper import load_config
from inpainting.models.tracking.siammask.custom import Custom
from inpainting.models.tracking.siammask.load_helper import load_pretrain
from inpainting.models.tracking.siammask.test import siamese_init, siamese_track
from inpainting.utils import mask_tensor, normalize, denormalize, invert_mask, tensor_to_cv_image
from inpainting.visualize import debug


class VideoInpaintingAlgorithm(abc.ABC):

    def initialize(self):
        pass

    def inpaint(self, images: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        images_result, masks_result = [], []
        for image, mask in zip(images, masks):
            image_result = self.inpaint_online(image, mask)
            images_result.append(image_result)
        return images_result

    @abc.abstractmethod
    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        pass


class SingleFrameVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self):
        self.model = DeepFillV1Model('models/external/deepfillv1/imagenet_deepfill.pth').cuda().eval()

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        return inpaint(current_image, current_mask)


class StableVideoInpaintingAlgorithm(SingleFrameVideoInpaintingAlgorithm):
    def __init__(self):
        super().__init__()
        self.previous_image = None
        self.previous_mask = None
        self.previous_image_result = None
        self.previous_available = False

    def initialize(self):
        self.previous_image = None
        self.previous_mask = None
        self.previous_image_result = None
        self.previous_available = False

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        current_image = current_image * (1 - current_mask)
        if self.previous_available:
            copy_mask = current_mask * self.previous_mask
            current_image = current_image + self.previous_image_result * copy_mask
            current_mask = current_mask - copy_mask

        current_image_result = super().inpaint_online(current_image, current_mask)

        self.previous_image = current_image
        self.previous_mask = current_mask
        self.previous_image_result = current_image_result
        self.previous_available = True

        return current_image_result


class FlowGuidedVideoInpaintingAlgorithm(SingleFrameVideoInpaintingAlgorithm):
    def __init__(self, eps=5):
        super().__init__()
        self.flow_model = LiteFlowNetModel().cuda().eval()
        self.eps = eps
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
        current_mask = invert_mask(current_mask)

        if self.previous_available:
            forward_flow = self.flow_model(self.previous_image, current_image)
            forward_flow_filled = inpaint(forward_flow, invert_mask(self.previous_mask))

            backward_flow = self.flow_model(current_image, self.previous_image)
            backward_flow_filled = inpaint(backward_flow, invert_mask(current_mask))

            grid = make_grid(forward_flow.size(), normalized=False)
            backward_grid = warp_tensor(grid, backward_flow, mode='nearest')
            forward_grid = warp_tensor(backward_grid, forward_flow, mode='nearest')
            flow_propagation_error = forward_grid - grid
            connected_pixels_mask = (torch.norm(flow_propagation_error, 2, dim=1) < self.eps).float().unsqueeze(1)

            current_mask_warped = warp_tensor(connected_pixels_mask * self.previous_mask_result, backward_flow_filled,
                                              mode='nearest')
            current_image_warped = warp_tensor(self.previous_image_result, backward_flow_filled, mode='nearest')

            current_mask_result = current_mask + current_mask_warped * (invert_mask(current_mask))
            current_image_result = current_image * current_mask + current_image_warped * current_mask_warped * (
                invert_mask(current_mask))

            debug(self.previous_image, '0_previous_image')
            debug(self.previous_mask, '0_previous_mask')
            debug(self.previous_image_result, '0_previous_image_result')
            debug(self.previous_mask_result, '0_previous_mask_result')
            debug(current_image, '0_current_image')
            debug(current_mask, '0_current_mask')
            debug(forward_flow, '1_forward_flow')
            debug(forward_flow_filled, '1_forward_flow_filled')
            debug(backward_flow, '1_backward_flow')
            debug(backward_flow_filled, '1_backward_flow_filled')
            debug(flow_propagation_error, '2_flow_propagation_error')
            debug(connected_pixels_mask, '2_connected_pixels_mask')
            debug(current_image_warped, '3_current_image_warped')
            debug(current_mask_warped, '3_current_mask_warped')
            debug(current_image_result, '4_current_image_result')
            debug(current_mask_result, '4_current_mask_result')
        else:
            current_mask_result = current_mask
            current_image_result = mask_tensor(current_image, current_mask)

        current_image_result = super().inpaint_online(current_image_result, invert_mask(current_mask_result))

        self.previous_mask = current_mask
        self.previous_image = current_image
        self.previous_mask_result = current_mask_result
        self.previous_image_result = current_image_result
        self.previous_available = True

        return current_image_result


class SiamMaskVideoTrackingAlgorithm:
    def __init__(self, mask_type='segmentation'):
        super().__init__()
        args = Namespace(
            config='models/external/siammask/config_davis.json'
        )
        self.cfg = load_config(args)
        self.model = Custom(anchors=self.cfg['anchors']).cuda().eval()
        load_pretrain(self.model, 'models/external/siammask/SiamMask_DAVIS.pth')
        self.state = None
        self.device = 'cuda'
        self.mask_type = mask_type

    def initialize(self, image, roi):
        image = tensor_to_cv_image(image)
        (x1, y1), (x2, y2) = roi
        w, h = x2 - x1, y2 - y1
        target_pos = np.array([x1 + w / 2, y1 + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(image, target_pos, target_sz, self.model, self.cfg['hp'], self.device)

    def find_mask(self, image):
        image = tensor_to_cv_image(image)
        self.state = siamese_track(self.state, image, mask_enable=True, refine_enable=True, device=self.device)
        mask = None
        if self.mask_type == 'segmentation':
            mask = self.state['mask'] > self.state['p'].seg_thr
        elif self.mask_type == 'box':
            mask = np.zeros(image.shape[:-1])
            rotated_bounding_box = self.state['ploygon'].astype(int)
            mask = cv.fillConvexPoly(mask, rotated_bounding_box, 1)
        else:
            raise ValueError(self.mask_type)
        return torch.tensor(mask).unsqueeze(0).float()


class NoopVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def initialize(self):
        pass

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        return current_frame
