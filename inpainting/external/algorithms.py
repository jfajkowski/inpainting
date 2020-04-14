import abc
from argparse import Namespace
from typing import List

import numpy as np
import cv2 as cv
import torch
from inpainting.external.deepflowguidedvideoinpainting.models import FlowNet2Model, DeepFillV1Model
from inpainting.external.deepflowguidedvideoinpainting.utils import fill_flow, warp_tensor, make_grid
from inpainting.external.deepvideoinpainting.model import generate_model
from inpainting.external.deepvideoinpainting.utils import repackage_hidden
from inpainting.external.freeformvideoinpainting.model import FreeFormVideoInpaintingModel
from inpainting.external.siammask.config_helper import load_config
from inpainting.external.siammask.custom import Custom
from inpainting.external.siammask.load_helper import load_pretrain
from inpainting.external.siammask.test import siamese_init, siamese_track
from inpainting.utils import mask_tensor, normalize, denormalize, invert_mask, tensor_to_cv_image
from inpainting.visualize import debug


class VideoInpaintingAlgorithm(abc.ABC):
    def inpaint(self, images: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        images_result, masks_result = [], []
        for image, mask in zip(images, masks):
            image_result = self.inpaint_online(image, mask)
            images_result.append(image_result)
        return images_result

    @abc.abstractmethod
    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        pass


class DeepFillVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self):
        self.model = DeepFillV1Model('models/external/deepfillv1/imagenet_deepfill.pth').cuda().eval()

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        current_mask = invert_mask(current_mask)
        current_image = normalize(current_image)
        current_image_masked = mask_tensor(current_image, current_mask)
        current_image_filled = self.model(current_image_masked, current_mask)
        current_image_result = denormalize(
            current_mask * current_image + (invert_mask(current_mask)) * current_image_filled)
        return current_image_result


class DeepFlowGuidedVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self, eps=20):
        self.flow_model = FlowNet2Model('models/external/flownet2/FlowNet2_checkpoint.pth.tar').cuda().eval()
        self.inpainting_model = DeepFillV1Model('models/external/deepfillv1/imagenet_deepfill.pth').cuda().eval()
        self.eps = eps
        self.previous_available = False
        self.previous_mask = None
        self.previous_image = None
        self.previous_mask_result = None
        self.previous_image_result = None

    def initialize(self):
        self.previous_available = False
        self.previous_mask = None
        self.previous_image = None
        self.previous_mask_result = None
        self.previous_image_result = None

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        current_mask = invert_mask(current_mask)

        if not self.previous_available:
            self.previous_mask = current_mask
            self.previous_image = current_image
            self.previous_mask_result = current_mask
            self.previous_image_result = mask_tensor(current_image, current_mask)
            self.previous_available = True

        forward_flow = self.flow_model(self.previous_image, current_image)
        forward_flow_masked = mask_tensor(forward_flow, self.previous_mask)
        forward_flow_filled = fill_flow(forward_flow_masked, self.previous_mask)

        backward_flow = self.flow_model(current_image, self.previous_image)
        backward_flow_masked = mask_tensor(backward_flow, current_mask)
        backward_flow_filled = fill_flow(backward_flow_masked, current_mask)

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
        debug(forward_flow_masked, '1_forward_flow_masked')
        debug(forward_flow_filled, '1_forward_flow_filled')
        debug(backward_flow, '1_backward_flow')
        debug(backward_flow_masked, '1_backward_flow_masked')
        debug(backward_flow_filled, '1_backward_flow_filled')
        debug(flow_propagation_error, '2_flow_propagation_error')
        debug(connected_pixels_mask, '2_connected_pixels_mask')
        debug(current_image_warped, '3_current_image_warped')
        debug(current_mask_warped, '3_current_mask_warped')
        debug(current_image_result, '4_current_image_result')
        debug(current_mask_result, '4_current_mask_result')

        self.previous_mask = current_mask
        self.previous_image = current_image
        self.previous_mask_result = current_mask_result
        self.previous_image_result = mask_tensor(current_image_result, current_mask_result)

        current_image, current_mask = current_image_result, current_mask_result

        current_image = normalize(current_image)
        current_image_masked = mask_tensor(current_image, current_mask)
        current_image_filled = self.inpainting_model(current_image_masked, current_mask)
        current_image_result = denormalize(
            current_mask * current_image + (invert_mask(current_mask)) * current_image_filled)

        debug(current_image, '0_current_image', denormalize)
        debug(current_mask, '0_current_mask')
        debug(current_image_masked, '1_current_image_masked', denormalize)
        debug(current_image_filled, '2_current_image_filled', denormalize)
        debug(current_image_result, '3_current_image_result')
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


class DeepVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self):
        args = Namespace(
            model='vinet_final',
            no_cuda=False,
            pretrain_path='models/external/vinet/save_agg_rec.pth',
            double_size=False,
            crop_size=256,
            search_range=4,
            batch_norm=False,
            prev_warp=True,
            loss_on_raw=False
        )
        self.model, _ = generate_model(args)
        self.model = self.model.eval()
        self.t_stride = 3
        self.previous_available = False
        self.ones = None
        self.lstm_state = None
        self.masks = []
        self.masked_images = []
        self.t = 0
        self.prev_mask = None
        self.prev_output = None

    def initialize(self):
        self.masks = []
        self.masked_images = []
        self.previous_available = False

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        current_image = normalize(current_image)
        current_image_masked = mask_tensor(current_image, invert_mask(current_mask))

        if not self.previous_available:
            self.masks = (4 * self.t_stride + 1) * [current_mask]
            self.masked_images = (4 * self.t_stride + 1) * [current_image_masked]
            self.ones = torch.ones(current_mask.size()).cuda()
            self.prev_mask = current_mask
            self.prev_output = current_image_masked
            self.previous_available = True
        elif self.t:
            self.masks.pop(0)
            self.masks = self.masks[2 * self.t_stride:] + (2 * self.t_stride + 1) * [current_mask]
            self.masked_images.pop(0)
            self.masked_images = self.masked_images[2 * self.t_stride:] + (2 * self.t_stride + 1) * [current_image_masked]

        masks = torch.stack(self.masks[::self.t_stride], dim=2)
        masked_inputs = torch.stack(self.masked_images[::self.t_stride], dim=2)
        prev_feed = torch.cat([self.prev_output, self.ones, self.ones * self.prev_mask], dim=1)

        result, _, self.lstm_state, _, _ = self.model(masked_inputs, masks, self.lstm_state, prev_feed, self.t)
        result = result.squeeze(2)
        self.lstm_state = repackage_hidden(self.lstm_state)
        self.t += 1

        self.prev_mask = current_mask * 0.5
        self.prev_output = result

        debug(current_image, '0_current_image', denormalize)
        debug(current_mask, '0_current_mask')
        debug(result, '1_result', denormalize)

        return denormalize(result)


class FreeFormVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self, n=15):
        self.model = FreeFormVideoInpaintingModel('models/external/lgtsm/model.pth').eval()
        self.model.model.generator.cuda()
        self.n = n
        self.images = []
        self.masks = []
        self.previous_available = False

    def initialize(self):
        self.images = []
        self.masks = []
        self.previous_available = False

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        current_image = normalize(current_image, mode='minmax')
        current_mask = invert_mask(current_mask)

        if not self.previous_available:
            self.images = self.n * [current_image]
            self.masks = self.n * [current_mask]
            self.previous_available = True
        else:
            self.images.pop(0)
            self.images.append(current_image)
            self.masks.pop(0)
            self.masks.append(current_mask)

        images = torch.stack(self.images, dim=1)
        masks = torch.stack(self.masks, dim=1)

        result = self.model(images, masks)[:, -1]

        return denormalize(result, mode='minmax')
