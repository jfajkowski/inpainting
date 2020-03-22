import abc
from argparse import Namespace
from typing import List

import numpy as np
import torch
from inpainting.external.deepflowguidedvideoinpainting.models import FlowNet2Model, DeepFillV1Model
from inpainting.external.deepflowguidedvideoinpainting.utils import fill_flow, warp_tensor, make_grid
from inpainting.external.deepvideoinpainting.model import generate_model
from inpainting.external.deepvideoinpainting.utils import repackage_hidden
from inpainting.external.siammask.config_helper import load_config
from inpainting.external.siammask.custom import Custom
from inpainting.external.siammask.load_helper import load_pretrain
from inpainting.external.siammask.test import siamese_init, siamese_track
from inpainting.utils import mask_tensor, normalize, denormalize, invert_mask
from inpainting.visualize import debug


class VideoInpaintingAlgorithm(abc.ABC):
    def inpaint(self, frames: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        frames_result, masks_result = [], []
        for frame, mask in zip(frames, masks):
            frame_result = self.inpaint_online(frame, mask)
            frames_result.append(frame_result)
        return frames_result

    @abc.abstractmethod
    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        pass


class DeepFlowGuidedVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self, eps=20):
        self.flow_model = FlowNet2Model('models/external/flownet2/FlowNet2_checkpoint.pth.tar').cuda().eval()
        self.inpainting_model = DeepFillV1Model('models/external/deepfillv1/imagenet_deepfill.pth').cuda().eval()
        self.eps = eps
        self.previous_available = False
        self.previous_mask = None
        self.previous_frame = None
        self.previous_mask_result = None
        self.previous_frame_result = None

    def reset(self):
        self.previous_available = False
        self.previous_mask = None
        self.previous_frame = None
        self.previous_mask_result = None
        self.previous_frame_result = None

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        if not self.previous_available:
            self.previous_mask = current_mask
            self.previous_frame = current_frame
            self.previous_mask_result = current_mask
            self.previous_frame_result = mask_tensor(current_frame, current_mask)
            self.previous_available = True

        forward_flow = self.flow_model(self.previous_frame, current_frame)
        forward_flow_masked = mask_tensor(forward_flow, self.previous_mask)
        forward_flow_filled = fill_flow(forward_flow_masked, self.previous_mask)

        backward_flow = self.flow_model(current_frame, self.previous_frame)
        backward_flow_masked = mask_tensor(backward_flow, current_mask)
        backward_flow_filled = fill_flow(backward_flow_masked, current_mask)

        grid = make_grid(forward_flow.size(), normalized=False)
        backward_grid = warp_tensor(grid, backward_flow, mode='nearest')
        forward_grid = warp_tensor(backward_grid, forward_flow, mode='nearest')
        flow_propagation_error = forward_grid - grid
        connected_pixels_mask = (torch.norm(flow_propagation_error, 2, dim=1) < self.eps).float().unsqueeze(1)

        current_mask_warped = warp_tensor(connected_pixels_mask * self.previous_mask_result, backward_flow_filled,
                                          mode='nearest')
        current_frame_warped = warp_tensor(self.previous_frame_result, backward_flow_filled, mode='nearest')

        current_mask_result = current_mask + current_mask_warped * (invert_mask(current_mask))
        current_frame_result = current_frame * current_mask + current_frame_warped * current_mask_warped * (
            invert_mask(current_mask))

        debug(self.previous_frame, '0_previous_frame')
        debug(self.previous_mask, '0_previous_mask')
        debug(self.previous_frame_result, '0_previous_frame_result')
        debug(self.previous_mask_result, '0_previous_mask_result')
        debug(current_frame, '0_current_frame')
        debug(current_mask, '0_current_mask')
        debug(forward_flow, '1_forward_flow')
        debug(forward_flow_masked, '1_forward_flow_masked')
        debug(forward_flow_filled, '1_forward_flow_filled')
        debug(backward_flow, '1_backward_flow')
        debug(backward_flow_masked, '1_backward_flow_masked')
        debug(backward_flow_filled, '1_backward_flow_filled')
        debug(flow_propagation_error, '2_flow_propagation_error')
        debug(connected_pixels_mask, '2_connected_pixels_mask')
        debug(current_frame_warped, '3_current_frame_warped')
        debug(current_mask_warped, '3_current_mask_warped')
        debug(current_frame_result, '4_current_frame_result')
        debug(current_mask_result, '4_current_mask_result')

        self.previous_mask = current_mask
        self.previous_frame = current_frame
        self.previous_mask_result = current_mask_result
        self.previous_frame_result = mask_tensor(current_frame_result, current_mask_result)

        current_frame, current_mask = current_frame_result, current_mask_result

        current_frame = normalize(current_frame)
        current_frame_masked = mask_tensor(current_frame, current_mask)
        current_frame_filled = self.inpainting_model(current_frame_masked, current_mask)
        current_frame_result = denormalize(
            current_mask * current_frame + (invert_mask(current_mask)) * current_frame_filled)

        debug(current_frame, '0_current_frame', denormalize)
        debug(current_mask, '0_current_mask')
        debug(current_frame_masked, '1_current_frame_masked', denormalize)
        debug(current_frame_filled, '2_current_frame_filled', denormalize)
        debug(current_frame_result, '3_current_frame_result')
        return current_frame_result


class SiamMaskVideoSegmentationAlgorithm:
    def __init__(self, init_image, init_roi, device='cuda'):
        super().__init__()
        args = Namespace(
            config='models/external/siammask/config_davis.json'
        )
        cfg = load_config(args)
        self.model = Custom(anchors=cfg['anchors']).cuda().eval()
        load_pretrain(self.model, 'models/external/siammask/SiamMask_DAVIS.pth')

        x, y, w, h = init_roi
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(init_image, target_pos, target_sz, self.model, cfg['hp'], device=device)
        self.device = device

    def find_mask(self, image):
        self.state = siamese_track(self.state, image, mask_enable=True, refine_enable=True, device=self.device)
        mask = self.state['mask'] > self.state['p'].seg_thr
        return mask


class DeepVideoInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self):
        args = Namespace(
            model='vinet_final',
            no_cuda=False,
            pretrain_path='models/external/vinet/save_agg_rec_512.pth',
            double_size=True,
            crop_size=512,
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
        self.masked_frames = []
        self.t = 0
        self.prev_mask = None
        self.prev_output = None

    def reset(self):
        self.masks = []
        self.masked_frames = []

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        current_frame = normalize(current_frame)
        current_frame_masked = mask_tensor(current_frame, current_mask)
        current_mask = invert_mask(current_mask)

        if not self.previous_available:
            self.masks = 5 * [current_mask]
            self.masked_frames = 5 * [current_frame_masked]
            self.ones = torch.ones(current_mask.size()).cuda()
            self.prev_mask = current_mask
            self.prev_output = current_frame_masked
            self.previous_available = True
        elif self.t % self.t_stride == 0:
            self.masks.pop(0)
            self.masks.append(current_mask)
            self.masked_frames.pop(0)
            self.masked_frames.append(current_frame_masked)

        masks = torch.stack(self.masks, dim=2)
        masked_inputs = torch.stack(self.masked_frames, dim=2)
        prev_feed = torch.cat([self.prev_output, self.ones, self.ones * self.prev_mask], dim=1)

        result, _, self.lstm_state, _, _ = self.model(masked_inputs, masks, self.lstm_state, prev_feed, self.t)
        result = result.squeeze(2)
        self.lstm_state = repackage_hidden(self.lstm_state)
        self.t += 1

        self.prev_mask = current_mask * 0.5
        self.prev_output = result

        return denormalize(result)
