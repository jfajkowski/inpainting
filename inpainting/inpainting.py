from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from inpainting.flow import estimate_flow, fill_flow, warp_tensor
from inpainting.utils import mask_tensor, normalize, denormalize
from inpainting.visualize import debug


class VideoInpaintingAlgorithm(ABC):

    def reset(self):
        pass

    def inpaint(self, frames: List[torch.Tensor], masks: List[torch.Tensor]) -> Tuple[
        List[torch.Tensor], List[torch.Tensor]]:
        frames_result, masks_result = [], []
        for frame, mask in zip(frames, masks):
            frame_result, mask_result = self.inpaint_online(frame, mask)
            frames_result.append(frame_result)
            masks_result.append(mask_result)
        return frames_result, masks_result

    @abstractmethod
    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        pass


class FlowInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self, flow_model, eps=1):
        self.flow_model = flow_model
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

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        if not self.previous_available:
            self.previous_available = True
            self.previous_mask = current_mask
            self.previous_frame = current_frame
            self.previous_mask_result = current_mask
            self.previous_frame_result = mask_tensor(current_frame, current_mask)

        forward_flow = estimate_flow(self.flow_model, self.previous_frame, current_frame)
        forward_flow_masked = mask_tensor(forward_flow, self.previous_mask)
        forward_flow_filled = fill_flow(forward_flow_masked, self.previous_mask)

        backward_flow = estimate_flow(self.flow_model, current_frame, self.previous_frame)
        backward_flow_masked = mask_tensor(backward_flow, current_mask)
        backward_flow_filled = fill_flow(backward_flow_masked, current_mask)

        flow_propagation_error = warp_tensor(
            (warp_tensor(forward_flow_filled, backward_flow_filled, mode='nearest') + backward_flow_filled) / 2,
            forward_flow_filled, mode='nearest')
        connected_pixels_mask = (torch.norm(flow_propagation_error, 2, dim=1) < self.eps).float().unsqueeze(1)

        current_mask_warped = warp_tensor(connected_pixels_mask * self.previous_mask_result, backward_flow_filled,
                                          mode='nearest')
        current_frame_warped = warp_tensor(self.previous_frame_result, backward_flow_filled, mode='nearest')

        current_mask_result = current_mask + current_mask_warped * (1 - current_mask)
        current_frame_result = current_frame * current_mask + current_frame_warped * current_mask_warped * (
                    1 - current_mask)

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

        return current_frame_result, current_mask_result


class FillInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self, fill_model: torch.nn.Module):
        self.inpainting_model = fill_model

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        current_frame = normalize(current_frame)
        current_frame_masked = mask_tensor(current_frame, current_mask)
        current_frame_filled = self.inpainting_model(current_frame_masked, current_mask)
        current_frame_result = denormalize(current_mask * current_frame + (1 - current_mask) * current_frame_filled)
        current_mask_result = torch.ones_like(current_mask)

        debug(current_frame, '0_current_frame', denormalize)
        debug(current_mask, '0_current_mask')
        debug(current_frame_masked, '1_current_frame_masked', denormalize)
        debug(current_frame_filled, '2_current_frame_filled', denormalize)
        debug(current_frame_result, '3_current_frame_result')
        return current_frame_result, current_mask_result


class FlowAndFillInpaintingAlgorithm(VideoInpaintingAlgorithm):

    def __init__(self, flow_model, fill_model, eps=1):
        self.flow_inpainting_algorithm = FlowInpaintingAlgorithm(flow_model, eps)
        self.fill_inpainting_algorithm = FillInpaintingAlgorithm(fill_model)

    def reset(self):
        self.flow_inpainting_algorithm.reset()
        self.fill_inpainting_algorithm.reset()

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        current_frame, current_mask = self.flow_inpainting_algorithm.inpaint_online(current_frame, current_mask)
        current_frame, current_mask = self.fill_inpainting_algorithm.inpaint_online(current_frame, current_mask)
        return current_frame, current_mask
