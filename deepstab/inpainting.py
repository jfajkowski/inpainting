from abc import ABC, abstractmethod
from typing import List

import torch

from deepstab.flow import estimate_flow, fill_flow, warp_tensor
from deepstab.utils import mask_tensor, normalize, debug


class VideoInpaintingAlgorithm(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def inpaint(self, frames: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        pass


class ImageInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self, inpainting_model: torch.nn.Module):
        self.inpainting_model = inpainting_model

    def reset(self):
        pass

    def inpaint(self, frames: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []
        for frame, mask in zip(frames, masks):
            result.append(self.inpaint_online(frame, mask))
        return result

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        current_frame = normalize(current_frame)
        current_frame_masked = mask_tensor(current_frame, current_mask)
        current_frame_filled = self.inpainting_model(current_frame_masked, current_mask)
        current_frame_result = current_mask * current_frame + (1 - current_mask) * current_frame_filled

        debug(current_frame, 'current_frame')
        debug(current_mask, 'current_mask')
        debug(current_frame_masked, 'current_frame_masked')
        debug(current_frame_filled, 'current_frame_filled')
        debug(current_frame_result, 'current_frame_result')
        return current_frame_result


class FlowInpaintingAlgorithm(VideoInpaintingAlgorithm):
    def __init__(self, flow_model, inpainting_model: torch.nn.Module, eps=1):
        self.flow_model = flow_model
        self.inpainting_model = inpainting_model
        self.eps = eps
        self.previous_available = False
        self.previous_frame = None
        self.previous_mask = None
        self.previous_output_frame = None
        self.previous_output_mask = None

    def reset(self):
        self.previous_available = False
        self.previous_frame = None
        self.previous_mask = None
        self.previous_output_frame = None
        self.previous_output_mask = None

    def inpaint(self, frames: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []
        for frame, mask in zip(frames, masks):
            result.append(self.inpaint_online(frame, mask))
        return result

    def inpaint_online(self, current_frame: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        if self.previous_available:
            forward_flow = estimate_flow(self.flow_model, self.previous_frame, current_frame)
            masked_forward_flow = mask_tensor(forward_flow, current_mask)
            filled_forward_flow = fill_flow(masked_forward_flow, current_mask)

            backward_flow = estimate_flow(self.flow_model, current_frame, self.previous_frame)
            masked_backward_flow = mask_tensor(backward_flow, self.previous_mask)
            filled_backward_flow = fill_flow(masked_backward_flow, self.previous_mask)

            masked_previous_output_frame = mask_tensor(self.previous_output_frame, self.previous_output_mask)

            propagation_error = warp_tensor(
                (warp_tensor(filled_forward_flow, filled_backward_flow, mode='nearest') + filled_backward_flow) / 2,
                filled_forward_flow, mode='nearest')
            connected_pixels_mask = (torch.norm(propagation_error, 2, dim=1) < self.eps).float()

            warped_mask = warp_tensor(connected_pixels_mask * self.previous_output_mask, filled_backward_flow,
                                      mode='nearest')
            warped_frame = warp_tensor(masked_previous_output_frame, filled_backward_flow, mode='bilinear')

            output_mask = current_mask + warped_mask * (1 - current_mask)
            output_frame = current_frame * current_mask + warped_frame * warped_mask * (1 - current_mask) + (
                    1 - output_mask)

        else:
            output_frame = mask_tensor(current_frame, current_mask)
            output_mask = current_mask

        self.previous_available = True
        self.previous_frame = current_frame
        self.previous_mask = current_mask
        self.previous_output_frame = output_frame
        self.previous_output_mask = output_mask

        output_frame = normalize(output_frame.clone())
        frame_masked = mask_tensor(output_frame, output_mask)
        frame_filled = self.inpainting_model(frame_masked, output_mask)
        return frame_filled
        # return output_frame
