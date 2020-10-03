import abc
from typing import List

from inpainting.flow import select_flow_model
from inpainting.inpainting import select_inpainting_model
from inpainting.inpainting.region_fill import inpaint
from inpainting.save import save_frame
from inpainting.tracking.siammask.model import SiamMaskModel
from inpainting.utils import make_grid, warp_tensor, normalize_flow, flow_tensor_to_image_tensor, dilate_mask, \
    mask_to_bbox

import torch


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
    def __init__(self, inpainting_model='DeepFillv1'):
        self.image_inpainting_model = select_inpainting_model(inpainting_model)

    def inpaint_online(self, current_image: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        return self.image_inpainting_model(current_image, current_mask)


class FlowGuidedVideoInpaintingAlgorithm(SingleFrameVideoInpaintingAlgorithm):
    def __init__(self, eps=1, flow_model='FlowNet2', inpainting_model='DeepFillv1'):
        super().__init__(inpainting_model)

        self.eps = eps

        self.flow_model = select_flow_model(flow_model)

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
            connected_pixels_mask = (1 - (warp_tensor(self.previous_mask_result, backward_flow_filled) > 0).float()) \
                                    * (torch.norm(flow_propagation_error, 2, dim=1) < self.eps).float().unsqueeze(1)

            current_image_result = current_image * (1 - current_mask) \
                                   + warp_tensor(self.previous_image_result,
                                                 backward_flow_filled) * connected_pixels_mask * current_mask
            current_mask_result = current_mask - connected_pixels_mask * current_mask

            save_frame(current_image, 'debug/i_a_current_image.png', frame_type='image')
            save_frame(current_mask, 'debug/i_b_current_mask.png', frame_type='mask')
            save_frame(self.previous_image, 'debug/i_c_previous_image.png', frame_type='image')
            save_frame(self.previous_mask, 'debug/i_d_previous_mask.png', frame_type='mask')
            save_frame(forward_flow, 'debug/i_e_forward_flow.png', frame_type='flowviz')
            save_frame(forward_flow_filled, 'debug/i_f_forward_flow_filled.png', frame_type='flowviz')
            save_frame(backward_flow, 'debug/i_g_backward_flow.png', frame_type='flowviz')
            save_frame(backward_flow_filled, 'debug/i_h_backward_flow_filled.png', frame_type='flowviz')
            save_frame(connected_pixels_mask, 'debug/i_i_connected_pixels_mask.png', frame_type='mask')
            save_frame(self.previous_image_result, 'debug/i_j_previous_image_result.png', frame_type='image')
            save_frame(self.previous_mask_result, 'debug/i_k_previous_mask_result.png', frame_type='mask')
            save_frame(current_image_result, 'debug/i_l_current_image_result.png', frame_type='image')
            save_frame(current_mask_result, 'debug/i_m_current_mask_result.png', frame_type='mask')
        else:
            current_mask_result = current_mask
            current_image_result = current_image

        self.previous_mask = current_mask
        self.previous_image = current_image
        self.previous_mask_result = current_mask_result
        self.previous_image_result = current_image_result
        self.previous_available = True

        current_image_result = super().inpaint_online(current_image_result, current_mask_result)
        save_frame(current_image_result, 'debug/i_n_current_image_result.png', frame_type='image')

        return current_image_result


class VideoTrackingAlgorithm:
    def __init__(self, dilation_size=5, dilation_iterations=3):
        super().__init__()
        self.tracking_model = SiamMaskModel(
            config_path='models/tracking/siammask/config_davis.json',
            weights_path='models/tracking/siammask/SiamMask_DAVIS.pth'
        ).cuda().eval()
        self.dilation_size = dilation_size
        self.dilation_iterations = dilation_iterations

    def initialize(self, image, roi):
        self.tracking_model.initialize(image, roi)

    def track(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []
        for image in images:
            result.append(self.track_online(image))
        return result

    def track_online(self, image: torch.Tensor) -> torch.Tensor:
        mask = self.tracking_model(image).unsqueeze(0)
        dilated_mask = dilate_mask(mask, self.dilation_size, self.dilation_iterations)

        save_frame(image, 'debug/s_a_image.png', frame_type='image')
        save_frame(image, 'debug/s_b_image_with_roi.png', frame_type='image', roi=mask_to_bbox(mask))
        save_frame(mask, 'debug/s_c_mask.png', frame_type='mask')
        save_frame(dilated_mask, 'debug/s_d_dilated_mask.png', frame_type='mask')
        return dilated_mask.squeeze(0)
