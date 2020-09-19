import pandas as pd
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

from .metrics import endpoint_error, object_coverage, background_coverage
from .utils import tensor_to_image


def evaluate_segmentation(target_masks, output_masks):
    results = []
    for frame_id, (target_mask, output_mask) in enumerate(zip(target_masks, output_masks)):
        results.append({
            'frame_id': frame_id,
            'object_coverage': float(object_coverage(target_mask, output_mask)),
            'background_coverage': float(background_coverage(target_mask, output_mask))
        })
    return pd.DataFrame(results)


def evaluate_flow(target_flows, output_flows):
    results = []
    for frame_id, (target_flow, output_flow) in enumerate(zip(target_flows, output_flows)):
        target_flow, output_flow = target_flow.numpy(), \
                                   output_flow.numpy()

        results.append({
            'frame_id': frame_id,
            'endpoint_error': endpoint_error(target_flow, output_flow)
        })
    return pd.DataFrame(results)


def evaluate_inpainting(target_images, output_images):
    results = []
    for frame_id, (target_image, output_image) in enumerate(zip(target_images, output_images)):
        target_image, output_image = tensor_to_image(target_image, rgb2bgr=False), \
                                     tensor_to_image(output_image, rgb2bgr=False)
        results.append({
            'frame_id': frame_id,
            'mean_squared_error': float(mean_squared_error(target_image, output_image)),
            'peak_signal_noise_ratio': float(peak_signal_noise_ratio(target_image, output_image)),
            'structural_similarity': float(structural_similarity(target_image, output_image, multichannel=True))
        })
    return pd.DataFrame(results)
