import pandas as pd
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

from .metrics import db_eval_iou, db_eval_boundary
from .utils import tensor_to_cv_image


def evaluate_segmentation(target_masks, output_masks):
    results = []
    for frame_id, (target_mask, output_mask) in enumerate(zip(target_masks, output_masks)):
        target_mask, output_mask = target_mask.numpy(), output_mask.numpy()
        results.append({
            'frame_id': frame_id,
            'region_similarity': float(db_eval_iou(target_mask, output_mask)),
            'contour_accuracy': float(db_eval_boundary(target_mask, output_mask))
        })
    return pd.DataFrame(results)


def evaluate_inpainting(target_images, output_images):
    results = []
    for frame_id, (target_image, output_image) in enumerate(zip(target_images, output_images)):
        target_image, output_image = tensor_to_cv_image(target_image), tensor_to_cv_image(output_image)
        results.append({
            'frame_id': frame_id,
            'mean_squared_error': float(mean_squared_error(target_image, output_image)),
            'peak_signal_noise_ratio': float(peak_signal_noise_ratio(target_image, output_image)),
            'structural_similarity': float(structural_similarity(target_image, output_image, multichannel=True))
        })
    return pd.DataFrame(results)
