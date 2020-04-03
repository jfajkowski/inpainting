from os import makedirs

import pandas as pd
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

from .davis.metrics import db_eval_iou, db_eval_boundary
from .utils import tensor_to_cv_image


def evaluate_tracking(target_masks, output_masks):
    results = []
    for t, (target_mask, output_mask) in enumerate(zip(target_masks, output_masks)):
        target_mask, output_mask = target_mask.numpy(), output_mask.numpy()
        results.append({
            't': t,
            'region_similarity': float(db_eval_iou(target_mask, output_mask)),
            'contour_accuracy': float(db_eval_boundary(target_mask, output_mask))
        })
    return pd.DataFrame(results)


def evaluate_inpainting(target_images, output_images):
    results = []
    for t, (target_image, output_image) in enumerate(zip(target_images, output_images)):
        target_image, output_image = tensor_to_cv_image(target_image), tensor_to_cv_image(output_image)
        results.append({
            't': t,
            'mean_squared_error': float(mean_squared_error(target_image, output_image)),
            'peak_signal_noise_ratio': float(peak_signal_noise_ratio(target_image, output_image)),
            'structural_similarity': float(structural_similarity(target_image, output_image, multichannel=True))
        })
    return pd.DataFrame(results)


def save_stats(df, dir):
    makedirs(dir, exist_ok=True)
    print(df.mean(), file=open(f'{dir}/mean.txt', mode='w'))
    print(df.median(), file=open(f'{dir}/median.txt', mode='w'))


def save_results(df, dir):
    makedirs(dir, exist_ok=True)
    df.to_csv(f'{dir}/results.csv', index=False)
