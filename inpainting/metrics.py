import numpy as np

eps = 1e-6


def object_coverage(target_mask, output_mask):
    target_mask, output_mask = target_mask.bool(), output_mask.bool()
    true_positive = (target_mask & output_mask).float().sum((1, 2))
    true = target_mask.float().sum((1, 2))
    return (true_positive + eps) / (true + eps)  # avoid 0/0


def background_coverage(target_mask, output_mask):
    target_mask, output_mask = ~target_mask.bool(), ~output_mask.bool()
    false_negative = (target_mask & output_mask).float().sum((1, 2))
    false = target_mask.float().sum((1, 2))
    return (false_negative + eps) / (false + eps)  # avoid 0/0


def endpoint_error(target_flow, output_flow):
    return np.linalg.norm(target_flow - output_flow, ord=2, axis=0).mean()
