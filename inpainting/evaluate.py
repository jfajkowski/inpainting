import numpy as np

from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def mean_absolute_error(image0, image1):
    """
    Compute the mean-absolute error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-absolute error (MAE) metric.
    """
    from skimage._shared.utils import check_shape_equality
    from skimage.metrics.simple_metrics import _as_floats
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean(np.abs(image0 - image1), dtype=np.float64)


def evaluate_video(video_true, video_test):
    return [evaluate_image(image_true, image_test) for image_true, image_test in zip(video_true, video_test)]


def evaluate_image(image_true, image_test):
    image_true = np.array(image_true)
    image_test = np.array(image_test)
    return {
        'mae': mean_absolute_error(image_true, image_test),
        'mse': mean_squared_error(image_true, image_test),
        'psnr': peak_signal_noise_ratio(image_true, image_test, data_range=255),
        'ssim': structural_similarity(image_true, image_test, multichannel=True)
    }
