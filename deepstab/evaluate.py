import numpy as np

from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_absolute_error


def evaluate_video(video_true, video_test):
    return [evaluate_image(image_true, image_test) for image_true, image_test in zip(video_true, video_test)]


def evaluate_image(image_true, image_test):
    image_true = np.array(image_true)
    image_test = np.array(image_test)
    return {
        'mae': mean_absolute_error(image_true, image_test),
        'mse': mean_squared_error(image_true, image_test),
        'psnr': peak_signal_noise_ratio(image_true, image_test),
        'ssim': structural_similarity(image_true, image_test, multichannel=True)
    }
