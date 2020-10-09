import abc

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


class Transform(abc.ABC):

    def __call__(self, *args):
        params = self.get_params()
        res = []
        for i in args:
            if isinstance(i, list):
                res.append(self.transform_sequence(i, params))
            elif isinstance(i, Image.Image) or isinstance(i, np.ndarray) or isinstance(i, torch.Tensor):
                res.append(self.transform_sample(i, params))
        return tuple(res)

    def get_params(self):
        return tuple()

    def transform_sequence(self, sequence, params):
        for i in range(len(sequence)):
            sequence[i] = self.transform_sample(sequence[i], params)
        return sequence

    @abc.abstractmethod
    def transform_sample(self, sample, params):
        pass


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ToTensor(Transform):

    def transform_sample(self, sample, params):
        return F.to_tensor(sample)


class CenterCrop(Transform):

    def __init__(self, ratio, frame_type):
        self.ratio = ratio
        self.frame_type = frame_type

    def transform_sample(self, sample, params):
        image_height, image_width = sample.shape[:2]

        if image_height > image_width:
            crop_height = image_height
            crop_width = round(crop_height * self.ratio)
        else:
            crop_width = image_width
            crop_height = round(crop_width / self.ratio)

        crop_left = int(round((image_width - crop_width) / 2.))
        crop_top = int(round((image_height - crop_height) / 2.))
        return sample[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width, :]


class Resize(Transform):

    def __init__(self, size, frame_type):
        self.size = size
        self.frame_type = frame_type

    def transform_sample(self, sample, params):
        if self.frame_type == 'image':
            interpolation = cv.INTER_LINEAR
        else:
            interpolation = cv.INTER_NEAREST
        return cv.resize(sample, tuple(self.size[::-1]), interpolation=interpolation)


class Lambda(Transform):

    def __init__(self, lambd):
        self.lambd = lambd

    def transform_sample(self, sample, params):
        return self.lambd(sample)
