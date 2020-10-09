import abc

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from inpainting.utils import normalize_flow, denormalize_flow


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
        image_height, image_width = sample.size[::-1] if self.frame_type == 'annotation' else sample.shape[:2]

        crop_height = image_height
        crop_width = round(crop_height * self.ratio)
        if crop_width > image_width:
            crop_width = image_width
            crop_height = round(crop_width / self.ratio)

        crop_left = int(round((image_width - crop_width) / 2.))
        crop_top = int(round((image_height - crop_height) / 2.))
        if self.frame_type == 'annotation':
            return F.crop(sample, crop_top, crop_left, crop_height, crop_width)
        elif self.frame_type == 'mask':
            return sample[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]
        elif self.frame_type == 'image' or self.frame_type == 'flow':
            return sample[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width, :]
        else:
            raise ValueError(self.frame_type)


class Resize(Transform):

    def __init__(self, size, frame_type):
        self.size = size
        self.frame_type = frame_type

    def transform_sample(self, sample, params):
        if self.frame_type == 'annotation':
            return F.resize(sample, self.size, interpolation=Image.NEAREST)
        elif self.frame_type == 'image':
            return cv.resize(sample, tuple(self.size[::-1]), interpolation=cv.INTER_LINEAR)
        elif self.frame_type == 'mask':
            return cv.resize(sample, tuple(self.size[::-1]), interpolation=cv.INTER_NEAREST)
        elif self.frame_type == 'flow':
            sample = normalize_flow(sample)
            sample = cv.resize(sample, tuple(self.size[::-1]), interpolation=cv.INTER_NEAREST)
            return denormalize_flow(sample)
        else:
            raise ValueError(self.frame_type)


class Lambda(Transform):

    def __init__(self, lambd):
        self.lambd = lambd

    def transform_sample(self, sample, params):
        return self.lambd(sample)
