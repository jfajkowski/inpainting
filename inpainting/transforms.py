import abc

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from inpainting.utils import tensor_to_flow


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

    def __init__(self, size, frame_type):
        self.size = size
        self.frame_type = frame_type

    def transform_sample(self, sample, params):
        if self.frame_type == 'flow':
            _, image_width, image_height = sample.shape
            crop_height, crop_width = self.size
            crop_left = int(round((image_width - crop_width) / 2.))
            crop_top = int(round((image_height - crop_height) / 2.))
            return sample[:, crop_left:crop_left + crop_width, crop_top:crop_top + crop_height]

        return F.center_crop(sample, self.size)


class Resize(Transform):

    def __init__(self, size, frame_type):
        self.size = size
        self.frame_type = frame_type

    def transform_sample(self, sample, params):
        if self.frame_type == 'flow':
            return tensor_to_flow(
                (torch.nn.functional.interpolate(F.to_tensor(sample).unsqueeze(0), self.size)).squeeze(0))
        elif self.frame_type == 'annotation' or self.frame_type == 'mask':
            return F.resize(sample, self.size, Image.NEAREST)
        else:
            return F.resize(sample, self.size, Image.BILINEAR)


class Lambda(Transform):

    def __init__(self, lambd):
        self.lambd = lambd

    def transform_sample(self, sample, params):
        return self.lambd(sample)
