import abc

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

    def __init__(self, size):
        self.size = size

    def transform_sample(self, sample, params):
        return F.center_crop(sample, self.size)


class Resize(Transform):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def transform_sample(self, sample, params):
        return F.resize(sample, self.size, self.interpolation)


class Lambda(Transform):

    def __init__(self, lambd):
        self.lambd = lambd

    def transform_sample(self, sample, params):
        return self.lambd(sample)
