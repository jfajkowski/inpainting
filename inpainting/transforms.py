import abc
import numbers
import random
import numpy as np
import torch
from typing import Sized

from PIL import Image
from torchvision.transforms import functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


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
        return None

    def transform_sequence(self, sequence, params):
        for i in range(len(sequence)):
            sequence[i] = self.transform_sample(sequence[i], params)
        return sequence

    @abc.abstractmethod
    def transform_sample(self, sample, params):
        pass


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(Transform):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def transform_sample(self, sample, params):
        """
        Args:
            sample (PIL Image or numpy.ndarray): Image to be converted to tensor.
            params (None or tuple): Random transformation parameters if needed.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(Transform):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Sized) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def transform_sample(self, sample, params):
        """
        Args:
            sample (PIL Image): Image to be scaled.
            params (None or tuple): Random transformation parameters if needed.

        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(sample, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Lambda(Transform):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def transform_sample(self, sample, params):
        return self.lambd(sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'
