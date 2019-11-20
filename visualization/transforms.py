import math
import random
from typing import Tuple

import torch
from PIL import Image
from torch.nn.functional import affine_grid, grid_sample
from torchvision.transforms import RandomCrop, Compose, Lambda

__all__ = ['Thumbnail', 'Jitter', 'VIS_TFMS']


# Transforms PIL.Image

class Thumbnail(object):

    def __init__(self, size: Tuple[int, int], resample: int = Image.BICUBIC):
        self.size = size
        self.resample = resample

    def __call__(self, img: Image.Image):
        img = img.copy()
        img.thumbnail(self.size, self.resample)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, resample={1})'.format(self.size, self.resample)


class Jitter(RandomCrop):
    """
    Emulates the "jitter" transform from Tensorflow/Lucid, which is basically a random crop but where the decrease in size is provided, instead of the
    final size.
    """

    def __init__(self, d: int, *args, **kwargs):
        # we need to have a valid value of size until an actual image comes
        super().__init__((0, 0), *args, **kwargs)
        self.d = d

    def __call__(self, img: Image.Image):
        w, h = img.size
        self.size = (h - self.d, w - self.d)
        return super().__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(d={0}, padding={1})'.format(self.d, self.padding)


"""
standard_transforms = [
    pad(12, mode='constant', constant_value=.5),
    jitter(8),
    random_scale([1 + (i-5)/50. for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5*[0]),
    jitter(4),
  ]
"""


# Transforms on torch.Tensor

class RandomCropTensor:

    def __init__(self, size=None, delta=None):
        assert (size is not None) ^ (delta is not None), "Exactly one of 'size' and 'delta' must be provided."
        self.size = size
        self.delta = delta

    def __call__(self, x):
        h, w = x.shape[-2:]
        size = self.size if self.size is not None else (h - self.delta, w - self.delta)
        offset_h = random.randint(0, h - size[0])
        offset_w = random.randint(0, w - size[1])
        return x[..., offset_h: offset_h + size[0], offset_w: offset_w + size[1]]


def affine(mat: torch.Tensor):
    def inner(x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # makes sure it is (N,C,W,H)
        grid = affine_grid(mat.to(x.device), x.size())
        return grid_sample(x, grid, padding_mode="reflection")

    return inner


def rotate(angle, radians: bool = False):
    rad = angle if radians else math.pi * angle / 180.0
    rot = torch.tensor([[math.cos(rad), -math.sin(rad), 0],
                        [math.sin(rad), math.cos(rad), 0]]).unsqueeze(0)
    return affine(rot)


def translate(x, y):
    rot = torch.tensor([[1.0, 0.0, -x],
                        [0.0, 1.0, y]]).unsqueeze(0)
    return affine(rot)


def shear(shear_factor):
    rot = torch.tensor([[1.0, shear_factor, 0.0],
                        [0.0, 1.0, 0.0]]).unsqueeze(0)
    return affine(rot)


def scale(scale_factor):
    mat = torch.tensor([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]]).unsqueeze(0) / scale_factor
    return affine(mat)


class RandomAffineTfm:
    """Randomly apply an affine transform on a tensor."""

    def __init__(self, tfm, interval=None, values=None):
        assert (interval is not None) ^ (values is not None), "Exactly one of 'interval' or 'values' has to be provided."
        self.tfm = tfm
        self.interval = interval if isinstance(interval, (tuple, list)) else (-interval, interval) if interval is not None else None
        self.values = values

    def __call__(self, x):
        value = random.choice(self.values) if self.values is not None else random.uniform(*self.interval)
        return self.tfm(value)(x)


VIS_TFMS = Compose([
    Lambda(lambda img: torch.nn.functional.pad(img, pad=[6, 6, 6, 6], mode='constant', value=0.5)),
    RandomCropTensor(delta=8),
    RandomAffineTfm(scale, values=[1 + (i - 5) / 50. for i in range(11)]),
    RandomAffineTfm(rotate, values=list(range(-10, 11)) + 5 * [0]),
    RandomCropTensor(delta=4)
])
