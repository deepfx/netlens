from typing import Tuple

from PIL import Image
from torchvision.transforms import RandomCrop, RandomAffine, ToPILImage, ToTensor, Compose

__all__ = ['Thumbnail', 'Jitter', 'VIS_TFMS']


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
        super(Jitter, self).__init__((0, 0), *args, **kwargs)
        self.d = d

    def __call__(self, img: Image.Image):
        w, h = img.size
        self.size = (h - self.d, w - self.d)
        return super.__call__(img)

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

VIS_TFMS = Compose([
    ToPILImage(),
    Jitter(8, padding=12, fill=128, padding_mode='constant'),  # pad + jitter from lucid together here
    RandomAffine(degrees=(-10, 10), scale=[1 + (i - 5) / 50. for i in range(11)]),  # random_scale + random_rotate from lucid together here
    Jitter(4),
    ToTensor()
])
