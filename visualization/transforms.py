from typing import Tuple

from PIL import Image

__all__ = ['Thumbnail']


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
