import torch
from torch.nn import Module, Parameter


class RawParam(Module):
    """
    A raw 'parameterized image', that just wraps a normal tensor.
    This has to be the first layer in the network. It wraps the input and is differentiable
    """

    def __init__(self, input: torch.Tensor, cloned: bool = True):
        super().__init__()
        self.param = Parameter(input.clone().detach().requires_grad_() if cloned else input)

    def forward(self):
        return self.param

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.param.shape}'


# class ImageParam(Module):
#     """Class to create a parameterized image.
#
#     Parameters:
#     size (int): size of image. Image will be square.
#     fft (bool): parameterise the image in the Fourier domain.
#     decorrelate (bool): decorrelate the colours of the image.
#     sigmoid (bool): apply sigmoid after decorrelation to ensure
#         values are in range(0,1)
#     mean (list): means of the images used to train the network.
#     std (list): standard deviations of the images used to train the network.
#     kwargs: passed on to the image function fourier_image or random_im.
#     """
#
#     def __init__(self, size, fft=True, decorrelate=True, sigmoid=True, **kwargs):
#         super().__init__()
#         self.fft = fft
#         self.decorrelate = decorrelate
#         self.size = size
#         self.sigmoid = sigmoid
#
#         im_func = fourier_image if fft else random_im
#         self.noise, self.get_image = im_func(size, **kwargs)
#         self.noise = Parameter(self.noise)
#
#     def forward(self):
#         im = self.get_image(self.noise)
#
#         if self.decorrelate:
#             im = _linear_decorrelate_color(im)
#
#         if self.sigmoid:
#             # if we sigmoid, we should also normalize
#             im = torch.sigmoid(im)
#             im = norm(im, input_range=(0, 1), unsqueeze=False, grad=False, mean=self.mean, std=self.std)
#
#         return im
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}: {self.size}px, fft={self.fft}, decorrelate={self.decorrelate}"
