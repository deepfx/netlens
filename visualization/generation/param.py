from typing import Tuple, Callable, Union

import numpy as np
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


# Decorrelation code ported from Lucid: https://github.com/tensorflow/lucid
color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]


def _get_default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _linear_decorrelate_color(t: torch.Tensor) -> torch.Tensor:
    """Multiply input by sqrt of empirical (ImageNet) color correlation matrix.

    If you interpret t's innermost dimension as describing colors in a
    decorrelated version of the color space (which is a very natural way to
    describe colors -- see discussion in Feature Visualization article) the way
    to map back to normal colors is multiply the square root of your color
    correlations.
    """
    assert t.shape[0] == 1  # must be (N,C,W,H)
    t_flat = t.squeeze(0).view((3, -1))
    color_correlation_normalized = torch.tensor(color_correlation_svd_sqrt / max_norm_svd_sqrt, device=t.device)
    t_flat = color_correlation_normalized @ t_flat
    t = t_flat.view(t.shape)
    return t


def rfft2d_freqs(h: int, w: int) -> np.ndarray:
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)
    return np.sqrt(fx * fx + fy * fy)


def fourier_image(size: Tuple[int, int], noise_scale: float = 0.01, decay_power: float = 1.0, device: str = None) -> Tuple[torch.Tensor, Callable]:
    """Image initialized in the Fourier domain"""
    device = device or _get_default_device()
    freqs = rfft2d_freqs(*size)
    noise = (noise_scale * torch.randn((3, *freqs.shape, 2))).to(device)  # dimensions: (C,W,H,Re/Im)

    def _get_image(_noise):
        # Normalize the input
        scale = 1.0 / np.maximum(freqs, 1.0 / max(*size)) ** decay_power
        scale *= np.sqrt(size[0] * size[1])
        scaled_spectrum_t = torch.tensor(scale, dtype=torch.float32, device=device)[None, ..., None] * _noise

        output = torch.irfft(scaled_spectrum_t, signal_ndim=2, onesided=False).unsqueeze(0)
        return output

    return noise, _get_image


def random_image(size: Tuple[int, int], sd: float = 0.5, device: str = None) -> Tuple[torch.Tensor, Callable]:
    """Create a random 'image' from a normal distribution"""
    device = device or _get_default_device()
    im = torch.randn(1, 3, *size, device=device) * sd
    return im, lambda x: x


class ImageParam(Module):
    """Class to create a parameterized image.

    Parameters:
    size: size of image, can be a tuple or an integer. If it's a tuple, the image will be square.
    fft (bool): parameterize the image in the Fourier domain.
    decorrelate (bool): decorrelate the colours of the image.
    sigmoid (bool): apply sigmoid after decorrelation to ensure values are in range(0,1)
    kwargs: passed on to the image function fourier_image or random_im.
    """

    def __init__(self, size: Union[int, Tuple[int, int]], fft: bool = True, decorrelate: bool = True, sigmoid: bool = True, **kwargs):
        super().__init__()
        self.fft = fft
        self.decorrelate = decorrelate
        self.size = (size, size) if isinstance(size, int) else size
        self.sigmoid = sigmoid

        im_func = fourier_image if fft else random_image
        self.noise, self.get_image = im_func(size, **kwargs)
        self.noise = Parameter(self.noise)

    def forward(self):
        im = self.get_image(self.noise)
        if self.decorrelate:
            im = _linear_decorrelate_color(im)
        if self.decorrelate and not self.sigmoid:
            im += color_mean
        if self.sigmoid:
            return torch.sigmoid(im)
        else:
            return im.clamp(min=0.0, max=1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.size}px, fft={self.fft}, decorrelate={self.decorrelate}"
