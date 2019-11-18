"Parameterizations of images"
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.parameter import Parameter

from .imagenet import imagenet_stats
from .transforms import resize_norm_transform
from .utils import denorm, norm

# Decorrelation code ported from Lucid: https://github.com/tensorflow/lucid
color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_mean = [0.48, 0.46, 0.41]


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


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)
    return np.sqrt(fx * fx + fy * fy)


def fourier_image(size, noise_scale=0.01, decay_power=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Image initialized in the Fourier domain"""
    freqs = rfft2d_freqs(size, size)
    noise = (noise_scale * torch.randn((3, *freqs.shape, 2))).to(device)  # dimensions: (C,W,H,Re/Im)

    def _get_image(_noise):
        # Normalize the input
        scale = 1.0 / np.maximum(freqs, 1.0 / max(size, size)) ** decay_power
        scale *= np.sqrt(size * size)
        scaled_spectrum_t = torch.tensor(scale, dtype=torch.float32, device=device)[None, ..., None] * _noise

        output = torch.irfft(scaled_spectrum_t, signal_ndim=2, onesided=False).unsqueeze(0)
        return output

    return noise, _get_image


def random_im(size, sd=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Create a random 'image' from a normal distribution"""
    # im = torch.randn(1, 3, size, size, device=device)*sd
    im = norm(torch.tensor(np.random.uniform(150, 180, (size, size, 3)), device=device).permute(2, 0, 1).float())
    return im, lambda x: x


class ImageParam(nn.Module):
    """Class to create a parameterized image.

    Parameters:
    size (int): size of image. Image will be square.
    fft (bool): parameterise the image in the Fourier domain.
    decorrelate (bool): decorrelate the colours of the image.
    sigmoid (bool): apply sigmoid after decorrelation to ensure
        values are in range(0,1)
    mean (list): means of the images used to train the network.
    std (list): standard deviations of the images used to train the network.
    kwargs: passed on to the image function fourier_image or random_im.
    """

    def __init__(self, size, fft=True, decorrelate=True, sigmoid=True, mean=imagenet_stats[0], std=imagenet_stats[1], **kwargs):
        super().__init__()
        self.fft = fft
        self.decorrelate = decorrelate
        self.size = size
        self.sigmoid = sigmoid
        self.mean = mean
        self.std = std

        im_func = fourier_image if fft else random_im
        self.noise, self.get_image = im_func(size, **kwargs)
        self.noise = Parameter(self.noise)

    def forward(self):
        im = self.get_image(self.noise)

        if self.decorrelate:
            im = _linear_decorrelate_color(im)

        if self.sigmoid:
            # if we sigmoid, we should also normalize
            im = torch.sigmoid(im)
            #im = norm(im, input_range=(0, 1), unsqueeze=False, grad=False, mean=self.mean, std=self.std)

        return im

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.size}px, fft={self.fft}, decorrelate={self.decorrelate}"


class ImageFile(nn.Module):
    """Create a parameterised image from a local image.

    Parameters:
    fn (str): image filename.
    size (int): size to resize image.
    decorrelate (bool): apply colour decorrelation to the image.
    kwargs: passed on to resize_norm_transform.
    """

    def __init__(self, fn, size, decorrelate=False, **kwargs):
        super().__init__()
        self.decorrelate = decorrelate
        self.size = size
        self.image = resize_norm_transform(size, **kwargs)(Image.open(fn))[None]
        self.image_tensor = Parameter(self.image)

    def forward(self):
        im = self.image_tensor

        if self.decorrelate:
            im = _linear_decorrelate_color(im)

        return im

    def _repr_html_(self):
        from IPython.display import display
        return display(denorm(self.image))
