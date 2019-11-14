from .color import *


def naive(shape, sd=None):
    raise NotImplementedError()


def _rfft2d_freqs(h, w):
    """Compute 2d spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[:w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[:w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1):
    b, h, w, ch = shape
    imgs = []
    for _ in range(b):
        freqs = _rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        sd = sd or 0.01
        # init_val = sd * np.random.randn(2, ch, fh, fw).astype("float32")
        # spectrum_var = tf.Variable(init_val)
        # spectrum = tf.complex(spectrum_var[0], spectrum_var[1])
        spectrum = torch.randn((2, ch, fh, fw), requires_grad=True)
        spectrum_scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** decay_power
        # Scale the spectrum by the square-root of the number of pixels
        # to get a unitary transformation. This allows to use similar
        # learning rates to pixel-wise optimisation.
        spectrum_scale *= np.sqrt(w * h)
        scaled_spectrum = spectrum * torch.from_numpy(spectrum_scale.astype('float32'))
        # img = tf.spectral.irfft2d(scaled_spectrum)
        img = torch.irfft(scaled_spectrum.permute((1, 2, 3, 0)), signal_ndim=2)
        # in case of odd input dimension we cut off the additional pixel
        # we get from irfft2d length computation
        img = img[:ch, :h, :w]
        img = img.permute((1, 2, 0))  # torch.transpose(img, [1, 2, 0])
        imgs.append(img)
    return torch.stack(imgs, dim=0) / 4.0
