# replacement for lucid

from .spatial import *


def image(w, h=None, batch=None, sd=None, decorrelate=True, fft=True, alpha=False):
    h = h or w
    batch = batch or 1
    channels = 4 if alpha else 3
    shape = (batch, channels, w, h)
    param_f = fft_image if fft else naive
    t = param_f(shape, sd=sd)
    rgb = to_valid_rgb(t[:,:3,:,:], decorrelate=decorrelate, sigmoid=True)
    print(f'Image: t.shape={t.shape}, rgb.shape={rgb.shape}') 
    if alpha:
        a = torch.sigmoid(t[:,3:,:,:])
        return torch.stack([rgb, a], dim=1)
    return rgb
