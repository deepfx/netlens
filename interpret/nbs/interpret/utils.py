import torch
from pathlib import Path
from PIL import Image
import numpy as np

from .imagenet import imagenet_stats

Path.ls = lambda c: list(c.iterdir())

def zoom(im, zoom=2):
    return im.transform((int(im.size[0]*zoom), int(im.size[1]*zoom)), Image.EXTENT, (0, 0, im.size[0], im.size[1]))

def denorm(im, mean=imagenet_stats[0], std=imagenet_stats[1], image=True):
    "Denormalize an image"
    if isinstance(im, torch.Tensor):
        im = im.detach().clone().cpu().squeeze()
    mean, std = torch.tensor(mean), torch.tensor(std)

    im *= std[..., None, None]
    im += mean[..., None, None]
    im *= 255
    im = im.permute(1, 2, 0).clamp(0,255).numpy()

    im = im.round().astype('uint8')
    if not image: return im
    return Image.fromarray(im)


def norm(im, input_range=(0,255), mean=imagenet_stats[0], std=imagenet_stats[1], unsqueeze=True, grad=True):
    "Normalize an image"
    if isinstance(im, Image.Image):
        im = torch.tensor(np.asarray(im)).permute(2,0,1).float()
    elif isinstance(im, np.ndarray):
        im = torch.tensor(im).float()
        size = im.size()
        assert len(size)==3 or len(size)==4, "Image has wrong number of dimensions."
        assert size[0]==3 or size[0]==1, "Image has invalid channel number. Should be 1 or 3."

    mean, std = torch.tensor(mean, device=im.device), torch.tensor(std, device=im.device)
    im = im + input_range[0]
    im = im / input_range[1]
    im = im - mean[..., None, None]
    im = im / std[..., None, None]
    if unsqueeze: im.unsqueeze_(0)
    if grad: im.requires_grad_(True)
    return im

def get_layer_names(m, upper_name='', _title=True):
    "Recursively show a network's named layers"
    name = ''
    if _title:
        print("{:^30} | {:^18} | {:^10} | {:^10}".format("Layer", "Class Name", "Input Size", "Output Size"))
        print(f"{'-'*30} | {'-'*18} | {'-'*10} | {'-'*10}")

    if type(m) == tuple:
        name, m = m

    if hasattr(m, 'named_children') and len(list(m.named_children()))!=0:
        for layer in m.named_children():
            get_layer_names(layer, upper_name=upper_name+name+"/" if name != '' else upper_name, _title=False)
    else:
        print("{:^30} | {:^18} | {:^10} | {:^10}".format(
                upper_name+name,
                m.__class__.__name__,
                m.weight.size(1) if hasattr(m, 'weight') and len(m.weight.shape)>1 else '-',
                m.weight.size(0) if hasattr(m, 'weight') else '-'))
