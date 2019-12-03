"""Core utilities for this project"""

from torch import nn

def get(s,i):
    if isinstance(i, int):
        if isinstance(s, nn.Sequential):
            return _orig_seq_get(s, i)
        else:
            return list(s.children())[i]
    elif isinstance(i, str):
        layers = i.split("/")
        l = s
        for layer in layers:
            l = getattr(l, layer)
        return l

def freeze(s,bn=False):
    def inner(m):
        if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad_(False)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad_(False)
    s.apply(inner)

def unfreeze(s):
    for p in s.parameters():
        p.requires_grad_(True)

_orig_seq_get = nn.Sequential.__getitem__
nn.Sequential.__getitem__ = get
nn.Module.__getitem__ = get
nn.Module.freeze = freeze
nn.Module.unfreeze = unfreeze
