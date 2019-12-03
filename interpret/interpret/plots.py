"""Plot images"""

from .utils import denorm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import math

def show_image(tensor, normalize=False, ax=None, **kwargs):
    img = denorm(tensor, **kwargs) if normalize else np.array(tensor.clone())
    plt.imshow(img) if ax is None else ax.imshow(img)

def show_images(batched_tensor, normalize=False, figsize=(5,5), axis=False, labels=None, title=None, **kwargs):
    r = math.ceil(math.sqrt(batched_tensor.size(0)))
    f,axes = plt.subplots(r,r,figsize=figsize)
    if title is not None: f.suptitle(title)
    for i,ax in enumerate(axes.flatten()):
        if i<batched_tensor.size(0):
            show_image(batched_tensor[i],normalize,ax, **kwargs)
            if labels is not None: ax.set_title(f'{labels[i]}')
        if not axis: ax.set_axis_off()

def plot(y, x=None, title=None, ax=None, x_lb=None, y_lb=None, label=None):
    if ax is None:
        plt.plot(y, label=label) if x is None else plt.plot(x, y, label=label)
        plt.title(title)
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)
        if label is not None: plt.legend()
    else:
        ax.plot(y, label=label) if x is None else ax.plot(x, y, label=label)
        ax.set_title(title)
        ax.set_xlabel(x_lb)
        ax.set_ylabel(y_lb)
        if label is not None: ax.legend()
