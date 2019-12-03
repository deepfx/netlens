"Visualise models"

import numpy as np
import torch
import torchvision
from PIL import Image

from ..core import *
from ..utils import *
from ..transforms import *
from ..utils import denorm
from .objective import *
from .param import *

VIS_TFMS = torchvision.transforms.Compose([
    RandomAffineTfm(scale, [0.9, 1.1]),
    RandomAffineTfm(rotate, 10),
])

class OptVis():
    """
    Class to visualise particular layers by optimisation. Visualisation
    follows the procedure outlined by Olah et al. [1] and
    implemented in Lucid [2].

    Parameters:
        model (nn.Module): PyTorch model.
        objective (Objective): The objective that the network will optimise.
            See factory methods from_layer.
        tfms (list): list of transformations to potentially apply to image.
        grad_tfms (list): list of transformations to apply to the gradient of the image.
        optim (torch.optim): PyTorch optimisation function.
        shortcut (bool): Attempt to shorten the computation by iterating through
            the layers until the objective is reached as opposed to calling the
            entire network. Only works on Sequential-like models.

    [1] - https://distill.pub/2017/feature-visualization/
    [2] - https://github.com/tensorflow/lucid
    """

    def __init__(self, model, objective, tfms=VIS_TFMS, grad_tfms=None, optim=torch.optim.Adam, shortcut=False):
        self.model = model
        self.objective = objective
        self.active = False
        self.tfms = tfms
        self.grad_tfms = grad_tfms
        self.optim_fn = optim
        self.shortcut = shortcut
        print(f"Optimising for {objective}")
        self.model.eval()

    def vis(self, img_param, thresh=(500,), transform=True, lr=0.05, wd=0., verbose=True):
        """
        Generate a visualisation by optimisation of an input. Updates img_param in-place.

        Parameters:
            img_param: object that parameterises the input noise.
            thresh (tuple): thresholds at which to display the generated image.
                Only displayed if verbose==True. Input optimised for max(thresh) iters.
            transform (bool): Whether to transform the input image using self.tfms.
            lr (float): learning rate for optimisation.
            wd (float): weight decay for self.optim_fn.
            verbose (bool): display input on thresholds.
        """
        if verbose:
            try:
                from IPython.display import display
            except ImportError:
                raise ValueError("Can't use verbose if not in IPython notebook.")

        freeze(self.model, bn=True)
        self.optim = self.optim_fn(img_param.parameters(), lr=lr, weight_decay=wd)
        for i in range(max(thresh)+1):
            img = img_param()

            if transform:
                img = self.tfms(img)

            loss = self.objective(img)
            loss.backward()

            # Potential debugging step.
            # print(img_param.noise.grad.abs().max(), img_param.noise.grad.abs().mean(),img_param.noise.grad.std())

            # Apply transforms to the gradient (normalize, blur, etc.)
            if self.grad_tfms is not None:
                with torch.no_grad():
                    img_param.noise.grad.data = self.grad_tfms(img_param.noise.grad.data)

            self.optim.step()
            self.optim.zero_grad()

            if verbose and i in thresh:
                print(i, loss.item())
                display(zoom(denorm(img), 2))

    @classmethod
    def from_layer(cls, model, layer, channel=None, neuron=None, shortcut=False, **kwargs):
        "Factory method to create OptVis from a LayerObjective. See respective classes for docs."
        if ":" in layer:
            layer, channel = layer.split(":")
            channel = int(channel)
        obj = LayerObjective(model, layer, channel, neuron=neuron, shortcut=shortcut)
        return cls(model, obj, **kwargs)

    @classmethod
    def from_dream(cls, model, layer, **kwargs):
        "Factory method for deepdream objective"
        return cls(model, DeepDreamObjective(model, layer), **kwargs)
