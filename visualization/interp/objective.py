"Visualisation Objectives"

import torch
from ..hooks import Hook
from ..core import *
from ..imagenet import imagenet_stats, imagenet_labels

class Objective():
    """Defines an Objective which OptVis will optimise. The
    Objective class should have a callable function which
    should return the loss associated with the forward pass.
    This class has the same functionality as Lucid: objectives
    can be summed, multiplied by scalars, negated or subtracted.
    """
    def __init__(self, objective_function, name=None):
        """
        Parameters:
        objective_function: function that returns the loss of the network.
        name (str): name of the objective. Used for display. (optional)
        """
        self.objective_function = objective_function
        self.name = name

    def __call__(self, x):
        return self.objective_function(x)

    @property
    def cls_name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.cls_name}" if self.name is None else self.name

    def __add__(self, other):
        if isinstance(other, (int,float)):
            name = " + ".join([self.__repr__(), other.__repr__()])
            return Objective(lambda x: other + self(x), name=name)
        elif isinstance(other, Objective):
            name = " + ".join([self.__repr__(), other.__repr__()])
            return Objective(lambda x: other(x) + self(x), name=name)
        else:
            raise TypeError(f"Can't add value of type {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int,float)):
            name = f"{other}*{self.__repr__()}"
            return Objective(lambda x: other * self(x), name=name)
        else:
            raise TypeError(f"Can't add value of type {type(other)}")

    def __sub__(self, other):
        return self + (-1*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.__mul__(-1.)

class LayerObjective(Objective):
    """Generate an Objective from a particular layer of a network.
    Supports the layer indexing that interpret provides as well as
    options for selecting the channel or neuron of the layer.

    Parameters:
    model (nn.Module): PyTorch model.
    layer (str or int): the layer to optimise.
    channel (int): the channel to optimise. (optional)
    neuron (int): the neuron to optimise. (optional)
    shortcut (bool): Whether to attempt to shortcut the network's
        computation. Only works for Sequential type models.
    """
    def __init__(self, model, layer, channel=None, neuron=None, shortcut=False):
        self.model = model
        self.layer = layer
        self.channel = channel
        self.neuron = neuron
        self.shortcut = shortcut
        if self.shortcut:
            self.active = False

        try:
            self.model[layer]
        except:
            raise ValueError(f"Can't find layer {layer}. Use 'get_layer_names' to print all layer names.")

    def objective_function(self, x):
        "Apply the input to the network and set the loss."
        def layer_hook(module, input, output):
            if self.neuron is None:
                if self.channel is None:
                    self.loss = -torch.mean(output)
                else:
                    self.loss = -torch.mean(output[:, self.channel])
            else:
                if isinstance(module, nn.Conv2d):
                    # TODO: Check if channel is None and handle
                    self.loss = -torch.mean(output[:, self.channel, self.neuron])
                elif isinstance(module, nn.Linear):
                    self.loss = -torch.mean(output[:, self.neuron])
            self.active = True

        with Hook(self.model[self.layer], layer_hook, detach=False, clone=True):
            if self.shortcut:
                for i, m in enumerate(self.model.children()):
                    x = m(x)
                    if self.active:
                        self.active = False
                        break
            else:
                x = self.model(x)

        return self.loss

    def __repr__(self):
        msg = f"{self.cls_name}: {self.layer}"
        if self.channel is not None:
            msg += f":{self.channel}"
        if self.neuron is not None:
            msg += f":{self.neuron}"
        if self.channel is None and self.neuron is not None and self.model[self.layer].weight.size(0)==1000:
            msg += f"  {imagenet_labels[self.neuron]}"
        return msg
