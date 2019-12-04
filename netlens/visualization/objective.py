from typing import Optional

import torch
from torch import nn

from netlens.hooks import ModuleHook
from netlens.modules import FlatModel


class Objective:

    def __init__(self, name: str = None):
        self.name = name

    def objective_function(self, x):
        raise NotImplementedError()

    @property
    def cls_name(self):
        return self.__class__.__name__

    def __call__(self, x):
        return self.objective_function(x)

    def __repr__(self):
        return self.name or self.cls_name


class LayerObjective(Objective):
    model: FlatModel

    """
    Generate an Objective from a particular layer of a network.

    Parameters:
    model (FlatModel):
    layer (str):
    channel (int):
    neuron (int):
    shortcut (bool): Whether to attempt to shortcut the network's computation.
    """

    def __init__(self, model: FlatModel, layer: str, channel: Optional[int] = None, neuron: Optional[int] = None, shortcut: bool = False):
        super().__init__('layer_obj')
        self.model = model
        self.layer = layer
        self.channel = channel
        self.neuron = neuron
        self.shortcut = shortcut

        self.module = self.model.get_module(self.layer)
        assert self.module is not None, f'No module with layer key {self.layer} found!'

    def objective_function(self, x):
        # in any case, we need a special hook to extract the value
        def _layer_hook(_module, _input, _output):
            output = _output.clone()
            if self.channel is None and self.neuron is None:
                return -torch.mean(output)

            # if a channel or neuron were given, only some modules are supported
            channel_idx = self.channel or slice(None)
            neuron_idx = self.neuron or slice(None)

            if isinstance(_module, nn.Conv2d):
                return -torch.mean(output[:, channel_idx, neuron_idx])
            elif isinstance(_module, nn.Linear):
                # there are no channels in a Linear module
                return -torch.mean(output[:, neuron_idx])
            else:
                raise Exception(f'LayerObject does not support module {_module.__class__.__name__} with channel {self.channel}, neuron {self.neuron}')

        with ModuleHook(self.module, _layer_hook, detach=False) as hook:
            self.model.forward(x, until_layer=self.layer if self.shortcut else None)
            return hook.stored

    def __repr__(self):
        return f'{self.cls_name}:{self.layer}:{self.channel}:{self.neuron}'
