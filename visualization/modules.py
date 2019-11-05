import copy
from collections import defaultdict
from typing import List, Tuple, Iterable, Optional, Any

import torch
from pydash import find_index
from torch import nn

from .utils import clean_layer, tuple_to_key, key_to_tuple

# GENERIC NAMES FOR DIFFERENT LAYERS
# externally exposed – to smoothen out PyTorch changes
MODULE_NAME_MAP = {
    nn.Conv2d: 'conv',
    nn.ReLU: 'relu',
    nn.MaxPool2d: 'pool',
    nn.BatchNorm2d: 'bn'
}


def get_module_name(module: nn.Module) -> str:
    clazz = module.__class__
    if clazz in MODULE_NAME_MAP:
        return MODULE_NAME_MAP[clazz]
    if hasattr(clazz, 'name'):
        return clazz.name
    return clazz.__name__


def generate_module_keys(modules: Iterable[nn.Module], use_tuples=False) -> Iterable[Tuple[Any, nn.Module]]:
    name_counts = defaultdict(int)

    def gen_key(m):
        name = get_module_name(m)
        nth = name_counts[name]
        name_counts[name] += 1
        return (name, nth) if use_tuples else tuple_to_key(name, nth)

    return [(gen_key(m), m) for m in modules]


class Normalization(nn.Module):
    name: str = 'normalization'

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class LayeredModule(nn.Module):
    layers: nn.ModuleDict
    hooked_layer_keys: Optional = None

    def __init__(self, layers, hooked_layer_keys=None):
        super(LayeredModule, self).__init__()
        self.layers = nn.ModuleDict(layers)

        self.hooks_forward = {}
        self.hooks_backward = {}

        self.layer_outputs = {}
        self.param_gradients = {}

        self.set_hooked_layers(hooked_layer_keys)

    @staticmethod
    def from_cnn(cnn, normalizer):
        """
        Converts a generic CNN into our standardized LayeredModule. The layer ids are inferred automatically from the CNN's layers.
        :param cnn:
        :param normalizer:
        :return:
        """
        cnn = copy.deepcopy(cnn)
        return LayeredModule(generate_module_keys([normalizer] + [clean_layer(layer) for layer in cnn.children()]))

    # TODO: implement similar static methods for other archs

    def set_layer_context(self, layer_key):
        def forward_hook_callback(_layer_, _input_, output):
            self.layer_outputs[layer_key] = output

        return forward_hook_callback

    def set_hooked_layers(self, layer_keys):
        """
        Allows to specify which layers (by keys) should be hooked. WARNING: it restarts all the hooks states!!!
        """
        self.hooked_layer_keys = layer_keys
        self.hook_forward()
        self.hook_backward()

    def is_hooked_layer(self, layer_key) -> bool:
        return self.hooked_layer_keys is None or layer_key in self.hooked_layer_keys

    def hook_forward(self):
        # remove any previously set hooks
        for hook in self.hooks_forward.values():
            hook.remove()
        self.hooks_forward.clear()
        self.layer_outputs.clear()

        for layer_key, layer in self.layers.items():
            if self.is_hooked_layer(layer_key):
                self.hooks_forward[layer_key] = layer.register_forward_hook(self.set_layer_context(layer_key))

    def hook_backward(self):
        # remove any previously set hooks
        for hook in self.hooks_backward.values():
            hook.remove()
        self.hooks_backward.clear()
        self.param_gradients.clear()

        def set_param_context(name):
            def hook_fn(grad):
                self.param_gradients[name] = grad
                return grad

            return hook_fn

        for name, param in self.layers.named_parameters():
            param.register_hook(set_param_context(name))

    def get_gradients_for_sample(self, input, target_class):
        # Put model in evaluation mode
        self.layers.eval()
        model_output = self.forward(input)

        self.zero_grad()
        num_classes = model_output.size()[-1]
        one_hot_output = torch.FloatTensor(1, num_classes).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)

        return self.param_gradients  # to numpy?? #numpy()[0]

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def get_modules(self, name: str) -> List[nn.Module]:
        return [layer for key, layer in self.layers.items() if key_to_tuple(key)[0] == name]

    def get_module(self, layer_key: str) -> nn.Module:
        return self.layers[layer_key]

    def get_layer_output(self, layer_key: str):
        return self.layer_outputs.get(layer_key)

    def insert_after(self, insertion_key: str, new_key: str, new_layer: nn.Module):
        layer_list = list(self.layers.items())
        idx = find_index(layer_list, lambda l: l[0] == insertion_key)
        if idx == -1:
            return
        layer_list.insert(idx + 1, (new_key, new_layer))
        self.layers = nn.ModuleDict(layer_list)

    def delete_all_after(self, last_key: Tuple[str, int]):
        layer_list = list(self.layers.items())
        idx = find_index(layer_list, lambda l: l[0] == last_key)
        if idx == -1:
            return
        self.layers = nn.ModuleDict(layer_list[:idx + 1])
