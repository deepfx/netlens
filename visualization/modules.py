import copy
from typing import List, Tuple, Iterable, Callable, Any, Collection, Optional

import fastai
import torch
import torchvision
from fastai.layers import Lambda
from pydash import find_index
from torch import nn, Tensor

from .hooks import HookDict, TensorHook
from .utils import clean_layer, get_name_from_key, get_parent_name, as_list, enumerate_module_keys, \
    insert_layer_after, delete_all_layers_after, update_set

# GENERIC NAMES FOR DIFFERENT LAYERS
# externally exposed – to smoothen out PyTorch changes
MODULE_NAME_MAP = {
    nn.Conv2d: 'conv',
    nn.ReLU: 'relu',
    nn.MaxPool2d: 'pool',
    nn.BatchNorm2d: 'bn',
    nn.AdaptiveAvgPool2d: 'avgpool',
    nn.Dropout: 'dropout',
    nn.Linear: 'linear'
}

NOT_FLATTEN_MODULES = {
    torchvision.models.resnet.BasicBlock,
    torchvision.models.resnet.Bottleneck,
    fastai.layers.AdaptiveConcatPool2d
}

MODELS_CONFIG = {
    'input_size': {
        'AlexNet': (224, 224)
    }
}


def get_module_name(module: nn.Module) -> str:
    clazz = module.__class__
    if clazz in MODULE_NAME_MAP:
        return MODULE_NAME_MAP[clazz]
    if hasattr(clazz, 'name'):
        return clazz.name
    return clazz.__name__


def get_module_names(modules: Iterable[nn.Module]) -> Iterable[Tuple[str, nn.Module]]:
    return [(get_module_name(m), m) for m in modules]


def get_flat_layers(model: nn.Module, prepended_layers=None) -> Iterable[Tuple[str, nn.Module]]:
    """
    Returns all the sub-modules of the given model as a list of named layers, assuming that the provided model is FLAT.
    Optionally pre-prepends some layers at the beginning.
    """
    layers = [clean_layer(layer) for layer in model.children()]
    return enumerate_module_keys(get_module_names(as_list(prepended_layers) + layers))


def get_nested_layers(model: nn.Module, dont_flatten: Collection[type] = None) -> Iterable[Tuple[str, nn.Module]]:
    """
    Returns all the sub-modules of the given model as a list of named layers, assuming that the provided model is NESTED. In that case, the names of
    the non-leaf 'parent' modules are prepended to the generated keys.
    """
    dont_flatten = dont_flatten or NOT_FLATTEN_MODULES
    parents = set(get_parent_name(name) for name, _ in model.named_modules())
    dont_flatten_names = set(name for name, layer in model.named_modules() if type(layer) in dont_flatten)

    def get_prefix(name):
        parent = get_parent_name(name)
        return parent.replace('.', '-') + '-' if len(parent) > 0 else ''

    return enumerate_module_keys((get_prefix(name) + get_module_name(layer), clean_layer(layer))
                                 for name, layer in model.named_modules()
                                 if (name not in parents or type(layer) in dont_flatten)
                                 and not any(name.startswith(p + '.') for p in dont_flatten_names))


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


CustomHookFunc = Callable[[nn.Module, str], Callable[[Tensor], Any]]


class LayeredModule(nn.Module):
    """
    **HOOKS**

    There are several places where you can hook to the network components.

    1. Layers (nn.Module)
       Forward hooks: they capture the OUTPUT of the module in the forward pass.

    2. Parameters (torch.Tensor)
       It is possible to hook to the parameter themselves (e.g. weights, biases) when their gradients are computed.

    3. Layer activation tensors (torch.Tensor)
       We can also hook to the output itself of each layer after the forward pass; to see the incoming gradients to it in the backward pass.

    """
    layers: nn.ModuleDict
    arch_name: str
    hooked_layer_keys = set()
    hooked_param_layer_keys = set()
    hooked_activation_keys = set()

    def __init__(self, layers, arch_name: str, hooked_layer_keys=None, hooked_activation_keys=None, hooked_param_layer_keys=None,
                 hook_to_activations: bool = False, custom_activation_hook_factory: CustomHookFunc = None):
        super(LayeredModule, self).__init__()
        self.layers = nn.ModuleDict(layers)
        self.arch_name = arch_name

        self.hooks_layers = None
        self.hooks_activations = None
        self.hooks_params = None

        self.set_hooked_layers(hooked_layer_keys)
        self.set_hooked_params(hooked_param_layer_keys)
        self.set_hooked_activations(hooked_activation_keys)

        self.hook_to_activations = hook_to_activations
        self.custom_activation_hook_factory = custom_activation_hook_factory

    def copy(self):
        return self.__class__(self.layers.items(), self.arch_name, self.hooked_layer_keys, self.hooked_activation_keys, self.hooked_param_layer_keys,
                              self.hook_to_activations, self.custom_activation_hook_factory)

    @classmethod
    def from_cnn(cls, cnn, prepended_layers=None, *args, **kwargs):
        """
        Converts a generic CNN into our standardized LayeredModule. The layer ids are inferred automatically from the CNN's layers.
        :param cnn:
        :param prepended_layers:
        :return:
        """
        cnn = copy.deepcopy(cnn)
        return cls(get_flat_layers(cnn, prepended_layers), cnn.__class__.__name__, *args, **kwargs)

    @classmethod
    def from_nested_cnn(cls, model, *args, **kwargs):
        layers = get_nested_layers(model)
        # we saw that the flattening is always before the first Linear layer
        idx = find_index(layers, lambda l: get_name_from_key(l[0]) == 'linear')
        if idx >= 0:
            layers.insert(idx, ('flatten', Lambda(lambda x: torch.flatten(x, 1))))
        return cls(layers, model.__class__.__name__, *args, **kwargs)

    # TODO: implement similar static methods for other archs

    def set_hooked_layers(self, layer_keys, keep: bool = True):
        """
        Allows to specify which layers (by keys) should be hooked. WARNING: it restarts all the hooks states!!!
        """
        update_set(self.hooked_layer_keys, layer_keys, keep)
        self.hook_layers()

    def set_hooked_params(self, layer_keys, keep: bool = True):
        update_set(self.hooked_param_layer_keys, layer_keys, keep)
        self.hook_parameters()

    def set_hooked_activations(self, layer_keys, keep: bool = True):
        update_set(self.hooked_activation_keys, layer_keys, keep)

    def hook_layers(self):
        # remove any previously set hook handlers
        self.hooks_layers = HookDict.from_modules(
            {layer_key: layer for layer_key, layer in self.layers.items() if layer_key in self.hooked_layer_keys},
            lambda _m, _input, output: output)

    def hook_parameters(self):
        # remove any previously set hook handlers
        self.hooks_params = HookDict.from_tensors(
            {param_key: param for param_key, param in self.layers.named_parameters() if get_parent_name(param_key) in self.hooked_param_layer_keys},
            lambda grad: grad)

    def _add_activation_hook(self, key: str, x: Tensor):
        _hook_func = None
        if self.custom_activation_hook_factory is not None:
            _hook_func = self.custom_activation_hook_factory(self, key)
        if _hook_func is None:
            _hook_func = lambda grad: grad
        self.hooks_activations[key] = TensorHook(x, _hook_func)

    def forward(self, x, until_layer: Optional[str] = None):
        if self.hook_to_activations:
            self.hooks_activations = HookDict()
            # here, 'x' has the input, so we can hook to it
            self._add_activation_hook('input', x)

        for layer_key, layer in self.layers.items():
            x = layer(x)  # all you need from external
            # if we enabled hooks to the outputs, add them now
            if self.hook_to_activations and layer_key in self.hooked_activation_keys:
                self._add_activation_hook(layer_key, x)
            if layer_key == until_layer:
                break
        return x

    def get_layer_output(self, key: str) -> Tensor:
        return self.hooks_layers.get_stored(key)

    def get_activation_gradient(self, key: str) -> Tensor:
        return self.hooks_activations.get_stored(key)

    def get_modules(self, name: str) -> List[nn.Module]:
        return [layer for key, layer in self.layers.items() if get_name_from_key(key) == name]

    def get_module(self, layer_key: str) -> nn.Module:
        return self.layers[layer_key]

    def insert_after(self, insertion_key: str, new_key: str, new_layer: nn.Module):
        layer_list = list(self.layers.items())
        self.layers = nn.ModuleDict(insert_layer_after(layer_list, insertion_key, new_key, new_layer))

    def delete_all_after(self, last_key: str):
        layer_list = list(self.layers.items())
        self.layers = nn.ModuleDict(delete_all_layers_after(layer_list, last_key))
