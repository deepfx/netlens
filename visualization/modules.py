import copy
from typing import List, Tuple, Optional, Iterable

import torch
from fastai.layers import Lambda
from torch import nn

from .math import one_hot_tensor
from .utils import clean_layer, key_to_tuple, get_parent_name, as_list, enumerate_module_keys, insert_layer_after, delete_all_layers_after

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


def get_nested_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """
    Returns all the sub-modules of the given model as a list of named layers, assuming that the provided model is NESTED. In that case, the names of
    the non-leaf 'parent' modules are prepended to the generated keys.
    """
    parents = set(get_parent_name(name) for name, _ in model.named_modules())

    def get_prefix(name):
        parent = get_parent_name(name)
        return parent.replace('.', '-') + '-' if len(parent) > 0 else ''

    return enumerate_module_keys((get_prefix(name) + get_module_name(layer), clean_layer(layer))
                                 for name, layer in model.named_modules() if name not in parents)


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


def _clear_hooks(hooks: dict) -> None:
    [h.remove() for h in hooks.values()]
    hooks.clear()


class LayeredModule(nn.Module):
    """
    **HOOKS**

    There are several places where you can hook to the network components.

    1. Layers (nn.Module)
       Forward hooks: they capture the OUTPUT of the module in the forward pass.

       - Handlers: self.hooks_layer_forward
       - Stored values: self.layer_outputs : a Mapping[str, Tensor], the key is the layer key, the value is the layer's output after the forward pass.
       Backward hooks: they capture the GRADIENTS of the module in the backward pass.

       - Handlers: self.hooks_layer_backward
       - Stored values: self.layer_gradients : a Mapping[str, Tuple[Tensor]], the key is the layer key, the value is the layer's INCOMING gradient(s).
         Note that there can be several gradients, hence a tuple.

    2. Parameters (torch.Tensor)
       It is possible to hook to the parameter themselves (e.g. weights, biases) when their gradients are computed.

       - Handlers: self.hooks_params
       - Stored values: self.param_gradients : a Mapping[str, Tensor], the key is the parameter name; the value is the incoming gradient to this
         tensor.

    3. Layer output tensors (torch.Tensor)
       We can also hook to the output itself of each layer after the forward pass; to see the incoming gradients to it in the backward pass.

       - Handlers: self.hooks_outputs
       - Stored values: self.output_gradients : a Mapping[str, Tensor]

    """
    layers: nn.ModuleDict
    hooked_layer_keys: Optional = None

    def __init__(self, layers, hooked_layer_keys=None, hook_to_activations=False):
        super(LayeredModule, self).__init__()
        self.layers = nn.ModuleDict(layers)

        self.hooks_layer_forward = {}
        self.hooks_layer_backward = {}
        self.hooks_activations = {}
        self.hooks_params = {}

        self.layer_outputs = {}
        self.layer_gradients = {}
        self.activation_gradients = {}
        self.param_gradients = {}

        self.set_hooked_layers(hooked_layer_keys)
        self.hook_to_activations = hook_to_activations

    @staticmethod
    def from_cnn(cnn, prepended_layers=None):
        """
        Converts a generic CNN into our standardized LayeredModule. The layer ids are inferred automatically from the CNN's layers.
        :param cnn:
        :param prepended_layers:
        :return:
        """
        cnn = copy.deepcopy(cnn)
        return LayeredModule(get_flat_layers(cnn, prepended_layers))

    @staticmethod
    def from_alexnet(model):
        layers = get_nested_layers(model)
        # the Pytorch implementation of AlexNet has a flatten in the forward, we need to insert it in the layers
        layers = insert_layer_after(layers, 'avgpool-0', 'flatten', Lambda(lambda x: torch.flatten(x, 1)))
        return LayeredModule(layers)

    # TODO: implement similar static methods for other archs

    def create_layer_callbacks(self, layer_key):
        def forward_hook_callback(_layer_, _input_, output):
            self.layer_outputs[layer_key] = output

        def backward_hook_callback(_layer_, grad_in, grad_out):
            # TODO: it will just have hardcoded indexes!!!
            self.layer_gradients[layer_key] = (grad_in, grad_out)

        return forward_hook_callback, backward_hook_callback

    def create_activation_callback(self, layer_key):
        def output_hook_callback(grad):
            self.activation_gradients[layer_key] = grad

        return output_hook_callback

    def set_hooked_layers(self, layer_keys):
        """
        Allows to specify which layers (by keys) should be hooked. WARNING: it restarts all the hooks states!!!
        """
        self.hooked_layer_keys = layer_keys
        self.hook_layers()
        self.hook_parameters()

    def is_hooked_layer(self, layer_key) -> bool:
        return self.hooked_layer_keys is None or layer_key in self.hooked_layer_keys

    def hook_layers(self):
        # remove any previously set hook handlers
        [_clear_hooks(hooks) for hooks in (self.hooks_layer_forward, self.hooks_layer_backward)]
        # clear stored values
        [v.clear() for v in (self.layer_outputs, self.layer_gradients)]

        for layer_key, layer in self.layers.items():
            if self.is_hooked_layer(layer_key):
                fwd_cb, bwd_cb = self.create_layer_callbacks(layer_key)
                self.hooks_layer_forward[layer_key] = layer.register_forward_hook(fwd_cb)
                self.hooks_layer_backward[layer_key] = layer.register_backward_hook(bwd_cb)

    def hook_parameters(self):
        # remove any previously set hook handlers
        _clear_hooks(self.hooks_params)
        # clear stored values
        self.param_gradients.clear()

        def create_parameter_callback(name):
            def hook_fn(grad):
                self.param_gradients[name] = grad.detach()

            return hook_fn

        for name, param in self.layers.named_parameters():
            self.hooks_params[name] = param.register_hook(create_parameter_callback(name))

    def get_gradients_for_sample(self, input, target_class):
        # Put model in evaluation mode
        self.layers.eval()
        model_output = self.__call__(input)

        self.zero_grad()
        one_hot_output = one_hot_tensor(num_classes=model_output.size()[-1], target_class=target_class)
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        return self.layer_gradients

    def forward(self, x):
        if self.hook_to_activations:
            _clear_hooks(self.hooks_activations)
            self.activation_gradients.clear()
            # here, 'x' has the input, so we can hook to it
            self.hooks_activations['input'] = x.register_hook(self.create_activation_callback('input'))

        for layer_key, layer in self.layers.items():
            x = layer(x)
            # if we enabled hooks to the outputs, add them now
            if self.hook_to_activations:
                self.hooks_activations[layer_key] = x.register_hook(self.create_activation_callback(layer_key))
        return x

    def get_modules(self, name: str) -> List[nn.Module]:
        return [layer for key, layer in self.layers.items() if key_to_tuple(key)[0] == name]

    def get_module(self, layer_key: str) -> nn.Module:
        return self.layers[layer_key]

    def get_layer_output(self, layer_key: str):
        return self.layer_outputs.get(layer_key)

    def insert_after(self, insertion_key: str, new_key: str, new_layer: nn.Module):
        layer_list = list(self.layers.items())
        self.layers = nn.ModuleDict(insert_layer_after(layer_list, insertion_key, new_key, new_layer))

    def delete_all_after(self, last_key: str):
        layer_list = list(self.layers.items())
        self.layers = nn.ModuleDict(delete_all_layers_after(layer_list, last_key))
