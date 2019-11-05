import copy
from collections import defaultdict
from typing import List, Tuple, Iterable, Optional, Any

import torch
import torch.nn.functional as F
from pydash import find_last
from torch import nn
from torch import optim
from torchvision import models

if __name__ == '__main__':
    # dirty hack for this to work when running directly
    from utils import gram_matrix, clean_layer
else:
    from .utils import gram_matrix, clean_layer

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


# FUNCTIONS TO ENCODE/DECODE LAYER KEYS (tuples) TO INTERNAL KEYS
# Example: key=('conv', 1) <--> int.key='conv-1'

def key_to_internal(name: str, nth: int) -> str:
    assert '-' not in name, "The name cannot contain a '-' character."
    return f'{name}-{nth}'


def internal_to_key(ikey: str) -> Tuple[str, int]:
    parts = ikey.rsplit('-', 1)
    return parts[0], int(parts[1])


def generate_module_keys(modules: Iterable[nn.Module], internal=False) -> Iterable[Tuple[Any, nn.Module]]:
    name_counts = defaultdict(int)

    def gen_key(m):
        name = get_module_name(m)
        nth = name_counts[name]
        name_counts[name] += 1
        return key_to_internal(name, nth) if internal else (name, nth)

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


class ContentLoss(nn.Module):
    name: str = 'content_loss'

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    name: str = 'style_loss'

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


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
        return LayeredModule(generate_module_keys([normalizer] + [clean_layer(layer) for layer in cnn.children()], internal=True))

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

        for layer_ikey, layer in self.layers.items():
            if self.is_hooked_layer(internal_to_key(layer_ikey)):
                self.hooks_forward[layer_ikey] = layer.register_forward_hook(self.set_layer_context(layer_ikey))

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
        return [layer for ikey, layer in self.layers.items() if internal_to_key(ikey)[0] == name]

    def get_module(self, name: str, nth: int) -> nn.Module:
        return self.layers[key_to_internal(name, nth)]

    def get_layer_output(self, layer_key):
        return self.layer_outputs.get(key_to_internal(*layer_key))

    def insert_after(self, insertion_key: Tuple[str, int], new_key: Tuple[str, int], new_layer: nn.Module):
        insertion_ikey = key_to_internal(*insertion_key)
        layers = []
        for ikey, layer in self.layers.items():
            layers.append((ikey, layer))
            if ikey == insertion_ikey:
                layers.append((key_to_internal(*new_key), new_layer))
        self.layers = nn.ModuleDict(layers)

    def delete_all_after(self, last_key: Tuple[str, int]):
        last_ikey = key_to_internal(*last_key)
        layers = []
        for ikey, layer in self.layers.items():
            layers.append((ikey, layer))
            if ikey == last_ikey:
                break
        self.layers = nn.ModuleDict(layers)


class StyleTransferModule(LayeredModule):
    content_target: torch.Tensor
    style_target: torch.Tensor

    def __init__(self, arch: LayeredModule,
                 content_target=None,
                 content_layer_keys=None,
                 style_target=None,
                 style_layer_keys=None):
        arch = copy.deepcopy(arch)
        super(StyleTransferModule, self).__init__(arch.layers.items())
        self.content_target = content_target
        self.style_target = style_target
        if content_target is not None and content_layer_keys is not None:
            self._insert_loss_layers(ContentLoss, content_target, content_layer_keys)
        if style_target is not None and style_layer_keys is not None:
            self._insert_loss_layers(StyleLoss, style_target, style_layer_keys)

        # remove the layers after the last loss layer, which are useless
        last = find_last(self.layers.items(), lambda l: isinstance(l[1], (ContentLoss, StyleLoss)))
        self.delete_all_after(internal_to_key(last[0]))

    def _insert_loss_layers(self, layer_class, target, insertion_keys):
        # do a forward pass to get the layer outputs
        self.forward(target)
        for key in insertion_keys:
            # create loss layer
            loss_layer = layer_class(self.get_layer_output(key).detach())
            # insert it after layer at key
            # we form the key of the new layer with the same 'nth' of the layer after which it was inserted
            self.insert_after(key, (layer_class.name, key[1]), loss_layer)

    def run_style_transfer(self, input_img, optimizer_class=optim.LBFGS, num_steps=300, style_weight=1000000, content_weight=1, verbose=True):
        optimizer = optimizer_class([input_img.requires_grad_()])

        style_losses = self.get_modules(StyleLoss.name)
        content_losses = self.get_modules(ContentLoss.name)

        print("Optimizing...")
        run = [0]
        while run[0] <= num_steps:
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                self.forward(input_img)
                style_score = style_weight * sum(sl.loss for sl in style_losses)
                content_score = content_weight * sum(cl.loss for cl in content_losses)
                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if verbose and run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}\n'.format(style_score.item(), content_score.item()))
                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img


if __name__ == '__main__':
    # just a test for debugging
    cnn = models.vgg19(pretrained=True).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    arch = LayeredModule.from_cnn(cnn, Normalization(cnn_normalization_mean, cnn_normalization_std))
    style_injects = [('conv', i) for i in range(5)]
    content_injects = [('conv', 3)]
    content_img = torch.zeros((1, 3, 128, 128))
    style_img = torch.zeros((1, 3, 128, 128))
    style_module = StyleTransferModule(arch, content_img, content_injects, style_img, style_injects)
