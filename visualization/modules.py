import copy
from typing import List, Tuple, Iterable

import torch
import torch.nn.functional as F
from pydash.arrays import find_last_index
from torch import nn
from torch import optim

from .utils import gram_matrix, clean_layer, is_instance_of_any

# GENERIC NAMES FOR DIFFERENT LAYERS

#externally exposed – to smoothen out pytorch changes
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


#implement gradient getter logic


"""
hook_layer(layer_key) -> {
    self.hooks[layer_key] = h
    hook
    
    def hook_closure(
}
"""

"""
*Register forward for all layers
*save hooks to dict
*save outputs to dict 
"""

class Show: #interpret
    @staticmethod
    def gradient(gradients):
        #convert to greyscale
        #plt.plot()
        pass
    @staticmethod
    def top_loss():
        pass
    @staticmethod
    def GradCAM(gradients):
        pass

class LayeredModule(nn.Module):
    layers: nn.ModuleList

    def __init__(self, layers):
        super(LayeredModule, self).__init__()
        self.layers = layers
        self.hooks_forward = {}
        self.hooks_backward = {}

        self.layer_outputs = {}
        self.layer_gradients = {}

    def set_layer_context(self, layer_key):
        def forward_hook_callback(self, _input_, output):
            self.layer_outputs[layer_key] = output
        return forward_hook_callback

    def hook_layers(self, layer_keys):
        return None

    #TODO: hook_all -> hook_specified
    def hook_all_forward(self):
        for layer_key, layer in self.layers._modules.items():
            self.hooks_forward[layer_key] = layer.register_forward_hook(self.set_layer_context(layer_key))

    #TODO: all -> specified
    def hook_all_backward(self):
        def set_param_context(name):
            def hook_fn(grad):
                self.layer_gradients[name] = grad
                return grad
            return hook_fn

        for name, param in self.layers.named_parameters():
            param.register_hook(set_param_context(name))

    def get_gradients(self, input, target_class):
        self.gradients = None
        # Put model in evaluation mode
        self.eval()
        model_output = self.forward(input)

        self.zero_grad()
        num_classes = model_output.size()[-1]
        one_hot_output = torch.FloatTensor(1 ,num_classes).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass

        #wrt
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)

        return self.layer_gradients #to numpy?? #numpy()[0]

    @staticmethod
    def from_cnn(cnn, normalizer):
        cnn = copy.deepcopy(cnn)
        layers = nn.ModuleList([normalizer] + [clean_layer(layer) for layer in cnn.children()])
        return LayeredModule(layers)

    # TODO: implement similar static methods for other archs

    def forward(self, x):
        for layer in self.layers:
            if layer is not None:
                x = layer(x)
        return x

    def find_indices(self, name: str) -> List[int]:
        return [i for i, layer in enumerate(self.layers) if get_module_name(layer) == name]

    def get_modules(self, name: str) -> List[nn.Module]:
        return [layer for layer in self.layers if get_module_name(layer) == name]

    def get_module(self, name: str, nth: int) -> nn.Module:
        return self.get_modules(name)[nth]

    # TODO: remove and integrate with forward hooks
    # def get_output(layer):
    def evaluate_at_layer(self, x: torch.Tensor, key: Tuple[str, int]):
        name, nth = key
        until_idx = self.find_indices(name)[nth]
        for layer in self.layers[:until_idx + 1]:
            x = layer(x)
        return x


    def evaluate_at_layers(self, x: torch.Tensor, keys: Iterable[Tuple[str, int]], compute_all: bool = False):
        """
        Allows to collect the activation values (outputs of some network layers) for a given input. Similar to :func:`evaluate_at_layer`
        but more efficient because it requires only one pass thru the network.
        :param x: the input tensor that will be fed to the network.
        :param keys: a list of "keys" ((module name, module nr) tuples) from which we want to fetch the activations, for the given input.
        :param compute_all: if True, all the layers will be computed, even if not requested. Default=False.
        :return: a map where the keys are the provided ones, and the values are the collected activation maps (tensors).
        """
        all_indices = [self.find_indices(m)[n] for m, n in keys]
        keys_indices = {idx: key for idx, key in zip(all_indices, keys)}

        values = {}
        layers = self.layers if compute_all else self.layers[:max(all_indices) + 1]
        for idx, layer in enumerate(layers):
            x = layer(x)
            if idx in keys_indices:
                values[keys_indices[idx]] = x
        return values

    def insert_after(self, insertion_key: Tuple[str, int], layer: nn.Module):
        name, nth = insertion_key
        idx = self.find_indices(name)[nth]
        self.layers.insert(idx + 1, layer)


class   StyleTransferModule(LayeredModule):
    content_target: torch.Tensor
    style_target: torch.Tensor

    def __init__(self, arch: LayeredModule,
                 content_target=None,
                 content_layer_keys=None,
                 style_target=None,
                 style_layer_keys=None):
        arch = copy.deepcopy(arch)
        super(StyleTransferModule, self).__init__(arch.layers)
        self.content_target = content_target
        self.style_target = style_target
        if content_target is not None and content_layer_keys is not None:
            self._insert_loss_layers(ContentLoss, content_target, content_layer_keys)
        if style_target is not None and style_layer_keys is not None:
            self._insert_loss_layers(StyleLoss, style_target, style_layer_keys)

        # remove the layers after the last loss layer, which are useless
        last = find_last_index(self.layers, is_instance_of_any([ContentLoss, StyleLoss]))
        self.layers = self.layers[:last + 1]

    def _insert_loss_layers(self, layer_constructor, target, insertion_keys):
        target_at_layers = self.evaluate_at_layers(target, insertion_keys)
        for key in insertion_keys:
            # create loss layer
            loss_layer = layer_constructor(target_at_layers[key].detach())
            # you could also just do (but it's kinda inefficient :))
            # loss_layer = layer_constructor(self.evaluate_at_layer(target, key).detach())
            # insert it after layer at key
            self.insert_after(key, loss_layer)

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
