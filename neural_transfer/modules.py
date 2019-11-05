import copy
from typing import Optional, List, Tuple, Iterable

import torch
import torch.nn.functional as F
from pydash.arrays import find_last_index
from torch import nn
from torch import optim

from .utils import gram_matrix, clean_layer, is_instance_of_any


class Normalization(nn.Module):
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

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class LayeredModule(nn.Module):
    preprocessor: Optional[nn.Module]
    layers: nn.ModuleList
    postprocessor: Optional[nn.Module]

    def __init__(self, layers, preprocessor=None, postprocessor=None):
        super(LayeredModule, self).__init__()
        self.preprocessor = preprocessor
        self.layers = layers
        self.postprocessor = postprocessor

    @staticmethod
    def from_cnn(cnn, normalizer):
        cnn = copy.deepcopy(cnn)
        layers = nn.ModuleList([clean_layer(layer) for layer in cnn.children()])
        return LayeredModule(layers, preprocessor=normalizer)

    # TODO: implement similar static methods for other archs

    def forward(self, x):
        for layer in ([self.preprocessor] + list(self.layers) + [self.postprocessor]):
            if layer is not None:
                x = layer(x)
        return x

    def find_indices(self, module: type) -> List[int]:
        return [i for i, l in enumerate(self.layers) if isinstance(l, module)]

    def get_modules(self, module: type) -> List[nn.Module]:
        return [layer for layer in self.layers if isinstance(layer, module)]

    def get_module(self, module: type, nth: int) -> nn.Module:
        return self.get_modules(module)[nth]

    def evaluate_at_layer(self, x: torch.Tensor, key: Tuple[type, int]):
        module, nth = key
        until_idx = self.find_indices(module)[nth]
        x = self.preprocessor(x)
        for layer in self.layers[:until_idx + 1]:
            x = layer(x)
        return x

    def evaluate_at_layers(self, x: torch.Tensor, keys: Iterable[Tuple[type, int]]):
        all_indices = [self.find_indices(m)[n] for m, n in keys]
        keys_indices = {idx: key for idx, key in zip(all_indices, keys)}

        values = {}
        x = self.preprocessor(x)
        for idx, layer in enumerate(self.layers[:max(all_indices) + 1]):
            x = layer(x)
            if idx in keys_indices:
                values[keys_indices[idx]] = x
        return values

    def insert_after(self, insertion_key: Tuple[type, int], layer: nn.Module):
        module, nth = insertion_key
        idx = self.find_indices(module)[nth]
        self.layers.insert(idx + 1, layer)


class StyleTransferModule(LayeredModule):
    content_target: torch.Tensor
    style_target: torch.Tensor

    def __init__(self, arch: LayeredModule,
                 content_target=None,
                 content_layer_keys=None,
                 style_target=None,
                 style_layer_keys=None):
        arch = copy.deepcopy(arch)
        super(StyleTransferModule, self).__init__(arch.layers, arch.preprocessor, None)
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

        style_losses = self.get_modules(StyleLoss)
        content_losses = self.get_modules(ContentLoss)

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
