from functools import partial

import torch
import torch.nn.functional as F
from pydash import find_last
from torch import nn, optim

from visualization.interp.objective import Objective
from .math import gram_matrix
from .modules import LayeredModule
from .utils import key_to_tuple, tuple_to_key


class FeatureLoss(nn.Module):

    def __init__(self, target: torch.Tensor, transform=None, loss_func=F.mse_loss):
        super(FeatureLoss, self).__init__()
        self.transform = transform
        self.loss_func = loss_func
        self.target = self._transformed(target).detach()
        self.loss = None

    def _transformed(self, t):
        return t if self.transform is None else self.transform(t)

    def forward(self, input):
        self.loss = self.loss_func(self._transformed(input), self.target)
        return input


def total_variation_loss(input):
    x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
    y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
    return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))


class StyleTransferModule(LayeredModule):
    content_target: torch.Tensor
    style_target: torch.Tensor

    def __init__(self, arch: LayeredModule,
                 content_target=None,
                 content_layer_keys=None,
                 style_target=None,
                 style_layer_keys=None,
                 loss_func=F.mse_loss):
        super(StyleTransferModule, self).__init__(arch.layers.items(), arch.arch_name, arch.flat_keys)
        self.content_target = content_target
        self.style_target = style_target

        if content_target is not None and content_layer_keys is not None:
            content_loss = partial(FeatureLoss, transform=None, loss_func=loss_func)
            self._insert_loss_layers('content_loss', content_loss, content_target, content_layer_keys)
        if style_target is not None and style_layer_keys is not None:
            style_loss = partial(FeatureLoss, transform=gram_matrix, loss_func=loss_func)
            self._insert_loss_layers('style_loss', style_loss, style_target, style_layer_keys)

        # remove the layers after the last loss layer, which are useless
        last = find_last(self.layers.items(), lambda l: isinstance(l[1], FeatureLoss))
        self.delete_all_from_key(last[0])

    def _insert_loss_layers(self, name, layer_constructor, target, insertion_keys):
        self.set_hooked_layers(insertion_keys)
        # do a forward pass to get the layer outputs
        self.forward(target)
        for i, key in enumerate(insertion_keys):
            # create loss layer
            loss_layer = layer_constructor(self.hooks_layers.get_stored(key))
            # insert it after layer at key
            if self.flat_keys:
                nth = i
            else:
                # we form the key of the new layer with the same 'nth' of the layer after which it was inserted
                _, nth = key_to_tuple(key)
            self.insert_at_key(key, tuple_to_key(name, nth), loss_layer)
        # we don't need to hook to the layers anymore
        self.set_hooked_layers(None, keep=False)

    def run_style_transfer(self, input_img, optimizer_class=optim.LBFGS, num_steps=100, style_weight=1, content_weight=1, tv_weight=0,
                           callback=None, in_place=False, verbose=True):

        if not in_place:
            input_img = input_img.clone().detach().requires_grad_()

        optimizer = optimizer_class([input_img])

        style_losses = self.get_modules('style_loss')
        content_losses = self.get_modules('content_loss')

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
                tv_score = tv_weight * total_variation_loss(input_img)
                loss = style_score + content_score + tv_score
                loss.backward()

                if callback:
                    callback(run[0], input_img, style_score.item(), content_score.item())

                run[0] += 1
                if verbose and run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}\n'.format(style_score.item(), content_score.item()))
                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img


class StyleTransferObjective(Objective):
    module: LayeredModule
    content_target: torch.Tensor
    style_target: torch.Tensor
    name = 'style_transfer_obj'

    def __init__(self, arch: LayeredModule,
                 content_target=None,
                 content_layer_keys=None,
                 style_target=None,
                 style_layer_keys=None,
                 style_weight=1, content_weight=1, tv_weight=0,
                 loss_func=F.mse_loss):
        self.module = LayeredModule(arch.layers.items(), arch.arch_name, arch.flat_keys)
        self.content_target = content_target
        self.style_target = style_target
        self.style_weight, self.content_weight, self.tv_weight = style_weight, content_weight, tv_weight

        if content_target is not None and content_layer_keys is not None:
            content_loss = partial(FeatureLoss, transform=None, loss_func=loss_func)
            self._insert_loss_layers('content_loss', content_loss, content_target, content_layer_keys)
        if style_target is not None and style_layer_keys is not None:
            style_loss = partial(FeatureLoss, transform=gram_matrix, loss_func=loss_func)
            self._insert_loss_layers('style_loss', style_loss, style_target, style_layer_keys)

        # remove the layers after the last loss layer, which are useless
        last = find_last(self.module.layers.items(), lambda l: isinstance(l[1], FeatureLoss))
        self.module.delete_all_from_key(last[0])

    def _insert_loss_layers(self, name, layer_constructor, target, insertion_keys):
        self.module.set_hooked_layers(insertion_keys)
        # do a forward pass to get the layer outputs
        self.module.forward(target)
        for i, key in enumerate(insertion_keys):
            # create loss layer
            loss_layer = layer_constructor(self.module.hooks_layers.get_stored(key))
            # insert it after layer at key
            if self.module.flat_keys:
                nth = i
            else:
                # we form the key of the new layer with the same 'nth' of the layer after which it was inserted
                _, nth = key_to_tuple(key)
            self.module.insert_at_key(key, tuple_to_key(name, nth), loss_layer)
        # we don't need to hook to the layers anymore
        self.module.set_hooked_layers(None, keep=False)

    def objective_function(self, x):
        style_losses = self.module.get_modules('style_loss')
        content_losses = self.module.get_modules('content_loss')

        self.module.forward(x)

        style_score = self.style_weight * sum(sl.loss for sl in style_losses) if self.style_weight != 0.0 else 0.0
        content_score = self.content_weight * sum(cl.loss for cl in content_losses) if self.content_weight != 0.0 else 0.0
        tv_score = self.tv_weight * total_variation_loss(x) if self.tv_weight != 0.0 else 0.0

        return style_score + content_score + tv_score
