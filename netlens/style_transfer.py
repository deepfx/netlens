from functools import partial

import torch
import torch.nn.functional as F
from pydash import find_last
from torch import nn, optim

from .math import gram_matrix
from .modules import FlatModel
from .utils import key_to_tuple, tuple_to_key
from .visualization.objective import Objective
from .visualization.param import RawParam
from .visualization.render import OptVis, OptVisCallback


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


class StyleTransferModule(FlatModel):
    content_target: torch.Tensor
    style_target: torch.Tensor

    def __init__(self, arch: FlatModel,
                 content_target=None,
                 content_layer_keys=None,
                 style_target=None,
                 style_layer_keys=None,
                 style_transform=gram_matrix,
                 loss_func=F.mse_loss):
        super(StyleTransferModule, self).__init__(arch.layers.items(), arch.arch_name, arch.flat_keys)
        self.content_target = content_target
        self.style_target = style_target

        if content_target is not None and content_layer_keys is not None:
            content_loss = partial(FeatureLoss, transform=None, loss_func=loss_func)
            self._insert_loss_layers('content_loss', content_loss, content_target, content_layer_keys)
        if style_target is not None and style_layer_keys is not None:
            style_loss = partial(FeatureLoss, transform=style_transform, loss_func=loss_func)
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

    def compute_losses(self, x: torch.Tensor, style_weight=1.0, content_weight=1.0, tv_weight=0.0):
        style_losses = self.get_modules('style_loss')
        content_losses = self.get_modules('content_loss')

        self.forward(x)

        # the conditions are to avoid unnecessary computation
        style_score = style_weight * sum(sl.loss for sl in style_losses) if style_weight != 0.0 else 0.0
        content_score = content_weight * sum(cl.loss for cl in content_losses) if content_weight != 0.0 else 0.0
        tv_score = tv_weight * total_variation_loss(x) if tv_weight != 0.0 else 0.0

        return style_score + content_score + tv_score, style_score, content_score, tv_score


class StyleTransferObjective(Objective):
    module: StyleTransferModule

    def __init__(self, module: StyleTransferModule, style_weight=1.0, content_weight=1.0, tv_weight=0):
        super(StyleTransferObjective, self).__init__('style_transfer_obj')
        self.module = module
        self.style_weight, self.content_weight, self.tv_weight = style_weight, content_weight, tv_weight
        # these just keep the last computed losses
        self.style_loss, self.content_loss, self.tv_loss = None, None, None

    # Objective.__call__
    def objective_function(self, x):
        total_loss, self.style_loss, self.content_loss, self.tv_loss = \
            self.module.compute_losses(x, self.style_weight, self.content_weight, self.tv_weight)
        return total_loss


class STCallback(OptVisCallback):
    """
    A callback class specific for style transfer
    """

    def on_step_begin(self, optvis, img, *args, **kwargs):
        img.data.clamp_(0.0, 1.0)

    def on_step_end(self, optvis, img, *args, **kwargs):
        if optvis.is_step_to_show():
            print(f'Style loss={optvis.objective.style_loss}, Content loss={optvis.objective.content_loss}, '
                  f'TV loss={optvis.objective.tv_loss}')

    def on_render_end(self, optvis, img, *args, **kwargs):
        img.data.clamp_(0.0, 1.0)


# TODO: lookup table for weights (we know optim and model at that point)
def generate_style_transfer(module: StyleTransferModule, input_img, num_steps=300, style_weight=1, content_weight=1, tv_weight=0, **kwargs):
    # create objective from the module and the weights
    objective = StyleTransferObjective(module, style_weight, content_weight, tv_weight)
    # the "parameterized" image is the image itself
    param_img = RawParam(input_img, cloned=True)
    render = OptVis(module, objective, optim=optim.LBFGS)
    thresh = (num_steps,) if isinstance(num_steps, int) else num_steps
    return render.vis(param_img, thresh, in_closure=True, callback=STCallback(), **kwargs)
