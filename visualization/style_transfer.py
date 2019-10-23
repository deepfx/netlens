import torch
import torch.nn.functional as F
from pydash import find_last
from torch import nn, optim

from .modules import LayeredModule
from .utils import gram_matrix, key_to_tuple, tuple_to_key


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


class StyleTransferModule(LayeredModule):
    content_target: torch.Tensor
    style_target: torch.Tensor

    def __init__(self, arch: LayeredModule,
                 content_target=None,
                 content_layer_keys=None,
                 style_target=None,
                 style_layer_keys=None):
        super(StyleTransferModule, self).__init__(arch.layers.items())
        self.content_target = content_target
        self.style_target = style_target
        if content_target is not None and content_layer_keys is not None:
            self._insert_loss_layers(ContentLoss, content_target, content_layer_keys)
        if style_target is not None and style_layer_keys is not None:
            self._insert_loss_layers(StyleLoss, style_target, style_layer_keys)

        # remove the layers after the last loss layer, which are useless
        last = find_last(self.layers.items(), lambda l: isinstance(l[1], (ContentLoss, StyleLoss)))
        self.delete_all_after(last[0])

    def _insert_loss_layers(self, layer_class, target, insertion_keys):
        # do a forward pass to get the layer outputs
        self.forward(target)
        for key in insertion_keys:
            # create loss layer
            loss_layer = layer_class(self.get_layer_output(key))
            # insert it after layer at key
            # we form the key of the new layer with the same 'nth' of the layer after which it was inserted
            _, nth = key_to_tuple(key)
            self.insert_after(key, tuple_to_key(layer_class.name, nth), loss_layer)

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
