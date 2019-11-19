from torch import nn

from pyimgy.optional.torch import *
from visualization.transforms import VIS_TFMS
from .objective import Objective


class OptVisCallback:

    def on_render_begin(self, optvis, *args, **kwargs):
        pass

    def on_step_begin(self, optvis, img, *args, **kwargs):
        pass

    def on_step_end(self, optvis, img, *args, **kwargs):
        pass

    def on_render_end(self, optvis, img, *args, **kwargs):
        pass


class OptVis:
    """
    A class that encapsulates the generation of an image from a given parameterized input, objective function and optimizer.
    """

    def __init__(self, model: nn.Module, objective: Objective, tfms=VIS_TFMS, optim=torch.optim.Adam, optim_params=None, show_step: int = 50):
        """
        :param model:
        :param objective:
        :param tfms:
        :param optim:
        :param optim_params: a dictionary with the parameters to pass to the optimizer, e.g. learning rate, weight decay
        """
        self.model = model
        self.objective = objective
        self.tfms = tfms
        self.optim_fn = optim
        self.optim_params = optim_params or {}
        self.show_step = show_step

        self.model.eval()

        # state variables
        self.optim = None
        self.run = 0

    def is_step_to_show(self):
        return self.run % self.show_step == 0

    def vis(self, img_param: nn.Module, thresh: Tuple[int, ...] = (100,), callback: OptVisCallback = None, transform: bool = True,
            in_closure: bool = False, verbose: bool = True, show: bool = False):
        """
        Generates the image

        :param img_param: a parameterized image, represented by a nn.Module where the only parameter is the image parameter
        :param thresh: the iterations count for which intermediate results will be shown if display=True; the max is the total number of iterations
        :param callback: an optional OptVisCallback to add custom behaviours.
        :param transform: if true, every image will be transformed by the provided chain.
        :param in_closure: if true, the optimizer.step() will be passed a closure with the actual loss computation, otherwise the loss computation
                           will be done outside. Default is false.
        :param verbose: if true, some output will be printed during the iterations.
        :param show: if true, intermediate results will be shown according to the values of thresh (only in a notebook).
        :return:
        """

        if show:
            try:
                from IPython.display import display
            except ImportError:
                raise ValueError("Can't use 'show' if not in IPython notebook.")

        self.optim = self.optim_fn(img_param.parameters(), **self.optim_params)
        self.run = 0

        if callback:
            callback.on_render_begin(self, img_param)

        while self.run <= max(thresh):
            # img_param is a "source" of images, a call to forward() just returns one
            img = img_param()

            if transform and self.tfms is not None:
                img = self.tfms(img)

            if callback:
                callback.on_step_begin(self, img)

            def closure():
                self.optim.zero_grad()

                _loss = self.objective(img)
                _loss.backward(retain_graph=in_closure)

                self.run += 1
                if verbose and self.is_step_to_show():
                    print(f'Run [{self.run}], loss={_loss.item():.4f}')
                if show and self.run in thresh:
                    display(convert_to_standard_pil(img))

                return _loss

            if in_closure:
                self.optim.step(closure)
            else:
                # "normal" case, without closure
                closure()
                self.optim.step()

            if callback:
                callback.on_step_end(self, img)

        # get the last image, for final tasks and returning it
        img = img_param()

        if callback:
            callback.on_render_end(self, img)

        return img
