from typing import Tuple, Optional

import numpy as np
import torch

from .modules import LayeredModule


##Refactor to named_params backprop
##

class VanillaBackprop:
    """
    Produces gradients generated with vanilla back propagation from the image.
    """

    def __init__(self, model: LayeredModule, target_layer: Optional[Tuple[str, int]] = None):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers(target_layer)

    def hook_layers(self, target_layer):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the given layer
        hooked_layer = self.model.get_module(*target_layer) if target_layer is not None else self.model.layers[0]
        hooked_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image: torch.Tensor, target_class: int) -> np.ndarray:
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        num_classes = model_output.size()[-1]
        one_hot_output = torch.FloatTensor(1, num_classes).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first dimension (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr
