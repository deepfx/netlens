from functools import partial
from typing import Mapping

from torch import Tensor

from pyimgy.optional.torch_utils import *
from visualization.image_proc import *
from visualization.math import one_hot_tensor
from visualization.modules import LayeredModule
from visualization.utils import get_name_from_key


class NetLens:

    def __init__(self, model: LayeredModule, input_image: Tensor, target_class: int = None, denormalize: bool = True):
        self.original_model, self.input_image, self.target_class = model, input_image, target_class
        # partial for the reconstruction of the original image from the input
        self.original_image = partial(recreate_image, self.input_image, denormalize)
        self.model = None

    def _prepare_model(self, guided: bool = False):
        self.model = self.original_model.copy()
        self.model.hook_to_activations = True

        if guided:
            # We have to make sure that the outputs of the relus are stored
            relu_keys = set(key for key in self.model.layers.keys() if get_name_from_key(key) == 'relu')
            self.model.set_hooked_layers(relu_keys)
            self.model.set_hooked_activations(relu_keys)

            # This is a bit tricky... we have to give the layered module a "factory", i.e. a function that creates hook functions;
            # it is a closure that has access to the whole context of the layered module (e.g. for seeing the stored layer outputs)
            def guided_relus_hook_factory(module: LayeredModule, key: str):
                if get_name_from_key(key) == 'relu':

                    def _hook_func(grad):
                        relu_output = module.hooks_layers[key].stored
                        return grad.clamp(min=0.0) * (relu_output > 0).float()

                    return _hook_func
                else:
                    return None

            self.model.custom_activation_hook_factory = guided_relus_hook_factory

        # Put model in evaluation mode
        self.model.eval()

    def _process_input(self, model_input: Tensor = None) -> Tensor:
        if model_input is None:
            model_input = self.input_image
        model_output = self.model(model_input)
        num_classes = model_output.size()[-1]
        target_class = self.target_class or torch.argmax(model_output).item()
        one_hot_output = one_hot_tensor(num_classes, target_class, device=model_output.device)
        # Backward pass
        self.model.zero_grad()
        model_output.backward(gradient=one_hot_output)
        return model_output

    def get_input_gradient(self, guided: bool = False) -> Mapping[str, Tensor]:
        self._prepare_model(guided)
        self._process_input()
        return self.model.get_activation_gradient('input')

    def _show_gradient_images(self, grads, name: str):
        grads = convert_image(grads, to_type=np.ndarray, shape='CWH')
        grads_color = normalize_to_range(grads)
        grads_gray = convert_to_grayscale(grads)
        show_images([self.original_image(), grads_color, grads_gray], ['Original image', f'{name} Color', f'{name} Grayscale'])

    def show_gradient_backprop(self, guided: bool = False):
        # Generate gradients
        input_grads = self.get_input_gradient(guided=guided)

        self._show_gradient_images(input_grads, 'Guided Backprop' if guided else 'Backprop')
        if guided:
            pos_sal, neg_sal = get_positive_negative_saliency(input_grads)
            show_images([pos_sal, neg_sal], ['GBP Positive Saliency', 'GBP Negative Saliency'])

    def generate_smooth_gradient(self, guided: bool = False, param_n: int = 50, param_sigma_multiplier: float = 4.0) -> Tensor:
        """
        Generates smooth gradients of the given type. You can use this with both vanilla and guided backprop

        :param guided: boolean, if false it's vanilla
        :param param_n: number of images used to smooth the gradient
        :param param_sigma_multiplier: sigma multiplier when calculating std of noise
        :return:
        """
        # Prepare model only once
        self._prepare_model(guided)

        # Generate an empty image/matrix
        smooth_grad = torch.zeros_like(self.input_image)

        mean = 0.0
        sigma = param_sigma_multiplier / (torch.max(self.input_image) - torch.min(self.input_image)).item()
        for _ in range(param_n):
            # Generate noise
            noise = torch.randn_like(self.input_image) * sigma + mean
            # Add noise to the image
            noisy_img = self.input_image + noise
            # Calculate gradients
            self._process_input(noisy_img)
            input_grads = self.model.get_activation_gradient('input')
            # Add gradients to smooth_grad
            smooth_grad += input_grads
        # Average it out
        smooth_grad = smooth_grad / param_n
        return smooth_grad

    def show_smooth_gradient(self, *args, **kwargs):
        self._show_gradient_images(self.generate_smooth_gradient(*args, **kwargs), 'Smooth Backprop')

    def generate_integrated_gradient(self, guided: bool = False, steps: int = 100) -> Tensor:
        # Prepare model only once
        self._prepare_model(guided)

        # Generate an empty image/matrix
        integrated_grads = torch.zeros_like(self.input_image)

        for step in np.linspace(0, 1, steps):
            xbar_image = self.input_image * step
            self._process_input(xbar_image)
            input_grads = self.model.get_activation_gradient('input')
            integrated_grads += input_grads / steps

        return integrated_grads

    def show_integrated_gradient(self, *args, **kwargs):
        self._show_gradient_images(self.generate_integrated_gradient(*args, **kwargs), 'Integrated Gradients')

    def generate_cam(self, target_layer: str, interpolate: bool = True) -> Tensor:
        self._prepare_model()
        self.model.set_hooked_layers(target_layer)
        self.model.set_hooked_activations(target_layer)

        # Full forward and backward pass
        self._process_input()

        # conv_output is the output of convolutions at specified layer
        conv_output = self.model.get_layer_output(target_layer)
        # Get hooked gradients
        guided_gradients = self.model.get_activation_gradient(target_layer).squeeze(0)
        # Get convolution outputs
        target = conv_output.squeeze(0)

        # Get weights from gradients
        weights = torch.mean(guided_gradients, dim=(1, 2))  # Take averages for each gradient
        # Multiply each weight with its conv output and then, sum
        cam = (target * weights[..., None, None]).sum(dim=0)
        cam.clamp_min_(0.0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize between 0-1
        height, width = self.input_image.shape[2:]
        return resize_as_torch(cam, width, height, mode='bilinear' if interpolate else 'nearest')

    def show_gradcam(self, target_layer: str, interpolate: bool = True):
        cam = self.generate_cam(target_layer, interpolate)
        original_image = self.original_image(to_pil=True)
        heatmap, heatmap_on_image = apply_colormap_on_image(original_image, convert_image(cam, to_type=np.ndarray), 'hsv')
        show_images([heatmap, heatmap_on_image, cam], [f'CAM Heatmap for {target_layer}', 'CAM Heatmap on image', 'CAM Grayscale'])

    def generate_guided_gradcam(self, *args, **kwargs) -> Tensor:
        """
        Guided grad cam is just the point-wise multiplication of cam mask and guided backprop mask
        """
        cam = self.generate_cam(*args, **kwargs)
        guided_backprop_grads = self.get_input_gradient(guided=True)
        return cam * guided_backprop_grads

    def show_guided_gradcam(self, *args, **kwargs):
        self._show_gradient_images(self.generate_guided_gradcam(*args, **kwargs), 'Guided GradCAM')

    def get_input_gradient_for_layer_activation(self, target_layer: str, target_channel: int, guided: bool = False) -> Tensor:
        self._prepare_model(guided)

        layer_output = self.model.forward(self.input_image, until_layer=target_layer).squeeze(0)
        layer_ch_output = torch.sum(torch.abs(layer_output[target_channel, :, :]))
        layer_ch_output.backward()

        return self.model.get_activation_gradient('input')

    def show_input_gradient_for_layer_activation(self, target_layer: str, target_channel: int, guided: bool = False):
        self._show_gradient_images(self.get_input_gradient_for_layer_activation(target_layer, target_channel, guided),
                                   ('Guided' if guided else 'Vanilla') + f' gradient for ({target_layer}, {target_channel})')
