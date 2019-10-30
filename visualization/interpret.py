from torch import Tensor

from visualization.image_proc import *
from visualization.math import one_hot_tensor
from visualization.modules import LayeredModule
from visualization.utils import get_name_from_key


def get_gradients_for_sample(model: LayeredModule, input_image: Tensor, target_class: int, guided: bool = False):
    model.hook_to_activations = True

    if guided:
        # This is a bit tricky... we have to give the layered module a "factory", i.e. a function that creates hook functions; it is a closure that
        # has access to the whole context of the layered module (e.g. for seeing the stored layer outputs)
        def guided_relus_hook_factory(module: LayeredModule, key: str):
            if get_name_from_key(key) == 'relu':

                def _hook_func(grad):
                    relu_output = module.hooks_layers[key].stored
                    return grad.clamp(min=0.0) * (relu_output > 0).float()

                return _hook_func
            else:
                return None

        model.custom_activation_hook_factory = guided_relus_hook_factory

    # Put model in evaluation mode
    model.eval()
    model_output = model(input_image)

    model.zero_grad()
    one_hot_output = one_hot_tensor(num_classes=model_output.size()[-1], target_class=target_class)
    # Backward pass
    model_output.backward(gradient=one_hot_output)
    return model.hooks_activations.stored


def show_gradient_backprop(model: LayeredModule, input_image: Tensor, target_class: int, guided: bool = False):
    # Generate gradients
    gradients = get_gradients_for_sample(model, input_image, target_class, guided)
    # we need to get the first layer, squeeze it an make it numpy
    input_grads = gradients['input'].squeeze(0).numpy()
    input_grads_color = normalize_to_range(input_grads)
    input_grads_gray = convert_to_grayscale(input_grads)

    images = [recreate_image(input_image), input_grads_color, input_grads_gray]
    titles = ['Original image', 'Backprop Color', 'Backprop Grayscale']

    if guided:
        pos_sal, neg_sal = get_positive_negative_saliency(input_grads)
        images += [pos_sal, neg_sal]
        titles += ['GBP Positive Saliency', 'GBP Negative Saliency']

    show_images(images, titles)


def generate_smooth_gradient(model: LayeredModule, input_image: Tensor, target_class: int, guided: bool = False,
                             param_n: int = 50, param_sigma_multiplier: float = 4.0):
    """
    Generates smooth gradients of the given type. You can use this with both vanillaand guided backprop

    :param model:
    :param input_image:
    :param target_class:
    :param guided: boolean, if false it's vanilla
    :param param_n: number of images used to smooth the gradient
    :param param_sigma_multiplier: sigma multiplier when calculating std of noise
    :return:
    """
    # Generate an empty image/matrix
    smooth_grad = torch.zeros_like(input_image)

    mean = 0.0
    sigma = param_sigma_multiplier / (torch.max(input_image) - torch.min(input_image)).item()
    for _ in range(param_n):
        # Generate noise
        noise = torch.randn_like(input_image) * sigma + mean
        # Add noise to the image
        noisy_img = input_image + noise
        # Calculate gradients
        grads = get_gradients_for_sample(model, noisy_img, target_class, guided)
        # Add gradients to smooth_grad
        smooth_grad = smooth_grad + grads['input']
    # Average it out
    smooth_grad = smooth_grad / param_n
    return smooth_grad


def generate_cam(model: LayeredModule, target_layer: str, input_image: Tensor, target_class: int = None, interpolate: bool = True):
    model.hook_to_activations = True
    model.eval()
    # Full forward pass
    # conv_output is the output of convolutions at specified layer
    # model_output is the final output of the model
    model_output = model(input_image)
    conv_output = model.hooks_layers.get_stored(target_layer)
    target_class = target_class or torch.argmax(model_output).item()
    # Target for backprop
    one_hot_output = one_hot_tensor(num_classes=model_output.size()[-1], target_class=target_class)
    # Zero grads
    model.zero_grad()
    # Backward pass with specified target
    model_output.backward(gradient=one_hot_output)
    # Get hooked gradients
    guided_gradients = model.hooks_activations.get_stored(target_layer).squeeze(0).numpy()
    # Get convolution outputs
    target = conv_output.detach().squeeze(0).numpy()

    # Get weights from gradients
    weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
    # Create empty numpy array for cam
    # Multiply each weight with its conv output and then, sum
    cam = (target * weights[..., None, None]).sum(axis=0)
    cam = np.maximum(cam, 0)
    cam = normalize_to_range(cam)  # Normalize between 0-1
    height, width = input_image.shape[2:]
    return resize_image(cam, width, height, resample=PIL.Image.ANTIALIAS if interpolate else PIL.Image.NEAREST)


def show_gradcam(*args, **kwargs):
    input_image = args[2]  # TODO: this is kinda ugly, perhaps there's a better way?
    cam = generate_cam(*args, **kwargs)
    heatmap, heatmap_on_image = apply_colormap_on_image(recreate_image(input_image, to_pil=True), cam, 'hsv')
    show_images([heatmap, heatmap_on_image, cam], ['CAM Heatmap', 'CAM Heatmap on image', 'CAM Grayscale'])
