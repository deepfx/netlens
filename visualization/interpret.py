from visualization.image_proc import *
from visualization.math import one_hot_tensor
from visualization.modules import LayeredModule


def show_vanilla_backprop(model: LayeredModule, input_image: torch.Tensor, target_class: int, name: str = 'Original image'):
    model.hook_to_activations = True
    # Generate gradients
    gradients = model.get_gradients_for_sample(input_image, target_class)
    # we need to get the first layer, squeeze it an make it numpy
    vanilla_grads = gradients['input'].squeeze(0).numpy()
    vanilla_grads_color = normalize_to_range(vanilla_grads)
    vanilla_grads_gray = convert_to_grayscale(vanilla_grads)
    show_images([recreate_image(input_image), vanilla_grads_color, vanilla_grads_gray], [name, 'Backprop Color', 'Backprop Grayscale'])


def generate_cam(model: LayeredModule, target_layer: str, input_image: torch.Tensor, target_class: int = None, interpolate: bool = True):
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
