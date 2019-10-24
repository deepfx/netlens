import numpy as np
import torch
from PIL import Image

from old_visual.utils.misc_functions import get_example_params, save_class_activation_images
from visualization.math import one_hot_tensor
from visualization.modules import LayeredModule


def generate_cam(model: LayeredModule, target_layer: str, input_image, target_class=None):
    model.eval()
    # Full forward pass
    # conv_output is the output of convolutions at specified layer
    # model_output is the final output of the model (1, 1000)
    model_output = model(input_image)
    conv_output = model.layer_outputs[target_layer]
    target_class = target_class or torch.argmax(model_output).item()
    # Target for backprop
    one_hot_output = one_hot_tensor(num_classes=model_output.size()[-1], target_class=target_class)
    # Zero grads
    model.zero_grad()
    # Backward pass with specified target
    model_output.backward(gradient=one_hot_output)
    # Get hooked gradients
    guided_gradients = model.activation_gradients[target_layer].numpy()[0]
    # Get convolution outputs
    target = conv_output.detach().numpy()[0]





    # Get weights from gradients
    weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
    # Create empty numpy array for cam
    # Multiply each weight with its conv output and then, sum
    cam = (target * weights[..., None, None]).sum(axis=0)
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                input_image.shape[3]), Image.ANTIALIAS)) / 255
    # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
    # supports resizing numpy matrices with antialiasing, however,
    # when I moved the repository to PIL, this option was out of the window.
    # So, in order to use resizing with ANTIALIAS feature of PIL,
    # I briefly convert matrix to PIL image and then back.
    # If there is a more beautiful way, do not hesitate to send a PR.
    return cam


if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = get_example_params(target_example)
    # Grad cam
    # grad_cam = GradCam(pretrained_model, target_layer=11)
    model = LayeredModule.from_alexnet(pretrained_model)
    model.hook_to_activations = True
    # Generate cam mask
    cam = generate_cam(model, 'features-relu-4', prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
