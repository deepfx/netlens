import torch

from old_visual.utils.misc_functions import get_example_params, convert_to_grayscale, save_gradient_images
from visualization.modules import LayeredModule


def check_equality(t1, t2):
    print(t1.shape, t2.shape, t1.shape == t2.shape and torch.all(torch.eq(t1, t2)))


if __name__ == '__main__':
    # Get params
    target_example = 1  # Snake
    (_, prep_img, target_class, file_name_to_export, pretrained_model) = get_example_params(target_example)
    # Vanilla backprop
    model = LayeredModule.from_alexnet(pretrained_model)
    model.hook_to_activations = True
    # Generate gradients
    gradients = model.get_input_gradient(prep_img, target_class)

    # TEST: here we confirm that the values captured by the nn.Module backward hook can be obtained by different means
    '''grad_in, grad_out = model.layer_gradients['features-conv-0']
    check_equality(grad_out[0], model.activation_gradients['features-conv-0'])
    check_equality(grad_in[0], model.activation_gradients['input'])
    check_equality(grad_in[1], model.param_gradients['features-conv-0.weight'])
    check_equality(grad_in[2], model.param_gradients['features-conv-0.bias'])'''

    # we need to get the first layer, squeeze it an make it numpy
    vanilla_grads = gradients['input'].squeeze(0).numpy()
    # Save colored gradients
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')
