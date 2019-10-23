from old_visual.utils.misc_functions import get_example_params, convert_to_grayscale, save_gradient_images
from visualization.modules import LayeredModule

if __name__ == '__main__':
    # Get params
    target_example = 1  # Snake
    (_, prep_img, target_class, file_name_to_export, pretrained_model) = get_example_params(target_example)
    # Vanilla backprop
    model = LayeredModule.from_alexnet(pretrained_model)
    # Generate gradients
    gradients = model.get_gradients_for_sample(prep_img, target_class)
    # we need to get the first layer, squeeze it an make it numpy
    vanilla_grads = gradients['features-conv-0'][0].data.numpy()[0]
    # Save colored gradients
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')
