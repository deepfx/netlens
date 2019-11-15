from .modules import *


# Code for transforming special architectures to our layered model

def googlenet_to_layers(model: torchvision.models.GoogLeNet):
    if model.aux_logits:
        raise NotImplementedError('Aux logits not yet there!')

    layers = get_flat_layers(copy.deepcopy(model), keep_names=True)

    if model.transform_input:
        # append the transformer at the beginning
        def transform_input(x):
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            return torch.cat((x_ch0, x_ch1, x_ch2), 1)

        layers.insert(0, ('transform_input', Lambda(transform_input)))

    idx = find_index(layers, lambda l: l[0] == 'dropout')
    if idx >= 0:
        layers.insert(idx, ('flatten', Lambda(lambda x: torch.flatten(x, 1))))

    return layers, True


MODEL_CONVERSION_MAP = {
    torchvision.models.GoogLeNet: googlenet_to_layers
}


def convert_to_layers(model):
    model_arch = type(model)
    converter = MODEL_CONVERSION_MAP.get(model_arch)
    assert converter is not None, f'No special converter for model architecture {model_arch}'
    return converter(model)
