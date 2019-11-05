from math import ceil
from typing import Tuple

import torch
import torch.nn.functional as F
from toolz import curry

from pyimgy.core import convert_image

config = {
    "model_input_size": {
        "AlexNet": (224, 224)
    }
}


@curry
def get_steps(distance: int, step: int):
    return ((x, min(x + step, distance)) for x in range(0, distance, step))


def get_masks(bbox: Tuple[int, int], window: Tuple[int, int]):
    height, width = bbox
    step_x, step_y = window
    return [
        (x1, x2, y1, y2)
        for x1, x2 in get_steps(width, step_x)
        for y1, y2 in get_steps(height, step_y)
    ]


@curry
def apply_mask(image, mask, replace_value=0, in_place=False):
    if not in_place:
        image = image.clone()
    x1, x2, y1, y2 = mask
    image[..., x1: x2, y1: y2] = replace_value
    return image


def resize_like_input(input_size, masks, probs):
    img = torch.zeros(input_size)
    for mask, prob in zip(masks, probs):
        apply_mask(img, mask, prob, in_place=True)
    return img


def occlusion(model, input, target_class, window=(30, 30)):
    input = convert_image(input, to_type=torch.Tensor)

    if len(input.shape) == 4:
        input = input.squeeze(0)

    masks = get_masks(config["model_input_size"]["AlexNet"], window)

    print(f'{len(masks)} masks obtained for the input of size {input.shape}. First={masks[0]}; Last={masks[-1]}')

    masked_imgs = torch.stack(
        [apply_mask(input, mask) for mask in masks],
        dim=0
    )

    print(f'Input for the model has shape {masked_imgs.shape}')

    logits = model(masked_imgs).detach()
    output = F.softmax(logits, dim=1)
    probs = output[:, target_class]

    heatmap_height = ceil(input.shape[1] / window[0])
    heatmap_width = ceil(input.shape[2] / window[1])

    hm = probs.view(heatmap_height, heatmap_width)
    class_map = torch.argmax(logits, dim=1).view(heatmap_height, heatmap_width)
    hm_scaled = resize_like_input(input.shape[1:], masks, probs)
    return hm, class_map, hm_scaled
