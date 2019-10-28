from math import ceil

import seaborn as sns
import torch
import torch.nn.functional as F
from flashtorch.utils import ImageNetIndex
from toolz import curry
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt

from image_proc import *
from tiling import *
from typing import Tuple

config = {
    "model_inputs": {
        "AlexNet": (224, 224)
    }
}


# getBoxes => (6,5) - ceil(w / step)...hm_w, hm_h
# boxes * step


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


@curry
def get_probs(model, inputs, logit_fn: nn.functional.softmax):
    logits = model(inputs)
    return logit_fn(logits, dim=1)


def rescale(input_size, masks, probs):
    img = torch.zeros(input_size)
    for mask, prob in zip(masks, probs):
        apply_mask(img, mask, prob, in_place=True)
    return img


def occlusion(model, input, target_class, window=(30, 30)):
    input = convert(input, torch.Tensor)

    if len(input.shape) == 4:
        input = input.squeeze(0)

    masks = get_masks(config["model_inputs"]["AlexNet"], window)

    masked_imgs = torch.stack(
        [apply_mask(input, mask) for mask in masks],
        dim=0
    )

    logits = model(masked_imgs)
    output = F.softmax(logits, dim=1)
    probs = output[:, target_class]

    heatmap_height = ceil(input.shape[1] / window[0])
    heatmap_width = ceil(input.shape[2] / window[1])

    class_map = torch.argmax(logits, dim=1).view(heatmap_height, heatmap_width)
    hm = probs.view(heatmap_height, heatmap_width)
    hm_scaled = rescale(input.shape[1:], masks, probs)
    return hm, class_map, hm_scaled


def main():
    original_img, name, target_class = get_example_data(1, img_path='../old_visual/input_images/')

    prep_img = preprocess_image(original_img)

    m_orig = models.vgg19_bn(pretrained=True)
    m_orig.eval()

    hm, cm, hm_scaled = occlusion(m_orig, prep_img, target_class)
    prob_no_occ = torch.max(hm)

    hmu = torch.unique(hm_scaled)
    print(hm_scaled.size(), hmu, hmu.size())

    # hm_scaled = skimage.transform.resize(hm, prep_img.shape[-2:], order=0)

    _, axes = plt.subplots(1, 3, figsize=(15, 8))

    sns.heatmap(hm, xticklabels=False, yticklabels=False, vmax=prob_no_occ, ax=axes[0])
    sns.heatmap(cm, xticklabels=False, yticklabels=False, ax=axes[1])
    axes[2].imshow(original_img)
    axes[2].imshow(hm_scaled, alpha=0.50)
    plt.show()
    # print(hm)

    det_classes, det_counts = torch.unique(cm, return_counts=True)

    ini = ImageNetIndex()
    in_labels = {v: k for k, v in ini.items()}

    print('DETECTED CLASSES')
    print('\n'.join(f'{int(cl)}\t{cnt}\t{in_labels[cl]}' for cl, cnt in zip(det_classes.tolist(), det_counts.tolist())))


if __name__ == '__main__':
    main()
