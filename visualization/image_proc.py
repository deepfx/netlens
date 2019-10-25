import math
from typing import Tuple

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage


# SOME BASIC CONVERSIONS

def normalize_numpy_for_pil(arr: np.ndarray) -> np.ndarray:
    return (arr * 255).astype(np.uint8)


IMAGE_CONVERSION_MAP = {
    (np.ndarray, Image): lambda img: Image.fromarray(normalize_numpy_for_pil(img))
}


def convert(img, to_type):
    from_type = type(img)
    if from_type == to_type:
        return img
    converter = IMAGE_CONVERSION_MAP.get((from_type, to_type))
    if converter is None:
        raise Exception(f'Not possible to convert from {from_type} to {to_type}!!!')
    return converter(img)


def convert_to_grayscale(im_as_arr: np.ndarray, use_percentile: bool = True) -> np.ndarray:
    """
    Converts a 3-channel image to grayscale
    :param im_as_arr: (numpy arr) RGB image with shape (D,W,H)
    :param use_percentile: if True, the 99-percentile will be used instead of the max
    :return: (numpy_arr) Grayscale image with shape (1,W,H)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    if use_percentile:
        return normalize_to_range(grayscale_im, a_max=np.percentile(grayscale_im, 99), clip=True)
    else:
        return normalize_to_range(grayscale_im)


def normalize_to_range(arr: np.ndarray, a_min=None, a_max=None, clip: bool = False) -> np.ndarray:
    a_min = a_min or np.min(arr)
    a_max = a_max or np.max(arr)
    norm_arr = (arr - a_min) / (a_max - a_min)
    return np.clip(norm_arr, 0, 1) if clip else norm_arr


def apply_colormap_on_image(org_im: Image, activation: np.ndarray, colormap_name: str) -> Tuple[PILImage, PILImage]:
    """
    Applies a heat map over the image.
    :param org_im: (PIL img) Original image
    :param activation: (numpy arr) Activation map (grayscale) 0-255
    :param colormap_name: name of the colormap
    :return: the opaque heat map, and the original image with the heat map applied over it.
    """
    # Get colormap
    color_map = matplotlib.cm.get_cmap(colormap_name)
    opaque_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = opaque_heatmap.copy()
    heatmap[:, :, 3] = 0.4

    # Apply heatmap on image
    heatmap_on_image = Image.alpha_composite(org_im.convert('RGBA'), convert(heatmap, to_type=Image))
    return convert(opaque_heatmap, to_type=Image), heatmap_on_image


def format_np_output(np_arr: np.ndarray) -> np.ndarray:
    """
    It converts all the outputs to the same format which is 3xWxH using successive if conditions.

    :param np_arr: (Numpy array) Matrix of shape 1xWxH or WxH or 3xWxH
    :return:
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if np_arr.ndim == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it save-able by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose((1, 2, 0))
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it save-able by PIL
    if np.max(np_arr) <= 1:
        np_arr = normalize_numpy_for_pil(np_arr)
    return np_arr


def save_image(im, path):
    """
    Saves a numpy matrix or PIL image as an image.
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


# IMAGE PREPARATION FOR PYTORCH

# mean and std list for channels (Imagenet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_image(pil_im: Image, resize_im=True) -> torch.Tensor:
    """
        Processes image for CNNs

    Args:
        pil_im (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    im_as_arr = (im_as_arr / 255 - IMAGENET_MEAN[..., None, None]) / IMAGENET_STD[..., None, None]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,W,H
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_ten.requires_grad_()
    return im_as_ten


def recreate_image(im_as_ten: torch.Tensor) -> np.ndarray:
    """
    Recreates images from a torch tensor, sort of reverse pre-processing
    """
    recreated_im = im_as_ten.numpy()[0].copy()
    recreated_im = recreated_im * IMAGENET_STD[..., None, None] + IMAGENET_MEAN[..., None, None]
    recreated_im = np.clip(recreated_im, 0, 1)
    recreated_im = normalize_numpy_for_pil(recreated_im)
    return recreated_im.transpose((1, 2, 0))


def get_positive_negative_saliency(gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates positive and negative saliency maps based on the gradient
    """
    pos_saliency = np.maximum(0, gradient) / gradient.max()
    neg_saliency = np.maximum(0, -gradient) / -gradient.min()
    return pos_saliency, neg_saliency


def get_example_data(example_index: int, img_path) -> Tuple[PILImage, str, int]:
    """
    :param example_index:
    :param img_path:
    :return:
        original_image (numpy arr): Original image read from the file
        img_name (str)
        target_class (int): Target class for the image
    """
    # Pick one of the examples
    example_list = (('snake.jpg', 56),
                    ('cat_dog.png', 243),
                    ('spider.png', 72),
                    ('pelican.jpg', 144))
    img_name = example_list[example_index][0]
    target_class = example_list[example_index][1]
    # Read image
    original_image = Image.open(img_path + img_name).convert('RGB')
    return original_image, img_name, target_class


def show_images(imgs, titles=None, r=1, figsize=(15, 8)):
    if isinstance(titles, list):
        assert len(imgs) == len(titles)
    elif titles is not None:
        titles = [titles] * len(imgs)

    c = math.ceil(len(imgs) / r)
    fig = plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(r, c, i + 1)
        ax.imshow(format_for_plotting(img))
        ax.axis('off')
        if titles is not None:
            ax.set_title(str(titles[i]))
    return fig


def format_for_plotting(img):
    if type(img) == np.ndarray and img.ndim == 3:
        if img.shape[0] == 3:
            img = img.transpose((1, 2, 0))
    return img
