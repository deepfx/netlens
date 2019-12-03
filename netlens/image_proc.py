import PIL.Image
import matplotlib.cm
import torchvision.transforms as T

import netlens.transforms as T2
from pyimgy.optional.torch import *


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


def apply_colormap_on_image(org_im: PILImage, activation: np.ndarray, colormap_name: str) -> Tuple[PILImage, PILImage]:
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
    heatmap_on_image = PIL.Image.alpha_composite(org_im.convert('RGBA'), convert_to_standard_pil(heatmap))
    return convert_to_standard_pil(opaque_heatmap), heatmap_on_image


# IMAGE PREPARATION FOR PYTORCH

# mean and std list for channels (Imagenet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_image(original_img, resize_to=(224, 224), thumbnail=True) -> torch.Tensor:
    transforms = []
    if resize_to is not None:
        if not isinstance(original_img, PILImage):
            transforms.append(T.ToPILImage())
        if thumbnail:
            transforms.append(T2.Thumbnail(resize_to))
        else:
            transforms.append(T.Resize(resize_to))

    transforms += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return T.Compose(transforms)(original_img).unsqueeze(0).requires_grad_()


def recreate_image(im_as_ten: torch.Tensor, denormalize: bool = True, to_pil: bool = False) -> Union[np.ndarray, PILImage]:
    """
    Recreates images from a torch tensor, sort of reverse pre-processing
    """
    recreated_im = convert_image(im_as_ten.detach(), to_type=np.ndarray, shape='3WH')
    if denormalize:
        recreated_im = recreated_im * IMAGENET_STD[..., None, None] + IMAGENET_MEAN[..., None, None]
    recreated_im = np.clip(recreated_im, 0, 1)
    return convert_to_standard_pil(recreated_im) if to_pil else convert_for_plot(recreated_im)


def get_positive_negative_saliency(gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates positive and negative saliency maps based on the gradient
    """
    pos_saliency = np.maximum(0, gradient) / gradient.max()
    neg_saliency = np.maximum(0, -gradient) / -gradient.min()
    return pos_saliency, neg_saliency
