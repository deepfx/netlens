# custom function to conduct occlusion experim
import math
import seaborn as sns
import skimage
import torch.nn.functional as F
from flashtorch.utils.imagenet import ImageNetIndex
from torchvision import models

from visualization.image_proc import *
from visualization.tiling import *


def show_occlusion_tiles(image, occ_size=50, occ_stride=50):
    image = np.asarray(image)
    width, height = image.shape[:2]

    # get occlusion tiles
    occ_tiles = get_tiles_positions(width, height, occ_size, occ_size, occ_stride, occ_stride)

    # show the tiles
    show_image_with_tiles(image, occ_size, occ_tiles)
    plt.show()


def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0):
    # get the width and height of the image
    height, width = image.shape[-2:]

    # setting the output image width and height
    # output_height = (height - occ_size) // occ_stride + 1
    # output_width = (width - occ_size) // occ_stride + 1
    # output_height = math.ceil(height / occ_stride)
    # output_width = math.ceil(width / occ_stride)
    def get_output_dim(dim):
        output_dim = math.ceil(dim / occ_stride)
        return output_dim - (1 if (output_dim - 1) * occ_stride == dim else 0)

    output_height = get_output_dim(height)
    output_width = get_output_dim(width)
    # create a white image of sizes we defined

    heatmap = torch.zeros((output_height, output_width))
    classmap = torch.zeros((output_height, output_width))
    scaled_map = torch.zeros((height, width))

    # iterate all the pixels in each column
    for h in range(0, output_height):
        for w in range(0, output_width):
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            if h_start == h_end or w_start == w_end: continue

            input_image = image.clone().detach()

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, h_start:h_end, w_start:w_end] = occ_pixel

            # run inference on modified image
            # setting the heatmap location to probability value
            prob, classmap[h, w] = prob_for_label(model, input_image, label)
            heatmap[h, w] = prob
            scaled_map[h_start:h_end, w_start:w_end] = prob  # * torch.ones((h_end - h_start, w_end - w_start))

    return heatmap, classmap, scaled_map


def prob_for_label(model, image, label):
    output = model(image)
    output = F.softmax(output, dim=1)
    pred_class = torch.argmax(output, dim=1).item()
    prob = output[0, label].item()
    return prob, pred_class


def main():
    original_img, name, target_class = get_example_data(1, img_path='../old_visual/input_images/')

    prep_img = preprocess_image(original_img)

    m_orig = models.vgg19_bn(pretrained=True)
    m_orig.eval()

    hm, cm, hm_scaled = occlusion(m_orig, prep_img, target_class, occ_size=50, occ_stride=50, occ_pixel=0)
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
