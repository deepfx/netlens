# custom function to conduct occlusion experim
import seaborn as sns
import torch
from torch import nn
import numpy as np
from torchvision import models
from visualization.modules import LayeredModule
from visualization.image_proc import *

def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):
    # get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    # setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))
    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))
    # iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            # run inference on modified image
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)

            prob = output.tolist()[0][label]

            # setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap

def prob_for_label(model, image, label):
    output = model(image)
    output = nn.functional.softmax(output, dim=1)

    prob = output.tolist()[0][label]
    return prob

if __name__ == '__main__':
    model = LayeredModule.from_alexnet(models.alexnet(pretrained=True))
    print(model.layers)

    original_img, name, target_class = get_example_data(0, img_path='../old_visual/input_images/')
    prep_img = preprocess_image(original_img)
    #output = model(prep_img)
    m_orig = models.alexnet(pretrained=True)
    p = prob_for_label(m_orig, prep_img, target_class)

    print(p, "prob", target_class, "targ")
    '''
        #interestingly changes a bit everytime
    print(output[-1][target_class], target_class)
    print(output[-1].max(), output[-1])
    hm = occlusion(model, prep_img, target_class, occ_size=25, occ_stride=10, occ_pixel=0  )
    prob_no_occ = torch.max(hm)

    #ax = sns.heatmap(uniform_data, )
    ax = sns.heatmap(hm, xticklabels=False, yticklabels=False, vmax=prob_no_occ)
    plt.show()
    #figure = imgplot.get_figure()
    print(hm)
    
    '''





