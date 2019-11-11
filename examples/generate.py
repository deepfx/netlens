import PIL.Image
import matplotlib.pyplot as plt
from torchvision import models

from visualization.generate import NetDreamer


def generate_class_sample():
    target_class = 130  # Flamingo
    pretrained_model = models.alexnet(pretrained=True)
    ns = NetDreamer(pretrained_model)
    pimg, rimg = ns.generate_class_sample(target_class)

    plt.imshow(pimg)
    plt.show()


def deep_dream():
    cnn_layer = 'conv-15'
    filter_pos = 94

    im_path = '../images/examples/dd_tree.jpg'
    img = PIL.Image.open(im_path).convert('RGB')

    # Fully connected layer is not needed
    pretrained_model = models.vgg19(pretrained=True).features
    ns = NetDreamer(pretrained_model)
    dd, _ = ns.deep_dream(img, cnn_layer, filter_pos, 20)

    plt.imshow(dd)
    plt.show()


if __name__ == '__main__':
    # generate_class_sample()
    deep_dream()
