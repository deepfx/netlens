from torchvision import models

from visualization.modules import Normalization
from visualization.style_transfer import *

if __name__ == '__main__':
    # just a test for debugging
    cnn = models.vgg19(pretrained=True).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    arch = LayeredModule.from_cnn(cnn, Normalization(cnn_normalization_mean, cnn_normalization_std))
    print(arch)

    style_injects = [f'conv-{i}' for i in range(5)]
    content_injects = ['conv-3']
    content_img = torch.zeros((1, 3, 128, 128))
    style_img = torch.zeros((1, 3, 128, 128))
    style_module = StyleTransferModule(arch, content_img, content_injects, style_img, style_injects)
    print(style_module)
