from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

paths = {"style": "images/picasso.jpg", "content": "images/dancing.jpg"}
#neural networks from the torch library are trained with tensor values ranging
# from 0 to 1. If you try to feed the networks with 0 to 255 tensor images,
# then the activated feature maps will be unable sense the intended content and style
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(path):
    return loader(Image.open(path))\
                .unsqueeze(0)\
                .to(device, torch.float)        # fake batch dimension required to fit network's input dimensions

style_img = image_loader(paths.style)
content_img = image_loader(paths.content)

#Todo: separate assertion loading and make PiPe
assert style_img.size() == content_img.size()

def display_img(img):
    pimg = tensor.cpu().clone()
    pimg = image.squeeze(0) # remove the fake BATCH dimension
    pimg = unloader(img)

    plt.imshow(pimg)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    #plt.subplots(nrows=1, ncols=1, figsize=)


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):
    #Important detail: although this module is named ContentLoss, it is not a
    # true PyTorch Loss function. If you want to define your content loss as a
    # PyTorch Loss function, you have to create a PyTorch
    #!! autograd function to recompute/implement!! the gradient manually in the backward method.
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps OR channel depth?
    # (c,d)=dimensions of a f. map (N=c*d) eg 512 x 512
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G =  features @ features.t() #maybe use matmull to be sure
    return G.div(a * b *c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature): #target = Gram Mat
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

