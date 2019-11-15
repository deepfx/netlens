from torchvision import transforms
from torch import nn
import torch
import random, math
from torch.nn.functional import affine_grid, grid_sample
from torch.nn import functional as F
from PIL import Image
from .imagenet import imagenet_stats

class Blur():
    "Apply a uniform blur to an input with kernel_size"
    def __init__(self, kernel_size):
        channels=3
        self.filter = nn.Conv2d(channels,channels,kernel_size, bias=False, groups=channels)
        kernel = torch.ones((channels,1,kernel_size,kernel_size))/(kernel_size*kernel_size)

        self.filter.weight.data = kernel
        self.filter.weight.requires_grad_(False)

    def __call__(self, x):
        if x.dim() == 3:
            x.unsqueeze_(0)
            return self.filter(x).squeeze()
        return self.filter(x)

class GaussianBlur():
    """
    Apply a blur to an input with a Gaussian kernel.

    The input should have the same number of channels as `channels`.
    The Gaussian kernel has size kernel_size, and standard deviation sigma.

    Sigma >> 1 will create an almost uniform kernel, while
    sigma << 1 will create a very focused kernel (less blurring).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.channels, self.kernel_size, self.sigma = channels, kernel_size, sigma
        self.get_kernel(self.channels, self.kernel_size, self.sigma, dim=2)
        self.conv = nn.Conv2d(channels, channels, kernel_size, groups=channels, bias=False, padding=kernel_size//2)
        self.conv.weight.data = self.kernel
        self.conv.weight.requires_grad_(False)
        self.conv.to(device)
        self.device = device

    def get_kernel(self, channels, kernel_size, sigma, dim=2):
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(kernel_size, dtype=torch.float32)
                for size in range(dim)
            ]
        )
        for mgrid in meshgrids:
            mean = (kernel_size - 1) / 2
            kernel *= 1 / (sigma * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / sigma) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        self.kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    def __call__(self, x):
        if x.dim() == 3:
            x.unsqueeze_(0)
            return self.conv(x).squeeze()
        return self.conv(x)

class ReducingGaussianBlur(GaussianBlur):
    "Dynamically reduce the sigma value after applying every blur."
    def __call__(self, x):
        self.sigma *= 0.95
        self.get_kernel(self.channels, self.kernel_size, self.sigma)
        self.conv.weight.data = self.kernel
        self.conv.weight.requires_grad_(False)
        self.conv.to(self.device)
        return super().__call__(x)

def get_transforms(size, mean=imagenet_stats[0], std=imagenet_stats[1], rotate=10,
                    flip_hor=True, flip_vert=False, perspective=True, color_jitter=True,
                    translate=None, zoom=None, shear=None):
    """
    Returns a set of default transformations.

    size - Resize image (int).
    mean and std - Normalize image.
    rotate - Rotate randomly between (-rotate, rotate) degrees.
    flip_hor - Randomly flip horizontally with p=0.5
    flip_vert - Randomly flip vertically with p=0.5
    perspective - Apply a perspective warp.
    color_jitter - Jitter the colors in the image.
    translate - Randomly translate the center of the
                image horizontally and vertically (tuple)
    zoom - Randomly scale the image (tuple)
    shear - Randomly shear the image (int or tuple)
    """
    tfms = [
        transforms.Resize((size, size)),
    ]
    # if rotate is not None: tfms += [transforms.RandomRotation(rotate)]
    tfms += [transforms.RandomAffine(rotate, translate=translate, scale=zoom,
                            shear=shear, resample=Image.BILINEAR, fillcolor=0)]
    if flip_hor: tfms += [transforms.RandomHorizontalFlip()]
    if flip_vert: tfms += [transforms.RandomVerticalFlip()]
    if perspective: tfms += [transforms.RandomPerspective()]

    if isinstance(color_jitter, tuple):
        brightness, contrast, saturation, hue = color_jitter
        if color_jitter: tfms += [transforms.ColorJitter(brightness,contrast,saturation,hue)]
    elif color_jitter:
        brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0
        if color_jitter: tfms += [transforms.ColorJitter(brightness,contrast,saturation,hue)]
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    return transforms.Compose(tfms)

def resize_norm_transform(size, mean=imagenet_stats[0], std=imagenet_stats[1]):
    "Return a list of transforms that only resizes and normalizes data."
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def affine(mat):
    "applies an affine transform"
    def inner(x):
        if x.dim() == 3: x=x.unsqueeze(0)
        grid = affine_grid(mat, x.size())
        rot_im = grid_sample(x, grid, padding_mode="reflection")
        return rot_im
    return inner

def rotate(angle=20):
    "rotates the image in degrees"
    rad = math.pi*angle/180
    rot = torch.tensor([[math.cos(rad), -math.sin(rad), 0],
                        [math.sin(rad), math.cos(rad), 0],
                        ]).unsqueeze(0)
    rot = rot.to('cuda' if torch.cuda.is_available() else 'cpu')
    return affine(rot)

def translate(x,y):
    "translates the center of image to the coordinate x,y"
    rot = torch.tensor([[1, 0, -x],
                        [0, 1, y],
                        ], dtype=torch.float32).unsqueeze(0)
    rot = rot.to('cuda' if torch.cuda.is_available() else 'cpu')
    return affine(rot)

def shear(shear):
    "shears the image"
    rot = torch.tensor([[1, shear, 0],
                        [0, 1, 0],
                        ], dtype=torch.float32).unsqueeze(0)
    rot = rot.to('cuda' if torch.cuda.is_available() else 'cpu')
    return affine(rot)

def scale(scale):
    "scales the image"
    mat = torch.tensor([[1, 0, 0],
                        [0, 1, 0],
                        ], dtype=torch.float32).unsqueeze(0)/scale
    mat = mat.to('cuda' if torch.cuda.is_available() else 'cpu')
    return affine(mat)

class RandomAffineTfm():
    "Randomly apply an affine transform on a tensor."
    def __init__(self, tfm, *args):
        self.tfm = tfm
        self.args = args

    def __call__(self, x):
        return self.tfm(*[random.random()*(a[1]-a[0])+a[0] if isinstance(a, list) else random.random()*a*2-a for a in self.args])(x)

class RandomEvery():
    def __init__(self, tfms, p=0.5):
        self.tfms = tfms
        self.p = p

    def __call__(self, x):
        for t in self.tfms:
            if random.random() < self.p:
                x = t(x)
        return x
