import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from enum import IntEnum
from PIL import Image
from pathlib import Path
import math
import matplotlib.pyplot as plt
from .data import random_shapes
from .plots import show_images

class DataType(IntEnum):
    Train = 1
    Valid = 2
    Test = 3

class DiabeticRetData(Dataset):
    """
    A dataset holding the diabetic retinopathy image data from
    https://www.kaggle.com/c/aptos2019-blindness-detection.

    data_type - Train, Valid or Test from the DataType enum.
    path - The path to the image data.
    tfms - Transforms to apply to each image.
    seed - seed the rng for train-test split.
    """
    def __init__(self, data_type, path=Path('data'), folder=Path('images'),
            csv_name=Path('train.csv'), tfms=None, seed=None, regres=False):
        self.df = pd.read_csv(path/csv_name)
        self.c = len(self.df.diagnosis.unique()) if not regres else 1

        # Split data into train validation and test sets
        np.random.seed(seed)
        train_idxs = np.random.choice(len(self.df), int(len(self.df)*0.7), replace=False)
        val_idxs = np.arange(len(self.df))
        val_idxs = np.setdiff1d(val_idxs, train_idxs)
        test_idxs = val_idxs.copy()
        val_idxs = np.random.choice(val_idxs, int(len(val_idxs)*0.5), replace=False)
        test_idxs = np.setdiff1d(test_idxs, val_idxs)

        self.data_type = data_type
        self.path = path
        self.folder = folder
        self.tfms = tfms
        self.regres = regres

        if self.data_type == DataType.Train:
            self.idxs = train_idxs
        elif self.data_type == DataType.Valid:
            self.idxs = val_idxs
        elif self.data_type == DataType.Test:
            self.idxs = test_idxs
        else:
            raise Exception(f"data_type must be a {DataType}")

    def __getitem__(self, idx):
        i = self.idxs[idx]
        filename, label = self.df.iloc[i]
        img = Image.open(self.path/self.folder/(filename+".png"))
        if self.tfms is not None:
            # img = self.tfms(image=np.asarray(img))['image'] # albumentations
            img = self.tfms(img) # torchvision
        return img, torch.tensor(label, dtype=torch.float32) if self.regres else label

    def __len__(self):
        return len(self.idxs)

    def show(self, num=9, figsize=(8,8), random=False):
        if self.tfms is None:
            r = math.ceil(math.sqrt(num))
            axes = plt.subplots(r,r,figsize=figsize)[1].flatten()
            for i,ax in enumerate(axes):
                if i<num:
                    im,label = self[np.random.randint(len(self))] if random else self[i]
                    if isinstance(im, torch.Tensor):
                        im = denorm(im)
                    ax.imshow(im)
                    ax.set_title(f'{self.decode_label(label)}')
                ax.set_axis_off()
        else:
            ims, lbs = zip(*[self[np.random.randint(len(self))] if random else self[i] for i in range(num)])
            batch = torch.stack(ims)
            show_images(batch, normalize=True, figsize=figsize, labels=lbs)

    @staticmethod
    def decode_label(lb):
        labels = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
        return labels[lb]

class RandomShape(Dataset):
    """
    Create a dataset of procedurally generated shapes.

    size - Size of the images.
    ds_length - length of the dataset.
    tfms - apply transforms to the dataset images.
    background - on of {'uniform', 'random', 'color'}
    number - The number of shapes in the image. If
             'random' then chosen between 1-number.
    random - Whether to choose a random number of images.
    weight - Weight the probability of seeing different classes.
             Normalized to sum to 1.
    """
    def __init__(self, size, ds_length, tfms=None, background='uniform', number=1, random=False, weight=None):
        self.size = size
        self.l = ds_length
        self.tfms = tfms
        self.bg = background
        self.num = number
        self.random = random
        self.probs = None if weight is None else np.array(weight)/np.array(weight).sum()

    def __len__(self):
        return self.l

    def __getitem__(self, _):
        im_number = np.random.randint(1,self.num) if self.random else self.num
        shape_choice = self.probs if self.probs is None else np.random.choice(['rectangle', 'circle', 'triangle'], p=self.probs)
        s = random_shapes(self.size, shape=shape_choice, min_size=self.size/3, max_size=self.size,
                             coord_limits=None, background=self.bg, number=im_number)
        return (self.tfms(s[0]),s[1]) if self.tfms is not None else s

    @staticmethod
    def decode_label(i):
        labels = {0: 'rectangle', 1: 'circle', 2: 'triangle'}
        return labels[i]
