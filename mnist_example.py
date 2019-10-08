import gzip
import pickle
from math import ceil
from pathlib import Path
from typing import Tuple, Callable, Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"


def download_data() -> Tuple[Tensor, ...]:
    PATH.mkdir(parents=True, exist_ok=True)
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        print(x_train, y_train)
        print(x_train.shape)
        print(y_train.min(), y_train.max())
        return tuple(map(torch.tensor, (x_train, y_train, x_valid, y_valid)))


def show_data_example(x: Tensor) -> None:
    plt.imshow(x.reshape((28, 28)), cmap="gray")
    plt.show()


class Trace(nn.Module):
    def __init__(self):
        super().__init__()
        self.stored = None

    def forward(self, input):
        self.stored = input
        return input


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Lambda(preprocess),
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Trace(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Trace(),
            nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Trace(),
            nn.AvgPool2d(4),
            Lambda(lambda x: x.view(x.size(0), -1)),
        )
        self.traces = tuple(l for l in self.layers if isinstance(l, Trace))

    def forward(self, input):
        return self.layers(input)


def get_data(train_ds: TensorDataset, valid_ds: TensorDataset, bs: int):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def loss_batch(model: nn.Module, loss_func: Callable, xb: Tensor, yb: Tensor, opt: Optional[Optimizer] = None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs: int, model: nn.Module, loss_func: Callable, opt: Optimizer, train_dl: DataLoader, valid_dl: DataLoader):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


def train(bs=64, lr=0.1, epochs=4):
    x_train, y_train, x_valid, y_valid = download_data()
    show_data_example(x_train[0])

    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

    model = MyNet()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_func = F.cross_entropy

    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    return model, opt, train_dl, valid_dl


def show_activations(model: nn.Module, x: Tensor, normalize_layers=True, ncols=10, figsize=(20, 10)):
    ypred = model(x)

    nrows = 1 + sum(ceil(t.stored.shape[1] / ncols) for t in model.traces)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(nrows, ncols, 1)
    ax.imshow(x.reshape((28, 28)), cmap="gray")

    pos = ncols + 1
    for l, layer in enumerate(model.traces):
        kernels = layer.stored.detach().numpy()
        num_channels = kernels.shape[1]
        vmin, vmax = (kernels.min(), kernels.max()) if normalize_layers else (None, None)
        for c in range(num_channels):
            ax = fig.add_subplot(nrows, ncols, pos + c)
            ax.imshow(kernels[0, c, :, :], vmin=vmin, vmax=vmax)
            ax.set_title('L%02d C%02d' % (l, c))
            ax.axis('off')
        pos += ceil(num_channels / ncols) * ncols

    plt.show()

    print(ypred)
    return ypred


if __name__ == '__main__':
    train()
