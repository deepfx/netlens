import torch
from torch import nn

class FlattenedLoss():
    "Flattens the inputs before passing on to loss_fn"
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, p, y):
        return self.loss_fn(p.view(-1), y.view(-1))

class FocalLoss(nn.Module):
    "https://arxiv.org/abs/1708.02002"
    def __init__(self, gamma=2, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class TverskyLoss(nn.Module):
    "https://arxiv.org/abs/1706.05721"
    def __init__(self, alpha=0.5, beta=0.5, regres=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.regres = regres

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)

        return 1 - Tversky

class DiceLoss(nn.Module):
    def __init__(self, regres=False):
        super().__init__()
        self.regres = regres

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
