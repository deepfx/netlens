from typing import Tuple

import torch
from torch import nn


def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def clean_layer(layer):
    return nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer


def find_indices(iterable, predicate):
    return [i for i, x in enumerate(iterable) if predicate(x)]


# FUNCTIONS TO ENCODE/DECODE LAYER KEYS (str) TO TUPLES
# Example: tuple=('conv', 1) <--> key='conv-1'

def tuple_to_key(name: str, nth: int) -> str:
    assert '-' not in name, "The name cannot contain a '-' character."
    return f'{name}-{nth}'


def key_to_tuple(key: str) -> Tuple[str, int]:
    parts = key.rsplit('-', 1)
    return parts[0], int(parts[1])
