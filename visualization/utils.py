import collections
from collections import defaultdict
from typing import Tuple, Iterable

from pydash import find_index
from torch import nn


def clean_layer(layer):
    return nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer


def find_indices(iterable, predicate):
    return [i for i, x in enumerate(iterable) if predicate(x)]


# FUNCTIONS TO ENCODE/DECODE LAYER KEYS (str) TO TUPLES
# Example: tuple=('conv', 1) <--> key='conv-1'

def tuple_to_key(name: str, nth: int) -> str:
    return f'{name}-{nth}'


def key_to_tuple(key: str) -> Tuple[str, int]:
    parts = key.rsplit('-', 1)
    return parts[0], int(parts[1])


def get_parent_name(name):
    return name.rsplit('.', 1)[0] if '.' in name else ''


def as_list(obj) -> list:
    if isinstance(obj, collections.Iterable):
        return list(obj)
    elif obj is not None:
        return [obj]
    else:
        return []


class KeyCounter:

    def __init__(self, as_tuples: bool = False):
        self.counts = defaultdict(int)
        self.as_tuples = as_tuples

    def get_next(self, name: str):
        nth = self.counts[name]
        self.counts[name] += 1
        return (name, nth) if self.as_tuples else tuple_to_key(name, nth)


def enumerate_module_keys(named_modules: Iterable[Tuple[str, nn.Module]]) -> Iterable[Tuple[str, nn.Module]]:
    key_counter = KeyCounter()
    return [(key_counter.get_next(name), module) for name, module in named_modules]


def insert_layer_after(layer_list, insertion_key: str, new_key: str, new_layer: nn.Module):
    idx = find_index(layer_list, lambda l: l[0] == insertion_key)
    if idx >= 0:
        layer_list.insert(idx + 1, (new_key, new_layer))
    return layer_list


def delete_all_layers_after(layer_list, last_key: str):
    idx = find_index(layer_list, lambda l: l[0] == last_key)
    return layer_list[:idx + 1] if idx >= 0 else layer_list
