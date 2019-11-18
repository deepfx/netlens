import pytest
from fastai.vision import *
from visualization.modules import get_nested_layers

def isFemale(path): return bool(int(path.name.split('_')[1]))
def get_gender(path): return "female" if isFemale(path) else "male"

if __name__ == '__main__':
    path = '../data/faces_test/'
    data = (ImageList.from_folder(path) \
            # .filter_by_func() \
            .split_by_rand_pct(0.2) \
            .label_from_func(get_gender) \
            .transform([]) \
            .databunch() \
            .normalize())

    learner = cnn_learner(data, models.resnet18, metrics=[accuracy])
    get_nested_layers(learner.model)
