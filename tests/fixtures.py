import pytest
from torchvision import models

from visualization.data import get_example_data
from visualization.interpret import NetLens, preprocess_image
from visualization.modules import FlatModel

'''
Setup to test models
'''


# TODO: add test for all model types!

@pytest.fixture
def layered_from_alex():
    return FlatModel.from_nested_cnn(models.alexnet(pretrained=True))


@pytest.fixture
def net_lens_alex(layered_from_alex):
    original_img, name, target_class = get_example_data(0, img_path="images/examples/")
    prep_img = preprocess_image(original_img)
    nl = NetLens(layered_from_alex, prep_img, target_class)
    # FIXME: running get_input bc model is set to None in NetLens.___init__ otherwise
    nl.input_gradient()
    return nl


def test_get_example_data():
    original_img, name, target_class = get_example_data(0, img_path="images/examples/")
    assert (224, 224) == original_img.size
    assert "snake.jpg" == name
    assert 56 == target_class
