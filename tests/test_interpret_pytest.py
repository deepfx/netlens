import torch
from visualization.image_proc import preprocess_image
from visualization.interpret import NetLens
from visualization.modules import LayeredModule
from torchvision import models
from visualization.data import get_example_data
import pytest

# use fixtures for intermediate (layeredModule)
@pytest.fixture
def layered_from_alex():
    return LayeredModule.from_nested_cnn(models.alexnet(pretrained=True))

alex_hooked_keys = {'classifier-relu-0',
 'classifier-relu-1',
 'features-relu-0',
 'features-relu-1',
 'features-relu-2',
 'features-relu-3',
 'features-relu-4'}

alex_layer_keys = (['features-conv-0', 'features-relu-0', 'features-pool-0', 'features-conv-1', 'features-relu-1', 'features-pool-1', 'features-conv-2', 'features-relu-2', 'features-conv-3', 'features-relu-3', 'features-conv-4', 'features-relu-4', 'features-pool-2', 'avgpool-0', 'flatten', 'classifier-dropout-0', 'classifier-linear-0', 'classifier-relu-0', 'classifier-dropout-1', 'classifier-linear-1', 'classifier-relu-1', 'classifier-linear-2'])


@pytest.fixture
def netLens_alex(layered_from_alex):
    original_img, name, target_class = get_example_data(0, img_path="images/examples/")
    prep_img = preprocess_image(original_img)
    return NetLens(layered_from_alex, prep_img, target_class)


def test_input_gradient(netLens_alex):
    gradient = netLens_alex.get_input_gradient()
    gradient_guided = netLens_alex.get_input_gradient(guided=True)

    assert gradient.shape == gradient_guided.shape == torch.Size([1, 3, 224, 224])

def test_netlens_setup(netLens_alex):
    print(netLens_alex, "netlens")
    assert netLens_alex.model.hooked_layer_keys == alex_hooked_keys
    assert netLens_alex.model.layers.keys == alex_layer_keys

def test_conv_shapes(netLens_alex):
    assert netLens_alex.model.layers['features-conv-0'].weight.shape == torch.Size([64, 3, 11, 11])
    conv_output = netLens_alex.model.forward(netLens_alex.input_image, until_layer='features-relu-4')
    assert conv_output.shape == torch.Size([1, 256, 13, 13])
    assert conv_output.requires_grad == True
    
def test_from_nested_cnn(layered_from_alex):
    assert len(layered_from_alex.layers) == 22


def test_get_example_data():
    original_img, name, target_class = get_example_data(0, img_path="images/examples/")
    assert (224, 224) == original_img.size
    assert "snake.jpg" == name
    assert 56 == target_class


# from examples/Visual-Gradient_backprop


#if __name__ == '__main__':

# def test_guided_gram_cam():
