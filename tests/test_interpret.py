import torch
from tests.fixtures import *


alex_hooked_keys = {'classifier-relu-0',
                    'classifier-relu-1',
                    'features-relu-0',
                    'features-relu-1',
                    'features-relu-2',
                    'features-relu-3',
                    'features-relu-4'}

alex_layer_keys = ['features-conv-0', 'features-relu-0', 'features-pool-0', 'features-conv-1',
                   'features-relu-1', 'features-pool-1', 'features-conv-2', 'features-relu-2',
                   'features-conv-3', 'features-relu-3', 'features-conv-4', 'features-relu-4',
                   'features-pool-2', 'avgpool-0', 'flatten', 'classifier-dropout-0', 'classifier-linear-0',
                   'classifier-relu-0', 'classifier-dropout-1', 'classifier-linear-1', 'classifier-relu-1',
                   'classifier-linear-2']


def test_input_gradient(net_lens_alex):
    gradient = net_lens_alex.get_input_gradient()
    gradient_guided = net_lens_alex.get_input_gradient(guided=True)

    assert gradient.shape == gradient_guided.shape == torch.Size([1, 3, 224, 224])


def test_netlens_setup(net_lens_alex):
    assert net_lens_alex.model.hooked_layer_keys == alex_hooked_keys


def test_conv_shapes(net_lens_alex):
    assert net_lens_alex.model.layers['features-conv-0'].weight.shape == torch.Size([64, 3, 11, 11])
    conv_output = net_lens_alex.model.forward(net_lens_alex.input_image, until_layer='features-relu-4')
    assert conv_output.shape == torch.Size([1, 256, 13, 13])
    assert conv_output.requires_grad == True
