from tests.fixtures import *

def test_from_nested_cnn(layered_from_alex):
    assert len(layered_from_alex.layers) == 22