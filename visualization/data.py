from typing import Tuple

import PIL.Image
from PIL.Image import Image as PILImage


def get_example_data(example_index: int, img_path) -> Tuple[PILImage, str, int]:
    """
    :param example_index:
    :param img_path:
    :return:
        original_image (numpy arr): Original image read from the file
        img_name (str)
        target_class (int): Target class for the image
    """
    # Pick one of the examples
    example_list = (('snake.jpg', 56),
                    ('cat_dog.png', 243),
                    ('spider.png', 72),
                    ('pelican.jpg', 144))
    img_name = example_list[example_index][0]
    target_class = example_list[example_index][1]
    # Read image
    original_image = PIL.Image.open(img_path + img_name).convert('RGB')
    return original_image, img_name, target_class
