import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# TILING


def get_tiles_positions(W, H, w, h, step_x=None, step_y=None):
    # if steps are not given, just the tile size
    step_x = step_x or w
    step_y = step_y or h

    used_w = ((W - w) // step_x + 1) * step_x
    used_h = ((H - h) // step_y + 1) * step_y

    return [(x, y) for x in range(0, used_w, step_x) for y in range(0, used_h, step_y)]


def get_image_tiles(img: np.ndarray, w, h, step_x=None, step_y=None):
    """
    Splits the given image in tiles of the specified weight and height; optionally giving a step for sliding tiles.
    It drops the remaining parts of the image that could not fit in a tile. It returns a list of "images" (numpy arrays).
    """
    # make sure it's a numpy array -- it will fail if not possible
    im_h, im_w = img.shape[0], img.shape[1]

    pos = get_tiles_positions(im_w, im_h, w, h, step_x, step_y)
    tiles = [img[y:y + h, x:x + w] for (x, y) in pos]
    return tiles, pos


def draw_tile_box(ax, pos, size, color='r', with_marker=False):
    rect = patches.Rectangle(pos, size, size, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    if with_marker:
        marker = patches.Rectangle(pos, 10, 10, linewidth=1, edgecolor=color, facecolor=color)
        ax.add_patch(marker)


def show_image_with_tiles(img: np.ndarray, tile_size: int, tile_pos: list) -> None:
    print('Image size: %s, total tiles: %d' % (img.shape, len(tile_pos)))
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.axis('off')
    # show the image itself
    ax.imshow(img)
    for pos in tile_pos: draw_tile_box(ax, pos, tile_size)
    draw_tile_box(ax, (0, 0), tile_size, color='b')
