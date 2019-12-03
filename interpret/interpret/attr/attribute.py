import torch
import matplotlib.pyplot as plt

from ..utils import denorm


class Attribute():
    """Class defining attribution maps over inputs. Contains
    useful plotting methods and implements mathematical
    operations on the underlying data.
    """
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        if isinstance(other, Attribute):
            return Attribute(self.data + other.data)
        elif isinstance(other, (int, float)):
            return Attribute(self.data + other)
        else:
            raise ValueError(f"Can't add type {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Attribute):
            # TODO: Attempt some shape checking
            return Attribute(self.data * other.data)
        elif isinstance(other, (int, float)):
            return Attribute(self.data * other)
        else:
            raise ValueError(f"Can't multiply by type {type(other)}")

    def __sub__(self, other):
        return self + (-1*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.__mul__(-1.)

    # TODO: Generalise this method to non-image data
    def show(self, ax=None, show_image=True, alpha=0.4, cmap='magma', colorbar=False):
        """Show the generated attribution map.

        Parameters:
        show_image (bool): show the denormalised input image overlaid on the heatmap.
        ax: axes on which to plot image.
        colorbar (bool): show a colorbar.
        cmap: matplotlib colourmap.
        alpha (float): transparency value alpha for heatmap.
        """
        if ax is None:
            _,ax = plt.subplots()

        sz = list(self.input_data.shape[2:])
        if show_image:
            input_image = denorm(self.input_data[0])
            ax.imshow(input_image)

        im = ax.imshow(self.data, alpha=alpha, extent=(0,*sz[::-1],0), interpolation='bilinear', cmap=cmap)
        if colorbar:
            ax.figure.colorbar(im, ax=ax)
