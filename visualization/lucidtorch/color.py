"""Functions for transforming and constraining color channels."""

import numpy as np
import torch
import torch.nn

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]


def _linear_decorelate_color(t: torch.Tensor) -> torch.Tensor:
    """Multiply input by sqrt of emperical (ImageNet) color correlation matrix.

    If you interpret t's innermost dimension as describing colors in a
    decorrelated version of the color space (which is a very natural way to
    describe colors -- see discussion in Feature Visualization article) the way
    to map back to normal colors is multiply the square root of your color
    correlations.
    """
    # check that inner dimension is 3?
    t_flat = t.reshape((-1, 3))
    color_correlation_normalized = torch.from_numpy(color_correlation_svd_sqrt / max_norm_svd_sqrt)
    t_flat = t_flat.matmul(color_correlation_normalized.t())
    t = t_flat.reshape(t.shape)
    return t


def to_valid_rgb(t: torch.Tensor, decorrelate: bool = False, sigmoid: bool = True) -> torch.Tensor:
    """Transform inner dimension of t to valid rgb colors.

    In practice this consistes of two parts:
    (1) If requested, transform the colors from a decorrelated color space to RGB.
    (2) Constrain the color channels to be in [0,1], either using a sigmoid
        function or clipping.

    Args:
      t: input tensor, innermost dimension will be interpreted as colors
        and transformed/constrained.
      decorrelate: should the input tensor's colors be interpreted as coming from
        a whitened space or not?
      sigmoid: should the colors be constrained using sigmoid (if True) or
        clipping (if False).

    Returns:
      t with the innermost dimension transformed.
    """
    if decorrelate:
        t = _linear_decorelate_color(t)
    if decorrelate and not sigmoid:
        t += color_mean
    if sigmoid:
        return torch.sigmoid(t)
    else:
        raise NotImplementedError()
