import torch


def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div_(a * b * c * d)


def gram_matrix_2(array, normalize_magnitude=True):
    channels = array.shape[1]
    array_flat = array.permute((0, 2, 3, 1)).view((-1, channels))
    gm = torch.mm(array_flat, array_flat.t())
    if normalize_magnitude:
        length = array_flat.shape[0]
        gm.div_(length)
    return gm


def gram_matrix_2(array, normalize_magnitude=True):
    channels = array.shape[1]
    array_flat = array.permute((0, 2, 3, 1)).view((-1, channels))
    gm = torch.mm(array_flat, array_flat.t())
    if normalize_magnitude:
        length = array_flat.shape[0]
        gm.div_(length)
    return gm


def one_hot_tensor(num_classes: int, target_class: int, device=None):
    one_hot = torch.zeros(1, num_classes, device=device)
    one_hot[0, target_class] = 1
    return one_hot


def alpha_norm(input, alpha):
    return (input.view(-1) ** alpha).sum()


def total_variation_norm(input, beta):
    to_check = input[:, :-1, :-1]  # Trimmed: right - bottom
    one_bottom = input[:, 1:, :-1]  # Trimmed: top - right
    one_right = input[:, :-1, 1:]  # Trimmed: top - right
    return (((to_check - one_bottom) ** 2 + (to_check - one_right) ** 2) ** (beta / 2)).sum()


def normalized_euclidean_loss(original, target):
    return alpha_norm(target - original, 2) / alpha_norm(original, 2)
