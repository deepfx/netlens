import torch


def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def one_hot_tensor(num_classes: int, target_class: int):
    one_hot = torch.zeros(1, num_classes)
    one_hot[0, target_class] = 1
    return one_hot
