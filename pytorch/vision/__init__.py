import math
import torch
import numpy as np
from pyhej.keras.image import load_img, img_to_array, array_to_img


def make_grid(tensor, nrow=8, padding=2, normalize=False, scale_each=False, pad_value=0):
    '''Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W).
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    '''
    if isinstance(tensor, list):
        if all(torch.is_tensor(t) for t in tensor):
            # if list of tensors, convert to a 4D mini-batch Tensor
            tensor = torch.stack(tensor, dim=0)
        elif all(isinstance(t, np.ndarray) for t in tensor):
            tensor = np.asarray(tensor)
            tensor = torch.from_numpy(tensor)
        elif all(isinstance(t, str) for t in tensor):
            tensor = [img_to_array(load_img(t), 'channels_first') for t in tensor]
            tensor = torch.from_numpy(tensor)
        else:
            raise TypeError('list of tensor or ndarray or str expected, got {}'.format(type(tensor[0])))

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))

    if tensor.dim() == 3:  # single image C x H x W
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place

        def func_norm(t):
            tmin = t.min()
            tmax = t.max()
            t.add_(-tmin).div_(tmax-tmin)

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                func_norm(t)
        else:
            func_norm(tensor)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2, normalize=False, scale_each=False, pad_value=0):
    '''Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    '''
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    img = array_to_img(ndarr)
    img.save(filename)