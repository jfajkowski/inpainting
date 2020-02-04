import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import color, io
from torchvision.utils import make_grid


# https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/utils.py


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def show_labels(image, labels):
    merged = color.label2rgb(labels, image, bg_label=0)
    io.imshow(merged)
    io.show()


def animate_sequence(*args):
    fig, axes = plt.subplots(1, len(args))
    for ax in axes:
        ax.axis('off')
    images = []
    for elements in zip(*args):
        images.append([ax.imshow(e, animated=True) for ax, e in zip(axes, elements)])
    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat_delay=1000)
    plt.close()
    return ani


def show_image(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    io.imshow(ndarr)
    io.show()
