import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io


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


def animate_sequence(upper_frames, lower_frames):
    fig, [ax1, ax2] = plt.subplots(2)
    ims = [[ax1.imshow(u, animated=True), ax2.imshow(l, animated=True)] for u, l in zip(upper_frames, lower_frames)]
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.close()
    return ani
