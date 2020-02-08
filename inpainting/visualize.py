import flowiz as fz
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms.functional import to_pil_image

DEBUG = True
DEBUG_PATH = './debug'


def flow_to_pil_image(tensor):
    return Image.fromarray(fz.convert_from_flow(tensor.numpy().transpose(1, 2, 0)))


def debug(tensor, name, denormalization_function=None):
    if DEBUG:
        if denormalization_function:
            tensor = denormalization_function(tensor.clone())

        example = tensor[0].cpu()
        if 'flow' in name:
            image = flow_to_pil_image(example)
        else:
            image = to_pil_image(example)
        image.save(f'{DEBUG_PATH}/{name}.png')


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
