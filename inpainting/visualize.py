from os import makedirs
from os.path import dirname

import flowiz as fz
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import cv2 as cv
import subprocess as sp

from PIL import Image
from torchvision.transforms.functional import to_pil_image

from inpainting.utils import tensor_to_cv_image, tensor_to_cv_mask

DEBUG = True
DEBUG_PATH = 'debug'


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
        images.append([ax.imshow(e, animated=True, cmap='Greys') for ax, e in zip(axes, elements)])
    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat_delay=1000)
    plt.close()
    return ani


def save_frames(frames, dir, frame_type='image'):
    extension = None
    if frame_type == 'image':
        extension = 'jpg'
    elif frame_type == 'mask':
        extension = 'png'
    else:
        ValueError(frame_type)

    for i, frame in enumerate(frames):
        save_frame(frame, f'{dir}/{i:05d}.{extension}', frame_type)


def save_frame(frame, path, frame_type='image', roi=None):
    if frame_type == 'image':
        frame = tensor_to_cv_image(frame)
    elif frame_type == 'mask':
        frame = tensor_to_cv_mask(frame)
    else:
        ValueError(frame_type)

    if roi:
        assert frame_type == 'image'
        frame = cv.rectangle(frame, roi[0], roi[1], (255, 0, 0))

    makedirs(dirname(path), exist_ok=True)
    cv.imwrite(path, frame)


def save_video(video, path, frame_type='image', frame_rate=24, codec=cv.VideoWriter_fourcc(*'H264')):
    if frame_type == 'image':
        video = [tensor_to_cv_image(t) for t in video]
    elif frame_type == 'mask':
        video = [tensor_to_cv_mask(t) for t in video]
    else:
        ValueError(frame_type)

    height, width = video[0].shape[-1], video[0].shape[-2]
    video_writer = cv.VideoWriter(path, codec, frame_rate, (width, height))
    for frame in video:
        video_writer.write(frame)
    video_writer.release()
