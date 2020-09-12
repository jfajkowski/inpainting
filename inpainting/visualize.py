from os import makedirs
from os.path import dirname

import cv2 as cv
import flowiz as fz
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid

from inpainting.utils import tensor_to_cv_image, tensor_to_cv_mask

DEBUG = False
DEBUG_PATH = 'debug'


def debug(tensor, name, denormalization_function=None):
    if DEBUG:
        if denormalization_function:
            tensor = denormalization_function(tensor.clone())

        example = tensor[0].cpu()
        if 'flow' in name:
            image = flow_tensor_to_image_tensor(example)
        else:
            image = to_pil_image(example)
        image.save(f'{DEBUG_PATH}/{name}.png')


def tensor_to_pil_image(tensor):
    assert 3 <= len(tensor.size()) <= 4
    if len(tensor.size()) == 4:
        tensor = make_grid(tensor)
    return to_pil_image(tensor.detach().cpu())


def flow_tensor_to_image_tensor(tensor):
    assert 3 <= len(tensor.size()) <= 4

    if len(tensor.size()) == 4:
        b, _, h, w = tensor.size()
        result = torch.zeros((b, 3, h, w))
        for i in range(b):
            result[i, :, :, :] = flow_tensor_to_image_tensor(tensor[i, :, :, :])
        return result

    return to_tensor(fz.convert_from_flow(tensor.detach().cpu().numpy().transpose(1, 2, 0)))


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
    if isinstance(frame, Image.Image):
        frame = to_tensor(frame)

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


def save_video(frames, path, frame_type='image', frame_rate=24, codec=cv.VideoWriter_fourcc(*'H264')):
    height, width = frames[0].shape[-1], frames[0].shape[-2]

    if frame_type == 'image':
        frames = [tensor_to_cv_image(t) for t in frames]
    elif frame_type == 'mask':
        frames = [tensor_to_cv_mask(t) for t in frames]
    else:
        ValueError(frame_type)

    makedirs(dirname(path), exist_ok=True)
    video_writer = cv.VideoWriter(path, codec, frame_rate, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
