from os import makedirs
from os.path import dirname

import cv2 as cv
import flowiz as fz
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid

from inpainting.utils import tensor_to_cv_image, tensor_to_cv_mask


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


def save_frames(frames, dir, frame_type='image'):
    extension = None
    if frame_type == 'image' or frame_type == 'mask' or frame_type == 'annotation':
        extension = 'png'
    else:
        ValueError(frame_type)

    for i, frame in enumerate(frames):
        save_frame(frame, f'{dir}/{i:05d}.{extension}', frame_type)


def save_frame(frame, path, frame_type='image', roi=None):
    makedirs(dirname(path), exist_ok=True)
    if isinstance(frame, Image.Image):
        frame.save(path)
    else:
        if frame_type == 'image':
            frame = tensor_to_cv_image(frame)
        elif frame_type == 'mask':
            frame = tensor_to_cv_mask(frame)
        elif frame_type == 'flow':
            frame = tensor_to_cv_image(flow_tensor_to_image_tensor(frame))
        else:
            raise ValueError(frame_type)

        if roi:
            assert frame_type == 'image'
            frame = cv.rectangle(frame, roi[0], roi[1], (255, 0, 0))

        cv.imwrite(path, frame)


def save_video(frames, path, frame_type='image', frame_rate=24, codec=cv.VideoWriter_fourcc(*'avc1')):
    height, width = frames[0].shape[-2], frames[0].shape[-1]

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


def save_dataframe(df, path):
    makedirs(dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
