import struct
from os import makedirs
from os.path import dirname

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from inpainting.utils import tensor_to_image, tensor_to_mask, flow_tensor_to_image_tensor, tensor_to_flow, \
    convert_tensor


def save_frames(frames, frames_dir, frame_type='image'):
    extension = None
    if frame_type == 'image' or frame_type == 'mask' or frame_type == 'annotation':
        extension = 'png'
    elif frame_type == 'flow':
        extension = 'flo'
    else:
        ValueError(frame_type)

    for i, frame in enumerate(frames):
        save_frame(frame, f'{frames_dir}/{i:05d}.{extension}', frame_type)


def save_frame(frame, path, frame_type='image', roi=None):
    makedirs(dirname(path), exist_ok=True)

    if isinstance(frame, Image.Image):
        frame.save(path)
        return

    if isinstance(frame, torch.Tensor):
        frame = convert_tensor(frame, frame_type)

    if frame_type == 'flow':
        h, w, _ = frame.shape
        with open(path, 'wb') as f_out:
            f_out.write(struct.pack('fii', float(202021.25), w, h))
            f_out.write(frame.astype(np.float32).tobytes())
    else:
        if roi:
            assert frame_type == 'image'
            frame = cv.rectangle(frame, roi[0], roi[1], (255, 0, 0))

        cv.imwrite(path, frame)


def save_video(frames, path, frame_type='image', frame_rate=24, codec=cv.VideoWriter_fourcc(*'avc1')):
    height, width = frames[0].shape[-2], frames[0].shape[-1]

    if frame_type == 'image':
        frames = [tensor_to_image(f) for f in frames]
    elif frame_type == 'mask':
        frames = [tensor_to_mask(f) for f in frames]
    elif frame_type == 'flow':
        frames = [tensor_to_image(flow_tensor_to_image_tensor(f)) for f in frames]
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
