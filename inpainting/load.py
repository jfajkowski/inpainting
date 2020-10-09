import random

import cv2 as cv
import flowiz as fz
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_frame(frame_path, frame_type):
    if frame_type == 'image' or frame_type == 'flowviz':
        return cv.cvtColor(cv.imread(frame_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    elif frame_type == 'mask':
        return cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
    elif frame_type == 'annotation':
        return np.array(Image.open(frame_path).convert('P'))
    elif frame_type == 'flow':
        return fz.read_flow(frame_path)
    else:
        raise ValueError(frame_type)


def load_dataframe(df_path):
    try:
        return pd.read_csv(df_path)
    except pd.errors.EmptyDataError:
        return None


class SequenceDataset(Dataset):

    def __init__(self, sequence_dirs, frame_type, sequence_length=None, transform=None):

        if not sequence_dirs:
            raise ValueError('Empty frame directory list given to VideoDataset')

        if sequence_length and sequence_length < 1:
            raise ValueError('Sequence length must be at least 1')

        self.sequence_dirs = sequence_dirs
        self.frame_type = frame_type
        self.sequence_length = sequence_length
        self.sequences = []
        self.sequence_count = 0
        self.transform = transform

        for sample_dir in sequence_dirs:
            frame_paths = get_paths(f'{sample_dir}/*')
            frame_count = len(frame_paths)
            if sequence_length and frame_count < sequence_length:
                print(f'Omitting {sample_dir} because it contains only {frame_count} upper_frames '
                      f'and sequence length is set to {sequence_length}')
            else:
                self.sequences += self.split_to_subsequences(frame_paths)
                self.sequence_count += frame_count - sequence_length if sequence_length else 0 + 1

    def split_to_subsequences(self, frame_paths):
        if not self.sequence_length:
            return [frame_paths]

        result = []
        for i in range(len(frame_paths) + 1 - self.sequence_length):
            result.append(frame_paths[i:i + self.sequence_length])
        return result

    def __getitem__(self, index: int):
        frame_paths = self.sequences[index]
        frames = [load_frame(frame_path, self.frame_type) for frame_path in frame_paths]
        if self.transform is not None:
            frames, = self.transform(frames)
        return frames

    def __len__(self) -> int:
        return self.sequence_count


class MergeDataset(Dataset):

    def __init__(self, datasets, transform=None):
        self.datasets = datasets
        self.transform = transform

    def __getitem__(self, index: int):
        sample = [d[index] for d in self.datasets]
        if self.transform is not None:
            sample = self.transform(*sample)
        return tuple(sample)

    def __len__(self) -> int:
        return len(self.datasets[0])


class RandomMaskDataset(Dataset):

    def __init__(self, frame_dataset, transform=None):
        self.frame_dataset = frame_dataset
        self.transform = transform

    def __getitem__(self, index: int):
        frames = self.frame_dataset[index]
        masks = self._random_masks(frames)
        sample = (frames, masks)
        if self.transform is not None:
            sample = self.transform(*sample)
        return tuple(sample)

    def _random_masks(self, frames):
        return list(map(self._random_mask, frames))

    def _random_mask(self, frame):
        h, w, _ = frame.shape
        s = random.choice([h, w])
        min_points, max_points = 1, s // 5
        min_thickness, max_thickness = 1, s // 5
        min_angle_dir, max_angle_dir = 0, 2 * np.pi
        min_angle_fold, max_angle_fold = - np.pi / 2, np.pi / 2
        min_length, max_length = 1, s // 5

        mask = np.zeros((h, w), dtype='int')
        points = random.randint(min_points, max_points)
        thickness = random.randint(min_thickness, max_thickness)

        prev_x = random.randint(0, w)
        prev_y = random.randint(0, h)

        angle_dir = random.uniform(min_angle_dir, max_angle_dir)
        for i in range(points):
            angle_fold = random.uniform(min_angle_fold, max_angle_fold)
            angle = angle_dir + angle_fold
            length = random.randint(min_length, max_length)
            x = int(prev_x + length * np.sin(angle))
            y = int(prev_y + length * np.cos(angle))
            mask = cv.line(mask, (prev_x, prev_y), (x, y), color=255, thickness=thickness)
            prev_x = x
            prev_y = y
        return Image.fromarray(mask).convert('L')

    def __len__(self) -> int:
        return len(self.frame_dataset)
