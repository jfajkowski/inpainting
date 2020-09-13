import glob
import random
import numpy as np
import cv2 as cv

from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from inpainting.utils import annotation_to_mask


def load_frame(image_path: str, image_type: str):
    if image_type == 'image':
        return Image.open(image_path).convert('RGB')
    elif image_type == 'mask':
        return Image.open(image_path).convert('L')
    elif image_type == 'annotation':
        return Image.open(image_path).convert('P')
    else:
        raise ValueError(image_type)


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
            frame_paths = sorted(glob.glob(f'{sample_dir}/*'))
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


def _random_mask(size):
    h, w = size
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


def _paste_object(background_image, foreground_image, foreground_mask):
    assert foreground_image.size == background_image.size == foreground_mask.size
    combined_image = background_image.copy()
    combined_image.paste(foreground_image, mask=foreground_mask)
    return combined_image


class VideoObjectRemovalDataset(IterableDataset):

    def __init__(self, background_dataset: SequenceDataset, foreground_dataset: MergeDataset, transform=None):
        self.background_dataset = background_dataset
        self.foreground_dataset = foreground_dataset
        self.transform = transform
        if len(self.background_dataset) != len(self.foreground_dataset):
            raise ValueError('Lengths of background dataset and foreground dataset don\'t match')

    def __iter__(self):
        while True:
            background_index = random.randint(0, len(self.background_dataset) - 1)
            foreground_index = random.randint(0, len(self.foreground_dataset) - 1)

            background_images = self.background_dataset[background_index]
            foreground_images, foreground_annotations = self.foreground_dataset[foreground_index]

            # 0 is background mask
            annotation_id = random.choice(set([_max_annotation_id(a) for a in foreground_annotations]))
            if annotation_id == 0:
                foreground_masks = [_random_mask(foreground_annotations[0].size)] * len(foreground_annotations)
            else:
                foreground_masks = [annotation_to_mask(a, annotation_id) for a in foreground_annotations]

            input_images = [_paste_object(bi, fi, fm) for bi, fi, fm in
                            zip(background_images, foreground_images, foreground_masks)]
            masks = foreground_masks[:len(input_images)]
            target_images = background_images[:len(input_images)]
            if self.transform is not None:
                input_images, masks, target_images = self.transform(input_images, masks, target_images)
            yield input_images, masks, target_images
