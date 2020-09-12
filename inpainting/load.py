import abc
import glob

import cv2 as cv
import flowiz as fz
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.utils import make_grid, save_image
import random

from inpainting.utils import annotation_to_mask


def load_sample(image_path: str, image_type: str):
    if image_type == 'image':
        return Image.open(image_path).convert('RGB')
    elif image_type == 'mask':
        return Image.open(image_path).convert('L')
    elif image_type == 'annotation':
        return Image.open(image_path).convert('P')
    elif image_type == 'flow':
        return fz.read_flow(image_path)
    else:
        raise ValueError(image_type)


def save_sample(image: np.ndarray, image_path: str, image_type: str):
    if image_type == 'image':
        Image.fromarray(image, 'RGB').save(image_path)
    elif image_type == 'mask':
        Image.fromarray(image, 'L').save(image_path)
    elif image_type == 'annotation':
        Image.fromarray(image, 'P').save(image_path)
    elif image_type == 'flow':
        Image.fromarray(fz.convert_from_flow(image)).save(image_path)
    else:
        raise ValueError(image_type)


class ImageDataset(Dataset):

    def __init__(self, sample_dirs, sample_type, transform=None):

        if not sample_dirs:
            raise ValueError('Empty image directory list given to ImageDataset')

        self.sample_dirs = sample_dirs
        self.sample_type = sample_type
        self.transform = transform
        self.considered_images = []

        for sample_dir in sample_dirs:
            self.considered_images += sorted(glob.glob(f'{sample_dir}/**/*', recursive=True))

    def __getitem__(self, index: int):
        sample = load_sample(self.considered_images[index], self.sample_type)
        if self.transform is not None:
            sample, = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.considered_images)


class VideoDataset(Dataset):

    def __init__(self, sample_dirs, sample_type, sequence_length=None, transform=None):

        if not sample_dirs:
            raise ValueError('Empty frame directory list given to VideoDataset')

        if sequence_length and sequence_length < 1:
            raise ValueError('Sequence length must be at least 1')

        self.sample_dirs = sample_dirs
        self.sample_type = sample_type
        self.sequence_length = sequence_length
        self.transform = transform
        self.considered_sequence_dirs = []
        self.sequence_count = 0

        for sample_dir in sample_dirs:
            frame_paths = sorted(glob.glob(f'{sample_dir}/*'))
            frame_count = len(frame_paths)
            if sequence_length and frame_count < sequence_length:
                print(f'Omitting {sample_dir} because it contains only {frame_count} upper_frames '
                      f'and sequence length is set to {sequence_length}')
            else:
                self.considered_sequence_dirs += self.split_to_sequence_paths(frame_paths)
                self.sequence_count += frame_count - sequence_length if sequence_length else 0 + 1

    def split_to_sequence_paths(self, frame_paths):
        if not self.sequence_length:
            return [frame_paths]

        result = []
        for i in range(len(frame_paths) + 1 - self.sequence_length):
            result.append(frame_paths[i:i + self.sequence_length])
        return result

    def __getitem__(self, index: int):
        frame_paths = self.considered_sequence_dirs[index]
        frames = [load_sample(frame_path, self.sample_type) for frame_path in frame_paths]
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


def _max_annotation_id(annotation):
    return np.amax(annotation)


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


class ImageObjectRemovalDataset(IterableDataset):

    def __init__(self, background_dataset: ImageDataset, foreground_dataset: MergeDataset, transform=None):
        self.background_dataset = background_dataset
        self.foreground_dataset = foreground_dataset
        self.transform = transform
        if len(self.background_dataset) != len(self.foreground_dataset):
            raise ValueError('Lengths of background dataset and foreground dataset don\'t match')

    def __iter__(self):
        while True:
            background_index = random.randint(0, len(self.background_dataset) - 1)
            foreground_index = random.randint(0, len(self.foreground_dataset) - 1)

            background_image = self.background_dataset[background_index]
            foreground_image, foreground_annotation = self.foreground_dataset[foreground_index]

            # 0 is background mask
            annotation_id = random.randint(0, _max_annotation_id(foreground_annotation))
            if annotation_id == 0:
                foreground_mask = _random_mask(foreground_annotation.size)
            else:
                foreground_mask = annotation_to_mask(foreground_annotation, annotation_id)

            input_image = _paste_object(background_image, foreground_image, foreground_mask)
            mask = foreground_mask
            target_image = background_image
            if self.transform is not None:
                input_image, mask, target_image = self.transform(input_image, mask, target_image)
            yield input_image, mask, target_image


class VideoObjectRemovalDataset(IterableDataset):

    def __init__(self, background_dataset: ImageDataset, foreground_dataset: MergeDataset, transform=None):
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
            annotation_id = random.randint(1, max([_max_annotation_id(a) for a in foreground_annotations]))
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import inpainting.transforms as T

    dataset = VideoDataset(list(glob.glob('../data/raw/DAVIS/JPEGImages/*')),
                           sample_type='image',
                           sequence_length=3,
                           transform=T.Compose([
                               T.ToTensor()
                           ]))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    sample = next(iter(dataloader))
    for i, s in enumerate(sample):
        save_image(make_grid(s), f'{i:03d}.jpg')
    print(f'Number of batches: {len(dataloader)}')
    print(f'Sequence length: {len(sample)}')
    print(f'Tensor size: {sample[0].size()}')
