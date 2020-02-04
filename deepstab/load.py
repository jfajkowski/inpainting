import glob

import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset

from deepstab.utils import extract_mask
from deepstab.visualize import color_map


class InpaintingImageDataset(Dataset):
    def __init__(self, image_dataset, mask_dataset, transform=None):
        self.image_dataset = image_dataset
        self.mask_generator = iter(mask_dataset)
        self.transform = transform

    def __getitem__(self, index: int):
        image = self.image_dataset.__getitem__(index)
        mask = next(self.mask_generator)

        if self.transform is not None:
            image, mask = tuple(map(self.transform, (image, mask)))

        return image, mask

    def __len__(self) -> int:
        return len(self.image_dataset)


class ImageDataset(Dataset):
    _repr_indent = 4

    def __init__(self, image_dirs, transform=None):

        if not image_dirs:
            raise ValueError('Empty image directory list given to ImageDataset')

        self.dir_list = image_dirs
        self.transform = transform
        self.considered_images = []

        for image_dir in image_dirs:
            self.considered_images += glob.glob(f'{image_dir}/**/*.jpg', recursive=True)

    def __getitem__(self, index: int):
        image = cv.imread(self.considered_images[index])
        # image = Image.open(self.considered_images[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.considered_images)


class RectangleMaskDataset(Dataset):
    def __init__(self, height, width, rectangle=None, transform=None):
        if not rectangle:
            rectangle = (int(width * 1 / 4), int(height * 1 / 4), int(width * 1 / 2), int(height * 1 / 2))
        self.mask = self._create_mask(height, width, rectangle)
        self.transform = transform

    def _create_mask(self, height, width, rectangle):
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = rectangle
        mask[y:y + h, x:x + w] = 1
        mask_img = Image.fromarray(mask)
        mask_img.putpalette(color_map().flatten().tolist())
        return extract_mask(mask_img)

    def __iter__(self):
        while True:
            mask = self.mask
            if self.transform is not None:
                mask = self.transform(mask.copy())
            yield mask


class FileMaskDataset(IterableDataset):
    def __init__(self, mask_dir, shuffle=True, transform=None):
        self.mask_paths = list(glob.glob(f'{mask_dir}/*'))
        self.mask_paths_generator = self.random_mask_path_generator()
        self.shuffle = shuffle
        self.transform = transform

    def __iter__(self):
        while True:
            mask = np.expand_dims(cv.imread(next(self.mask_paths_generator), cv.IMREAD_GRAYSCALE), axis=2)
            # mask = Image.open(next(self.mask_paths_generator)).convert('L')
            if self.transform is not None:
                mask = self.transform(mask)
            yield mask

    def random_mask_path_generator(self):
        while True:
            if self.shuffle:
                np.random.shuffle(self.mask_paths)
            for mask_path in self.mask_paths:
                yield mask_path


class DynamicMaskVideoDataset(Dataset):
    def __init__(self, frame_dataset, mask_dataset, transform=None):
        self.frame_dataset = frame_dataset
        self.mask_dataset = mask_dataset
        self.transform = transform
        if len(self.frame_dataset) != len(self.mask_dataset):
            raise ValueError('Lengths of frames dataset and masks dataset don\'t match')

    def __getitem__(self, index: int):
        frames, frame_dir = self.frame_dataset.__getitem__(index)
        masks, _ = self.mask_dataset.__getitem__(index)
        for i in range(len(frames)):
            frame = frames[i]
            mask = masks[i]
            if self.transform is not None:
                frame, mask = tuple(map(self.transform, (frame, mask)))

            frames[i] = frame
            masks[i] = mask

        return frames, masks, frame_dir

    def __len__(self) -> int:
        return len(self.frame_dataset)


class StaticMaskVideoDataset(Dataset):
    def __init__(self, frame_dataset, mask_dataset, transform=None):
        self.frame_dataset = frame_dataset
        self.mask_dataset = mask_dataset
        self.mask_generator = iter(mask_dataset)
        self.transform = transform

    def __getitem__(self, index: int):
        frames, frame_dir = self.frame_dataset.__getitem__(index)
        masks = [next(self.mask_generator)] * len(frames)
        for i in range(len(frames)):
            frame = frames[i]
            mask = masks[i]
            if self.transform is not None:
                frame, mask = tuple(map(self.transform, (frame, mask)))

            frames[i] = frame
            masks[i] = mask

        return frames, masks, frame_dir

    def __len__(self) -> int:
        return len(self.frame_dataset)


class VideoDataset(Dataset):
    _repr_indent = 4

    def __init__(self, frame_dirs, sequence_length=None, transform=None):

        if not frame_dirs:
            raise ValueError('Empty frame directory list given to VideoDataset')

        if sequence_length and sequence_length < 1:
            raise ValueError('Sequence length must be at least 1')

        self.dir_list = frame_dirs
        self.sequence_length = sequence_length
        self.transform = transform
        self.considered_videos = []

        for frames_dir in frame_dirs:
            frames_count = len(glob.glob(f'{frames_dir}/*'))
            if sequence_length and frames_count < sequence_length:
                print(f'Omitting {frames_dir} because it contains only {frames_count} upper_frames '
                      f'and sequence length is set to {sequence_length}')
            else:
                self.considered_videos.append(frames_dir)

    def __getitem__(self, index: int):
        frames = []
        frame_dir = self.considered_videos[index]
        frame_paths = sorted(glob.glob(f'{frame_dir}/*'))
        frame_paths = frame_paths[:self.sequence_length] if self.sequence_length else frame_paths
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)
        return frames, frame_dir

    def __len__(self) -> int:
        return len(self.considered_videos)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    dataset = VideoDataset(['data/raw/DAVIS/JPEGImages/480p/butterfly'], 4,
                           transform=transforms.Compose([
                               transforms.Resize((854, 480)),
                               transforms.ToTensor()
                           ]))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
