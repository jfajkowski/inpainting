import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset

from deepstab.utils import cutout_mask, extract_mask
from deepstab.visualize import color_map


class InpaintingImageDataset(Dataset):
    def __init__(self, image_dataset, mask_dataset, mask_mode='L', transform=None):
        self.image_dataset = image_dataset
        self.mask_generator = iter(mask_dataset)
        self.mask_mode = mask_mode
        self.transform = transform

    def __getitem__(self, index: int):
        image = self.image_dataset.__getitem__(index)[0]
        mask = next(self.mask_generator).convert(self.mask_mode)
        image_masked = cutout_mask(image, mask)

        if self.transform is not None:
            image, mask, image_masked = tuple(map(self.transform, (image, mask, image_masked)))

        return image, mask, image_masked

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
            self.considered_images += glob.glob(f'{image_dir}/*')

    def __getitem__(self, index: int):
        image = Image.open(self.considered_images[index])
        if self.transform is not None:
            image = self.transform(image)
        return image,

    def __len__(self) -> int:
        return len(self.considered_images)


class RectangleMaskDataset(IterableDataset):
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
        return mask_img

    def __iter__(self):
        while True:
            mask = self.mask
            if self.transform is not None:
                mask = self.transform(mask)
            yield mask


class IrregularMaskDataset(IterableDataset):
    def __init__(self, mask_dir, transform=None):
        self.mask_paths = list(glob.glob(f'{mask_dir}/*'))
        self.transform = transform

    def __iter__(self):
        while True:
            mask = Image.open(np.random.choice(self.mask_paths))
            if self.transform is not None:
                mask = self.transform(mask)
            yield mask


class ObjectMaskVideoDataset(Dataset):
    def __init__(self, frame_dirs, mask_dirs, sequence_length=None, mask_mode='L', frame_transform=None,
                 mask_transform=None, transform=None):
        self.frame_dataset = VideoDataset(frame_dirs, sequence_length, frame_transform)
        self.mask_dataset = VideoDataset(mask_dirs, sequence_length, mask_transform)
        self.mask_mode = mask_mode
        self.transform = transform
        if len(self.frame_dataset) != len(self.mask_dataset):
            raise ValueError('Lengths of frames dataset and masks dataset don\'t match')
        self.length = len(self.frame_dataset)

    def __getitem__(self, index: int):
        frames = []
        masks = []
        frames_masked = []
        for frame, mask in zip(self.frame_dataset.__getitem__(index), self.mask_dataset.__getitem__(index)):
            mask = extract_mask(mask).convert(self.mask_mode)
            frame_masked = cutout_mask(frame, mask)

            if self.transform is not None:
                frame, mask, frame_masked = tuple(map(self.transform, (frame, mask, frame_masked)))

            frames.append(frame)
            masks.append(mask)
            frames_masked.append(frame_masked)

        return frames, masks, frames_masked

    def __len__(self) -> int:
        return self.length


class SquareMaskVideoDataset(Dataset):

    def __init__(self, frame_dirs, sequence_length=None, mask_mode='L', frame_transform=None, mask_transform=None,
                 transform=None):
        self.frame_dataset = VideoDataset(frame_dirs, sequence_length, frame_transform)
        self.mask_transform = mask_transform
        self.transform = transform
        self.mask_mode = mask_mode

    def __getitem__(self, index: int):
        frames = []
        masks = []
        frames_masked = []
        for frame in self.frame_dataset.__getitem__(index):
            mask = extract_mask(self.create_mask((frame.size[1], frame.size[0]))).convert(self.mask_mode)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

            frame_masked = cutout_mask(frame, mask)

            if self.transform is not None:
                frame, mask, frame_masked = tuple(map(self.transform, (frame, mask, frame_masked)))

            frames.append(frame)
            masks.append(mask)
            frames_masked.append(frame_masked)

        return frames, masks, frames_masked

    def create_mask(self, size):
        mask = np.zeros(size, dtype=np.uint8)
        mask[int(size[0] * 1 / 4):int(size[0] * 3 / 4), int(size[1] * 1 / 4):int(size[1] * 3 / 4)] = 1
        mask_img = Image.fromarray(mask)
        mask_img.putpalette(color_map().flatten().tolist())
        return mask_img

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
        frame_paths = glob.glob(f'{self.considered_videos[index]}/*')
        frame_paths = frame_paths[:self.sequence_length] if self.sequence_length else frame_paths
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)
        return frames

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
