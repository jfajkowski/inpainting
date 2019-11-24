import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset

from deepstab.utils import cutout_mask
from deepstab.visualize import color_map


class ObjectMaskVideoDataset(Dataset):

    def __init__(self, frame_dirs, mask_dirs, sequence_length=None, transforms=None):
        self.frame_dataset = VideoDataset(frame_dirs, sequence_length, transforms)
        self.mask_dataset = VideoDataset(mask_dirs, sequence_length, transforms)

        if len(self.frame_dataset) != len(self.mask_dataset):
            raise ValueError('Lengths of frames dataset and masks dataset don\'t match')
        self.length = len(self.frame_dataset)

    def __getitem__(self, index: int):
        return {'frames': self.frame_dataset.__getitem__(index), 'masks': self.mask_dataset.__getitem__(index)}

    def __len__(self) -> int:
        return self.length

    def __repr__(self):
        return '\n'.join([repr(self.frame_dataset), repr(self.mask_dataset)])


class SquareMaskVideoDataset(Dataset):

    def __init__(self, frame_dirs, sequence_length=None, transforms=None):
        self.frame_dataset = VideoDataset(frame_dirs, sequence_length, transforms)

    def __getitem__(self, index: int):
        frames = []
        masks = []
        for frame in self.frame_dataset.__getitem__(index):
            mask = self.create_mask(frame.size)
            frames.append(frame)
            masks.append(mask)
        return {'frames': frames, 'masks': masks}

    def create_mask(self, size):
        mask = np.zeros(size, dtype=np.uint8)
        mask[int(size[0] * 1 / 4):int(size[0] * 3 / 4), int(size[1] * 1 / 4):int(size[1] * 3 / 4)] = 1
        mask_img = Image.fromarray(mask)
        mask_img.putpalette(color_map().flatten().tolist())
        return mask_img

    def __len__(self) -> int:
        return len(self.frame_dataset)

    def __repr__(self):
        return '\n'.join([repr(self.frame_dataset)])


class RectangleMaskDataset(IterableDataset):
    def __init__(self, height, width, rectangle=None, transforms=None):
        if not rectangle:
            rectangle = (int(width * 1 / 4), int(height * 1 / 4), int(width * 1 / 2), int(height * 1 / 2))
        self.mask = self._create_mask(height, width, rectangle)
        self.transforms = transforms

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
            if self.transforms is not None:
                mask = self.transforms(mask)
            yield mask


class IrregularMaskDataset(IterableDataset):
    def __init__(self, mask_dir, transforms=None):
        self.mask_paths = list(glob.glob(f'{mask_dir}/*'))
        self.transforms = transforms

    def __iter__(self):
        while True:
            mask = Image.open(np.random.choice(self.mask_paths))
            if self.transforms is not None:
                mask = self.transforms(mask)
            yield mask


class InpaintingImageDataset(Dataset):
    def __init__(self, image_dataset, mask_dataset, transforms=None):
        self.image_dataset = image_dataset
        self.mask_generator = iter(mask_dataset)
        self.transforms = transforms

    def __getitem__(self, index: int):
        image = self.image_dataset.__getitem__(index)[0]
        mask = next(self.mask_generator).convert('RGB')
        image_masked = cutout_mask(image, mask)

        if self.transforms is not None:
            image, mask, image_masked = tuple(map(self.transforms, (image, mask, image_masked)))

        return image, mask, image_masked

    def __len__(self) -> int:
        return len(self.image_dataset)


class ImageDataset(Dataset):
    _repr_indent = 4

    def __init__(self, image_dirs, transforms=None):

        if not image_dirs:
            raise ValueError('Empty image directory list given to ImageDataset')

        self.dir_list = image_dirs
        self.transforms = transforms
        self.considered_images = []

        for image_dir in image_dirs:
            self.considered_images += glob.glob(f'{image_dir}/*')

    def __getitem__(self, index: int):
        image = Image.open(self.considered_images[index])
        if self.transforms is not None:
            image = self.transforms(image)
        return image,

    def __len__(self) -> int:
        return len(self.considered_images)

    def __repr__(self):
        head = f'Dataset {self.__class__.__name__}'
        body = [f'Number of datapoints: {self.__len__()}']
        if hasattr(self, 'transforms') and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [' ' * self._repr_indent + line for line in body]
        return '\n'.join(lines)


class VideoDataset(Dataset):
    _repr_indent = 4

    def __init__(self, frame_dirs, sequence_length=None, transforms=None):

        if not frame_dirs:
            raise ValueError('Empty frame directory list given to VideoDataset')

        if sequence_length and sequence_length < 1:
            raise ValueError('Sequence length must be at least 1')

        self.dir_list = frame_dirs
        self.sequence_length = sequence_length
        self.transforms = transforms
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
            if self.transforms is not None:
                frame = self.transforms(frame)
            frames.append(frame)
        return frames

    def __len__(self) -> int:
        return len(self.considered_videos)

    def __repr__(self):
        head = f'Dataset {self.__class__.__name__}'
        body = [f'Number of datapoints: {self.__len__()}']
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transforms') and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [' ' * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return f'Sequence length: {self.sequence_length}'


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((854, 480)),
        transforms.ToTensor()
    ])
    dataset = VideoDataset(['data/raw/DAVIS/JPEGImages/480p/butterfly'], 4, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
