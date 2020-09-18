import glob
import flowiz as fz
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_frame(frame_path, frame_type):
    if frame_type == 'image':
        return Image.open(frame_path).convert('RGB')
    elif frame_type == 'mask':
        return Image.open(frame_path).convert('L')
    elif frame_type == 'annotation':
        return Image.open(frame_path).convert('P')
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
