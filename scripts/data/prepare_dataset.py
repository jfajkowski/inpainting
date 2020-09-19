import argparse
import glob
import random
from os.path import basename

from tqdm import tqdm

from inpainting.load import SequenceDataset, load_dataframe
from inpainting.save import save_frames
from inpainting.utils import annotation_to_mask

parser = argparse.ArgumentParser()
parser.add_argument('--frames-dir', type=str, default='data/interim/DAVIS/JPEGImages/480p')
parser.add_argument('--annotations-dir', type=str, default='data/interim/DAVIS/Annotations/480p')
parser.add_argument('--object-stats-dir', type=str, default='data/interim/DAVIS/ObjectStats')
parser.add_argument('--processed-dir', type=str, default='data/processed/image_inpainting')
parser.add_argument('--limit-samples', type=int, default=3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mode', type=str, default='image_inpainting')
opt = parser.parse_args()

random.seed(opt.seed)

frames_dirs = list(sorted(glob.glob(f'{opt.frames_dir}/*')))
annotations_dirs = list(sorted(glob.glob(f'{opt.annotations_dir}/*')))
object_stats_dirs = list(sorted(glob.glob(f'{opt.object_stats_dir}/*')))
background_sequence_names = list(map(basename, frames_dirs))
foreground_sequence_names = list(map(basename, annotations_dirs))

frame_type = 'flow' if 'flow' in opt.mode else 'image'
frames_dataset = SequenceDataset(
    frames_dirs,
    frame_type=frame_type
)
annotations_dataset = SequenceDataset(
    annotations_dirs,
    frame_type='annotation'
)
object_stats_dataset = list(map(lambda x: load_dataframe(f'{x}/valid_object_stats.csv'), object_stats_dirs))

# Prepare potential candidates for background-foreground pairs
background = list(range(len(frames_dataset)))
foreground = []
for i, df in enumerate(object_stats_dataset):
    if df is None:
        continue
    for _, row in df.iterrows():
        foreground.append((i, row))

candidates = []
for b in background:
    for f in foreground:
        if ('inpainting' in opt.mode and b != f[0]) \
                or (opt.mode == 'tracking_and_segmentation' and b == f[0]):
            candidates.append((b, f))

# Limit candidates with random sampling
if len(candidates) > opt.limit_samples:
    candidates = random.sample(candidates, opt.limit_samples)

for background_index, (foreground_index, foreground_object_stats_row) in tqdm(candidates,
                                                                              desc=f'Preparing {opt.mode} dataset',
                                                                              unit='sequence'):
    background_sequence_name = background_sequence_names[background_index]
    background_frames = frames_dataset[background_index]

    foreground_sequence_name = foreground_sequence_names[foreground_index]
    foreground_annotations = annotations_dataset[foreground_index]

    object_id = int(foreground_object_stats_row['object_id'])
    first_frame_id = int(foreground_object_stats_row['first_frame_id'])
    last_frame_id = int(foreground_object_stats_row['last_frame_id'])
    considered_foreground_annotations = foreground_annotations[first_frame_id:last_frame_id]
    considered_foreground_masks = [annotation_to_mask(a, object_id) for a in considered_foreground_annotations]

    frames, masks = [], []
    for background_frame, foreground_mask in zip(background_frames, considered_foreground_masks):
        frames.append(background_frame)
        masks.append(foreground_mask)

    sequence_name = f'{background_sequence_name}_{foreground_sequence_name}_{object_id:03d}'
    save_frames(frames, f'{opt.processed_dir}/{frame_type.capitalize()}s/{sequence_name}', frame_type)
    save_frames(masks, f'{opt.processed_dir}/Masks/{sequence_name}', 'mask')
