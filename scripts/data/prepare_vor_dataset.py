import argparse
import glob
import random
from os.path import basename

import pandas as pd
from tqdm import tqdm

from inpainting.load import SequenceDataset
from inpainting.utils import annotation_to_mask
from inpainting.visualize import save_frames

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/interim/demo/AdjustedJPEGImages')
parser.add_argument('--annotations-dir', type=str, default='data/interim/demo/AdjustedAnnotations')
parser.add_argument('--object-stats-dir', type=str, default='data/interim/demo/ObjectStats')
parser.add_argument('--processed-dir', type=str, default='data/processed/demo')
parser.add_argument('--limit-samples', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--min-presence', type=float, default=0.5)
parser.add_argument('--min-mean-size', type=float, default=0.01)
parser.add_argument('--max-mean-size', type=float, default=0.25)
opt = parser.parse_args()

random.seed(opt.seed)

images_dirs = list(sorted(glob.glob(f'{opt.images_dir}/*')))
annotations_dirs = list(sorted(glob.glob(f'{opt.annotations_dir}/*')))
object_stats_dirs = list(sorted(glob.glob(f'{opt.object_stats_dir}/*')))
sequence_names = list(map(basename, images_dirs))

images_dataset = SequenceDataset(
    images_dirs,
    frame_type='image'
)
annotations_dataset = SequenceDataset(
    annotations_dirs,
    frame_type='annotation'
)
object_stats_dataset = list(map(lambda x: pd.read_csv(f'{x}/object_stats.csv'), object_stats_dirs))

# Validate objects in foreground sequences
valid_object_stats_dataset = []
for object_stats in tqdm(object_stats_dataset, desc='Validating objects', unit='sequence'):
    valid_rows = []
    for _, row in object_stats.iterrows():
        if row['presence'] > opt.min_presence \
                and opt.min_mean_size <= row['mean_size'] <= opt.max_mean_size:
            valid_rows.append(row)
    valid_object_stats_dataset.append(valid_rows)

# Prepare potential candidates for background-foreground pairs
background = list(range(len(valid_object_stats_dataset)))
foreground = []
for i, rows in enumerate(valid_object_stats_dataset):
    for row in rows:
        foreground.append((i, row))

candidates = []
for b in background:
    for f in foreground:
        if b != f[0]:
            candidates.append((b, f))

# Limit candidates with random sampling
if len(candidates) > opt.limit_samples:
    candidates = random.sample(candidates, opt.limit_samples)


# Paste objects onto sequences
def _paste_object(background_image, foreground_image, foreground_mask):
    assert foreground_image.size == background_image.size == foreground_mask.size
    combined_image = background_image.copy()
    combined_image.paste(foreground_image, mask=foreground_mask)
    return combined_image


for background_index, (foreground_index, foreground_object_stats_row) in tqdm(candidates, desc='Preparing VOR dataset',
                                                                              unit='sequence'):
    background_sequence_name = sequence_names[background_index]
    background_images = images_dataset[background_index]

    foreground_sequence_name = sequence_names[foreground_index]
    foreground_images = images_dataset[foreground_index]
    foreground_annotations = annotations_dataset[foreground_index]

    object_id = int(foreground_object_stats_row['object_id'])
    first_frame_id = int(foreground_object_stats_row['first_frame_id'])
    last_frame_id = int(foreground_object_stats_row['last_frame_id'])
    considered_foreground_images = foreground_images[first_frame_id:last_frame_id]
    considered_foreground_annotations = foreground_annotations[first_frame_id:last_frame_id]
    considered_foreground_masks = [annotation_to_mask(a, object_id) for a in considered_foreground_annotations]

    input_images = [_paste_object(bi, fi, fm) for bi, fi, fm in
                    zip(background_images, considered_foreground_images, considered_foreground_masks)]
    masks = considered_foreground_masks[:len(input_images)]
    target_images = background_images[:len(input_images)]

    sequence_name = f'{background_sequence_name}_{foreground_sequence_name}_{object_id:03d}'
    save_frames(input_images, f'{opt.processed_dir}/InputImages/{sequence_name}', 'image')
    save_frames(masks, f'{opt.processed_dir}/Masks/{sequence_name}', 'mask')
    save_frames(target_images, f'{opt.processed_dir}/TargetImages/{sequence_name}', 'image')
